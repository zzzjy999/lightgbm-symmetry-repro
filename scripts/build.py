"""Build pristine/patched CPU libraries in isolated directories; never edit input sources."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
COMMAND_LOG=None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call(command, cwd=None, binary=False, log_path=None, timeout=1800):
    """Save both output streams before propagating command failures or timeouts."""
    command=list(map(str,command))
    paths=list(dict.fromkeys(Path(p) for p in (COMMAND_LOG,log_path) if p is not None))
    def as_bytes(value):
        return value.encode('utf-8') if isinstance(value,str) else value or b''
    def record(outcome,stdout=b'',stderr=b''):
        header=f'\n[{datetime.now(timezone.utc).isoformat()}] {outcome}\nCommand: {json.dumps(command)}\nCwd: {cwd or Path.cwd()}\n'
        payload=(f'[binary stdout omitted: {len(stdout or b"")} bytes]'.encode()
                 if binary else as_bytes(stdout))
        for path in paths:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('ab') as stream:
                stream.write(header.encode('utf-8')+b'--- stdout ---\n'+payload+b'\n--- stderr ---\n'+as_bytes(stderr)+b'\n')
    def note(error):
        if paths:
            error.add_note('Command diagnostics saved to: '+', '.join(map(str,paths)))
    try:
        result=subprocess.run(command,cwd=cwd,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired as error:
        record(f'TIMEOUT after {timeout} seconds',error.stdout,error.stderr)
        note(error)
        raise
    except OSError as error:
        record('LAUNCH FAILED',stderr=str(error))
        note(error)
        raise
    record(f'EXIT {result.returncode}',result.stdout,result.stderr)
    try:
        result.check_returncode()
    except subprocess.CalledProcessError as error:
        note(error)
        raise
    return result.stdout if binary else result.stdout.decode('utf-8',errors='replace').strip()


def git(source,*args,binary=False):
    return call(['git','-c',f'safe.directory={source.as_posix()}','-C',source,*args],binary=binary)


def export_commit(source,commit,destination):
    destination.mkdir(parents=True,exist_ok=False)
    content=git(source,'archive','--format=tar',commit,binary=True)
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        archive.extractall(destination,filter='data')
    entries=git(source,'ls-tree','-r',commit).splitlines()
    for line in entries:
        info,path=line.split('\t',1)
        mode,kind,object_id=info.split()
        if mode=='160000':
            # The parent archive includes an empty gitlink directory.
            child=destination/path
            if child.exists():
                assert not any(child.iterdir())
                child.rmdir()
            export_commit(source/path,object_id,child)


def main():
    global COMMAND_LOG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='Existing initialized LightGBM clone; exports pinned Git objects, not working-tree changes')
    parser.add_argument('--work-dir',type=Path,default=ROOT/'build')
    parser.add_argument('--cmake',default='cmake')
    parser.add_argument('--ninja',default='ninja')
    parser.add_argument('--cc',default='gcc')
    parser.add_argument('--cxx',default='g++')
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    work=args.work_dir.resolve()
    if work.exists():
        raise SystemExit('Use a new work directory; existing builds are never overwritten.')
    work.mkdir(parents=True)
    COMMAND_LOG=work/'commands.log'
    cfg=json.loads((ROOT/'config/build.json').read_text())
    assert sha(ROOT/cfg['patch'])==cfg['patch_sha256']
    source=args.source.resolve() if args.source else work/'upstream'
    if not args.source:
        print('Cloning pinned upstream tag and submodules...',flush=True)
        call(['git','clone','--recursive','--depth','1','--branch',cfg['upstream_tag'],cfg['upstream_url'],source])
    assert git(source,'rev-parse',cfg['upstream_tag']+'^{commit}')==cfg['upstream_commit']
    resolved={name:shutil.which(getattr(args,name)) for name in ('cmake','ninja','cc','cxx')}
    if not all(resolved.values()):
        raise SystemExit(f'Missing build executable: {resolved}')
    start=time.perf_counter()
    record=dict(status='BUILDING',created_utc=datetime.now(timezone.utc).isoformat(),
                upstream_commit=cfg['upstream_commit'],patch_sha256=cfg['patch_sha256'],
                cmake_version=call([resolved['cmake'],'--version']).splitlines()[0],
                compiler_version=call([resolved['cxx'],'--version']).splitlines()[0],
                ninja_version=call([resolved['ninja'],'--version']),variants={})
    manifest=work/'build_manifest.json'
    manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    for variant in ('baseline','child_reeval'):
        src=work/variant/'source'
        export_commit(source,cfg['upstream_commit'],src)
        # Local repositories are only build scratch space (ignored by this repository).
        call(['git','init','-q',src])
        if variant=='child_reeval':
            git(src,'apply','--check',str(ROOT/cfg['patch']))
            git(src,'apply',str(ROOT/cfg['patch']))
        out=work/variant/'cmake'
        command=[resolved['cmake'],'-S',src,'-B',out,'-G',cfg['generator'],
                 f'-DCMAKE_MAKE_PROGRAM={Path(resolved["ninja"]).as_posix()}',
                 f'-DCMAKE_C_COMPILER={Path(resolved["cc"]).as_posix()}',
                 f'-DCMAKE_CXX_COMPILER={Path(resolved["cxx"]).as_posix()}']
        command += [f'-D{k}={v}' for k,v in cfg['cmake_options'].items()]
        print(f'Configuring {variant}...',flush=True)
        call(command,log_path=work/variant/'configure.log')
        print(f'Compiling {variant} with {args.jobs} jobs...',flush=True)
        call([resolved['cmake'],'--build',out,'--target','_lightgbm','--parallel',args.jobs],
             log_path=work/variant/'compile.log')
        candidates=[p for p in src.glob('lib_lightgbm.*') if p.suffix in ('.dll','.so','.dylib')]
        assert len(candidates)==1,candidates
        library=candidates[0]
        runtime=work/'runtime'/variant/'lightgbm'
        shutil.copytree(src/'python-package/lightgbm',runtime)
        shutil.copyfile(src/'VERSION.txt',runtime/'VERSION.txt')
        (runtime/'lib').mkdir()
        shutil.copyfile(library,runtime/'lib'/library.name)
        record['variants'][variant]=dict(runtime=str(runtime.parent),library=str(runtime/'lib'/library.name),
            library_sha256=sha(library),serial_tree_learner_sha256=sha(src/'src/treelearner/serial_tree_learner.cpp'),
            compiler_bin=str(Path(resolved['cxx']).parent),configure_command=list(map(str,command)),
            cmake_cache_sha256=sha(out/'CMakeCache.txt'))
        manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(f'{variant} built: {sha(library)}',flush=True)
    record.update(status='COMPLETED',elapsed_seconds=time.perf_counter()-start)
    manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'Build complete. Run: python reproduce.py --build-manifest "{manifest}"',flush=True)


if __name__=='__main__':
    main()
