"""Run separate native-library processes on generated data and check the mechanism."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CONDITIONS=('clean','iid','odd_rms_matched','iid_positive_gate')
CONTINUOUS_KEYS=(('baseline','clean'),('baseline','iid'),
                 ('child_reeval','clean'),('child_reeval','iid'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def single_tree_parameters(gate=0.0):
    return dict(objective='regression',metric='None',verbosity=-1,num_threads=1,
        deterministic=True,force_col_wise=True,boost_from_average=False,
        learning_rate=1.0,num_leaves=4,max_depth=2,min_data_in_leaf=1,
        min_sum_hessian_in_leaf=0.0,lambda_l1=0.0,lambda_l2=0.0,
        min_gain_to_split=gate,bagging_fraction=1.0,bagging_freq=0,
        feature_fraction=1.0,feature_fraction_bynode=1.0,feature_pre_filter=False,
        seed=20260917,max_bin=15)


def diagnostics(tree):
    counts=[0,0]
    def visit(node):
        if 'split_feature' in node:
            counts[int(node['split_feature'])]+=1
            visit(node['left_child']); visit(node['right_child'])
    visit(tree)
    return dict(root_split_feature=['beta','z'][tree['split_feature']] if 'split_feature' in tree else 'none',
                root_split_gain=float(tree.get('split_gain',0.0)),geometry_splits=counts[1],beta_splits=counts[0])


def worker(args,build):
    spec=build['variants'][args.worker]
    dll_handle=None
    if os.name=='nt':
        dll_handle=os.add_dll_directory(spec['compiler_bin'])
    sys.path.insert(0,spec['runtime'])
    import lightgbm as lgb
    from lightgbm.basic import _LIB
    import numpy as np
    from datasets import four_cells,noise_pair,continuous_shapes
    loaded=Path(_LIB._name).resolve()
    assert loaded==Path(spec['library']).resolve()
    assert sha(loaded)==spec['library_sha256']
    assert lgb.__version__=='4.6.0'
    seeds=[11] if args.suite=='smoke' else [11,12,13,14,15]
    rows=[]
    for a in (0,1):
        x,y=four_cells(a)
        for seed in seeds:
            iid,odd=noise_pair(y,seed)
            for condition in CONDITIONS:
                noise=np.zeros_like(y) if condition=='clean' else odd if condition=='odd_rms_matched' else iid
                gate=1e-5 if condition=='iid_positive_gate' else 0.0
                model=lgb.train(single_tree_parameters(gate),lgb.Dataset(x,label=y+noise,feature_name=['beta','z']),num_boost_round=1)
                pred=model.predict(x,num_threads=1)
                tree=model.dump_model()['tree_info'][0]['tree_structure']
                rows.append(dict(variant=args.worker,a=a,seed=seed,condition=condition,
                    NRMSE_clean=float(np.sqrt(np.mean((pred-y)**2))/np.std(y)),
                    **diagnostics(tree)))
    x,y,_=continuous_shapes(seed=20261009)
    xv,yv,_=continuous_shapes(seed=20261010)
    par=single_tree_parameters()
    par.update(boost_from_average=True,learning_rate=.05,num_leaves=31,max_depth=-1,
               min_data_in_leaf=8,max_bin=255,seed=11)
    continuous=[]
    clean=None
    for condition in ('clean','iid'):
        target=y.copy()
        if condition=='iid':
            noise=np.random.default_rng(11).standard_normal(len(y))
            target+=noise*(1e-4*np.std(y)/np.sqrt(np.mean(noise**2)))
        start=time.perf_counter()
        model=lgb.train(par,lgb.Dataset(x,label=target,feature_name=['beta','z0','z1','z2','z3','z4']),num_boost_round=300)
        elapsed=time.perf_counter()-start
        mirror=xv.copy(); mirror[:,0]*=-1
        projected=(model.predict(xv,num_threads=1)-model.predict(mirror,num_threads=1))/2
        geom=int(model.feature_importance('split')[1:].sum())
        continuous.append(dict(variant=args.worker,condition=condition,
            validation_NRMSE=float(np.sqrt(np.mean((projected-yv)**2))/np.std(yv)),
            geometry_splits=geom,fit_seconds=elapsed))
    save(args.output/f'worker_{args.worker}.json',dict(version=lgb.__version__,library_sha256=sha(loaded),
        numpy=np.__version__,single_tree=rows,continuous=continuous))
    print(f'{args.worker}: {len(rows)} single trees + {len(continuous)} continuous-data fits completed.',flush=True)


def continuous_checks(rows):
    """Check the full four-condition matrix, including the combined intervention."""
    checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
    keys=[(r.get('variant'),r.get('condition')) for r in rows]
    complete=len(keys)==len(CONTINUOUS_KEYS) and all(keys.count(k)==1 for k in CONTINUOUS_KEYS)
    check('Continuous results contain each of the four conditions exactly once',complete)
    def valid(row):
        error=row.get('validation_NRMSE'); splits=row.get('geometry_splits')
        return (isinstance(error,Real) and not isinstance(error,bool)
                and math.isfinite(error) and error>=0
                and isinstance(splits,Integral) and not isinstance(splits,bool) and splits>=0)
    well_formed=complete and all(valid(r) for r in rows)
    check('Continuous metrics are finite and nonnegative with integer split counts',well_formed)
    c=dict(zip(keys,rows)) if complete else {}
    check('Continuous baseline/clean omits geometry',well_formed and c['baseline','clean']['geometry_splits']==0)
    for key in CONTINUOUS_KEYS[1:]:
        label='/'.join(key)
        check(f'Continuous {label} restores geometry splits',well_formed and c[key]['geometry_splits']>0)
        check(f'Continuous {label} reduces validation NRMSE',
              well_formed and c[key]['validation_NRMSE']<c['baseline','clean']['validation_NRMSE'])
    return checks


def verify_results(rows,continuous):
    """Validate fresh or recorded results without fitting a model."""
    import numpy as np
    assertions=[]
    def check(name,value):
        assertions.append(dict(check=name,passed=bool(value)))
    clean=[r for r in rows if r['condition']=='clean']
    check('XOR has no profitable root in either library',all(r['root_split_feature']=='none' and abs(r['NRMSE_clean']-1)<1e-12 for r in clean if r['a']==0))
    check('Baseline a=1 uses beta but omits geometry',all(r['root_split_feature']=='beta' and r['geometry_splits']==0 and abs(r['NRMSE_clean']-1/np.sqrt(2))<1e-12 for r in clean if r['a']==1 and r['variant']=='baseline'))
    check('Patched a=1 restores two geometry splits and zero error',all(r['geometry_splits']==2 and r['NRMSE_clean']<1e-12 for r in clean if r['a']==1 and r['variant']=='child_reeval'))
    check('IID perturbation repairs the zero-gate baseline',all(r['geometry_splits']>0 and r['NRMSE_clean']<1e-3 for r in rows if r['variant']=='baseline' and r['condition']=='iid'))
    check('Positive threshold retains the distinction between libraries',all((r['geometry_splits']==0 if r['variant']=='baseline' else r['geometry_splits']==2) for r in rows if r['a']==1 and r['condition']=='iid_positive_gate'))
    with (ROOT/'expected/archived_single_tree.csv').open(encoding='utf-8-sig',newline='') as stream:
        reference={(r['variant'],int(r['a']),int(r['seed']),r['condition']):r for r in csv.DictReader(stream)}
    diffs=[abs(r['NRMSE_clean']-float(reference[(r['variant'],r['a'],r['seed'],r['condition'])]['NRMSE_clean'])) for r in rows]
    check('Single-tree NRMSE agrees with archived results (absolute tolerance 1e-10)',max(diffs)<1e-10)
    assertions.extend(continuous_checks(continuous))
    return dict(status='PASS' if all(x['passed'] for x in assertions) else 'FAIL',
                assertions=assertions,single_tree_fits=len(rows),continuous_fits=len(continuous),
                max_archived_NRMSE_difference=max(diffs),continuous_results=continuous,
                real_data_access=False)


def controller(args,build):
    from datasets import export
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    export(out/'data')
    save(out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        suite=args.suite,script_sha256=sha(__file__),datasets_sha256=sha(ROOT/'datasets.py'),
        single_tree_parameters=single_tree_parameters(),noise_seeds=[11] if args.suite=='smoke' else list(range(11,16)),
        data='Generated from functions; no aerodynamic database access.',
        continuous_train_seed=20261009,continuous_validation_seed=20261010,
        single_tree_endpoint='NRMSE against clean labels over the balanced four cells (structural fit).',
        continuous_endpoint='Projected NRMSE on independently sampled synthetic geometries.',
        library_sha256={v:s['library_sha256'] for v,s in build['variants'].items()}))
    rows=[]; continuous=[]
    for variant in ('baseline','child_reeval'):
        env=os.environ.copy()
        env['OMP_NUM_THREADS']='1'; env['OPENBLAS_NUM_THREADS']='1'; env['MKL_NUM_THREADS']='1'
        env['PATH']=build['variants'][variant]['compiler_bin']+os.pathsep+env.get('PATH','')
        command=[sys.executable,str(Path(__file__).resolve()),'--build-manifest',str(args.build_manifest.resolve()),
                 '--output',str(out),'--suite',args.suite,'--worker',variant]
        completed=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
        (out/f'{variant}.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        if completed.returncode:
            raise RuntimeError(f'{variant} failed with exit code {completed.returncode}; see {out}/{variant}.log. No automatic retry.')
        print(completed.stdout,end='',flush=True)
        block=json.loads((out/f'worker_{variant}.json').read_text())
        rows.extend(block['single_tree']); continuous.extend(block['continuous'])
    for name,data in [('single_tree.csv',rows),('continuous.csv',continuous)]:
        with (out/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    result=verify_results(rows,continuous)
    result['results_sha256']=sha(out/'single_tree.csv')
    save(out/'verification.json',result)
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':
        raise SystemExit('Verification failed. Inspect saved results before any rerun.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-manifest',type=Path,default=ROOT/'build/build_manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/full')
    parser.add_argument('--suite',choices=('smoke','full'),default='full')
    parser.add_argument('--worker',choices=('baseline','child_reeval'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    build=json.loads(args.build_manifest.read_text(encoding='utf-8'))
    assert build['status']=='COMPLETED'
    if args.worker:
        worker(args,build)
    else:
        controller(args,build)
