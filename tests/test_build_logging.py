"""Exercise command diagnostics with tiny subprocesses and injected timeouts."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import build


class BuildLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.log=self.root/'stage.log'
        self.global_log=self.root/'commands.log'
        self.override=patch.object(build,'COMMAND_LOG',self.global_log)
        self.override.start()
        self.addCleanup(self.override.stop)

    def test_success_retains_stderr_and_return_value(self):
        output=build.call([sys.executable,'-c',"import sys; print('output'); print('warning',file=sys.stderr)"],log_path=self.log)
        self.assertEqual(output,'output')
        for path in (self.log,self.global_log):
            text=path.read_text(encoding='utf-8')
            self.assertIn('EXIT 0',text)
            self.assertIn('--- stdout ---\noutput',text)
            self.assertIn('--- stderr ---\nwarning',text)

    def test_nonzero_exit_saves_both_streams_before_raising(self):
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            build.call([sys.executable,'-c',"import sys; print('compiler output'); print('fatal diagnostic',file=sys.stderr); sys.exit(7)"],log_path=self.log)
        self.assertEqual(raised.exception.returncode,7)
        for path in (self.log,self.global_log):
            text=path.read_text(encoding='utf-8')
            self.assertIn('EXIT 7',text)
            self.assertIn('compiler output',text)
            self.assertIn('fatal diagnostic',text)
        self.assertIn(str(self.log),' '.join(raised.exception.__notes__))

    def test_timeout_saves_partial_output(self):
        error=subprocess.TimeoutExpired(['compiler'],1,output=b'partial output',stderr=b'partial diagnostic')
        with patch('scripts.build.subprocess.run',side_effect=error):
            with self.assertRaises(subprocess.TimeoutExpired):
                build.call(['compiler'],log_path=self.log,timeout=1)
        for path in (self.log,self.global_log):
            text=path.read_text(encoding='utf-8')
            self.assertIn('TIMEOUT after 1 seconds',text)
            self.assertIn('partial output',text)
            self.assertIn('partial diagnostic',text)

    def test_launch_failure_is_logged(self):
        missing=self.root/'missing-compiler.exe'
        with self.assertRaises(OSError):
            build.call([missing],log_path=self.log)
        self.assertIn('LAUNCH FAILED',self.log.read_text(encoding='utf-8'))
        self.assertIn('missing-compiler.exe',self.global_log.read_text(encoding='utf-8'))

    def test_binary_stdout_is_returned_intact_without_polluting_log(self):
        command=[sys.executable,'-c',"import sys; sys.stdout.buffer.write(b'\\x00\\xffarchive'); sys.stderr.write('archive warning')"]
        output=build.call(command,binary=True,log_path=self.log)
        self.assertEqual(output,b'\x00\xffarchive')
        text=self.log.read_text(encoding='utf-8')
        self.assertIn('[binary stdout omitted: 9 bytes]',text)
        self.assertIn('archive warning',text)
        self.assertNotIn(b'\x00\xff',self.log.read_bytes())

    def test_global_log_covers_calls_without_a_stage_log_and_appends(self):
        build.call([sys.executable,'-c',"print('first command')"])
        with self.assertRaises(subprocess.CalledProcessError):
            build.call([sys.executable,'-c',"import sys; print('second command'); sys.exit(2)"])
        text=self.global_log.read_text(encoding='utf-8')
        self.assertIn('EXIT 0',text)
        self.assertIn('EXIT 2',text)
        self.assertIn('first command',text)
        self.assertIn('second command',text)


if __name__=='__main__':
    unittest.main()
