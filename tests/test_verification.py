"""Regression tests use recorded results; no LightGBM fitting is performed."""
import argparse
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import reproduce


def recorded_results():
    with (reproduce.ROOT/'expected/rebuilt_single_tree.csv').open(newline='',encoding='utf-8') as stream:
        single=list(csv.DictReader(stream))
    for row in single:
        for name in ('a','seed','geometry_splits','beta_splits'):
            row[name]=int(row[name])
        for name in ('NRMSE_clean','root_split_gain'):
            row[name]=float(row[name])
    continuous=json.loads((reproduce.ROOT/'expected/rebuilt_verification.json').read_text(encoding='utf-8'))['continuous_results']
    return single,continuous


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.single,self.continuous=recorded_results()

    def test_recorded_results_pass_all_checks(self):
        result=reproduce.verify_results(self.single,self.continuous)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(len(result['assertions']),15)
        self.assertTrue(all(c['passed'] for c in result['assertions']))

    def test_all_three_interventions_reject_bad_error_or_missing_splits(self):
        for key in reproduce.CONTINUOUS_KEYS[1:]:
            for field,value in (('validation_NRMSE',999.0),('geometry_splits',0)):
                with self.subTest(key=key,field=field):
                    continuous=deepcopy(self.continuous)
                    row=next(r for r in continuous if (r['variant'],r['condition'])==key)
                    row[field]=value
                    self.assertEqual(reproduce.verify_results(self.single,continuous)['status'],'FAIL')

    def test_every_condition_is_required_and_duplicates_are_rejected(self):
        for index in range(4):
            with self.subTest(index=index):
                missing=self.continuous[:index]+self.continuous[index+1:]
                self.assertEqual(reproduce.verify_results(self.single,missing)['status'],'FAIL')
                duplicate=missing+[deepcopy(missing[0])]
                self.assertEqual(reproduce.verify_results(self.single,duplicate)['status'],'FAIL')
        self.assertEqual(reproduce.verify_results(self.single,self.continuous+[deepcopy(self.continuous[0])])['status'],'FAIL')

    def test_invalid_metrics_are_rejected_for_every_condition(self):
        for index in range(4):
            for field,values in (
                ('validation_NRMSE',[float('nan'),float('inf'),-1.0,None,True,'0.1']),
                ('geometry_splits',[float('nan'),float('inf'),-1,0.5,None,True])):
                for value in values:
                    with self.subTest(index=index,field=field,value=value):
                        continuous=deepcopy(self.continuous)
                        continuous[index][field]=value
                        self.assertEqual(reproduce.verify_results(self.single,continuous)['status'],'FAIL')

    def test_controller_saves_fail_and_exits_on_corrupt_patched_iid(self):
        continuous=deepcopy(self.continuous)
        next(r for r in continuous if (r['variant'],r['condition'])==('child_reeval','iid'))['validation_NRMSE']=999.0
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'results'
            args=argparse.Namespace(output=out,suite='full',build_manifest=Path(directory)/'unused.json')
            build={'variants':{v:{'compiler_bin':directory,'library_sha256':'fixture'} for v in ('baseline','child_reeval')}}
            def fake_worker(command,**kwargs):
                variant=command[-1]
                reproduce.save(out/f'worker_{variant}.json',dict(
                    single_tree=[r for r in self.single if r['variant']==variant],
                    continuous=[r for r in continuous if r['variant']==variant]))
                return subprocess.CompletedProcess(command,0,stdout='',stderr='')
            with patch('reproduce.subprocess.run',side_effect=fake_worker), patch('sys.stdout',new_callable=StringIO):
                with self.assertRaises(SystemExit) as raised:
                    reproduce.controller(args,build)
            self.assertNotEqual(raised.exception.code,0)
            result=json.loads((out/'verification.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'FAIL')
            failed=[c['check'] for c in result['assertions'] if not c['passed']]
            self.assertIn('Continuous child_reeval/iid reduces validation NRMSE',failed)


if __name__=='__main__':
    unittest.main()
