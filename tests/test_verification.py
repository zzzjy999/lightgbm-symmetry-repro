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
        self.assertEqual(len(result['assertions']),17)
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


class SingleTreeValidationTests(unittest.TestCase):
    def setUp(self):
        self.single,self.continuous=recorded_results()
        self.smoke=[r for r in self.single if r['seed']==11]

    def verify(self,rows,suite='full'):
        return reproduce.verify_results(rows,self.continuous,suite=suite)

    def test_full_and_smoke_require_the_declared_suite(self):
        for rows,suite,count in ((self.single,'full',80),(self.smoke,'smoke',16)):
            with self.subTest(suite=suite):
                result=self.verify(rows,suite)
                self.assertEqual(result['status'],'PASS')
                self.assertEqual(result['expected_single_tree_fits'],count)
                self.assertEqual(result['single_tree_fits'],count)
                self.assertEqual(result['max_archived_NRMSE_difference'],0.0)
                self.assertEqual(self.verify(list(reversed(rows)),suite)['status'],'PASS')
        self.assertEqual(self.verify(self.smoke)['status'],'FAIL')
        self.assertEqual(self.verify(self.single,'smoke')['status'],'FAIL')
        with self.assertRaises(ValueError):
            self.verify(self.single,'typo')

    def test_reported_missing_group_equal_size_duplicates_and_one_row_fail(self):
        missing=[r for r in self.single if not (r['variant']=='child_reeval' and r['a']==1 and r['condition']=='clean')]
        self.assertEqual(len(missing),75)
        for rows in (missing,missing+[deepcopy(missing[0]) for _ in range(5)],self.single[:1],[]):
            with self.subTest(count=len(rows)):
                result=self.verify(rows)
                self.assertEqual(result['status'],'FAIL')
                self.assertIsNone(result['max_archived_NRMSE_difference'])
                self.assertFalse(next(c['passed'] for c in result['assertions'] if c['check'].startswith('Patched a=1')))

    def test_every_full_and_smoke_key_is_required_exactly_once(self):
        for rows,suite in ((self.single,'full'),(self.smoke,'smoke')):
            for index in range(len(rows)):
                with self.subTest(suite=suite,index=index):
                    missing=rows[:index]+rows[index+1:]
                    self.assertEqual(self.verify(missing,suite)['status'],'FAIL')
                    self.assertEqual(self.verify(missing+[deepcopy(missing[0])],suite)['status'],'FAIL')
            self.assertEqual(self.verify(rows+[deepcopy(rows[0])],suite)['status'],'FAIL')

    def test_unexpected_or_malformed_keys_fail(self):
        cases=(('variant','other'),('condition','other'),('seed',99),('seed','11'),
               ('seed',11.0),('a',True),('a',0.0),('a',None))
        for field,value in cases:
            with self.subTest(field=field,value=value):
                rows=deepcopy(self.single); rows[0][field]=value
                self.assertEqual(self.verify(rows)['status'],'FAIL')
        for field in ('variant','condition','a','seed'):
            rows=deepcopy(self.single); del rows[0][field]
            self.assertEqual(self.verify(rows)['status'],'FAIL')
        rows=deepcopy(self.single); rows[0]=None
        self.assertEqual(self.verify(rows)['status'],'FAIL')
        shifted=[dict(r,seed=12) for r in self.smoke]
        self.assertEqual(self.verify(shifted,'smoke')['status'],'FAIL')

    def test_nan_in_every_position_fails_before_archive_maximum(self):
        for index in range(len(self.single)):
            with self.subTest(index=index,condition=self.single[index]['condition']):
                rows=deepcopy(self.single); rows[index]['NRMSE_clean']=float('nan')
                result=self.verify(rows)
                self.assertEqual(result['status'],'FAIL')
                self.assertIsNone(result['max_archived_NRMSE_difference'])
                json.dumps(result,allow_nan=False)

    def test_single_tree_numeric_fields_reject_invalid_values(self):
        cases={
            'NRMSE_clean':(float('inf'),float('-inf'),-1,None,True,'0.1',1+0j),
            'root_split_gain':(float('nan'),float('inf'),float('-inf'),-1,None,True,'0'),
            'geometry_splits':(float('nan'),float('inf'),-1,0.5,None,True,'0'),
            'beta_splits':(float('nan'),float('inf'),-1,0.5,None,True,'0')}
        for field,values in cases.items():
            for value in values:
                with self.subTest(field=field,value=value):
                    rows=deepcopy(self.single); rows[2][field]=value
                    result=self.verify(rows)
                    self.assertEqual(result['status'],'FAIL')
                    self.assertIsNone(result['max_archived_NRMSE_difference'])
            rows=deepcopy(self.single); del rows[2][field]
            self.assertEqual(self.verify(rows)['status'],'FAIL')

    def test_bad_archive_values_and_duplicate_reference_keys_fail(self):
        with (reproduce.ROOT/'expected/archived_single_tree.csv').open(newline='',encoding='utf-8') as stream:
            archived=list(csv.DictReader(stream))
        for fault in ('nan','duplicate','missing'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); (root/'expected').mkdir()
                changed=deepcopy(archived)
                if fault=='nan': changed[2]['NRMSE_clean']='NaN'
                elif fault=='duplicate': changed.append(deepcopy(changed[2]))
                else: changed.pop(2)
                with (root/'expected/archived_single_tree.csv').open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(changed[0])); writer.writeheader(); writer.writerows(changed)
                with patch.object(reproduce,'ROOT',root):
                    result=self.verify(self.single)
                self.assertEqual(result['status'],'FAIL')
                self.assertIsNone(result['max_archived_NRMSE_difference'])

    def controller_result(self,rows,suite,expect_fail):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'results'
            args=argparse.Namespace(output=out,suite=suite,build_manifest=Path(directory)/'unused.json')
            build={'variants':{v:{'compiler_bin':directory,'library_sha256':'fixture'} for v in ('baseline','child_reeval')}}
            def fake_worker(command,**kwargs):
                variant=command[-1]
                block=dict(single_tree=[r for r in rows if r['variant']==variant],
                    continuous=[r for r in self.continuous if r['variant']==variant])
                # Deliberately permit injected NaN to exercise validation of external records.
                (out/f'worker_{variant}.json').write_text(json.dumps(block),encoding='utf-8')
                return subprocess.CompletedProcess(command,0,stdout='',stderr='')
            with patch('reproduce.subprocess.run',side_effect=fake_worker), patch('sys.stdout',new_callable=StringIO):
                if expect_fail:
                    with self.assertRaises(SystemExit) as raised:
                        reproduce.controller(args,build)
                    self.assertNotEqual(raised.exception.code,0)
                else:
                    reproduce.controller(args,build)
            return json.loads((out/'verification.json').read_text(encoding='utf-8'))

    def test_controller_propagates_smoke_suite(self):
        result=self.controller_result(self.smoke,'smoke',expect_fail=False)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['expected_single_tree_fits'],16)

    def test_controller_saves_fail_for_missing_duplicate_empty_and_nan_records(self):
        missing=[r for r in self.single if not (r['variant']=='child_reeval' and r['a']==1 and r['condition']=='clean')]
        nan=deepcopy(self.single); nan[2]['NRMSE_clean']=float('nan')
        for rows in (missing,missing+[deepcopy(missing[0]) for _ in range(5)],self.single[:1],[],nan):
            with self.subTest(count=len(rows)):
                result=self.controller_result(rows,'full',expect_fail=True)
                self.assertEqual(result['status'],'FAIL')
                self.assertIsNone(result['max_archived_NRMSE_difference'])


if __name__=='__main__':
    unittest.main()
