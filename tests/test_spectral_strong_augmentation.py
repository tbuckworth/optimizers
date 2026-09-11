"""Synthetic resource/archive/selection fixtures; no model or scientific inputs."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiments import spectral_strong_augmentation as runner


class StrongRunnerFixtures(unittest.TestCase):
    def validation_records(self):
        return [{'step':step,'validation_ce':2.,'readout_receipt':{'path':f'logits-fake-h{step:05d}.npz',
                 'size_bytes':0,'sha256':'0'*64}} for step in runner.EVAL_STEPS]

    def branches(self):
        values=np.zeros((2,10),dtype=np.float32)
        labels=np.array([0,1],dtype=np.int64)
        stats=runner.scalar.classification(values,labels)
        return [{**r,'name':runner.branch_name(r),'metrics':[{'step':x['step'],'validation':stats.copy(),
                 'readout_receipt':x['readout_receipt']} for x in self.validation_records()]} for r in runner.roster()]

    def test_roster_exposure_readouts_inventory(self):
        rows=runner.roster()
        self.assertEqual(len(rows),12)
        self.assertEqual(runner.STEPS,782*72)
        self.assertEqual(runner.STEPS*12,675648)
        self.assertEqual(runner.EPOCHS*runner.TRAIN*12,43200000)
        self.assertEqual(runner.EVAL_STEPS,(0,100)+tuple(782*i for i in range(1,73)))
        self.assertEqual(len(runner.EVAL_STEPS),74)
        for i,seed in enumerate(runner.SEEDS):
            self.assertEqual([(r['policy'],r['augmentation']) for r in rows if r['seed']==seed],
                             list(runner.CELLS[i:]+runner.CELLS[:i]))
        inventory=runner.byte_inventory()
        self.assertLess(inventory['total_upper_bytes']+runner.RESERVE_BYTES,8*1024**3)
        self.assertGreater(inventory['component_upper_bytes']['27_snapshot_upper']/27,128*1024**2)

    def test_selector_exact_ties_and_exposures(self):
        records=self.validation_records()
        self.assertEqual(runner.choose_validation(records)['selected_step'],0)
        records[1]['validation_ce']=1.
        records[2]['validation_ce']=1.
        choice=runner.choose_validation(records)
        self.assertEqual(choice['selected_step'],100)
        self.assertEqual(choice['selected_example_count'],6400)
        records[-1]['validation_ce']=.5
        self.assertEqual(runner.choose_validation(records)['selected_example_count'],3600000)

    def test_selector_rejects_reporting_and_nonfinite(self):
        records=self.validation_records()
        records[0]['reporting_ce']=object()
        with self.assertRaises(RuntimeError):
            runner.choose_validation(records)
        for bad in (float('nan'),float('inf'),-1.,True):
            records=self.validation_records()
            records[0]['validation_ce']=bad
            with self.assertRaises(RuntimeError):
                runner.choose_validation(records)

    def test_selection_readback_recomputed_and_tampered(self):
        branches=self.branches()
        choices=runner.selection_rows(branches)
        labels={s:np.array([0,1],dtype=np.int64) for s in runner.SEEDS}
        loader=lambda receipt,count:np.zeros((count,10),dtype=np.float32)
        self.assertTrue(runner.verify_selection(branches,choices,labels,loader))
        choices[0]['selected_step']=100
        with self.assertRaisesRegex(RuntimeError,'readback'):
            runner.verify_selection(branches,choices,labels,loader)
        choices=runner.selection_rows(branches)
        with self.assertRaisesRegex(RuntimeError,'validation metrics'):
            runner.verify_selection(branches,choices,labels,lambda receipt,count:np.ones((count,10),dtype=np.float32)*np.arange(10,dtype=np.float32))

    def test_all_validation_before_any_reporting(self):
        branches=self.branches()
        labels={s:{'validation':np.array([0,1],dtype=np.int64),'reporting':np.array([0,1],dtype=np.int64)} for s in runner.SEEDS}
        events=[]
        with tempfile.TemporaryDirectory() as temp:
            run=runner.Run(temp,device='cpu')
            run.check=lambda:None
            def panel(directory,receipt,which,count):
                events.append(which)
                if which=='reporting':
                    self.assertTrue((Path(temp)/'selection.json').is_file())
                    verification=json.loads((Path(temp)/'selection-verification.json').read_text())
                    self.assertEqual(verification['status'],'PASS')
                    self.assertFalse(verification['reporting_metrics_computed'])
                return np.zeros((count,10),dtype=np.float32)
            with patch.object(runner,'load_panel',side_effect=panel):
                selection,verification,timing=runner.finish_reporting(run,branches,labels)
            self.assertEqual(events,['validation']*888+['reporting']*888)
            before=json.loads((Path(temp)/'validation-complete.json').read_text())
            self.assertTrue(all('reporting' not in row for b in before['branches'] for row in b['metrics']))
            self.assertTrue(all('reporting' in row for b in branches for row in b['metrics']))
            self.assertGreaterEqual(timing['selection_seconds'],0.)
            with self.assertRaises(FileExistsError):
                run.save('selection.json',{})

    def test_selection_failure_never_scores_reporting(self):
        branches=self.branches()
        labels={s:{'validation':np.array([0,1],dtype=np.int64),'reporting':np.array([0,1],dtype=np.int64)} for s in runner.SEEDS}
        with tempfile.TemporaryDirectory() as temp:
            run=runner.Run(temp,device='cpu')
            run.check=lambda:None
            with patch.object(runner,'verify_selection',side_effect=RuntimeError('tampered')),patch.object(runner,'load_panel') as load:
                with self.assertRaisesRegex(RuntimeError,'tampered'):
                    runner.finish_reporting(run,branches,labels)
                load.assert_not_called()
            self.assertFalse((Path(temp)/'selection-verification.json').exists())

    def test_archive_caps_and_exclusive_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            run=runner.Run(temp,device='cpu')
            run.check=lambda:None
            with patch.object(runner,'MAX_BYTES',256),patch.object(runner,'RESERVE_BYTES',32):
                run.save('small.json',{'x':1})
                with self.assertRaisesRegex(RuntimeError,'byte cap'):
                    run.save('overflow.json',{'x':'a'*1000})
                self.assertLessEqual(sum(p.stat().st_size for p in Path(temp).iterdir()),224)
                with self.assertRaises(FileExistsError):
                    run.save('small.json',{})
            with self.assertRaises(RuntimeError):
                run.save('../outside.json',{})

    def test_receipt_tamper_and_npz_step_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            run=runner.Run(temp,device='cpu')
            run.check=lambda:None
            values={'step':np.array([100],dtype=np.int64),**{k:np.zeros((2,10),dtype=np.float32) for k in ('train','validation','reporting')}}
            receipt=run.save('logits-fixture-h00100.npz',values,'npz')
            self.assertEqual(runner.load_panel(temp,receipt,'validation',2).shape,(2,10))
            with self.assertRaises(RuntimeError):
                runner.read_receipt(temp,{**receipt,'sha256':'0'*64})
            receipt=run.save('logits-fixture-h00200.npz',values,'npz')
            with self.assertRaisesRegex(RuntimeError,'step/name'):
                runner.load_panel(temp,receipt,'validation',2)

    def test_resources_and_inert_cli(self):
        effective={'memory.max':str(16*1024**3),'memory.swap.max':'0','cpu.max':'100000 100000'}
        service={'Type':'exec','RuntimeMaxUSec':'3h 30min','Restart':'no','KillMode':'control-group'}
        runner.validate_bounds(effective,service)
        for bad in ({**effective,'cpu.max':'200000 200000'},{**effective,'memory.swap.max':'1'}):
            with self.assertRaises(RuntimeError):
                runner.validate_bounds(bad,service)
        with self.assertRaises(RuntimeError):
            runner.validate_bounds(effective,{**service,'RuntimeMaxUSec':'20min'})
        with patch.object(runner,'configure') as config,patch.object(runner,'source_pins') as pins:
            with self.assertRaisesRegex(RuntimeError,'explicit --execute'):
                runner.main(['--output-dir','/unread/acquisition-001'])
            config.assert_not_called()
            pins.assert_not_called()

    def test_fixed_summary_contrasts_and_interaction(self):
        branches=self.branches()
        endpoints={('raw','none'):.5,('raw','translate'):.52,
                   ('native200','none'):.6,('native200','translate'):.61}
        for branch in branches:
            for row in branch['metrics']:
                accuracy=.4 if row['step']<=100 else endpoints[branch['policy'],branch['augmentation']]
                for panel in ('train_true','train_assigned','wrong_true','wrong_assigned','reporting'):
                    row[panel]={'accuracy':accuracy,'ce':2.-accuracy}
            branch['selection']={'selected_index':0}
        summary=runner.summarize(branches)
        key='native200/translate minus raw/translate'
        for metric in ('accuracy','ce'):
            self.assertAlmostEqual(summary['reporting_contrasts'][key]['endpoint'][metric]['mean'],.09)
            self.assertAlmostEqual(summary['reporting_contrasts'][key]['selected'][metric]['mean'],0.)
            self.assertAlmostEqual(summary['reporting_augmentation_interaction']['endpoint'][metric]['mean'],-.01)
            self.assertAlmostEqual(summary['reporting_augmentation_interaction']['late61_72'][metric]['mean'],-.01)


if __name__=='__main__':
    unittest.main()
