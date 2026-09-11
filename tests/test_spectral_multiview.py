"""Fabricated scalar fixtures; no models, scientific data or GPU execution."""
import unittest
from unittest.mock import patch

import numpy as np
from experiments import spectral_multiview as runner


class MultiviewRunnerFixtures(unittest.TestCase):
    def record(self, policy, step=1):
        views = 4 if policy in ('observer4','raw4') else 1
        native = policy in ('native1','observer4')
        return {'losses': np.array([.5]*views+[0.]*(4-views), dtype=np.float64),
                'observation_norm': .3 if native else 0., 'delivery_norm': .4,
                'applied_norm': .2, 'data_step_norm': .001,
                'gradient_evaluations': views, 'observer_steps': step if native else 0,
                'adam_steps': step, 'basis_rank': 2 if native else 0}

    def test_roster_fixed_rotation(self):
        rows = runner.branch_roster()
        self.assertEqual(len(rows),12)
        self.assertEqual(len({(r['seed'],r['policy']) for r in rows}),12)
        for i, seed in enumerate(runner.SEEDS):
            self.assertEqual([r['policy'] for r in rows if r['seed']==seed],
                             list(runner.POLICIES[i:]+runner.POLICIES[:i]))
        self.assertEqual(sum(4000*(4 if r['policy'] in ('observer4','raw4') else 1) for r in rows),120000)

    def test_stream_schema_and_records(self):
        for policy in runner.POLICIES:
            stream = runner.empty_stream()
            self.assertEqual(set(stream),{'losses',*runner.FLOAT_STREAMS,*runner.INT_STREAMS})
            self.assertEqual(stream['losses'].shape,(4000,4))
            for key in runner.FLOAT_STREAMS:
                self.assertEqual(stream[key].dtype,np.float64)
            for key in runner.INT_STREAMS:
                self.assertEqual(stream[key].dtype,np.int64)
            row = self.record(policy)
            runner.record_step(stream,0,row,policy)
            for key in row:
                np.testing.assert_array_equal(stream[key][0],row[key])
            self.assertFalse(stream['losses'][1:].any())

    def test_loss_zero_valid_and_invalid_values(self):
        row = self.record('raw1')
        row['losses'][:] = 0
        runner.record_step(runner.empty_stream(),0,row,'raw1')
        for bad in (-1.,float('nan'),float('inf')):
            row = self.record('raw1')
            row['losses'][0] = bad
            with self.assertRaises(RuntimeError):
                runner.record_step(runner.empty_stream(),0,row,'raw1')
        row = self.record('raw1')
        row['losses'][1] = .1
        with self.assertRaises(RuntimeError):
            runner.record_step(runner.empty_stream(),0,row,'raw1')

    def test_counter_and_norm_errors(self):
        for key,value in (('gradient_evaluations',4),('adam_steps',2),('observer_steps',1),
                          ('basis_rank',1),('data_step_norm',-1.),('delivery_norm',float('nan'))):
            row = self.record('raw1')
            row[key] = value
            with self.assertRaises(RuntimeError):
                runner.record_step(runner.empty_stream(),0,row,'raw1')
        for bad in (True,1.):
            row = self.record('native1')
            row['adam_steps'] = bad
            with self.assertRaises(RuntimeError):
                runner.record_step(runner.empty_stream(),0,row,'native1')
        with self.assertRaises(RuntimeError):
            runner.record_step(runner.empty_stream(),0,{**self.record('raw1'),'extra':0},'raw1')

    def test_counter_update100_and101(self):
        for policy in runner.POLICIES:
            stream = runner.empty_stream()
            for step in (100,101,4000):
                runner.record_step(stream,step-1,self.record(policy,step),policy)
                self.assertEqual(stream['adam_steps'][step-1],step)

    def test_guard_no_execute_no_admission(self):
        with patch.object(runner,'configure') as configure, patch.object(runner,'source_pins') as pins:
            with self.assertRaisesRegex(RuntimeError,'explicit --execute'):
                runner.main(['--output-dir','/does-not-exist/acquisition-001'])
            configure.assert_not_called()
            pins.assert_not_called()

    def test_inventory_and_resource_identity(self):
        inventory = runner.byte_inventory()
        self.assertEqual(inventory['cap_bytes'],1024**3)
        self.assertLess(inventory['total_upper_bytes']+runner.base.RESERVE_BYTES,1024**3)
        self.assertEqual(runner.UNIT,'spectral-multiview-clean-001.service')
        self.assertEqual(runner.base.DEADLINE_SECONDS,900)
        self.assertEqual(runner.EVAL_STEPS,(0,100)+tuple(range(200,4001,200)))
        self.assertEqual(len(runner.EVAL_STEPS),22)
        bounds={'memory.max':str(16*1024**3),'memory.swap.max':'0','cpu.max':'100000 100000'}
        service={'Type':'exec','RuntimeMaxUSec':'20min','Restart':'no','KillMode':'control-group'}
        runner.base.validate_bounds(bounds,service)
        with self.assertRaises(RuntimeError):
            runner.base.validate_bounds(bounds,{**service,'Restart':'always'})


if __name__=='__main__':
    unittest.main()
