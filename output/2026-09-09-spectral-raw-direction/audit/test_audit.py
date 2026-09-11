"""Synthetic-only fixtures: no real artifact access, model or producer import."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from common import Audit
from audit_tensors_histories import norm_law, cosine, tree_hash, vector, audit_adam_scalars
from readout_math import behavior, probe, defect, pooled, paired_stats
from audit_readouts import summary_row, paired, METRICS, CONTRASTS, POLICY, SEEDS


def flags(direction,target,delivered,prefix='raw'):
    denominator=max(direction,1e-30)
    absolute=abs(delivered-target)
    data={'scale':target/denominator,'absolute_mismatch':absolute,
          'relative_mismatch':absolute/max(target,1e-30),'relative_tolerance':10*2**-23,
          'degenerate':direction<=1e-30,'denominator_clamped':direction<1e-30,
          'exact':direction>=1e-30 or (direction==0 and target==0)}
    return {prefix+'_norm_match_'+key:value for key,value in data.items()}


class AuditFixtures(unittest.TestCase):
    def test_norm_domains_are_not_tolerance_flags(self):
        for direction,target,delivered in ((2.,6.,6.),(0.,0.,0.),(1e-31,1.,.1),(2.,0.,0.)):
            audit=Audit();norm_law(audit,flags(direction,target,delivered),'raw',direction,target,delivered)
            self.assertEqual(audit.errors,[])
        audit=Audit();bad=flags(2.,6.,5.)
        self.assertTrue(bad['raw_norm_match_exact'])
        norm_law(audit,bad,'raw',2.,6.,5.)
        self.assertTrue(any('postcast tolerance' in x for x in audit.errors))

    def test_zero_raw_positive_target_rejected(self):
        with self.assertRaises(ValueError):
            norm_law(Audit(),flags(0.,1.,0.),'raw',0.,1.,0.)

    def test_distinct_raw_projected_directions_and_zero_cosine(self):
        raw=torch.tensor([3.,4.]);q=torch.tensor([3.,0.])
        self.assertAlmostEqual(cosine(raw,q),.6)
        self.assertIsNone(cosine(raw,torch.zeros(2)))
        self.assertNotEqual(tree_hash(raw),tree_hash(q))
        audit=Audit();vector(audit,raw,q,'not equal')
        self.assertTrue(audit.errors)

    def test_zero_action_carries_moments_and_decay_in_equations(self):
        m=torch.tensor([.2,-.3],dtype=torch.float64);v=torch.tensor([.4,.5],dtype=torch.float64)
        zero=torch.zeros(2,dtype=torch.float64);old=torch.tensor([1.,2.],dtype=torch.float64)
        m1=.9*m+.1*zero;v1=.98*v+.02*zero.square()
        direction=(m1/(1-.9**1501))/((v1/(1-.98**1501)).sqrt()+1e-8)
        movement=-.001*direction-.001*old
        self.assertGreater(float(movement.norm()),0)
        self.assertTrue(torch.equal(m1,.9*m))

    def test_later_adam_scalar_bounds_detect_misreport(self):
        scalars={'adaptive_direction_norm':100.,'adaptive_movement_norm':.1,'decay_movement_norm':.03,
                 'actual_displacement_norm':.11,'decomposition_residual_norm':1e-6,'decomposition_residual_max_abs':1e-7,
                 'm_norm':1.,'v_norm':2.}
        audit=Audit();audit_adam_scalars(audit,scalars,227313);self.assertEqual(audit.errors,[])
        scalars['actual_displacement_norm']=10
        audit_adam_scalars(audit,scalars,227313);self.assertTrue(audit.errors)

    def test_stable_ce_margin_and_accuracy(self):
        logits=np.array([[1000.,1000.],[1001.,1000.]])
        result=behavior(logits,np.array([0,1]),np.array([0,1]))
        self.assertAlmostEqual(result['loss'],(math.log(2)+math.log1p(math.e))/2)
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['correct_class_margin_mean'],-.5)

    def test_probe_known_fourier_signal_and_nulls(self):
        p=7;labels=np.tile(np.arange(p),4)
        angle=labels*2*np.pi/p
        x=np.column_stack((np.cos(angle),np.sin(angle)))
        fit=np.arange(14);ev=np.arange(14,28)
        permutation=np.random.default_rng(83).permutation(len(labels))
        result=probe(x,labels,fit,ev,[permutation],p=p,top_k=1)
        self.assertEqual(result['runs'][0]['selected_frequencies'],[1])
        self.assertGreater(result['runs'][0]['eval_r2'][0],.99)
        self.assertLess(result['runs'][1]['selected_eval_mean_r2'],.5)
        self.assertEqual(len(result['runs']),2)

    def test_symmetry_known_equivariance_and_undefined(self):
        p=7;pair=np.column_stack((np.repeat(np.arange(p),p),np.tile(np.arange(p),p)))
        logits=np.cos((np.arange(p)[None,:]-pair.sum(axis=1)[:,None])*2*np.pi/p)
        centered=logits-logits.mean(axis=1,keepdims=True)
        source=np.arange(p*p);target=((source//p+1)%p)*p+source%p
        edge=np.column_stack((source,target))
        good=defect(logits,centered,edge,1,p);bad=defect(logits,centered,edge,2,p)
        self.assertLess(good['value'],1e-25);self.assertGreater(bad['value'],.1)
        self.assertAlmostEqual(pooled([good,good],p)['value'],good['value'])
        empty=defect(np.zeros_like(logits),np.zeros_like(logits),edge,1,p)
        self.assertIsNone(empty['value']);self.assertFalse(empty['energy_defined'])

    def test_paired_sd_signs_and_missing_no_subset(self):
        result=paired_stats([1.,2.,3.,4.,5.],[2.,2.,2.,2.,2.],'lower')
        self.assertEqual(result['mean_difference'],1)
        self.assertAlmostEqual(result['sample_sd'],math.sqrt(2.5))
        self.assertEqual((result['positive_count'],result['negative_count'],result['zero_count'],result['favorable_count']),(3,1,1,1))
        missing=paired_stats([1.,None,3.,4.,5.],[2.]*5,'lower')
        self.assertEqual(missing['defined_count'],4)
        for key in ('mean_difference','sample_sd','sample_se','positive_count','negative_count','favorable_count'):
            self.assertIsNone(missing[key])
        self.assertIsNone(paired_stats([1.]*5,[0.]*5,'descriptive')['favorable_count'])

    def test_explicit_summary_schema_transformation(self):
        state={'seed':100,'behavior':{'test':{'loss':1.}},'full_symmetry':{'not_compact':1}}
        item={'raw':{'a':1},'scalar':{'b':2},'analyzed_state':{'c':3}}
        value=summary_row(state,item)
        self.assertNotIn('full_symmetry',value)
        self.assertEqual(value['source_state_receipt'],item['analyzed_state'])
        self.assertIn('full_symmetry',state)

    def test_full_paired_schema_and_tampered_arithmetic(self):
        def row(seed,policy,step,offset):
            value={'seed':seed,'policy':policy,'step':step}
            for path,_,_,_ in METRICS.values():
                target=value
                for key in path[:-1]:target=target.setdefault(key,{})
                target[path[-1]]=seed/100+offset
            return value
        new=[row(seed,POLICY,step,.2) for seed in SEEDS for step in (1501,2000,2500)]
        old_rows=[row(seed,policy,step,0) for seed in SEEDS for policy in ('orthogonal','norm_matched') for step in (1501,2000,2500)]
        native=[row(seed,'native',step,-.1) for seed in SEEDS for step in (1500,2000,2500)]
        old={'new_state_rows':old_rows,'archived_native_reference_rows':native}
        manifest={'metric_definitions':{name:{'path':list(path),'favorable_direction':direction,'role':role,'unit':unit}
                  for name,(path,direction,role,unit) in METRICS.items()}}
        summary={'schema':'grokking_raw_direction_analysis_summary_v1','new_state_count':15,
                 'new_state_rows':new,'archived_action_reference_rows':old_rows,
                 'archived_native_reference_rows':native,'paired_endpoint_contrasts':[]}
        for step in (2000,2500):
            for name,other,role in CONTRASTS:
                contrast={'step':step,'contrast':name,'left_policy':POLICY,'right_policy':other,
                          'contrast_role':role,'endpoint_role':'primary' if step==2500 else 'fixed_earlier','metrics':{}}
                for metric,(_,direction,metric_role,unit) in METRICS.items():
                    left=[s/100+.2 for s in SEEDS];right=[s/100+(-.1 if other=='native' else 0) for s in SEEDS]
                    actual=paired_stats(left,right,direction)
                    differences=actual.pop('differences')
                    actual.update(paired=[{'seed':s,'left':l,'right':r,'difference':d} for s,l,r,d in zip(SEEDS,left,right,differences)],
                                  n_total_seeds=5,difference_definition='left_minus_right',role=metric_role,unit=unit)
                    contrast['metrics'][metric]=actual
                summary['paired_endpoint_contrasts'].append(contrast)
        audit=Audit();paired(audit,summary,manifest,new,old);self.assertEqual(audit.errors,[])
        summary['paired_endpoint_contrasts'][0]['metrics']['heldout_cross_entropy']['mean_difference']=9.
        audit=Audit();paired(audit,summary,manifest,new,old);self.assertTrue(audit.errors)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
