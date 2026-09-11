"""Tiny synthetic equations only. No model, experiment artifact or optimizer."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from check_saved import Check,NAMES,projection,response,recompute,acceptance,aggregate_check,compare_tree,adam_residuals,EPS32


def fixture(p=3):
    pairs=np.column_stack((np.repeat(np.arange(p),p),np.tile(np.arange(p),p)))
    sums=pairs.sum(1)%p
    base=np.random.default_rng(7).normal(size=(p*p,p))*.1
    useful=np.eye(p)[sums]*.03
    nuisance=(pairs[:,0,None]-1)*np.array([.01,-.02,.01])[None,:]
    return {'before':base,'raw':base+useful+nuisance,'trunc':base+useful,
            'projected':base+2*useful,'zero':base-.01,
            'pairs':pairs,'sums':sums,'train_indices':np.arange(0,p*p,2),'test_indices':np.arange(1,p*p,2)}


class SavedFixtures(unittest.TestCase):
    def test_projection_orthogonal_and_not_subset(self):
        arrays=fixture();values=arrays['raw']
        projected=projection(values,arrays['sums'])
        self.assertTrue(np.allclose(projection(projected,arrays['sums']),projected))
        self.assertAlmostEqual(float(np.sum(projected*(values-projected))),0,places=12)
        with self.assertRaises(ValueError):projection(values[:-1],arrays['sums'][:-1])

    def test_known_usefulness_and_nonadditive_energy(self):
        arrays=fixture();metrics=recompute(arrays)
        check=Check();acceptance(check,metrics,arrays);self.assertEqual(check.errors,[])
        self.assertGreater(metrics['responses_from_before']['projected']['finite_improvement']['test']['ce'],0)
        contrast=metrics['contrasts']
        self.assertEqual(set(contrast),{'raw_to_trunc','trunc_to_projected','raw_to_projected','zero_to_raw','zero_to_trunc','zero_to_projected'})
        for split in ('train','test'):
            for value in metrics['finite_scalar_telescoping_residual'][split].values():self.assertLess(abs(value),1e-12)
        # Responses are not assumed to be additive squared magnitudes.
        self.assertGreater(metrics['responses_from_before']['projected']['energy']['total'],metrics['responses_from_before']['trunc']['energy']['total'])

    def test_classwise_constant_shift_has_zero_response(self):
        arrays=fixture()
        for name in NAMES[1:]:arrays[name]=arrays['before']+3.
        metrics=recompute(arrays)
        for value in metrics['responses_from_before'].values():
            self.assertLess(value['energy']['total'],1e-25)
            self.assertLess(abs(value['finite_improvement']['test']['ce']),1e-12)

    def test_energy_utility_decomposition_signed_component(self):
        arrays=fixture();change=arrays['raw']-arrays['before'];weight=np.eye(3)[arrays['sums']]-1/3
        measured=response(change,arrays['sums'],weight,{'train':arrays['train_indices'],'test':arrays['test_indices']})
        self.assertAlmostEqual(measured['energy']['total'],measured['energy']['sum_consistent']+measured['energy']['within_sum'],places=12)
        self.assertGreater(measured['linear_ce_utility']['test']['total'],0)

    def test_saved_scalar_comparison_detects_fabrication(self):
        expected=recompute(fixture());actual=copy.deepcopy(expected)
        actual['behavior']['raw']['test']['ce']+=1
        check=Check();compare_tree(check,actual,expected);self.assertTrue(check.errors)

    def test_exact_five_seed_signs_and_se(self):
        values=[-2.,-1.,0.,1.,2.]
        aggregate={'values':values,'mean':0.,'sample_se':math.sqrt(.5),'positive_count':2,'negative_count':2,'zero_count':1}
        check=Check();aggregate_check(check,aggregate,values);self.assertEqual(check.errors,[])
        aggregate['positive_count']=3
        with self.assertRaises(ValueError):aggregate_check(Check(),aggregate,values)

    def test_small_residual_signs_preserved_from_saved_values(self):
        values=[-1e-18,1e-18,0.,2e-18,-2e-18]
        aggregate={'values':values,'mean':0.,'sample_se':math.sqrt(.5)*1e-18,'positive_count':2,'negative_count':2,'zero_count':1}
        check=Check();aggregate_check(check,aggregate,values);self.assertEqual(check.errors,[])

    def test_adam_zero_and_nonzero_pure_recurrences(self):
        group={'params':[0],'betas':(.9,.98),'lr':.001,'weight_decay':1.,'eps':1e-8,'amsgrad':False,'maximize':False}
        before={'parameter_identity':[{'name':'w','shape':[2]}],'model_state':{'w':torch.tensor([1.,-2.])},
                'optimizer_state':{'param_groups':[group],'state':{0:{'step':torch.tensor(1500.),'exp_avg':torch.tensor([.2,-.1]),'exp_avg_sq':torch.tensor([.4,.3])}}}}
        for action in (torch.zeros(2),torch.tensor([.5,-.3])):
            after=copy.deepcopy(before);old=before['optimizer_state']['state'][0];new=after['optimizer_state']['state'][0]
            new['step']=torch.tensor(1501.)
            new['exp_avg']=(.9*old['exp_avg'].double()+.1*action.double()).float()
            new['exp_avg_sq']=(.98*old['exp_avg_sq'].double()+.02*action.double().square()).float()
            direction=new['exp_avg'].double()/(1-.9**1501)/((new['exp_avg_sq'].double()/(1-.98**1501)).sqrt()+1e-8)
            after['model_state']['w']=(.999*before['model_state']['w'].double()-.001*direction).float()
            self.assertLess(max(adam_residuals(before,after,action).values()),32*EPS32)
            self.assertFalse(torch.equal(before['model_state']['w'],after['model_state']['w']))
        after['optimizer_state']['state'][0]['step']=torch.tensor(1500.)
        with self.assertRaises(ValueError):adam_residuals(before,after,action)


if __name__=='__main__':unittest.main()
