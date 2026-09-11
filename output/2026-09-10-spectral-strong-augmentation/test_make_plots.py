"""Fabricated scalar fixtures only; no production input reads or PNG writes."""
import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('strong_scalar_renderer',Path(__file__).with_name('make_plots.py'))
plots=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def metric(count,accuracy=.5,ce=1.):
    correct=int(count*accuracy)
    return {'count':count,'correct':correct,'accuracy':correct/count,'ce':ce,'ce_sum':ce*count}


def fixture():
    branches=[]
    for si,seed in enumerate(plots.SEEDS):
        for policy,augmentation in plots.ARMS:
            name=f's{seed}-{policy}-{augmentation}'
            rows=[]
            for i,step in enumerate(plots.STEPS):
                rows.append({'step':step,'train_true':metric(50000),'train_assigned':metric(50000),
                             'wrong_true':metric(40000+si),'wrong_assigned':metric(40000+si),
                             'validation':metric(5000,ce=.3 if i==3 else 1.+i*.01),
                             'reporting':metric(5000,accuracy=.99 if i==10 else .2+i*.001,ce=2.-i*.01)})
            selected={'name':name,'seed':seed,'policy':policy,'augmentation':augmentation,
                      'selected_index':3,'selected_step':plots.STEPS[3],
                      'selected_example_count':plots.EXPOSURES[3],'validation_ce':.3,
                      'readout_receipt':{'path':f'logits-{name}-h{plots.STEPS[3]:05d}.npz',
                                         'size_bytes':2401024,'sha256':'a'*64}}
            branches.append({'name':name,'seed':seed,'policy':policy,'augmentation':augmentation,
                             'metrics':rows,'selection':selected,'gradient_evaluation_count':56304,
                             'training_example_count':3600000})
    return {'schema':plots.AUDIT_SCHEMA,'status':'PASS','trajectories':12,
            'logical_evaluation_records':888,'validation_choices_verified_before_reporting':12,
            'gradient_evaluation_count':675648,'training_example_count':43200000,
            'summary':{'seeds':list(plots.SEEDS)},'branches':branches,
            'input_dir':'/fabricated/not-opened/archive'}


class PlotFixtures(unittest.TestCase):
    def test_full_fixture_and_exposure_map(self):
        self.assertEqual(len(plots.validate_audit(fixture())),12)
        self.assertEqual(len(plots.EXPOSURES),74)
        self.assertEqual(plots.EXPOSURES[:4],(0,6400,50000,100000))
        self.assertEqual(plots.EXPOSURES[-1],3600000)
        self.assertEqual(plots.STEPS[-1],56304)

    def test_uses_selected_index_not_reporting_best(self):
        indexed=plots.validate_audit(fixture())
        selected=plots.comparison_values(indexed,'selected','accuracy')
        first=indexed[plots.SEEDS[0],*plots.ARMS[0]]
        self.assertEqual(selected[0][0],first['metrics'][3]['reporting']['accuracy'])
        self.assertNotEqual(selected[0][0],first['metrics'][10]['reporting']['accuracy'])
        self.assertEqual(plots.comparison_values(indexed,'endpoint','ce')[0][0],first['metrics'][-1]['reporting']['ce'])

    def test_default_inert_and_hash_mandatory_on_execute(self):
        with patch.object(Path,'open',side_effect=AssertionError('no file reads')),patch.object(plots,'render') as render,patch('sys.stdout',new=io.StringIO()):
            self.assertEqual(plots.main([]),0)
            with self.assertRaisesRegex(ValueError,'audit-sha256'):
                plots.main(['--execute'])
            render.assert_not_called()

    def test_schema_status_and_roster_rejection(self):
        for key,value in (('schema','other'),('status','FAIL'),('logical_evaluation_records',887),('gradient_evaluation_count',True)):
            audit=fixture()
            audit[key]=value
            with self.assertRaises(ValueError):
                plots.validate_audit(audit)
        audit=fixture()
        audit['branches'].pop()
        with self.assertRaises(ValueError):
            plots.validate_audit(audit)
        audit=fixture()
        audit['branches'][-1]=copy.deepcopy(audit['branches'][0])
        with self.assertRaises(ValueError):
            plots.validate_audit(audit)

    def test_domains_counts_and_nonfinite(self):
        for panel,key,value in (('reporting','ce',float('nan')),('reporting','accuracy',1.2),
                                ('train_true','count',5000),('wrong_assigned','count',50000),
                                ('validation','ce_sum',-1.)):
            audit=fixture()
            audit['branches'][0]['metrics'][0][panel][key]=value
            with self.assertRaises(ValueError):
                plots.validate_audit(audit)

    def test_selection_mismatch_rejected(self):
        for key,value in (('selected_index',10),('selected_step',100),('selected_example_count',6400),
                          ('validation_ce',.1),('seed',202609173)):
            audit=fixture()
            audit['branches'][0]['selection'][key]=value
            with self.assertRaises(ValueError):
                plots.validate_audit(audit)

    def test_json_duplicates_and_nonfinite(self):
        for payload in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                plots.strict_json(payload)

    def test_audited_exact_tie_is_not_reselected_from_alternate_arithmetic(self):
        audit=fixture()
        branch=audit['branches'][0]
        # The frozen canonical selector chose index3. The audit's independent
        # metric reduction can round two equal losses in the opposite order.
        branch['metrics'][3]['validation']=metric(5000,ce=.3+1e-12)
        branch['metrics'][4]['validation']=metric(5000,ce=.3-1e-12)
        indexed=plots.validate_audit(audit)
        self.assertEqual(indexed[plots.SEEDS[0],*plots.ARMS[0]]['selection']['selected_index'],3)
        self.assertEqual(plots.comparison_values(indexed,'selected','accuracy')[0][0],
                         branch['metrics'][3]['reporting']['accuracy'])

    def test_fabricated_in_memory_render(self):
        # No PNG files, production audit, referenced path, or scientific array.
        images,version=plots.render(plots.validate_audit(fixture()),'0'*64)
        self.assertEqual(version,plots.MATPLOTLIB_VERSION)
        self.assertEqual(set(images),{plots.CURVES,plots.WRONG,plots.SELECTED})
        for image in images.values():
            self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertLessEqual(len(image),plots.PNG_CAP)


if __name__=='__main__':
    unittest.main()
