"""Fabricated saved-array audit and adversarial fixtures; no scientific acquisition."""
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiments import spectral_strong_augmentation_audit as audit


def producer_summary(branches,d):
    """Fabricate producer-shaped summaries from fixture rows, not saved science."""
    indexed={(b['seed'],b['policy'],b['augmentation']):b for b in branches}
    panels=('train_true','train_assigned','wrong_true','wrong_assigned','validation','reporting')
    windows=('endpoint','warmup','selected','late61_72')
    def at(b,w,p,m):
        rows=b['metrics']
        if w=='selected':
            return rows[b['selection']['selected_index']][p][m]
        if w=='late61_72':
            return sum(r[p][m] for r in rows[2:])/d.epochs
        return rows[-1 if w=='endpoint' else 1][p][m]
    def est(values):
        return {'values':values,'mean':sum(values)/len(values)}
    groups={p+'/'+a:{w:{panel:{m:est([at(indexed[s,p,a],w,panel,m) for s in audit.SEEDS])
            for m in ('accuracy','ce')} for panel in panels} for w in windows} for p,a in audit.ARMS}
    progress={p+'/'+a:{m:est([sign*(at(indexed[s,p,a],'endpoint','reporting',m)
               -at(indexed[s,p,a],'warmup','reporting',m)) for s in audit.SEEDS])
               for m,sign in (('accuracy',1),('ce',-1))} for p,a in audit.ARMS}
    pairs=(('native200','translate','raw','translate'),('native200','none','raw','none'),
           ('raw','translate','raw','none'),('native200','translate','native200','none'))
    contrasts={f'{lp}/{la} minus {rp}/{ra}':{w:{m:est([sign*(at(indexed[s,lp,la],w,'reporting',m)
               -at(indexed[s,rp,ra],w,'reporting',m)) for s in audit.SEEDS])
               for m,sign in (('accuracy',1),('ce',-1))} for w in windows} for lp,la,rp,ra in pairs}
    interaction={w:{m:est([sign*(at(indexed[s,'native200','translate'],w,'reporting',m)
               -at(indexed[s,'native200','none'],w,'reporting',m)-at(indexed[s,'raw','translate'],w,'reporting',m)
               +at(indexed[s,'raw','none'],w,'reporting',m)) for s in audit.SEEDS])
               for m,sign in (('accuracy',1),('ce',-1))} for w in windows}
    return {'seeds':list(audit.SEEDS),'groups':groups,'reporting_progress_from100':progress,
            'reporting_contrasts':contrasts,'reporting_augmentation_interaction':interaction}


class StrongAuditFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)/'archive'
        self.directory.mkdir()
        self.source_root=Path(self.temp.name)/'source'
        self.d=audit.Dimensions(train=20,validation=10,reporting=10,epochs=2,batch=8,warmup=1)
        self.labels=np.tile(np.arange(10,dtype=np.int64),4)
        self.receipts=[]
        self.pins={}
        for name in audit.required_sources():
            payload=(audit.ROOT/name).read_bytes()
            target=self.source_root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload)
            self.pins[name]=audit.base.sha(payload)
        self.results=self.make_archive()

    def save(self,name,value,kind='json'):
        if kind=='npz':
            buffer=io.BytesIO()
            np.savez(buffer,**value)
            payload=buffer.getvalue()
        elif kind=='bytes':
            payload=value
        else:
            payload=json.dumps(value,allow_nan=False).encode()
        (self.directory/name).write_bytes(payload)
        receipt={'path':name,'size_bytes':len(payload),'sha256':audit.base.sha(payload)}
        self.receipts.append(receipt)
        return receipt

    def make_archive(self):
        d=self.d
        attempt={'schema':audit.SCHEMA,'commit':'a'*40,'source_pins':self.pins,'data_pins':audit.base.DATA_PINS,
                 'unit':audit.UNIT,'output_dir':str(self.directory.resolve()),'started_utc':'2026-09-10T00:00:00Z',
                 'pid':123,'argv':['fixture','--execute']}
        self.save('provenance.json',attempt)
        (self.source_root/audit.DOCS_REL/'attempt.json').write_text(json.dumps(attempt))
        branches=[]
        for seed in audit.SEEDS:
            plan=audit.regenerate_plan(self.labels,seed,d)
            self.save(f'plan-s{seed}.npz',plan,'npz')
            self.save(f'plan-s{seed}.json',{'seed':seed,'numpy_version':np.__version__,
                      'array_hashes':{k:audit.array_digest(v) for k,v in plan.items()},
                      'corruption_selected':int(plan['corruption_mask'].sum()),'actually_wrong':int(plan['changed_mask'].sum()),
                      'true_class_counts':{role:np.bincount(plan[role+'_labels'],minlength=10).tolist() for role in audit.PANELS}})
            self.save(f'initial-s{seed}.pt',b'opaque initial; never unpickle','bytes')
            for cell in (r for r in audit.roster() if r['seed']==seed):
                p,a=cell['policy'],cell['augmentation']
                name=f's{seed}-{p}-{a}'
                rows=[]
                for step in d.eval_steps:
                    arrays={'step':np.array([step],dtype=np.int64)}
                    for role in audit.PANELS:
                        truth=plan[role+'_labels']
                        values=np.zeros((len(truth),10),dtype=np.float32)
                        strength=0. if step<=d.warmup else step*.1+(audit.ARMS.index((p,a))+1)*.03
                        values[np.arange(len(truth)),truth]=np.float32(strength)
                        arrays[role]=values
                    receipt=self.save(f'logits-{name}-h{step:05d}.npz',arrays,'npz')
                    mask=plan['changed_mask']
                    # Producer uses a different stable-logsumexp implementation.
                    metric=audit.base.classification
                    rows.append({'step':step,'readout_receipt':receipt,
                         'train_true':metric(arrays['train'],plan['train_labels']),
                         'train_assigned':metric(arrays['train'],plan['assigned_labels']),
                         'wrong_true':metric(arrays['train'][mask],plan['train_labels'][mask]),
                         'wrong_assigned':metric(arrays['train'][mask],plan['assigned_labels'][mask]),
                         'validation':metric(arrays['validation'],plan['validation_labels']),
                         'reporting':metric(arrays['reporting'],plan['reporting_labels'])})
                native=p=='native200'
                stream={**{k:np.ones(d.updates,dtype=np.float64) for k in audit.FLOATS},
                        'adam_steps':np.arange(1,d.updates+1,dtype=np.int64),
                        'observer_steps':np.arange(1,d.updates+1,dtype=np.int64) if native else np.zeros(d.updates,dtype=np.int64),
                        'basis_rank':np.full(d.updates,2 if native else 0,dtype=np.int64)}
                b={**cell,'name':name,'metrics':rows,'initial_model_sha256':'1'*64,'warmup_learning_sha256':'2'*64,
                   'gradient_evaluation_count':d.updates,'training_example_count':d.epochs*d.train,
                   'training_seconds':1.,'augmentation_seconds':.1,'evaluation_seconds':.1,'wall_seconds':2.}
                for field,prefix in (('warmup_receipt','warmup'),('final_receipt','final')):
                    b[field]=self.save(f'{prefix}-{name}.pt',b'opaque fixture','bytes')
                b['stream_receipt']=self.save(f'stream-{name}.npz',stream,'npz')
                branches.append(b)
        stripped=[{k:([{q:v for q,v in r.items() if q!='reporting'} for r in value] if k=='metrics' else value)
                   for k,value in b.items()} for b in branches]
        self.save('validation-complete.json',{'schema':audit.SCHEMA,'branches':stripped,'reporting_metrics_computed':False})
        choices=[]
        for b in branches:
            index=min(range(len(b['metrics'])),key=lambda i:(b['metrics'][i]['validation']['ce'],b['metrics'][i]['step']))
            r=b['metrics'][index]
            choice={k:b[k] for k in ('name','seed','policy','augmentation')}
            choice.update(selected_index=index,selected_step=r['step'],selected_example_count=d.epochs*d.train,
                          validation_ce=r['validation']['ce'],readout_receipt=r['readout_receipt'])
            choices.append(choice)
            b['selection']=choice
        selected=self.save('selection.json',{'schema':'spectral_strong_selection_v1','rule':'minimum_float64_validation_ce_earliest_exact_tie',
                  'reporting_metrics_computed':False,'choices':choices,'created_utc':'2026-09-10T00:00:01Z'})
        verified=self.save('selection-verification.json',{'status':'PASS','selection_receipt':selected,'verified_choices':12,
                           'reporting_metrics_computed':False,'verified_utc':'2026-09-10T00:00:02Z'})
        result={'schema':audit.SCHEMA,'status':'complete','source_pins':self.pins,'data_pins':audit.base.DATA_PINS,
                'roster':audit.roster(),'eval_steps':list(d.eval_steps),'branches':branches,
                'selection_receipt':selected,'selection_verification_receipt':verified,'receipts':self.receipts,
                'gradient_evaluation_count':12*d.updates,'training_example_count':12*d.epochs*d.train,
                'phase_timings':{'reporting_started_utc':'2026-09-10T00:00:03Z'},'summary':producer_summary(branches,d)}
        self.write_results(result)
        return result

    def write_results(self,result=None):
        (self.directory/'results.json').write_text(json.dumps(result or self.results,allow_nan=False))

    def run_audit(self):
        return audit.audit_saved(self.directory,self.labels,dimensions=self.d,source_root=self.source_root)

    def rewrite_npz(self,name,mutator):
        with np.load(self.directory/name,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        mutator(arrays)
        buf=io.BytesIO()
        np.savez(buf,**arrays)
        payload=buf.getvalue()
        (self.directory/name).write_bytes(payload)
        receipt=next(r for r in self.receipts if r['path']==name)
        receipt.update(size_bytes=len(payload),sha256=audit.base.sha(payload))
        # Rebind the pre-report stream receipt too, so stream fixtures exercise
        # semantic counter reconstruction rather than an earlier hash mismatch.
        before_path=self.directory/'validation-complete.json'
        before=json.loads(before_path.read_text())
        for branch in before['branches']:
            if branch['stream_receipt']['path']==name:
                branch['stream_receipt']=receipt.copy()
        before_bytes=json.dumps(before,allow_nan=False).encode()
        before_path.write_bytes(before_bytes)
        before_receipt=next(r for r in self.receipts if r['path']=='validation-complete.json')
        before_receipt.update(size_bytes=len(before_bytes),sha256=audit.base.sha(before_bytes))
        self.write_results()

    def test_complete_archive_and_independent_arithmetic(self):
        report=self.run_audit()
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['verified_artifacts'],len(audit.expected_files(self.d)))
        self.assertEqual(report['logit_artifacts_read_twice'],48)
        self.assertEqual(report['validation_choices_verified_before_reporting'],12)
        self.assertLess(report['max_absolute_scalar_error'],1e-10)
        self.assertGreater(report['summary']['contrasts']['final']['native200/translate-raw/translate']['ce']['mean'],0.)

    def test_plan_rehashed_tamper(self):
        self.rewrite_npz(f'plan-s{audit.SEEDS[0]}.npz',lambda a:a['shifts'].fill(0))
        with self.assertRaisesRegex(audit.base.AuditError,'plan reconstruction'):
            self.run_audit()

    def test_opaque_tamper(self):
        path=self.directory/f'initial-s{audit.SEEDS[0]}.pt'
        payload=bytearray(path.read_bytes())
        payload[0]^=1
        path.write_bytes(payload)
        with self.assertRaisesRegex(audit.base.AuditError,'hash/size'):
            self.run_audit()

    def test_extra_file(self):
        (self.directory/'extra').write_bytes(b'x')
        with self.assertRaisesRegex(audit.base.AuditError,'inventory'):
            self.run_audit()

    def test_reporting_metric_tamper(self):
        self.results['branches'][0]['metrics'][-1]['reporting']['ce']+=.01
        self.write_results()
        with self.assertRaisesRegex(audit.base.AuditError,'metric differs'):
            self.run_audit()

    def test_validation_phase_mismatch(self):
        self.results['branches'][0]['metrics'][0]['validation']['ce']+=.01
        self.write_results()
        with self.assertRaisesRegex(audit.base.AuditError,'pre-report snapshot'):
            self.run_audit()

    def test_stream_counter_rehashed(self):
        self.rewrite_npz(f'stream-s{audit.SEEDS[0]}-native200-none.npz',lambda a:a['observer_steps'].__setitem__(2,99))
        with self.assertRaisesRegex(audit.base.AuditError,'observer counters'):
            self.run_audit()

    def test_summary_tamper(self):
        self.results['summary']['reporting_augmentation_interaction']['endpoint']['ce']['mean']+=.01
        self.write_results()
        with self.assertRaisesRegex(audit.base.AuditError,'summary scalar'):
            self.run_audit()

    def test_selection_exact_tie_and_reject_reporting(self):
        rows=[{'step':0,'ce':2.},{'step':1,'ce':1.},{'step':3,'ce':1.}]
        self.assertEqual(audit.validation_choice(rows),1)
        rows[0]['reporting_ce']=0.
        with self.assertRaisesRegex(audit.base.AuditError,'validation-only schema'):
            audit.validation_choice(rows)

    def test_uniform_offset_exact_tie_and_near_tie(self):
        labels=np.array([0,1],dtype=np.int64)
        zero=np.zeros((2,10),dtype=np.float32)
        offset=np.full((2,10),2.,dtype=np.float32)
        first=audit.canonical_selection_ce(zero,labels)
        second=audit.canonical_selection_ce(offset,labels)
        self.assertEqual(first,second)
        self.assertEqual(first,audit.base.classification(zero,labels)['ce'])
        self.assertEqual(audit.validation_choice([{'step':0,'ce':first},{'step':1,'ce':second}]),0)
        offset[np.arange(2),labels]+=np.float32(1e-5)
        better=audit.canonical_selection_ce(offset,labels)
        self.assertLess(better,first)
        self.assertEqual(audit.validation_choice([{'step':0,'ce':first},{'step':1,'ce':better}]),1)

    def test_source_attempt_tamper(self):
        path=self.source_root/audit.DOCS_REL/'attempt.json'
        value=json.loads(path.read_text())
        value['commit']='b'*40
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(audit.base.AuditError,'attempt/provenance'):
            self.run_audit()

    def test_data_plan_matches_independent_helper(self):
        from experiments import spectral_strong_augmentation_data as producer
        dimensions=producer.Dimensions(source_count=40,train_count=20,validation_count=10,reporting_count=10,epochs=2,batch_size=8)
        for seed in audit.SEEDS:
            own=audit.regenerate_plan(self.labels,seed,self.d)
            other=producer.make_plan(self.labels,seed,dimensions)
            self.assertEqual(set(own),set(other))
            for key in own:
                self.assertEqual(own[key].dtype,other[key].dtype)
                np.testing.assert_array_equal(own[key],other[key])


if __name__=='__main__':
    unittest.main()
