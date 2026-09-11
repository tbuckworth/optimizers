"""Fabricated complete archives and tamper checks; no model/data acquisition."""
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from experiments import spectral_multiview_audit as audit


class AuditFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)/'archive'
        self.directory.mkdir()
        self.dim=audit.Dimensions(per_class=1,updates=200,batch=2,eval_steps=(0,100,200))
        self.labels=np.repeat(np.arange(10,dtype=np.int64),4)
        self.receipts=[]
        self.source_root=Path(self.temp.name)/'source'
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
        branches=[]
        attempt={'schema':audit.SCHEMA,'source_pins':self.pins,'data_pins':audit.base.DATA_PINS,
                 'unit':audit.UNIT,'output_dir':str(self.directory.resolve()),'commit':'a'*40,
                 'started_utc':'2026-09-10T00:00:00Z'}
        self.save('provenance.json',attempt)
        (self.source_root/audit.DOCS_REL/'attempt.json').write_text(json.dumps(attempt))
        for seed in audit.SEEDS:
            plan=audit.regenerate_plan(self.labels,seed,self.dim)
            self.save(f'plan-s{seed}.npz',plan,'npz')
            self.save(f'plan-s{seed}.json',{'seed':seed,'numpy_version':np.__version__,
                      'array_hashes':{k:audit.array_digest(v) for k,v in plan.items()}})
            self.save(f'initial-s{seed}.pt',b'opaque fabricated initial, never unpickle','bytes')
            for roster in (r for r in audit.roster() if r['seed']==seed):
                policy=roster['policy']
                name=f's{seed}-{policy}'
                values=np.zeros((3,10,10),dtype=np.float32)
                values[-1,np.arange(10),np.arange(10)]=np.float32((audit.POLICIES.index(policy)+1)/4)
                logits={'steps':np.array(self.dim.eval_steps,dtype=np.int64),
                        **{k:values.copy() for k in audit.PANELS}}
                metrics=[{'step':step,**{k:audit.base.classification(logits[k][i],
                         plan['train_labels' if k=='train' else 'eval_labels']) for k in audit.PANELS}}
                         for i,step in enumerate(self.dim.eval_steps)]
                native=policy in ('native1','observer4')
                views=4 if policy in ('observer4','raw4') else 1
                stream={'losses':np.zeros((200,4),dtype=np.float64),
                        **{k:np.full(200,.1,dtype=np.float64) for k in audit.FLOATS},
                        'gradient_evaluations':np.full(200,views,dtype=np.int64),
                        'adam_steps':np.arange(1,201,dtype=np.int64),
                        'observer_steps':np.arange(1,201,dtype=np.int64) if native else np.zeros(200,dtype=np.int64),
                        'basis_rank':np.full(200,2 if native else 0,dtype=np.int64)}
                if not native:
                    stream['observation_norm'][:]=0
                branch={**roster,'name':name,'initial_model_sha256':'1'*64,
                        'warmup_learning_sha256':('2' if policy!='raw4' else '3')*64,
                        'metrics':metrics,'gradient_evaluation_count':200*views,
                        'training_seconds':1.,'augmentation_seconds':.1,'evaluation_seconds':.1,'wall_seconds':2.}
                for field,prefix in (('warmup_receipt','warmup'),('final_receipt','final')):
                    branch[field]=self.save(f'{prefix}-{name}.pt',b'opaque fixture','bytes')
                branch['logits_receipt']=self.save(f'logits-{name}.npz',logits,'npz')
                branch['stream_receipt']=self.save(f'stream-{name}.npz',stream,'npz')
                branches.append(branch)
        results={'schema':audit.SCHEMA,'status':'complete','source_pins':self.pins,
                 'data_pins':audit.base.DATA_PINS,'roster':audit.roster(),
                 'eval_steps':list(self.dim.eval_steps),'branches':branches,
                 'gradient_evaluation_count':6000,'receipts':self.receipts}
        self.write_results(results)
        return results

    def write_results(self,results=None):
        (self.directory/'results.json').write_text(json.dumps(results or self.results,allow_nan=False))

    def run_audit(self):
        return audit.audit_saved(self.directory,self.labels,dimensions=self.dim,source_root=self.source_root)

    def replace_npz(self,name,mutate):
        with np.load(self.directory/name,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        mutate(arrays)
        payload=io.BytesIO()
        np.savez(payload,**arrays)
        data=payload.getvalue()
        (self.directory/name).write_bytes(data)
        receipt=next(r for r in self.receipts if r['path']==name)
        receipt.update(size_bytes=len(data),sha256=audit.base.sha(data))
        self.write_results()

    def test_complete_archive_and_summary(self):
        report=self.run_audit()
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['verified_artifacts'],58)
        self.assertEqual(report['logical_evaluation_records'],36)
        self.assertEqual(report['panel_evaluations'],108)
        self.assertEqual(report['gradient_evaluation_count'],6000)
        self.assertEqual(report['max_absolute_scalar_error'],0.)
        contrast=report['summary']['endpoint_contrasts']['observer4-native1']['heldout']['ce']
        self.assertGreater(contrast['mean'],0.)
        self.assertEqual(len(contrast['values']),3)

    def test_extra_inventory_rejected(self):
        (self.directory/'unexpected').write_bytes(b'x')
        with self.assertRaisesRegex(audit.base.AuditError,'inventory'):
            self.run_audit()

    def test_opaque_checkpoint_tamper(self):
        path=self.directory/f'initial-s{audit.SEEDS[0]}.pt'
        payload=bytearray(path.read_bytes())
        payload[0]^=1
        path.write_bytes(payload)
        with self.assertRaisesRegex(audit.base.AuditError,'hash/size'):
            self.run_audit()

    def test_plan_wrong_stream_even_rehashed(self):
        self.replace_npz(f'plan-s{audit.SEEDS[0]}.npz',lambda a:a['extra_shifts'].fill(0))
        with self.assertRaisesRegex(audit.base.AuditError,'plan reconstruction'):
            self.run_audit()

    def test_counter_tamper_even_rehashed(self):
        self.replace_npz(f'stream-s{audit.SEEDS[0]}-observer4.npz',lambda a:a['observer_steps'].__setitem__(5,7))
        with self.assertRaisesRegex(audit.base.AuditError,'observer counters'):
            self.run_audit()

    def test_metrics_tamper(self):
        self.results['branches'][0]['metrics'][-1]['heldout']['ce']+=.01
        self.write_results()
        with self.assertRaisesRegex(audit.base.AuditError,'metric differs'):
            self.run_audit()

    def test_warmup_logit_tamper(self):
        self.replace_npz(f'logits-s{audit.SEEDS[0]}-observer4.npz',lambda a:a['heldout'].__setitem__((1,0,0),1.))
        with self.assertRaisesRegex(audit.base.AuditError,'warmup prediction pairing'):
            self.run_audit()

    def test_unused_loss_slot_and_nonfinite(self):
        self.replace_npz(f'stream-s{audit.SEEDS[0]}-raw1.npz',lambda a:a['losses'].__setitem__((0,1),.1))
        with self.assertRaisesRegex(audit.base.AuditError,'unused views'):
            self.run_audit()

    def test_duplicate_json_rejected(self):
        with self.assertRaises(audit.base.AuditError):
            audit.base.strict_json(b'{"status":"PASS","status":"FAIL"}')

    def test_attempt_binding_tamper(self):
        path=self.source_root/audit.DOCS_REL/'attempt.json'
        value=json.loads(path.read_text())
        value['commit']='b'*40
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(audit.base.AuditError,'attempt/provenance'):
            self.run_audit()

    def test_old_pin_tamper(self):
        self.results['source_pins']['spectral_filter.py']='0'*64
        self.write_results()
        with self.assertRaisesRegex(audit.base.AuditError,'old dependency pins'):
            self.run_audit()


if __name__=='__main__':
    unittest.main()
