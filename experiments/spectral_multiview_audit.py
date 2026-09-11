#!/usr/bin/env python3
"""Independent NumPy saved-artifact audit; no model or training replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import resource
import time

import numpy as np
from experiments import spectral_general_augmentation_audit as base

ROOT = Path(__file__).resolve().parents[1]
SCHEMA, AUDIT_SCHEMA = 'spectral_multiview_clean_v1', 'spectral_multiview_clean_audit_v1'
SEEDS = (202609161, 202609162, 202609163)
POLICIES = ('raw1', 'native1', 'observer4', 'raw4')
PANELS = ('train', 'heldout', 'heldout_translated')
FLOATS = ('observation_norm', 'delivery_norm', 'applied_norm', 'data_step_norm')
INTS = ('gradient_evaluations', 'observer_steps', 'adam_steps', 'basis_rank')
CONTRASTS = (('observer4','native1'), ('observer4','raw4'), ('observer4','raw1'),
             ('raw4','raw1'), ('native1','raw1'))
BASE_SHA = '9f2c38c48a9d0dc14c733930a1022fe094caf623cb10a524bf1a262612d2cb15'
UNIT = 'spectral-multiview-clean-001.service'
DOCS_REL = Path('output/2026-09-10-spectral-multiview-clean')
OLD_PINS = {
    'spectral_filter.py': '9280c7d360aef4c775baf0d781a5afd56e016e9cfb93ef829c3ae4b8cc3df943',
    'experiments/spectral_general_augmentation.py': 'abffd85aa30f350cc2a6c857758a315d878c27de170c1a4a2adee2f39101538e',
    'experiments/spectral_general_augmentation_data.py': '4404d59fe79e294abc946cd3a58963d1b6e84d65e47d29c33bcebbde7003396a',
    'experiments/spectral_general_augmentation_audit.py': BASE_SHA,
    'output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py':
        '706851828b7120bea0d7a41bd0fa75474ac800b2df71087941d0140c8d1970ec',
}
OUTPUT_CAP = 32*1024**2
require = base.require
Dimensions = base.Dimensions


def required_sources():
    return {'spectral_filter.py',
            'output/2026-09-06-spectral-optimizer-investigation/continuation/iteration-009/neural_core.py',
            'experiments/spectral_general_augmentation.py',
            'experiments/spectral_general_augmentation_data.py',
            'experiments/spectral_general_augmentation_audit.py',
            *{f'experiments/spectral_multiview{s}.py' for s in ('','_core','_data','_audit')},
            *{f'tests/test_spectral_multiview{s}.py' for s in ('','_core','_data','_audit')},
            'output/2026-09-10-spectral-multiview-clean/protocol.md',
            'output/2026-09-10-spectral-multiview-clean/implementation-check.md'}


def roster():
    return [{'seed': seed, 'policy': policy} for i,seed in enumerate(SEEDS)
            for policy in POLICIES[i:]+POLICIES[:i]]


def regenerate_plan(raw_labels, seed, dimensions=Dimensions()):
    plan = base.regenerate_plan(raw_labels, seed, dimensions)
    rng = lambda stream: np.random.Generator(np.random.PCG64(np.random.SeedSequence([stream,seed])))
    plan['extra_shifts'] = rng(3).integers(-2,3,(dimensions.updates,3,dimensions.batch,2),dtype=np.int8)
    plan['eval_shifts'] = rng(4).integers(-2,3,(dimensions.examples,2),dtype=np.int8)
    return plan


def array_digest(value):
    value = np.ascontiguousarray(value)
    header = json.dumps([value.dtype.str,list(value.shape)],separators=(',',':'))
    return base.sha(header.encode()+b'\n'+value.tobytes())


def expected_files():
    names = {'provenance.json'}
    for seed in SEEDS:
        names.update((f'initial-s{seed}.pt',f'plan-s{seed}.npz',f'plan-s{seed}.json'))
        for policy in POLICIES:
            name = f's{seed}-{policy}'
            names.update((f'warmup-{name}.pt',f'final-{name}.pt',f'logits-{name}.npz',f'stream-{name}.npz'))
    return names


def validate_stream(stream, policy, dimensions=Dimensions()):
    views = 4 if policy in ('observer4','raw4') else 1
    native = policy in ('native1','observer4')
    steps = np.arange(1,dimensions.updates+1,dtype=np.int64)
    require(np.all(stream['losses'] >= 0) and np.all(stream['losses'][:,views:] == 0), 'loss validity/unused views')
    require(np.all(stream['gradient_evaluations'] == views), 'gradient evaluation counters')
    require(np.array_equal(stream['adam_steps'],steps), 'Adam counters')
    require(np.array_equal(stream['observer_steps'],steps if native else np.zeros_like(steps)), 'observer counters')
    require(np.all((stream['basis_rank'] >= 0)&(stream['basis_rank'] <= 32)), 'rank bounds')
    for key in FLOATS:
        require(np.all(stream[key] >= 0), 'negative norm: '+key)
    if not native:
        require(np.all(stream['basis_rank'] == 0) and np.all(stream['observation_norm'] == 0), 'raw observer fields')
    # At warmup and for raw policies delivery is applied verbatim. Norms are
    # measured from the identical FP32 vector by the same FP64 norm operation.
    stop = min(100,dimensions.updates) if native else dimensions.updates
    require(np.array_equal(stream['delivery_norm'][:stop],stream['applied_norm'][:stop]), 'unfiltered delivery norm differs')
    if policy == 'native1':
        require(np.array_equal(stream['delivery_norm'],stream['observation_norm']), 'native1 observes delivery')
    return int(stream['gradient_evaluations'].sum())


def summarize(branches, endpoint):
    indexed = {(b['seed'],b['policy']):b for b in branches}
    require(len(indexed)==12 and set(indexed)=={(s,p) for s in SEEDS for p in POLICIES}, 'summary roster')
    groups,progress,contrasts,timing,work = {},{},{},{},{}
    for policy in POLICIES:
        groups[policy],progress[policy] = {},{}
        for panel in PANELS:
            groups[policy][panel],progress[policy][panel] = {},{}
            for metric,sign in (('accuracy',1),('ce',-1)):
                ends,warm = [],[]
                for seed in SEEDS:
                    rows = indexed[seed,policy]['metrics']
                    require(rows[-1]['step']==endpoint and rows[1]['step']==100,'fixed endpoint/warmup')
                    ends.append(rows[-1][panel][metric])
                    warm.append(rows[1][panel][metric])
                groups[policy][panel][metric] = base.estimate(ends)
                progress[policy][panel][metric] = base.estimate([sign*(e-w) for e,w in zip(ends,warm)])
        timing[policy] = {key:base.estimate([indexed[s,policy][key] for s in SEEDS])
                         for key in ('training_seconds','augmentation_seconds','evaluation_seconds','wall_seconds')}
        work[policy] = {'gradient_evaluations':base.estimate([indexed[s,policy]['gradient_evaluation_count'] for s in SEEDS])}
    for left,right in CONTRASTS:
        contrasts[left+'-'+right] = {panel:{metric:base.estimate([
            sign*(indexed[s,left]['metrics'][-1][panel][metric]-indexed[s,right]['metrics'][-1][panel][metric])
            for s in SEEDS]) for metric,sign in (('accuracy',1),('ce',-1))} for panel in PANELS}
    return {'seeds':list(SEEDS),'endpoint_step':endpoint,'warmup_step':100,'groups':groups,
            'warmup_progress':progress,'endpoint_contrasts':contrasts,'timings_seconds':timing,'work':work,
            'benefit_convention':'accuracy: left minus right; CE: right minus left; warmup progress uses endpoint versus update100',
            'independent_units':'three paired fresh training seeds, not views or epochs',
            'uncertainty':'descriptive all-seed values and arithmetic means; no significance or best-epoch selection',
            'timing_limit':'rotated sequential execution, not a randomized hardware benchmark'}


def audit_saved(input_dir, raw_labels, *, dimensions=Dimensions(), source_root=ROOT, check=lambda:None):
    started = time.monotonic()
    require(base.sha(Path(base.__file__).read_bytes()) == BASE_SHA,'reused independent reader changed')
    directory = Path(input_dir)
    require(directory.is_dir() and not directory.is_symlink(),'ordinary archive directory')
    path = directory/'results.json'
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= base.RESULTS_CAP,'results admission')
    payload = path.read_bytes()
    results = base.strict_json(payload)
    require(results['schema']==SCHEMA and results['status']=='complete','source status/schema')
    require(results['eval_steps']==list(dimensions.eval_steps) and 100 in dimensions.eval_steps,'evaluation schedule')
    require(results['data_pins']==base.DATA_PINS,'source dataset pins')
    pins = results['source_pins']
    require(type(pins) is dict and set(pins)==required_sources(),'exact source pins missing/differ')
    require(all(pins.get(k)==v for k,v in OLD_PINS.items()),'accepted old dependency pins differ')
    for name,expected in pins.items():
        check()
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts and base.is_hash(expected),'source pin schema')
        source = Path(source_root)/name
        require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(Path(source_root).resolve()),'source path confinement')
        require(base.sha(source.read_bytes())==expected,'source hash mismatch: '+name)
    require(results['roster']==roster(),'fixed rotated roster')
    branches = results['branches']
    require(type(branches) is list and len(branches)==12
            and [{'seed':b['seed'],'policy':b['policy']} for b in branches]==roster(),'ordered branch roster')
    table = base.validate_inventory(directory,results['receipts'])
    require(set(table)==expected_files() and len(table)==58,'fixed 58-artifact inventory')
    require(sum(r['size_bytes'] for r in table.values())+len(payload)<=base.INPUT_CAP,'total input cap')
    base.check_embedded_receipts(branches,table)
    reader = base.Reader(directory,table,check)
    provenance = base.strict_json(reader.load('provenance.json',retain=True))
    require(provenance['schema']==SCHEMA and provenance['source_pins']==pins
            and provenance['data_pins']==base.DATA_PINS,'provenance binding')
    require(provenance['unit']==UNIT and provenance['output_dir']==str(directory.resolve())
            and type(provenance['commit']) is str and re.fullmatch('[0-9a-f]{40}',provenance['commit']) is not None,
            'provenance unit/archive/commit binding')
    attempt_path = Path(source_root)/DOCS_REL/'attempt.json'
    require(attempt_path.is_file() and not attempt_path.is_symlink()
            and attempt_path.stat().st_size<=base.RESULTS_CAP,'attempt admission')
    attempt_payload = attempt_path.read_bytes()
    attempt = base.strict_json(attempt_payload)
    attempt_keys = {'schema','commit','source_pins','data_pins','unit','output_dir','started_utc'}
    require(set(attempt)==attempt_keys and all(attempt[k]==provenance[k] for k in attempt_keys),
            'guarded attempt/provenance identity differs')
    plans = {}
    for seed in SEEDS:
        expected = regenerate_plan(raw_labels,seed,dimensions)
        plan = reader.npz(f'plan-s{seed}.npz',{k:(v.dtype,v.shape) for k,v in expected.items()})
        require(all(np.array_equal(plan[k],v) for k,v in expected.items()),'independent plan reconstruction')
        require(not np.intersect1d(plan['train_ids'],plan['eval_ids']).size,'source split overlap')
        metadata = base.strict_json(reader.load(f'plan-s{seed}.json',retain=True))
        require(metadata['seed']==seed and metadata['numpy_version']==np.__version__
                and metadata['array_hashes']=={k:array_digest(v) for k,v in plan.items()},'plan metadata binding')
        plans[seed] = plan
    initial,warmup,initial_predictions,warmup_predictions = {},{},{},{}
    verified,max_error,total_count = [],0.,0
    n,u,ne = dimensions.examples,dimensions.updates,len(dimensions.eval_steps)
    for branch in branches:
        check()
        seed,policy = branch['seed'],branch['policy']
        name = f's{seed}-{policy}'
        require(branch['name']==name,'branch name')
        for field,prefix,suffix in (('logits_receipt','logits','.npz'),('stream_receipt','stream','.npz'),
                                    ('warmup_receipt','warmup','.pt'),('final_receipt','final','.pt')):
            require(branch[field]==table[f'{prefix}-{name}{suffix}'],'branch artifact receipt binding')
        for key in ('initial_model_sha256','warmup_learning_sha256'):
            require(base.is_hash(branch[key]),'producer state digest schema')
        require(seed not in initial or initial[seed]==branch['initial_model_sha256'],'initial state pairing')
        initial[seed]=branch['initial_model_sha256']
        if policy!='raw4':
            require(seed not in warmup or warmup[seed]==branch['warmup_learning_sha256'],'warmup learning pairing')
            warmup[seed]=branch['warmup_learning_sha256']
        timing = {}
        for key in ('training_seconds','augmentation_seconds','evaluation_seconds','wall_seconds'):
            timing[key]=base.finite_number(branch[key],key)
            require(timing[key]>=0,'negative timing')
        require(sum(timing[k] for k in ('training_seconds','augmentation_seconds','evaluation_seconds'))
                <=timing['wall_seconds']+1e-6,'timing intervals exceed wall time')
        logits = reader.npz(f'logits-{name}.npz',{'steps':(np.int64,(ne,)),
            **{k:(np.float32,(ne,n,10)) for k in PANELS}})
        require(np.array_equal(logits['steps'],dimensions.eval_steps),'logit steps')
        stream = reader.npz(f'stream-{name}.npz',{'losses':(np.float64,(u,4)),
            **{k:(np.float64,(u,)) for k in FLOATS},**{k:(np.int64,(u,)) for k in INTS}})
        count = validate_stream(stream,policy,dimensions)
        require(type(branch['gradient_evaluation_count']) is int and branch['gradient_evaluation_count']==count,'branch work count')
        total_count += count
        for panel in PANELS:
            key=(seed,panel)
            require(key not in initial_predictions or np.array_equal(initial_predictions[key],logits[panel][0]),'initial prediction pairing')
            initial_predictions[key]=logits[panel][0].copy()
            if policy!='raw4':
                require(key not in warmup_predictions or np.array_equal(warmup_predictions[key],logits[panel][1]),'warmup prediction pairing')
                warmup_predictions[key]=logits[panel][1].copy()
        require(type(branch['metrics']) is list and len(branch['metrics'])==ne,'metrics length')
        rebuilt=[]
        for i,step in enumerate(dimensions.eval_steps):
            check()
            actual=branch['metrics'][i]
            require(type(actual) is dict and set(actual)=={'step',*PANELS}
                    and type(actual['step']) is int and actual['step']==step,'metric schema/step')
            row={'step':step}
            for panel in PANELS:
                row[panel]=base.classification(logits[panel][i],plans[seed]['train_labels' if panel=='train' else 'eval_labels'])
                max_error=max(max_error,base.compare_metrics(actual[panel],row[panel]))
            rebuilt.append(row)
        verified.append({'seed':seed,'policy':policy,'name':name,'metrics':rebuilt,
                         'gradient_evaluation_count':count,**timing})
    require(total_count==30*dimensions.updates and type(results['gradient_evaluation_count']) is int
            and results['gradient_evaluation_count']==total_count,'total work count')
    for name in table:
        if name not in reader.read:
            reader.load(name)  # Opaque checkpoints: no unpickling.
    require(reader.read==set(table),'artifact read completion')
    return {'schema':AUDIT_SCHEMA,'status':'PASS','input_dir':str(directory.resolve()),
            'input_results_sha256':base.sha(payload),'source_pins':pins,'data_pins':base.DATA_PINS,
            'input_attempt_sha256':base.sha(attempt_payload),'source_commit':provenance['commit'],
            'verified_artifacts':len(reader.read),'input_bytes_read_once':reader.bytes+len(payload),
            'trajectories':12,'logical_evaluation_records':12*ne,'panel_evaluations':12*ne*3,
            'gradient_evaluation_count':total_count,'metric_atol':base.ATOL,'metric_rtol':base.RTOL,
            'max_absolute_scalar_error':max_error,'summary':summarize(verified,dimensions.eval_steps[-1]),
            'branches':verified,'wall_seconds':time.monotonic()-started,
            'limits':['No model, optimizer, autograd, observer or training replay.',
                      'Full checkpoint bytes are hashed only; model/Adam digest pairing remains a producer assertion.',
                      'Saved logits, metrics, plans, counts and pairing independently checked; actual use of plans is not replayed.',
                      'Pinned raw training-label IDX only; no image/test IDX or checkpoint deserialization.',
                      'Three paired fresh seeds; descriptive fixed-endpoint comparisons, no best-epoch selection.',
                      'Clean-only study cannot certify retained wrong-label protection or safety benefits.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args(argv)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key)=='1','set '+key+'=1')
    source=args.input_dir.resolve(strict=True)
    output=args.output_json.parent.resolve(strict=True)/args.output_json.name
    require(not output.is_relative_to(source) and not output.exists() and not output.is_symlink(),'exclusive output outside archive')
    started=time.monotonic()
    def check():
        require(time.monotonic()-started<300,'300-second cooperative audit deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'2GiB audit memory cap')
    with output.open('xb') as handle:
        try:
            report=audit_saved(source,base.read_fixed_labels(),check=check)
            check()
            payload=(json.dumps(report,allow_nan=False,separators=(',',':'))+'\n').encode()
            require(len(payload)<=OUTPUT_CAP,'32MiB output cap')
            handle.write(payload)
        except BaseException as exc:
            handle.write((json.dumps({'schema':AUDIT_SCHEMA,'status':'FAIL','type':type(exc).__name__,
                                      'message':str(exc)[:2000]},allow_nan=False)+'\n').encode())
            raise
    print(json.dumps({'status':'PASS','path':str(output),'size_bytes':len(payload),'sha256':base.sha(payload)}))


if __name__=='__main__':
    main()
