#!/usr/bin/env python3
"""Independent15-new-NPZ arithmetic and exact six-contrast JSON audit."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from common import Audit, BATCH, OLD, POLICY, REPO, SEEDS, STEPS, batch_contract
from readout_math import behavior, probe, edges, audit_symmetry, check_array, paired_stats

OLD_SUMMARY=REPO/'output/2026-09-09-spectral-grokking-action/results/summary.json'
OLD_SUMMARY_SHA='dc79ddbc9261607468c6adb5c8ff48b3e1249803f4651397aca134bcbc02a7a1'
OLD_MEASURE_SHA='fe11e1404e8ab20c792d8d441477f325f48a2bb9fdc4c868620a3ffeeafbb9ee'
FIXED=[9,33,32,49,11]
STRUCTURE=('pairs','sums','train_ids','test_ids','probe_fit_indices','probe_eval_indices','null_permutations')
METRICS={
 'heldout_cross_entropy':(('behavior','test','loss'),'lower','primary','nats_per_example'),
 'heldout_correct_margin_mean':(('behavior','test','correct_class_margin_mean'),'higher','primary','logit'),
 'final_hidden_selected_five_heldout_r2':(('probes','final_hidden','selected_eval_mean_r2'),'higher','primary','R2'),
 'heldout_accuracy':(('behavior','test','accuracy'),'higher','secondary','fraction'),
 'final_hidden_fixed_panel_heldout_r2':(('probes','final_hidden','fixed_panel_eval_mean_r2'),'higher','secondary','R2'),
 'final_hidden_null_max_heldout_r2':(('probes','final_hidden','null_max_eval_mean_r2'),'descriptive','secondary','R2'),
 'pre_attention_selected_five_heldout_r2':(('probes','pre_attention','selected_eval_mean_r2'),'descriptive','secondary','R2'),
 'pre_attention_fixed_panel_heldout_r2':(('probes','pre_attention','fixed_panel_eval_mean_r2'),'descriptive','secondary','R2'),
 'pre_attention_null_max_heldout_r2':(('probes','pre_attention','null_max_eval_mean_r2'),'descriptive','secondary','R2'),
 'heldout_correct_shift_defect':(('symmetry','heldout_shift_pooled','correct','value'),'lower','secondary','normalized_squared_defect'),
 'heldout_wrong_shift_defect':(('symmetry','heldout_shift_pooled','wrong_shift','value'),'descriptive','secondary','normalized_squared_defect'),
 'exchange_defect':(('symmetry','exchange','correct','value'),'lower','secondary','normalized_squared_defect'),
 'training_membership_excess':(('symmetry','training_membership_pooled','excess'),'descriptive','secondary','normalized_squared_defect_difference'),
 'centered_logit_rms_heldout':(('symmetry','centered_logit_rms','test'),'descriptive','secondary','logit'),
}
CONTRASTS=(('raw_minus_norm_matched','norm_matched','primary'),('raw_minus_native','native','secondary'),('raw_minus_orthogonal','orthogonal','secondary'))


def structural(audit,arrays,contract):
    audit.require(set(arrays)==set(STRUCTURE)|{'logits','final_hidden','pre_attention'},'Exact raw array members')
    for key in STRUCTURE:
        value=np.ascontiguousarray(arrays[key])
        digest=hashlib.sha256(key.encode()+b'\0'+str(value.dtype).encode()+b'\0'+json.dumps(list(value.shape)).encode()+b'\0'+value.tobytes()).hexdigest()
        audit.check(digest==contract['structural_array_sha256'][key],'Accepted structural-array hash '+key)
        audit.require(value.dtype==np.int64,'Structural int64 dtype '+key)
    pairs=np.column_stack((np.repeat(np.arange(113),113),np.tile(np.arange(113),113)))
    audit.check(np.array_equal(arrays['pairs'],pairs) and np.array_equal(arrays['sums'],pairs.sum(axis=1)%113),'Pairs and modular labels')
    train,test=arrays['train_ids'],arrays['test_ids'];fit,ev=arrays['probe_fit_indices'],arrays['probe_eval_indices']
    audit.require(len(train)==int(12769*.3) and np.array_equal(np.sort(np.r_[train,test]),np.arange(12769)),'Train/test partition')
    audit.require(np.array_equal(np.sort(np.r_[fit,ev]),np.arange(len(test))),'Probe partition')
    audit.require(arrays['null_permutations'].shape==(20,len(test)),'Exactly20 null draws')
    for perm in arrays['null_permutations']:
        audit.check(np.array_equal(np.sort(perm),np.arange(len(test))) and not np.array_equal(perm,np.arange(len(test))),'Null row permutation')
    for key,width in (('logits',113),('final_hidden',128),('pre_attention',256)):
        value=arrays[key]
        audit.require(value.shape==(12769,width) and np.isfinite(value).all(),'New activation shape/finiteness '+key)


def audit_probe(audit,arrays,scalar,row):
    ids=arrays['test_ids'];fit=arrays['probe_fit_indices'];ev=arrays['probe_eval_indices']
    for feature in ('final_hidden','pre_attention'):
        expected=probe(arrays[feature][ids],arrays['sums'][ids],fit,ev,arrays['null_permutations'])
        actual=scalar['probes'][feature];compact=row['probes'][feature]
        check_array(audit,actual['fit_transform']['feature_mean'],expected['feature_mean'],'Probe transform:mean')
        audit.close(actual['fit_transform']['feature_rms_scalar'],expected['feature_rms_scalar'],'Probe transform:RMS')
        audit.require(len(actual['null']['runs'])==20,'Exactly20 stored probe nulls')
        null=[]
        for index,derived in enumerate(expected['runs']):
            stored=actual['observed'] if index==0 else actual['null']['runs'][index-1]
            per=stored['per_frequency']
            audit.require([r['frequency'] for r in per]==list(range(1,57)),'Exactly56 frequency identities')
            for key in ('fit_r2','eval_r2'):
                check_array(audit,[r[key] for r in per],derived[key],'Probe scores:'+key)
            check_array(audit,[[r['fit_target_mean_cos'],r['fit_target_mean_sin']] for r in per],derived['target_mean'],'Probe target:mean',atol=1e-10)
            audit.check(stored['selected_frequencies']==derived['selected_frequencies'],'Fit-selected five frequencies')
            audit.close(stored['selected_eval_mean_r2'],derived['selected_eval_mean_r2'],'Probe selected:R2',rtol=1e-8,atol=1e-8)
            if index==0:
                audit.check(compact['selected_frequencies']==derived['selected_frequencies'] and compact['fixed_panel_frequencies']==FIXED,'Compact fixed/selected panel')
                audit.close(compact['selected_eval_mean_r2'],derived['selected_eval_mean_r2'],'Compact:selected',rtol=1e-8,atol=1e-8)
                audit.close(compact['fixed_panel_eval_mean_r2'],float(derived['eval_r2'][np.asarray(FIXED)-1].mean()),'Compact:fixed',rtol=1e-8,atol=1e-8)
            else:
                null.append(derived['selected_eval_mean_r2'])
        for key,value in (('mean',np.mean(null)),('std_population',np.std(null)),('max',max(null))):
            audit.close(actual['null']['selected_eval_mean_r2_'+key],float(value),'Null:'+key,rtol=1e-8,atol=1e-8)
        audit.close(compact['null_max_eval_mean_r2'],max(null),'Compact:null max',rtol=1e-8,atol=1e-8)


def summary_row(state,item):
    """Explicit producer schema transformation, not numerical producer reuse."""
    value={key:value for key,value in state.items() if key!='full_symmetry'}
    value.update(raw_receipt=item['raw'],source_scalar_receipt=item['scalar'],source_state_receipt=item['analyzed_state'])
    return value


def paired(audit,summary,manifest,rows,old):
    audit.require(summary['schema']=='grokking_raw_direction_analysis_summary_v1' and summary['new_state_count']==15,'Raw paired summary schema')
    audit.check(summary['new_state_rows']==rows,'Exact transformed new state rows')
    audit.check(summary['archived_action_reference_rows']==old['new_state_rows'] and summary['archived_native_reference_rows']==old['archived_native_reference_rows'],'Unchanged archived reference JSON rows')
    audit.require(set(manifest['metric_definitions'])==set(METRICS),'Exactly14 fixed metrics')
    lookup={(row['seed'],row['policy'],row['step']):row for row in rows+old['new_state_rows']+old['archived_native_reference_rows']}
    expected_order=[(step,name,other,role) for step in (2000,2500) for name,other,role in CONTRASTS]
    audit.require(len(summary['paired_endpoint_contrasts'])==6,'Exactly six contrasts')
    for contrast,(step,name,other,role) in zip(summary['paired_endpoint_contrasts'],expected_order,strict=True):
        audit.require((contrast['step'],contrast['contrast'],contrast['left_policy'],contrast['right_policy'])==(step,name,POLICY,other),'Fixed contrast identity/order')
        audit.check(contrast['contrast_role']==role and contrast['endpoint_role']==('primary' if step==2500 else 'fixed_earlier'),'Fixed contrast/endpoint role')
        audit.require(set(contrast['metrics'])==set(METRICS),'All contrast metrics')
        for metric,(path,direction,metric_role,unit) in METRICS.items():
            definition=manifest['metric_definitions'][metric]
            audit.check(definition['path']==list(path) and definition['favorable_direction']==direction and definition['role']==metric_role and definition['unit']==unit,'Fixed metric definition '+metric)
            def value(row):
                for key in path: row=row[key]
                return row
            left=[value(lookup[seed,POLICY,step]) for seed in SEEDS]
            right=[value(lookup[seed,other,step]) for seed in SEEDS]
            derived=paired_stats(left,right,direction);actual=contrast['metrics'][metric]
            expected_pairs=[{'seed':seed,'left':a,'right':b,'difference':d} for seed,a,b,d in zip(SEEDS,left,right,derived.pop('differences'),strict=True)]
            audit.check(actual['paired']==expected_pairs,'Five paired sources/differences '+metric)
            audit.check(actual['n_total_seeds']==5 and actual['difference_definition']=='left_minus_right' and actual['role']==metric_role and actual['unit']==unit,'Paired units/role '+metric)
            for key,value in derived.items():
                if isinstance(value,bool): audit.check(actual[key] is value,'Paired boolean '+key)
                else: audit.close(actual[key],value,'Paired:'+key,rtol=1e-12,atol=1e-12)


def run(audit,args):
    _,completed=batch_contract(audit,args.batch_sha)
    measure=BATCH/'measurement-001';analysis=BATCH/'analysis-001'
    audit.require(not (measure/'failure.json').exists() and not (analysis/'failure.json').exists(),'No measurement/analysis failures')
    done=audit.read(measure/'complete.json',args.measurement_sha)
    am_done=audit.read(analysis/'complete.json',args.analysis_sha)
    audit.require(done['status']=='complete' and done['state_count']==15 and am_done['status']=='complete','Completed measurement/analysis')
    audit.require(done['schema']=='grokking_raw_direction_measurement_complete_v1' and am_done['schema']=='grokking_raw_direction_analysis_complete_v1' and am_done['new_state_count']==15 and am_done['contrast_count']==6,'Completion schemas/counts')
    manifest=audit.read(audit.receipt(done['manifest'],measure/'manifest.json'))
    am_manifest=audit.read(audit.receipt(am_done['manifest'],analysis/'manifest.json'))
    summary=audit.read(audit.receipt(am_done['summary'],analysis/'summary.json'))
    old=audit.read(OLD_SUMMARY,OLD_SUMMARY_SHA)
    old_complete=audit.read(OLD/'measurement-001/complete.json',OLD_MEASURE_SHA)
    old_manifest=audit.read(audit.receipt(old_complete['manifest'],OLD/'measurement-001/manifest.json'))
    audit.require(manifest['prior_recipe']==old_manifest['prior_recipe'] and manifest['recipe']==old_manifest['recipe'],'Accepted immutable readout recipe')
    audit.require(manifest['schema']=='grokking_raw_direction_measurement_v1' and am_manifest['schema']=='grokking_raw_direction_analysis_v1','Manifest schemas')
    audit.require(done['measurement_source_sha256']==manifest['measurement_source_sha256'] and am_done['analysis_source_sha256']==summary['analysis_source_sha256']==am_manifest['analysis_source_sha256'],'Source maps bound through completion')
    audit.check(manifest['fixed_frequency_panel']['frequencies']==FIXED,'Fixed five-frequency panel')
    audit.sources(manifest['measurement_source_sha256']);audit.sources(summary['analysis_source_sha256'])
    audit.require(done['input_batch_completion']['sha256']==args.batch_sha and manifest['raw_batch']['completion']==done['input_batch_completion'],'Measurement raw batch binding')
    audit.require(am_manifest['measurement']['sha256']==args.measurement_sha and am_manifest['archived_summary']['sha256']==OLD_SUMMARY_SHA,'Paired accepted inputs')
    audit.check(summary['input_receipts']['measurement_completion']['sha256']==args.measurement_sha and summary['input_receipts']['archived_summary']['sha256']==OLD_SUMMARY_SHA,'Summary accepted inputs')
    expected_roster=[(seed,POLICY,step) for seed in SEEDS for step in STEPS]
    audit.require([(r['seed'],r['policy'],r['step']) for r in done['accepted_results']]==expected_roster,'Exact15 NPZ states')
    audit.check(manifest['roster']==[list(x) for x in expected_roster],'Measurement manifest roster')
    checkpoint_by_key={(value['seed'],step):artifacts[f'{POLICY}-step-{step:06d}.pt'] for value,artifacts in completed for step in STEPS}
    seed_by_key={value['seed']:value for value,_ in completed};cached_edges={};rows=[]
    for item in done['accepted_results']:
        seed,step=item['seed'],item['step'];stem=f'seed{seed}-{POLICY}-step{step:06d}'
        bound=checkpoint_by_key[seed,step]
        audit.check(item['checkpoint']=={key:bound[key] for key in ('path','sha256','size_bytes')},'Measured checkpoint bound to new acquisition')
        for key,folder,suffix in (('raw','raw','.npz'),('scalar','scalars','.json'),('analyzed_state','states','.json')):
            audit.receipt(item[key],measure/folder/(stem+suffix))
        scalar=audit.read(item['scalar']['path']);state=audit.read(item['analyzed_state']['path'])
        audit.require((scalar['seed'],scalar['policy'],scalar['step'])==(seed,POLICY,step) and (state['seed'],state['policy'],state['step'])==(seed,POLICY,step),'State identity')
        audit.require(scalar['schema']=='grokking_raw_direction_measurement_state_v1' and state['schema']=='grokking_raw_direction_analyzed_state_v1','State schema')
        audit.check(scalar['raw_activations']==item['raw'] and scalar['source_sha256']==manifest['measurement_source_sha256'],'Scalar source/raw bindings')
        provenance=scalar['checkpoint_provenance']
        audit.check(provenance==state['checkpoint_provenance'] and provenance['raw_direction_checkpoint']==item['checkpoint'],'Scalar/analyzed checkpoint provenance')
        audit.check(provenance['parent_legacy_checkpoint']==seed_by_key[seed]['parent_checkpoint'],'Accepted original parent identity')
        audit.check(state['source']['source_npz_sha256']==item['raw']['sha256'] and state['source']['source_scalar_sha256']==item['scalar']['sha256'] and state['source']['raw_direction_checkpoint_sha256']==item['checkpoint']['sha256'],'Analyzed source hashes')
        audit.check(state['source']['stage']=='raw_direction' and (provenance['outer_seed'],provenance['outer_policy'],provenance['outer_step'])==(seed,POLICY,step),'Analyzed source stage/outer identity')
        contract=old_manifest['prior_recipe']['seeds'][str(seed)]
        audit.check(scalar['prior_recipe_contract']==contract and state['prior_recipe_contract']==contract,'Accepted per-seed structural contract')
        with np.load(item['raw']['path'],allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        structural(audit,arrays,contract)
        for split in ('train','test'):
            expected=behavior(arrays['logits'],arrays['sums'],arrays[split+'_ids'])
            for key,value in expected.items():
                audit.close(state['behavior'][split][key],value,'Behavior:'+key,rtol=2e-6 if key=='loss' else 1e-9,atol=2e-6 if key=='loss' else 1e-9)
                if key!='correct_class_margin_mean': audit.close(scalar['behavior'][split][key],value,'Scalar behavior:'+key,rtol=2e-6 if key=='loss' else 1e-9,atol=2e-6 if key=='loss' else 1e-9)
        audit_probe(audit,arrays,scalar,state)
        if seed not in cached_edges: cached_edges[seed]=edges(arrays['train_ids'],arrays['test_ids'])
        audit_symmetry(audit,arrays,state,cached_edges[seed])
        rows.append(summary_row(state,item));del arrays,scalar,state;gc.collect()
        print(json.dumps({'audited_state':[seed,POLICY,step],'checks':audit.checks,'errors':len(audit.errors)}),flush=True)
    paired(audit,summary,am_manifest,rows,old)
    expected_copies={(kind,f'seed{seed}-{POLICY}-step{step:06d}') for kind in ('scalar','analyzed_state') for seed in SEEDS for step in STEPS}
    audit.require({(r['kind'],r['seed_policy_step']) for r in summary['source_copies']}==expected_copies and len(summary['source_copies'])==30,'All30 byte-identical source copies')
    accepted={(f"seed{r['seed']}-{POLICY}-step{r['step']:06d}",kind):r[key] for r in done['accepted_results'] for kind,key in (('scalar','scalar'),('analyzed_state','analyzed_state'))}
    for rec in summary['source_copies']:
        folder='source-scalars' if rec['kind']=='scalar' else 'source-states'
        audit.receipt(rec,analysis/folder/(rec['seed_policy_step']+'.json'))
        audit.check(rec['sha256']==accepted[rec['seed_policy_step'],rec['kind']]['sha256'],'Source-copy SHA identity')
    return {'scope':'Independent NumPy equations on15 new NPZs and six fixed14-metric paired contrasts; archived values reused only',
            'states':expected_roster,'batch_sha256':args.batch_sha,'measurement_sha256':args.measurement_sha,'analysis_sha256':args.analysis_sha,
            'numpy_version':np.__version__,'limitations':['Saved-readout numerical corroboration, not new inference/training replication.',
                'Archived reference tensors/readouts not re-audited.', 'Small or mixed effects do not establish equivalence or semantic/safety claims.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','measurement','analysis'):parser.add_argument('--'+name+'-sha',required=True)
    args=parser.parse_args();audit=Audit();audit.start('readout-paired-audit-001')
    try: payload=run(audit,args)
    except Exception as error:
        audit.errors.append(f'{type(error).__name__}: {error}');payload={'scope':'Incomplete readout audit; failure preserved'}
    sys.exit(0 if audit.finish(payload) else 1)
