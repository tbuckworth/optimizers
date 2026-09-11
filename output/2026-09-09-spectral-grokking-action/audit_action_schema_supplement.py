#!/usr/bin/env python3
"""JSON-only correction of the raw auditor's summary-row schema comparison.

Preserves the first audit and does not re-execute raw tensor/array arithmetic.
"""
import hashlib
import json
from pathlib import Path
import time

BATCH=Path('/tmp/spectral-experiment-artifacts/spectral-grokking-action-20260909.ghvEvD')
REPO=Path(__file__).resolve().parents[2]
OUT=BATCH/'schema-audit-supplement-001.json'
CHECKS=0
ERRORS=[]


def check(ok,message):
    global CHECKS
    CHECKS+=1
    if not ok:
        ERRORS.append(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()


def rec(value):
    check(sha(value['path'])==value['sha256'],'receipt '+value['path'])
    if 'size_bytes' in value:
        check(Path(value['path']).stat().st_size==value['size_bytes'],'size '+value['path'])


def main():
    started=time.monotonic(); assert not OUT.exists()
    raw_audit=read(BATCH/'readout-audit-001.json')
    check(raw_audit['errors']==['analysis rows exactly match measured states'], 'sole anticipated auditor schema defect')
    summary=read(BATCH/'analysis-001/summary.json')
    measurement=read(BATCH/'measurement-001/complete.json')
    check(len(summary['new_state_rows'])==35, 'all 35 new rows')
    for observed,item in zip(summary['new_state_rows'],measurement['accepted_results'],strict=True):
        state=read(item['analyzed_state']['path']); scalar=read(item['scalar']['path'])
        expected={key:value for key,value in state.items() if key!='full_symmetry'}
        for key,source in (('raw_receipt','raw'),('source_scalar_receipt','scalar'),('source_state_receipt','analyzed_state')):
            expected[key]={k:item[source][k] for k in ('path','sha256','size_bytes')}
        check(observed==expected,'schema-correct exact row '+str((item['seed'],item['policy'],item['step'])))
        check(state['checkpoint_provenance']==scalar['checkpoint_provenance'], 'scalar/state provenance')
        check(state['source']['source_scalar_sha256']==item['scalar']['sha256'], 'scalar hash binding')
        check(state['source']['source_npz_sha256']==item['raw']['sha256'], 'raw hash binding')
        check(state['source']['action_checkpoint_sha256']==item['checkpoint']['sha256'], 'checkpoint hash binding')
        check(scalar['raw_activations']==item['raw'], 'scalar raw receipt')
        for relative,value in scalar['source_sha256'].items():
            check(sha(REPO/relative)==value,'measurement scientific source '+relative)
    archive_receipt=summary['input_receipts']['archived_summary']; rec(archive_receipt)
    archive=read(archive_receipt['path'])
    originals={(row['seed'],row['step']):row for row in archive['rows'] if row['arm']=='legacy' and row['step'] in (1500,2000,2500)}
    archived_raw={}
    for stage in ('calibration','remaining'):
        for item in archive['input_receipts'][stage]['raw_receipts']:
            if item['arm']=='legacy':
                archived_raw[item['seed'],item['step']]=item
    for row in summary['archived_native_reference_rows']:
        key=row['seed'],row['step']
        expected={**originals[key], 'policy':'native','reference_origin':'accepted_archived_legacy','raw_receipt':archived_raw[key]}
        check(row==expected,'archived native source identity '+str(key))
        rec(row['raw_receipt'])
    for item in summary['source_copies']:
        rec(item)
    for name,value in summary['analysis_source_sha256'].items():
        check(sha(REPO/name)==value,'analysis source '+name)
    result={'status':'PASS' if not ERRORS else 'FAIL','checks':CHECKS,'errors':ERRORS,
            'scope':'JSON-only correction of one auditor schema mismatch; no numerical replay',
            'original_raw_audit_sha256':sha(BATCH/'readout-audit-001.json'),
            'script_sha256':sha(__file__),'analysis_summary_sha256':sha(BATCH/'analysis-001/summary.json'),
            'elapsed_seconds':time.monotonic()-started}
    with OUT.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False); handle.write('\n')
    print(json.dumps(result),flush=True)
    assert not ERRORS


if __name__=='__main__':
    main()
