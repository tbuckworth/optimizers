"""Small scalar corroboration of the NEW 16-text panel; never imports a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output')
NEW = WORK / '2026-09-11-j-lens-content-template'
DIRECTIONS = WORK / '2026-09-10-j-lens-fresh-content/preparation/directions.npz'
PINS = {HERE/'dataset.json': '841d86753f857d686165e53d76278bc1fa40840f97e09ffad61f0c89d57fcc7d',
        HERE/'pairs.json': 'd47d27b93e48b38deb5e4871596c906057ba46e18305e3849b02ad1b0e337ac0',
        HERE/'protocol.md': '10488950ab8234932dd1f7aa2ed74fc0d32caaef4cf5e8468d724e7206eb8c45',
        DIRECTIONS: '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check(condition, why):
    if not condition:
        raise ValueError(why)

def scalar(h, u, mean):
    return np.array([[math.fsum((float(x)-float(m))*float(v) for x,m,v in zip(row,mean,u[:,a],strict=True))
                      for a in range(u.shape[1])] for row in h], dtype=np.float64)

def differences(scores, rows, pairs):
    index = {r['id']: i for i,r in enumerate(rows)}
    return np.array([[float(scores[index[p['observation_id']],a])-float(scores[index[p['provision_id']],a])
                      for a in range(scores.shape[1])] for p in pairs], dtype=np.float64)

def summarize(gaps):
    check(gaps.shape == (8,4) and np.isfinite(gaps).all(), 'gap shape/nonfinite')
    return {f'PC{a+1}': {'positive': int((gaps[:,a]>0).sum()),
              'zero': int((gaps[:,a]==0).sum()), 'negative': int((gaps[:,a]<0).sum()),
              'all_eight_positive': bool((gaps[:,a]>0).all()),
              'template_sign_changes': [f'{i+1:02}' for i in range(4)
                   if np.sign(gaps[2*i,a]) != np.sign(gaps[2*i+1,a])]}
            for a in range(4)}

def fixtures():
    h=np.array([[3,5],[2,1]],dtype=np.float32)
    u=np.array([[1,0],[0,1]],dtype=np.float32)
    m=np.array([1,2],dtype=np.float64)
    np.testing.assert_array_equal(scalar(h,u,m), [[2,3],[1,-1]])
    rows=[{'id':'O'},{'id':'P'}]; pairs=[{'observation_id':'O','provision_id':'P'}]
    np.testing.assert_array_equal(differences(scalar(h,u,m),rows,pairs), [[1,4]])
    g=np.ones((8,4)); g[1,3]=-2; g[2,3]=0
    s=summarize(g)['PC4']
    check(s=={'positive':6,'zero':1,'negative':1,'all_eight_positive':False,'template_sign_changes':['01','02']}, 'sign summary')
    check(summarize(np.ones((8,4)))['PC4']['all_eight_positive'], 'positive fixture')
    try: summarize(np.full((8,4),np.nan))
    except ValueError: pass
    else: raise AssertionError('nonfinite accepted')
    print('PASS: exact scalar projections, O-P orientation, strict-zero and template-sign fixtures; no actual data loaded')

def run(receipt_sha):
    out=HERE/'measurement-audit'
    out.mkdir(exist_ok=False)
    (out/'attempt.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sha(Path(__file__)), 'automatic_retry':False})+'\n')
    for p,d in PINS.items(): check(sha(p)==d, 'input pin: '+str(p))
    receipt_path=NEW/'forwards/receipt.json'
    check(sha(receipt_path)==receipt_sha,'measurement receipt pin')
    receipt=json.loads(receipt_path.read_text())
    check(receipt['status']=='complete' and receipt['forward_count']==16 and receipt['parameters_unchanged'], 'measurement incomplete')
    check(sha(NEW/'forward.py')==receipt['source_sha256'],'producer source changed')
    for item in receipt['outputs']:
        check(sha(NEW/'forwards'/item['path'])==item['sha256'],'producer output changed')
    rows=json.loads((HERE/'dataset.json').read_text())['rows']
    pairs=json.loads((HERE/'pairs.json').read_text())['pairs']
    with np.load(DIRECTIONS,allow_pickle=False) as f: u,mean=f['u32'],f['source_mean64']
    with np.load(NEW/'forwards/features.npz',allow_pickle=False) as f:
        h,s,g=f['activation_11'],f['scores64'],f['gaps64']
    check(h.shape==(16,1024) and h.dtype==np.float32 and np.isfinite(h).all(),'h shape/type')
    check(s.shape==(16,4) and s.dtype==np.float64 and np.isfinite(s).all(),'score shape/type')
    check(g.shape==(8,4) and g.dtype==np.float64 and np.isfinite(g).all(),'gap shape/type')
    check(u.shape==(1024,4) and u.dtype==np.float32 and mean.shape==(1024,) and mean.dtype==np.float64,'directions shape/type')
    np.testing.assert_array_equal(g,differences(s,rows,pairs))
    independent=scalar(h,u,mean)
    np.testing.assert_allclose(s,independent,rtol=0,atol=1e-12)
    independent_gaps=differences(independent,rows,pairs)
    np.testing.assert_allclose(g,independent_gaps,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(np.sign(g),np.sign(independent_gaps))
    exported=json.loads((NEW/'forwards/scores.json').read_text())['scores']
    check([r['axis'] for r in exported]==['PC1','PC2','PC3','PC4'],'axis roster')
    for a,axis in enumerate(exported):
        check(axis['values']=={r['id']:float(s[i,a]) for i,r in enumerate(rows)},'score JSON values')
    exported_gaps=json.loads((NEW/'forwards/gaps.json').read_text())['pairs']
    check(exported_gaps==[{**p,'gaps64':g[i].tolist()} for i,p in enumerate(pairs)],'gap JSON values')
    records=json.loads((NEW/'forwards/inputs.json').read_text())['records']
    preflight=json.loads((NEW/'token-preflight/tokens.json').read_text())['records']
    check(records==preflight,'input IDs differ from preflight')
    for record,row in zip(records,rows,strict=True):
        check(all(record[k]==v for k,v in row.items()),'text/metadata changed')
        check(record['layer']==11 and record['captured_position']==len(record['input_ids'])-1,'capture location')
    result={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),
       'source_sha256':sha(Path(__file__)),'measurement_receipt_sha256':receipt_sha,
       'scalar_scores_checked':64,'scalar_gaps_checked':32,
       'max_score_error':float(np.abs(s-independent).max()),'max_gap_error':float(np.abs(g-independent_gaps).max()),
       'axes':summarize(g),'pairs':exported_gaps,
       'scope':'Independent scalar arithmetic only; not semantic validation or repeated model measurement.'}
    with (out/'receipt.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fixtures','run']);p.add_argument('--receipt-sha');a=p.parse_args()
    if a.mode=='fixtures':fixtures()
    else:run(a.receipt_sha)
