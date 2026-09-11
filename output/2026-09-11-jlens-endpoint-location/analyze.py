"""One saved-array endpoint comparison; no model imports or old-stage execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output')
NEW=WORK/'2026-09-11-j-lens-endpoint-location'
OLD=WORK/'2026-09-11-j-lens-content-template'
UFILE=WORK/'2026-09-10-j-lens-fresh-content/preparation/directions.npz'
PINS={HERE/'dataset.json':'bda6dbefe07ebfa62cbf2efc483b6164cc819debfd1d124460e5153bdd2b7ee8',
      HERE/'pairs.json':'d47d27b93e48b38deb5e4871596c906057ba46e18305e3849b02ad1b0e337ac0',
      HERE/'protocol.md':'6095b51f81993be06210c81f32d621b00640130ec77da783e1c774e4566cded3',
      UFILE:'47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa',
      OLD/'forwards/scores.json':'3d363e6a9ba927e79b7698d50a31115fe128762be8e1809be799b6025422c8eb',
      OLD/'forwards/gaps.json':'e0ccddc94ef0144b44be2aefabaa7a48a01601cc8fe06de37502d15e160aa822'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(x,m):
    if not x:raise ValueError(m)
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def projections(h,u,m):
    return np.array([[[math.fsum((float(x)-float(c))*float(v) for x,c,v in zip(pos,m,u[:,a],strict=True))
                       for a in range(u.shape[1])] for pos in row] for row in h],dtype=np.float64)
def gaps(s,rows,pairs):
    idx={r['id']:i for i,r in enumerate(rows)}
    return np.stack([s[idx[p['observation_id']]]-s[idx[p['provision_id']]] for p in pairs])
def summaries(g):
    return {loc:{f'PC{a+1}':{'positive':int((g[:,j,a]>0).sum()),'zero':int((g[:,j,a]==0).sum()),
        'negative':int((g[:,j,a]<0).sum()),'all_eight_positive':bool((g[:,j,a]>0).all()),
        'template_sign_changes':[f'{i+1:02}' for i in range(4) if np.sign(g[2*i,j,a])!=np.sign(g[2*i+1,j,a])]}
        for a in range(4)} for j,loc in enumerate(['verb','sentence_end','old_full_end'])}

def fixtures():
    h=np.array([[[2,4],[6,8]],[[1,2],[3,4]]],dtype=np.float32)
    s=projections(h,np.eye(2,dtype=np.float32),np.ones(2,dtype=np.float64))
    np.testing.assert_array_equal(s,[[[1,3],[5,7]],[[0,1],[2,3]]])
    np.testing.assert_array_equal(gaps(s,[{'id':'O'},{'id':'P'}],[{'observation_id':'O','provision_id':'P'}]),[[[1,2],[3,4]]])
    g=np.ones((8,3,4));g[0,0,3]=0;g[1,0,3]=-1
    v=summaries(g)['verb']['PC4']
    check(v['positive']==6 and v['zero']==1 and v['negative']==1 and not v['all_eight_positive'],'strict signs')
    print('PASS: fabricated two-position projections, O-P gaps and strict-zero summary; no scientific input')

def run(receipt_sha):
    target=HERE/'analysis';target.mkdir(exist_ok=False)
    save(target/'attempt.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'automatic_retry':False})
    for p,d in PINS.items():check(sha(p)==d,'input pin '+str(p))
    rp=NEW/'forwards/receipt.json';check(sha(rp)==receipt_sha,'new receipt pin')
    rec=json.loads(rp.read_text());check(rec['status']=='complete' and rec['forward_count']==16 and rec['parameters_unchanged'],'incomplete measurement')
    check(sha(NEW/'forward.py')==rec['source_sha256'],'producer changed')
    for x in rec['outputs']:check(sha(NEW/'forwards'/x['path'])==x['sha256'],'output changed')
    rows=json.loads((HERE/'dataset.json').read_text())['rows'];pairs=json.loads((HERE/'pairs.json').read_text())['pairs']
    locs=['verb','sentence_end'];check(rec['capture_locations']==locs,'receipt locations')
    with np.load(NEW/'forwards/features.npz',allow_pickle=False) as f:
        h,s,g=f['activation_11'],f['scores64'],f['gaps64']
        check(f['locations'].tolist()==locs,'array locations')
    tp=NEW/'token-preflight/receipt.json'
    check(sha(tp)==rec['preflight_receipt_sha256'],'preflight binding')
    tr=json.loads(tp.read_text());check(tr['source_sha256']==rec['source_sha256'],'preflight source')
    check(sha(NEW/'token-preflight/tokens.json')==tr['outputs'][0]['sha256'],'preflight token bytes')
    inputs=json.loads((NEW/'forwards/inputs.json').read_text())['records']
    check(inputs==json.loads((NEW/'token-preflight/tokens.json').read_text())['records'],'input record binding')
    check(len(inputs)==16,'input count')
    for row,inp in zip(rows,inputs,strict=True):
        check(all(inp[k]==v for k,v in row.items()),'input row/verb metadata')
        check(list(inp['captured_positions'])==locs and inp['layer']==11,'capture schema')
        start,end=row['verb_span'];offsets=inp['offset_mapping']
        j=inp['captured_positions']['verb'];last=inp['captured_positions']['sentence_end']
        check(offsets[j][0]<end and offsets[j][1]==end and last==len(inp['input_ids'])-1,'capture boundary')
        check(offsets[last][0]<=len(row['text'])-1 and offsets[last][1]==len(row['text']),'period boundary')
    with np.load(UFILE,allow_pickle=False) as f:u,m=f['u32'],f['source_mean64']
    for x,shape,dtype in [(h,(16,2,1024),np.float32),(s,(16,2,4),np.float64),(g,(8,2,4),np.float64),
                           (u,(1024,4),np.float32),(m,(1024,),np.float64)]:
        check(x.shape==shape and x.dtype==dtype and np.isfinite(x).all(),'array shape/dtype/nonfinite')
    sc=projections(h,u,m);np.testing.assert_allclose(s,sc,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(g,gaps(s,rows,pairs));np.testing.assert_allclose(g,gaps(sc,rows,pairs),atol=2e-12,rtol=0)
    np.testing.assert_array_equal(np.sign(g),np.sign(gaps(sc,rows,pairs)))
    exported=json.loads((NEW/'forwards/scores.json').read_text())['locations']
    check(list(exported)==locs,'JSON locations')
    for j,loc in enumerate(locs):
        check(exported[loc]==[{'axis':f'PC{a+1}','values':{r['id']:float(s[i,j,a]) for i,r in enumerate(rows)}} for a in range(4)],'score JSON binding')
    gj=json.loads((NEW/'forwards/gaps.json').read_text())
    check(gj['locations']==locs and gj['axes']==['PC1','PC2','PC3','PC4'] and gj['orientation']=='observation minus provision','gap metadata')
    check(gj['pairs']==[{**p,'gaps64':g[i].tolist()} for i,p in enumerate(pairs)],'gap JSON binding')
    old=json.loads((OLD/'forwards/scores.json').read_text())['scores']
    check([a['axis'] for a in old]==['PC1','PC2','PC3','PC4'],'old axes')
    oldscores=np.array([[a['values'][r['id']] for a in old] for r in rows],dtype=np.float64)
    oldg=gaps(oldscores,rows,pairs)
    np.testing.assert_array_equal(oldg,np.array([p['gaps64'] for p in json.loads((OLD/'forwards/gaps.json').read_text())['pairs']]))
    allg=np.concatenate([g,oldg[:,None,:]],axis=1)
    stats=summaries(allg)
    result={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
        'new_receipt_sha256':receipt_sha,'input_pins':{str(p):d for p,d in PINS.items()},
        'locations':['verb','sentence_end','old_full_end'],'axes':['PC1','PC2','PC3','PC4'],
        'new_scalar_scores_checked':128,'new_scalar_gaps_checked':64,'max_score_error':float(abs(s-sc).max()),
        'primary_all_eight_verb_PC4_positive':stats['verb']['PC4']['all_eight_positive'],'summaries':stats,
        'pairs':[{**p,'gaps_by_location':allg[i].tolist(),'verb_minus_old':(allg[i,0]-allg[i,2]).tolist(),
                  'sentence_end_minus_old':(allg[i,1]-allg[i,2]).tolist()} for i,p in enumerate(pairs)],
        'scope':'Four reused contents/two templates. One new model run, two new locations; old final endpoints reused. Scalar corroboration, not semantic or causal validation.'}
    save(target/'results.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fixtures','run']);p.add_argument('--receipt-sha');a=p.parse_args()
    if a.mode=='fixtures':fixtures()
    else:run(a.receipt_sha)
