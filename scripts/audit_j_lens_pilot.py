"""Self-audit from raw arrays and pinned weights; no new inference or tuning."""
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/2026-09-09-j-lens-pilot'
WEIGHTS=Path('/private-artifacts/storage/cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/model.safetensors-00001-of-00001.safetensors')


def main():
    torch.set_num_threads(1)
    data=json.loads((OUT/'dataset.json').read_text())
    metrics=json.loads((OUT/'metrics.json').read_text())
    readouts=json.loads((OUT/'readouts.json').read_text())
    acquisition=json.loads((OUT/'acquisition.json').read_text())
    start=json.loads((OUT/'acquisition-start.json').read_text())
    assert acquisition['provenance']==start
    for name,value in start['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
    assert len(data['rows'])==72 and len({r['text'] for r in data['rows']})==72
    fit=np.array([r['split']=='fit' for r in data['rows']])
    labels=np.array([metrics['groups'].index(r['group']) for r in data['rows']])
    assert sum(fit)==48 and sum(~fit)==24
    raw=np.load(OUT/'features.npz')
    early=np.load(OUT/'acquired_features.npz')
    arrays=np.load(OUT/'analysis_arrays.npz')
    checks={}
    with safe_open(WEIGHTS,framework='pt') as weights:
        unembed=weights.get_tensor('model.language_model.embed_tokens.weight')
        norm=weights.get_tensor('model.language_model.norm.weight').float()
    for name,row in metrics['analyses'].items():
        x=raw[name].astype(np.float64)
        assert x.shape==(72,1024) and np.isfinite(x).all()
        np.testing.assert_array_equal(x,early[name])
        mean=x[fit].mean(0)
        _,s,vt=np.linalg.svd(x[fit]-mean,full_matrices=False)
        np.testing.assert_allclose(s**2/47,row['eigenvalues'],atol=1e-12,rtol=1e-10)
        basis=arrays[name+'_basis']
        np.testing.assert_allclose(np.abs(vt[:4]@basis),np.eye(4),atol=1e-8)
        scores=(x-mean)@basis
        np.testing.assert_allclose(scores,arrays[name+'_scores'],atol=1e-12)
        center=np.stack([scores[fit&(labels==i)].mean(0) for i in range(4)])
        pred=np.argmin(np.sum((scores[~fit,None]-center[None])**2,axis=-1),axis=1)
        assert np.array_equal(pred,row['evaluation']['predictions'])
        assert np.mean(pred==labels[~fit])==row['evaluation']['centroid_accuracy']
        assert adjusted_rand_score(labels[~fit],row['evaluation']['cluster_predictions'])==row['evaluation']['kmeans_ari']
        assert abs(np.mean([v['centroid_accuracy'] for v in row['random_evaluations']])-row['random_accuracy_mean'])<1e-12
        rng=np.random.default_rng(20260909)
        random=np.linalg.qr(rng.normal(size=(1024,4)))[0]
        dirs=np.concatenate([basis,random],axis=1).T
        signed=np.concatenate([dirs,-dirs])
        layer=name.rsplit('_',1)[1]
        check={}
        for control,vectors in [('jlens',signed@raw['J_'+layer].T),('plain',signed)]:
            # Independent implementation of the pinned Qwen RMSNorm + tied head.
            z=torch.tensor(vectors,dtype=torch.float32).bfloat16().float()
            normalized=(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*(1+norm)).bfloat16()
            with torch.no_grad():
                logits=(normalized@unembed.T).float()
            record=readouts[name+'_'+control]
            ids=torch.tensor(record['token_ids'])
            recorded=torch.tensor(record['scores'])
            recomputed=logits.gather(1,ids)
            error=float((recomputed-recorded).abs().max())
            # BF16 CPU/GPU kernels can differ in last rounding bits.
            assert error<=.125,(name,control,error)
            threshold=logits.topk(12,dim=-1).values[:,-1:]
            assert torch.all(recomputed>=threshold-.125)
            assert torch.isfinite(logits).all()
            centered=logits-logits.mean(-1,keepdim=True)
            unit=torch.nn.functional.normalize(centered,dim=-1)
            cos_error=float((unit@unit.T-torch.tensor(record['cosine'])).abs().max())
            assert cos_error<.005,(name,control,cos_error)
            check[control]=dict(max_score_error=error,max_cosine_error=cos_error)
        checks[name]=dict(svd_and_report_checks='PASS',readouts=check)
        print(name,'PASS',flush=True)
    result=dict(status='PASS',scope='Executor self-audit; independent SVD and CPU weight readout, not independent-agent replication',checks=checks,
                dataset_hash_verified=True,source_pins_verified=True,acquisition_provenance_verified=True)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
