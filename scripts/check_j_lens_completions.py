"""Executor self-check of archived outputs; independent SVD and centroid arithmetic."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/2026-09-10-j-lens-completions'


def main():
    rows=json.loads((OUT/'dataset.json').read_text())
    raw=np.load(OUT/'features.npz')
    analysis=np.load(OUT/'analysis_arrays.npz')
    metrics=json.loads((OUT/'metrics.json').read_text())
    provenance=json.loads((OUT/'acquisition-start.json').read_text())
    inputs=json.loads((OUT/'inputs.json').read_text())
    fit=np.array([r['split']=='fit' for r in rows])
    y=np.array([metrics['input_summary']['groups'].index(r['group']) for r in rows])
    checks={}
    for entry in provenance['files']:
        assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    assert len(rows)==48 and sum(fit)==32 and len({r['pair'] for r in rows if r['split']=='heldout'})==8
    assert [r['id'] for r in rows]==[r['id'] for r in inputs]
    assert all(len(r['token_losses'])==len(r['ids'])-r['boundary']>=2 for r in inputs)
    for name in raw.files:
        x=raw[name].astype(np.float64)
        assert x.shape==(48,1024) and np.isfinite(x).all()
        mu=x[fit].mean(0)
        _,s,vt=np.linalg.svd(x[fit]-mu,full_matrices=False)
        v=analysis[name+'_basis']
        overlap=float(np.sum((vt[:4]@v)**2)/4)
        assert abs(overlap-1)<1e-10
        assert np.allclose(s[:4]**2/31,metrics[name]['eigenvalues'][:4])
        score=(x-mu)@v
        centers=np.stack([score[fit&(y==c)].mean(0) for c in range(4)])
        pred=np.array([min(range(4),key=lambda c:float(np.dot(z-centers[c],z-centers[c]))) for z in score[~fit]])
        assert pred.tolist()==metrics[name]['pca']['predictions']
        checks[name]=dict(svd_top4_overlap=overlap,centroid_match=True)
    result=dict(status='PASS',scope='executor self-check, not independent-agent replication',checks=checks,
        source_hashes_match=True,completion_boundaries_valid=True,heldout_unique_pairs=8)
    with (OUT/'self-check.json').open('x') as f:
        json.dump(result,f,indent=2)
    print('Self-check PASS: source hashes, boundaries, independent SVD, centroid predictions.')


if __name__=='__main__':
    main()
