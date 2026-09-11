"""CPU-only, fixed analysis of the archived frozen-model pilot."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/2026-09-09-j-lens-pilot'
spec = importlib.util.spec_from_file_location('pilot', ROOT/'experiments/j_lens_spectral_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
torch.set_num_threads(1)


def evaluate(scores, labels, fit):
    centroids = np.stack([scores[fit & (labels == c)].mean(0) for c in range(4)])
    pred = ((scores[~fit, None] - centroids[None])**2).sum(-1).argmin(1)
    km = KMeans(n_clusters=4, random_state=20260909, n_init=10).fit(scores[fit])
    return dict(centroid_accuracy=float(np.mean(pred == labels[~fit])),
                kmeans_ari=float(adjusted_rand_score(labels[~fit], km.predict(scores[~fit]))),
                predictions=pred.tolist(), cluster_predictions=km.predict(scores[~fit]).tolist())


def streaming(features, final_basis):
    x = np.array(features, dtype=np.float64)
    order = np.random.default_rng(20260909).permutation(len(x))
    mu = np.zeros(x.shape[1])
    scatter = np.zeros((x.shape[1], x.shape[1]))
    factor = np.zeros((x.shape[1], 0))
    history = []
    for n, index in enumerate(order, start=1):
        delta = x[index] - mu
        mu += delta/n
        update = delta * np.sqrt((n-1)/n)
        scatter += np.outer(update, update)
        augmented = np.column_stack([factor, update])
        u, s, _ = np.linalg.svd(augmented, full_matrices=False)
        factor = u[:, :4] * s[None, :4]
        if n in (12, 24, 36, 48):
            _, _, exact_basis = pilot.pca(x[order[:n]])
            history.append(dict(n=n,
                exact_to_final_overlap=float(np.sum((exact_basis.T @ final_basis)**2)/4),
                truncated_to_final_overlap=float(np.sum((u[:, :4].T @ final_basis)**2)/4),
                truncated_to_current_exact_overlap=float(np.sum((u[:, :4].T @ exact_basis)**2)/4)))
    direct = (x-x.mean(0)).T @ (x-x.mean(0))
    return dict(order=order.tolist(), history=history,
                welford_relative_error=float(np.linalg.norm(scatter-direct)/np.linalg.norm(direct)))


def fixture():
    x = np.random.default_rng(7).normal(size=(48, 16)) @ np.diag(np.linspace(4, 0.1, 16))
    mu, values, v = pilot.pca(x)
    direct = (x-mu).T @ (x-mu)/47
    assert np.max(np.abs(v.T @ v-np.eye(4))) < 1e-12
    assert np.max(np.abs(direct @ v-v*values[:4])) < 1e-12
    stream = streaming(x, v)
    assert stream['welford_relative_error'] < 1e-12
    assert abs(stream['history'][-1]['exact_to_final_overlap']-1) < 1e-12
    print('CPU fixture PASS: Gram lift, eigensystem, orthonormality, exact Welford, final subspace.')


def main():
    data = json.loads((OUT/'dataset.json').read_text())
    groups = list(dict.fromkeys(r['group'] for r in data['rows']))
    labels = np.array([groups.index(r['group']) for r in data['rows']])
    fit = np.array([r['split'] == 'fit' for r in data['rows']])
    readouts = json.loads((OUT/'readouts.json').read_text())
    features = np.load(OUT/'features.npz')
    results = dict(groups=groups, fit_n=int(sum(fit)), heldout_n=int(sum(~fit)), analyses={})
    saved = {}
    for name in ['gradient_11','gradient_17','activation_11','activation_17']:
        x = features[name].astype(np.float64)
        mu, values, v = pilot.pca(x[fit])
        scores = (x-mu) @ v
        random = []
        for seed in range(32):
            rv = np.linalg.qr(np.random.default_rng(20260909+seed).normal(size=(x.shape[1],4)))[0]
            random.append(evaluate((x-mu) @ rv, labels, fit))
        cosine = np.array(readouts[name+'_jlens']['cosine'])[:4,:4]
        plain_cosine = np.array(readouts[name+'_plain']['cosine'])[:4,:4]
        raw_corr = [float(np.corrcoef(np.linalg.norm(x,axis=1), scores[:,i])[0,1]) for i in range(4)]
        results['analyses'][name] = dict(eigenvalues=values.tolist(),
            variance_fractions=(values/values.sum()).tolist(), top4_variance=float(values[:4].sum()/values.sum()),
            evaluation=evaluate(scores, labels, fit), random_evaluations=random,
            random_accuracy_mean=float(np.mean([r['centroid_accuracy'] for r in random])),
            random_accuracy_range=[float(min(r['centroid_accuracy'] for r in random)),float(max(r['centroid_accuracy'] for r in random))],
            random_ari_mean=float(np.mean([r['kmeans_ari'] for r in random])),
            jlens_max_abs_pc_cosine=float(np.max(np.abs(cosine-np.eye(4)))),
            plain_max_abs_pc_cosine=float(np.max(np.abs(plain_cosine-np.eye(4)))),
            feature_norm_pc_correlations=raw_corr, streaming=streaming(x[fit],v),
            gram_orthonormal_error=float(np.max(np.abs(v.T@v-np.eye(4)))))
        saved[name+'_scores'], saved[name+'_basis'], saved[name+'_mean'] = scores, v, mu
    np.savez_compressed(OUT/'analysis_arrays.npz', **saved)
    pilot.write_json(OUT/'metrics.json', results)
    print(json.dumps({k:dict(top4_variance=v['top4_variance'],evaluation=v['evaluation'],
                           random_accuracy_mean=v['random_accuracy_mean']) for k,v in results['analyses'].items()},indent=2))


if __name__ == '__main__':
    fixture() if '--fixture' in sys.argv else main()
