"""CPU-only fixed follow-up analysis; never acquires model features."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('completion',ROOT/'experiments/j_lens_completions.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
OUT = experiment.OUT


def evaluate(x, y, fit, style):
    classes = np.unique(y)
    centers = np.stack([x[fit & (y==c)].mean(0) for c in classes])
    pred = classes[((x[~fit,None]-centers[None])**2).sum(-1).argmin(1)]
    km = KMeans(n_clusters=4,random_state=experiment.SEED,n_init=10).fit(x[fit])
    nearest = ((x[~fit,None]-x[fit][None])**2).sum(-1).argmin(1)
    style_centers = np.stack([x[fit & (style==s)].mean(0) for s in [0,1]])
    style_pred = ((x[~fit,None]-style_centers[None])**2).sum(-1).argmin(1)
    return dict(centroid_accuracy=float(np.mean(pred==y[~fit])),
        kmeans_ari=float(adjusted_rand_score(y[~fit],km.predict(x[~fit]))),
        individual_1nn_accuracy=float(np.mean(y[fit][nearest]==y[~fit])),
        style_accuracy=float(np.mean(style_pred==style[~fit])),
        per_style_topic_accuracy=[float(np.mean(pred[style[~fit]==s]==y[~fit][style[~fit]==s])) for s in [0,1]],
        predictions=pred.tolist())


def main():
    import torch
    torch.set_num_threads(1)
    rows = json.loads((OUT/'dataset.json').read_text())
    groups = list(experiment.PAIRS)
    y = np.array([groups.index(r['group']) for r in rows])
    style = np.array([int(r['style']=='note') for r in rows])
    fit = np.array([r['split']=='fit' for r in rows])
    features = np.load(OUT/'features.npz')
    results,saved = {},{}
    for name in features.files:
        x = features[name].astype(np.float64)
        mu,values,v = experiment.pca(x[fit])
        scores = (x-mu)@v
        random = []
        for seed in range(32):
            q = np.linalg.qr(np.random.default_rng(experiment.SEED+seed).normal(size=(x.shape[1],4)))[0]
            random.append(evaluate((x-mu)@q,y,fit,style))
        mean_direction = mu/max(np.linalg.norm(mu),1e-30)
        results[name] = dict(pca=evaluate(scores,y,fit,style),full=evaluate(x-mu,y,fit,style),
            mean_direction=evaluate(((x-mu)@mean_direction)[:,None],y,fit,style),
            constant_mean_accuracy=.25,random=random,eigenvalues=values.tolist(),
            top4_variance=float(values[:4].sum()/values.sum()),norms=np.linalg.norm(x,axis=1).tolist())
        saved[name+'_scores'],saved[name+'_basis'],saved[name+'_mean'] = scores,v,mu
    inputs = json.loads((OUT/'inputs.json').read_text())
    results['input_summary'] = dict(completion_lengths=[len(r['token_losses']) for r in inputs],
        mean_losses=[float(np.mean(r['token_losses'])) for r in inputs],groups=groups)
    experiment.write_json(OUT/'metrics.json',results)
    np.savez_compressed(OUT/'analysis_arrays.npz',**saved)
    print(json.dumps({k:v['pca'] for k,v in results.items() if k!='input_summary'},indent=2))


if __name__ == '__main__':
    main()
