"""Frozen completion follow-up. See output/2026-09-10-j-lens-completions/protocol.md."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/2026-09-10-j-lens-completions'
MODEL = 'Qwen/Qwen3.5-0.8B'
MODEL_REV = '2fc06364715b967f1860aea9cf38778875588b17'
LENS_REV = '0731326edff4ae730ffc5356fe1a4728c748b3a6'
LENS_FILE = 'qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt'
LAYERS = [11, 17]
SEED = 20260910
# Prefixes deliberately stop before a multiword semantic clause.
PAIRS = {
 'astronomy': [
  ('The telescope revealed', ' a distant galaxy surrounded by faint stars.'),
  ('The spacecraft photographed', ' deep craters on the surface of Mars.'),
  ('The comet released', ' a long tail of dust near the sun.'),
  ('The observatory detected', ' radio pulses from a rotating neutron star.'),
  ('The satellite measured', ' changes in the orbit of the moon.'),
  ('The astronomer identified', ' a new planet beyond the solar system.')],
 'cooking': [
  ('The chef prepared', ' a vegetable soup with fresh herbs.'),
  ('The baker kneaded', ' the bread dough before putting it in the oven.'),
  ('The cook added', ' chopped garlic and butter to the sauce.'),
  ('The recipe combined', ' flour and eggs into a smooth batter.'),
  ('The kitchen assistant sliced', ' ripe tomatoes for the evening salad.'),
  ('The pastry chef decorated', ' the chocolate cake with whipped cream.')],
 'football': [
  ('The striker scored', ' the winning goal with a powerful header.'),
  ('The goalkeeper blocked', ' a penalty shot near the left post.'),
  ('The referee awarded', ' a free kick after a dangerous tackle.'),
  ('The midfielder delivered', ' a precise pass across the football pitch.'),
  ('The defender cleared', ' the ball away from the penalty area.'),
  ('The team captain lifted', ' the league trophy after the final match.')],
 'programming': [
  ('The programmer fixed', ' a bug in the array sorting function.'),
  ('The compiler translated', ' source code into executable machine instructions.'),
  ('The developer tested', ' the database query with an empty table.'),
  ('The debugger located', ' an invalid memory access inside the loop.'),
  ('The software engineer implemented', ' an exception handler for network errors.'),
  ('The application returned', ' a formatted response to the client request.')],
}


def dataset():
    rows = []
    for group, pairs in PAIRS.items():
        for i, (prefix, completion) in enumerate(pairs):
            for style, frame in [('plain', ''), ('note', 'A brief factual note:\n')]:
                rows.append(dict(id=f'{group}-{i}-{style}', pair=f'{group}-{i}',
                    group=group, style=style, split='fit' if i < 4 else 'heldout',
                    prefix=frame+prefix, completion=completion))
    return rows


def write_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def completion_loss(logits, ids, boundary):
    # ids[boundary:] are targets; prediction sites begin at boundary-1.
    return torch.nn.functional.cross_entropy(logits[0, boundary-1:-1].float(),
                                             ids[0, boundary:], reduction='none')


def pca(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0)
    z = x-mean
    values, u = np.linalg.eigh(z @ z.T/(len(z)-1))
    values, u = values[::-1].copy(), u[:, ::-1]
    assert values[3] > 1e-14
    v = z.T @ u[:, :4]/np.sqrt((len(z)-1)*values[:4])
    v *= np.sign(v[np.abs(v).argmax(0), np.arange(4)])
    return mean, values.clip(0), v


def acquire():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    import jlens
    assert os.environ.get('JLENS_PARENT_GPU_RELEASE') == '1', 'Parent coordination required'
    rows = dataset()
    paths = [Path(__file__), OUT/'protocol.md']
    provenance = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        files=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT.parent/'jacobian-lens', text=True).strip(),
        invocation=os.environ.get('INVOCATION_ID'), started_utc=datetime.now(timezone.utc).isoformat())
    write_json(OUT/'acquisition-start.json', provenance)
    write_json(OUT/'dataset.json', rows)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    start = time.monotonic()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='eager').cuda()
    model = jlens.from_hf(hf, tok)
    assert not hf.training and all(not p.requires_grad and p.grad is None for p in hf.parameters())
    assert model.d_model == 1024 and model.n_layers == 24
    versions = [p._version for p in hf.parameters()]
    lens_path = hf_hub_download('neuronpedia/jacobian-lens', LENS_FILE, revision=LENS_REV, local_files_only=True)
    lens = jlens.JacobianLens.load(lens_path)
    features = {f'{kind}_{l}': [] for kind in ['activation','gradient','first_gradient'] for l in LAYERS}
    inputs = []
    for i, row in enumerate(rows):
        enc = tok(row['prefix']+row['completion'], return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc['offset_mapping']
        boundary_char = len(row['prefix'])
        assert not any(a < boundary_char < b for a,b in offsets)
        boundary = next(k for k,(a,b) in enumerate(offsets) if a >= boundary_char)
        ids = torch.tensor([enc['input_ids']], device='cuda')
        assert 0 < boundary < ids.shape[1]-1 and ids.shape[1] <= 96
        captured = {}
        def hook(layer):
            def capture(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                if layer == LAYERS[0]:
                    h = h.detach().requires_grad_(True)
                captured[layer] = h
                return (h,*output[1:]) if isinstance(output,tuple) else h
            return capture
        handles = [model.layers[l].register_forward_hook(hook(l)) for l in LAYERS]
        try:
            result = model.forward(ids)
            # Decoder last_hidden_state already passed final norm. Do not norm twice.
            logits = model._lm_head(result.last_hidden_state)
            losses = completion_loss(logits, ids, boundary)
            assert len(losses) == ids.shape[1]-boundary
            gs = torch.autograd.grad(losses.mean(), [captured[l] for l in LAYERS], retain_graph=True)
            first = torch.autograd.grad(losses[0], [captured[l] for l in LAYERS])
            for l,g,f in zip(LAYERS,gs,first):
                for kind, tensor in [('activation',captured[l]),('gradient',g),('first_gradient',f)]:
                    features[f'{kind}_{l}'].append(tensor[0,boundary-1].detach().float().cpu().numpy())
            inputs.append(dict(id=row['id'], ids=ids[0].tolist(), boundary=boundary,
                               token_losses=losses.detach().float().cpu().tolist()))
        finally:
            for h in handles:
                h.remove()
        del result, logits, losses, gs, first, captured
        if (i+1)%8 == 0:
            print(f'{i+1}/{len(rows)} acquired; {time.monotonic()-start:.1f}s', flush=True)
    arrays = {k:np.stack(v) for k,v in features.items()}
    assert all(np.isfinite(x).all() for x in arrays.values())
    np.savez_compressed(OUT/'features.npz', **arrays)
    write_json(OUT/'inputs.json', inputs)
    fit = np.array([r['split']=='fit' for r in rows])
    readouts = {}
    for name,x in arrays.items():
        l = int(name.rsplit('_',1)[1])
        mu,values,v = pca(x[fit])
        random = np.linalg.qr(np.random.default_rng(SEED).normal(size=(1024,4)))[0].T
        sign = -1 if 'gradient' in name else 1
        directions = np.concatenate([sign*x[~fit], sign*mu[None], v.T, -v.T, random, -random])
        names = [r['id'] for r in rows if r['split']=='heldout']+['mean']+[
            f'{kind}{j+1}{s}' for kind in ['PC','random'] for s in ['+','-'] for j in range(4)]
        norms = np.linalg.norm(directions, axis=1)
        assert (norms > 0).all()
        directions = directions/norms[:,None]
        for method in ['jlens','plain']:
            tokens,scores,indices = [],[],[]
            for direction in directions:
                h = torch.tensor(direction,device='cuda',dtype=torch.float32)
                with torch.no_grad():
                    z = lens.transport(h,l) if method=='jlens' else h
                    readout = model.unembed(z).float()
                assert torch.isfinite(readout).all()
                top = readout.topk(12)
                indices.append(top.indices.cpu().tolist())
                scores.append(top.values.cpu().tolist())
                tokens.append([tok.decode([k]) for k in indices[-1]])
            readouts[f'{name}_{method}'] = dict(names=names,norms=norms.tolist(),token_ids=indices,tokens=tokens,scores=scores)
    write_json(OUT/'readouts.json',readouts)
    assert versions == [p._version for p in hf.parameters()]
    assert all(p.grad is None and not p.requires_grad for p in hf.parameters()) and not hf.training
    write_json(OUT/'acquisition.json',dict(model=MODEL,model_revision=MODEL_REV,lens_revision=LENS_REV,
        lens_sha256=hashlib.sha256(Path(lens_path).read_bytes()).hexdigest(),lens_n_prompts=lens.n_prompts,
        elapsed_seconds=time.monotonic()-start,gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
        parameters_unchanged=True,parameter_gradients_absent=True,provenance=provenance))


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    j = torch.tensor([[1.,2.],[0.,3.]],dtype=torch.float64)
    h = torch.tensor([.7,-.2],dtype=torch.float64,requires_grad=True)
    w = torch.randn(5,2,dtype=torch.float64)
    def norm(z):
        return z/torch.sqrt(z.square().mean()+1e-5)
    z = j@h
    logits = w@norm(z)
    loss = -logits.log_softmax(0)[2]
    g, = torch.autograd.grad(loss,h)
    dn = torch.autograd.functional.jacobian(norm,z)
    e = torch.nn.functional.one_hot(torch.tensor(2),5)
    expected = j.T@dn.T@w.T@(logits.softmax(0)-e)
    assert torch.allclose(g,expected)
    assert not torch.allclose(j@g,j.T@g)
    assert float(g@(-g)) < 0
    ids = torch.tensor([[0,1,2,3,4]])
    logits = torch.randn(1,5,5)
    actual = completion_loss(logits,ids,2)
    expected = torch.stack([-logits[0,t-1].log_softmax(0)[ids[0,t]] for t in range(2,5)])
    assert torch.allclose(actual,expected)
    x = np.random.default_rng(4).normal(size=(16,8))
    mu,val,v = pca(x)
    c = np.cov(x,rowvar=False)
    assert np.allclose(c@v,v*val[:4]) and np.allclose(v.T@v,np.eye(4))
    rows = dataset()
    assert len(rows)==48
    assert not ({r['pair'] for r in rows if r['split']=='fit'} & {r['pair'] for r in rows if r['split']=='heldout'})
    for group in PAIRS:
        for style in ['plain','note']:
            assert sum(r['group']==group and r['style']==style and r['split']=='fit' for r in rows)==4
    print('PASS: completion indexing; postnorm VJP; descent sign; J direction; PCA; content-disjoint balanced split.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['fixture','acquire'])
    args = parser.parse_args()
    {'fixture':fixture,'acquire':acquire}[args.stage]()
