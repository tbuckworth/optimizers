"""Bounded frozen-model covariance pilot; see its pre-acquisition protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/2026-09-09-j-lens-pilot'
MODEL = 'Qwen/Qwen3.5-0.8B'
MODEL_REV = '2fc06364715b967f1860aea9cf38778875588b17'
LENS_REV = '0731326edff4ae730ffc5356fe1a4728c748b3a6'
LENS_FILE = 'qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt'
LAYERS = [11, 17]

SENTENCES = {
 'astronomy': [
  'The telescope revealed a distant galaxy beyond the bright stars.',
  'Astronomers measured the orbit of a planet around its sun.',
  'A comet released a long tail of dust as it approached the star.',
  'The observatory recorded radiation from a massive black hole.',
  'Saturn has rings made from countless pieces of ice and rock.',
  'The moon cast a shadow across Earth during the solar eclipse.',
  'A supernova scattered heavy elements into interstellar space.',
  'Scientists searched the night sky for a newly discovered asteroid.',
  'The spacecraft photographed craters on the surface of Mars.',
  'Gravity bends the light of distant galaxies into luminous arcs.',
  'Jupiter is a gas giant with powerful storms in its atmosphere.',
  'The Milky Way contains billions of stars and clouds of gas.',
  'An infrared detector found a cool dwarf star near the nebula.',
  'The rocket carried a probe toward the outer solar system.',
  'A pulsar sends regular beams of radio waves through space.',
  'The lunar lander collected samples of ancient volcanic rock.',
  'Two neutron stars collided and released gravitational waves.',
  'The redshift of the galaxy indicates that it is moving away.'
 ],
 'cooking': [
  'The chef chopped fresh onions before heating oil in the pan.',
  'Bake the bread until its crust becomes golden and crisp.',
  'The recipe calls for flour, butter, sugar, and two eggs.',
  'Simmer the soup gently while stirring in the chopped vegetables.',
  'A pinch of salt brings out the flavor of ripe tomatoes.',
  'The dough needs time to rise before it goes into the oven.',
  'Roast the chicken with garlic and herbs until it is tender.',
  'The cook whisked cream into the sauce to make it smooth.',
  'Boil the pasta in salted water and drain it carefully.',
  'Fresh basil and olive oil give the salad a rich aroma.',
  'The pastry was filled with apples and dusted with cinnamon.',
  'Slice the mushrooms thinly and fry them over medium heat.',
  'The baker folded melted chocolate into the cake batter.',
  'Steam the dumplings in a covered basket above boiling water.',
  'A sharp kitchen knife makes it easier to dice the carrots.',
  'Marinate the fish in lemon juice before placing it on the grill.',
  'The risotto absorbed the warm stock as the rice softened.',
  'Sprinkle grated cheese over the dish just before serving.'
 ],
 'football': [
  'The striker scored a goal after receiving a precise pass.',
  'The goalkeeper dived to save a powerful shot from the winger.',
  'A defender cleared the ball away from the penalty area.',
  'The referee awarded a free kick after the dangerous tackle.',
  'The team trained on the pitch before the important match.',
  'Supporters cheered as the captain lifted the league trophy.',
  'The midfielder controlled possession and started another attack.',
  'A corner kick created a chance for the tall central defender.',
  'The coach made a substitution during the second half.',
  'The visiting club won the game with a late penalty.',
  'An offside flag stopped the forward as he ran toward goal.',
  'The players celebrated their victory in the crowded stadium.',
  'The fullback crossed the ball toward a teammate near the net.',
  'A yellow card warned the captain after his second foul.',
  'The tournament final ended with a dramatic penalty shootout.',
  'The manager changed formation to strengthen the midfield.',
  'The home crowd applauded a skillful dribble past two opponents.',
  'A header from the center forward struck the crossbar.'
 ],
 'programming': [
  'The programmer fixed a bug in the function that sorts the array.',
  'A compiler translates source code into executable instructions.',
  'The variable stores an integer that is updated inside the loop.',
  'The software test failed because the expected value was incorrect.',
  'A database query retrieves records that match the specified condition.',
  'The developer added an exception handler around the network request.',
  'A recursive function calls itself until the base case is reached.',
  'The application crashed when it tried to access invalid memory.',
  'The class defines methods for creating and modifying each object.',
  'A version control system tracks changes to files in the repository.',
  'The script parses the input and writes the result to a file.',
  'The server returns a response when the client sends a request.',
  'The debugger paused execution at the breakpoint in the method.',
  'A missing semicolon caused a syntax error during compilation.',
  'The interface accepts a callback that handles incoming events.',
  'The algorithm uses a hash table to find duplicate entries.',
  'A thread acquired the lock before modifying the shared buffer.',
  'The package manager installed the library and its dependencies.'
 ]}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare():
    rows = []
    for group, sentences in SENTENCES.items():
        for j, sentence in enumerate(sentences):
            rows.append(dict(id=f'{group}-{j:02}', group=group,
                             split='fit' if j < 12 else 'heldout', text=sentence))
    write_json(OUT/'dataset.json', dict(target_text=' The', layers=LAYERS, rows=rows))
    print('Prepared 72 fixed sentences, 48 fit and 24 heldout.', flush=True)


def acquire():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import jlens
    from huggingface_hub import hf_hub_download
    protected = ['acquisition.json', 'features.npz', 'readouts.json',
                 'acquired_features.npz', 'acquired_inputs.json', 'acquisition-start.json']
    if any((OUT/name).exists() for name in protected):
        raise FileExistsError('Acquisition evidence already exists; reconcile without rerunning')
    source_paths = ['experiments/j_lens_spectral_pilot.py',
                    'output/2026-09-09-j-lens-pilot/dataset.json',
                    'output/2026-09-09-j-lens-pilot/protocol.md']
    provenance = dict(
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(jlens.__file__).parents[1],text=True).strip(),
        sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths},
        systemd_invocation_id=os.environ.get('INVOCATION_ID'),
        started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    with (OUT/'acquisition-start.json').open('x') as handle:
        json.dump(provenance, handle, indent=2)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260909)
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    started = time.monotonic()
    data = json.loads((OUT/'dataset.json').read_text())
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV,
        dtype=torch.bfloat16, local_files_only=True, attn_implementation='eager').cuda()
    model = jlens.from_hf(hf, tok)
    assert not hf.training and all(not p.requires_grad for p in hf.parameters())
    assert model.n_layers == 24 and model.d_model == 1024
    lens_path = hf_hub_download('neuronpedia/jacobian-lens', LENS_FILE,
        revision=LENS_REV, local_files_only=True)
    lens = jlens.JacobianLens.load(lens_path)
    assert lens.d_model == model.d_model and all(l in lens.jacobians for l in LAYERS)
    versions = [p._version for p in hf.parameters()]
    target_ids = tok.encode(data['target_text'], add_special_tokens=False)
    assert len(target_ids) == 1, target_ids
    target = torch.tensor(target_ids, device='cuda')
    arrays = {f'{kind}_{layer}': [] for layer in LAYERS for kind in ('activation', 'gradient')}
    losses, token_ids = [], []
    for i, row in enumerate(data['rows']):
        captured = {}
        def hook(layer):
            def capture(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                if layer == LAYERS[0]:
                    h = h.detach().requires_grad_(True)
                captured[layer] = h
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return capture
        handles = [model.layers[l].register_forward_hook(hook(l)) for l in LAYERS]
        ids = model.encode(row['text'], max_length=96)
        token_ids.append(ids[0].cpu().tolist())
        try:
            result = model.forward(ids)
            logits = model._lm_head(result.last_hidden_state[:, -1])
            loss = torch.nn.functional.cross_entropy(logits.float(), target)
            gs = torch.autograd.grad(loss, [captured[l] for l in LAYERS])
            for l, g in zip(LAYERS, gs):
                arrays[f'activation_{l}'].append(captured[l][0, -1].detach().float().cpu().numpy())
                arrays[f'gradient_{l}'].append(g[0, -1].detach().float().cpu().numpy())
            losses.append(float(loss))
        finally:
            for handle in handles:
                handle.remove()
        del result, logits, loss, gs, captured
        if (i + 1) % 12 == 0:
            print(f'acquired {i+1}/72; {time.monotonic()-started:.1f}s; peak GPU {torch.cuda.max_memory_allocated()/1024**3:.2f}GiB', flush=True)
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    assert all(np.isfinite(x).all() for x in arrays.values())
    # Preserve completed inference before readout/analysis, so later errors do not
    # require another model acquisition.
    np.savez_compressed(OUT/'acquired_features.npz', **arrays)
    write_json(OUT/'acquired_inputs.json', dict(target_ids=target_ids, token_ids=token_ids, losses=losses))
    fit = np.array([r['split'] == 'fit' for r in data['rows']])
    readouts = {}
    for name, features in list(arrays.items()):
        layer = int(name.rsplit('_', 1)[1])
        mean, values, vectors = pca(features[fit])
        rng = np.random.default_rng(20260909)
        random = np.linalg.qr(rng.normal(size=(model.d_model, 4)))[0]
        dirs = np.concatenate([vectors, random], axis=1).T
        signed = np.concatenate([dirs, -dirs])
        for control, transformed in [('jlens', signed @ lens.jacobians[layer].numpy().T), ('plain', signed)]:
            with torch.no_grad():
                scores = model.unembed(torch.tensor(transformed, device='cuda', dtype=torch.float32)).float().cpu()
            assert torch.isfinite(scores).all(), f'Nonfinite readout {name}/{control}'
            top = scores.topk(12, dim=-1)
            centered = scores - scores.mean(dim=1, keepdim=True)
            unit = torch.nn.functional.normalize(centered, dim=1)
            readouts[f'{name}_{control}'] = dict(
                order=['PC1+','PC2+','PC3+','PC4+','Random1+','Random2+','Random3+','Random4+',
                       'PC1-','PC2-','PC3-','PC4-','Random1-','Random2-','Random3-','Random4-'],
                tokens=[[tok.decode([int(t)]) for t in row] for row in top.indices],
                token_ids=top.indices.tolist(), scores=top.values.tolist(),
                cosine=(unit @ unit.T).tolist())
        arrays[f'J_{layer}'] = lens.jacobians[layer].numpy() if f'J_{layer}' not in arrays else arrays[f'J_{layer}']
    assert versions == [p._version for p in hf.parameters()]
    assert all(p.grad is None and not p.requires_grad for p in hf.parameters())
    np.savez_compressed(OUT/'features.npz', **arrays)
    write_json(OUT/'readouts.json', readouts)
    write_json(OUT/'acquisition.json', dict(model=MODEL, model_revision=MODEL_REV,
        lens_revision=LENS_REV, lens_file=LENS_FILE, lens_n_prompts=lens.n_prompts,
        lens_sha256=hashlib.sha256(Path(lens_path).read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256((OUT/'dataset.json').read_bytes()).hexdigest(),
        target_ids=target_ids, token_ids=token_ids, losses=losses,
        elapsed_seconds=time.monotonic()-started, gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
        parameters_unchanged=True, parameter_gradients_absent=True,
        torch_version=torch.__version__, layers=LAYERS, provenance=provenance))
    print('ACQUISITION COMPLETE', flush=True)


def pca(features):
    x = torch.tensor(features, dtype=torch.float64)
    assert torch.isfinite(x).all(), 'Nonfinite features'
    mean = x.mean(0)
    x -= mean
    values, u = torch.linalg.eigh(x @ x.T / (len(x)-1))
    values, u = values.flip(0).clamp_min(0), u.flip(1)
    assert torch.isfinite(values).all() and torch.all(values[:4] > 0), 'Invalid leading eigenvalues'
    vectors = (x.T @ u[:, :4]) / torch.sqrt((len(x)-1)*values[:4])[None]
    # A fixed sign convention for reproducible plots; decode both signs regardless.
    sign = torch.sign(vectors[vectors.abs().argmax(0), torch.arange(4)])
    return mean.numpy(), values.numpy(), (vectors*sign).numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'acquire'])
    args = parser.parse_args()
    {'prepare': prepare, 'acquire': acquire}[args.stage]()
