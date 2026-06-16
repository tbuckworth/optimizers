# Legacy

Superseded code, kept for provenance.

- `weight_cov_optimizer.py` — the **v1** spectral filter. Replaced by the
  canonical implementation at the repo root: `../../spectral_filter.py`
  (historically `weight_cov_optimizer_v2.py`). The v2 streaming rank-1 SVD update
  is cheaper and is what every current experiment uses.
- `run_single_weight_cov.py`, `run_weight_cov_experiments.py` — v1 run harnesses,
  superseded by `../run_single_weight_cov_v2.py`.

These still import and run, but you almost certainly want the root module instead.
