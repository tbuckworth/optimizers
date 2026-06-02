#!/usr/bin/env python3
"""Per-sample gradient-covariance filter (rank-B, UNCENTERED, project-onto-top-k).

Difference from WeightCovarianceFilterV2:
  - V2 tracks the covariance of the *batch-mean* gradient over *steps* (temporal),
    centred by a running mean.
  - This tracks the *uncentered second moment of the per-sample gradients across
    examples within a step*: C = EMA[ (1/B) sum_i g_i g_i^T ]. No mean subtraction.

Then it descends on the batch-mean gradient PROJECTED ONTO the top-k eigenspace of C
(same "keep the top-k" operation as V2, just sourced from per-sample structure).

The covariance update is a streaming rank-B EMA SVD (the rank-B generalization of the
rank-1 update in weight_cov_optimizer_v2.py; see research/rank1_svd_explainer.html §10):
  C' = lam*C + ((1-lam)/B) * sum_i g_i g_i^T
done via a (k+B)x(k+B) Gram eigendecomposition, never materializing p x p.

Per-sample gradients are obtained with torch.func.vmap(grad(...)) on a subsample of
size `cov_batch` each step (the EMA accumulates cross-sample structure over steps).
"""
import math
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad


class PerSampleCovarianceFilter:
    def __init__(self, model, base_optimizer, rank=50, decay=0.99, warmup=100,
                 cov_batch=256, energy_threshold=None):
        self.model = model
        self.base_optimizer = base_optimizer
        self.rank = rank
        self.decay = decay
        self.warmup = warmup
        self.cov_batch = cov_batch
        self.energy_threshold = energy_threshold
        self.param_list = list(model.parameters())
        self.n_params = sum(p.numel() for p in self.param_list)
        self.V = None   # (p, k)
        self.S = None   # (k,) on CPU
        self.step_count = 0

    # ---- per-sample gradients via functorch ----
    def _persample_grads(self, X, y):
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        names = list(params.keys())

        def loss_fn(ps, bs, x, t):
            out = functional_call(self.model, (ps, bs), (x.unsqueeze(0),))
            return F.cross_entropy(out, t.unsqueeze(0))

        g = vmap(grad(loss_fn), in_dims=(None, None, 0, 0))(params, buffers, X, y)
        B = X.shape[0]
        return torch.cat([g[k].reshape(B, -1) for k in names], dim=1)  # (B, p)

    # ---- streaming rank-B uncentered EMA SVD update ----
    def _update_svd_rankB(self, Ws):
        # Ws: (p, B)
        device = Ws.device
        p, B = Ws.shape
        lam = self.decay
        sd = math.sqrt(lam)
        sn = math.sqrt((1 - lam) / B)

        if self.V is None:
            U, Sv, _ = torch.linalg.svd(Ws, full_matrices=False)   # U (p, B)
            s = Sv * sn                                            # second-moment singular values
            k = min(self.rank, U.shape[1])
            self.V = U[:, :k].contiguous()
            self.S = s[:k].detach().cpu()
            return

        k = self.V.shape[1]
        S = self.S.to(device)
        P = self.V.T @ Ws                                          # (k, B)

        G = torch.zeros(k + B, k + B, device=device)
        G[:k, :k] = torch.diag(lam * S * S)
        cross = sd * sn * (S.unsqueeze(1) * P)                     # (k, B)
        G[:k, k:] = cross
        G[k:, :k] = cross.T
        G[k:, k:] = (sn * sn) * (Ws.T @ Ws)                        # (B, B)

        evals, evecs = torch.linalg.eigh(G)
        evals = evals.flip(0); evecs = evecs.flip(1)
        posmask = evals > 1e-12
        evals = evals[posmask]; evecs = evecs[:, posmask]
        new_k = min(self.rank, len(evals))
        if self.energy_threshold is not None and len(evals) > 0:
            frac = torch.cumsum(evals, 0) / evals.sum()
            k_energy = int(torch.searchsorted(frac, self.energy_threshold).item()) + 1
            new_k = max(1, min(self.rank, k_energy, len(evals)))
        evals = evals[:new_k]; evecs = evecs[:, :new_k]
        s_new = evals.sqrt()

        coeffs = evecs / s_new.unsqueeze(0)                        # (k+B, new_k)
        top = sd * S.unsqueeze(1) * coeffs[:k]                     # (k, new_k)
        bot = sn * coeffs[k:]                                      # (B, new_k)
        V_new = self.V @ top + Ws @ bot                            # (p, new_k)
        self.V = V_new.contiguous()
        self.S = s_new.detach().cpu()

    def _project(self, g):
        if self.V is None:
            return g
        return self.V @ (self.V.T @ g)

    def _flat_grad(self):
        return torch.cat([p.grad.reshape(-1) for p in self.param_list])

    def _set_flat_grad(self, flat):
        off = 0
        for p in self.param_list:
            n = p.numel()
            p.grad = flat[off:off + n].reshape(p.shape)
            off += n

    def step(self, X, y):
        self.step_count += 1
        # descent gradient = full-batch mean
        self.base_optimizer.zero_grad()
        loss = F.cross_entropy(self.model(X), y)
        loss.backward()
        mean_grad = self._flat_grad()

        # covariance from a per-sample subsample
        B = X.shape[0]
        if self.cov_batch < B:
            idx = torch.randperm(B, device=X.device)[:self.cov_batch]
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y
        psg = self._persample_grads(Xs, ys)        # (b, p)
        self._update_svd_rankB(psg.T.contiguous())  # (p, b)

        if self.step_count > self.warmup and self.V is not None:
            self._set_flat_grad(self._project(mean_grad))
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
        return loss.item()

    def effective_rank(self):
        if self.S is None:
            return 0.0
        s2 = self.S ** 2; s2 = s2[s2 > 1e-12]
        if len(s2) == 0:
            return 0.0
        pr = s2 / s2.sum()
        return math.exp(-(pr * pr.log()).sum().item())


# --------------------------- self-test ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    def run_case(p, B, rank, lam, steps, true_dim):
        f = PerSampleCovarianceFilter.__new__(PerSampleCovarianceFilter)
        f.rank, f.decay, f.energy_threshold = rank, lam, None
        f.V, f.S = None, None
        Q, _ = torch.linalg.qr(torch.randn(p, true_dim))  # fixed true subspace
        C = torch.zeros(p, p); first = True
        for _ in range(steps):
            Ws = Q @ torch.randn(true_dim, B) if true_dim < p else torch.randn(p, B)
            M = (Ws @ Ws.T) / B
            C = ((1 - lam) * M) if first else (lam * C + (1 - lam) * M)
            first = False
            f._update_svd_rankB(Ws)
        evals_bf, evecs_bf = torch.linalg.eigh(C)
        evals_bf = evals_bf.flip(0); evecs_bf = evecs_bf.flip(1)
        r = min(rank, f.V.shape[1], true_dim)
        eig_err = (evals_bf[:r] - f.S[:r] ** 2).abs().max().item()
        overlap = (torch.linalg.matrix_norm(evecs_bf[:, :r].T @ f.V[:, :r]) ** 2 / r).item()
        return eig_err, overlap, evals_bf[:5].tolist(), (f.S[:5] ** 2).tolist()

    # Correctness regime: true rank (6) <= kept rank (12) -> no truncation loss, must match.
    eig_err, overlap, bf, est = run_case(p=20, B=4, rank=12, lam=0.9, steps=40, true_dim=6)
    print(f"[exact regime, true_dim=6<=rank=12]")
    print(f"  top eigenvalue max-abs-err = {eig_err:.3e}")
    print(f"  subspace overlap (1.0=perfect) = {overlap:.5f}")
    print(f"  brute-force top5 = {[round(x,4) for x in bf]}")
    print(f"  streaming   top5 = {[round(x,4) for x in est]}")
    assert eig_err < 1e-4, "eigenvalue mismatch in exact regime"
    assert overlap > 0.9999, "subspace mismatch in exact regime"

    # Approximation regime: true rank (20) > kept rank (8) -> truncated, report quality only.
    eig_err2, overlap2, _, _ = run_case(p=20, B=4, rank=8, lam=0.9, steps=40, true_dim=20)
    print(f"[truncated regime, true_dim=20>rank=8] top8 err={eig_err2:.3f} overlap={overlap2:.3f} (approx, expected)")
    print("SELF-TEST PASSED (exact regime matches brute force)")
