#!/usr/bin/env python3
"""Spectral gradient filter — the core idea of this repo, in one file.

Wrap any PyTorch optimizer. Before each update, project the gradient onto the
top-k eigendirections of a streaming estimate of the gradient covariance, so the
optimizer only ever steps in directions the gradient has been *consistently*
pointing. It is a "coherence amplifier": it keeps whatever the gradient agrees
about across steps and discards the rest.

Why it's cheap: we never form the p×p covariance. We keep a rank-k factorization
(V, S) and update it with a streaming rank-1 SVD each step. Only a tiny
(k+1)×(k+1) eigendecomposition runs on CPU; everything else stays on the GPU.
Cost is ~2× a bare Adam step.

--------------------------------------------------------------------------------
Quick start — drop it into ANY training loop (any loss, any model):

    from spectral_filter import SpectralGradientFilter

    base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    filt = SpectralGradientFilter(model, base_opt, rank=200)

    for x, y in loader:
        base_opt.zero_grad()
        loss = my_loss_fn(model(x), y)
        loss.backward()
        filt.filter_grad()        # <-- filters .grad in place; call between
        base_opt.step()           #     backward() and the optimizer step

For plain classification there is a one-call convenience that does the
forward/backward for you:

    for x, y in loader:
        loss, diagnostics = filt.step(x, y)   # cross-entropy under the hood

--------------------------------------------------------------------------------
Knobs (see __init__ for the full list):
  rank        hard cap on the number of eigendirections kept (default 200)
  decay       EMA decay of the covariance estimate (default 0.99)
  warmup      steps to observe before filtering kicks in (default 100)
  weighting   "hard" top-k projection, or "soft" eigenvalue^alpha reweighting
  normalize   basis: "none" (covariance), "var" (correlation), "degree" (affinity)
  adaptive    rank rule: "none", "effrank", or "gap"
"""

import math

import torch
import torch.nn.functional as F


class SpectralGradientFilter:
    def __init__(self, model, base_optimizer, rank=200, decay=0.99,
                 warmup=100, filter_strength=1.0, energy_threshold=None,
                 adaptive="none", normalize="none",
                 weighting="hard", alpha=1.0, soft_residual=True,
                 stable_update=True,
                 relative_eig_tol=1e-8, absolute_eig_floor=0.0,
                 stabilize_every=100, parameters=None):
        self.model = model
        self.base_optimizer = base_optimizer
        self.rank = rank            # hard cap on kept directions
        self.decay = decay
        self.warmup = warmup
        self.filter_strength = filter_strength
        # Graded eigenvalue weighting (the "soft" filter).
        #   weighting="hard"  -> project onto top-k subspace, all kept dirs weight 1
        #                        (the original hard top-k filter)
        #   weighting="soft"  -> reweight each retained eigendirection i by
        #                        w_i = (lambda_i / lambda_max)^alpha, lambda_i = S_i^2,
        #                        then renormalize the result to preserve ||g|| (so
        #                        alpha is a pure "where to point" knob, decoupled from
        #                        the global learning rate).
        # The alpha spectrum (soft):
        #   alpha = 0          -> if soft_residual: identity (no filter, == base opt);
        #                         else: uniform weights == hard top-k filter
        #   alpha = 1          -> update proportional to how agreed-upon a direction is
        #   alpha -> +inf      -> collapses toward the single dominant direction
        #   alpha < 0          -> whitening / natural-gradient (amplify rare dirs)
        # soft_residual=True keeps g's component OUTSIDE the retained basis at weight 1
        # so that alpha=0 is a genuine no-filter control; False drops it (subspace-only).
        self.weighting = weighting
        self.alpha = alpha
        self.soft_residual = soft_residual
        # If set (e.g. 0.99), keep the smallest number of top eigenvectors that
        # capture this fraction of the spectral energy, capped at `rank`. The
        # eigenvalues are already computed each step, so this is ~free.
        self.energy_threshold = energy_threshold
        # Which BASIS to project onto (one categorical knob):
        #   "none"   -> eigenvectors of the covariance C            (default)
        #   "var"    -> eigenvectors of the correlation D^-1/2 C D^-1/2, D=diag(C)
        #               (per-weight variance divided out: structure over magnitude)
        #   "degree" -> eigenvectors of D^-1/2 C D^-1/2, D=row-sums of C
        #               (the normalized-affinity / spectral-clustering basis;
        #                degree of a SIGNED covariance can be <=0, so it is clamped)
        # All three project onto the top-k eigenspace of D^-1/2 C D^-1/2, which is the
        # column space of A = D^-1/2 V diag(S); we project via A (AᵀA)^-1 Aᵀ — no
        # eigendecomposition needed. normalize!="none" ignores the adaptive proj_k.
        self.normalize = normalize
        # Adaptive rank rule (overrides energy_threshold if not "none"):
        #   "none"    -> keep `rank` (fixed), or energy_threshold if set
        #   "effrank" -> keep round(effective rank) = round(exp(entropy of spectrum))
        #   "gap"     -> keep up to the largest multiplicative gap in the log-spectrum
        # Each step the rank-1 update already produces k+1 candidate eigenpairs
        # (the +1 is the new gradient's component orthogonal to the current basis),
        # so the rule only chooses the truncation cutoff; rank can grow by at most
        # 1 per step (a single rank-1 observation adds at most one new direction).
        self.adaptive = adaptive
        self.stable_update = bool(stable_update)
        # Numerical controls for the streaming eigensystem. Eigenvalues below
        # max(absolute_eig_floor, relative_eig_tol * lambda_max) are discarded.
        # Periodic repair re-orthogonalizes V while preserving the represented
        # covariance; set stabilize_every=None to disable scheduled repairs.
        if relative_eig_tol < 0 or absolute_eig_floor < 0:
            raise ValueError("eigenvalue tolerances must be non-negative")
        if stabilize_every is not None and stabilize_every < 1:
            raise ValueError("stabilize_every must be positive or None")
        self.relative_eig_tol = float(relative_eig_tol)
        self.absolute_eig_floor = float(absolute_eig_floor)
        self.stabilize_every = stabilize_every

        model_parameters = {id(parameter) for parameter in model.parameters()}
        optimizer_parameters = [
            parameter
            for group in base_optimizer.param_groups
            for parameter in group["params"]
        ]
        optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
        if not optimizer_ids.issubset(model_parameters):
            raise ValueError(
                "base_optimizer contains a parameter that is not in model"
            )
        selected = optimizer_parameters if parameters is None else list(parameters)
        if any(id(parameter) not in optimizer_ids for parameter in selected):
            raise ValueError("filtered parameters must belong to base_optimizer")
        if any(id(parameter) not in model_parameters for parameter in selected):
            raise ValueError("filtered parameters must belong to model")
        self.param_list = []
        seen_parameters = set()
        for parameter in selected:
            if parameter.requires_grad and id(parameter) not in seen_parameters:
                self.param_list.append(parameter)
                seen_parameters.add(id(parameter))
        if not self.param_list:
            raise ValueError("base_optimizer has no trainable model parameters")
        self.n_params = sum(p.numel() for p in self.param_list)

        self.V = None  # (p, k) top eigenvectors
        self.S = None  # (k,) singular values
        self.proj_k = None  # adaptive #directions to project onto (None = all of V)
        self.step_count = 0

        # Running mean for centering
        self.grad_mean = None
        self.stabilization_count = 0
        self.max_orthogonality_error = 0.0

    def _get_flat_grad(self):
        if any(parameter.grad is None for parameter in self.param_list):
            raise RuntimeError(
                "all filtered optimizer parameters must have gradients; "
                "use a separate filter for conditionally active parameters"
            )
        if any(parameter.grad.is_sparse for parameter in self.param_list):
            raise RuntimeError("sparse gradients are not supported")
        return torch.cat([p.grad.reshape(-1) for p in self.param_list])

    def _set_flat_grad(self, flat_grad):
        offset = 0
        for p in self.param_list:
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].reshape(p.shape)
            offset += numel

    @staticmethod
    def _sym_eigh_desc(matrix_cpu_f64):
        """Return eigenpairs of a small symmetric matrix, largest first."""
        matrix_cpu_f64 = (matrix_cpu_f64 + matrix_cpu_f64.T) * 0.5
        eigvals, eigvecs = torch.linalg.eigh(matrix_cpu_f64)
        return eigvals.flip(0), eigvecs.flip(1)

    def _truncate_eigensystem(self, eigvals, eigvecs):
        """Apply scale-aware positivity and configured rank truncation."""
        if eigvals.numel() == 0:
            return eigvals, eigvecs
        largest = eigvals[0].clamp_min(0.0)
        floor = max(
            self.absolute_eig_floor,
            self.relative_eig_tol * float(largest.item()),
        )
        keep = eigvals > floor
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]
        if eigvals.numel() == 0:
            return eigvals, eigvecs

        # Adaptive projection rules retain a broad estimation basis. Energy
        # thresholding retains its historical basis-truncating semantics.
        new_k = min(self.rank, eigvals.numel())
        if self.adaptive == "none" and self.energy_threshold is not None:
            frac = torch.cumsum(eigvals, 0) / eigvals.sum()
            k_energy = int(torch.searchsorted(frac, self.energy_threshold).item()) + 1
            new_k = max(1, min(new_k, k_energy))
        return eigvals[:new_k], eigvecs[:, :new_k]

    def _repair_representation(self):
        """Re-orthogonalize V without changing its represented covariance."""
        if self.V is None or self.V.shape[1] == 0:
            return
        q, r = torch.linalg.qr(self.V, mode="reduced")
        r64 = r.detach().double().cpu()
        variances = self.S.detach().double().square()
        small = (r64 * variances.unsqueeze(0)) @ r64.T
        eigvals, rotation = self._sym_eigh_desc(small)
        eigvals, rotation = self._truncate_eigensystem(eigvals, rotation)
        if eigvals.numel() == 0:
            self.V = None
            self.S = None
            self.proj_k = None
            return
        self.V = (q @ rotation.to(device=q.device, dtype=q.dtype)).contiguous()
        self.S = eigvals.sqrt().cpu()
        self.stabilization_count += 1
        self._update_proj_k()

    def orthogonality_error(self):
        """Spectral norm of V^T V - I; zero when no basis exists."""
        if self.V is None:
            return 0.0
        gram = self.V.T @ self.V
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return float(torch.linalg.matrix_norm(gram - eye, ord=2).item())

    def _update_svd(self, g):
        if self.stable_update:
            return self._update_svd_stable(g)
        return self._update_svd_legacy(g)

    def _update_svd_legacy(self, g):
        """Historical streaming update retained for result reproducibility."""
        device = g.device

        if self.grad_mean is None:
            self.grad_mean = g.detach().clone()
        else:
            self.grad_mean.mul_(self.decay).add_(g.detach(), alpha=1 - self.decay)
        centered = g.detach() - self.grad_mean

        if self.V is None:
            norm = centered.norm()
            if norm > 1e-12:
                self.V = (centered / norm).unsqueeze(1).contiguous()
                self.S = norm.unsqueeze(0).cpu()
            return

        k = self.V.shape[1]
        sqrt_decay = math.sqrt(self.decay)
        sqrt_observation = math.sqrt(1 - self.decay)
        coefficients = self.V.T @ centered
        centered_norm_sq = centered.dot(centered).item()

        singular_values = self.S
        coefficients_cpu = coefficients.cpu()
        gram = torch.zeros(k + 1, k + 1)
        gram[:k, :k] = torch.diag(
            sqrt_decay * sqrt_decay * singular_values * singular_values
        )
        cross = (
            sqrt_decay
            * sqrt_observation
            * singular_values
            * coefficients_cpu
        )
        gram[:k, k] = cross
        gram[k, :k] = cross
        gram[k, k] = sqrt_observation * sqrt_observation * centered_norm_sq

        eigvals, eigvecs = torch.linalg.eigh(gram)
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)
        positive = eigvals > 1e-12
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]
        new_k = min(self.rank, len(eigvals))
        if (
            self.adaptive == "none"
            and self.energy_threshold is not None
            and len(eigvals) > 0
        ):
            fraction = torch.cumsum(eigvals, 0) / eigvals.sum()
            energy_k = (
                int(torch.searchsorted(fraction, self.energy_threshold).item())
                + 1
            )
            new_k = max(1, min(self.rank, energy_k, len(eigvals)))
        eigvals = eigvals[:new_k]
        eigvecs = eigvecs[:, :new_k]
        new_singular_values = eigvals.sqrt()

        reconstruction = eigvecs / new_singular_values.unsqueeze(0)
        top = (
            sqrt_decay
            * singular_values.unsqueeze(1)
            * reconstruction[:k]
        ).to(device)
        bottom = (sqrt_observation * reconstruction[k]).to(device)
        self.V = (
            self.V @ top + centered.unsqueeze(1) * bottom.unsqueeze(0)
        ).contiguous()
        self.S = new_singular_values
        self._update_proj_k()

    def _update_svd_stable(self, g):
        """Stably update the low-rank streaming covariance representation.

        The covariance is diagonalized directly in the orthonormal augmented
        basis [V, q]. Heavy p-by-k operations stay on the compute device; only
        the small k-by-k symmetric eigensystem is solved on the CPU in fp64.
        """
        device, dtype = g.device, g.dtype

        if self.grad_mean is None:
            self.grad_mean = g.detach().clone()
        else:
            self.grad_mean.mul_(self.decay).add_(g.detach(), alpha=1 - self.decay)
        centered = g.detach() - self.grad_mean

        if self.V is None:
            norm = centered.norm()
            if norm > 0:
                self.V = (centered / norm).unsqueeze(1).contiguous()
                self.S = norm.detach().double().cpu().unsqueeze(0)
            return

        if self.stabilize_every and self.step_count % self.stabilize_every == 0:
            self._repair_representation()
            if self.V is None:
                return

        # Two-pass Gram-Schmidt controls residual error when V has accumulated
        # small fp32 orthogonality drift.
        coefficients = self.V.T @ centered
        residual = centered - self.V @ coefficients
        correction = self.V.T @ residual
        residual = residual - self.V @ correction
        coefficients = coefficients + correction
        residual_norm = residual.norm()
        residual_floor = max(
            1e-12,
            10 * torch.finfo(dtype).eps * float(centered.norm().item()),
        )
        has_residual = bool(residual_norm > residual_floor)

        k = self.V.shape[1]
        coefficients64 = coefficients.detach().double().cpu()
        old_variances = self.S.detach().double().square()
        decay, observation_weight = float(self.decay), float(1 - self.decay)
        size = k + int(has_residual)
        small = torch.zeros(size, size, dtype=torch.float64)
        small[:k, :k] = (
            torch.diag(decay * old_variances)
            + observation_weight * torch.outer(coefficients64, coefficients64)
        )

        residual_direction = None
        if has_residual:
            residual_norm64 = float(residual_norm.double().item())
            cross = observation_weight * coefficients64 * residual_norm64
            small[:k, k] = cross
            small[k, :k] = cross
            small[k, k] = observation_weight * residual_norm64 * residual_norm64
            residual_direction = residual / residual_norm

        eigvals, rotation = self._sym_eigh_desc(small)
        eigvals, rotation = self._truncate_eigensystem(eigvals, rotation)
        if eigvals.numel() == 0:
            self.V = None
            self.S = None
            self.proj_k = None
            return

        rotation_device = rotation.to(device=device, dtype=dtype)
        updated = self.V @ rotation_device[:k]
        if has_residual:
            updated = (
                updated
                + residual_direction.unsqueeze(1)
                * rotation_device[k].unsqueeze(0)
            )
        self.V = updated.contiguous()
        self.S = eigvals.sqrt().cpu()
        self._update_proj_k()

        # Check often during basis growth, then periodically. Repair only when
        # the measured drift is material; scheduled repair handles large ranks.
        if self.V.shape[1] <= 256 and (
            self.step_count <= 10 or self.step_count % 50 == 0
        ):
            error = self.orthogonality_error()
            self.max_orthogonality_error = max(
                self.max_orthogonality_error, error
            )
            if not math.isfinite(error) or error > 5e-3:
                self._repair_representation()

    def _update_proj_k(self):
        """How many of the (broad) top directions to actually project onto this step.

        Measured on the full retained spectrum self.S, so it can jump to the right
        value immediately rather than ratcheting up one per step.
        """
        if self.adaptive == "none" or self.S is None or len(self.S) == 0:
            self.proj_k = None
            return
        ev = (self.S ** 2)
        ev = ev[ev > 1e-12]
        n = len(ev)
        if n == 0:
            self.proj_k = None
            return
        if self.adaptive == "effrank":
            p = ev / ev.sum()
            eff = float(torch.exp(-(p * (p + 1e-30).log()).sum()).item())
            self.proj_k = max(1, min(self.rank, n, int(round(eff))))
        elif self.adaptive == "gap":
            if n == 1:
                self.proj_k = 1
            else:
                logs = (ev + 1e-30).log()
                drops = logs[:-1] - logs[1:]
                self.proj_k = max(1, min(self.rank, n, int(torch.argmax(drops).item()) + 1))
        else:
            self.proj_k = None

    def _project_gradient(self, g):
        if self.V is None:
            return g
        if self.normalize == "none" and self.weighting == "soft":
            V = self.V if self.proj_k is None else self.V[:, :self.proj_k]
            S = self.S.to(g)
            if self.proj_k is not None:
                S = S[:self.proj_k]
            lam = S * S
            lam_max = lam.max().clamp_min(1e-30)
            ratio = (lam / lam_max).clamp_min(1e-12)        # (k,) in (0, 1]
            w = ratio.pow(self.alpha).clamp(max=1e3)        # graded weights
            coeffs = V.T @ g                                # (k,)
            if self.soft_residual:
                # g + V diag(w-1) Vᵀg : retained dirs reweighted, complement untouched
                g_projected = g + V @ ((w - 1.0) * coeffs)
            else:
                g_projected = V @ (w * coeffs)              # subspace-only
            gn = g.norm()
            pn = g_projected.norm().clamp_min(1e-12)
            g_projected = g_projected * (gn / pn)           # preserve ‖g‖
        elif self.normalize == "none":
            V = self.V if self.proj_k is None else self.V[:, :self.proj_k]
            g_projected = V @ (V.T @ g)
        else:
            # Project onto col(A), A = D^{-1/2} V diag(S) — the top-k eigenspace of
            # the diagonally-normalized covariance. No eigendecomposition needed.
            S = self.S.to(g)
            Vs = self.V * S.unsqueeze(0)                  # (p, k) = V diag(S)
            if self.normalize == "var":
                D = (Vs * Vs).sum(1)                      # C_ii = per-weight variance
            else:  # "degree": row-sums of C = Vs @ (S * colsum)
                colsum = self.V.sum(0)                    # (k,)
                D = Vs @ (S * colsum)                     # (p,)
                D = D.abs()                               # signed degree -> clamp |.|
            Dinv = (D.clamp_min(1e-12)).rsqrt()           # (p,)
            A = Vs * Dinv.unsqueeze(1)                    # (p, k)
            G = A.T @ A                                   # (k, k)
            G += 1e-6 * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
            coeffs = torch.linalg.solve(G, A.T @ g)       # (k,)
            g_projected = A @ coeffs
        if self.filter_strength < 1.0:
            return (1 - self.filter_strength) * g + self.filter_strength * g_projected
        return g_projected

    def filter_grad(self):
        """Filter the gradients currently held in model.parameters(), in place.

        Call this AFTER loss.backward() and BEFORE base_optimizer.step(). It is
        task-agnostic — no assumption of classification, any loss / model / output
        shape works. Each call updates the streaming covariance estimate and, once
        past `warmup` steps, replaces every parameter's .grad with its projection
        onto the top-k gradient-covariance eigenspace.

        Returns a diagnostics dict (singular values, effective/basis/kept rank).
        """
        self.step_count += 1
        flat_grad = self._get_flat_grad()

        # Update covariance estimate every step
        self._update_svd(flat_grad)

        # Apply filter after warmup
        if self.step_count > self.warmup:
            filtered_grad = self._project_gradient(flat_grad)
            self._set_flat_grad(filtered_grad)

        return self._diagnostics()

    def step(self, inputs, targets):
        """Convenience classification step (cross-entropy).

        Does zero_grad → forward → cross_entropy → backward → filter_grad() →
        base_optimizer.step() for you. For any other loss/model, run your own loop
        and call filter_grad() between backward() and the optimizer step instead.

        Returns (loss_value, diagnostics_dict).
        """
        self.base_optimizer.zero_grad()
        loss = F.cross_entropy(self.model(inputs), targets)
        loss.backward()
        loss_val = loss.item()

        diagnostics = self.filter_grad()

        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

        diagnostics["loss"] = loss_val
        return loss_val, diagnostics

    def _diagnostics(self):
        diagnostics = {
            "step": self.step_count,
            "filtering_active": self.step_count > self.warmup,
            "stabilization_count": self.stabilization_count,
            "max_orthogonality_error": self.max_orthogonality_error,
            "relative_eig_tol": self.relative_eig_tol,
            "stable_update": self.stable_update,
        }
        if self.S is not None:
            diagnostics["top_singular_values"] = self.S[:5].tolist()
            total = (self.S ** 2).sum().item()
            if total > 1e-12:
                diagnostics["variance_in_top5"] = (self.S[:5] ** 2).sum().item() / total
            diagnostics["effective_rank"] = self._effective_rank()
            # basis_rank: directions tracked; kept_rank: directions actually projected
            # onto (== basis_rank unless an adaptive rule narrows the projection).
            basis_rank = self.V.shape[1] if self.V is not None else 0
            diagnostics["basis_rank"] = basis_rank
            diagnostics["kept_rank"] = self.proj_k if self.proj_k is not None else basis_rank
        return diagnostics

    def _effective_rank(self):
        if self.S is None:
            return 0.0
        s2 = self.S ** 2
        s2 = s2[s2 > 1e-12]
        if len(s2) == 0:
            return 0.0
        p = s2 / s2.sum()
        entropy = -(p * p.log()).sum().item()
        return math.exp(entropy)

    def reset(self):
        self.V = None
        self.S = None
        self.proj_k = None
        self.step_count = 0
        self.grad_mean = None
        self.stabilization_count = 0
        self.max_orthogonality_error = 0.0

    def zero_grad(self):
        self.base_optimizer.zero_grad()


# Backwards-compatible alias. The class was historically named
# WeightCovarianceFilterV2; existing experiment scripts import that name.
WeightCovarianceFilterV2 = SpectralGradientFilter
