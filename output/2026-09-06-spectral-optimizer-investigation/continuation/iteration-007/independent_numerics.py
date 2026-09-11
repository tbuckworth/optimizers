"""Independent, import-inert NumPy arithmetic for the I7 numerical contract.

No producer, state, loss, response, experiment or Torch imports. This module
does not load artifacts or choose scientific cells. Native tensor arguments
must be finite native-endian NumPy float32 arrays, with the documented shapes;
labels must be int64. No normalization, deduplication or dtype repair occurs.
Reference arithmetic uses float64. All returned arrays own their storage and
all returned scalars are Python primitives. Malformed/nonfinite inputs and
nonfinite intermediate results raise ValueError. Screening violations instead
return passed=False with diagnostics: callers must preserve and stop, not turn
them into scientific nulls. These are engineering screens, not CUDA proofs.
"""

import math
from numbers import Integral, Real

import numpy as np

U32, U64 = 2.0**-24, 2.0**-53
H32, H64, K = 2.0**-126, 2.0**-1022, 32.0
ADAMW_OPTIONS = {
    "lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .01,
    "amsgrad": False, "maximize": False, "foreach": False, "capturable": False,
    "differentiable": False, "fused": False, "decoupled_weight_decay": True,
}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        converted = float(value)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real scalar")
    return converted


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite {name}")
    return value


def _native(value, name, ndim=None, shape=None):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must be a native-endian float32 NumPy array")
    if not value.size or (ndim is not None and value.ndim != ndim) or (shape is not None and value.shape != shape):
        raise ValueError(f"invalid {name} shape")
    _finite(value, name)
    return np.array(value, dtype=np.float64, order="C", copy=True)


def _vector64(value, name):
    original = np.asarray(value)
    if original.dtype.kind not in "fiu" or original.ndim != 1 or not original.size:
        raise ValueError(f"{name} must be a nonempty real vector")
    return _finite(np.array(original, dtype=np.float64, copy=True), name)


def gamma(n, unit_roundoff):
    n, unit_roundoff = _integer(n, "n"), _scalar(unit_roundoff, "unit_roundoff")
    product = n*unit_roundoff
    if not 0 < product < 1:
        raise ValueError("gamma requires 0 < n*unit_roundoff < 1")
    return product/(1-product)


def loss_tolerance(producer, auditor):
    """Frozen TL for scalar CPU64 CE comparisons."""
    a, b = _scalar(producer, "producer CE"), _scalar(auditor, "auditor CE")
    return 1e-10 + 1e-11*max(1., abs(a), abs(b))


def gradient_tolerance(producer, auditor):
    """Frozen Tq for two same-length finite real vectors; returns owned float64."""
    a, b = _vector64(producer, "producer gradient"), _vector64(auditor, "auditor gradient")
    if a.shape != b.shape:
        raise ValueError("gradient shapes differ")
    return _finite(1e-12 + 1e-9*np.maximum(np.abs(a), np.abs(b)), "Tq")


def sum_roundoff(terms):
    """Rsum: allowance for two K-term CPU64 reductions, not relative error."""
    terms = _vector64(terms, "terms")
    value = 2*gamma(2*terms.size+4, U64)*np.abs(terms).sum() + 4*terms.size*H64
    return _scalar(value, "Rsum")


def dot_roundoff(left, right):
    """Rdot with the supplied vector length p (50890 for the study model)."""
    x, y = _vector64(left, "left"), _vector64(right, "right")
    if x.shape != y.shape:
        raise ValueError("dot shapes differ")
    with np.errstate(over="raise", invalid="raise"):
        try:
            value = 2*gamma(2*x.size+4, U64)*np.abs(x*y).sum() + 4*x.size*H64
        except FloatingPointError as error:
            raise ValueError("nonfinite Rdot intermediate") from error
    return _scalar(value, "Rdot")


def numerical_sign(producer, auditor, fixed_bound):
    """Resolve only same nonzero signs with min(abs(value)) > fixed ceiling.

    A zero, disagreement or value within the fixed ceiling remains reported
    with status='unresolved'; this is not equivalence or a statistical test.
    Domain-null values are not accepted here and must remain a separate mask.
    """
    a, b, bound = (_scalar(x, n) for x, n in
                   ((producer, "producer"), (auditor, "auditor"), (fixed_bound, "fixed_bound")))
    if bound < 0:
        raise ValueError("fixed_bound must be nonnegative")
    sa, sb = (int(x > 0)-int(x < 0) for x in (a, b))
    resolved = sa == sb and sa != 0 and min(abs(a), abs(b)) > bound
    reason = None if resolved else ("sign_disagreement" if sa != sb else
                                   "zero_estimate" if sa == 0 else "within_fixed_ceiling")
    return dict(producer=a, auditor=b, fixed_bound=bound, producer_sign=sa, auditor_sign=sb,
                status=("resolved_positive" if sa > 0 else "resolved_negative") if resolved else "unresolved",
                reason=reason)


def native_ce_concordance(native32_ce, cpu64_ce):
    """Finite discordance is retained, not raised; nonfinite CE raises ValueError."""
    native, reference = _scalar(native32_ce, "native CE"), _scalar(cpu64_ce, "CPU64 CE")
    error = _scalar(abs(native-reference), "CE discrepancy")
    threshold = 5e-6*max(1., abs(reference))
    return dict(native32_ce=native, cpu64_ce=reference, absolute_error=error,
                flag_threshold=threshold, discordant=bool(error > threshold))


def mlp_ce_gradient(parameters, inputs, labels, chunk_size=500, *, guard=None):
    """Explicit NumPy Linear-ReLU-Linear mean CE and chain-rule gradient.

    parameters is [W1(h,d), b1(h), W2(c,h), b2(c)] in fixed flattening order.
    inputs is (n,d) native float32; labels is (n,) int64 with 0 <= label < c.
    Positive n,d,h,c and chunk_size are required. Ordered contiguous chunks
    use sum CE/sum gradients, then divide once by n; duplicate rows count each
    time. Exact-zero ReLU derivative is zero. Returns mean_ce, flat gradient,
    separately owned parameter_gradients, chunk_means and chunk_counts. Optional
    guard(label) checks bracket copies/validation, chunks and reduction; an
    observational callback must preserve RNG and its exceptions propagate.
    """
    if guard is not None and not callable(guard):
        raise ValueError("guard must be callable or None")
    def check(label):
        if guard is not None:
            guard(label)
    check("independent_loss.before_validation_copy")
    if not isinstance(parameters, (tuple, list)) or len(parameters) != 4:
        raise ValueError("parameters must contain W1,b1,W2,b2 in that order")
    w1 = _native(parameters[0], "W1", ndim=2)
    h, d = w1.shape
    b1 = _native(parameters[1], "b1", shape=(h,))
    w2 = _native(parameters[2], "W2", ndim=2)
    c = w2.shape[0]
    if w2.shape[1] != h:
        raise ValueError("W2 hidden dimension differs")
    b2 = _native(parameters[3], "b2", shape=(c,))
    x = _native(inputs, "inputs", ndim=2)
    if x.shape[1] != d:
        raise ValueError("input feature dimension differs")
    n = x.shape[0]
    if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype(np.int64) or labels.shape != (n,):
        raise ValueError("labels must be an int64 NumPy vector matching the input count")
    if np.any(labels < 0) or np.any(labels >= c):
        raise ValueError("label outside class range")
    chunk_size = _integer(chunk_size, "chunk_size")
    check("independent_loss.after_validation_copy")
    totals = [np.zeros_like(a) for a in (w1, b1, w2, b2)]
    total_loss, chunk_means, chunk_counts = 0., [], []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            check(f"independent_loss.before_chunk.{start}.{end}")
            try:
                xx, yy = x[start:start+chunk_size], labels[start:start+chunk_size]
                z = xx@w1.T+b1
                hidden = np.maximum(z, 0.)
                logits = hidden@w2.T+b2
                shifted = logits-logits.max(axis=1, keepdims=True)
                exponentials = np.exp(shifted)
                denominator = exponentials.sum(axis=1, keepdims=True)
                chunk_sum = float((np.log(denominator[:, 0])-shifted[np.arange(len(yy)), yy]).sum())
                total_loss += chunk_sum
                chunk_means.append(chunk_sum/len(yy))
                chunk_counts.append(int(len(yy)))
                dlogits = exponentials/denominator
                dlogits[np.arange(len(yy)), yy] -= 1.
                dz = (dlogits@w2)*(z > 0.)
                pieces = (dz.T@xx, dz.sum(axis=0), dlogits.T@hidden, dlogits.sum(axis=0))
                for total, piece in zip(totals, pieces):
                    total += piece
            except FloatingPointError as error:
                raise ValueError("nonfinite MLP intermediate") from error
            check(f"independent_loss.after_chunk.{start}.{end}")
    check("independent_loss.before_reduction")
    gradients = [np.array(_finite(total/n, "MLP gradient"), copy=True) for total in totals]
    result = dict(mean_ce=_scalar(total_loss/n, "mean CE"),
                gradient=np.concatenate([a.ravel() for a in gradients]),
                parameter_gradients=gradients, chunk_means=chunk_means, chunk_counts=chunk_counts)
    check("independent_loss.after_reduction")
    return result


def validate_adamw_options(options):
    """Require exactly the frozen arithmetic option keys (no parameter objects)."""
    if not isinstance(options, dict) or set(options) != set(ADAMW_OPTIONS):
        raise ValueError("AdamW arithmetic option membership differs")
    for key, expected in ADAMW_OPTIONS.items():
        value = options[key]
        if isinstance(expected, bool):
            valid = type(value) is bool and value is expected
        elif key == "betas":
            valid = isinstance(value, tuple) and len(value) == 2 and all(
                _scalar(a, key) == b for a, b in zip(value, expected))
        else:
            valid = _scalar(value, key) == expected
        if not valid:
            raise ValueError(f"AdamW option {key} differs")


def adamw_reference(theta, delivered, moment, variance, next_step, *, options):
    """Native before arrays share one nonempty shape; old variance must be >=0.

    next_step is the exact positive post-step integer counter, not t-1.
    Returns owned float64 moment/variance/theta and corresponding K32 envelopes.
    It does not substitute reference theta for a saved scientific endpoint.
    """
    validate_adamw_options(options)
    t = _integer(next_step, "next_step")
    theta = _native(theta, "theta")
    a, m, v = [_native(x, name, shape=theta.shape) for x, name in
               ((delivered, "delivered"), (moment, "moment"), (variance, "variance"))]
    if np.any(v < 0):
        raise ValueError("old variance is negative")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            M, V = .9*m+.1*a, .999*v+.001*a*a
            b, s, q = math.sqrt(1-.999**t), .001/(1-.9**t), 1-.001*.01
            D = np.sqrt(V)/b+1e-8
            em, ev = K*U32*(np.abs(m)+.1*np.abs(a))+K*H32, K*U32*V+K*H32
            lo, hi = np.sqrt(np.maximum(V-ev, 0))/b+1e-8, np.sqrt(V+ev)/b+1e-8
            ed = np.maximum(D-lo, hi-D)+K*U32*hi+K*H32
            lower = np.maximum(D-ed, 1e-8/2)
            eq = em/lower+np.abs(M)*ed/(D*lower)
            ep = abs(s)*eq+K*U32*(np.abs(q*theta)+abs(s)*(np.abs(M)/D+eq))+K*H32
            endpoint = q*theta-s*M/D
        except FloatingPointError as error:
            raise ValueError("nonfinite Adam reference intermediate") from error
    values = dict(moment=M, variance=V, theta=endpoint, moment_envelope=em,
                  variance_envelope=ev, theta_envelope=ep)
    return {key: np.array(_finite(value, key), copy=True) for key, value in values.items()}


def _screen(observed, expected, envelope):
    error = np.abs(observed-expected)
    failed = error > envelope
    zero_failure = bool(np.any(failed & (envelope == 0)))
    positive = envelope > 0
    return dict(passed=not bool(failed.any()), max_absolute_error=float(error.max()),
                max_envelope=float(envelope.max()),
                max_error_envelope_ratio=None if zero_failure else
                (float((error[positive]/envelope[positive]).max()) if positive.any() else 0.),
                ratio_null_reason="zero_envelope_mismatch" if zero_failure else None,
                failing_flat_indices=np.flatnonzero(failed).tolist(),
                error_l2=float(np.linalg.norm(error.ravel())), envelope_l2=float(np.linalg.norm(envelope.ravel())))


def audit_adamw(theta, delivered, moment, variance, next_step, native_after, *, options):
    """native_after has exactly theta,moment,variance,next_step; no hidden option defaults."""
    if not isinstance(native_after, dict) or set(native_after) != {"theta", "moment", "variance", "next_step"}:
        raise ValueError("native_after membership differs")
    t = _integer(next_step, "next_step")
    if _integer(native_after["next_step"], "native post counter") != t:
        raise ValueError("native post counter differs")
    reference = adamw_reference(theta, delivered, moment, variance, t, options=options)
    checks = {}
    for key in ("theta", "moment", "variance"):
        observed = _native(native_after[key], f"native after {key}", shape=reference[key].shape)
        if key == "variance" and np.any(observed < 0):
            raise ValueError("new variance is negative")
        checks[key] = _screen(observed, reference[key], reference[key+"_envelope"])
    displacement = _native(native_after["theta"], "native after theta")-_native(theta, "theta")
    norm = float(np.linalg.norm(displacement.ravel()))
    return dict(passed=all(item["passed"] for item in checks.values()), checks=checks,
                native_displacement_norm=norm,
                endpoint_envelope_to_displacement=None if norm == 0 else checks["theta"]["envelope_l2"]/norm,
                displacement_ratio_null_reason="zero_native_displacement" if norm == 0 else None)


def projection_reference(raw, basis):
    """raw is (p,), basis is None or (p,k), 1<=k<=min(p,32), native float32.

    Uses the represented B B^T operator, not QR or an exact orthoprojector.
    None means native identity, with a zero envelope (signed-zero audit exact).
    """
    g = _native(raw, "raw", ndim=1)
    if basis is None:
        return dict(candidate=g.copy(), envelope=np.zeros_like(g), operator="identity", rank=None)
    B = _native(basis, "basis", ndim=2)
    p, k = B.shape
    if p != g.size or not 1 <= k <= min(p, 32):
        raise ValueError("basis shape/rank differs")
    with np.errstate(over="raise", invalid="raise"):
        try:
            z = B.T@g
            ez = gamma(2*p, U32)*(np.abs(B).T@np.abs(g))
            envelope = np.abs(B)@ez + gamma(2*k, U32)*(np.abs(B)@(np.abs(z)+ez))
            envelope += 32*(p+k+1)*H32
            candidate = B@z
        except FloatingPointError as error:
            raise ValueError("nonfinite projection reference intermediate") from error
    return dict(candidate=np.array(_finite(candidate, "projection"), copy=True),
                envelope=np.array(_finite(envelope, "projection envelope"), copy=True),
                operator="basis_outer_product", rank=int(k))


def audit_projection(raw, basis, native_candidate):
    """Return component screen; None-basis additionally requires exact native value bytes."""
    reference = projection_reference(raw, basis)
    native = _native(native_candidate, "native candidate", shape=raw.shape)
    result = _screen(native, reference["candidate"], reference["envelope"])
    exact = raw.tobytes(order="C") == native_candidate.tobytes(order="C") if basis is None else None
    result.update(operator=reference["operator"], rank=reference["rank"], identity_bytes_equal=exact,
                  identity_mismatch_flat_indices=[])
    if exact is False:
        result["passed"] = False
        result["identity_mismatch_flat_indices"] = np.flatnonzero(
            raw.ravel().view(np.uint32) != native_candidate.ravel().view(np.uint32)).tolist()
    return result
