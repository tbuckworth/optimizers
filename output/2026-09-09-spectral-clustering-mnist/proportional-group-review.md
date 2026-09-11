# Proportional-group algebra review

Date: 2026-09-09
Verdict: **PASS — the stated identities and qualifications are correct.**

For `g_i = a_i z` with `a_i > 0` and `z != 0`, unweighted group averaging gives

`||Cg||^2 / ||g||^2 = (sum_i a_i)^2 / (n sum_i a_i^2)`.

With population variance, this equals `mean(a)^2 / mean(a^2) =
1 / (1 + CV(a)^2)`. It is one for equal amplitudes and approaches `1/n` as one
positive amplitude dominates and the others approach zero. The word
“approach” is important because strict positivity excludes attaining the
one-sparse endpoint.

The vectors `v_c = a_c / ||a_c||` have disjoint support, hence are orthonormal
across groups. Therefore `C_a = sum_c v_c v_c^T` plus singleton isolate terms is
an orthogonal projector, the component formula is correct, and it exactly
preserves a group vector proportional to `a_c`. Positive weights do not cover
anti-proportional coordinates; the note correctly avoids claiming otherwise.

For any orthogonal projector `C`, decomposing `g = Cg + (I-C)g` verifies

`||(.5I + .5C)g||^2 / ||g||^2 = .25 + .75 ||Cg||^2 / ||g||^2`.

Thus the lower bound of one quarter in squared norm, or one half in vector norm,
is exact, while the mixture generally does not recover the amplitude-weighted
direction.

The only material assumption to preserve in later discussion is that
`||F_i||` is merely a prospective historical-amplitude proxy. With centered,
decayed, rank-truncated covariance it need not equal the current coefficient
`a_i`, even when a local proportional model is approximately useful. The note
already frames this as a feasible unlabeled candidate rather than an oracle and
separates incoming-gradient geometry from AdamW dynamics.

This is a theoretical construct check, not evidence that proportional groups
occur in the neural run, not a measured mechanism, and not authorization for a
new arm or a retrofit of the locked experiment.
