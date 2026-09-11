# Prospective interpretation of the legacy action substitution

Written without reading action-intervention outcomes. This note is algebra and prospective interpretation, not evidence about which arm won.

## What is isolated

At the first post-checkpoint update, all three actions use the same step-1500 model, carried AdamW moments, raw gradient, and already-updated legacy estimator. Let a thin numerical SVD of the retained basis be $V=Q\Sigma W^T$. Then

$$
r_{\mathrm{native}}=VV^Tg=Q\Sigma^2Q^Tg,\qquad
r_{\mathrm{orth}}=QQ^Tg,\qquad
r_{\mathrm{norm}}=c\,r_{\mathrm{orth}}.
$$

Thus native and orthogonal use the same numerical column space when no singular direction is discarded, but native applies singular-value-squared gains within it. They are generally not collinear. The scalar $c$ can match their incoming norms, but cannot reproduce native's relative coefficients unless all gradient-active singular values are equal. This is the clean distinction between **within-span conditioning** and **span membership**.

There is also an exact local raw-gradient comparison. Assume full retained numerical rank, exact real arithmetic, and $q=QQ^Tg\neq0$. Write $z=Q^Tg$, $\lambda_i=\sigma_i^2$, and $w_i=z_i^2/\|z\|^2$. Then

$$
\cos(r_{\mathrm{native}},q)
=\frac{\mathbb E_w[\lambda]}{\sqrt{\mathbb E_w[\lambda^2]}}
=\frac{1}{\sqrt{1+\operatorname{CV}_w(\lambda)^2}},
\qquad
c=\sqrt{\mathbb E_w[\lambda^2]}.
$$

Consequently,

$$
g^T(cq)=\|q\|^2\sqrt{\mathbb E_w[\lambda^2]}
\geq \|q\|^2\mathbb E_w[\lambda]
=g^Tr_{\mathrm{native}}.
$$

A two-coordinate quadratic shows why the first-order result is not a finite-step impossibility theorem. Take $g=(1,1)$, Hessian $\operatorname{diag}(100,1)$, $VV^T=\operatorname{diag}(.01,1)$, full-span $QQ^T=I$, and $\eta=.04$. Native uses $(.01,1)$ and changes the quadratic loss by
$-.04(1.01)+\tfrac12(.04)^2(1.01)=-.039592$. Its equal-norm orthogonal counterpart is $\sqrt{1.0001/2}(1,1)$ and changes loss by approximately $-.01616733$. Both descend, but native wins because it suppresses the high-curvature coordinate despite its smaller first-order dot product. This only proves that curvature shaping is a coherent route; it neither models Adam nor shows that learned legacy gains estimate inverse curvature.

The common first update identifies the conditional effect of changing this action map at that saved state. It does not identify how the legacy trajectory formed the state, whether the span was sufficient from initialization, or a mediation pathway from optimizer to representation. After that update, parameters and Adam moments differ; later raw gradients, bases, scales, and decay terms differ too. The endpoint contrast is therefore a policy effect from step 1500 onward, not repeated application of one common projector.

## Exact one-coordinate AdamW response

PyTorch AdamW first applies decoupled parameter decay, then updates exponential first and second moments, bias-corrects them, and applies the adaptive step; weight decay does not enter those moments ([PyTorch 2.11 AdamW documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html)). Assume $0<\beta_1,\beta_2<1$ and $\epsilon>0$ except where epsilon-zero limits are stated explicitly. For coordinate $i$, let $m_i,v_i$ be the uncorrected carried moments before update $t$, and consider a current delivered coordinate $c q_i$ with $c\geq0$. Define

$$
\begin{aligned}
a&=\beta_1m_i, & b&=(1-\beta_1)q_i,\\
d&=\beta_2v_i, & e&=(1-\beta_2)q_i^2,\\
B_1&=1-\beta_1^t, & B_2&=1-\beta_2^t.
\end{aligned}
$$

The bias-corrected adaptive direction before multiplying by the learning rate is exactly

$$
F_i(c)=
\frac{a+bc}{B_1\left[\sqrt{(d+ec^2)/B_2}+\epsilon\right]}
=
\frac{\sqrt{B_2}}{B_1}
\frac{a+bc}{H(c)+\delta},
$$

where $H(c)=\sqrt{d+ec^2}$ and $\delta=\epsilon\sqrt{B_2}$. The total first-step displacement is

$$
\Delta\theta_i(c)=-\eta\lambda\theta_i-\eta F_i(c).
$$

The decay term is common across arms only at this first update because $\theta$ is then common. Equal incoming action norms do not imply equal $F$, adaptive-displacement norm, total displacement, or adaptive-to-decay balance.

For $H(c)>0$, differentiation gives

$$
F_i'(c)=
\frac{\sqrt{B_2}}{B_1}
\frac{b[d+\delta H(c)]-aec}
{H(c)[H(c)+\delta]^2}.
$$

This is an exact sign test. To express it without dependence on the sign of $q_i$, take $s=\operatorname{sign}(q_i)$, $A=sa=\beta_1m_i s$, and $C=s b=(1-\beta_1)|q_i|$. The derivative of the response aligned with $q_i$ has the sign of

$$
G(c)=C[d+\delta H(c)]-Aec.
$$

When $q_i\neq0$ and $d>0$:

- if $A\leq C\delta/\sqrt e$, the aligned response is increasing for every finite $c\geq0$;
- if $A>C\delta/\sqrt e$, it increases and then decreases, with one unique maximum at the positive solution of $Aec=C[d+\delta H(c)]$.

The threshold simplifies to

$$
\beta_1m_i\operatorname{sign}(q_i)>
(1-\beta_1)\epsilon\sqrt{\frac{B_2}{1-\beta_2}}.
$$

With $\epsilon=0$ and a positive denominator, the turning point is $c^*=Cd/(Ae)$. Hence even a positive scalar increase can reduce a coordinate's adaptive response after the maximum: carried aligned first moment competes with the new contribution to the second moment. This is distinct from native's cross-coordinate anisotropy.

If $d=0$, the sign for $c>0$ is constant rather than turning: increasing, flat, or decreasing according as $A$ is below, equal to, or above $C\delta/\sqrt e$. With moments initialized at zero and $0<\beta_2<1$, an exact attainable Adam history with $v_i=0$ has only zero past gradients in that coordinate and therefore $m_i=0$ as well.

Zero and limiting cases are informative:

- If $q_i=0$, $F_i(c)$ is independent of $c$ but can remain nonzero because carried $m_i,v_i$ still move the parameter.
- If $m_i=v_i=0$ and $q_i\neq0$, the aligned magnitude rises from zero only through the epsilon effect and approaches a scale-independent limit. For $\epsilon=0$ it is constant for $c>0$, while $c=0$ is undefined.
- As $c\to\infty$,
  $F_i(c)\to \frac{\sqrt{B_2}}{B_1}\frac{1-\beta_1}{\sqrt{1-\beta_2}}\operatorname{sign}(q_i)$.
- At $c=0$ with $v_i>0$, the carried response remains
  $\frac{\sqrt{B_2}}{B_1}\frac{\beta_1m_i}{\sqrt{\beta_2v_i}+\delta}$.

The original Adam paper describes invariance to diagonal gradient rescaling ([Kingma and Ba, 2017 version](https://arxiv.org/abs/1412.6980)). The relevant exact special case is a constant positive scaling of the **entire** gradient history from zero moments: $\widehat m$ scales by $c$ and $\widehat v$ by $c^2$, giving

$$
\frac{c\widehat m}{c\sqrt{\widehat v}+\epsilon}
=\frac{\widehat m}{\sqrt{\widehat v}+\epsilon/c}.
$$

It is exactly scale-invariant when epsilon is zero and the denominator is positive, and approximately so when epsilon is negligible. Scaling one new action while retaining unscaled $m,v$ is the different function $F_i(c)$ above.

## Honest readings of endpoint patterns

| Prospective pattern | Supported hypothesis, conditional on diagnostics | Not established |
|---|---|---|
| Native better than both orthogonal arms | Legacy within-span gains or their interaction with carried Adam/history matter after step 1500. | Semantic usefulness of those gains; span formation from scratch. |
| Norm-matched approaches native while unmatched orthogonal does not | A global incoming-norm/Adam interaction is plausible, especially if measured post-Adam displacements also approach native. | Exact dose matching or scalar mediation. |
| Both orthogonal arms approach native | The two post-fork policies are compatible with native endpoints conditional on the inherited legacy starting state. | A frozen shared future span, or that span selection caused the original advantage or is sufficient from initialization. |
| Unmatched orthogonal approaches native but norm-matched does not | The imposed scalar changes Adam/decay balance; native incoming norm is not itself a sufficient dose explanation. | That smaller or larger gradients are generically preferable. |
| Both orthogonal arms outperform native | Native within-span anisotropy is harmful over this post-fork interval. | Negation of the earlier observed learning effect. |
| All three are similar | No material post-step-1500 action-policy dependence is detected at the available precision/horizon. | Equivalence, or irrelevance before step 1500. |

Mixed seed signs, metric disagreement, or a gap between incoming norms and actual displacement call for a mixed explanation rather than selecting whichever mechanism fits one endpoint. The strongest steelman is that $\Sigma^2$ conditions the chosen span toward rule-building or cleanup directions. The competing account is purely numerical: legacy nonorthogonality changes coordinatewise Adam transients without semantic selection. The recorded common-$Q$ coefficients, $m/v$, adaptive direction, decay movement, and total displacement check necessary operator and delivery distinctions locally; they cannot by themselves distinguish semantic rule-building from a numerical route.

Finally, the native endpoint is archived rather than contemporaneously replayed, so later differences close to known CUDA trajectory sensitivity need qualification. Numerical rank loss also changes “same span” to “retained numerical span.” Incoming norm matching can still leave post-Adam displacement and later decay/dose imbalanced. None of these qualifications invalidates the one-step intervention; they bound what its short trajectories can establish.
