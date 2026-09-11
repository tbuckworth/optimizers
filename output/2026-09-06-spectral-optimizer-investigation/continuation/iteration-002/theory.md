# Prospective theory: projection, diagonal scaling and descent

These deductions precede numerical execution. Let g be the gradient of a
smooth scalar loss at the current parameters, P an orthogonal projector, and
h=Pg. Ordinary hard projection preserves a weak descent direction:

\[
g^T(-\eta Pg)=-\eta\|Pg\|^2\leq0.
\]

This is a directional statement; finite step loss reduction additionally
depends on curvature and step size. For `L(theta)=||theta||²/2`, the exact
change under projected SGD is
`(-eta+eta²/2)||Pg||²`, strictly negative for `0<eta<2` when Pg is nonzero.

For the first Adam step with zero initial moments and conventional bias
correction, `m_hat=h` and `v_hat=h²`, independently of the chosen betas.
With positive epsilon, the update is exactly

\[
\delta=-\eta D_hh,\qquad
D_h=\operatorname{diag}[(|h_i|+\epsilon)^{-1}]\succ0.
\]

Its directional change for the *original* gradient is
`g^T delta=-eta*g^T D_h P g`. Positive definiteness of D_h and P does not
guarantee that this product quadratic form is nonnegative: D_h P is generally
nonsymmetric and can rotate the step outside range(P).

For a fixed symmetric positive-definite D and orthogonal P, the condition
`g^T D P g>=0` for **every** g is equivalent to `DP=PD`. To see this, decompose
the space into range(P) and null(P). The symmetric part of DP has blocks

\[
\frac{DP+PD}{2}=
\begin{pmatrix}D_{11}&D_{12}/2\\D_{12}^T/2&0\end{pmatrix}.
\]

If D12 is nonzero, choose v in range(P), w in null(P) with `w^T D v!=0`.
Then `g=v+t*w` gives `Pg=v` and
`g^T DPg=v^T Dv+t*w^T Dv`, which is negative for sufficiently large t of the
opposite sign. Conversely, D12=0 means D commutes with P and leaves range(P)
invariant, so `g^T DPg=(Pg)^T D(Pg)>=0`.

Adam's D depends on h, so that fixed-D theorem is not automatically a global
characterization of its nonlinear update map. But along `g=v+t*w`, h=v and
therefore D_h remain fixed. Whenever `(I-P)D_v v!=0`, choosing
`w=(I-P)D_v v` and sufficiently negative t yields first-step ascent for the
original gradient. This supplies an existence mechanism, not its prevalence
on learned neural-network trajectories.

For the fixed candidate `g=(-2,1.1)` and `u=(1,2)/sqrt(5)`,
`P=[[1,2],[2,4]]/5` and `h=(.04,.08)`. Hence `g^T h=.008`. First Adam gives

\[
\delta=-.1\left(\frac{.04}{.04+10^{-8}},
                       \frac{.08}{.08+10^{-8}}\right).
\]

The original-gradient directional change is positive exactly when
`1/(.04+epsilon)>1.1/(.08+epsilon)`, or `epsilon<.36`; epsilon=1e-8 satisfies
this by a wide margin. At this fixed theta, positive directional change also
implies an increase in the chosen quadratic for every positive learning rate,
because its second-order term `||delta||²/2` is positive. With eta=.1 the
expected change is approximately .10, while projected SGD changes it by
exactly -.00076.

Reprojecting the displacement from filtered Adam is a different map:
`delta_post=-eta*P*D_h*P*g`. It satisfies
`g^T delta_post=-eta*(Pg)^T D_h(Pg)<=0`. This restores first-step directional
descent under these assumptions, but does not ensure finite-step decrease for
arbitrary eta, nor later-step descent with arbitrary historical momentum.
For the fixed candidate the finite-eta loss change is predicted to be about
-.003. Unfiltered first Adam also has
`g^T delta=-eta*sum(g_i²/(abs(g_i)+epsilon))<=0`.

The result, if validated, is only a fixed-basis first-step counterexample to
the claim that hard gradient projection followed by Adam must be a descent
step. It is neither a failure of orthogonal projection itself, a proof that
Adam generally ascends, nor evidence that this occurs often in the trained
spectral optimizer. The latter requires logging real gradients, projectors,
moment state and actual parameter updates in a separate prospective study.
