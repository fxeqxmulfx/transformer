# Scope of the Lean proofs used by the experiment

The local sources, not theorem names alone, determine the implementations.
`lake build Transformer.AMSGradW Transformer.MagnitudeDirection` succeeded
without warnings. The full-tree audit reports159 explicit sorry, zero
proved declarations resting on sorry, and zero extra axioms. The W/MD
theorems below are individually audited with `#print axioms`.

## AMSGradW

`AMSGradW.trainingStep` updates the three original AMSGrad buffers from the
true gradient, then subtracts eta*decay times the old position. There is no
debiasing, projection or guard. Its running maximum is monotone. Bounds on
actual positions and moments are derived under L < decay*epsilon; these
bounds give a finite positive limit for every actual denominator D_i.

`trainingRun_convergence` uses Banach contraction of the frozen map
x_i -> -gradient_i(x)/(decay*D_i). It bounds the joint position/momentum
error in a weighted maximum norm by a factor strictly below one, plus
forcing from the actual changing denominators. That forcing tends to zero,
so actual weights and first moments converge, as does the original loss.
The conclusion is gradient_i(star)+decay*D_i*star_i=0, rather than a zero
unregularized gradient. With convexity, `trainingRun_minimum_convergence`
proves a unique minimum of f(x)+decay/2*sum_i D_i*x_i^2.

Assumptions: differentiable fixed objective; coordinate maximum-norm
Lipschitz gradient, L>=0; eta,epsilon,decay>0; 0<=beta1<1;
0<=beta2<=1; eta*decay<=1; L<decay*epsilon. No bounded-trajectory,
alignment, existing minimum, or assumed metric-convergence premise is hidden.
`original_stationarity_counterexample` proves a nonzero limiting original
gradient on the shifted quadratic. The CPU contract reproduces this case.

## AMSGradMD

`amsgradMDProposal` is a concrete variant of arXiv:2606.25971v2, Appendix A,
Algorithm2: recover D=W/(softplus(row)*softplus(col)); calculate all factor
gradients using old gains and old D; update distinct first/second/maximum
histories for D, raw rows and raw columns; project D to its fixed sphere;
fuse the new softplus gains with projected D. Every callback uses AMSGrad,
where the manuscript's experiments use Adam for gains.

`originalAMSGradMD_training_counterexample` retains every buffer, projection
and raw gain. Starting with W=1, epsilon1, beta1=beta2=0, and both rates1/4
on (W+1)^2/2, the unguarded weights stay positive and the true gradient
norm stays at least one. Thus ordinary AMSGrad convergence cannot simply
be inherited through MD reparameterization.

`safeguardedAMSGradMD_convergence` proves convergence for a changed algorithm:
check the complete post-projection/post-gain displacement, keep a certified
proposal, otherwise use a true-gradient step (halve if it would hit zero).
Keep all proposed optimizer memories, and restore sphere storage by common
row rescaling without changing fused weights. The guard gives a descent
bound sigma^2/(2L)*||gradient||^2; a lower loss bound makes the squared
gradients summable, hence their norms tend to zero. Monotone bounded loss
also has a finite limit. Strong convexity and a stationary minimizer yield
weight convergence and a geometric gap bound in the minimum theorem.

Assumptions: a fixed differentiable objective with a quadratic smooth upper
model, L>0, 0<sigma<=1/2, a lower loss bound. Positive c and a valid initial
sphere ensure nonzero storage at every finite time. The counterexample's
corrected run tends to W=-1. CPU tests reproduce both outcomes and check
moment evolution after rejection, factor derivatives and storage repair.

## Relation to GPTMini measurements

The benchmark has stochastic minibatches, float32 arithmetic and a
nonconvex objective. Smoothness of the whole GPT loss (especially with
Sparsemax) and W's decay-domination assumption are not proved here.
The chosen Python MD wrapper applies the exact single-matrix proposal to
hidden Linear matrices and ordinary AMSGrad to auxiliary tensors. Its
global product guard is an explicit multi-block numerical extension, tested
on the full displacement; it is not a separate Lean GPT convergence theorem.
See PLAN.md for every parameter choice and routing rule. A good measured
test loss does not establish any of these mathematical hypotheses.
