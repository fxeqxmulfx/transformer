# Geometry and optimizer dynamics of training

Moved unchanged from [`../flat_spectrum.md`](../flat_spectrum.md) on 2026-11-19: the sections
*Observable spectral geometry* through *Query-weighted feature covariance*. They follow the
directions that change class comparisons during training (OG, CG), a modified Forward–Forward
loss (LF), the motion of the main-model AdamW (AD, DP, FF1–FF4), control of the cross-entropy
through momentum, feature metrics and probabilities (CL, FM, PM), and the motion of the full
encoder (EM, PA, QG). The Lean is mostly in `Transformer.Grokking.GradientEvidence`, where
its modules from `FlatSpectrum.FeatureCodes` moved the same day, with modules in
`Transformer.Grokking.Geometry`, `Transformer.Grokking.ForwardForward` and
`Transformer.Rotate90.FeatureMotion`.

These labels are this file's; its FF1–FF4 are not the FF1–FF3 of [`transport.md`](transport.md),
where FT1–FT3 are. The earlier labels are in [`../flat_spectrum.md`](../flat_spectrum.md).

## Observable spectral geometry

Task #198 extends the geometry after the user's request to close both the
compressed-certificate and learning-mechanism gaps. The extension concerns
the directions that change class comparisons. It does not assert that the
full hidden feature spectrum becomes flat. The
[protocol](../../experiments/readout_geometry/PROTOCOL.md), committed before
candidate measurements, fixes both trajectories, all 26 probes, ranks,
execution allowances and the actual AdamW attribution.

### OG1. Remove common readout offsets

For finitely many classes with readout rows `E_c`, define

```
mean(E) = (sum_c E_c) / number_of_classes
C_c = E_c - mean(E).
```

Every pairwise score difference is unchanged:
`<C_c,x>-<C_t,x>=<E_c,x>-<E_t,x>`. A common added readout row also cancels
from every comparison. Linear scoring commutes with the class mean. Thus
the classification geometry should not demand recovery of a shared logit
offset. The finite Lean results also cover empty class sets, for which the
contrast claims are vacuous and the sums are zero.

### OG2. The observable Gram form

For a feature error `e`, define

```
Q = sum_c C_c C_c^T
observable_energy(E,e) = sum_c <C_c,e>^2 = e^T Q e.
```

The equality is an exact finite-sum identity. This nonnegative form may be
singular: zero energy is equivalent to `<E_c,e>=<E_t,e>` for every pair
of classes. Such errors can change every absolute score while preserving
every margin. No inverse, positive smallest eigenvalue, or exact feature
recovery is assumed. This is the natural readout-dependent extension of
the earlier recovery geometry; it does not replace a missing optimizer
argument with a definition.

### OG3. A sharp finite margin budget

For distinct classes, their two centered squared score errors are at most
the total energy. The scalar inequality `(a-b)^2 <= 2(a^2+b^2)` gives

```
|<E_c,e>-<E_t,e>| <= sqrt(2*observable_energy(E,e)).
margin_t,c(x+e) >= margin_t,c(x) - sqrt(2*observable_energy(E,e)).
```

The bound also holds when the classes coincide. If every old target margin
exceeds this budget, the target remains the strict winner. These are
conditional theorems: a prediction must obtain its error budget before
observing the future error. The native experiment changes basis selection
using known old training states and readouts; it retains the existing
signed transport residual and orthogonality-defect bounds. It does not
insert a measured future observable energy into a forecast.

The readout-aware constructor first projects all centered readout rows into
the available centered-training-state span, then takes the leading right
singular vectors. It uses the training pseudoinverse only to construct
coefficients; the actual reconstruction residual and basis defects remain
in the certificate. Thus small training singular values, rank truncation
and imperfect arithmetic cannot be silently treated as exact recovery.
Rank 32 has the same budget as the hidden-variance baseline. The existing
signed-transport theorem accepts either computed basis; optimality of a
floating-point SVD is not a premise of correctness.

### OG4. Nonzero and sharp witnesses

Take two readout rows `(-1,3),(1,3)` and error `e=(0,7)`. Both scores shift
by 21, although observable energy is zero and the hidden error is nonzero.
Every margin is preserved. For rows `-1,1` in one dimension and `e=1`,
the squared contrast error is 4 and observable energy is 2, attaining the
factor two in OG3. Lean proves both witnesses and instantiates the strict
winner theorem with a nonzero invisible error and positive old margins.

### OG5. Attribute the actual adaptive update

The fixed source has weight decay one. Write `g_R` for the full-training
partial readout derivative, `g_F` for the full tied-embedding derivative,
`g_B` for the actual minibatch derivative, and `m_hat` for the updated
bias-corrected first moment. Keep each coordinate's actual diagonal
preconditioner `D`. The known rounded embedding increment obeys

```
Delta E = -eta E - eta D g_R - eta D(g_F-g_R)
          - eta D(g_B-g_F) - eta D(m_hat-g_B) + rounding.
```

The rounding term is defined by the difference between the executed FP32
increment and this real-valued decomposition, and is explicitly reported.
It includes any discrepancy in the reconstructed adaptive scaling. The
seventh parameter component contains the actual projection-matrix changes.
The instrumented update must match ordinary source `train_step` byte for
byte. Full-training partial readout derivatives receive an independent
analytic check, and the complete tied derivative receives source finite
difference checks.

Propagate each parameter component through the old source Jacobian. If
`Delta w=sum_i dw_i` and `v=sum_i v_i`, finite algebra gives

```
Delta forecast_score
  = sum_i (<dw_i,x> + <w,v_i>) + <Delta w,v>.
new_score = old_score + Delta forecast_score + <w',x'-(x+v)>.
```

All signed terms, the mixed term and the nonlinear remainder are retained.
Lean proves these identities for arbitrary finite decompositions, with
explicit satisfying examples. It also gives a witness where one positive
component coexists with a negative total update. In the native study the
parameter, tangent and logit sums must close, and the independent verifier
reconstructs the total predicted change from frozen features and independent
old logits. Attribution fixes the old worst competitor; correctness bounds
still cover every competitor.

These are contributions to one actual update, not seven alternative trained
trajectories or a proof of long-run causal necessity. Readout-only CE credit
cannot stand in for adaptive tied-weight dynamics. The
[findings](../../experiments/readout_geometry/README.md) distinguish reduced
certificate slack, measured update contributions and the remaining discovery
and convergence question.

| Claim | Lean declaration |
| --- | --- |
| Class-centering and common shifts | `GradientEvidence.centeredReadout_contrast`, `common_readout_shift_preserves_contrast` |
| Observable quadratic form and nonnegativity | `GradientEvidence.observableEnergy_eq_gram`, `observableEnergy_nonneg` |
| Exact invisible-error characterization | `GradientEvidence.observableEnergy_eq_zero_iff_centered`, `observableEnergy_eq_zero_iff_contrasts` |
| Sharp contrast and finite margin bounds | `GradientEvidence.observable_contrast_sq_le`, `observable_contrast_abs_le`, `observable_margin_lower` |
| Invisible errors and strict winners | `GradientEvidence.invisible_error_preserves_contrast`, `strict_winner_of_observable_error` |
| Nonzero and sharp witnesses | `GradientEvidence.nonzero_invisible_error_witness`, `observable_contrast_sharp_witness` |
| Signed component and remainder identities | `GradientEvidence.forecast_component_identity`, `finite_update_component_identity` |
| Positive component, negative total | `GradientEvidence.positive_component_negative_update_witness` |

## Class-contrast discovery and persistence

Task #199 follows the user's request to pursue the remaining learning
question. The [fixed protocol](../../experiments/class_geometry/PROTOCOL.md)
replays two original trajectories and six replication seeds for all 10000
updates. The [findings](../../experiments/class_geometry/README.md) distinguish
a proved supervised damping mechanism, exact attribution of realized
trajectories, and the still missing convergence and alignment argument.
These claims extend the local CE derivative and observable geometry above;
they are authored derivations, not claims attributed to another manuscript.

### CG1. A supervised dissipative direction

Let `T_i` be the current features, `y_i` the current minibatch targets, and
`z_ic=<E_c,T_i>`. The existing derivative theorem `readoutGradient_eq`
identifies the actual partial derivative of mean cross-entropy:

```
g_R,c = mean_i ((p_ic - 1[c=y_i]) T_i),
p_ic = exp(z_ic) / sum_v exp(z_iv).
```

For two classes `c,d` absent from the minibatch targets, their label terms
vanish. Expanding the finite sums gives the exact identity

```
<E_c-E_d, g_R,c-g_R,d>
  = mean_i ((z_ic-z_id)(p_ic-p_id)) >= 0.
```

The sign follows from monotonicity of the exponential and a positive
softmax denominator. Features are fixed only for this partial derivative;
the theorem can be applied anew after each feature or minibatch change.
For any nonempty group U containing no targets, the pairwise identity

```
sum_u <E_u-mean_U(E), g_R,u-mean_U(g_R)>
  = (1/(2*|U|)) sum_u,v <E_u-E_v, g_R,u-g_R,v> >= 0
```

turns pair damping into damping of the whole centered unused group. In
particular, for nonnegative step scales `eta_t,alpha_t`, its cumulative
isotropic radial contribution `sum_t -2*eta_t*alpha_t*<Q_U E_t,Q_U g_R,t>`
is nonpositive for every finite prefix, even with changing features and
minibatches. This is a theorem about one signed contribution. It does not
identify the complete tied, adaptive optimizer with isotropic readout descent.

### CG2. Keep the finite step

For any finite vectors `w,g` and real `eta`,

```
||w-eta*g||^2 = ||w||^2 - 2*eta*<w,g> + eta^2*||g||^2.
```

Thus `eta>=0` and the current-state condition
`eta*||g||^2 <= 2*<w,g>` imply nonincrease. A dissipative derivative alone
does not imply contraction for every learning rate. Neither this condition
nor isotropic dynamics is assumed for the complete source AdamW step.
The native audit checks its separate CG1 contribution on every update and
keeps the complete finite quadratic term in CG4. A nonzero two-coordinate
witness satisfies the finite-step condition with positive learning rate.

### CG3. Why 32 class directions appear

Split the readout rows into nonempty groups A and U. With the overall mean
`mu`, and their computed group means `mu_A,mu_U`, finite centering gives

```
||E-centered||_F^2
 = sum_a ||E_a-mu_A||^2 + sum_u ||E_u-mu_U||^2
   + |A|*||mu_A-mu||^2 + |U|*||mu_U-mu||^2.
```

The three terms are active within-group contrasts, unused within-group
contrasts, and between-group energy. Cross terms vanish because each
within-group centered sum is zero. The Lean scalar identity holds for any
chosen `mu`; summing it over coordinates and choosing the overall mean
gives this Frobenius identity.

For `n+1` rows, centering also has an explicit reconstruction:

```
E_c-mean(E)
 = sum_(i=1..n) (1[c=i]-1/(n+1)) (E_i-E_0).
```

Hence at most `n` generators are needed; linear independence is not assumed.
In this experiment the training labels have 33 distinct answers and 64
unused classes. Active contrasts therefore need at most 32 generators;
the two group means add at most one more direction, since their weighted
centered sum is zero. This dimensional fact holds at initialization too.
Learned concentration in these directions is a separate assertion, and a
rank-33 SVD need not select exactly the between-group direction.

### CG4. Attribute the actual trajectory

Here `g_R` is the partial derivative on the **actual minibatch** and `g_B`
its full tied-embedding derivative. This differs deliberately from OG5's
full-training/sampling split. With actual updated moment `m_hat`, actual
coordinatewise preconditioner `D`, and `alpha=min_coordinate(D)>0`, write

```
Delta E = -eta E - eta alpha g_R - eta(D-alpha I)g_R
          -eta D(g_B-g_R) -eta D(m_hat-g_B) + rounding.
```

The six terms are decay, isotropic readout, adaptive readout correction,
tied-encoder credit, moment memory, and the measured FP32 remainder. The
remainder is bounded and reported; no exact floating-point implementation
theorem is claimed. Class centering is applied after each coordinatewise
increment: `D` need not commute with the class projections.

For any finite component family `d_i`, the real identity is

```
||x+sum_i d_i||^2-||x||^2
  = sum_i 2<x,d_i> + ||sum_i d_i||^2.
```

The last square includes all cross-component products. If every realized
step satisfies `x_(t+1)=x_t+sum_i d_(t,i)`, summing these contributions
telescopes to `||x_n||^2-||x_0||^2` for every prefix n. Apply this separately
to each fixed class projection. Lean proves both identities with a nonzero
trajectory example; the experiment checks parameter closure, all projected
energy identities, and every cumulative prefix without dropping terms.

### CG5. Relative concentration and persistence

Absolute active energy is insufficient to explain its share. Let `S` be
active energy and `V` unused plus between-group energy. For nonzero old and
new total energies, algebra gives

```
(S+ds)/(S+V+ds+dv) - S/(S+V)
  = (V*ds-S*dv) / ((S+V)*(S+V+ds+dv)).
```

Substitute all CG4 radial terms and the total quadratic term into `ds,dv`.
This yields an exact signed attribution of share change, which can be
summed over every prefix. With positive denominators, `S*dv <= V*ds` is
the explicit improvement condition. The native trajectory does not satisfy
monotone improvement at every step; these conditions are not asserted to
follow from training loss or the local dissipative sign.

Common nonzero scaling preserves `S/(S+V)`. Pure weight decay scales both
energies by the same square and cannot by itself select active contrasts.
Its radial share numerator is zero. Mixed-step cross products still remain
in the total quadratic term, so decay is not removed from the full optimizer.
Signed attribution explains one realized trajectory, not the outcome of
retraining after deleting an optimizer component.

### CG6. Concentration still needs alignment

There is a simple obstruction to identifying class concentration with
successful prediction. In one feature dimension take active readout rows
`+1,-1`, unused row `0`, and query feature `-1` with target the `+1` row.
All centered energy is in the active contrast space, yet the other active
class wins strictly. Lean proves the two energies equal 2 and the strict
wrong-class comparison. Thus concentration needs a separate query alignment
and positive-margin argument such as OG3 and signed transport.

The fixed study finds a reproducible increase in active-class share and
useful sampled late geometry, with signed opposition from tied-encoder and
memory terms. It also retains fluctuations, persistent residual readout
energy, and failures to sustain 99% coverage. The remaining theorem must
control coupled feature learning, adaptive scales and moments strongly
enough to obtain query alignment and margins over time. CG1–CG5 alone do
not prove that theorem or universal optimizer discovery.

| Claim | Lean declarations |
| --- | --- |
| Actual CE pair identity and sign | `GradientEvidence.unused_pair_gradient_identity`, `unused_pair_gradient_damping`, `logit_probability_difference_nonneg` |
| Whole-group and cumulative damping | `GradientEvidence.unused_group_gradient_damping`, `cumulative_unused_readout_credit_nonpos` |
| Finite isotropic identity and condition | `GradientEvidence.isotropic_pair_energy_identity`, `isotropic_pair_energy_nonincrease` |
| Computed class means and energy split | `GradientEvidence.class_center_sum_zero`, `class_translation_energy`, `class_split_energy` |
| Pairwise-to-centered product identities | `GradientEvidence.zero_sum_pair_product`, `centered_pair_product` |
| Explicit centered-row generators | `GradientEvidence.centered_rows_from_anchor` |
| Finite and cumulative component identity | `GradientEvidence.component_energy_identity`, `trajectory_component_energy_identity` |
| Relative share identity and improvement | `GradientEvidence.class_share_change`, `class_share_component_identity`, `class_share_nondecrease` |
| Common scaling preserves share | `GradientEvidence.common_scaling_preserves_class_share` |
| Full concentration with a wrong prediction | `GradientEvidence.active_concentration_insufficient_witness` |

## Forward–Forward reformulation

User direction, 2026-11-08: reformulate Forward–Forward and modify its loss.
The reference is Geoffrey Hinton, *The Forward-Forward Algorithm: Some
Preliminary Investigations*, arXiv:2212.13345, §§2–3.3, read locally in
`papers/arXiv-2212.13345/paper.txt`. The following is an explicitly changed
candidate, not a claim made by that paper or a new convergence guarantee.
The [fixed native diagnostic](../../experiments/forward_forward/PROTOCOL.md)
precedes any full training comparison.

### LF1. Factor the label out of the encoder

Compute a label-independent local feature `h=f(x)` once. For class vector
`e_c`, define corrected distance goodness

`G(h,e_c) = (||e_c||² - ||h-e_c||²)/2 = Re<h,e_c> - ||h||²/2`.

The class-norm correction matters: uncorrected nearest-prototype distance
would add a class-dependent bias and change the source classifier. The
common feature-norm term cancels in every positive/negative difference:
`G(h,e_y)-G(h,e_c) = Re<h,e_y-e_c>`. Thus the existing real dot-product
readout, or the real part of a Hermitian product in complex coordinates,
computes all goodness comparisons. Inference remains the same classifier.
Complex multiplication inside the encoder is unchanged.

Hinton's supervised label-conditioned encoder can require one encoder
evaluation per candidate label (§3.3); this factorization uses one encoder
evaluation and a class readout. It gives up label-conditioned feature
extraction. It does not eliminate readout work or prove an overall speedup.
Putting these goodness scores into a softmax recovers ordinary CE exactly;
that relabeling alone would not change the learning algorithm.

### LF2. Do not use normalized energy as the only goodness

Hinton measures squared activities **before** normalization (§2). In the
actual source model, a post-RMS squared norm is
`W*q/(q+W*epsilon)`, where `q=||x||²`. It depends only on radius and saturates
near W. At zero epsilon it is exactly W for every nonzero input. Its
derivative with respect to q is `W²*epsilon/(q+W*epsilon)²`.
Equal-energy features can have opposite label preferences. LF1 retains
that directional information and avoids treating post-normalization
magnitude as a rich classification signal. No change to RMS is required.

### LF3. Change the local loss to paired directional margins

For local scores `s_c=Re<h,e_c>`, let

`L_pair = sum_{c != y} softplus(mu+s_c-s_y)/(C-1)`.

The fixed candidate uses mu=1 and all C-1=96 negative labels. No label
requires a second encoder evaluation. Unlike Hinton's separate positive
and negative binary objectives, this loss directly compares each negative
with its positive partner. Its logit derivatives are

`q_c = sigmoid(mu+s_c-s_y)/(C-1)` for c != y,
`q_y = -sum_{c != y} q_c`.

The local linear-head derivatives are `dL/dh=sum_c q_c e_c` and
`dL/de_c=q_c h`. Complex derivatives mean the corresponding real-coordinate
derivatives, with the conjugation in the Hermitian product retained. At
mu=0 and two classes this is CE. With multiple competing classes its
gradient need not be a scalar multiple of the CE gradient. The distinct
competitor weights are a real change to the objective.

The **sum** of pair penalties below softplus(mu) guarantees positive margin
against every competitor. The native **mean** therefore needs a threshold
softplus(mu)/(C-1). A small average penalty or a 90% certificate coverage
must not be presented as 100% model accuracy. This is a per-query labeled
loss implication; it gives no held-out generalization guarantee.

### LF4. Specify local credit, not just a different loss

At a boundary, cache the incoming feature u and optimize the next block's
local loss with u fixed. Local Jacobians, outer products and optimizer
updates still require computation. No derivative from an upper block may
be silently propagated into an earlier block and called Forward–Forward.
For the two-layer scalar example `u=a*x`, `v=b*u`, the local first-layer
loss softplus(-a) has derivative -1/2 at a=0, x=1. The terminal loss
softplus(-2*a) has derivative -1 there. The two update rules are different.

The source E is shared by the input embedding and output readout. The
local readout derivative is only one contribution to this shared parameter;
changing E also changes the next encoder evaluation. The native one-step
probe therefore reports both frozen-feature and actual tied-network loss
changes. It is not a complete local-learning trainer. Before task #190,
specify losses/prototypes for every trainable block, detachment boundaries,
and accumulation of all local contributions before one shared-parameter
update. Count their storage and computation, and compare training from
scratch at matched validation quality. A loss reformulation alone supplies
neither a faster optimizer nor a solution to the flat-spectrum obstruction.

| Claim | Lean declarations in `ForwardForward` |
| --- | --- |
| Corrected distance, class order and CE identity | `goodness_eq_inner`, `goodness_difference`, `goodness_order`, `goodness_crossEntropy` |
| Normalized energy and its sensitivity | `normalized_energy`, `normalized_energy_zero_eps`, `normalized_energy_derivative`, `equal_energy_opposite_preferences` |
| Actual pairwise derivative and invariances | `softplus_derivative`, `pairLoss_directional`, `pairLoss_shift`, `pairLoss_goodness` |
| Per-query strict correctness threshold | `pair_term_le`, `winner_of_pairLoss_lt` |
| Local versus terminal derivatives | `local_linear_derivative`, `terminal_first_derivative`, `local_global_derivative_witness` |

### LF5. Complete local training exposes the missing link

The [frozen task #190 pilot](../../experiments/forward_training/README.md)
specifies local objectives for all blocks and sums both direct contributions
to the tied E before one original AdamW update. Its controls separately
change terminal loss, add fixed auxiliary heads and stop cross-block credit.
All 224 directional checks pass, but local training ends at 2.23–10.50%
validation accuracy across eight cases, compared with 99.28–99.87% for CE.
Full auxiliary backpropagation is also worse than CE. This is empirical
evidence against that particular objective/boundary choice, not a
counterexample to the local derivative theorems or to all Forward–Forward
methods. Complex multiplication is unchanged.

A useful extension must connect a local parameter displacement to terminal
task improvement, with all tied-parameter contributions included. Local loss
descent alone supplies no such implication. The next diagnostic is their
directional alignment under the actual optimizer, before selecting new local
heads or boundaries. Any full-gradient measurements used for diagnosis must
remain separate from the information available to a proposed local trainer.
The pilot found no efficiency gain, and simultaneous foreign GPU training
also prevents a clean-device speed comparison from its retained timings.

### LF6. Local gradient, optimizer displacement and finite step differ

The [task #193 diagnostic](../../experiments/forward_alignment/README.md)
measures all three at 40 saved states. Complex directional pairings use
the signed pullback to independent real coordinates. Validation and
full-training gradients are charged diagnostic information, never free
signals for a proposed local trainer.

At all eight failed local endpoints, the hidden block A's current gradient
aligns with terminal validation CE, but its actual AdamW displacement is
an ascent direction in five. Positive finite-change remainders turn two
other first-order descent predictions into actual increases. A lowers its
own frozen local loss in all eight probes while raising full-training CE
in all eight and validation CE in seven. B-only updates decrease full-training
and validation CE in every local endpoint, while validation pair loss falls
in five of eight. The vocabulary-E removal worsens complete
validation CE in all eight, so the present probes do not support removing
that term as the immediate repair.

These numerical observations distinguish three obligations for a useful
local method: task alignment of the available gradient, preservation of
useful direction by the optimizer, and control of finite-step error. They
do not prove a trajectory-level cause or a generalization guarantee. The
next diagnostic should isolate A's direction and scale before changing
objective and optimizer together. No new training outcome or speed claim
follows from these backward-assisted measurements.

### LF7. Smaller steps do not remove a direction defect

The [task #194 follow-up](../../experiments/forward_step/README.md) compares the
actual A displacement with its norm-matched negative current local gradient
at five fixed scales, retaining all 40 states and 400 probes. At the eight
local endpoints, every gradient-direction probe reduces full-training and
validation CE. AdamW improves validation CE in one case at full scale and
three at scales 1/64 and 1/256: shrinking the step recovers the two overturned
descent predictions but retains five ascent cases. This separates observed
finite-step overshoot from an observed direction defect without assigning
that defect solely to moment carryover, adaptive scaling or decay.

All 400 probes reduce their local loss, yet the other state origins still
have mixed terminal outcomes. Accuracy remains mixed even where CE falls.
These are finite numerical observations, not a theorem about an entire
training trajectory. They support testing a current-gradient A update from
scratch while holding the local objective and other block rules fixed.
Its step-size policy must be specified independently: borrowing AdamW's
radius in this diagnostic establishes neither a standalone optimizer nor
an efficiency advantage. Keep validation derivatives out of the proposed
trainer and use a separately frozen confirmation cohort. No complex-product
definition or formal theorem changes in this study.

### LF8. A useful one-step direction need not yield useful local training

The [task #195 comparison](../../experiments/forward_normalized/README.md)
replaces only A's optimizer with its normalized current local gradient,
using a fixed 1%-of-A-norm radius and ten-update warmup. It leaves local
objectives, E/B AdamW and the classifier unchanged. All 24 fresh-seed runs
complete 4000 updates, yet the candidate reaches only 2.23–11.63% validation
accuracy versus 99.45–99.87% CE. Its endpoint validation CE is worse than
original local-pair in all eight cases; accuracy improves in two and falls
in six. Numerical and source checks pass, so this is a negative learning
result for the specified rule and budget, not a failed coordinate check.

This does not contradict LF7's finite same-state observations: the radius
and subsequent trajectory differ. Neither local descent nor favorable
terminal alignment at selected states proves trajectory-wide alignment,
finite-step control or generalization. A next empirical control should
cross local/full credit with original/normalized A updates while keeping
the auxiliary objective set fixed. This separates a possible step-policy
failure from stopped credit without blaming the head or optimizer from
this study alone. The result supplies no new theorem, accuracy repair,
memory saving or clean-device speed advantage.

## Main-model AdamW motion

Task #203 pursues the remaining discovery/persistence question under the
[fixed protocol](../../experiments/geometry_dynamics/PROTOCOL.md). These are
authored derivations for the main-model AdamW copied from GPT commit
`6bb9d03a`, with beta1=.9, beta2=.95 and decay .1. Exact-real arithmetic and
the numerical FP32 audit remain distinct. The gradient sequence below may
come from the full coupled model, a minibatch, clipping, or any other rule.

### AD1. A moment invariant independent of the gradient direction

For one coordinate, write

```
m+ = (9/10)m + (1/10)g,
v+ = (19/20)v + (1/20)g^2.
```

Expanding the actual recurrences gives the identity

```
(19/14)v+ - (m+)^2
  = (19/20)((19/14)v - m^2) + (14m-9g)^2/1400.
```

Thus `m^2 <= (19/14)v` is preserved for every real gradient g. Zero
initial moments satisfy it, so induction proves it at every time without
assuming an improving loss, fixed features, isotropic scales, or a favorable
gradient sign. In particular, v is nonnegative. This applies separately to
each independent coordinate before restoring the complex parameter ties.

If the actual buffer updates instead add errors `em,ev`, their excess obeys

```
(m+ + em)^2 - (19/14)(v+ + ev)
 <= (19/20)(m^2-(19/14)v)
    + 2|m+||em| + em^2 + (19/14)|ev|.
```

The square in the first identity is dropped only with its proved sign.
The errors are explicit; this is not a proof that the FP32 recurrence
satisfies the unperturbed cone. The native audit checks every saved moment
state using the protocol's fixed numerical allowance.

### AD2. Bias-corrected adaptive directions are bounded

For a nonnegative variance, `m^2 <= K^2 v`, `K>=0`, positive bias factors
`b1,b2`, and positive epsilon, elementary square-root algebra yields

```
| (m/b1) / (sqrt(v)/b2 + eps) | <= K*b2/b1.
```

This includes v=0. At zero-based update t, AD1 supplies the explicit bound

```
B_t = sqrt(19/14) * sqrt(1-(19/20)^(t+1)) / (1-(9/10)^(t+1)).
```

This is a bound on the actual buffered adaptive direction, including its
two different bias corrections. It does not replace momentum by a current
gradient. At late times B_t approaches `sqrt(19/14)`, about 1.165.

### AD3. Finite-window motion with continuing updates

For a coordinate update `p+ = (1-eta*rho)p - eta*u + e`, assume `eta>=0`
and `eta*rho<=1`. If `|p|<=R` and `|u|<=B`, then

```
|p+| <= (1-eta*rho)R + eta*B + |e|.
```

If also `rho>=0`, displacement obeys

```
|p+-p| <= eta*(rho*|p|+B) + |e|.
```

The invariant box `|p|<=R` is preserved by exact steps when `B<=rho*R`.
For a sharper finite interval starting at an arbitrary saved time s, set
`R_0=|p_s|` and `R_(j+1)=(1-eta_(s+j)*rho)R_j+eta_(s+j)*B_(s+j)`.
Induction and the triangle inequality give

```
|p_(s+n)-p_s| <= sum_(j<n) eta_(s+j)*(rho*R_j+B_(s+j)).
```

Lean proves the result for the actual exact-real main-model moment and
parameter recurrences, with arbitrary gradients and any finite window.
The recurrence is explicitly identified with `bufferedAdamUpdate`.
It permits nonzero updates throughout the window. The native FP32 trace
audit retains its recorded coordinate rounding remainder in both radius
and displacement recurrences and checks all recorded readout updates.
It does not prove a worst-case FP32 error bound for all future executions.

### AD4. What motion control must still supply

Let w be a target-minus-rival readout contrast and x a query feature.
For coordinate bounds `|w'_k-w_k|<=r_k` and `|x'_k-x_k|<=s_k`,

```
|<w',x'>-<w,x>| <= sum_k (|w_k|s_k + |x_k|r_k + r_k*s_k).
```

The mixed term remains. A positive old margin lower bound larger than this
budget guarantees a positive new contrast. Apply the statement to every
rival; a contrast's readout budget is the sum of its two row budgets.
This turns useful motion bounds into a maintenance guarantee, including
feature movement explicitly. AD3 alone does not control the nonlinear
feature map tightly enough, nor does it establish the starting margins.

Consequently these theorems add gradient-independent control of exact
AdamW step sizes, but do not prove discovery of the useful geometry.
Small individual steps can accumulate a large displacement. A constant
positive learning rate over arbitrarily many steps does not provide a
small total budget. The study reports whether the explicit bound is useful
on the declared hold interval; a loose bound is not counted as certification.

### AD5. Valid certificate envelopes

The new single basis uses the rows `E_a-mu_U` for training-observed labels a,
where mu_U is the mean of unused rows. Each equals
`(E_a-mu_A)+(mu_A-mu_U)`: the CG3 active contrasts plus the between-group
direction. There are at most 33 generators here. Rank 32 still truncates;
rank 33 is a distinct control. All rival classes and all residual terms
remain in the arbitrary-basis signed-transport bound.

If a,b,c are three valid lower bounds for the same margin M, then
`max(a,max(b,c))<=M`. A positive constituent remains positive in this
envelope, and a positive envelope guarantees `M>0`. Lean proves validity,
retention and soundness. No query outcome is required to take the maximum.
Preparing three bases has a separate cost: this is not one rank-32 basis
or a speed claim. The independent artifact verifier checks the maximum
and every emitted bound directly.

The formal modules are `GradientEvidence.MainAdamMoments`, `MainAdamMotion`
and `MainAdamCertificates`; `scripts/MainAdamAxioms.lean` lists all 21
theorems for an explicit axiom audit. Each theorem with hypotheses has a
nonzero admissible example. The [experiment README](../../experiments/geometry_dynamics/README.md)
records the full numerical comparison and the remaining limitations.

## Observed finite-window persistence

Task #204 follows the failure of AD3's uniform coordinate bound to certify
the observed low-rate hold. The following refinement is conditional on
realized forcing budgets. It does not predict unknown future gradients,
prove their alignment, or explain discovery from random initialization.
The [frozen study](../../experiments/geometry_persistence/PROTOCOL.md) checks
every hold update and separately transports predictions over finite windows.

### DP1. Damped forcing tubes

Let P be a fixed linear projection on the readout matrix, including an
active-class contrast, unused-class contrast, or between-group projection
from CG3. For an actual AdamW update with all rounding in e,

```
E+ = (1-eta*rho) E - eta*u + e,
P E+ - (1-eta*rho) P E = -eta*P u + P e.
```

Thus the forcing norm is at most `eta*||P u|| + ||P e||` for nonnegative
eta. The adaptive direction u includes the saved moments and the complete
tied-parameter gradient. No partial CE term is substituted. Alternatively
the realized residual `F_t = P E_(t+1) - a_t P E_t` can be measured directly;
the native checker includes projection arithmetic defects in its radius.

For any path x in a real normed space with nonnegative scales a_t and
`||x_(t+1)-a_t*x_t|| <= b_t`, define

```
q_0 = 1,     q_(t+1) = a_t*q_t,
R_0 = 0,     R_(t+1) = a_t*R_t + b_t.
```

Induction and the triangle inequality give

```
||x_n-q_n*x_0|| <= R_n,
max(0,q_n*||x_0||-R_n) <= ||x_n|| <= q_n*||x_0||+R_n.
```

The inductive residual is the sum of the next forcing and a_t times the
previous residual; this retains common decay exactly. Reindex a window at
zero to obtain every prefix statement from an arbitrary anchor. This works
with nonzero movement at every step. A uniform balance condition
`b_t <= (1-a_t)*B` further implies `R_n <= (1-q_n)*B`. The algebraic statement
holds for any real B and nonnegative a; the usual bounded-radius reading
also requires B nonnegative and scales at most one. Establishing that balance
for future adaptive forcing is an additional problem, not a consequence of
small individual learning rates.

The native audit can identify forcing from already executed training steps.
Those are observed-path bounds, even though they use no query outcomes.
This differs from AD1–AD3's gradient-independent guarantee available before
the window. The stronger hypothesis buys a potentially much smaller radius.

### DP2. Readout share certificates

For actual nonnegative signal and competing energies S,V, lower and upper
bounds `0<=L<=S`, `V<=U` give

```
L/(L+U) <= S/(S+V)
```

when both denominators are positive. Multiplying by the denominators reduces
the claim to `L*V<=S*U`. Apply this to squared active norms and the sum of
squared unused/between norms, using the lower and upper DP1 radii. CG3's
orthogonal decomposition identifies the denominator with centered readout
energy. When the lower active norm is exhausted the implementation emits
zero; it does not count nonnegative energy as a useful persistence result.

An energy-share guarantee does not establish query classification, a flat
nonzero singular spectrum, or feature alignment. Those limitations persist
even when the radius is small. The native study also checks prediction
certificates with the whole nonlinear feature map included.

### DP3. Transporting certificates across a finite displacement

Let M be a saved margin lower bound for a training-derived target and let
g_c denote the old rival-minus-target gap. Suppose the full future model's
gap g'_c is enclosed above by U_c. If `U_c-g_c<=D` for every rival, then

```
M <= -g_c,   g'_c <= U_c  implies  M-D <= -g'_c.
```

Consequently `D<M` preserves the target prediction against every rival. The
native D is the maximum of all 96 changes `U_c-g_c`, with no omitted unused
class. The starting M comes from task #203's stored envelope; rank 128 is a
separate control. A direct forecast without the old geometry slack is also
reported separately. A claim of retained certification additionally requires
the old bound to have been positive.

The FT1–FT3 forecast equations apply to an arbitrary finite parameter delta,
so the native study uses the difference between two checkpoints. It retains
the old-feature tangent, mixed products, nonlinear remainder, and unchanged
execution allowances. Endpoint parameters are known to the constructor;
endpoint evaluation and labels are not used to construct its intervals.
This certifies an endpoint after many updates. It says nothing about the
intermediate query predictions, which are not recorded at every update.

The modules `GradientEvidence.DampedPersistence` and
`GradientEvidence.PersistenceCertificates` formalize DP1–DP3;
`scripts/PersistenceAxioms.lean` audits their ten theorems. Native numerical
checking remains separate from Lean's real-arithmetic proof.

## Future forcing from checkpoint moments

Task #205 asks what can be guaranteed before later gradients or updates are
read. DP1's tube is useful with observed forcing, but that observation is an
input to its earlier experiment. The following exact-real bounds remove that
input for main-model AdamW. They do not assert reliable discovery, favorable
gradient alignment, or a flat nonzero singular spectrum.

### FF1. A sharp scalar cone update

At a checkpoint let `v>=0`, `r>=0`, and `m^2<=r*v`. For any next real gradient
`g`, retain the copied main-model recurrences

```
m' = .9*m + .1*g,     v' = .95*v + .05*g^2.
Q = 81/95,           C = 19/14.
((Q*r+1/5)*v' - m'^2)*v
  = Q*(r*v-m^2)*v' + (19*v-9*m*g)^2/1900.
```

When `v>0`, nonnegativity of the right side proves
`m'^2<=(Q*r+1/5)*v'`. When `v=0`, the initial cone forces `m=0`, and the same
inequality follows directly for all `g`, including zero. For `v>0` and
`m!=0`, set `r=m^2/v` and `g=19*v/(9*m)`; the square vanishes and the next
ratio attains `Q*r+1/5`. This sharpness concerns one scalar step over
unrestricted real gradients. It does not claim that the tied model's CE
gradient, or a globally clipped sequence, attains every worst-case step.
There is a nonzero clipped-compatible example: `m=.1`, `v=.01`, `g=19/90<1`.

Iteration yields the checkpoint cone

```
K_h(r) = C + (r-C)*Q^h,    m_h^2 <= K_h(r)*v_h.
K_(h+1)(r) = Q*K_h(r)+1/5.
K_h(r)-K_h(s) = (r-s)*Q^h.
```

These statements require no independence, fixed-feature, small-gradient,
loss-decrease, or favorable-direction assumption. They permit arbitrary
adaptation to the past. The ratio is zero when both buffers are zero; positive
variance permits the exact checkpoint ratio. The formula is nonnegative for
every nonnegative initial ratio. Its rapidly vanishing memory describes the
worst-case magnitude envelope, not a theorem that the learned geometry itself
forgets its initialization.

### FF2. Actual resumed moments and absolute-time normalization

`warmMoments` continues the same recurrences from saved buffers.
`main_moments_resume` identifies this continuation with the original history,
and `main_direction_resume` does the same for the normalized direction. For
the h-th future update (`h>=1`) after absolute checkpoint time t, write

```
b(t,h) = sqrt(1-.95^(t+h))/(1-.9^(t+h)),
u_h = (m_h/(1-.9^(t+h))) /
      (sqrt(v_h)/sqrt(1-.95^(t+h)) + eps).
```

For `eps>0`, AD2 and FF1 give `|u_h|<=sqrt(K_h(r))*b(t,h)`, including zero
variance. Epsilon remains in the actual update; dropping it only enlarges
the upper bound. The Lean function uses zero-based update index `h`, hence
its next buffer and bias exponent are `h+1` and `t+h+1`.

Simultaneously reversing the initial first moment and every future gradient
negates every first moment while preserving every second moment. Thus the
same moment magnitudes and cone can support opposite directions. This
symmetry limits what magnitude-only information implies; it is not a
counterexample to high-probability discovery under the actual training loss.

### FF3. Readout blocks and class centering

For a coordinate block J of size d, with checkpoint ratios r_k,

```
sum_(k in J) K_h(r_k) = d*C + (sum_(k in J) r_k-d*C)*Q^h,
||u_(J,h)||_2 <= b(t,h)*sqrt(sum_(k in J) K_h(r_k)).
```

This gives separate budgets for the whole readout and its observed-answer
row block. CG3's mean-subtraction identity at center zero proves
`sum_c (x_c-mean(x))^2 <= sum_c x_c^2`; summing across feature coordinates
therefore bounds the active class-contrast projection by the active-row
budget. The complementary centered projection has norm at most the whole
readout norm by CG3's orthogonal decomposition. Neither step assumes that
the actual gradient lies in the active subspace.

### FF4. A checkpoint-only finite tube and its limits

For the exact-real continuation `p_(h+1)=a_h*p_h-eta_h*u_(h+1)`, with
`a_h=1-eta_h*rho>=0` and `eta_h>=0`, set `q_0=1`, `R_0=0` and

```
q_(h+1) = a_h*q_h,
R_(h+1) = a_h*R_h + eta_h*B_(h+1),
B_h = b(t,h)*sqrt(sum_k K_h(r_k)).
```

DP1 now gives `||p_h-q_h*p_0||<=R_h` with B computed entirely from the
checkpoint and known schedule. `warm_adam_source` identifies each coordinate
with the existing buffered AdamW definition, and `main_adam_resume` identifies
the whole continuation with the original parameter history. `warm_adam_tube` has no premise
bounding a future gradient or observed forcing. Its norm bounds compose with
DP2 to bound active-energy share using the active-row and whole-readout
radii; `warm_adam_projected_tube` gives the tube for any fixed contractive
linear projection. A zero lower share is an exhausted certificate, not preserved useful
geometry.

The native forecast adds the protocol's predeclared numerical allowances:
relative direction allowance 1e-6 and a per-coordinate rounding cap
`8*eps32*(1+M_h+eta_h*Bmax_h*(1+1e-6))`. The coordinate cap M is itself
propagated from the checkpoint, schedule, and these predicted bounds. It
never substitutes observed future rounding into a forecast. These allowances
are audited against every saved FP32 update; their universal IEEE/CUDA
validity is not a Lean theorem. Exact-real proofs and finite numerical
verification remain distinct.

The missing discovery argument is still about the actual loss: why typical
initialization and minibatch sequences create the relevant class contrasts
and align the changing features with them. Long useful persistence also needs
more than a magnitude bound when the prospective radius exhausts the active
norm. The sign symmetry and the measured failures motivate controlling that
directional interaction; they do not prove such control impossible.

## Loss-specific momentum control

This continuation strengthens CG1's sign of unused-class CE damping to a
quantitative inequality, then absorbs arbitrary saved first moments in a
quadratic potential. These are authored consequences of the actual partial
CE derivative and the main model's beta1=.9 recurrence. They are not claims
that the flat-spectrum manuscript proves full tied AdamW training dynamics.
The corresponding files are `GradientEvidence/CECocoercivity.lean`,
`MomentumPotential.lean`, and `CEMomentumControl.lean`.

### CL1. Quantitative damping of the actual partial CE gradient

For a finite class set, let p be softmax(z). The exponential tangent gives
`exp(a)-exp(b) <= (a-b)*exp(a)`, with either ordering of a,b. Divide by the
common positive softmax denominator. When p_c>=p_d, multiply the resulting
inequality by p_c-p_d and use p_c<=q; when p_d>=p_c, swap the endpoints.
CG1's sign `(z_c-z_d)*(p_c-p_d)>=0` justifies both cases. Thus

```
(p_c-p_d)^2 <= q*(z_c-z_d)*(p_c-p_d),   p_c<=q, p_d<=q.
```

For minibatch features T_i and readout rows E_c, logits are `<E_c,T_i>`.
If neither class c nor d occurs as a target, their actual partial gradient
difference is `G_c-G_d = mean_i ((p_ic-p_id)*T_i)`. Holding T fixed here
specifies the derivative, not its behavior at later training steps.
Finite Cauchy–Schwarz bounds the squared norm of this mean by the product of
mean squared probability difference and mean feature energy. The displayed
softmax bound and CG1's gradient identity then give

```
||G_c-G_d||^2 <= L*<E_c-E_d,G_c-G_d>,
L = q*mean_i ||T_i||^2.
```

Sum over all pairs of unused classes and divide by twice their count. CG3's
computed centering identity yields `||G_U||^2 <= L*<E_U,G_U>` for the whole
within-unused contrast block. Its radial product is nonnegative by CG1.
No fixed feature table, fixed minibatch, probability independence, or
spectral-flatness assumption enters this pointwise statement. The Lean
statements also permit repeated class indices (the same weighted finite
pair identity) and empty batches (the existing zero-mean convention);
the experiment uses 64 distinct unused classes and nonempty minibatches.
For clipping a>=0, g=a*G_U has coefficient aL and nonnegative radial product.
The probability ceiling q=1 is always valid. A tighter measured q only
bounds the current minibatch unless a separate future bound is available.

### CL2. Saved momentum is dissipative in a suitable potential

Write w=E_U and let m be any saved within-unused first moment, including
arbitrary encoder/minibatch history. For a scalar effective step s define

```
m_R = .9*m+.1*g,       w_R = w-s*m_R,
V_s(w,m) = ||w-9*s*m||^2+90*s^2*||m||^2.
V_s(w_R,m_R)-V_s(w,m) =
  -2*s*<w,g>+19*s^2*||g||^2-(171/10)*s^2*||m-g||^2.
```

This is a polynomial identity with the main model's actual first-moment
coefficient. If s>=0, `<w,g>>=0`, `||g||^2<=L*<w,g>` and `19*s*L<=2`,
the first two terms sum to at most zero; the last is nonpositive for every
m. No alignment or smallness premise on the saved moment is needed.
The exact identity

```
19*V_s(w,m)-10*||w||^2 = 9*||w-19*s*m||^2
```

also gives `||w||^2 <= (19/10)*V_s(w,m)`. The potential bounds ordinary
unused-class contrast energy, not just a transformed position coordinate.
Induction yields nonincrease over every finite prefix of the declared
fixed-s recurrence. The CE specialization allows features, targets and
clipping to change at each step, provided the classes remain unused and
the explicit probability/feature-energy step condition holds each time.
The generic version instead takes a common L. Nonzero examples accompany
the conditional statements; the CE example has distinct unused readout
rows and nonzero saved memory. The sequence example demonstrates consistent
premises; it is not a discovery example.

This reference omits decay, coordinate-dependent scaling and the tied
encoder's current gradient. It does not show that unused energy itself is
monotone, that active contrast grows, or that prediction margins improve.
Changing s also changes V_s; one-step results with different s cannot be
chained without a bound on that change of potential.

### CL3. Exact full-model perturbations

Let g_R be clipped partial CE and g_E the clipped encoder contribution.
On a full readout coordinate, retain the actual new second moment v and
absolute-time index t+1. Write `b1=1-.9^(t+1)` and
`D=1/(sqrt(v)/sqrt(1-.95^(t+1))+eps)`. Let e_m and e_p denote moment
and parameter execution residuals relative to these exact-real expressions:

```
m_new = .9*m+.1*(g_R+g_E)+e_m,
w_new = (1-eta*rho)*w-eta*D*m_new/b1+e_p.
dm = m_new-m_R = .1*g_E+e_m,
dw = w_new-w_R = -eta*rho*w
     +(s-eta*D/b1)*m_new-s*.1*g_E+(e_p-s*e_m).
```

The scalar Lean identities use the existing buffered AdamW definitions,
with beta1=.9, beta2=.95 and both bias corrections. They are valid without
assuming the residuals small. Project the four position terms onto the
unused contrast block; projection commutes with scalar multiplication and
addition, but **not** with coordinate-dependent D. The native replay
therefore computes the adaptive term as `s*P_U(m_new)-eta*P_U(D*m_new)/b1`,
not by multiplying separately centered arrays. Encoder and execution terms
also carry their respective moment perturbations; decay and adaptive
scaling have zero moment perturbation in this decomposition.

For any reference state (w_R,m_R) and total perturbation (dw,dm),

```
V_s(w_R+dw,m_R+dm)-V_s(w_R,m_R) =
  2*<w_R-9*s*m_R,dw-9*s*dm>+180*s^2*<m_R,dm>+V_s(dw,dm).
```

The signed linear credits add over the four sources. The quadratic term
uses their **total** perturbation, preserving all cross terms. Adding CL2's
reference change gives the exact full-update potential identity. A concrete
one-dimensional example has a decreasing reference and an increasing
perturbed potential; it demonstrates that algebra alone does not sign the
perturbation, not a failure of the actual CE or GPT trajectory. A prospective
full-model certificate must bound these terms before the unknown update.

### CL4. Scope of the numerical diagnosis

The frozen `experiments/ce_lyapunov/PROTOCOL.md` uses the original next
minibatch at 144 saved states, preserving the actual model and optimizer.
An independent FP64 calculation supplies partial CE from executed features;
GPU/CPU derivative disagreement joins the execution residual. The raw
reference uses `s=eta*min(D)/b1` over the full readout. A separate stable
reference caps s by `1/(19*L)` when L>0, without changing training. These
statewise quantities use the actual next second moment and current q;
they are diagnoses, not forecasts. The declared tolerances test finite
executions and are not universal floating-point theorems. Results and
complete reproduction commands belong in the experiment README. Neither
these replays nor CL1–CL3 prove reliable discovery, full-model future
gradient alignment, or a useful active-share floor over the entire hold.

## Checkpoint-fixed feature metrics

The scalar reference in CL2 treats every coordinate-scale difference as a
perturbation. The following authored extension absorbs feature-dependent
scaling that is shared across unused classes. It leaves class-dependent
scaling and tied encoder updates explicit. The files are
`GradientEvidence/FeatureMetricCE.lean`, `FeatureMetricMomentum.lean`, and
`FeatureMetricControl.lean`; these do not assert a new manuscript theorem.

### FM1. Weighted damping from actual partial CE

Fix nonnegative feature weights r_k. Apply finite Cauchy–Schwarz to each
coordinate of `mean_i A_i*T_ik`, multiply by r_k, and sum. This yields

```
sum_k r_k*(mean_i A_i*T_ik)^2 <=
  (mean_i A_i^2)*(mean_i sum_k r_k*T_ik^2).
```

Use CL1's pair probability inequality with A_i=p_ic-p_id. For two classes
that never appear as targets, the actual partial-gradient pair obeys

```
sum_k r_k*(G_ck-G_dk)^2 <= L_r*<E_c-E_d,G_c-G_d>,
L_r = q*mean_i sum_k r_k*T_ik^2.
```

The radial product is the same unweighted nonnegative product as CG1.
Multiply the centered pair-product identity at feature k by r_k, then sum
over features. Applying the pair inequality to every unused pair gives
`sum_(u,k) r_k*G_U(u,k)^2 <= L_r*<E_U,G_U>`. The class means are computed;
the feature weights do not depend on the unused class index. This is why
scaling and centering commute. No corresponding commutation is asserted
for the actual row-dependent Adam diagonal. Empty batches use the existing
zero-mean convention; the experiment uses nonempty batches and 64 distinct
unused classes. As in CL1, repeated class indices are also permitted formally.

Common clipping a>=0 replaces G_U by g=a*G_U and L_r by aL_r.
If r_k<=R and mean feature energy is at most M, weighted mean feature energy
is at most R*M. Hence q=1 and 0<=a<=1 give a coefficient at most R*M without
observing a future probability or clipping scalar. This still requires the
declared feature-energy bound. In particular, a future sequence needs it
at each step; a single measured feature norm does not supply that premise.

### FM2. A fixed metric absorbs feature scaling and arbitrary memory

Require r_k>0. For centered unused readout w and arbitrary saved first
moment m, use the actual main beta1=.9 and the scaled reference

```
m_R = .9*m+.1*g,        w_R = w-s*r*m_R,
V_(r,s)(w,m) = sum_(u,k) ((w-9*s*r*m)^2/r+90*s^2*r*m^2).
```

Coordinatewise expansion and cancellation of positive r gives the identity

```
V_(r,s)(w_R,m_R)-V_(r,s)(w,m) =
  -2*s*<w,g>+19*s^2*sum r*g^2-(171/10)*s^2*sum r*(m-g)^2.
```

Thus s>=0 and `19*s*L_r<=2`, with the clipped coefficient, prove
nonincrease for every saved moment. No alignment, zero-memory or isotropic
feature assumption is needed. The universal-probability corollary instead
uses `19*s*R*M<=2` and the stated metric, feature and clipping bounds.
Induction proves the fixed-r, fixed-s result over any finite prefix of the
declared partial-CE recurrence. Features, targets and clipping may change,
with unused-class and coefficient hypotheses made explicit at every step.

The exact coercivity identity is

```
19*V_(r,s)(w,m)-10*sum w^2/r = 9*sum (w-19*s*r*m)^2/r.
```

Consequently, for 0<=R and every r_k<=R,
`||w||^2 <= (19/10)*R*V_(r,s)(w,m)`. The largest-weight penalty cannot be
omitted when comparing with ordinary geometric energy. Nonzero CE and
memory examples accompany the conditional statements. A sequence example
shows their premises are consistent, not that discovery occurs. The metric
and scalar step cannot change silently when chaining this theorem.

### FM3. Class-dependent and encoder remainders remain explicit

Apply CL3's buffered AdamW identity with scalar reference step s*r_k at
each full readout coordinate. For absolute time t+1, actual updated v and
`D=1/(sqrt(v)/sqrt(1-.95^(t+1))+eps)`, retain

```
dm = .1*g_E+e_m,
dw = -eta*rho*w +(s*r-eta*D/(1-.9^(t+1)))*m_new
     -s*r*.1*g_E+(e_p-s*r*e_m).
```

Project the four full-coordinate position terms onto unused class
contrasts. Feature scaling r commutes with that projection; D generally
does not. The adaptive remainder is therefore exactly
`s*r*P_U(m_new)-eta*P_U(D*m_new)/(1-.9^(t+1))`. It includes both differences
between class rows and changes since the checkpoint metric was frozen.
The encoder and execution sources retain their moment perturbations.

For reference (w_R,m_R) and total perturbation (dw,dm), direct expansion gives

```
V_(r,s)(w_R+dw,m_R+dm)-V_(r,s)(w_R,m_R) =
  sum [2*(w_R-9*s*r*m_R)*(dw-9*s*r*dm)/r+180*s^2*r*m_R*dm]
  +V_(r,s)(dw,dm).
```

The multi-source theorem distributes only the signed linear term, keeping
the quadratic energy of the total perturbation. The full-step theorem adds
FM2's exact reference change. These are identities, not advance estimates
on the signs or magnitudes of unknown encoder and adaptive remainders.

### FM4. Checkpoint construction and scope of the comparison

Task #207 fixes each metric from the old checkpoint before its separate
verifier opens any #206 next-update evidence. For t>0, compute the old
bias-corrected D_old; at the zero state take D_old=1/eps. Let alpha be its
minimum over the complete readout and set
`r_k=min_(unused c) D_old(c,k)/alpha`. The isotropic control has r=1 and
uses the same checkpoint scalar `s=eta(t)*alpha/(1-.9^(t+1))`. Thus the
comparison does not use the next second moment to choose its metric.

The next minibatch's exact-real partial CE is independently computed in
FP64 from the executed features. Its measured q supplies a diagnostic L_r;
the separately reported conservative condition uses q=1, clip<=1 and the
declared numerical feature cap. A stable reference caps s using the current
L_r, so it is explicitly a diagnostic rather than a forecast. All four
metric/step tracks retain every case, source, full increase and coefficient
failure. Since different metrics define different potentials, changes in
their numerical values are not effects of intervening on the optimizer.
The exact-real Lean results and the finite numerical allowances remain
separate. Full-model future control, discovery and prediction certificates
are not consequences of these comparisons.

## Prospective probability control

This continuation addresses FM4's future-probability premise. These are
authored consequences of the actual finite softmax and bilinear readout,
not new claims of the original flat-spectrum manuscript. The files are
`Geometry/ProbabilityMotion.lean`, `GradientEvidence/ProbabilityMotion.lean`
and `GradientEvidence/ProspectiveCE.lean`.

### PM1. Keep the class's own denominator term

For actual old and future logits z,z', fix a class u. Suppose every rival
satisfies `(z'_u-z'_c)-(z_u-z_c)<=D_c`. Rearranging and exponentiating gives
`exp(z'_c-z'_u)>=exp(z_c-z_u-D_c)`. Therefore

```
p'_u = 1/(1+sum_(c!=u) exp(z'_c-z'_u))
     <= 1/(1+sum_(c!=u) exp(z_c-z_u-D_c)).
```

No rival is discarded, and the self term is exactly one. When every D_c=D,
the old odds sum is `(1-p_u)/p_u`. Substitution gives the sharp ceiling

```
F(p_u,D) = p_u*exp(D)/(1-p_u+p_u*exp(D)).
```

For `0<=p<=q<=1`, direct cross multiplication gives `F(p,D)<=F(q,D)`;
the denominator is positive and `0<=F(q,D)<=1`, for either sign of D.
For D>=0 this also implies the less precise exponential bound `p*exp(D)`.
For negative D the self term matters: that simpler exponential bound need
not hold. The binary witness z=(log 4,0), z'=(log 4,log 2) has old
probability 1/5 and new probability 1/3 for the second class; F attains 1/3.

### PM2. Accumulate motion before using the ceiling

If each increment of `z_u-z_c` is at most D_t, telescoping gives total
deterioration at most `sum_(t<n) D_t` at every finite horizon n. PM1 then
bounds its future probability using only the initial probability ceiling
and those declared budgets. Budgets may be signed; this identity does not
assert that the actual probabilities or contrasts change monotonically.
To cover an arbitrary future training minibatch, these bounds must hold
for each input in the training population. The maximum probability on one
observed minibatch is not a ceiling over that population.

### PM3. Account for full readout and feature motion

Write the old class contrast as w=E_u-E_c and the feature as x, including
final normalization. At a later state they are w+dw and x+dx. The exact
score change is `<dw,x>+<w,dx>+<dw,dx>`. If

```
||w||<=A, ||x||<=X, ||dw||<=R, ||dx||<=S,
```

Cauchy–Schwarz gives absolute change at most `R*X+A*S+R*S`. The Lean
statements use the equivalent squared-energy premises and nonnegative
caps. For executed logits, an explicit allowance on the change of the
execution contrast adds e. Use that complete budget as D_c in PM1.

The readout and feature changes can be coupled: independence is unnecessary,
and the mixed term is retained. Class-dependent scaling is not assumed to
commute with centering. The parameter and feature radii must come from a
valid source-motion analysis; this theorem does not infer them from small
loss, a measured future feature displacement or a single checkpoint norm.
In particular, the feature radius must cover every moving encoder parameter
and normalization. Existing warm-moment Adam tubes address parameter motion;
their usefulness after propagation through the nonlinear model remains to
be measured in the new experiment.

### PM4. Supply the actual CE coefficient prospectively

Suppose old logits on every possible training input give unused-class
probabilities at most q. A uniform valid contrast-motion budget D gives
the new ceiling F(q,D) by PM1. If the feature weights satisfy r_k<=R,
the new mean ordinary feature energy is at most M, and clipping is in [0,1],
the weighted partial-CE cocoercivity coefficient is at most `F(q,D)*R*M`.
Hence

```
19*s*F(q,D)*R*M <= 2
```

implies the weighted reference potential's nonincrease at that future state,
for arbitrary saved first moment. This composes PM1 with the actual
partial-CE theorem FM2. The future probability is no longer a separate
smallness assumption. The motion and feature-energy premises are explicit,
and their prospective validity is still required.

The reference conclusion does not control the signed full-step perturbations
of FM3. Nor do statewise conditions with different s or r automatically chain
into one potential over time. The complete task still requires useful motion
and forcing bounds, a geometry/margin bridge and a discovery argument. The
new [experiment](../../experiments/probability_tube/README.md) records source
recovery and verification status separately from those open obligations.

## Full encoder motion

This continuation discharges PM3's separate feature-motion premise from
checkpoint parameter caps. Its reference graph is the plain arm of the
committed `experiments/flat_approx/fixtures/reference.rs` and `src/model.rs`
at 62638521. The Lean namespace is `Transformer.Rotate90.FeatureMotion`.
The original trainer's exact bytes and execution remain prerequisites for
recreating the training evidence; a reference forward equation is insufficient
to recover its initializer, sampler and full training process.
The trainer has now been recovered at b70811e with its exact pinned digest;
its matrix aliases, buffer layout, four normalizations and product kernel
agree with the reference graph inspected here. New execution evidence is pending.

### EM1. Moving inputs and linear layers

Let each shared embedding row have displacement at most R_c. The fixed
quarter-turn J is an isometry, so the input E[a]+J E[b] moves by at most
R_a+R_b, also when a=b. A real linear layer with checkpoint matrix A,
operator cap L>=||A||, matrix displacement ||A'-A||<=R and input displacement
||x'-x||<=r satisfies

```
||A'x'-Ax|| <= (L+R)*r + R*||x||.
```

Expand the difference as `A*dx+dA*x+dA*dx`; the final product is essential.
All terms on the right are checkpoint quantities or declared displacement
caps. In contrast, a one-update forecast that computes ||A'|| from the known
proposed parameters cannot be used unchanged for unknown future updates.

`matrixMap_apply` identifies the continuous real map with ordinary dense
matrix multiplication; `matrixMap_sub` identifies its displacement.
`matrixMap_norm_le` supplies the explicit Frobenius cap via FT2. The native
constructor uses these Frobenius caps, including every repeated entry of an
expanded tied matrix. It does not substitute an estimated leading singular
value. The expanded input width is 128, hidden width 682 and parameter-buffer
length 274304. The existing `optimizer_efficiency/layout.rs` records signed
matrix/first-moment aliases and unsigned second-moment aliases; counting
expanded coordinates conservatively does not assert that they are independent
trainable owners.

### EM2. Every normalization retains its checkpoint denominator

For real width d>0 and epsilon>0, put s(x)=sqrt(||x||^2+d*epsilon).
The exact source RMS is `sqrt(d)*x/s(x)`, as `rms_eq_source` proves.
Given an incoming displacement cap r, intersect three valid output caps:

```
2*sqrt(d)                                      (range diameter),
2*sqrt(d)*r/s(x)                               (FF2),
sqrt(d)*r/sqrt(s(x)*(s(x)-r))   when r<s(x)      (FT1).
```

When r>=s(x), omit the local secant candidate. No future denominator enters.
The first bound follows from ||RMS(x)||<=sqrt(d); the final feature therefore
has squared energy at most d at every parameter state. The same argument
applies to the input, branch-input, branch-output and final normalizations.
Near-zero inputs retain epsilon rather than assuming a nonzero raw norm.

### EM3. Complex products retain both moving factors

For the actual paired kernel
`P(g,u)_j=((g_re*u_re-g_im*u_im),(g_re*u_im+g_im*u_re))/sqrt(2)`,
the scalar modulus identity gives

```
||P(g,u)||^2 = sum_j |g_j|^2*|u_j|^2 / 2
            <= ||g||^2*||u||^2 / 2.
```

Expanding the finite change into `P(dg,u)+P(g,du)+P(dg,du)` gives
`(r_g*||u||+||g||*r_u+r_g*r_u)/sqrt(2)`. The last term is required even
when both old factors vanish. The input and output linear maps may be
arbitrary real maps; source complex-linear ties are a valid special case.
Logical pairs replace the native hidden buffer's separated real/imaginary
halves, without changing its Euclidean norm or calibrated product.

### EM4. Compose the whole encoder and its tied readout

Evaluate the checkpoint graph once. Transport `R_a+R_b` through the first
two RMS bounds, then separately through the moving gate and value maps.
Apply EM3, the moving output map and branch RMS. Add the first RMS radius
for the residual, then apply the final RMS bound. `feature_motion` proves
that the resulting explicit S bounds the final feature displacement. Its
only motion premises are parameter caps. `feature_eq_source` and
`logit_eq_source` prove equality with the previously defined expanded
complex-linear source graph, including all normalizations and ties.

For readout classes u,c, R_u+R_c bounds their contrast-row displacement.
PM3 then gives

```
D_uc = (R_u+R_c)*||x|| + ||E_u-E_c||*S + (R_u+R_c)*S.
```

`probability_le_encoder_motion` feeds these bounds into PM1 for every rival.
The old executed logits are the old exact scores plus their execution
residuals. A separately bounded CHANGE of the execution contrast is added
to D_uc. Dropping the old residual while calling a future residual a change
would be incorrect; the native interface accepts both explicitly.
`feature_energy_le` also supplies PM4's ordinary energy cap d, rather than
assuming it of an unknown future feature. These results still control the
partial-CE reference condition, not the signed full-update perturbations FM3.

### EM5. Obtain full-block parameter caps from saved Adam moments

For the selected main-model adapter's betas (0.9,0.95), retain checkpoint
p,m,v for any complete coordinate block, with v>=0 and
m_k^2<=r_k*v_k. FF4 gives an arbitrary-gradient future tube about q_n*p,
where q_n is the known product of weight-decay factors. Recenter it at p:

```
||p_n-p|| <= |q_n-1|*||p|| + forcingRadius_n.
forcingRadius_(h+1) = (1-eta_h*rho)*forcingRadius_h
                   + eta_h*bias(start+h+1)*sqrt(sum_k futureCone(r_k,h+1)).
futureCone(r,h) = (81/95)^h*r + (1-(81/95)^h)*(19/14).
```

`warm_adam_checkpoint_motion` proves this for every finite prefix and any
future gradients, including absolute-time bias correction and arbitrary
valid saved moments. Applying it separately to each embedding row and each
full matrix yields the native parameter caps. The constructor never receives
future parameters, moments, gradients or feature values. Bounds count the
whole model, rather than the earlier readout-only subset. A zero variance
with nonzero first moment is rejected. No observed future direction norm is
smuggled into a hypothesis.

These are exact-real optimizer bounds evaluated in FP64 by the native code.
The floating-point optimizer recurrence, serialized source correspondence,
outward rounding and execution-error allowances remain separate obligations.
The tests use explicitly synthetic parameter vectors with the source ties;
they do not invoke the original initializer or train a replacement baseline.
They check each intermediate stage, original dimensions, near-zero inputs,
all class contrasts and probabilities, and 24 arbitrary-gradient warm-Adam
prefixes across every parameter block. Eight native tests pass, including
the four earlier PM tests. This establishes implementation checks, not a
useful horizon on any of the 144 frozen training checkpoints.

## Pairwise encoder amplification

### PA1. Use the pointwise structure of the complex product

EM3's full-vector amplification is conservative when gate and value energy
is distributed over many logical pairs. For the unchanged calibrated source
product, the complex-modulus identity gives

```
||product(g,u)||² = (1/2) sum_j |g_j|² |u_j|².
```

If every old gate pair has amplitude at most G and every old value pair has
amplitude at most U, the two linear terms in finite product motion obey

```
||product(g+dg,u+du)-product(g,u)||
  <= (U*||dg|| + G*||du|| + ||dg||*||du||)/sqrt(2).
```

G and U depend only on the checkpoint activations. The mixed term cannot
use a small old amplitude: both factors are unknown perturbations. Apply
this bound inside EM4, keeping all RMS normalizations, moving matrices and
residual terms. Intersect the resulting feature bound with EM4's original
one. Every checkpoint operator cap still requires its own proof; the
existing FT2 Gram bound is an available alternative to Frobenius caps.
No estimated singular value is treated as a certified operator norm.

This continuation is independent of the frozen `1fb060b9` training audit.
It does not change that audit's formulas or numerical allowances. A sharper
local formula alone is not a useful training horizon or a bound on the
full-update potential; those remain separate experimental and mathematical
obligations.

### PA2. Use a proved Gram cap for the same moving maps

For an old real matrix A, FT2 bounds its operator norm by
`(sum_ij (Aᵀ*A)_ij²)^(1/4)`; the row Gram gives the same valid bound with
the smaller matrix dimension when appropriate. The source wrapper
`matrixMap` represents exactly that real matrix action. PA2 exposes this
correspondence for both orientations and for an upper bound on Gram energy.
The native continuation reuses FT2's outward-expanded scalar Gram calculation,
including its checks against identity, rank-one and transposed matrices.
This certifies no whole forward execution: the remaining activation/radius
arithmetic and CUDA execution retain their separate numerical obligations.

## Query-weighted feature covariance

The complete recreated comparison in `experiments/probability_tube` validates
both encoder forecasts but yields zero useful positive raw CE horizons. This
continuation examines another loss of information: replacing every example's
probability by the maximum, then replacing a feature operator by its trace.
The mathematical reference remains actual partial CE with arbitrary saved
momentum; full tied AdamW perturbations remain separate.

### QG1. Retain individual probabilities and the feature operator

For a batch of n examples, let T_i be its actual feature vectors, q_i>0 a
ceiling on every unused-class probability for that example, and r_k>0 the
fixed feature metric. Define the matrix with feature rows and example columns

```
W_ki = sqrt(r_k)*T_ik*sqrt(q_i),
L_op = ||W||_op^2/n,
L_trace = mean_i q_i*sum_k r_k*T_ik^2.
```

For arbitrary real a_i, apply W to a_i/sqrt(q_i). The probability weights
cancel from the true mean, yielding

```
sum_k r_k*(mean_i a_i*T_ik)^2 <= L_op*mean_i (a_i^2/q_i).
L_op <= L_trace <= Q*mean_i sum_k r_k*T_ik^2,   q_i <= Q.
```

The first inequality is the L2 operator bound; the second is the Frobenius
bound, and the third is nonnegative weighted summation. Two orthogonal unit
features with q_i=r_k=1 give L_op=1/2 and L_trace=1. A flat feature covariance
can therefore improve this coefficient without assuming that training
discovers such a covariance. The Lean convention also permits empty batches,
for which means and coefficients are zero; the source uses nonempty batches.

### QG2. Apply the operator to the actual CE derivative

For two unused classes c,d, write a_i=p_ic-p_id. CL1 gives
`a_i^2 <= q_i*(z_ic-z_id)*a_i`. Each q_i is positive because it bounds an
actual finite-logit probability. Divide before summing, then use QG1 and
CG1's exact unused-pair gradient identity:

```
sum_k r_k*(G_ck-G_dk)^2 <= L_op*<E_c-E_d,G_c-G_d>.
```

Sum the same coefficient over every unused pair and apply the computed
centering identity. Thus the whole centered unused gradient satisfies
`sum r*g_U^2 <= L_op*<E_U,g_U>`. Common clipping a>=0 changes the coefficient
to a*L_op. FM2's exact momentum identity then uses `19*s*a*L_op <= 2` in the
same fixed metric, with no alignment premise on saved memory. Changing the
coefficient does not change the executed optimizer or remove its encoder,
class-dependent scaling, decay and execution terms.

### QG3. Transport the coefficient through full feature motion

Use the same q_i and r_k in an old and a future table. Suppose r_k<=R,
R>=0 and `||T'_i-T_i|| <= S_i`. The weighted matrix difference has squared
Frobenius norm at most `R*sum_i q_i*S_i^2`. For any certified old operator
cap B, the triangle inequality gives

```
L_op(T') <= (B + sqrt(R*sum_i q_i*S_i^2))^2/n.
```

FT2 supplies B from the complete row or column Gram, including correlations.
In the smaller feature Gram, define `C_kl=sum_i q_i*T_ik*T_il`; its squared
Frobenius energy is `sum_kl r_k*r_l*C_kl^2`. An upper bound U on that explicit
energy gives `B=U^(1/4)`. A pointwise future feature-energy cap M also gives
`L_op(T') <= R*M*mean_i q_i`; intersecting both bounds retains the trace
control. These formulas use every probability and every feature radius.

The feature radius must cover executed features, not just the ideal encoder.
The previous ideal radius plus both endpoint execution errors supplies this
radius. Likewise, a GPU probability ceiling and a readout dot error at most
e give a real-CE ceiling via PM1 with an additional contrast allowance 2e.
The new numerical comparison must retain this conversion explicitly. It must
not infer a universal floating-point theorem from endpoint checks.

### QG4. Known batches and the remaining full-model obligation

The original source plan uses batches of 512, with a shorter final batch in
each epoch. Apply each coefficient to its actual finite batch and take the
maximum over the known batch sequence in the declared window. Varying batch
sizes, changing features and changing clipping do not change the fixed-r,
fixed-s momentum argument. Bounds and conditions must hold at every prefix;
an endpoint diagnosis alone cannot supply them.

The frozen native continuation must keep all 144 anchors, all 1800 horizons,
both source probability forecasts and both metrics. A useful partial-CE
condition still needs separate control of full adaptive/encoder perturbations
before it becomes a training stability result. No new numerical gain is
asserted by this derivation.
