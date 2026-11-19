# Training-state transport and feature forecasts

Moved unchanged from [`../flat_spectrum.md`](../flat_spectrum.md) on 2026-11-19: the sections
*Training-state transport* through *Training-to-forecast bridge*. They bound the actual tied
readout on unseen queries from observed training states and the trained weights: transport of
query displacements (TT, ST), what a small loss controls (LG, PT), residual and adaptive
dynamics (RD, AI), single-update evidence and difficult queries (JC), and forecasts of the
features (FF1–FF3, FT, TF). The Lean is mostly in `Transformer.Grokking.GradientEvidence`, with
transport modules in `Transformer.Grokking.Geometry`, `FlatSpectrum.FeatureCodes` and
`Transformer.Rotate90.ComplexProduct`.

These labels are this file's; its FF1–FF3 are not the FF1–FF4 of [`dynamics.md`](dynamics.md).
AC, CP and the other earlier labels are in [`../flat_spectrum.md`](../flat_spectrum.md).

## Training-state transport

Added 2026-11-08, task #180. AC4 still required a measured error against an
ideal answer code and a source-alignment term. Here the bound works directly
with the actual tied readout. Its inputs are observed training labels/states,
the trained readout weights, raw query operands and the unlabeled query state.
It is an input-dependent deterministic condition, not an optimizer theorem,
a statistical coverage claim, or a prediction made without running the model.

### TT1. Reconstruct a query displacement

Let `T_i` be real training states in a group that preserves the task answer,
`mu` a center computed from them, and `x` the query state. The diagnostic uses
the training mean as `mu`. For arbitrary real coefficients `a_i`, define

```
H_i = T_i - mu
r = x - mu - sum_i a_i H_i.
```

The residual is calculated from states and is never assumed to vanish. For
any linear contrast `L_w(z)=sum_k w_k z_k`, exact algebra gives

```
L_w(x)-L_w(mu) = sum_i a_i (L_w(T_i)-L_w(mu)) + L_w(r).
```

Neither a fitted ideal decoder nor its alignment with the source is needed.

### TT2. Joint training/residual energy

For nonzero real `s`, put `lambda=s^2>0` and

```
G = sum_i a_i^2 + ||r||_2^2/lambda
B_w = sum_i (L_w(T_i)-L_w(mu))^2 + lambda*||w||_2^2.
```

Apply Cauchy–Schwarz to the concatenated vectors
`(a_i, r_k/s)` and `(L_w(T_i)-L_w(mu), s*w_k)`. This proves

```
|L_w(x)-L_w(mu)| <= sqrt(G*B_w).
```

All terms on the right use training contrasts, known weights and query-state
geometry. No query target error occurs. Omitting the residual term is false:
zero training states and center, with scalar query and readout both one, have
zero training variation and query contrast error one. This counterexample is
proved explicitly.

A separate Euclidean-center baseline is
`|L_w(x)-L_w(mu)| <= ||x-mu||_2*||w||_2`. It uses displacement from a
training center, which can be much smaller than the complete query state.

The numerical diagnostic chooses `a` by regularized linear reconstruction:
`z=(H^T H+lambda I)^(-1)(x-mu)`, `a=H z`, `r=x-mu-H^T a`. The theorem
does not depend on inversion correctness or optimality: any coefficients with
their actual residual satisfy its inequality. Floating-point evaluation of
the coefficients, residual and inequalities still needs arithmetic certification.

### TT3. Actual source classification and valid bound selection

For target label `t` supplied by training and competitor `v`, take
`w_v=realify(E_t)-realify(E_v)`, the difference of the actual tied embedding
rows. The existing real/complex correspondence proves
`L_wv(realify(state)) = rawLogit_t-rawLogit_v` exactly. Put `g_v=L_wv(mu)`.
The source classifier's executed comparison is correct whenever

```
sqrt(G_v*B_v) + 2*eta*D < g_v
```

for every wrong class, where `D` is the actual final RMS denominator and
`eta` bounds each executed logit's error. A separate theorem supplies the
Euclidean baseline. Another permits selecting a different valid upper bound
for each competitor: every candidate must bound that contrast error before
selection. This justifies the diagnostic's fixed scale grid and its mixture
with the Euclidean baseline without using query target errors to select scales.
There is no theorem that AdamW discovers small training variation or transport
energy, nor a claim that every query satisfies the strict budget.

### TT4. Transfer a training label along a task-preserving route

In `ZMod 97`, each `u` in `{1,35,61}` satisfies `u^3=1`. The route

```
(a,b) -> (u*a+t, u*b-t)
```

therefore preserves `(a+b)^3`. Conversely, a checked sum relation
`c+d=u*(a+b)` supplies the route with `t=c-u*a`. A theorem composes this
identity with an observed training label. The numerical constructor groups
equal sums, or sums in the same three-element orbit, and checks an explicit
allowed multiplier and translation for every member. Groups without training
anchors abstain. The constructor does not cube query sums to obtain labels.
Task symmetry is known here; no equivariance of the trained network is assumed.

### TT5. Concrete witnesses and practical scope

Take two real training states `(1,1),(1,-1)`, center `(1,0)`, query
`(9/8,1/4)`, contrast weights `(2,0)`, coefficients `(1/8,-1/8)` and
`s=1/2`. The residual is `(1/8,0)`, and `G*B=3/32`, so its square root
is strictly below the center margin two. Lean proves both the nonzero residual
and the positive budget. The source theorem also has CP5's exact-center witness.
That witness remains a residual-path example, not a learned active product.

The [native diagnostic](../../experiments/contrast_transport/README.md) constructs
and hashes all candidate labels and bounds before a separate audit accesses
query labels and logits. Positive FP64 budgets are useful evidence about these
saved states, but they do not constitute verified backend arithmetic or an
independent statistical test. The source-specific Lean bridge covers plain
complex product; the generic linear-readout inequalities apply to all four arms.

| Claim | Lean declaration |
| --- | --- |
| Center-displacement baseline | `FeatureCodes.center_contrast_error_le` |
| Exact training/residual decomposition | `FeatureCodes.linearScore_transport` |
| Weighted Cauchy–Schwarz and query error | `FeatureCodes.weighted_transport_sq_le`, `training_transport_error_le` |
| Necessary residual and nonzero positive witness | `FeatureCodes.residual_penalty_necessary`, `training_transport_nonzero_witness` |
| Exact cubic route and observed-label transfer | `FeatureCodes.cubic97_route`, `cubic97_route_of_sum`, `cubic97_transfer_label` |
| Actual source contrast identity | `ComplexProduct.linearScore_contrastRow` |
| Source decision from valid bound choices | `ComplexProduct.executed_correct_of_center_bounds` |
| Euclidean and training-transport source budgets | `ComplexProduct.executed_correct_of_euclidean_center`, `executed_correct_of_training_transport` |
| Source witness | `ComplexProduct.copy_training_transport_classifies` |

## Loss-controlled geometry

Added 2026-11-08, task #181. TT1–TT5 bound query contrasts from training
geometry but do not say that the objective reduces raw state or contrast
scatter. Before assuming such a mechanism, distinguish the quantity that
small cross-entropy actually controls. The results below apply to the real
finite-class CE used by the source models, including all competing labels.

### LG1. The objective controls missing target probability

For arbitrary finite logits `z`, write

```
p_c = exp(z_c) / sum_j exp(z_j)
ell(z,t) = log(sum_j exp(z_j)) - z_t = -log(p_t).
```

A target index witnesses that the denominator is positive. Thus every `p_c`
is positive, each is at most one, and the probabilities sum to one.
The elementary inequality `log(x) <= x-1` for `x>0` yields

```
1-p_t <= ell(z,t).
```

This is a theorem about the actual objective, with no correct-classification,
large-margin, optimizer or feature-collapse premise. The finite softmax in
`Geometry.probability` agrees with the source's log-sum-exp CE definition in
`Landscape.Neural.Loss`.

### LG2. Small loss controls squared probability error

Let `e_t` be the one-hot target vector. Since `sum_c p_c^2 <= sum_c p_c = 1`,

```
||p-e_t||_2^2 = sum_c p_c^2 + 1 - 2*p_t
              <= 2*(1-p_t) <= 2*ell(z,t).
```

Lean derives this from the finite sums. The constant two is a convenient
universal bound; no sharpness is claimed. This does not bound raw logits or
hidden states: softmax is nonlinear and does not have a uniform inverse
Lipschitz constant in the high-confidence regime.

### LG3. Training probability scatter follows from training loss

For a finite same-answer group of rows `S`, define the computed probability
mean `pbar = sum_{i in S} p_i / |S|`. Coordinatewise least-squares optimality
of that mean, already proved by `Geometry.mean_minimizes_error`, gives

```
sum_i ||p_i-pbar||_2^2 <= sum_i ||p_i-e_t||_2^2 <= 2*sum_i ell_i.
```

Consequently mean CE at most `epsilon` implies mean within-group probability
scatter at most `2*epsilon`. Every same-answer training group obeys this;
summing the group inequalities gives the corresponding weighted split bound.
The exact centering identity also separates the two terms:

```
mean_i ||p_i-e_t||_2^2 = mean_i ||p_i-pbar||_2^2 + ||pbar-e_t||_2^2.
```

The mean could be confidently wrong if only its within-group scatter were
examined. CE controls the error to the correct target as well. These statements
apply to exactly the rows where CE is measured. Small training CE gives no
held-out loss bound. A decreasing loss lowers this upper bound; it need not
make the actual scatter decrease at every step. No optimizer convergence rate
or existence of a small-loss iterate is asserted.

### LG4. Actual shared-parameter descent can expand raw contrasts

Consider two same-label binary examples with logits `(theta,0)` and
`(2*theta,0)`, both targeting the first class. Their one shared trainable
scalar has the actual objective

```
F(theta) = (CE(theta,0) + CE(2*theta,0))/2
F'(theta) = ((sigmoid(theta)-1) + 2*(sigmoid(2*theta)-1))/2 < 0.
```

Their raw class contrasts are `theta` and `2*theta`; subtracting the actual
mean `3*theta/2` leaves total centered energy `theta^2/2`. For every `eta>0`
and `theta>=0`, the genuine gradient step
`theta_next = theta - eta*F'(theta)` increases theta, strictly lowers F and
strictly increases the raw contrast energy. Lean proves the derivative and
strict decrease rather than inserting a prescribed gradient in a definition.
There is no step-size restriction here because this particular scalar loss
is strictly decreasing on the entire real line.

This refutes unconditional raw-contrast contraction from CE descent, even with
shared parameters and no label conflict. It is a two-example classifier,
not a claim that the native transformer realizes this exact scalar trajectory.
The normalized source has additional parameter coupling and state geometry;
those must be established separately to explain its training dynamics.

### LG5. Fixed-cohort comparison and remaining obligation

The [native diagnostic](../../experiments/loss_geometry/README.md) measures both
geometries on all 24 original saved states, grouping by equal sums or cubic
answer orbits. Training and validation means are computed independently.
The protocol fixes the cohort, metrics and numerical tolerances before the
measurement. Adjacent checkpoints are finite training intervals, not gradient
steps; their geometry changes cannot identify a causal optimizer mechanism.

This extension establishes which output geometry follows from the observed
objective. It does not upgrade TT3 to a generalization theorem: its raw contrast
transport terms still need control, or a separate probability-space transfer
bound must connect unobserved inputs to training evidence. Validation versions
of LG1–LG3 use validation CE and are retrospective checks only. All numerical
calculations are FP64 evidence, not certified execution arithmetic.

| Claim | Lean declaration |
| --- | --- |
| Probability simplex and actual CE identity | `Geometry.probability_pos`, `sum_probability`, `probability_le_one`, `crossEntropy_eq_neg_log_probability` |
| Objective controls missing probability and one-hot error | `Geometry.missing_probability_le_loss`, `probability_error_le_twice_loss` |
| Group scatter and its loss ceiling | `Geometry.probability_scatter_le_twice_loss`, `mean_probability_scatter_le` |
| Actual shared-scalar derivative and sign | `GradientEvidence.sharedContrastLoss_hasDerivAt`, `sharedContrastLoss_deriv_neg` |
| Strict loss decrease and computed raw scatter | `GradientEvidence.sharedContrastLoss_strictAnti`, `sharedContrast_scatter` |
| Actual descent with expanding raw contrasts | `GradientEvidence.descent_increases_contrast_scatter` |

## Probability transfer

Added 2026-11-08, task #182. LG1–LG4 bound probability geometry only on rows
whose loss is known. The next step is a pointwise transfer inequality with
training probabilities and unlabeled query-state geometry on its right side.
It remains conditional on the measured query geometry; it is not a theorem
that training reaches small loss or that future inputs have small distance.

### PT1. Express confidence through competitor odds

For finite logits `z` and target `t`, ordinary softmax gives

```
p_c/p_t = exp(z_c-z_t)
R_true = sum_(c != t) exp(z_c-z_t)
p_t = 1/(1+R_true),       CE(z,t) = log(1+R_true).
```

All competing classes are included. Lean derives these identities from the
same `Geometry.probability` and actual CE definition used in LG1. A target
index ensures a nonempty class set; no finite logits have zero softmax
denominator. A one-class task has zero wrong odds and is covered by the
identities, although the numerical experiment uses all 97 classes.

### PT2. Valid envelopes transfer probability and loss

Suppose `z_c-z_t <= U_c` for every wrong class. Define
`R = sum_(c != t) exp(U_c)`. Monotonicity of exp, reciprocal and log gives

```
p_t >= 1/(1+R),       CE(z,t) <= log(1+R).
```

Average the latter over any declared finite query set to obtain a mean CE
ceiling; every query stays in the average, including failed classification
conditions. These are proved pointwise and finite-mean inequalities, not a
statistical confidence interval or a claim about an unseen distribution.

The condition `R<1` implies target probability above one half and a strict
target winner. The weaker condition `U_c<0` for every competitor also proves
the target wins, but need not put more than half the probability mass there.
Both decision implications are proved and reported separately.

### PT3. Derive the envelope from normalized state geometry

Let `E_c` be the actual tied readout rows, `h_i` normalized training states,
and `h` the normalized query state. For a selected training reference `i(c)`,
put

```
d_i = ||h-h_i||_2
L_c = ||E_c-E_t||_2
U_c = log(p_i(c),c / p_i(c),t) + d_i(c)*L_c.
```

The training log odds are evaluated stably as training logit differences.
The earlier Cauchy–Schwarz contrast bound proves

```
z_c(h)-z_t(h) <= z_c(h_i)-z_t(h_i) + d_i*L_c = U_c.
```

Thus the probability and loss bounds are derived from actual training scores,
readout weights and state distances. They do not require a measured query
target error or a query class-score evaluation on the right side. Different
competitors may select different valid references. Taking a finite minimum
of such valid bounds uses no query targets and preserves the inequality.

A computed mean of normalized training states is also a valid reference.
Its linear logits are the means of training logits, so its log odds are
mean training log odds. Its softmax probabilities are not the arithmetic
means of training probabilities. The generic geometric theorem allows this
reference without claiming that the center is itself a network input.

### PT4. Actual source normalization and numerical execution

For the plain complex-product source, the reference coordinates are
`realify(state_i)/D_i` and the query coordinates are `realify(state)/D`, with
each actual final RMS denominator retained. Lean proves that linear readout
of these coordinates equals the independently expanded source's real logits.
This includes all four original normalizations; replacing the denominators
by a fixed radius or by a mean denominator is not part of the result.

If every executed query logit differs from the exact source by at most `eta`,
replace each `U_c` by `U_c+2*eta`. The source probability and CE theorems
retain that hypothesis explicitly. For positive `R`, the probability
condition has available error budget `eta < -log(R)/2`, because the total
bounded wrong odds become `exp(2*eta)*R`. The diagnostic initially sets eta
to zero and reports this allowance; it does not establish backend arithmetic
accuracy or verify the saved floating-point computation in Lean.

Task symmetry supplies the candidate answer separately. The existing cubic
routes `(a,b) -> (u*a+t,u*b-t)`, `u in {1,35,61}`, transfer an observed
training label. No equivariance of the learned network is assumed. The
generic linear-readout result applies to every numerical arm; the source
bridge is specific to the formalized plain complex product.

### PT5. Witnesses, fixed diagnostic and limits

In the odds witness the training target logit is `log 4` with other logit
zero; the query target logit drops by `log 2`. The wrong-class odds envelope
is exactly `1/2`, giving target probability at least `2/3` and a strict winner.
The geometric witness uses scalar readout rows two and zero, training state
two and query state `3/2`. Its nonzero distance gives relative-logit bound
`-3` and total wrong odds below one. Actual-source examples instantiate the
bridge with nonzero complex branch parameters and distinct input pairs.

The [fixed Rust diagnostic](../../experiments/probability_transfer/README.md)
uses all 24 original checkpoints. It constructs both same-sum and cubic-orbit
bounds from training labels/scores and normalized query states, hashes them,
then separately audits query labels, probabilities, losses and logits.
The model/oracle evaluator does compute logits separately; the constructor
does not receive them. Training queries exclude themselves from both the
anchor family and center. Missing references cause abstention.

The protocol compares the training center, nearest anchor, best complete
anchor, per-competitor anchor envelope, and the envelope including the center.
All methods and failed conditions remain in the report. Query evaluation and
known task symmetry are required. Conditional correctness at these saved
states does not imply an optimizer-discovery theorem, independent test-set
coverage, or a priori generalization without running the query model.

| Claim | Lean declaration |
| --- | --- |
| Exact probability ratios and odds identities | `Geometry.probability_ratio`, `probability_eq_one_div_odds`, `crossEntropy_eq_log_odds` |
| Envelope nonnegativity and comparison | `Geometry.oddsEnvelope_nonneg`, `odds_le_envelope` |
| Query probability floor and CE ceiling | `Geometry.probability_ge_of_contrast_bounds`, `crossEntropy_le_of_contrast_bounds` |
| Mean query loss ceiling | `Geometry.meanCrossEntropy_le_of_contrast_bounds` |
| Stronger probability and weaker argmax conditions | `Geometry.strict_winner_of_odds_envelope`, `strict_winner_of_negative_contrast_bounds` |
| Nonzero odds witness | `Geometry.nonzero_probability_transfer_witness` |
| Training-center linearity and geometric bound | `FeatureCodes.linearScore_mean`, `relative_logit_le_anchor` |
| Probability and loss from training-state geometry | `FeatureCodes.probability_ge_anchor_bound`, `crossEntropy_le_anchor_bound` |
| Nonzero geometric witness | `FeatureCodes.nonzero_anchor_geometry_witness` |
| Exact normalized source readout | `ComplexProduct.readoutState_logit` |
| Execution allowance and source bounds | `ComplexProduct.executed_relative_le_anchor`, `executed_probability_ge_anchor`, `executed_crossEntropy_le_anchor` |

## Signed directional transport

Added 2026-11-08, task #183. PT3 uses a norm penalty for the entire displacement
from a training reference. TT1 already identifies an exact signed contribution
from reconstructed training directions. This extension preserves that term
and bounds only the remaining state/readout interaction. It proves a pointwise
bound, not a claim that training discovers a useful low-dimensional subspace.

### ST1. Preserve the signed training contribution

For normalized training states `T_i`, a center `mu`, query `x`, readout contrast
`w=E_c-E_t` and coefficients `a_i`, define

```
r = x - mu - sum_i a_i*(T_i-mu)
prediction = <w,mu> + sum_i a_i*(<w,T_i>-<w,mu>)
<w,x> = prediction + <w,r>.
```

The equality follows from TT1's exact transport decomposition. The coefficients
may have either sign, and reconstruction may extrapolate beyond the training
states. No convexity assumption or omitted residual is introduced. In the
diagnostic `mu` is the mean of normalized training states, so linearity gives
mean training log odds for `<w,mu>`; the weighted terms use only training
log-odds deviations. The baseline bound is

```
abs(<w,x> - prediction) <= ||r||_2 * ||w||_2.
```

Unlike TT2's covariance bound, the reconstructed training contribution is kept
signed. It is not replaced by an upper bound on its absolute value.

### ST2. Separate the remaining readout directions

For any finite vectors `B_j` and coefficients `b_j`, let

```
u = w - sum_j b_j*B_j
<w,r> = <u,r> + sum_j b_j*<B_j,r>
abs(<w,r>) <= ||r||_2*||u||_2 + sum_j abs(b_j)*abs(<B_j,r>).
```

Lean proves the decomposition and bound without assuming the vectors are
orthonormal. The second term is an explicit geometric orthogonality defect.
If every `<B_j,r>=0`, the two residual norms suffice, as a separate theorem
shows. This hypothesis cannot silently be imposed on floating-point SVD.
The counterexample `B=r=w=b=1` has zero readout residual but interaction one,
so omitting the defect without orthogonality is false even for a unit basis.

The theorem allows arbitrary vectors; the diagnostic restricts `B` to a basis
computed solely from centered training states, without query states or readout
weights. It uses `b_j=<B_j,w>`. The constructor computes query scores against
this training basis, but never the original query class readout. The remaining
readout norm is computed from weights and the training basis.

### ST3. Transfer probabilities and retain actual source execution

Set `U_c=prediction_c+residual_bound_c` for each wrong class. ST1–ST2 give
`z_c(x)-z_t(x)<=U_c`; PT2 then supplies

```
R = sum_(c!=t) exp(U_c)
p_t >= 1/(1+R),       CE <= log(1+R).
```

The same finite-mean theorem includes every query, including failed decision
conditions. `R<1` and `max U_c<0` retain their distinct probability-majority
and argmax meanings. Task-preserving cubic routes supply the candidate answer
from observed training labels; no learned equivariance is assumed.

The plain-product source bridge uses its actual normalized `readoutState`,
including all four RMS normalizations. With per-logit execution error at most
`eta`, each bound increases by `2*eta`. The numerical diagnostic starts at
eta zero and reports an available allowance. It does not prove an FP64 error
bound for its decomposition, training readouts or source evaluator.

### ST4. Rank control and the full-reconstruction limit

The [frozen native protocol](../../experiments/signed_transport/PROTOCOL.md) uses
thin SVD of centered normalized training states, with singular cutoff
`1e-10*s_max` and fixed ranks `0,1,2,4,8,16,32,64,128`. Each rank is clamped to
the available training rank. It uses the state projection to construct signed
training coefficients, then explicitly recomputes the residual from those
coefficients and training states. No exact SVD identity is assumed by the bound.

Rank zero gives the previous normalized training-center baseline. At the other
extreme, Lean proves `r=0` implies the training prediction equals the original
score. Full-dimensional reconstruction can therefore re-express the source's
classification without explaining its learning or compression. The diagnostic
retains every rank; successful full-rank reconstruction must not be reported
as evidence of a useful low-rank representation. Finite training rank and the
cutoff can also leave a residual at the requested ambient rank 128.

Validation queries use all training references in their group. Training queries
use the opposite half of alternating sorted group training indices, avoiding
self reconstruction. Their reference sets differ from validation's. Both
complete constructions are hashed before a separate target/score audit.

### ST5. Nonzero witness and scope

Take normalized-coordinate training states `(1/2,0)` and `(3/2,0)`, center
`(1,0)`, query `(3,1/4)`, contrast `w=(-2,1)`, and coefficients `(-2,2)`.
The signed prediction is `-6`; the residual `(0,1/4)` is genuinely nonzero.
The training basis `(1,0)` with readout coefficient `-2` leaves readout residual
`(0,1)`. Its defect is zero, and the upper contrast bound is exactly `-23/4`.
Both observed training contrasts are negative. Lean proves the nonzero residual
and exact upper bound; the native test also checks the resulting probability.

Known task symmetry and query evaluation remain necessary. A successful saved
state bound is neither optimizer convergence nor an independent statistical
generalization result. The generic linear-readout bound applies to every native
arm; the formal source bridge is specific to the plain complex product.

| Claim | Lean declaration |
| --- | --- |
| Exact signed reconstruction and baseline error | `FeatureCodes.linearScore_signed_residual`, `signed_transport_norm_le` |
| Readout residual identity and directional bound | `FeatureCodes.linearScore_readout_residual`, `directional_residual_abs_le` |
| Signed transport error and upper bound | `FeatureCodes.signed_transport_abs_le`, `signed_transport_upper` |
| Zero-residual re-expression limit | `FeatureCodes.exact_transport_of_zero_residual` |
| Orthogonal specialization and necessary defect | `FeatureCodes.orthogonal_residual_abs_le`, `orthogonality_defect_necessary` |
| Relative logit, probability and loss transfer | `FeatureCodes.relative_logit_le_signed`, `probability_ge_signed_bound`, `crossEntropy_le_signed_bound` |
| Nonzero extrapolation witness | `FeatureCodes.signed_transport_nonzero_witness` |
| Actual source with execution allowance | `ComplexProduct.executed_relative_le_signed`, `executed_probability_ge_signed`, `executed_crossEntropy_le_signed` |

## Residual learning dynamics

This project extension, task #184, asks what learning mechanism could reduce
ST2's directional residuals. It derives a fixed-feature readout result and
separately checks the limitations imposed by adaptive updates and moving
features. Complex multiplication is unchanged.

### RD1. Differentiate the actual readout objective

For normalized training states T_i, labels y_i, readout rows E_c and
L(E) = mean_i CE((<E_c,T_i>)_c,y_i), differentiate a scalar perturbation of
each coordinate of E. The exact result is

```
G_c = mean_i (p_i,c - 1[c=y_i]) T_i.
```

Lean defines G using the actual derivative and proves this equality; it does
not assume an update direction. This is the partial final-readout derivative
with states frozen. When the embedding is tied to the encoder, backpropagation
has another path. Neither the full tied gradient nor AdamW's update is G.
The line-derivative proof follows the local finite-logit calculation in
`Imbalance.hasDerivAt_crossEntropy_line` (arXiv:2402.19449v2, §3.2).

### RD2. Bound gradient forcing using training CE

For any fixed linear map Q, Cauchy–Schwarz and LG1's actual probability-error
inequality give

```
||QG||_F² <= mean_i sum_c (p_i,c - 1[c=y_i])² * mean_i ||QT_i||²
          <= 2 L(E) * mean_i ||QT_i||².
```

All training rows enter both means. Choosing Q from an answer group's
centered PCA does not remove other groups from the feature factor. No
orthogonality assumption on Q is needed for this inequality. A small loss
controls gradient forcing relative to projected feature scale, not the
current readout residual, an adaptive normalized direction, or future drift.

### RD3. The sufficient Euclidean descent mechanism

For fixed T, consider the explicitly defined Euclidean update with decay
E+ = (1-eta*rho)E - eta*G. Linearity gives the exact projected update.
If Q annihilates **every** training feature, QG=0, and induction yields
QE_n=(1-eta*rho)^n QE_0. With 0<eta*rho<1 the residual decays;
rho=0 gives invariance. The theorem allows arbitrary rates, so its identity
does not assert contraction outside the stated rate range.

For eta>=0, rho>0 and eta*rho<=1, the scalar square identity

```
(1-eta*rho)u² + (eta/rho)g² - ((1-eta*rho)u-eta*g)²
  = (eta/rho)(1-eta*rho)(rho*u+g)² >= 0
```

and RD2 imply

```
||QE+||_F² <= (1-eta*rho)||QE||_F²
             + (2eta/rho)L(E) mean_i ||QT_i||².
```

This is a derived decay-plus-forcing estimate. Small forcing relative to
the current energy is a sufficient one-step contraction condition. It is
not assumed to hold for the saved models. Concrete Lean examples use
nonzero training features, nonzero initial residuals and positive rates.

### RD4. Account for movement of the learned subspace

For arbitrary old/new residual maps and vectors, linearity gives

```
Q1 w1 = Q0 w0 + Q0(w1-w0) + (Q1-Q0)w1.
```

The corresponding energy difference can be split exactly by inserting
||Q0 w1||²: first change w with Q frozen, then change Q at the new w.
The same identity applies to readout contrasts and normalized query
displacements h-mu. The frozen native protocol retains both terms for all
adjacent saved intervals. These are retrospective changes over 5000 steps,
not an individual update or an identified causal effect of weight decay.

### RD5. An actual adaptive step can create a nullspace residual

One fixed feature is (1,2), with binary logits (u+2v,0), first-class target,
and zero initial readout. Differentiating actual CE gives (-1/2,-1).
Euclidean descent yields (eta/2,eta), which stays in the feature span.
The existing bias-corrected Adam first-update formula, with betas 0.9/0.999,
epsilon 1 and decay 0, instead gives (eta/3,eta/2). Its squared orthogonal
residual is

```
(2u-v)²/5 = eta²/180 > 0  for eta>0.
```

The correct margin strictly increases, so CE strictly decreases. Lean proves
the actual derivative, Adam formula, residual projection identity and strict
loss decrease. Epsilon is positive and deliberately fixed to one for this
counterexample; this is not the checkpoint training recipe or a replay of
that trajectory. It refutes transferring the Euclidean feature-span
invariance/decay argument unchanged to coordinatewise adaptive updates.

| Claim | Lean declaration in `GradientEvidence` |
| --- | --- |
| Actual coordinate derivative | `readoutLoss_coordinate_hasDerivAt`, `readoutGradient_eq` |
| Objective-controlled projected gradient | `projected_readoutGradient_energy_le` |
| Annihilated directions and iterated decay | `projected_gradient_zero`, `projected_readoutRun` |
| Derived decay and loss-controlled forcing | `decay_forcing_sq_le`, `readout_step_energy_le` |
| Changing weights and basis | `moving_residual_identity` |
| Actual adaptive counterexample | `axisReadoutGradient_initial`, `axisAdamStep_eq`, `adam_creates_readout_residual` |

## Adaptive interaction control

Task #185 extends RD1–RD5 to coordinate-adaptive forcing and moving residual
interactions. This is a sufficient-control theory with explicit missing terms;
it does not conclude that the optimizer makes those terms small.

### AI1. Retain the actual buffered Adam direction

The source is the local Rust/Kernite rotate90_complex implementation at source
hash 14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da:
`engine.rs::update`, `product.rs::moment`, `ops.rs::adamw`, and the buffer
layout in `storage.rs::state_bytes/decode`. The checked cohort uses coordinate
AdamW, betas 0.9/0.98, epsilon 1e-8, decay 1 and rate 0.001 after ten warmup
updates. Embedding coordinates are independent in all four arms; the complex
projection constraints do not change this embedding update.

Let m_t,v_t denote the buffers *after* incorporating update t's gradient.
In exact real arithmetic the source factors as

```
M_t = m_t/(1-beta1^t)
D_t,k = 1/(sqrt(v_t,k)/sqrt(1-beta2^t)+epsilon)
w_new = (1-eta*rho) w_old - eta D_t M_t.
```

Lean proves the factorization and projected update without identifying M_t
with a current reference gradient. The source uses t>=1 and nonnegative
second moments; the initial checkpoint has no historical bias-corrected
direction. The native audit retains that case with an explicit unavailable
status. It evaluates real-form equations in FP64 on saved FP32 buffers, not
a bitwise replay of the FP32 CUDA kernel. In a trained checkpoint, the
historical moments summarize minibatch gradients at earlier parameters.

### AI2. Coordinate scaling does not preserve an arbitrary subspace

For a residual map Q and diagonal scale D, define

```
[Q,D]g = Q(Dg) - D(Qg)
([Q,D]g)_k = sum_l Q_kl (D_l-D_k) g_l.
```

Constant D commutes with every Q. For any reference gradient G and moment M,

```
QDM = DQG + [Q,D]G + QD(M-G).
```

With RD1's actual current full-training partial readout derivative G, the
three terms isolate scaled projected CE forcing, diagonal mixing, and the
combined discrepancy from the saved moment. The last term includes encoder,
minibatch, temporal and momentum effects; the decomposition does not identify
them separately. In particular M-G is not just a momentum error or the current
encoder gradient. The historical direction is not a predicted next update.

The RD5 example makes the mixing term concrete: Q is orthogonal projection
perpendicular to (1,2), G=(-1/2,-1), D=(2/3,1/2). Then QG=0 but [Q,D]G has
squared norm 1/180. Lean proves this directly. The native test also checks
zero commutator under a constant diagonal scale.

### AI3. Bound the part controlled by CE and keep the remainder

For class-dependent D_c,k with D_c,k²<=dmax², finite sums give

```
||D Q G||_F² <= dmax² ||QG||_F²
             <= dmax² 2 L_train mean_all_training ||Qh_i||².
```

This uses every training row, as RD2 requires. Coordinatewise
(a+b+c)²<=3(a²+b²+c²) gives the sufficient adaptive-forcing bound

```
||QDM||_F² <= 3*(dmax² 2 L_train mean_all_training ||Qh_i||²
                 + ||[Q,D]G||_F² + ||QD(M-G)||_F²).
```

No commutator or discrepancy is assumed zero. Large inverse scales can
amplify the CE contribution, and cross terms can make the sum of individual
energies very conservative. The native audit retains those cross terms,
the direct three-term bound and the objective-based bound separately.

### AI4. Control the signed interaction, not just separate energies

Let J=<u,r>, where u=Qw and r=Qx; x is normalized query displacement
relative to its training center. For arbitrary new vectors define
du=u_new-u and dr=r_new-r. Bilinearity yields

```
J_new-J = <du,r> + <u,dr> + <du,dr>
|J_new-J| <= ||du||||r|| + ||u||||dr|| + ||du||||dr||.
```

The mixed term is retained. An old bound |J|<=B transfers to the new state
with this explicit drift allowance. This is a retrospective finite-change
statement; it does not identify a 5000-step checkpoint interval with an update.
Changing Q is already included in du and dr.

A nonzero witness has u=r=(1,0), u_new=(0,2), r_new=(3,0). The interaction
falls from 1 to 0 while the norm product rises from 1 to 6. Signed changes
are -1, +2 and -2. This shows why tracking signs and cancellation can be
more informative than requiring both residual norms to contract.

For computed Q, <Qw,Qx> differs from the source residual <w,Qx> by a
projection defect unless the required projection identity holds. The audit
retains that defect. Neither interaction alone replaces ST2's certificate.

### AI5. A sufficient condition for an actual moving adaptive update

Write a=1-eta*rho, f=DM, w'=a w-eta f, u=Qw, r=Qx, u'=Q'w', r'=Q'x'.
This buffered direction describes one readout row. For a class contrast
w=E_c-E_target, use f=D_c M_c-D_target M_target: the two preconditioners
need not coincide. Lean proves this actual contrast update and its moving
interaction bound separately; it never replaces f by D*(M_c-M_target)
without a shared-scale assumption.
The readout residual satisfies

```
u' = a*u - eta*Qf + b,          b=(Q'-Q)w'
<u',r'> = a<u,r> - eta<Qf,r> + <b,r> + <u',r'-r>.
```

Consequently,

```
|<u',r'>| <= |a| |<u,r>| + Gamma
Gamma = |eta| ||Qf||||r|| + ||b||||r|| + ||u'||||r'-r||.
```

Lean derives this bound, instantiates it with the actual buffered Adam
equations, and proves strict contraction when
Gamma < (1-|a|)|<u,r>|. The hypothesis controls concrete forcing and drift
terms, not the desired new interaction. A concrete example has Q=Q'=1,
a=eta=1/2, w=4, f=1, x=x'=3: nonzero forcing and residuals satisfy it.
AI3 supplies control for one part of Qf; it supplies no control over the
other two parts or over feature/basis movement. The condition is therefore
sufficient, not a claim of optimizer discovery or a consequence of CE alone.

| Claim | Lean declaration in `GradientEvidence` |
| --- | --- |
| Source update and projection | `bufferedAdamUpdate_factor`, `projected_bufferedAdamUpdate` |
| Mixing and three forcing terms | `diagonalCommutator_eq_sum`, `adaptive_forcing_decomposition`, `diagonal_mixing_witness` |
| Loss-controlled contribution and full bound | `scaled_readoutGradient_energy_le`, `adaptive_forcing_energy_le` |
| Exact interaction change and drift control | `interaction_change`, `interaction_change_abs_le`, `interaction_bound_transport` |
| Cancellation witness | `interaction_cancellation_witness` |
| Actual update with moving geometry | `moving_update_interaction`, `buffered_interaction_le` |
| Row-specific preconditioners in actual class contrasts | `buffered_contrast_eq_directionStep`, `buffered_contrast_interaction_le` |
| Sufficient strict contraction | `moving_update_interaction_contracts` |

## Actual single-update evidence

Task #186 implements the [fixed native audit](../../experiments/single_update/README.md)
of AI1–AI5 on all 24 original checkpoints. Each next scheduled AdamW update
replays bitwise against the pinned source trainer. All 48 independent scalar
directional derivative checks and 9360 projected decompositions pass; the
original 73 input files remain unchanged. Eight endpoint probes extend the
sampler by one post-budget step and do not change the original ranking.

At the old parameters let R be the full-training readout embedding gradient,
F the full-training total embedding gradient and B the next-minibatch total
gradient. M and D are taken from the new moment buffers, with bias correction.
The measured decomposition is

```
M = R + (F-R) + (B-F) + (M-B)
QDM = DQR + (QDR-DQR) + QD(F-R) + QD(B-F) + QD(M-B).
```

This distinguishes the tied encoder, sampling and moment-memory terms
without treating them as independent causal interventions. It retains each
class row's own D before subtraction. For the plain-product post-budget
probes at rank-32 orbit bases, individual forcing energies sum to
57923.68/4407.75, whereas actual energies are 3786.84/885.70. Cross terms
contribute -54136.84/-3522.05. These sums run over 33 different fitted bases;
they are not a single global model energy. A sum of separate norm bounds
discards substantial cancellation.

For actual FP32 movement write `w'=a*w-eta*f+rho`, retaining the measured
rounding remainder rho. The audit evaluates AI5's interaction identity with
that extra projected remainder, and further splits
`r'-r = Q(x'-x)+(Q'-Q)x'`. Both x and x' are normalized, training-centered
feature displacements. Feature and basis contributions can be much larger
than individual optimizer terms, but their absolute magnitudes do not add.

The margin diagnostic keeps the signed training-affine reconstruction
`s=<w,mu+(I-Q)(x-mu)>`, interaction `J=<Qw,Q(x-mu)>`, projection defect
`e=<w,Q(x-mu)>-J` and measured executed-logit discrepancy. Each executed
competitor gap is reconstructed as `s+J+e+execution`. A sufficient condition
uses an absolute residual/defect/execution allowance for every competitor.
This is a retrospective measurement of the actual interaction, not a new
training-only certificate or an improvement to ST's norm-bound coverage.

Both plain-product endpoint probes increase mean validation margin, while
validation CE rises and correctness stays 4689/4705 and 4690/4705. On
1082/1163 queries margin improves despite a growing maximum residual budget.
Across the cohort, 17/24 probes lower training CE and five of those increase
rank-32 orbit validation mean absolute interaction; among trained states
the counts are 9/16 and 1/9. Five trained probes improve mean validation
margin while worsening validation CE.

The remaining theoretical obligation is control of worst competitor margins
and difficult-query losses under joint forcing and moving features/bases.
Residual contraction and average-margin improvement alone do not supply it.
This evidence motivates that extension; it introduces no additional Lean
claim of convergence or discovery and leaves complex multiplication unchanged.

## Joint difficult-query control

Task #187 extends AI's interaction analysis to the actual competing-class
score gaps and individual query losses. Source equations are the native
coordinate AdamW update at 14cff3b5 and the finite-class softmax CE in
`Transformer.Landscape.Neural.Loss`. The mathematical statements below are
exact-real identities and sufficient bounds; their numerical audit retains
explicit parameter rounding and executed-logit discrepancies.

### JC1. Keep signed parameter and feature changes together

For a competitor-minus-target readout w and a normalized feature x, write
the new readout as `w'=w+p+rho` and the new feature as `x'=x+dx`. The intended
increment p and rounding remainder rho are distinct. Then

```
L = <p,x> + <w,dx>
gap' - gap = L + <p,dx> + <rho,x'> + (e'-e)
R = ||p|| ||dx|| + ||rho|| ||x'|| + |e'-e|
gap + L - R <= gap' <= gap + L + R.
```

The two first-order terms are combined before taking any absolute value.
Cauchy–Schwarz bounds the mixed term and rounding term; a triangle inequality
adds the explicit execution change. The separated allowance
`S=||p||||x||+||w||||dx||+R` also bounds the full change in both directions.
The signed upper allowance L+R is at most S. This does not assert small
feature movement or small forcing as a consequence of training CE.

The nonzero scalar witness w=2, p=1, x=2, dx=-1 has L=0, R=1 and S=5.
The score changes by -1 through the mixed term. Thus omitting that term is
incorrect even when the two first-order contributions cancel exactly.

### JC2. Control every competing class under the actual update

For each competitor c and target t, use w=E_c-E_t and its own increment
contrast. Define `U_c=gap_c+L_c+R_c` and `V_c=gap_c+L_c-R_c` from JC1.
Every new gap lies in `[V_c,U_c]`. Hence if all U_c are at most -m, the
new target has margin at least m against every competitor. In particular,
all U_c<0 suffice for a strict correct prediction. Checking only the old
worst competitor would miss changes in its identity.

The source Adam increment is proved to be

```
p_c = -eta * (decay*E_c + D_c*M_c)
p_c-p_t = -eta * (decay*(E_c-E_t) + D_c*M_c-D_t*M_t).
```

M and D come from the already updated moment buffers, retaining bias
correction and epsilon. Each class has its own diagonal scale. Adding p
reconstructs `bufferedAdamUpdate`; adding rho connects the joint bound to
the actual rounded parameters. A concrete margin witness has nonzero
moments, decay, parameter movement and feature movement, with different
preconditioners for the two class rows.

### JC3. Control actual query loss with old probability weights

For arbitrary old/new logits z,z', define
`delta_c=(z'_c-z'_t)-(z_c-z_t)` and the actual old softmax p_c. Lean proves

```
CE(z',t)-CE(z,t) = log(sum_c p_c * exp(delta_c)).
```

The target increment is zero. If every delta_c is bounded by B_c, the
same expression with B gives an upper bound on loss change. A weighted
exponential moment at most one is sufficient for loss nonincrease. This
condition permits signed compensation across classes but does not replace
their old probabilities by uniform weights or replace CE by a mean margin.

Taking `B_c=L_c+R_c` instantiates this bound with the actual joint step.
Equivalently, the target contributes its fixed zero gap, and the wrong-class
upper/lower gaps U/V give upper/lower CE bounds through log-sum-exp. Both
directions are proved, allowing detection of worsening as well as improving
query losses. Parameter and feature contributions remain signed inside L.

### JC4. Preserve difficult queries and retain the limitation

Independently valid query loss ceilings average over any fixed subset.
The native protocol freezes the lowest-margin decile of each old split
before evaluating the new state, and retains old errors as another subset.
It does not select difficult rows using the new outcome.

A proved two-query binary counterexample uses old margins `(log 8,log 8)`
and new margins `(log 256,log 2)`. Both new predictions remain strictly
correct and the mean margin increases, but mean actual CE increases:
the old loss product is `(9/8)^2=81/64`, whereas the new one is
`(257/256)*(3/2)=771/512 > 81/64`. The second query's margin deteriorates.
This formally rules out an average-margin argument for loss improvement.

These are conditional local guarantees. The audit measures dx and numerical
execution error after the update. Deriving a useful bound on feature movement
from pre-update training quantities, and validating it on further seeds/tasks,
remain separate obligations. No convergence or optimizer-discovery theorem
is inferred. The new bounds use ambient normalized features and need no PCA
basis, avoiding a dependence on how fitted subspace coordinates move.

| Claim | Lean declarations in `GradientEvidence` |
| --- | --- |
| Exact finite step and joint interval | `score_step_identity`, `score_step_error_bound`, `score_step_interval` |
| Comparison with separated norms | `joint_drift_abs_le`, `score_step_separate_bound`, `joint_upper_le_separate` |
| Nonzero mixed-term witness | `joint_score_cancellation_witness` |
| All-competitor bounds and margins | `gap_step_error_bound`, `new_gap_le_joint_upper`, `joint_lower_le_new_gap`, `all_competitor_margin`, `strict_winner_of_joint_step` |
| Actual Adam and row-specific scales | `adam_increment_update`, `adam_increment_contrast`, `adam_gap_le_joint_upper`, `adam_margin_of_joint_condition` |
| Exact probability-weighted CE change | `crossEntropy_relative_change`, `loss_change_le_relative_bound` |
| Joint query and subset loss bounds | `loss_change_le_joint_step`, `loss_nonincrease_of_joint_step`, `crossEntropy_le_joint_step`, `crossEntropy_ge_joint_step`, `meanCrossEntropy_le_joint_step` |
| Mean-margin counterexample | `mean_margin_loss_counterexample` |

### Native audit of JC1–JC4

The [frozen #187 extension](../../experiments/single_update/MARGIN_PROTOCOL.md)
passes all 24 original probes and reproduces every #186 new-state hash.
Across 225816 query updates and 21678336 competing-class pairs, all joint
and separated gap, margin and CE intervals hold, with zero false certificates.
Maximum scaled identity and weighted-CE errors are below 4.39e-15 and
2.79e-15 respectively. The earlier source, derivative and geometry checks
also remain green; original checkpoint inputs are unchanged.

For the 16 trained-state validation probes, joint conditions certify the
direction of CE change on 74677/75280 observations; 603 remain inconclusive.
The fixed old-margin bottom deciles give 7489/7536, with 47 inconclusive.
At both plain-product post-budget endpoints, even the fixed difficult set's
mean margin rises while its mean CE increases (0.172718 → 0.174981 and
0.167636 → 0.168661). Averaging the lower CE bounds certifies both increases.
Thus an average-margin argument remains insufficient after restricting to
difficult queries; JC3's individual probability-weighted control is material.

The [full authored findings](../../experiments/single_update/README.md#joint-margins-and-difficult-query-losses)
retain every arm, seed and inconclusive case. The good coverage uses observed
feature displacement and is conditional, not a pre-update prediction. A
useful bound on that displacement from training quantities remains open.

## Feature forecasts

Task #188 replaces JC's observed feature displacement with a forecast made
from the old parameters and the proposed rounded Adam step. The source is
the same native trainer (14cff3b5); the
[frozen protocol](../../experiments/single_update/FORECAST_PROTOCOL.md) specifies
every propagation rule and numerical allowance before measuring outcomes.

### FF1. Propagate the old tangent and a finite error ball

At each node let a be the old value, v its old directional derivative along
the proposed parameter step, and r a radius for the unknown true new value
around a+v. The embedding gather is affine, so its radius starts at zero.
Residual addition adds radii. For a linear layer with old matrix A and
increment D, the predicted tangent is Av+Da and

```
(A+D)y - (Aa+Av+Da) = (A+D)(y-a-v) + Dv
radius_out <= ||A+D|| * r + ||Dv||.
```

The mixed term Dv is known before evaluating the future network. Native
code uses the Frobenius norm as a conservative operator bound. For bilinear
products, expand around the two nominal centers, retaining the known product
of tangents, both center/error products and the error/error product. Coordinate
and complex block maximum norms bound those linear factors; the complex
source keeps its 1/sqrt(2) multiplier. No product term is silently discarded.

For a nonlinear map f, a proposed tangent u and an anchored sensitivity L at
a+v give the finite bound

```
||f(y)-(f(a)+u)|| <= L*r + ||f(a+v)-f(a)-u||.
```

The nominal defect on the right is computable from a and v. Its value is
used only in the scalar error budget; the next node receives f(a) and u,
not f(a+v). Thus the center remains the old-network tangent. The certificate
is valid for any u; derivative correctness is independently checked against
central differences of the scalar source oracle.

### FF2. Use anchored nonlinear sensitivities

Define `softNorm(eps,x)=sqrt(||x||^2+eps)` and
`softNormalize(eps,x)=x/softNorm(eps,x)`. Positive epsilon gives

```
|softNorm(eps,y)-softNorm(eps,x)| <= ||y-x||
||softNormalize(eps,y)-softNormalize(eps,x)||
    <= 2/softNorm(eps,x) * ||y-x||.
```

The first bound follows by factoring the difference of squared regularized
lengths and using the reverse triangle inequality. For the second, split
the vector difference and the reciprocal-denominator difference; the
regularized denominator dominates the other vector's norm. Scaling by
sqrt(width) and using eps=width*epsilon gives the source RMS coefficient
2/s(x). Anchor x at the known nominal input, avoiding the global inverse
sqrt(epsilon) factor. This works without an assumed small relative step.

The radial complex gate uses the first bound with two real coordinates.
The sigmoid has global derivative magnitude at most 1/4; its nominal defect
therefore transports incoming error with coefficient 1/4. Real SwiGLU,
complex diagonal, radial product and plain product retain their source
equations and every normalization site.

### FF3. Read out a forecast ball

The proposed rounded readout W is known before future feature evaluation.
If center z has certified feature radius r and each executed readout logit
has an additional error at most delta, then the competitor gap lies within

```
<W_c-W_t,z> +/- (||W_c-W_t||*r + 2*delta).
```

Checking every upper gap controls worst-competitor margins even when that
competitor changes. Log-sum-exp of lower/upper gaps encloses actual query
CE; independently valid upper bounds average over fixed subsets. Errors and
inconclusive conditions remain. These are actual-loss bounds, not an
inference from increasing mean margins.

The native forecast adds a fixed 1e-3 Euclidean feature allowance and 1e-3
per-class readout-logit allowance. They are separately audited numerical
assumptions, not proved CUDA error bounds. Forecasts are persisted before
the candidate network is evaluated. The real graph and its tangent are
checked against source computations; Lean proves the propagation lemmas
and feature-to-outcome bridge, not a verified implementation of the Rust/CUDA
graph. This is local control given a proposed step, not optimizer discovery,
convergence or a statistical generalization theorem.

| Claim | Lean declarations in `GradientEvidence` |
| --- | --- |
| Finite nominal-defect transport | `forecast_transport`, `linear_forecast`, `residual_forecast` |
| Mixed product terms | `product_forecast_identity`, `product_forecast_bound`, `product_forecast_mixed_witness` |
| Source sigmoid | `sigmoid_quarter_lipschitz`, `sigmoid_forecast` |
| Radial and normalization sensitivity | `softNorm_error`, `softNorm_forecast`, `softNormalize_error`, `scaled_softNormalize_error`, `softNormalize_forecast` |
| Explicit execution allowance | `forecast_execution_ball` |
| All-competitor control | `forecast_contrast_error`, `forecast_gap_interval`, `forecast_margin` |
| Actual query/subset loss | `forecast_loss_upper`, `forecast_loss_lower`, `forecast_mean_loss_upper` |

### Native audit of FF1–FF3

All 24 frozen forecasts pass, with unchanged #186 state hashes and original
inputs. All 225816 feature balls and 21678336 competitor bounds enclose the
later outcomes. Fixed FP32 allowances have zero violations; independent
scalar tangent checks have maximum scaled error 5.47e-9. The native report
retains all per-query errors, radii, intervals and frozen feature centers.

Forecast correctness coverage is 68040/75280 (90.38%) on trained validation
observations, but no loss-change direction is certified anywhere in the
24-case cohort. Mean validation margin interval width is 4.947498, versus
0.001595 for JC's retrospective interval. Plain-product endpoint ratios of
mean radius to mean actual forecast error are 109.10 and 196.37. The
[authored findings](../../experiments/single_update/README.md#feature-forecasts-before-the-updated-forward-pass)
show all arms and seeds. Future feature measurements are no longer needed
for these bounds, but repeated scalar-ball propagation is too conservative
for loss control. The next obligation is directional error transport through
the actual block and readout, not an architecture change or a fitted allowance.

## Forecast tightening

Task #196 keeps FF1–FF3's old values, directional tangents and explicit mixed
terms. The question is whether sharper, computable error bounds can close the
gap between 90.38% trained-validation certificate coverage and the 99.56%
observed correctness of the same updates. These are different quantities:
neither a certificate failure nor an inconclusive interval is a wrong answer.
The [frozen protocol](../../experiments/forecast_tightening/PROTOCOL.md) fixes the
24 source states and all ablations before reading their candidate outcomes.

### FT1. A regularized secant retains radial cancellation

For a real inner-product space, let

\[
s_x=\sqrt{\|x\|^2+\varepsilon},\qquad
F(x)=x/s_x,\qquad \varepsilon>0.
\]

Expanding the two squared distances gives the exact identity

\[
\|F(y)-F(x)\|^2(s_xs_y)^2
 +(s_x-s_y)^2(s_xs_y+\varepsilon)
 =\|y-x\|^2s_xs_y.
\tag{FT1}
\]

Indeed, multiply the normalized difference by the positive product
\(s_xs_y\), expand \(\|s_xy-s_yx\|^2\), and substitute
\(\|x\|^2=s_x^2-\varepsilon\) and its counterpart for y. The second
term is nonnegative, so the secant coefficient is at most
\(1/\sqrt{s_xs_y}\). If a known ball gives \(\|y-x\|\le r<s_x\),
the already proved regularized reverse triangle bound gives
\(s_y\ge s_x-r>0\). Consequently,

\[
\|F(y)-F(x)\|\le {r\over\sqrt{s_x(s_x-r)}}.
\]

For source RMS at width n, set the regularization above to n times the
source epsilon and multiply F by \(\sqrt n\). Use the smaller of this
coefficient and FF2's \(2\sqrt n/s_x\). If \(r\ge s_x\), retain
the old global bound; no unknown future denominator is substituted. At a
forecast center \(x=a+v\), add the unchanged nominal nonlinear defect
\(\|F(a+v)-(F(a)+u)\|\). Computing this defect does not make F(a+v)
the next layer's feature: the center remains the old value plus tangent.
The Lean witnesses include opposite nonzero vectors and a nonzero radius.

### FT2. Gram energy bounds linear amplification

For any finite real rectangular matrix M with its Euclidean operator norm,

\[
\|M\|_{\rm op}
 \le \sqrt{\|M^\top M\|_F}
 =\left(\sum_{ij}(M^\top M)_{ij}^2\right)^{1/4}.
\tag{FT2}
\]

Rowwise Cauchy–Schwarz proves \(\|G\|_{\rm op}\le\|G\|_F\).
The Euclidean C-star identity
\(\|M^\top M\|_{\rm op}=\|M\|_{\rm op}^2\) then proves FT2.
Applying the same argument to the transpose gives the row Gram version.
The implementation chooses the smaller Gram dimension and takes the minimum
with the independent Frobenius upper bound. There is no estimated singular
vector, spectral flatness assumption or trained parameter change.

For the updated matrix M=A+D, FF1's radius becomes

\[
B r+\|Dv\|,\qquad B\ge\|A+D\|_{\rm op}.
\]

The actual mixed vector Dv remains. Lean also proves the conditional
statement that any U above the exact Gram squared Frobenius norm gives the
bound \(U^{1/4}\). This exposes the numerical obligation. The Rust routine
enlarges every FP64 product, sum and square root outward, including signed
Gram dot products; the whole Rust/BLAS/CUDA program is not formally verified.
Nonzero square and rectangular examples inhabit the Lean statements; native
tests also cover identity, rank-one and transposed layouts.

### FT3. Preserve signed contrasts across the final RMS

Let \(\|y-x\|\le r\) describe the feature before the final normalization,
and let w be one updated readout contrast. Cauchy–Schwarz gives
\(a\le\langle w,y\rangle\le b\), where
\(a=\langle w,x\rangle-\|w\|r\) and
\(b=\langle w,x\rangle+\|w\|r\). Independently,

\[
0<l=\sqrt{\max(0,\|x\|-r)^2+\varepsilon}
 \le s_y\le
u=\sqrt{(\|x\|+r)^2+\varepsilon}.
\]

The signed quotient interval is

\[
\begin{split}
L&=\begin{cases}a/u&a\ge0,\\a/l&a<0,\end{cases}\\
U&=\begin{cases}b/l&b\ge0,\\b/u&b<0,\end{cases}\qquad
L\le\langle w,F(y)\rangle\le U.
\end{split}
\tag{FT3}
\]

The sign cases follow by multiplying through positive denominators; using
the same denominator endpoint for both signs would be wrong. Regularization
keeps l positive even if the ball contains the origin. Source RMS contributes
the factor \(\sqrt n\), or equivalently divides the denominator interval
by \(\sqrt n\). The native experiment then adds the independently audited
execution allowance \(\|w\|10^{-3}+2\cdot10^{-3}\).

Intersect FT3 with the independent combined final-feature interval by taking
the maximum lower endpoint and minimum upper endpoint. Both components are
audited separately. A strictly negative upper competitor gap certifies that
the target beats that competitor; applying the existing all-competitor
margin and log-sum-exp bounds gives correctness and CE intervals. No observed
future feature, score or error selects a radius, denominator or class direction.
Strictness uses only the old score and frozen endpoints; future scores are
used to check the resulting certificates.

`ForecastSecant.lean`, `ForecastGram.lean` and `ForecastDirectional.lean`
contain 18 proved theorems, each conditional theorem with an explicit
satisfying example. `scripts/ForecastTighteningAxioms.lean` checks all of them.
These extend the forecast bounds, not the convergence theory: they neither
raise model accuracy nor establish an efficient forward-only training method.
The [experiment findings](../../experiments/forecast_tightening/README.md) retain
all predeclared ablations, execution checks and preparation costs.

## Training-to-forecast bridge

Task #197 joins ST1–ST3's signed training reconstruction to FT1–FT3's
pre-update forecasts. The [frozen protocol](../../experiments/transport_forecast/PROTOCOL.md)
uses the actual plain complex-product source on `(a+b)^3 mod 97`. This is
the next conditional connection in this development, not a replacement for
the failed componentwise spectral diagnostic or a solution of `a^3+ab`.

### TF1. Retain training geometry across the proposed step

Fix old normalized training states `T`, their center `mu`, old query state
`x`, coefficients `a`, and a computed basis `B`. For the competitor-minus-target
readout `w`, ST gives an old signed prediction `P=transportPrediction` and
directional remainder `R=directionalResidualBound` with

```
<w,x> <= P + R.
```

Neither the basis nor the coefficients need be exactly orthogonal. The
existing defect term stays in `R`. Keep this old geometry fixed across one
actual parameter update. Let `w'` be the known rounded updated readout,
`v` the old-network directional feature tangent, and suppose FT propagation
supplies `||x'-(x+v)|| <= r`, with `r>=0`. Then the exact finite identity is

```
<w',x'> = <w,x> + Delta + <w',x'-(x+v)>
Delta = <w'-w,x> + <w',v>.
```

The second term includes `<w'-w,v>`; it is not the first-order readout
`<w,v>` alone. Cauchy–Schwarz gives the complete bridge

```
<w',x'> <= P + R + Delta + ||w'||*r.
```

The finite-coordinate Lean theorem takes the equivalent squared-ball premise.
The native implementation uses FT1/FT2's combined radius and retains the
fixed feature/logit execution allowances. A two-coordinate witness has
training states `(1/2,0),(3/2,0)`, center `(1,0)`, query `(3,1/4)`, old
readout `(-2,1)`, new readout `(-3,0)`, tangent `(1/4,0)`, and future state
`(27/8,1/4)`. Both the old residual and forecast remainder are nonzero.
Its future score is `-81/8`, below the bridge endpoint `-75/8`.

### TF2. Transport the directional endpoint without hiding its slack

If FT3 gives an independently valid future gap endpoint `U`, its signed
drift allowance is `U-<w,x>`. The corresponding training bridge is

```
U_bridge = P + R + (U-<w,x>)
         = U + (P+R-<w,x>).
```

Lean proves both the bound and `U<=U_bridge`. Training reconstruction adds
nonnegative slack; this composition cannot improve the independent forecast's
coverage. Its value is testing how much of that coverage can also carry a
training-to-query explanation. Reporting `min(U,U_bridge)` as the bridge
would erase that question. The diagnostic reports both separately.

### TF3. Preserve every competitor and the actual loss

Apply TF1 to every class contrast with the same actual future state. If
executed scores differ from their exact linear readouts by at most `eta`,
add `2*eta` to each endpoint. Negative upper endpoints for all wrong classes
give a strict correct winner, and endpoints at most `-m` give margin `m`.
The existing odds-envelope lemmas give probability floor
`1/(1+sum exp(U_c))` and CE ceiling `log(1+sum exp(U_c))`, summing wrong
classes. The worst competitor may change. These are per-query sufficient
conditions; no average-margin argument or future error measurement enters
their construction.

### TF4. Actual source and temporal scope

`ComplexProduct.executed_gap_le_transport_forecast` instantiates TF1/TF3
with the old and new `readoutState`, the actual tied embeddings and all four
source normalizations. The endpoint adapter instantiates TF2. Feature-ball
and numerical-execution premises remain explicit, just as in FF and FT;
this does not formally verify the full Rust/BLAS/CUDA program.

The execution adapter also separates a squared feature discrepancy `delta^2`
from per-logit arithmetic error `eta`. It derives the native contrast allowance
`||w'||*delta+2*eta`, rather than silently treating the feature error as a
per-logit error. The feature radius becomes `r+delta`; both nonnegativity and
both error premises have explicit witnesses.

The frozen native study replays both original seeds, preserving their full
parameter/moment hashes at 0, 5000 and 10000, and inserts 26 declared probes
through the trajectory. Training-only answer-orbit bases transfer observed
labels via the proved cubic routes. Rank 32 is primary; rank 0 and rank 128
are center and reconstruction-limit controls. Every class endpoint is
persisted before the updated-network evaluation. Existing seeds and the
previously examined validation split provide a temporal diagnostic, not an
independent statistical generalization test. A correctness certificate cannot
precede the correctness it certifies. The test asks whether useful coverage
appears before the first sampled 99% accuracy, with no eventual-convergence
claim. The original flat-spectrum learning obstruction remains open.

| Claim | Lean declaration |
| --- | --- |
| Signed finite update with mixed term | `FeatureCodes.linearScore_forecast_identity` |
| Euclidean remainder and training composition | `FeatureCodes.linearScore_forecast_error_le`, `signed_transport_forecast_upper` |
| Directional endpoint and unavoidable old slack | `FeatureCodes.signed_transport_endpoint_upper`, `endpoint_le_transportEndpointUpper` |
| Nonzero update/residual witness | `FeatureCodes.transport_forecast_nonzero_witness` |
| Executed gap, margin and strict winner | `FeatureCodes.executed_gap_le_transportForecast`, `margin_of_transportForecast`, `strict_winner_of_transportForecast` |
| Actual probability and CE | `FeatureCodes.probability_ge_transportForecast`, `crossEntropy_le_transportForecast` |
| Complete source adapters | `ComplexProduct.executed_gap_le_transport_forecast`, `executed_gap_le_transport_endpoint` |
| Separate feature and arithmetic execution errors | `FeatureCodes.signed_transport_forecast_execution_upper`, `ComplexProduct.executed_gap_le_transport_forecast_with_feature_error` |
