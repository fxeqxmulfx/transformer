# Sequence models: capacity and learning bridges

Moved unchanged from [`../flat_spectrum.md`](../flat_spectrum.md) on 2026-11-19: the sections
*General positional models* through *Quantitative RoPE approximation*, and *Local attention
routing*. They ask what the actual normalized RoPE transformer can represent and what its
learner sees: general positional models (G1–G5), bridges from the matrix results to the actual
learner (R1, S1–S2, T1–T4, E1–E8), finite capacity and information limits (N1–N2, F1–F6),
prescribed and uniform RoPE (P1–P6, U1–U4), depth, width and precision (R1–R5), quantitative
RoPE approximation (A1–A6) and per-position retrieval (LR1–LR4). The Lean is in
`Transformer.SequenceModel` and `Transformer.CRASP`.

These labels are this file's; R1 names two items, as it did in the note. The rounding
certificates B1–B9 that *Quantitative RoPE approximation* calls "below" are in
[`execution.md`](execution.md). Theorem A, *Actual network expressivity for every exponent* and
the other earlier labels are in [`../flat_spectrum.md`](../flat_spectrum.md).

## General positional models

The next extension follows the user's request for arbitrary rotations, more
than two input variables, other positional mixers and attention. The supplied
normalization reference is
`/home/misha/python_projects/gpt/src/diff_gpt/domain/model`, at commit
`b98c665bd8faa30729fd368d257a13b6d58712a1`. The five inspected source-file hashes
and the exact scope are retained in `review/sequence_model_source.json`.
All results below concern real arithmetic at finite architecture sizes.

**G1, arbitrary inputs and full predictors.** `SequenceModel` takes an input
type X, a finite set of scalar trainable coordinates I, a finite label type V,
and the complete score function `forward : R^I -> X -> V -> R`. X can be
`Fin N -> ZMod P`, a token tuple, or a variable-length sequence. Sample IDs
can be independent of their data values, retaining multiplicity of repeated
examples and losses at distinct token positions. The source Schedule-Free
recurrence is stated independently, then identified with F1--F4's run.
Its finite matrix-product/normalized-gradient decomposition and zero-gradient
parameter-plane invariant therefore apply without a two-operand assumption.
The nonlinear forward is evaluated after adding the parameter response; logits
and attention probabilities are not assumed to depend linearly on it.

The source identities hold for arbitrary update oracles. The sampled classical
loss gradient is justified at differentiable queries by an explicit theorem
with that hypothesis. An arbitrary forward is not declared differentiable by
definition, and the clipped XSA normalizer does not acquire a global smoothness
guarantee. At a clipping boundary, relating a chosen autodiff convention to an
update oracle is an additional step, not an implicit assumption of this theory.

**G2, arbitrary rotations and positional mixing.** The rotary operator has an
arbitrary real angle for every channel pair. Lean proves angle addition under
composition, the inverse rotation, preservation of squared length and norm,
and the relative-position dot-product identity for every frequency vector:

```
<R(s*omega) x, R(t*omega) y> = <x, R((t-s)*omega) y>.
```

The half-split layout agrees with the existing actual RoPE operator. The
geometric frequency grid is an instance, and pi/2 is another instance; neither
is required by the general theorem. A positional mixer can also transform
each token embedding by an arbitrary position-dependent map before summing.
The ensuing predictor retains the shared embedding, input/final RMSNorm and
the full Peri-RMSNorm SwiGLU stack. The two-input zero/pi-over-two instance
has exactly the previous Rotate90 scores. For arbitrary fixed rotary phase
tables and positive epsilon, the pooled forward is analytic jointly in all
its trainable weights, at every finite tuple length, width and depth.

**G3, information lost in pooling.** An invertible rotation of each vector
does not make a sum of rotated vectors injective. If the positional operators
repeat under a permutation, the pooled representation is invariant under that
permutation, for every embedding. In the special sequence of fixed quarter
turns, positions 0 and 4 use the same rotation. Consequently
`[a,b,c,d,e]` and `[e,b,c,d,a]` have identical pooled representations and all
downstream logits. Lean proves that no finite width, depth or weights of that
special pooled network can strictly predict the first of five binary variables
on every input. This is a limitation of the repeated-phase pooling rule;
the theorem does not impose it on arbitrary RoPE or attention operating on
the unpooled token stream. It supplies no universal variable-count threshold
at which attention becomes necessary.

**G4, the supplied normalized attention model.** `SequenceModel.Attention`
formalizes the supplied diff_gpt no-cache training forward with the following
operations in their source order:

1. Normalize the learned token embeddings by parameter-free RMSNorm.
2. In every block, normalize the attention input; project fused QKV; reshape
   heads; RMS-normalize each Q/K head; apply its specified rotary phases;
   use causal softmax with scale `1/sqrt(head_dim)` on unnormalized values.
3. Apply XSA with clipped L2 normalization of the own value, and explicitly
   zero the first position's attention output. Merge heads and project.
   RMS-normalize that branch output before adding it to the residual stream.
4. Normalize the updated stream; apply bias-free SwiGLU; normalize its output;
   add that branch to the residual stream.
5. Apply final RMSNorm and the original shared embedding as tied readout.

RMS epsilon remains explicit: it is 2^-23 for float32 in the supplied source.
XSA's L2 epsilon is separately 1e-12. The proof does not replace either
normalizer by an exactly unit-length vector. RoPE can be disabled independently
per layer by zero angles, or the forward can take arbitrary angle tables.
The source uses hidden size H=(8W+1)//3; other finite H are a stated extension.
The coordinate encoder and decoder are inverse and cover every source weight,
with parameter count `V*W + D*(4*W^2 + 3*W*H)`. This includes embedding, QKV,
attention output and all three SwiGLU matrices; there are no learned norm
gains, head temperatures or biases. The Schedule-Free theorems now instantiate
this complete model for arbitrary finite input length. KV-cache execution and
floating-point rounding are outside the formal forward's scope. The mathematical
forward receives its phase table directly; correspondence with Python execution
uses positions within the configured block_size. Larger finite contexts are
a mathematical extension or require correspondingly larger source tables.

**G5, normalization changes the quantitative theory.** With positive RMS
epsilon and positive width, Lean proves, for every choice of weights and phases:

```
|head_score| <= sqrt(d_head)
||block(x)_i|| <= ||x_i|| + 2*sqrt(W)
||hidden_L(i)|| <= (1 + 2*L)*sqrt(W)
|logit(i,v)| <= sqrt(W)*||E[v]||.
```

Both normalized branch outputs are responsible for the residual bound; it
does not require a bound on the attention or FFN matrices. The final logit
bound still contains the tied embedding row norm. From the actual normalized
Q/K score bound and the positive causal softmax denominator, every attention
weight with N visible positions satisfies

```
attention(i,j) <= exp(2*sqrt(d_head)) / N,   N = i+1.
```

The bound can exceed one for short contexts. It limits concentration of one
attention row; it does not bound the computational power of the entire stack
or prove failure on all tasks with many variables. Conversely, these bounds
and the source-response identity do not guarantee learning. The static
flat-spectrum/nuclear-norm surrogate still does not describe the complete
normalized nonlinear network's optimizer or generalization.

Source correspondence was checked with `review/sequence_model_source_check.py`
against the pinned Python files, using Python 3.15.2 and Torch 2.15.0+cu132.
Three CPU float64 cases cover width 8/16, depth 1/2/3, length 1/5/7 and all,
mixed or disabled RoPE layers. Logits agree within 2.78e-16 and every trainable
gradient within 6.89e-15. A CUDA float32 max-autotune/fullgraph/static case
agrees within 5.97e-8 in logits and 1.50e-6 in gradients. The equation checker
uses the source's rounded RoPE tables; it is a numerical source check, not
Lean extraction or a floating-point proof. No training outcome is inferred.
All configurations, parameter counts, errors and hashes are retained in
`review/sequence_model_source_check.json`.

Validation adds 37 proved theorems. The complete `lake build` passes (13096
jobs); individual `#print axioms` checks use only the three allowed standard
axioms. Audit remains at 36 existing `sorry`, with zero results resting on
them and zero extra axioms. The regenerated index has 3999 modules and 20682
theorems, with zero vacuous or placeholder declarations; `./run.py forbidden`
passes. Every new module is reachable from `Transformer.lean`.

The 40 lab tests selected by `./run.py test rotary compiled schedule_free`
pass on an unchanged retry after a CUDA graph-capture error in the existing
power-division resume test. The following 31 optimizer tests report two
failures in the float32/float64 Schedule-Free Muon allocation subtests. Both
failures reproduce on unchanged main `2006c06b` with
`./run.py test schedule_free_warmed_cuda_steps_allocate_no_device_storage`:
each records 75 positive allocations of 8519680 bytes. The AdamW allocation
subtests pass, and `./run.py test schedule_free_adamw` separately passes all
six selected tests. No production Python or optimizer code changes here.
The unsuccessful attempts and final checks are retained in the root board's
`.kanban/artifacts/122/`; the broader Python command is not reported green.

## Transformer learning bridges

This extension addresses the review at `cabf2119`. Static matrix completion,
representational capacity, and learning under the actual optimizer have
different hypotheses; the bridge results keep those hypotheses explicit.

**R1, actual local regularity.** With positive RMS and XSA epsilons, every
score of the supplied finite Transformer is analytic jointly in every
embedding and matrix entry at weights where each actual own-value norm
avoids the XSA clipping threshold. `Attention.Regular` spells out this
condition on the values computed from the actual normalized hidden states,
at each layer, position and head. This is a sufficient condition; it is not
claimed necessary at exceptional boundary points where cancellation occurs.

The proof follows scalar coordinates through fused QKV, head reshaping,
Q/K RMSNorm, arbitrary fixed rotary phases, causal softmax, clipped XSA,
attention projection, Peri-RMSNorm, SwiGLU, final RMSNorm and tied readout.
Both open sides of the clipping sphere are covered. Zero value vectors are
inside the clipped region and are included, not excluded as singular. The
causal softmax denominator is proved positive. Induction through all layers
then proves score analyticity and analyticity of the full sampled softmax
cross-entropy, with independent example IDs retaining repeated examples.

The condition is witnessed by zero weights at every positive finite source
configuration, context, rotary phase table and positive clipping epsilon.
The component proofs also have nonconstant and nonzero witnesses on both
sides of the clipping sphere. This discharges differentiability of the
actual loss at regular query points; it does not assign a classical gradient
to arbitrary clipping-boundary points or assert that a training trajectory
always avoids those points.

**S1, the rotary spectrum correction.** Norm preservation of individual
tokens does not preserve the spectrum of their Gram table when different
positions use different rotations. `SequenceModel.RotarySpectrum` proves
an exact counterexample using the genuine rotary operator: the two unit
basis vectors have Gram matrix `[[1,0],[0,1]]`; RoPE frequency pi/2 at
positions 0 and 1 changes it to `[[1,-1],[-1,1]]`. The independent vectors
`[1,1]` and `[1,-1]` change eigenvalues from `(1,1)` to `(0,2)`. Both Gram
diagonal entries remain one. The same lost mode persists after positive
common normalization/scaling. This corrects a possible extrapolation from
orthogonal invariance; it does not change the existing valid common-rotation
or relative-position identities. The counterexample concerns pre-softmax
Gram scores, not a claim that softmax has those same eigenvalues.

**T1--T2, the actual learning operator.** Let `F` be the complete finite
logit table, indexed by independent example IDs and classes, and let `ell`
be ordinary sampled softmax cross-entropy in those logit coordinates. At a
regular query `y`, the proved chain rule gives

```
g = J_train(y)^T r,      r = gradient ell(F_train(y)).
P_t = diag(1 / actual_Adam_denominator(t, history, g)).
K_eval,train = J_eval(x) P_t J_train(y)^T.
```

Here `y` is the source optimizer's query and `x` its evaluation average.
Every embedding and matrix coordinate participates in both Jacobians.
`P_t` includes the actual second-moment history, current-gradient update,
bias correction and optimizer epsilon. Positive optimizer epsilon proves
its nonnegative quadratic form. The same-Jacobian operator `J P_t J^T`
is positive semidefinite. Across splits or parameter points, symmetry and
positivity are not automatic. The full table can include held-out examples;
only the sampled IDs contribute to `ell` and its residual `r`.

**T3, finite updates, not just infinitesimal identities.** Write `c_t` for
the source averaging coefficient, `eta_t` for its rate and `lambda` for
decay. Its exact evaluation displacement is

```
delta = drift - c_t eta_t P_t g,
drift = c_t (base - x) - c_t eta_t lambda y.
predicted_change = J_eval(x) drift - c_t eta_t K_eval,train r.
```

The derivation requires nonzero source beta1 to recover the evaluation
average. It includes the source's drift and query decay; replacing it with
plain gradient descent would give a different formula. If the actual
network is regular along `[x,x+delta]` and
`||J_eval(w)-J_eval(x)|| <= C` on that segment, the mean-value theorem gives

```
||F_eval(x+delta) - F_eval(x) - predicted_change|| <= C ||delta||.
```

`Attention.actual_finite_prediction_error` discharges differentiability and
the gradient factorization for the full supplied forward from its actual
activation regularity. A predicted correct-class margin exceeding
`2 C ||delta||` therefore implies true finite-step strict classification on
every indexed example (`Transfer.source_step_correct`). A scalar version
proves true loss decrease when `D(delta) + C ||delta|| < 0`.
These conditions are certificates to establish or measure; no theorem says
they hold automatically throughout a run, or assigns a useful numerical C
solely from width, depth or flatness. The margin theorem is distribution-free
over its explicit finite row set, not a claim about unseen distributions.

**T4, alignment cannot be replaced by a spectrum.** Negating an evaluation
Jacobian leaves its entire self Gram operator `J P J^T` unchanged but
reverses `J P K^T` against a fixed training Jacobian K. Lean proves both
identities and opposite signs of residual transfer with the same self Gram.
Thus even knowing its whole self spectrum does not determine whether a
held-out residual improves. The necessary generalization of the static
spectral view is to retain cross-split alignment, changing Jacobians, the
actual preconditioner and finite nonlinear error. No equivalence between
nonlinear CE training and fixed-matrix nuclear-norm minimization is asserted.
The existing flat-spectrum completion theorems retain their original scope.

**E1--E8, constructive full-Transformer expressivity.** For every nonempty
finite vocabulary of size p and every binary function `f(a,b)`, the actual
source-shaped model has one fixed set of weights giving the correct label
a strictly larger logit than every rival, on every ordered pair. A sufficient
family is

```
W = 2 (p+1)^2 + 2 padding,
H = floor((8 W + 1)/3),
heads = 1,      depth = extraDepth + 1,      context length = 2.
```

Both padding and extraDepth are arbitrary naturals. This is a finite,
deliberately generous construction, not a minimal-width or efficient-learning
claim. RMS epsilon must be positive and XSA epsilon at most one; the actual
float32 source values `2^-23` and `1e-12` satisfy these conditions. All
embedding/final, Q/K and branch normalizers remain in their actual positions.

The proof has eight concrete steps, each connected to the actual forward:

1. **E1:** positive-epsilon RMSNorm has the explicit inverse
   `sqrt(W eps / (W-||v||^2)) v` for `||v||^2 < W`.
2. **E2:** two opposed actual SwiGLU gates compute one exact product, by
   `swiglu(g,u) - swiglu(-g,u) = g u`, with optional zero padding.
3. **E3:** normalized embeddings are unit token vectors. Zero Q/K gives
   uniform causal scores at any RoPE phases. With unit values, XSA keeps
   half of the earlier token when operands differ and returns zero when
   they agree. The current token's residual is retained.
4. **E4:** a fixed projection and its output RMSNorm produce the ordered
   code `x_ab = e_b + [a!=b] e'_a` in two disjoint banks. Its squared norm is
   one or two. Squares identify diagonal pairs, and context/current products
   identify unequal pairs, canceling the square contribution. Fixed output
   weights compensate both exact RMSNorm input scales.
5. **E5:** the source's fused-QKV cuts and head reshape are proved to realize
   that construction. Attention values, XSA and the context projection are
   active. The arbitrary RoPE table is present, but the chosen zero Q/K
   means this existence witness does not need position-dependent Q/K scores.
6. **E6:** the source H contains all `2(p+p^2)` opposed hidden features;
   residual width contains both token banks.
7. **E7:** the FFN interpolates the inverse-normalized displacement
   `e_f(a,b) - x_ab`, whose squared norm is at most three. Its output RMSNorm
   and residual therefore give exactly `e_f(a,b)`.
8. **E8:** extra blocks are identity residuals. The final normalizer and
   tied raw embedding give logits `L * 1[q=f(a,b)]` with `L>0`.

The main theorem is `Attention.Expressivity.exists_binary_classifier`.
`exists_power_division_classifier` instantiates it with `(a/b)^k` over
`ZMod p`, for every prime p and every natural k. It covers zero numerators,
exponent zero and all nonzero denominators; its stronger statement also
specifies Lean's totalized value at denominator zero. The weights depend
on the task, never on the queried operands. Concrete Lean examples include
p=2,3,97,193, exponents 0,5,17, varied depth and padding. Neither an infinite
width nor an infinite depth limit is needed.

The forward, regularity and learning results cover arbitrary finite context
lengths, heads, widths and depths. The constructive arbitrary-task theorem
above specifically covers binary tasks with one head and two token positions.
The binary theorem alone does not prove arbitrary-arity universality. The
F1--F6 extension below supplies a finite-arity theorem with a different,
explicit readout protocol and constructed active rotary phases. Neither
construction requires a learned model to use its particular circuit or
predicts the required training set.

**S2, representability does not force optimizer convergence.** At all-zero
parameters the full model's hidden state and tied embedding both vanish.
Lean differentiates the actual forward and proves every logit derivative,
the full logit Jacobian and the ordinary CE parameter gradient are zero.
This point is inside the clipped-XSA region and is locally analytic, so the
result is not an artifact of a total derivative at a singularity. Starting
with zero query/base buffers, actual Schedule-Free AdamW keeps both buffers
zero for every finite step, even with averaging, weight decay and evolving
Adam history. All logits remain zero. With multiple classes this cannot
strictly classify any example. The result holds at every positive finite
architecture size: more width, depth, epochs or altered optimizer parameters
alone cannot establish a universal learning guarantee.

Together, the exact successful weights and the genuine stationary run close
the logical gap between capacity and guaranteed learning: capacity is proved,
unconditional convergence is refuted, and finite learning conclusions require
the explicit transfer, regularity and remainder conditions above. Whether
typical random initializations satisfy sufficiently strong conditions on the
experiment tasks remains a mathematical and empirical question.

Validation of this extension: the full `lake build` passes (13121 jobs).
There are 97 new public theorems, plus two private witness lemmas, across
25 new modules; every module is reachable from `Transformer.lean`.
Individual `#print axioms` checks for all 97 public theorems use only the
allowed standard axioms. `./run.py audit` reports 36 existing `sorry`, zero
results resting on them and zero extra axioms. The regenerated index has
4024 modules and 20781 theorems, with zero vacuous or placeholder declarations.
`./run.py forbidden` and `git diff --check` pass. Every theorem with explicit
mathematical hypotheses has a concrete Lean witness, including both sides
of clipping, ordered/equal token pairs, real source epsilons, and nonempty CE
training rows. `./run.py test schedule_free_adamw` passes all six selected
optimizer tests on Python 3.15.2. No production Python code changes or new
training run is part of this proof extension; the previously recorded wider
Muon allocation failures are not claimed fixed. Logs and the theorem-level
axiom script are retained in `.kanban/artifacts/123/`.

## Finite sequence capacity and information limits

**N1, what the actual learner observes.** For an arbitrary parameterized
predictor, two target functions agreeing on every row sampled during the
first n updates induce the same CE function at each update, hence the same
full parameter gradient and the same complete Schedule-Free AdamW state.
This includes query/base buffers, moments, averaging, weight decay, any
initial state and a nonzero starting step. A label at an unobserved row
can therefore be replaced without changing the trained weights. Two
different replacements cannot both be strictly classified. The Lean
theorems are `Information.trajectory_eq_of_labels`,
`unseen_trajectory_eq`, `unseen_correct_incompatible` and
`exists_unseen_failure`. The sampling plan is fixed independently of the
replacement; no claim about label-adaptive sampling is hidden here.

**N2, a target independent of the random seed.** Let a finite seed space
carry arbitrary nonnegative masses summing to one. Each seed may select
its own initialization and sampling plan, provided the same designated row
is unobserved throughout the finite training horizon. There is one fixed
replacement label v such that

```
card(V) * Pr_seed[strictly correct on that row when its target is v] <= 1.
```

The order is `exists v, probability bound over seeds`, not a different
adversarial target chosen after observing each seed. For any seed, at most
one label can strictly win; summing those winner masses and averaging over
labels proves the bound. `Information.exists_unseen_low_success` connects
this counting argument to the actual CE/SFAdamW trajectory using N1.
Ties count as failures, and an equally weighted two-seed/two-label Lean
example attains the bound. Other examples use nonempty observed rows,
nonzero initial weights, the source optimizer parameters and 100 updates.

N2 covers finite distributions, including finite ensembles of PRNG seeds;
it does not assert a measure-theoretic result for a continuous Gaussian.
More importantly, these are **unconstrained target extensions**: changing
one label can leave a known modular-arithmetic family. The result does not
refute learning that family from few examples. It identifies the missing
assumption in a guarantee for arbitrary tasks: information about unseen
labels must come from additional observations or a justified restriction
on the target family. More width, depth, epochs or random starts alone
cannot supply it. Positive transfer certificates T1--T4 retain their stated
scope; convergence from typical starts on the algebraic experiments remains
unproved.

Validation of N1--N2: all 11 new public theorems have individual axiom
checks using only `propext`, `Classical.choice` and `Quot.sound`. The full
build passes (13123 jobs), with no warnings in the new modules. Audit
remains at 36 existing `sorry`, zero dependent results and zero extra
axioms. Index regeneration reports 4026 modules, 20792 theorems and zero
vacuous or placeholder declarations. Forbidden checks and all six selected
Schedule-Free AdamW Python tests pass. No new training run is claimed.
Logs and axiom checks are retained in `.kanban/artifacts/124/`.

**F1--F6, finite capacity for arbitrary arity.** For every nonempty finite
vocabulary of size p, every natural N, and every table
`f : (Fin N -> Fin p) -> Fin p`, there are fixed finite Transformer weights
and a fixed positional rotary phase table strictly classifying every input.
The input is the N ordered operands **followed by the fixed token 0 for
readout**. That token already belongs to the vocabulary. Its use is part of
the theorem's input protocol; this is not an unstated claim about the
unaugmented N-token protocol. A sufficient source-shaped family is

```
W = 2 (p+1)^2 + 6 p^N + 2 padding,
H = floor((8 W + 1)/3),
heads = 1,      depth = extraDepth + 1,      context length = N + 1.
```

Padding and extraDepth are arbitrary naturals, RMS epsilon is positive, and
XSA epsilon is at most one. The main theorem is
`Attention.Finite.exists_finite_sequence_classifier`. Unlike E8, its query
and key are nonzero and its rotary attention weights depend on position.
It constructs a phase table; that theorem alone does **not** quantify over
prescribed tables. P1--P6 below establish capacity for the usual source
integer-position geometric grid, using the original input tokens.

1. **F1, bounded separation:** any finite collection of distinct real
   vectors can be separated by one linear functional chosen in any open
   ball. A finite union of proper hyperplanes cannot contain an open ball.
   Applied to all zero/one subsets of N positions, this gives coefficients
   `1 < w_j < exp(1/sqrt(W))` with distinct subset sums. They can be chosen
   arbitrarily close to uniform. The proof does not demand unbounded
   attention temperatures or increasingly concentrated softmax weights.
2. **F2, actual finite SwiGLU interpolation:** positive common-channel
   values u_i and distinct projective input directions admit a linear
   measurement with distinct ratios alpha_i. Choose a positive common
   separation delta. Three actual SiLU gates at each center, using the
   common channel for offsets, converge after gain rescaling to a triangular
   feature. The evaluation matrices converge to a diagonal matrix with
   entries `u_i^2 delta > 0`. Continuity of the determinant gives an
   invertible matrix at a **finite** positive gain. Its inverse supplies
   fixed output weights interpolating arbitrary vector targets exactly.
   The model activation remains SiLU throughout; ReLU occurs only in the
   auxiliary limit argument. There are three hidden channels per input row.
3. **F3, actual normalization and branch reachability:** these features
   compile into the source's gate, value and output matrices with zero
   padding. Positive-epsilon RMSNorm is injective, and projective coordinates
   relative to a positive common channel survive its row-dependent scale.
   For a finite context table, one fixed positive projection scale makes all
   normalized contexts have norm below one. A unit readout residual and unit
   answer then differ by norm below three. Width above nine places every
   desired displacement inside the true branch-output ball; E1 supplies
   its exact inverse-normalized raw target. Both FFN normalizers remain.
4. **F4, ordered context after XSA:** a disjoint context bank stores
   `sum_j w_j e_(x_j)` with the readout label's coordinate removed. Distinct
   subset sums recover the positions of each other label; remaining
   positions necessarily contain the readout label. Thus this code is
   injective on ordered sequences, including repeated tokens and any finite
   N. The readout residual supplies a fixed common coordinate equal to one.
5. **F5, connection to actual rotary attention:** normalized embeddings are
   unit token vectors. A fixed linear sum of their coordinates creates the
   same nonzero query/key at each position, normalized to a unit vector in
   one actual half-split rotary pair. Set the readout phase to zero and
   operand phase j to `acos(log(w_j) sqrt(W))`. The normalized score
   exponentiates to w_j. The actual causal denominator is
   `D = sum_j w_j + exp(1/sqrt(W))`, including self attention. Unit values
   are above the XSA clip at the source epsilon. Actual own-value subtraction
   deletes exactly the readout coordinate, and a fixed projection copies
   the retained code to the workspace. The explicit zero output at the first
   position handles N=0. Neither a modified mask nor an assumed context
   representation enters the forward.
6. **F6, full source model:** the exact FFN maps each normalized context to
   the inverse-normalized displacement to the answer's unit token vector.
   The actual residual reaches that answer. Additional blocks have zero
   projections and are identity residuals. Final RMSNorm and the tied raw
   embedding give logits `L * 1[q=f(x)]` with `L>0`, so every rival has a
   strictly smaller score. The source hidden-width ratio has room for all
   `3 p^N` channels, and its context window contains the appended token.

This closes the finite-arity **capacity** gap for the stated input protocol
and freely chosen rotary phases. The sufficient size is exponential in N,
and the real-valued generic coefficients and finite gate gain need not be
well conditioned or easy to discover numerically. No useful parameter-size
optimum, floating-point error bound, new training result, or typical-start
convergence claim follows. In particular, N1--N2 still separate available
label information from expressivity; T1--T4 still state the explicit
conditions needed for finite optimizer progress. P1--P6 below close the
prescribed-source-phase and unaugmented-input capacity questions. Learning
the algebraic task family from typical random starts remains a separate question.

Validation of F1--F6: the full `lake build` passes (13138 jobs), with no
warnings in the 15 new modules. All 55 new public theorems have individual
`#print axioms` checks using only the three standard allowed axioms; seven
private witness lemmas are also covered by the whole-project audit. Audit
remains at 36 existing `sorry`, zero dependent results and zero extra axioms.
The regenerated index has 4041 modules and 20854 theorems, with zero vacuous
or placeholder declarations. `./run.py forbidden` and `git diff --check`
pass. Every conditional construction has a concrete Lean witness, including
nonzero Q/K, positive-epsilon normalization, nonempty inputs, N=0, p=2,3,97,193,
three/four-variable tasks, source hidden widths, padding and varied depth.
No production Python change or training run is included. Logs and the
theorem-level axiom script are retained in `.kanban/artifacts/124/`.

## Prescribed RoPE capacity

The source's actual integer-position geometric grid suffices. For every
nonempty finite vocabulary of size p, every **positive** finite arity m,
and every table `f : (Fin m -> Fin p) -> Fin p`, Lean proves that one finite
set of model weights strictly classifies all original input tuples at the
last original token. Positions are `0,...,m-1`, every layer enables RoPE,
and its angles are exactly `position * base^(-2*pair/head_dim)`.
No positional table is selected by the proof and no readout token is added.
The full forward retains embedding/final RMSNorm, both Peri-RMSNorm branches,
Q/K RMSNorm, causal softmax, clipped XSA, bias-free SwiGLU and tied readout.

The public theorem is `Attention.Grid.exists_finite_sequence_classifier`.
Its Lean index is `N=m-1`, so the original input type is `Fin (N+1) -> Fin p`.
A sufficient source-shaped family, with arbitrary natural padding and extraDepth, is

```
W = 2 (p+1)^2 + 6 p^m + 2 padding,
H = floor((8 W + 1)/3),
heads = 1,      depth = extraDepth + 1,      context length = m.
```

RMS epsilon is positive and XSA epsilon is at most one; the source values
`2^-23` and `1e-12` satisfy these hypotheses. The displayed family uses
base 10000. `Grid.exists_classifier` works for any valid configured positive
base, with one head, width above nine, disjoint token/context banks, and
`3 p^m <= H`. The family theorem configures enough source table positions
for the entire input. As in G4, the generic mathematical forward can accept
longer directly supplied phase tables; Python execution requires them to
fit the configured context window.

1. **P1, fixed integer offsets:** source rotary pair zero has frequency one,
   independently of the base or width. Its final-row cosine values are
   `cos(N-j)` for prefix positions `j<N`. Cosines of distinct nonnegative
   integers are distinct: equality would make an integer difference or sum
   equal to `2 k pi`; irrationality of pi forces `k=0`, then equality of the
   original integers. Thus the fixed slopes
   `a_j = cos(N-j)/sqrt(W)` are injective. This proof concerns the actual
   integer positions, not an assumption that arbitrary phases separate inputs.
2. **P2, a bounded ordinary query gain:** for distinct slopes, the functions
   `s -> exp(a_j s)` are linearly independent by Dedekind's theorem for
   multiplicative characters of the additive reals. Every difference of two
   unequal positional subset sums is therefore nonzero somewhere and is
   globally real analytic. Analytic uniqueness makes its nonzero set dense
   and open. There are finitely many subset pairs, so their nonzero sets
   intersect every open gain interval. In particular, some `0<s<1` gives
   weights `w_j=exp(a_j s)` with all subset sums distinct. The proof neither
   assumes independent learned features nor adjusts the prescribed RoPE phases.
3. **P3, genuine fixed-grid attention:** normalize the query to `s e_0` and
   key to `e_0` in the first actual half-split rotary pair. Both are inside
   the positive-epsilon RMSNorm image and have exact finite raw preimages.
   Fixed fused-QKV weights generate these same nonzero queries/keys at every
   token; values are unit token vectors. Source scores are
   `s cos(N-j)/sqrt(W)`, and the self score is `s/sqrt(W)`. The real causal
   denominator is `D=sum_j w_j + exp(s/sqrt(W))`. Actual clipped XSA removes
   the final label's value coordinate. A fixed projection multiplies by D
   and copies the rest to the disjoint context bank. For m=1 the source's
   explicit first-position attention zero gives the empty prefix context.
4. **P4, original last-token readout:** the final label is an input variable,
   not a fixed extra token. Its unit residual has zero coordinates in the
   context bank, while the normalized context has zero coordinates in the
   entire residual token bank. Equality of resulting states first forces
   equality of final labels. Distinct subset sums then recover every prefix
   token, including repeated occurrences of the final label. Positive-epsilon
   RMSNorm preserves this injectivity. The sum of residual token coordinates
   is one on every state, giving the common positive FFN channel even when
   the last label varies. `split_input` proves the prefix/last decomposition
   reassembles the exact original token tuple.
5. **P5, complete normalized model:** choose one positive projection scale
   making the finitely many normalized contexts have norm below one. The
   F2--F3 finite SwiGLU interpolator uses `3 p^m` channels to reach each
   requested unit answer by the actual normalized residual branch. Every
   matrix is fixed across all inputs. Additional identity blocks preserve
   that answer at any positive finite depth. Final RMSNorm and tied raw
   embeddings give logits `L * 1[q=f(x)]`, with `L>0`. The conclusion uses
   `sourceAngles cfg (fun _ => true) (fun j => j.val)` directly; only
   `weights`, not phases or input tokens, is existentially quantified.
6. **P6, finite source dimensions:** the displayed even width contains both
   token banks, has width above nine, and its source FFN ratio contains all
   interpolation channels. Its context window is exactly m, with no hidden
   position beyond the original input. Concrete Lean instances cover m=1,
   two-, three- and four-variable tasks, p=2,3,97,193, padding and varied depth.

The modules follow proof dependencies: `Grid.Separation` proves P1--P2,
`Grid.Head` the actual P3 scores and XSA code, `Grid.Encoder` the fixed matrix
and P4 injectivity statements, and `Grid.Model` P5--P6 for the full forward.
The source remains diff_gpt commit `b98c665bd8faa30729fd368d257a13b6d58712a1`;
its geometric table, half-split layout, source normalization order and
context-window conditions were reread beside these statements.

This closes two capacity gaps left by F1--F6: a prescribed source grid and
the original input protocol. It does not change the earlier flat-spectrum
matrix-completion theorem into a learning impossibility theorem for this
nonlinear model. The sufficient width grows exponentially in arity. The
argument supplies finite real weights but no useful bound on feature
separation, gate size, numerical conditioning or floating-point error.
It also makes no claim that every smaller architecture can implement every
table. Typical-random-start Schedule-Free AdamW convergence, efficient
learning of the modular family, and generalization from partial labels
remain unproved; N1--N2's information limits and T1--T4's conditional
progress statements retain their existing scope. No new training result
is asserted by this capacity proof.

Validation of P1--P6: `lake build` passes (13142 jobs), with no warnings in
the four new modules. All 29 new public theorems have individual
`#print axioms` checks using only `propext`, `Classical.choice` and
`Quot.sound`; the private concrete-witness lemma is covered by the project
audit. Audit remains at 36 existing `sorry`, zero dependent results and
zero extra axioms. The generated index has 4045 modules, 20884 theorems,
zero vacuous declarations and zero placeholders. Forbidden checks and
`git diff --check` pass. Every conditional theorem has a concrete Lean
instance, including a nonempty prefix, both possible final labels, the
source epsilons, a bounded nonzero query gain, and the finite source-shaped
width/depth examples above. All five source hashes still match the pinned
manifest. Logs and the theorem-level axiom script are retained in
`.kanban/artifacts/125/`.

The required `./run.py test` run uses Python 3.15.2. All 418 lab tests pass
with three skips. The standalone optimizer suite runs 104 tests and reports
the two previously recorded Schedule-Free Muon CUDA allocation failures
(float32 and float64); each reports 75 allocations of 8519680 bytes in the
warmed-step allocation assertion. Schedule-Free AdamW tests pass. Thus the
complete Python command exits 1, rather than being reported as green. No
Python or optimizer implementation changed in this proof extension, and
those existing Muon failures are not claimed fixed. The complete log is
retained as `.kanban/artifacts/125/tests.log`.

## Depth, width and precision

The earlier capacity constructions establish existence at a sufficient finite
width. Their additional blocks are identities, so they do not establish a
benefit from active depth or a necessary depth/width tradeoff. A flat spectrum
alone does not supply either conclusion. The following results add explicit
resource accounting and a counting-logic model of the normalized architecture.
They distinguish stored parameter bits, stored activation bits, counting depth,
fixed context length, and optimization success.

### R1: finite parameter information

Fix the architecture, input set X, label set V, and a parameter alphabet before
choosing a target table. One setting of the parameters can strictly classify
at most one complete table X -> V: two different labels on the same row would
require contradictory strict score inequalities. Therefore, universality needs

```
|V|^|X| <= number of parameter settings.
```

With b bits in each of M parameter coordinates and at least two labels, this
implies `|X| <= b*M`. Below that budget there is one fixed table which every
allowed weight setting fails to classify strictly. The target is quantified
before the weights. This argument permits any nonlinear forward and even
exact real arithmetic after decoding the finite weight alphabet. It does not
require finite-precision activations, a kernel approximation, or a rank bound.

### R2: the actual source parameter count

For vocabulary p, residual width W, FFN width H and D blocks, the full source
parameter count is

```
M = p*W + D*(4*W^2 + 3*W*H).
```

The tied embedding is counted once; fused QKV, output projection and all three
SwiGLU matrices are included. For the source ratio `H = (8*W+1)/3`, integer
division gives `M <= p*W + D*(12*W^2+W)`. Consequently, universality on the
complete length-N input space requires

```
p^N <= b*[p*W + D*(4*W^2+3*W*H)]
     <= b*[p*W + D*(12*W^2+W)].
```

The implemented source obstruction uses its actual RMSNorm/RoPE/XSA/SwiGLU
scores, arbitrary fixed phases and any fixed b-bit value codebook. A concrete
instance uses p=3, W=4, H=11, D=2, N=8 and b=8: M=404, hence 3232 stored bits
are fewer than the 6561 input rows. Some complete task cannot be represented
with that budget, even with exact forward evaluation.

This is a worst-case statement about representing **all** tables. It is not
a necessary budget for `(a/b)^K mod P`, whose algebraic structure can make it
much cheaper, nor a claim that increasing the budget guarantees learning.
The precise table-count inequality in R1 is stronger than the one-bit-per-row
consequence displayed here. Inputs restricted to nonzero denominators use
their own finite input set in R1.

### R3: extend the counting model to parallel heads and local nonlinearities

The construction extends the finite-state argument of arXiv:2506.16055v3,
Appendix B.2 and Appendix F, rather than identifying the source block with the
paper's different transformer definition. A counted layer has a denominator
and numerator table for each head coordinate, followed by an arbitrary local
finite-state map. Quotients are compiled using the existing proved rounded
count-cell formulas. Enumerating the query state and the rounded head vector
then implements the local map by Boolean tests.

All parallel heads add **at most one counting level per layer**. The number of
heads and width affect the number of coordinates and enumerated states, not
the number of nested counting operations. The induction proves exact state
simulation for every finite stack and every positive prefix. Initial finite
embedding and final local readout add no counting level.

This compiler is not claimed efficient. With activation precision r>=1 bits,
W coordinates, T possible position tags and a separate sentinel, the state
count is exactly `T * 2^(r*W) + 1`. The proof exposes that width/precision cost
rather than treating arbitrary finite local maps as a bounded-size circuit.

### R4: normalized source adapter and unchanged token inputs

The concrete counted layer uses the source fused-QKV split, head reshape,
RMS-normalized Q/K, arbitrary half-split rotary phases, scaled logits, clipped
XSA and its explicit zero at the first source position. It preserves both
branches' input and output RMSNorm, both residual additions and SwiGLU.
`local_sourceAverage` proves that substituting the unrounded source averages
into this local computation gives exactly the existing real `Attention.block`.
The source exponential establishes the denominator nonnegativity required by
the counting-cell proof.

The **rounded variant** has explicit numerical semantics: round exponential
coefficients and weighted value coordinates before summing; accumulate their
integer mantissas without overflow; round the quotient; then round the stored
residual after each complete block. Round the normalized embedding once on
entry. A zero rounded denominator produces zero attention. The local maps and
final normalized tied scores use real arithmetic. This differs from the
paper's uniform-attention zero-denominator fallback and from IEEE float32
execution, including its accumulation and intermediate rounding.

For a fixed context bound T, residues modulo T+1 identify every positive
position on words of length at most T. The compiler uses those predicates to
recover zero-based source position tags without changing the input vocabulary
or inserting a source token. Its sentinel contributes zero to attention sums.
Thus `source_classifier_mem` and `source_classifier_correct` give an
end-to-end depth-D counting formula on the **original token words**, including
embedding normalization, all rounded blocks, final RMSNorm and tied readout.
The readout tests strict classification; ties are rejected, consistently with
the capacity results. There is no hypothesis assuming that trained logits
already solve the target task.

### R5: exact source RoPE is outside the existing periodic hierarchy

Appendix F, `sec:rope_pes`, assumes rational multiples of pi so that RoPE is
periodic in the integer position. The actual geometric source grid has first
frequency exactly one radian, regardless of its base. Lean proves that this
is not a rational multiple of pi, that its first-pair cosine has no positive
integer period, and that **no finite positional code** can reproduce that
cosine at every natural position. The latter follows from cosine injectivity
on natural numbers and the finite pigeonhole principle.

The bounded-context compiler in R4 is therefore not a uniform simulation for
unbounded context: its modulo T+1 predicates and state space change with T.
It does not transfer the paper's all-length depth hierarchy unchanged to the
exact source model. This obstruction concerns exact real rotations; it makes
no assertion about the eventual behavior of a particular float32 RoPE routine.

| Lean module | Results |
| --- | --- |
| `SequenceModel.Resources.Parameters` | unique strict target, table/bit counting, complete parameter count, source depth/width/precision obstruction |
| `CRASP.Normalized.Counting` | parallel attention-cell compilation and arbitrary local output tests |
| `CRASP.Normalized.Program` | stack simulation, depth-zero embedding, exact RoPE positional obstruction |
| `CRASP.Normalized.Source` | actual source local computation, rounded adapter, stack simulation, activation-state count |
| `CRASP.Normalized.Input` | normalized tied readout and complete bounded-context compiler on original tokens |

The remaining questions are sharper: a task-specific active-depth advantage,
useful tight width/depth lower bounds for powered modular division, a uniform
hierarchy allowing the source's nonperiodic positions, and a quantitative
connection between this rounding model and IEEE execution. None is replaced
by identity padding or inferred from the flat spectrum. Training convergence
and optimal Schedule-Free AdamW settings also remain separate questions.

Validation of R1--R5: the full `lake build` passes (13149 jobs), with no
warnings in the new modules. Individual `#print axioms` checks cover all
31 new public theorems and use only `propext`, `Classical.choice` and
`Quot.sound`; the private lookup witness is also covered by the project
audit. Audit remains at 36 existing `sorry`, zero results resting on a
`sorry`, and zero extra axioms. The regenerated index has 4052 modules,
20916 theorems, zero vacuous declarations and zero placeholders. Forbidden
checks and `git diff --check` pass. Concrete witnesses cover the resource
obstruction, finite state counts, initial-state predicates, positive readable
prefixes, and the actual first geometric frequency. All five source hashes
match the pinned manifest. Proof and validation logs are retained in
`.kanban/artifacts/126/`.

The required Python 3.15.2 regression run passes all 418 lab tests (three
skips). Its 104-test optimizer suite reproduces the two pre-existing
Schedule-Free Muon warmed CUDA allocation failures, float32 and float64,
each reporting 75 allocations of 8519680 bytes. Schedule-Free AdamW tests
pass. Thus `./run.py test` exits unsuccessfully, as at the parent commit;
this Lean-only extension does not fix those unrelated optimizer failures.
The full log is `.kanban/artifacts/126/tests.log`.

## Uniform nonperiodic RoPE

The model's RoPE does not need to be made periodic. The theory needs a
position representation that can express the actual source coefficients.
R5 rules out an exact finite periodic code; it does not rule out a counting
language that directly uses a query/source position pair. U1--U4 provide that
extension and remove the context-bound restriction from the simulation theorem.

### U1: position-dependent weighted counts

The new syntax contains token tests, a first-position test, Boolean operations,
integer constants, addition, integer scaling, comparisons, and the operation

```
Count_K[phi](w, i) = sum_{1 <= j <= i} K(i,j) * 1[phi(w,j)].
```

K is fixed independently of the input word. The count retains the query
position i while evaluating phi at each source position j. A nested count in
phi uses j as its own query position. The constant-one kernel recovers ordinary
causal counting; `Term.ordinary_count` proves this identity.

The grammar does not contain attention, a transformer forward, rounded
division, or arbitrary predicates of the whole word. Rounded quotients are
compiled into integer comparisons of numerator and denominator sums. For
positive D, the unsaturated cell for mantissa y is

```
y*D <= 2^s*N < (y+1)*D.
```

The lower or upper comparison is omitted at the corresponding saturation
endpoint. A zero denominator returns zero, matching R4. Lean proves these
endpoint-aware cells from `Fx.round_eq_iff`, including signed numerators and
complete denominator underflow. Thus division adds no counting level.

### U2: one uniform formula for a finite-state stack

Each coefficient name records its layer, numerator/denominator choice,
fixed query state, head coordinate and fixed source state. Its interpretation
is the corresponding integer mantissa at the position pair (i,j). It cannot
inspect the input word. The actual query state is selected by its previous-layer
formula, and source states are selected inside the count. Reordering finite
sums proves that this enumeration recovers the actual attention sums.

Parallel heads use parallel coordinate formulas. The arbitrary local finite-state
map is then a Boolean test of the current state and rounded aggregate vector.
The induction proves exact simulation with one additional counting level per
block. Tokenwise embedding, the first-position flag and final local readout
add no counting level.

The order of quantifiers is essential: for fixed layers, embedding, output test
and depth D, construct **one formula**, then quantify over **all finite token
words and positive readable positions**. Neither its syntax nor its kernels
depend on a word's length. There is no context bound or annotated token alphabet.
`classifier_correct` proves this statement; `classifier_depth` bounds its
counting depth by D.

Each positional coefficient remains a fixed-point mantissa in
`[-2^(r-1), 2^(r-1))`, even when its dependence on positions is nonperiodic.
This is proved by `kernel_bounds`. Prefix sums still use unbounded integers,
as in the source paper's idealized `sumdiv`; this is not an IEEE accumulation
guarantee.

### U3: construct the kernels from the supplied RoPE architecture

The concrete kernels evaluate the source head score: pre-attention RMSNorm,
fused-QKV projection and reshape, head Q/K RMSNorm, exact half-split rotary
phases, division by sqrt(head_dim), and exponentiation. Numerator kernels
also multiply by the source's unnormalized value coordinate before rounding.
The source phase is exactly

```
angle(layer, position, pair) =
  position * theta^(-2*pair/head_dim)
```

at enabled layers, and zero at disabled layers. Position is an unrestricted
natural number, with no modulus or rational-angle assumption. The logic's
one-based positions are converted to the source's zero-based positions before
computing these phases. `sourceAngles_restrict` identifies this formula with
the existing finite source angle table.

The local map retains clipped XSA, its exact first-position zero, output
projection, both branches' input/output RMSNorm, the two residual additions
and SwiGLU. Only the first-position bit is retained in the state; the absolute
position is supplied to the coefficient computation. That bit is proved
invariant under the stack. For r>=1 activation bits and width W, the finite
state count is `2 * 2^(r*W)`, independent of input length.

`denominator_restrict`, `numerator_restrict` and `finish_restrict` identify
the new operators with the earlier rounded source model at every finite
position. `coefficient_sum` then proves equality of the complete prefix sums.
`aggregate_restrict` covers the rounded attention quotient, including the
zero-denominator case, and `step_restrict` covers the whole normalized block.
The final normalized tied readout is also unchanged. These are exact
agreement results under R4's specified rounding semantics.

### U4: the complete normalized model at every finite length

`uniform_source_classifier` instantiates the compiler with the complete
source parameter table, the exact geometric RoPE grid, arbitrary per-layer
enable flags, normalized/rounded token embeddings, all configured blocks,
and the final RMSNorm and tied embedding readout. It proves

```
for every fixed model and target label,
  there exists one formula phi with depth(phi) <= model.n_layers,
    such that for every finite original-token word w and 0 < i <= length(w),
      phi(w,i) = the rounded source model's strict classification test at i.
```

The formula is chosen before the word, and no premise asserts that trained
logits already solve a task. The stored residual state and formula no longer
grow with a prescribed context bound. This closes the nonperiodic-position
coverage gap in the rounded model's counting simulation. It leaves the
actual RoPE frequencies unchanged and preserves the earlier finite-code
impossibility result.

This is an extension of the language, not a proof that the periodic language's
strict lower-bound hierarchy also holds in the extension. Its source-generated
kernel family must be analyzed to establish new lower bounds; arbitrary kernel
families can have greater power. The upper simulation bound alone does not
prove that every extra layer increases expressivity. The exact real phase
calculation and the fixed-point/real-local-map rounding semantics also remain
distinct from actual float32 execution. No new training or convergence result
is asserted, and the mathematical all-position grid does not enlarge the
Python implementation's configured context storage or KV cache.

| Lean module | Contents |
| --- | --- |
| `CRASP.Rotary.Logic` | explicit syntax, weighted-count semantics, ordinary-count special case |
| `CRASP.Rotary.Cells` | exact quotient cells and position-sensitive state-sum identity |
| `CRASP.Rotary.Layer` | bounded coefficient alphabet and one-layer compilation |
| `CRASP.Rotary.Program` | uniform stack/input/output simulation and counting-depth bound |
| `CRASP.Rotary.Source` | source-generated kernels, exact phases, finite states and tied readout |
| `CRASP.Rotary.Model` | complete prefix/block agreement and the all-length source classifier |

Validation of U1--U4: full `lake build` passes (13156 jobs), with no warnings
in the new modules. All 43 public theorems have individual `#print axioms`
checks using only `propext`, `Classical.choice` and `Quot.sound`. The audit
remains at 36 existing `sorry`, zero dependent results and zero extra axioms.
The regenerated index has 4059 modules, 20959 theorems, zero vacuous results
and zero placeholders. Forbidden checks and `git diff --check` pass. All
five pinned source-file hashes still match. Concrete witnesses cover Boolean
state predicates, positive and zero rounding denominators, nonempty prefix
sums, finite activation-state counts and the full source classifier.

The required Python 3.15.2 lab run completed 418 tests with three skips and
one CUDA graph-capture error in
`test_compiled_training_resumes_with_float32_tolerance`. Rerunning that test
alone in a fresh process passed (one test, 8.544 seconds); the original full
run is still recorded as unsuccessful. The separately completed 104-test
optimizer suite reproduces the two existing Schedule-Free Muon allocation
failures, float32 and float64, each with 75 allocations of 8519680 bytes.
Schedule-Free AdamW tests pass. No Python implementation was changed.
The theorem script, source checks and all original/retry validation logs
are retained in `.kanban/artifacts/127/`.

## Quantitative RoPE approximation

This extension keeps exact source RoPE in the model and asks what changing
its frequencies or rounding its attention coefficients can change. The
source is still `diff_gpt/domain/model/{gpt,block,mha,feed_forward,rope}.py`
at `b98c665b`, with the hashes in `sequence_model_source.json`. The periodic
comparison is with arXiv:2506.16055v3, Appendix F, `sec:rope_pes`, read from
`papers/`. As in the earlier correction, periodic frequencies are rational
multiples of pi, rather than rational numbers of radians.

### A1. Quantitative phase and frequency errors

For the actual half-split rotation, including an unchanged odd leftover
coordinate in the mathematical generalization,

\[
 \|R_a x-R_b x\|_2\le\delta\|x\|_2
 \quad\text{if}\quad |a_c-b_c|\le\delta\text{ for every pair }c.
\]

The proof bounds the squared chord between two unit-circle phases by the
squared angle difference and sums over the source channel pairs. With
positive RMS epsilon and positive head width, the normalized source logit
therefore obeys

\[
 |s(a,b;q,k)-s(a',b';q,k)|
 \le (\delta_q+\delta_k)\sqrt d.
\]

This holds for arbitrary query/key magnitudes. For frequencies f and g
with coordinate errors at most eta, the phase error at position t is at
most `|t| eta`. A fixed frequency error does not supply a fixed phase-error
budget at arbitrarily large positions. These statements bound exact plane
rotations; separately rounded sine/cosine entries need not form an exact
plane rotation and are not automatically covered by a phase-error premise.

### A2. Causal softmax and the actual XSA head

If all unmasked logits at a query move by at most r, the L1 change in the
entire causal softmax row is at most `8 r`. The constant is conservative.
For the same value stream with `norm(v_j) <= B`, its average changes by at
most `8 r B`. Applying the source XSA correction with nonnegative clipped-L2
epsilon gives a whole-head phase-error bound

\[
 \|H_a(q,k,v)_i-H_b(q,k,v)_i\|_2
 \le 32\delta\sqrt d B.
\]

The explicit first-position zero is retained. The constants do not grow
with the number of keys when the phase-error and value-norm premises are
fixed. This theorem holds Q/K/V fixed; it is not by itself a stack bound,
since changing early phases changes later residuals and hence later Q/K/V.

### A3. Continuity through the complete source stack

Joint continuity is proved through every actual operation: epsilon RMSNorm,
half-split rotations, fused QKV and head reshaping, causal softmax, clipped
XSA, projection, all four branch normalizers, SwiGLU, residual additions,
normalized embedding, final RMSNorm and tied readout. Weights are arbitrary
and fixed, depth and width are arbitrary valid finite source dimensions,
and both normalizer epsilons are positive.

This includes the XSA clipping sphere. Continuity needs no exclusion of
that sphere, although the earlier analyticity/gradient theorem still does.
The entire residual stream is recomputed in every subsequent block. There
is no premise assuming continuity of a black-box model or assuming a bound
on the final change in logits.

### A4. What a finite-context periodic replacement preserves

`frequencyAngles` varies the shared frequency vector, retaining the source
per-layer RoPE flags and positions. Substitution of `invFreq` is exactly
`sourceAngles`. Finite vectors of rational multiples of pi are dense; for
any rational vector q the actual half-split rotations have a common positive
integer period `2 * product(q_c.den)`.

For every fixed finite T, fixed weights and positive error tolerance eta,
one such rational-pi frequency vector approximates **every logit, every
readout position and every input tuple of length T** to error less than
eta. This is uniform over the finite input domain, not just selected
training examples. A strict source classifier at this fixed T consequently
has a periodic frequency replacement with the same weights, width, depth
and strict decisions. A nonconstant two-token binary sum task witnesses
the hypotheses using the existing exact-source capacity theorem.

The replacement frequencies may cease to be a geometric grid. The theorem
does not claim that merely changing the source's single base parameter can
produce the replacement. Its period and tolerance may depend on T, the
weights, the epsilons and the strict margins. Ties have no such certificate.
The frequency tolerance is existential; this proof does not numerically
compute it for a trained model.

The quantifiers are crucial for the depth question:

\[
 \forall T\ \exists q_T\ \text{(same finite-context decisions)},
 \qquad\text{not established:}\qquad
 \exists q\ \forall T\ \text{(same language)}.
\]

Thus the finite-context replacement preserves the given depth, but does
not transfer the periodic language's strict depth lower bounds to exact
source RoPE. The earlier impossibility of a finite exact position code
remains true. The weighted-count compiler in U1--U4 still supplies the
uniform upper bound of one count nesting per source block. A strict lower
hierarchy for its particular source-generated nonperiodic kernels remains
unproved; the present approximation argument does not remove that gap.

### A5. Quantitative readout and classification certificates

For a final residual error `norm(x-y) <= E`, positive RMS epsilon eps and
a tied embedding row e, the logit error is bounded by

\[
 \frac{2}{\sqrt{\mathrm{eps}}}E\|e\|_2.
\]

A target-versus-competitor decision is preserved if the original strict
margin exceeds

\[
 \frac{2}{\sqrt{\mathrm{eps}}}E
 (\|e_{\rm target}\|_2+\|e_{\rm competitor}\|_2).
\]

The raw tied embedding norms are necessary; final RMSNorm does not bound
them. This certificate requires a separately established final residual
error E. It does not silently treat the local head estimate as that error.

### A6. Rounding certificates for the existing nonperiodic counting head

`Fx.NoSaturation p s x` checks that `floor(x * 2^s)` lies in the format's
mantissa interval. Under that condition, the existing floor-and-saturate
`Fx.round` has absolute error strictly smaller than `u = 2^-s`. There is
also a theorem that no finite uniform absolute error bound exists without
a range restriction, and a concrete saturated counterexample at p=8,s=0.

The source normalized logits give every unrounded exponential coefficient
a floor `a = exp(-sqrt(d))`. If `u < a` and every exponential coefficient
avoids saturation, the sum of the rounded coefficients is strictly positive
at every nonempty prefix. This rules out the counting model's zero-denominator
fallback under these assumptions, rather than assuming division is safe.

For a value coordinate bounded by B, if all exponential coefficients,
weighted numerator coefficients **and the final quotient** avoid saturation,
then the actual `CRASP.Rotary.sourceLayer` attention aggregate has error

\[
 \left|\widehat y-y\right|
 \le \frac{u(1+B)}{\exp(-\sqrt d)-u}+u.
\]

The comparison y is the exact real softmax average on the **same stored
residual stream**. The proof relates the implementation's integer-mantissa
quotient to the ratio of interpreted coefficients, bounds both sums, and
cancels their common prefix-length factor. The last u accounts for the
existing final quotient rounding. Concrete source-layer witnesses retain
the source epsilons and nonzero positions.

This remains the specified counting model: exact integer sums, rounded
coefficients and quotient, real local nonlinearities, and separately rounded
stored residuals. It is not IEEE round-to-nearest. In particular, fixed-point
u is not float32 machine epsilon. B1--B9 below propagate this local estimate
through the whole specified rounded residual stack. Hardware error bounds
for projections, RMSNorm, SwiGLU and RoPE table generation remain outside
these results. No new training or optimizer convergence guarantee follows.

| Lean module | Result |
| --- | --- |
| `SequenceModel.RotaryStability` | phase chord, half-split rotation, normalized logit and frequency-error bounds |
| `Attention.Stability.Softmax` | causal row, average and source XSA phase-error bounds |
| `Attention.Stability.Continuity`, `Model` | joint phase/residual continuity of the complete source |
| `Attention.Stability.Periodic` | genuine rational-pi periods and simultaneous finite logit approximation |
| `Attention.Stability.Classification` | finite-context same-depth classifier replacement and quantitative tied readout certificate |
| `CRASP.FixedError` | conditional quantum error, no-underflow criterion and saturation counterexample |
| `Attention.Stability.Quotient`, `Rounding` | length-independent coefficient error, exact source-aggregate connection and denominator positivity |

Validation of A1--A6: full `lake build` passes (13166 jobs), with no warnings
in the new modules. Individual `#print axioms` checks for all 41 new public
theorems use only `propext`, `Classical.choice` and `Quot.sound`. The audit
remains at 36 existing `sorry`, zero dependent results and zero extra axioms.
The regenerated index has 4069 modules, 21004 theorems, zero vacuous results
and zero placeholders. Forbidden checks and `git diff --check` pass; all
five pinned source hashes still match. Witnesses include nonzero phases,
nonzero rounding errors, saturation, source epsilon values, positive rounded
denominators and strict decisions on all four binary input pairs.

The required Python regression run passed all 418 lab tests with three skips.
The 104-test optimizer suite reproduced the two existing Schedule-Free Muon
allocation failures (float32 and float64), each with 75 allocations of
8519680 bytes. Schedule-Free AdamW tests pass. No Python implementation was
changed. All validation logs, source review and the theorem axiom-check
script are retained in `.kanban/artifacts/128/`.

## Local attention routing (2026-11-01)

The user's proposed mechanism is per-position retrieval and composition:
each query forms a weighted sum of relevant earlier value vectors, and
successive feed-forward/residual blocks combine these pieces. A context of
N tokens has N separate states of width W. A whole-table interpolation
construction with exponential width is a sufficient universality result;
it is not a necessary width for structured arithmetic. The next construction
target is polynomial resources, ideally width independent of chain length
at fixed prime with growing active depth. That construction and its learning
guarantee remain open in tasks#139 and#133.

The following local bounds are formalized in
`Transformer.SequenceModel.Attention.Retrieval`. They use the existing actual
source `causalWeight` and `headScore`, including positive RMS epsilon and
arbitrary rotary phases. Causal softmax includes the current position before
the source XSA operation; a useful set may consist solely of earlier tokens.

### LR1. Weighted pieces and retrieval error

Fix one causal row i, coefficients alpha_j, values v_j and target t.
The source coefficients are nonnegative and sum to one. If
`norm(v_j-t) <= e_j` at every position, then

```
norm(sum_j alpha_j*v_j - t) <= sum_j alpha_j*e_j.
```

For a useful set S, let alpha=sum_(j in S) alpha_j. If useful values have
error at most g and remaining values error at most b, the error is at most
`alpha*g + (1-alpha)*b`. Multiple positions may support the desired feature;
no unique best position is assumed. These are bounds on the raw attention
average, before XSA, output projection and Peri-RMSNorm. Propagation through
those operations needs the existing source stability/execution bounds; a
small raw error alone is not a final classification certificate.

### LR2. Useful mass under actual Q/K normalization

Let N=i+1 be the visible prefix size and r=card(S). For scores bounded in
absolute value by B, the useful mass is at most `r*exp(2B)/N`. For the actual
source normalized rotary head of positive width d and positive epsilon,
the proved score bound gives B=sqrt(d). Hence

```
sum_(j in S) alpha_j <= r*exp(2*sqrt(d))/N.
```

The upper bound can exceed one. Future positions contribute zero, so the
bound remains valid without assuming every member of S is visible. A larger
useful set changes the bound: selecting one distinguished token and
aggregating a broad set are different local requirements.

### LR3--LR4. A necessary local score/width budget

Suppose S is nonempty and reaches mass at least mu>0. Then

```
log(N*mu/r) <= 2*sqrt(d),
max(0, log(N*mu/r))^2 <= 4*d.
```

For fixed r and mu this necessary single-head concentration condition grows
as log(N)^2, not exponentially in N. It is not a sufficient construction,
a task-specific width lower bound, or a multi-layer impossibility theorem.
It does not assert that training creates the required query/key scores.
In particular, it supplies no unconditional constant-width routing guarantee
for arbitrarily long contexts in this source with bounded normalized scores.

| Claim | Lean declaration |
| --- | --- |
| Nonnegative coefficients | `causalWeight_nonneg` |
| Weighted retrieval error | `retrieval_error_le` |
| Useful-set error | `retrieval_error_of_useful_set` |
| Bounded-score useful mass | `useful_mass_le` |
| Actual normalized useful mass | `normalized_useful_mass_le` |
| Necessary log score budget | `retrieval_log_budget` |
| Necessary squared-log width budget | `retrieval_width_budget` |

Each theorem with hypotheses has a concrete witness. These local statements
refine what attention must accomplish; they neither assume nor establish an
optimizer convergence theorem. The source-model experiments are intended to
find counterexamples to current learning interpretations. The frozen early
screen, subsequent profiling and bounded long-grokking phase are tracked in
`EXPERIMENT_PLAN.md`, tasks#138/#140--#143. A finite plateau is not permanent
failure, and a transformer succeeding at a flat task would invalidate an
impossibility inference, not the exact nuclear-norm completion theorem.

Validation of LR1--LR4: `lake build` passes13258 jobs. The seven new
theorems use only propext, Classical.choice and Quot.sound. The audit has
the unchanged36 sorry, zero results resting on sorry and zero extra axioms.
The regenerated index contains4161 modules and21367 theorems, with zero
vacuous statements and placeholders; the forbidden-pattern check passes.
