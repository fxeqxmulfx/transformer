# Convex channels

Moved unchanged from `review/flat_spectrum.md` on 2026-11-19. These sections separate the
flat-spectrum obstruction of that note from convex learning: recall and categorical channels
trained by ordinary answer cross-entropy, their capacity, what the training answers identify,
and convex spectral upper bounds. Flat spectrum enters only as a capacity bound. The thread
continues in [`moments.md`](moments.md).

The labels CR, CC, B, MF, CFt, CF, O, CI and SB are this file's. Its CC1–CC8 and B1–B3 are
not the codec items CC1–CC7 or the rounding certificates B1–B9 of
[`../flat_spectrum.md`](../flat_spectrum.md), to which Theorem A, R, AR, WD, XE and the other
earlier labels refer. CM is in [`moments.md`](moments.md).

## Convex answer-loss recall

Added 2026-11-11. This construction separates the flat-spectrum obstruction
from a blanket obstruction to convex learning. R2 and AR3 concern nuclear-norm
completion of an answer table. They do not quantify over all convex models
that receive the dictionary as input. The construction below uses ordinary
answer cross-entropy, not an auxiliary key-matching or calibration loss.

### CR1. Model and objective

Let `κ` be a finite key set, `ν` a finite value set, and `S` the finite training
sample set. The learned parameters are **all entries** of `A ∈ ℝ^(κ×κ)`,
constrained by `A(q,k) ≥ 0` and `Σ_k A(q,k) = 1`. For an input query `q` and
input dictionary `D : κ → ν`, the class logits are

```
z_A(q,D,c) = Σ_k A(q,k) 1[D(k) = c].
L(A) = Σ_s (log Σ_c exp z_A(q_s,D_s,c) − z_A(q_s,D_s,D_s(q_s))).
```

The dictionary's values are input features; the label is its queried value.
The forward pass never receives that label separately. Categorical value
copying and the key identities are fixed. The matrix is learned directly,
without a factorization, learned embedding, attention softmax, or nuclear-norm
penalty. Its `m²` entries, for `m = |κ|`, share across all `|ν|^m` dictionaries;
there is no learned parameter per dictionary. This is an explicit, restricted
architecture, not a convex replacement for an arbitrary transformer.

Every logit is nonnegative and at most one. At `A = I` the logits are the
one-hot vector of `D(q)`. The constraints bound confidence: the best correct
softmax probability is `e / (e + |ν| − 1)`. Perfect argmax accuracy is possible;
zero cross-entropy is not claimed. Unlike unbounded separable logits in XE6,
this compact feasible set admits a finite minimum without weight decay.

### CR2. Joint convexity

The row-simplex domain is convex. For every fixed input, the complete logit
vector is linear in **all** entries of `A`. Ordinary softmax cross-entropy is
convex in its logits, by the first-order inequality of XE1. Therefore `L` is
jointly convex, even on the unrestricted matrix space. Its restriction to
the stated convex domain is a convex optimization problem, including the
usual cross-entropy loss.

### CR3. What answer-only training identifies

Assume `|ν| ≥ 2`. Say the input dataset **separates keys** when

```
∀ q k, k ≠ q → ∃ s, q_s = q and D_s(k) ≠ D_s(q).
```

This is a checkable condition on supplied inputs. It mentions no learned
parameters, matching labels, test outputs, or successful training.

For any nonnegative logits `z` with `z_t ≤ 1`, each competitor's relative
logit `z_c − z_t` is at least that of the ideal one-hot vector `e_t`.
Consequently `CE(e_t,t) ≤ CE(z,t)`. If `z_t < 1`, a competitor exists and the
inequality is strict. Applying this to every sample gives `L(I) ≤ L(A)` for
every feasible `A`. Thus `I` actually attains the minimum.

If `L(A) ≤ L(I)`, every sample has the ideal loss and its target logit is one.
Nonnegative row mass then vanishes on any key whose input value differs from
the target. Separation supplies such a sample for every off-diagonal entry,
so all off-diagonal entries vanish. Row sums force `A(q,q) = 1`, hence `A = I`.
The optimum is unique, and every optimizer recalls **every** test dictionary,
including those never supplied during training, with a strict correct argmax.

### CR4. Small separating data and a negative control

An injective code `code : κ → (ι → ν)` constructs separating input data:
use dictionary `D_r(k) = code(k,r)` at each coordinate `r`, queried at every
key. The code is only a way to construct data; it is not an additional input
to the model. In the formal example, four keys are Boolean pairs, and the two
coordinate dictionaries give eight training samples. The theorem applies to
all sixteen Boolean dictionaries, not only those two. More generally binary
codes allow `m b` samples when `m` keys have distinct `b`-bit codes.

Without separation, convexity does not guarantee generalization. For two
keys, let sample `s ∈ Bool` query key `s` in the constant dictionary `D_s = s`.
Both labels occur. The matrix that swaps the two keys is feasible and attains
the global minimum of the same ordinary answer CE. On the test dictionary
`D = id`, it gives a strictly wrong argmax. This is a proved counterexample,
not a failed optimization experiment.

### CR5. Causal MQAR

For the aligned pairs of Zoology §4 / Appendix, use

```
z_A(x,i,v) = Σ_{j<i} A(query_i,key_j) 1[value_j = v].
```

This is also linear in the complete parameter matrix. After CR3's training,
`A = I`, and it equals the existing paired causal attention operator exactly.
The repository's `paired_attention_solves_mqar` theorem therefore proves that
it produces every prior-binding answer on every distinct-key stream, at
any length. It returns the zero vector when no key matches, as specified by
`expectedPairedAnswer`; no learned null-class decision is claimed. Training
and stream values may use different alphabets because only key matching is
learned. The theorem assumes the key/value pairing shift and token identities
are already supplied. Repeated-key overwrite semantics and raw-token parsing
are outside this result. The training theorem is for complete dictionary
inputs; arbitrary partially observed MQAR training streams are not assumed
to satisfy it.

### CR6. Flat spectrum and the scope of the earlier obstruction

For a fixed scalar readout `φ : ν → ℂ`, define the model's output table by
`K_A(q,D) = Σ_c z_A(q,D,c) φ(c)`. CR3 implies `K_A(q,D) = φ(D(q))` at every
optimizer. Thus AR2 applies: if `Σ_c φ(c) = 0`,

```
K_A K_Aᴴ = (|ν|^(|κ|−1) Σ_c |φ(c)|²) I.
```

For a nonzero readout it has rank `|κ|` and a positive flat singular scale.
The model keeps the capacity needed to carry every key's binding, consistent
with AR4's width lower bound. It does not force a low-rank context summary.
For `φ = ±1` and two Boolean keys, the squared singular value is four.

If every `φ(c)` is nonzero, AR3 still says this very same trained kernel is
**not** a nuclear-norm minimum from any proper observation set. Both the
positive convex-learning theorem and the negative nuclear-completion theorem
are proved together. The nuclear-norm results have not been disproved; using
them to exclude all convex dictionary-aware learners would exceed their
hypotheses. The old calibration-based `Transformer.ConvexRecall` construction
also did not establish ordinary answer-CE training. The present construction
supplies that narrower missing result. It makes no claim to learn a general
flat arithmetic task such as the cubic kernel from two bare numeric tokens.

### CR7. What convergence is proved

The feasible matrix set is closed and contained in the finite coordinate
cube `[0,1]^(κ×κ)`, hence compact. `L` is continuous. Under CR3's separation
condition, **every feasible sequence `A_n` with `L(A_n) → L(I)` satisfies
`A_n → I`**: continuity makes each cluster point an optimizer, and uniqueness
forces every cluster point to be `I`. Since the limiting logit gap is one and
the input and class sets are finite, eventually **all** dictionary queries
have a strict correct argmax, uniformly over test dictionaries.

This is a theorem about minimizing sequences. It does not prove that a
specific projected-gradient, mirror-descent, SGD, or AdamW implementation has
convergent loss, nor give a convergence rate. That algorithmic obligation
remains separate; no optimizer run or empirical result is reported here.

### Lean

The modules under `src/Transformer/Grokking/FlatSpectrum/ConvexRecall/` prove
these statements with ordinary `Landscape.Neural.crossEntropy`:

| Item | Principal declarations |
| --- | --- |
| CR1–CR2 | `IsStochastic`, `logits`, `answerLoss`, `convex_stochastic`, `answerLoss_convex` |
| CR3 | `Separates`, `isMinOn_answerLoss_one`, `eq_one_of_answerLoss_le`, `trained_logits`, `trained_strict_argmax` |
| CR4 | `separates_of_injective_code`, `twoBitCode_injective`, `wrong_global_minimum_without_separation` |
| CR5 | `streamLogits_linear`, `trained_solves_mqar` |
| CR6 | `trained_outputKernel`, `trained_isFlat`, `trained_rank`, `trained_not_nuclear_minimum` |
| CR7 | `isCompact_stochastic`, `continuous_answerLoss`, `tendsto_one_of_loss_tendsto`, `eventually_all_strict_argmax` |

Reproduce with native Lean tooling:

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexRecall
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexRecallAxioms.lean
```

## Convex channels and capacity

The next extension learns generated outputs as well as routing. Flat spectrum
is used as a necessary capacity bound. It is not an arithmetic feature map,
an answer oracle, or a sufficient condition for generalization. The inputs
remain categorical query identities and supplied key/value bindings. The
earlier compact joint-embedding transformer replacement remains a separate,
unresolved requirement.

### CC1. A joint parameter domain

For finite query, key, value and output sets `Q,K,V,C`, directly train
`W(q,k,v,c)` subject to

```
W(q,k,v,c) ≥ 0,
Σ_c W(q,k,v,c) = ρ(q,k), independent of v,
Σ_k ρ(q,k) = 1.

z_W(q,D,c) = Σ_k W(q,k,D(k),c),
L(W) = Σ_s CE(z_W(q_s,D_s), y_s).
```

`ρ` is the output marginal of `W`, not a separately multiplied parameter.
The implementation stores the joint tensor, with `|Q||K||V||C|` entries.
The forward receives input bindings and a query, never the answer or the
rule used to construct labels. The loss is ordinary log-sum-exp cross-entropy.
Summing class logits gives one for every dictionary: the common routing
mass lets each key use its own observed value. Logits lie in `[0,1]`, so the
confidence limitation of CR1 remains: ideal CE is `log(e+|C|−1)−1`.

### CC2. Joint convexity

Nonnegativity and affine equalities make the entire feasible set convex.
For fixed inputs, `z_W` is linear in every trained entry. XE1's logit
convexity therefore makes ordinary answer CE jointly convex in routing and
output generation. A product of separately learned routing and decoding
matrices is neither needed nor used.

### CC3. Copying is a strict subfamily

Every stochastic matcher `A` from CR1 embeds as
`W(q,k,v,c) = A(q,k) 1[c=v]`, with exactly the old forward logits. The enlarged
family also contains, for every categorical rule `g : K×V → C`,

```
W_g(q,k,v,c) = 1[k=q] 1[c=g(q,v)],
z_Wg(q,D,c) = 1[c=g(q,D(q))].
```

This constructs a parameter witness; `g` is not part of the model's input
features or forward code. For two Boolean keys, `g(q,v)=not v` produces
`true` on an all-false dictionary. No old copying matrix can do so.

### CC4. General categorical two-input capacity

With a single context slot, `W_f(q,(),v,c)=1[c=f(q,v)]` represents every
finite two-input classification table. This is a representation theorem.
It does not infer the labels of unobserved pairs. With several slots the
family sums their contributions; it is not a universal model of arbitrary
multi-slot interactions. New interactions would require additional input
features and a fresh capacity/identification analysis.

### CC5. What answer-only training can identify

Assume `|C|≥2`, labels `y_s=g(q_s,D_s(q_s))`, and the following finite-data
condition, called `SeparatesAtoms`:

```
∀ q,k,v,c, (k≠q or c≠g(q,v)) →
  ∃ s, q_s=q and D_s(k)=v and y_s≠c.
```

Every incorrect routing/output atom must be active in some training input
whose ordinary answer contradicts it. This condition refers to the actual
data and target rule, not to learned parameters or training success. It
requires value coverage beyond the simpler copy-only condition CR3.

As in CR3, `W_g` attains the ideal CE sample by sample. Equal aggregate
loss forces target mass one at each observed sample. Any active atom
predicting another class has zero weight. Separation eliminates all
incorrect atoms; unit mass forces the remaining entries of `W` to equal
`W_g`. Therefore every global optimizer generates `g(q,D(q))` with a strict
correct argmax on every test dictionary. The Lean witness uses all Boolean
two-key dictionaries with negated answer labels and verifies separation.
This theorem is conditional on reaching a global optimum; it is not a
proof of convergence of the native optimization loop.

### CC6. Use flat spectrum as the bound

For a fixed class readout `φ:C→ℂ` and any supplied family `D_x`, the actual
model output has the exact factorization

```
F_W(q,x) = Σ_c z_W(q,D_x,c) φ(c) = A_W B,
A_W(q,(k,v)) = Σ_c W(q,k,v,c) φ(c),
B((k,v),x) = 1[D_x(k)=v].
```

These are task-independent categorical binding features. For any flat
target `T` of squared singular value `σ²>0` and rank `r=min(|Q|,|X|)`, R3 gives

```
‖T−F_W‖_F² ≥ σ² (r−|K||V|).
```

Exact representation requires `|K||V|≥r`. This is an upper feature count,
not a claim that all those features are independent. A further factorization
through width `d` has the stronger corresponding bound `σ²(r−d)`. The bound
concerns scalar-output approximation, not CE or classification error; AR5
is the separate sign/argmax result. It need not impose a positive error
floor once capacity exceeds rank. The feasible one-slot Boolean AND table,
read out as `false↦1,true↦−1`, is `[[1,1],[1,−1]]`: a full-rank flat kernel
with `σ²=2`, realized at width two. No arithmetic features are supplied.

### CC7. Flat capacity does not identify missing labels

The two tables with rows `[[1,1],[1,−1]]` and `[[1,1],[−1,1]]` are both
flat with `σ²=2`. Their categorical labels agree on the entire first query
row and disagree on the second. Both feasible channel tensors globally
minimize the same ordinary CE on observations from the first row. Thus even
known flatness, full rank and sufficient capacity leave two incompatible
held-out predictions. The Lean counterexample proves feasibility, both
flatness equations, both global-optimality statements, observed agreement
and held-out disagreement together. It does not rule out learning with
additional structure or identifying examples such as CC5.

### CC8. An exact training oracle

For a gradient tensor `G`, choose `decode(q,k,v)` minimizing `G(q,k,v,c)` over
classes. Choose `key(q)` minimizing `Σ_v min_c G(q,k,v,c)` over keys. The
vertex

```
S(q,k,v,c) = 1[k=key(q)] 1[c=decode(q,k,v)]
```

globally minimizes `Σ_qkvc G(q,k,v,c) W(q,k,v,c)` on the feasible set. Indeed,
each key's cost is at least its class-minimum sum times its routing mass;
the routing masses are nonnegative and sum to one. This bound is attained
by the selected vertex. The finite search costs `O(|Q||K||V||C|)` and does
not fit a nonconvex inner model. Convex updates `(1−η)W+ηS`, `0≤η≤1`, remain
feasible. These facts justify the native Frank-Wolfe step; no convergence
rate for that implementation is asserted by the Lean result.

### Lean

The eight modules under `Grokking/FlatSpectrum/ConvexChannel/` prove CC1–CC8.
Principal results are `convex_feasible`, `answerLoss_convex`,
`generates_absent_value`, `eq_generator_of_answerLoss_le`, `trained_logits`,
`flat_error_bound`, `flat_width_necessary`, `flat_capacity_does_not_identify`
and `linear_oracle`. Every theorem with premises has a satisfying example.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexChannel
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean
```

## Balancedness as a capacity certificate (added)

This extension isolates what the balancedness update in WD14.6 supplies for
the convex-model question. Let `U : C^(m × k)`, `V : C^(n × k)` and assume
`Uᴴ U = Vᴴ V`. No loss, stationary point, optimizer or successful training is
assumed in B1–B2. The statement concerns a factorization of a matrix; applying
it to a model requires identifying the matrix that the model represents.

### B1. Identities from balance alone

The trace of the common Gram matrix gives `||U||_F² = ||V||_F²`. Also

```
(U Vᴴ)ᴴ (U Vᴴ) = (V Vᴴ)²,
||U Vᴴ||_* = tr(V Vᴴ) = (||U||_F² + ||V||_F²)/2,
rank(U Vᴴ) = rank U = rank V = rank [U; V].
```

The positive semidefinite matrix `V Vᴴ` is a root of the product's Gram
matrix, which proves the nuclear-norm identity. The first three ranks follow
from equality of Gram matrices; the stack has Gram matrix `2 Uᴴ U`, which
has the rank of `U`.

If `n > 0`, the stack rank is at most `m < m+n`. Thus an exactly balanced
start cannot satisfy WD5's full-stack row-rank hypothesis. This only limits
that sufficient theorem; it does not imply that all balanced starts fail or
that discrete gradient steps preserve balance. The stationary points in
WD14 satisfy the hypothesis by its separate stationarity argument.

### B2. Use flatness as a bound on actual rank

For `IsFlat K c`, `c > 0`, put `r₀ = min(m,n)`. R3 and B1 give

```
c (r₀ - rank U) ≤ ||K - U Vᴴ||_F².
```

The subtraction here is in the real numbers. Allocated width can exceed
the actual rank: adding zero columns does not remove this obstruction.
Equivalently, an error budget `||K-U Vᴴ||_F² ≤ ε` requires
`rank U ≥ r₀ - ε/c`; this is a necessary capacity condition.
The theorem does not identify unseen labels, certify optimization, or
guarantee that a particular representation attains the lower bound.

### B3. Balance is neither a convex constraint nor an optimum certificate

Even for real scalars, the balanced set `{(u,v) : u² = v²}` is not convex:
`(1,1)` and `(1,-1)` belong to it, but their midpoint `(1,0)` does not.

For a nonzero stationary example with a convex loss, take
`K = (3/2) I₂`, `L(X) = ||X-K||_F²/2`, `λ = 1/2` and
`U = V = diag(1,0)`. The actual product gradient is
`G = U Vᴴ-K = diag(-1/2,-3/2)`. Both factor gradients vanish:
`G V + λ U = 0` and `Gᴴ U + λ V = 0`. The factors are balanced and
have rank one. The target is flat with `c = 9/4`, so B2 gives an error
lower bound of `9/4`. Nevertheless their decayed loss is `7/4`, strictly
above the `5/4` attained by the balanced same-width competitor `U = V = I₂`.
Thus even nonzero balanced stationarity does not certify optimality.
This squared-loss witness refutes a loss-independent inference; it is not
a cross-entropy experiment or a replacement for the existing ordinary CE.

The existing convex-channel learner optimizes the joint coefficients directly.
It has no separately trained `U,V` whose balancing would repair its predictions.
An exact refactorization preserving the represented matrix also preserves
every linear readout of that matrix. B1–B3 support checking effective capacity
and the hypotheses of a factorized optimizer. The minibatch oracle bias,
query-specific missing records and fixed logit scale discussed in the
experiment README need their own remedies and proofs.

Lean: `BalancedFactorization.lean` proves B1–B2, including the stack-rank
identity and full-row-rank obstruction. `Training/Scope/BalanceObstruction.lean`
proves B3 and checks the numerical matrix witness, its actual rank, flat target,
stationarity, both losses and the resulting failure of `IsDecayedMin`.

## Repairing minibatch and full-batch failures (added)

### MF1. Exact singleton CE oracles can fail forever

Take two Boolean inputs `D_b(false) = false`, `D_b(true) = b`, with answer
`b`. For any current joint channel `W`, one sample's ordinary CE gradient is
`G_b(k,v,c) = 1[v=D_b(k)] (softmax(logits_W(D_b))_c - 1[c=b])`.
The Lean first-order inequality checks this tensor against ordinary CE.
At the observed value, class `b` minimizes the residual. All other value
rows have zero gradient, so every key has the same best singleton cost.
The first-key tie rule is therefore an exact linear minimization oracle.

After its initial step of size one, this rule places all routing mass on
the constant slot. Every subsequent convex update with another first-key
vertex leaves the separating slot at zero, regardless of label order and
retained choices on unobserved rows. Both inputs receive the same logits.
Any deterministic Boolean readout is exactly 50% accurate on the two inputs.
Yet the feasible vertex routing to the second slot with decoder `id` assigns
the correct class logit one and the other class zero on both inputs.

`ConvexChannel/Minibatch/Gradient.lean` and `Failure.lean` prove this exact
two-key/two-class instance. The native frozen regression uses three classes
and the larger causal adapter and exhibits the same constant-key failure.
No universal stochastic-convergence conclusion follows from an exact oracle
for each noisy minibatch objective.

### MF2. Retain sample gradients, not only decoder choices

Store one gradient per observed training example and replace it on revisit.
Replacing entry `g_i` by `g'_i` updates the aggregate by `g'_i-g_i`; repeated
visits do not increase an example's weight. If all entries are refreshed at
one model state, this is its full empirical gradient. With minibatches,
entries may instead describe different earlier states.

For MF1, even two arbitrary evaluation states `W_0,W_1` suffice: the
separating vertex minimizes `G_false(W_0)+G_true(W_1)`. Every constant-answer
vertex has strictly larger cost, because the probability assigned to the
other class is strictly positive. Thus this memory removes the demonstrated
singleton tie. `Minibatch/Memory.lean` proves this statement and the exact
replacement identity. It does not prove general convergence with stale gradients.

For a sparse implementation, subtract the class-independent softmax background
from each sample gradient. On a nonempty value type, a tensor
`B(k,v,c)=1[v=D(k)] beta` has linear cost exactly `beta` against every feasible
joint channel. Therefore subtracting this baseline preserves oracle comparisons
and FW gaps. The constant-cost identity is also checked in `Memory.lean`.

### MF3. Shared joint parameters address a separate full-batch failure

The earlier full-batch recall model fitted observed labels, but each query had
an independent decoder. An unseen query/value pair could keep its default even
when another query had learned that value. This is a structural identification
problem, not a nonzero empirical optimization gap.

Fix an input-only slot permutation `align(q)` and train one joint tensor `A`.
Its expansion is `W(q,k,v,c)=A(align(q)(k),v,c)`. Equivalently it reads the
aligned dictionary `D(align(q)^(-1)(j))`. The expansion preserves all feasibility
constraints and is linear in `A`. Ordinary CE remains convex, including after
multiplication of logits by any fixed scale. Equal aligned dictionaries have
equal predictions under different queries. `ConvexChannel/Shared.lean` proves
these statements. A positive fixed scale preserves a model's argmax ordering;
learning the scale jointly is not covered by this convexity result.

The intended causal alignment leaves lag slots fixed and relabels each
token-indexed successor/presence block relative to the visible final query.
All output classes and value decodings remain learned. This explicitly shares
information across queries; flat spectrum alone does not imply that sharing
is appropriate for an arbitrary task. B1–B3 remain capacity and scope checks,
and do not supply optimization or missing-label identification by themselves.

### MF4. Prefer sharing without discarding query-specific capacity

Mandatory sharing is a restriction, not a general repair. For two Boolean input
slots `D(a,b)=(a,b)`, every shared additive logit satisfies the rectangle identity
`z(0,0)+z(1,1)=z(0,1)+z(1,0)`. Applying it to the difference between two class
logits rules out strict XOR classification on all four inputs, even without
feasibility constraints. The full query-indexed domain, with visible query `a`,
represents XOR by selecting the second slot and learning `decode(a,b)=a xor b`.
This is a capacity witness, not an arithmetic feature in the stress model.

Retain the original convex domain and compute its exact query-wise linear
oracle `V`. An aligned shared feasible candidate `S` can replace it whenever
`<G,S> <= <G,V>`. Since `V` minimizes the linear objective on the full domain,
the replacement is also a minimizer. Convex combinations with either candidate
remain feasible, and the ordinary CE forward map stays linear in all joint
coordinates. `ConvexChannel/Adaptive.lean` proves the XOR obstruction, the
unrestricted witness, and this oracle preference. It does not guarantee that
the resulting optimizer discovers every rule or identifies arbitrary unseen
labels. A forward-only learner may similarly use a shared consistent solution
when one exists and otherwise retain the query-specific consistency family.

### MF5. Complete unseen rows without changing the oracle cost

The unrestricted fallback can still keep a default on an unobserved query/value
row even when other queries have learned that value. If `G(q,k,v,c)=0` for every
class on such a row, replace its decoder choice by a pooled choice. This leaves
the linear cost of every routing vertex unchanged, so an exact oracle remains
exact. The completed vertex and its convex updates remain feasible in the
original domain. No product of separately trained factors is introduced.

For any finite retained residual sum
`G(q,k,v,c)=sum_s 1[q_s=q and D_s(k)=v] r_s(c)`, a row absent from every consumed
sample is zero, regardless of the states at which residuals were evaluated.
`ConvexChannel/Completion.lean` proves this support fact, cost invariance and
oracle preservation. The missing-row mask must be derived from consumed
training inputs. It must not mark an observed row just because its current
gradient happens to be small. The shared decoder's correctness on an unseen
row is additional structure; this theorem supplies no arbitrary-label guarantee.

The native completion uses the pooled oracle's decoder for the same aligned
key, only for values absent from that query's retained observations. It does
not replace observed choices, alter oracle costs or read held-out answers.

## Causal features and convex capacity

### CFt1. Add features without discarding the old model

For any additional input-only slots `L`, embed a channel on `K` into one on
`K + L` by setting the new slots' joint weights to zero. Nonnegativity and
value-independent routing mass remain true; the total routing mass stays one.
If the value alphabet also grows, first pull back the old rows through a fixed
retraction: old values retain their rows, and each new unused value reads any
fixed old row. Feasibility and predictions on the embedded old inputs are
preserved. The key extension then applies on this common enlarged alphabet.
On every input, the new slots contribute zero, so the old logits and ordinary
CE are exactly preserved. The enlarged domain remains convex and the logits
remain linear in every learned coordinate, for any fixed feature extractor.
Consequently the optimal empirical loss cannot increase. This is a capacity
claim; a particular optimizer or tie rule can still generalize differently.

### CFt2. Query-specific coordinates do not remove additive obstructions

Fix one query and expose two independent Boolean slots. Even an unconstrained
query-specific channel obeys the rectangle identity of MF4 on that query.
Therefore it cannot strictly classify XOR on all four inputs. Adding more
iterations or increasing a fixed positive logit scale cannot remove that
identity. This small counterexample is not claimed to be a proof about every
feature of the native stress adapter; the latter has additional causal slots.

### CFt3. Counts add interactions without learned-factor products

Append the occurrence count of each token in the visible prefix. The extractor
receives only that prefix and the vocabulary, with no labels, arithmetic
operation or task identifier. A count is between zero and the context bound,
so it is a finite categorical value in a bounded-context experiment. The
usual joint tensor can select that slot and learn any output rule of its count.
In particular, the count of `true` represents two-bit XOR with a unit margin,
strictly extending the separated-slot model of CFt2. Parity appears only in
the mathematical parameter witness; the executable learns its decoder from
ordinary labels and does not compute a parity answer in its feature map.

`ConvexChannel/Counts.lean` proves the embedding, exact predictions and loss,
convex CE, fixed-query obstruction, context bound and count-rule witness.
The generic finite categorical flat-spectrum bound CC6 still applies to the
enlarged feature alphabet. Counts do not identify labels at unseen counts,
and do not assert sufficiency for order-sensitive tasks or modular division.

### CFt4. Pair visible features; keep selection assumptions explicit

Given a categorical dictionary `D : K -> V`, expose the ordered pairs
`(D(k), D(l))` as fixed features indexed by `K x K`. This map reads input
values only, with no target rule or output labels. Any decoder of two slots
then has a feasible single-key vertex on the paired alphabet. Its actual
logits are the decoded one-hot answer. The full channel tensor still has
convex feasibility and ordinary full-vocabulary CE; paired feature discovery
or candidate selection is not asserted jointly convex. CFt1 permits retaining
the original features alongside the extra pairs, and the generic finite
categorical CC6 capacity bound applies to the enlarged alphabet.

The CF1 learner can scan paired dictionaries with the original raw labels.
If an enumerated pair explains the observations, the actual scan succeeds.
For its selected pair, an observed matching pair value transfers its observed
answer to a new dictionary, conditional on the labels following that pair's
rule. A good pair somewhere in the catalogue does not establish coverage or
correctness of whichever other pair the scan selects. The four Boolean-pair
examples, with the pair itself as their four class names, eliminate every
individual-key decoder while the paired scan succeeds. The rule occurs only
in the specification and witness weights, never in the feature extractor.

The existing token-presence feature is unchanged by permutations of the
visible prefix. Any channel using only these features has the same logits
under that permutation when its query is fixed. This invariance can motivate
a candidate-order prior for position-invariant tasks, but observed consistency
alone does not justify that prior. Two constant dictionaries with labels
`false` and `true` fit both individual Boolean keys. Trying one key first or
the other first produces different predictions on the unseen dictionary whose
two slots differ. Both learners fit all observed labels and cover both input
values of their chosen key. Distinguishing the intended target still requires
structure or separating observations, as in CF4.

`ConvexChannel/CausalPairs.lean` proves these claims and nonempty examples.
It supplies a representation and selection boundary for a future generic
causal-feature repair; it does not establish an original stress-task pass,
a practical search cost, or extrapolation to unseen pair values.

### CFt5. Bind the complex initializer to the convex head

Fix a finite categorical state set I, a finite raw class vocabulary C and a
unitary code matrix U whose rows are state features. For a recorded state
assignment y : I -> C and a real margin a, let T[s,c] be a when c=y(s), and
zero otherwise. Construct H = U* T by the conjugate transform. Its actual
real logits are Re(sum_k U[s,k] H[k,c]); splitting H into real and imaginary
coordinates and using features (Re U, -Im U) gives exactly the CD8 head.
Both components affect predictions. U H = T follows from U U* = I, so the
initializer reproduces its declared state margins and has ordinary CE
log(1+(|C|-1) exp(-a)) at every assigned state. If a is positive, its declared
class has strictly larger logit than every other class.

Unitary multiplication also gives ||H||_F^2 = ||T||_F^2 = |I| a^2. For each
class, Cauchy--Schwarz on its 2|I| real coordinates bounds the squared L1
norm by 2|I|^2 a^2. In particular, every column has L1 norm at most
2|a||I|. The native initializer with margin24 therefore lies in the CD8
head domain of radius64|I|. This is a sufficient budget bound, not an
assertion that the initialized or trained head has a flat spectrum.
Every unitary-code entry has modulus at most one, so both real feature
coordinates have absolute value at most one. A CD8 head with column budget R
therefore has absolute logits bounded by R after any feasible update.

For every nonzero N, take U = F_N/sqrt(N), with F_N the existing character
Fourier matrix. The proved identities F_N F_N* = F_N* F_N = N I give
unitarity and singular values exactly one. Each N-state code has entry
modulus 1/sqrt(N), nuclear norm N and squared Frobenius norm N. A finite
block diagonal is unitary even when the query blocks have different sizes.
Taking I as the disjoint union of their categorical states gives the native radius
64 times the total number of states, with the same exact logit/CE formulas.
The generic theorem uses any unitary code; no arithmetic target is used.

The assignments y are the learned observed dictionary entries plus the
explicit unknown-state default. This construction certifies what the head
computes from those assignments. It does not prove that an arbitrary new
prefix uses the correct selected feature, that an unknown default is true,
or that discovering the feature map is jointly convex. CFt4's counterexample
continues to apply. Numerical floating-point audits of saved native states
remain separate from these exact-real identities.

Lean: `ConvexDepth/ComplexHead.lean` formalizes the actual coordinates,
inversion, CE and logit bound; `ComplexHeadBudget.lean` proves the energy and
L1 budget; `FourierCode.lean` supplies arbitrary nonzero Fourier dimensions
and unequal query blocks. These statements correspond to
`experiments/convex_stress_causal_pairs/src/model.rs` and
`model/initialize.rs` at `7ce28651`. The integer categorical state is merely
the recorded dictionary position, not the numerical meaning of an input.

## Forward-only channel learning

This extension gives a finite categorical learner for the realizable
generated-recall family of CC3 and CC5. It uses ordinary answer labels and
constructs a global minimum of the same answer cross-entropy as CC1. No
gradient, backward pass, contrastive surrogate or optimizer convergence
assumption is needed. It does not solve arbitrary noisy or multi-key
classification in the larger convex channel family.

### CF1. Record answers and eliminate conflicting keys

Group the training dictionaries and labels by query. For each query, try
the candidate keys in a supplied finite order. For one candidate `k`, start
with an empty partial table from values to classes, then scan its examples
from left to right:

1. Read the input value `v=D(k)` and the ordinary answer `y`.
2. If `v` has not been seen, record `v -> y`.
3. If its recorded answer is `y`, retain the table.
4. Otherwise eliminate `k` permanently.

Return the first surviving key and its recorded decoder. Complete unseen
values with a fixed, supplied default class. If every candidate is
eliminated, return `none`. The target key and rule are never algorithm
inputs. `learn` scans the data once per attempted key; it does not require
keeping every candidate's table live simultaneously.

The Lean state is `Option (V -> Option C)`: the outer `none` is an
eliminated candidate; the inner `none` is an unobserved value. `step`,
`scan`, `trainKey` and `learn` are executable definitions. Functional
updates are a mathematical table representation, not a claim of constant
time memory access in the compiled Lean implementation.

### CF2. Exact scan invariant and explicit failure

Let a decoder extend a table when it agrees with every recorded entry.
After each observation, the surviving decoders are exactly the previous
ones that predict this answer from this candidate value:

```
Allows(step(state,v,y), f) <-> Allows(state,f) and f(v)=y.
Allows(trainKey(k,samples), f) <-> every (D,y) in samples has f(D(k))=y.
```

Thus any realizable candidate survives; every returned decoder fits all
observations. Two samples with the same candidate value and different
labels eliminate that candidate. The learner returns `none` exactly when
no supplied key has a decoder fitting every observation. This is failure
of exact single-key realizability, not infeasibility of the convex domain
or proof that its best approximate classifier is poor.

### CF3. A constructed global ordinary-CE optimum

Suppose every query's row is realizable by some enumerated key and decoder.
The computed learner returns `(k_q,f_q)` and hence the feasible channel

```
W(q,k,v,c) = 1[k=k_q] 1[c=f_q(v)].
```

All training samples have unit target logits. CC5's sample lower bound
then proves global minimality of ordinary answer CE over **all** feasible
joint channels, including nondeterministic ones. The minimum CE retains
the bounded-logit value `log(e+|C|-1)-1` per example, which is positive
when there are at least two classes.
This is finite completion of the specified learner, rather than an
assumption that an iterative optimizer reached a minimum.

`learnedChannel` uses an explicit fallback to make the mathematical tensor
total. Its optimality theorem assumes realizability and proves a successful
return on every row, so the fallback is unused. The executable `learn`
continues to report failure explicitly. The finite CE sample family must
be included in the rows actually supplied to the learner.

### CF4. Conditions identifying held-out answers

For the target `g(q,D(q))`, assume:

* the correct key is enumerated and all supplied labels follow this rule;
* every value at the correct key appears in the query's training row;
* for every wrong candidate key, some two observations have the same value
  at that key and different answer labels.

One collision per wrong key eliminates it. Coverage at the true key then
fixes its complete decoder. The learner therefore returns exactly
`(q,g(q,·))`, and the assembled channel equals `generator g`. It answers
every dictionary, including unobserved combinations, correctly. The
theorem does not infer labels for uncovered true-key values.

The Boolean witness trains negation from two of the four dictionaries
per query: the first has `false` at the queried key and `true` elsewhere;
the second has `true` at both keys. Their labels are `true` and `false`.
Each wrong key is constant across these examples and therefore conflicts.
The learned model produces `true` on the held-out all-false dictionary,
although that value is absent from the input. Reading its actual outputs
with `false -> 1, true -> -1` over all four dictionaries gives a flat
rank-two kernel with squared singular value four (`bool_learned_flat`).
Separate inconsistent data produce explicit failure. Flat spectrum supplies
no extra identifying information to this procedure.

### CF5. Counted work and scope of the efficiency claim

`countedScan` increments a counter for every executed call to `step`.
`countedLearn` accumulates these counters while running the actual key
selection. Its returned result equals `learn`, and its count is at most

```
keys.length * samples.length
```

for each query row. If each key is listed once, this proves an abstract
`O(|K| N)` bound on candidate observations over `N` total training examples.
Each successful result can
be represented by one key and a categorical decoder, with no dense
`|Q||K||V||C|` floating-point parameter tensor. An array-backed port could
perform constant-time record accesses and retain at most one candidate
table per query during the sequential search. Those runtime and storage
claims need a separate native implementation and measurements; the Lean
counter does not model allocations, lookup cost or wall-clock time.

The theorems under `ConvexChannel/Forward/` establish the invariant,
success/failure characterization, exact CE optimality, identification,
held-out Boolean witness and operation count. No claim is made here about
the proposed soft multiplicative update for inconsistent data.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean
```

### CF6. Persistent minibatches and exact boundary invariance

Retain a partial record and an elimination marker for every `(query,key)`
candidate. Each batch updates every candidate using CF1's `scan`. A snapshot
selects the first surviving key in the fixed candidate order and supplies the
same default on unseen values, without modifying any records. In particular,
neither selecting a key nor predicting a default discards information needed
by later batches.

For any initial candidate state, scanning `xs ++ ys` equals scanning `xs` and
then continuing on `ys`. Induction over batches proves exact state equality
with scanning their concatenation. The executable minibatch learner therefore
returns exactly the same key and decoder, or the same failure, as `learn` on
the flattened observations. This holds for empty batches, unequal batch sizes,
partial coverage and inconsistent labels. It requires the same ordered examples,
candidate order and default; it does not assume realizability. Assembled channel
tensors are equal too, so CF3's CE optimality and CF4's identification conclusions
transfer when their hypotheses hold for the combined observations.

A concrete Boolean example exposes premature selection: the first batch has
dictionary `(false,false)` and label `true`; a snapshot selects key `false`.
The next nonempty batch has dictionary `(false,true)` and label `false`.
Key `false` is now contradictory. Key `true` survives and decodes negation,
using the first batch's `false -> true` entry as well as the new entry.
Keeping only the first snapshot would lose that alternative's earlier evidence.

An array implementation needs `|Q||K||V|` optional categorical entries and
`|Q||K|` elimination flags, independent of the number of examples. It visits
all candidates, rather than CF5's early stopping between candidates. Its
runtime and memory constants are measured separately in
`experiments/convex_channel_minibatch`; batch-boundary invariance alone is
not a claim that streaming is faster or that an arbitrary prefix generalizes.

## Observable features determine unseen predictions

This extension supplies a sufficient identification condition in addition to
CC6's capacity bound. It neither removes the CC7 counterexample nor assumes
that a flat spectrum determines unobserved labels. All channels and losses
are the unfactorized model and ordinary cross-entropy of CC1-CC2.

### O1. Training labels determine observed logits

Let the finite training data be `(q_s,D_s,y_s)`. Assume a feasible reference
channel `A` realizes these labels at unit logit scale:

```
z_A(q_s,D_s,c) = 1[c=y_s]                 for every observed s and class c.
```

There is no condition on unobserved answers in this hypothesis. Every
feasible channel has nonnegative logits summing to one. Each example's CE
is therefore bounded below by the CE of its target vector, and `A` attains
all those bounds. Equality of total CE forces equality on every sample.
For at least two classes, equality forces target mass one. Nonnegativity
and unit mass then force all other coordinates to zero. Consequently every
feasible global CE optimizer agrees with `A` on the complete observed
logit vectors, not only on their argmax. This proves identification on the
training observations; it does not assert that an optimization algorithm
has reached that optimum.

### O2. An input-only certificate determines an unseen prediction

Define categorical features `x(D)_(k,v)=1[D(k)=v]`. For a proposed unseen
input `(q,D)`, supply finite real coefficients `a_s` satisfying

```
a_s = 0 whenever q_s != q,
x(D) = Σ_s a_s x(D_s).
```

This is an explicit finite system of linear equations involving only
observed inputs and the proposed test input. Neither training answers,
test answers nor trained parameters occur in the certificate. Coefficients
may be negative. Direct expansion of the actual channel forward gives

```
z_W(q,D,c) = Σ_s a_s z_W(q_s,D_s,c)        for every channel W and class c.
```

Thus agreement on observed logits implies agreement on this unseen input.
Together with O1, every feasible global optimizer has the same unseen
logits as the feasible reference channel. Equivalently, the inferred vector
is the explicit combination `Σ_s a_s 1[c=y_s]` of observed target vectors.
To call its argmax a correct ground-truth answer requires that the target
rule on the test input is represented by that reference channel. Training
data cannot exclude an arbitrary ground-truth rule outside the model.

The certificate is sufficient, not necessary. Nonnegativity and the shared
routing constraints can identify additional predictions, as in CC5. Failure
to find this certificate alone is not an impossibility theorem. A query with
no observed samples has no nonzero categorical test vector in the indicated
span, which is consistent with CC7's unseen-query ambiguity.

### O3. Error amplification is measurable

For any two channels, any class and any real error bound `ε`, suppose
the absolute difference of the observed logits is at most `ε` per sample.
The same input certificate and the triangle inequality give

```
|z_W(q,D,c) - z_A(q,D,c)| <= κ ε,       κ = Σ_s |a_s|.
```

This is a deterministic logit error bound. It is not a CE convergence rate
or a statistical sampling guarantee. Large coefficients can amplify error;
finding a small-norm reconstruction is therefore useful beyond merely
checking that a reconstruction exists. If a reference answer has a logit
margin greater than `2 κ ε`, this bound preserves its strict argmax (this
last elementary margin consequence is explanatory, not a separate Lean
theorem in this extension).

### O4. A genuinely unobserved input

Use two Boolean dictionary slots and a fixed query selecting the second.
Observe inputs `00,01,10` with labels `false,true,false`. The all-true input
`11` is absent. Coordinate by coordinate,

```
x(11) = -x(00) + x(01) + x(10).
```

The copying channel is a feasible witness for the observed labels. O1-O2
therefore force every feasible global CE optimizer to predict `true` on
`11`. This is a new input, not an added training example. The certificate's
amplification factor is three. Lean proves the absence of `11` from the
observations, the feature identity, realizability, the factor three and the
conclusion for every feasible optimizer. This example uses shared input
structure; flatness remains a separate rank/capacity bound.

The three `ConvexChannel/Observation/` modules prove O1-O4 and provide
satisfying examples for every conditional theorem. Principal declarations
are `observed_logits_eq`, `logits_reconstruct`, `unseen_logits_of_min`,
`logit_error_of_reconstruct` and `every_optimizer_predicts_unseen`.

## Convex channel minibatch updates

### CM1. Independent query steps preserve the domain

For any feasible joint tensors `A,B` and query-dependent step sizes
`0 <= eta(q) <= 1`, define

```
W(q,k,v,c) = (1-eta(q)) A(q,k,v,c) + eta(q) B(q,k,v,c).
```

Nonnegativity follows term by term. Summing over classes preserves each
key's value-independent routing marginal. Summing over keys then gives
`(1-eta(q))*1 + eta(q)*1 = 1`. Hence `W` is feasible even when different
query blocks use different step sizes. Its logits undergo the same convex
combination for that query. When `eta(q)=0`, every coefficient of that query
is retained. This matters for a minibatch containing only some queries:
the absent ones must not be reset to an arbitrary zero-gradient oracle.

`ConvexChannel/Batch.lean` proves these facts as `feasible_mixRows`,
`logits_mixRows` and `mixRows_zero`, with Boolean witnesses. Combined with
CC8's feasible oracle vertices, they justify blockwise minibatch feasibility.
CC2 already supplies convexity of ordinary CE on each finite batch. These
statements do not assert stochastic optimization convergence or transfer of
unseen labels. CF6 separately proves persistent forward-only batch equality.

## Inference-time learning

### CI1. Retained state can learn from later observations

After main training, retain `trainedState(history)`, the complete candidate
records of CF6. When a correct answer becomes available, whether through
feedback or an observable target in the incoming stream, pass that labeled
example to the same `advance` operation. For arbitrary history and feedback,

```
advance(trainedState(history), feedback) = trainedState(history ++ feedback).
finish(keys, advance(trainedState(history), feedback), default)
  = learn(keys, history ++ feedback, default).
```

These are exact computed-state and model equalities, including failure.
If one enumerated key and decoder realize both phases, the next snapshot
succeeds and fits every old and new observation. The update receives the
labeled data, not that target key or decoder. This is continued categorical
learning without backpropagation; it is not a theorem that arbitrary unlabeled
inputs reveal their correct answers. A task-specific observable-target adapter
must justify the labels it supplies. The theorem guarantees fitting the combined
observations; it asserts no general monotonicity of held-out accuracy. The finite
example in CI4 demonstrates correction of a previously wrong prediction.

### CI2. Self-labeling has a fixed prediction function

Given the current successful snapshot `(k,f)`, label an arbitrary list of
inputs by `(D,f(D(k)))` and really apply `advance` to all candidates. Every
earlier candidate remains eliminated. The selected key survives: at a recorded
value its prediction is the recorded label; at an unobserved value it inserts
the same default that its decoder already returned. Its entire decoded function
therefore stays equal to `f`. Thus the next snapshot is exactly `(k,f)`.

`inferBatch` performs this unverified update; if no candidate survives, it
retains the failed state. `inferBatches` refreshes the snapshot between batches.
For every initial state and every finite list of arbitrary input batches,

```
finish(keys, inferBatches(keys, batches, state, default), default)
  = finish(keys, state, default).
```

The result includes exact key and function equality, so all predictions and
the corresponding deterministic channel remain unchanged. Refitting the original
learner after appending its own hard predictions gives the same result too.
This is a claim about this deterministic consistency learner with fixed keys,
default and hypothesis family, not about all uses of self-supervision.

The candidate **state** need not stay unchanged: unobserved cells can acquire
default guesses, and alternatives can be eliminated. Those mutations can harm
future learning even though the current predictions have not improved.

### CI3. Fix unverified updates with a conservative agreement certificate

`certify(keys,state,D)` returns an answer only when at least one candidate
survives, every surviving enumerated candidate has already observed its value
`D(k)`, and all those observed entries have the same class. It returns `none`
on an unknown value or disagreement. It uses neither a completed decoder's
default nor a confidence threshold. This sufficient certificate is deliberately
conservative; no completeness claim is made.

If the certificate returns `c`, every enumerated surviving key and decoder
consistent with its observations predicts `c` on `D`. In particular, the answer
is correct if the true key is enumerated and its decoder is still consistent.
Those hypotheses matter: corrupted prior labels or a target outside the
single-key family are not repaired by agreement among the remaining candidates.

`guardedObserve` updates only on a certificate and otherwise abstains. For
every enumerated candidate its post-update record is exactly its old record.
If all keys are enumerated, the entire state is unchanged. Accepted entries
were already known, so this guard prevents contamination but supplies no new
training information. Keep new unverified guesses out of the trusted records;
use independently observed answers to add information through CI1. Records
already contaminated by guesses must be rebuilt from trusted observations.

### CI4. Useful feedback, poisoned defaults and an ambiguity witness

Take Boolean values/classes, enumerate key `false`, and train on dictionary
`id` with label `false`. The value at that key is `false`; the true decoder
may be `id`, but the default completion predicts `false` everywhere.

* Revealing label `true` on the all-true dictionary changes the learned decoder
  to `id`, correcting the previously wrong prediction at the new value.
* Self-labeling that dictionary with the current prediction `false` first
  leaves the prediction function unchanged, but records the guess. The same
  genuine feedback then eliminates the only enumerated key and learning fails.
* The certificate abstains at this unobserved value. Guarding that inference
  observation preserves the record, and the genuine feedback still recovers `id`.

Separate examples show acceptance of an observed answer and abstention when
two surviving candidates have observed but conflicting answers on a new input.
Finally, for **any** proposed Boolean answer at value `true`, there is a Boolean
decoder fitting the same observed `false -> false` label that disagrees with
that answer. Both possible rules belong to the single-key family. Thus no
procedure using only the common observations can guarantee that unseen label
without additional evidence or restrictions on the rule. This ambiguity
statement does not rule out a valid task-specific self-supervised signal.

The `ConvexChannel/Forward/Inference/` modules prove CI1-CI4. They formalize the
existing categorical update and the new conservative guard; no neural teacher,
noisy-label optimizer or Rust deployment of the guard is asserted here.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexChannelAxioms.lean
```

## Convex spectral upper bounds

### SB1. Direct complex coordinates with ordinary cross-entropy

Use the matrix entries `X ∈ ℂ^(m×n)` as the learned coordinates, with
`|X_ij| ≤ 1` and `||X||op ≤ R`. Both constraints are convex over the reals.
Fixed probes give logits `Re tr(A_scᴴ X)`, so ordinary empirical
cross-entropy is jointly convex as well. The spectral condition is an
upper bound, not equality of singular values, orthogonality, or a product
of trained factors. For code probes the logits are
`Re(conj(phi(c)) X[cell(s)])`. A flat unit-modulus target of squared
singular value `c` meets the spectral bound `R = sqrt(c)`.

`SpectralBall/Basic.lean` proves domain convexity, zero feasibility,
convexity of the usual probe CE, and the squared vector-energy bound from
the actual L2 operator norm. The size of a useful radius is additional
information; flatness does not select the missing labels by itself.

### SB2. Repair of the XE15 unseen-sign counterexample

For XE15's `K = [[1,1],[-1,1]]`, use `R = sqrt(2)` and observe only
`(0,0), (0,1), (1,0)`, with their binary labels. The target is feasible.
An entry in the unit disk has signed real part at most one, so the target
minimizes each observed ordinary binary CE term. Strict convexity in
the observed real parts, together with the disks, forces every global
CE minimum to equal the three observed complex entries of K.

The spectral bound says `2I - XᴴX` is positive semidefinite. Its `(0,0)`
entry vanishes because the first column is `(1,-1)`. A zero diagonal
entry in a positive-semidefinite matrix forces its column to vanish;
the off-diagonal equation is `1 - X_11 = 0`. Thus every constrained CE
minimum is K, including the unobserved `+1`. `SpectralBall/Recovery.lean`
proves this from the norm constraint and an empirical-loss inequality,
with K as a satisfying example. No assumption of correct unseen
predictions is used. This changes the model constraint; XE15's negative
claim about nuclear-norm training remains valid.

This witness is not a theorem for random half-observed modular tables,
looser radii, arbitrary labels, long contexts, or a specific finite-step
optimizer. Those require separate evidence. Numerical projections and
finite-difference updates are floating-point algorithms, not Lean-certified
solutions of the constrained optimization problem.

### SB3. A tight bound can still leave multiple optimal unseen answers

Let

```
H = [ 1  1  1  1 ]      H' = [ -1  1 -1  1 ]
    [ 1 -1  1 -1 ]           [ -1 -1 -1 -1 ]
    [ 1  1 -1 -1 ]           [  1  1 -1 -1 ]
    [ 1 -1 -1  1 ]           [  1 -1 -1  1 ].
```

Both satisfy `HᴴH = 4I`; H' exchanges and negates the first two rows.
Observe every cell except rows `{0,1}` crossed with columns `{0,2}`.
The twelve observations are distinct, cover every row and column, and
have identical labels under H and H'. Each table belongs to the same
tight domain `|X_ij| ≤ 1`, `||X||op ≤ 2`. Each also minimizes every
observed ordinary binary CE term over the unit disks. Thus both are
global CE minima over that convex domain, yet their unseen `(0,0)`
labels are opposite. `SpectralBall/Ambiguity.lean` proves these claims,
including the observation coverage and both Gram identities.

This refutes a universal extension of SB2 even at 75% observation,
with genuinely flat targets and all rows and columns represented.
Additional assumptions must eliminate observationally indistinguishable
targets; changing batch size or solving the same empirical objective more
accurately cannot supply such information. This example is not a theorem
that the particular modular stress instances are unrecoverable.
