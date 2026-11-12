# Convex causal probabilities for the full stress catalog

Lean-only extension, board task #198, 2026-11-12. The requested scope is a
common model supporting the outputs and losses of all 165 catalog cases,
with explicit conditions for learning unseen answers. It is not a claim
that all 165 historical targets can be attained. No experiment or native
model implementation is part of this change.

Source catalog: `/home/misha/python_projects/diff-gpt/stress_tests_for_gpt.md`,
SHA-256 `d358edc9f3c9104f4a19f4ccbbd35f8547a745cf47cf611b23bbf7f032ca143f`.
Its 165 cases include the random-table and random-language controls, noisy
labels, measured weather relationships, numeric decoders and language CE.
The 35-case short/long selection is a proper subset and is not the scope here.

This extension addresses two model restrictions. Probability coordinates used
directly as logits have a fixed confidence ceiling. Additive categorical
features and fixed pair-phase classes also exclude some sequence interactions.
The repair changes the readout and supplies full positional interaction
features, while retaining structured causal libraries as smaller alternatives.
The universal alternative is expensive and does not itself generalize.

## CCa1. Calibrated probabilities and actual ordinary CE

Let `X` be visible inputs, typically complete causal prefixes, and let `C`
be the nonempty original finite output vocabulary. Joint coordinates satisfy

```
Q(x,c) >= 0,                 sum_c Q(x,c) = 1.
p_Q(x,c) = (1-epsilon) Q(x,c) + epsilon / |C|,
z_Q(x,c) = log p_Q(x,c),      0 < epsilon <= 1.
```

`epsilon` is fixed before optimization. The forward map receives no target.
All probabilities are positive, their sum is one, and softmax of `z_Q` is
exactly `p_Q`. Thus the repository's ordinary full-vocabulary log-sum-exp CE
is exactly `-log p_Q(x,y)`. The logarithm readout is nonlinear, but its
negative log likelihood is convex in every joint coordinate of `Q`, since
`p_Q` is affine. No factorized learned embedding or router/decoder product
is introduced. Original sample multiplicities remain in the empirical sum.

`Basic.lean`: `normalized_logits`, `crossEntropy_eq`, `loss_convex`.
At `epsilon = 1` the output is uniform; identification uses `epsilon < 1`.

## CCa2. Complete causal rules in one convex mixture

For a fixed finite bank `E_h : X -> C -> R` of probability-valued rules, train
one simplex vector `w` and compute `Q_w(x,c) = sum_h w_h E_h(x,c)`.
Each rule executes its whole computation on the actual visible prefix before
predictions are mixed. It can inspect order, count, compose operations or
maintain state through that prefix. The rules are fixed during optimization;
their internal parameters are not jointly trained under this theorem.

`Mixture.lean` proves that the resulting rows normalize, that the forward
map is linear in all weights, and that calibrated ordinary CE is convex
on the full simplex. A vertex exactly retains each original rule. This is
an ensemble in complete prediction coordinates, not an inexpensive new
parameterization of arbitrary raw transformer weights.

## CCa3. A universal bounded-context interaction head

For an input alphabet `A` and declared maximum prefix length `B`, represent
a prefix by `x : Fin B -> Option A`, padding with `none`. The encoding is
injective on lists of length at most `B`; position and length are retained.
For every positional template `t`, compute the input-only feature

```
phi_t(x) = product_i 1[x_i = t_i].
Q_W(x,c) = sum_t W(t,c) phi_t(x).
```

Only `W` is learned. Products involve fixed categorical input indicators,
not separately trained factors. Lean proves that `phi_t(x)` is one precisely
when the whole template matches, and zero otherwise. Executing the sum then
gives `Q_W(x,c) = W(x,c)`. Therefore every conditional coordinate table on
bounded prefixes is represented exactly, with convex row constraints and
the calibrated convex loss of CCa1. This includes arbitrary deterministic
token rules and stochastic conditional distributions in the coordinates.
After the readout, deterministic rules have a strict correct argmax and
positive smoothing mass; exact calibrated distributions must exceed the
declared floor, as CCa6 states.

This is a full prefix lookup expressed by categorical interaction features.
There are exactly `(|A|+1)^B` allocated features, including padded arrays
that do not encode valid prefixes, and that many coefficients per class.
It proves universal finite-context capacity, not compactness, statistical
learning or efficient execution. The horizon must include all generated
prefixes and declared length probes. It says nothing beyond that bound.

`Features.lean`: `encode_injective`, `feature_eq`, `evaluate_eq`,
`universal_rows`, `context_card`. The witness includes XOR of two visible
positions, which independent additive slot features cannot strictly classify.

## CCa4. Exact finite CE minima, with no optimization assumption

For probability coordinates the largest possible target probability is
`1-epsilon + epsilon/|C|`. Each observed CE is at least

```
ideal = -log(1-epsilon + epsilon/|C|).
```

A feasible table with `Q(x_s,y_s)=1` at every observed sample attains the
sum of these lower bounds and is a global minimum. Conversely, for
`0 < epsilon < 1`, loss at this bound forces each observed target mass to
one. Positivity and strict logarithm monotonicity justify the converse;
it is not an assumption that a solver fitted the observations.

`Optimality.lean` proves both directions for the actual CE. Its constructed
minima can still disagree on unseen inputs without the next conditions.

## CCa5. A training-only finite learner and recovery condition

For deterministic candidates `f_h : X -> C`, the learner receives only the
fixed bank, training inputs and ordinary labels. It explicitly computes

```
survivors = {h | every sample has f_h(x_s) = y_s}.
```

It returns `inconsistent` for no survivor, `ambiguous` for more than one,
or the unique learned rule. The implementation uses a finite filter and
cardinality tests; an order on the bank supplies the singleton's element.
It does not use an existential completion oracle. Uniform weights on a
nonempty surviving set give real joint parameters for CCa2. Lean proves
that their predictions fit every sample and attain CCa4's global ordinary
CE minimum, without assuming a successful optimizer run. Real probability
arithmetic and the mathematical maximum readout are noncomputable Lean
definitions; the finite candidate filter and verdict are computational.

The identifying condition is input-only relative to the fixed bank:

```
for each pair of distinct candidates h,k,
some observed input x_s has f_h(x_s) != f_k(x_s).
```

This condition is a finite comparison test on candidate outputs at observed
inputs, independent of the unknown true rule and unseen labels. Assume the
complete true causal rule belongs to the bank and the observed labels follow
it. Then every other candidate is eliminated, the learner returns the true
rule, and the constructed distribution is its one-hot table on every input.

`Recovery.lean` also proves the result for **every** optimizer over the entire
convex simplex of mixtures. At a minimum each observed target mass is one.
Every candidate contradicted at that observation has zero weight. Separation
eliminates every incorrect weight; unit total mass forces the target vertex.
This convex domain is not replaced by the set of deterministic vertices.

`Learner.lean` proves exact intersection of survivors across batches and
invariance under a surjective reindexing, including repetitions. Retaining
survivors costs at most the bank size; testing them requires a bank evaluation
for each examined candidate/sample pair. No compiled runtime bound, compact
universal bank or automatic bank discovery is proved. A bank that was chosen
using labels needs its own justification of realizability and generalization.

The concrete parity witness has two possible orientations of a two-bit rule.
One observed input distinguishes them. The computed model recovers genuinely
unseen inputs with both output classes. Additional witnesses cause explicit
ambiguity and inconsistency. This is conditional structured learning; it does
not demonstrate learning parity from the unrestricted universal lookup family.

## CCa6. Noisy and probabilistic outputs

For any conditional distribution `P` with `P(x,c) >= epsilon/|C|` and
`epsilon < 1`, the joint coordinates

```
Q(x,c) = (P(x,c) - epsilon/|C|) / (1-epsilon)
```

are feasible and the readout returns `P` exactly. Thus the probability
parameterization supports calibrated, non-one-hot outputs. For positive
normalized distributions `p,q`, Lean proves the logarithm inequality

```
sum_c p_c (-log p_c) <= sum_c p_c (-log q_c).
```

The same statement is proved for the existing ordinary softmax CE of the
emitted logits. This supplies the correct population objective for noisy
series and language modeling; it does not prove finite-sample estimation
of the unknown conditional distribution or make the deterministic candidate
filter robust to corrupted labels.

The deterministic `ideal` tends to zero as `epsilon` tends to zero. There
is no fixed unit-logit confidence ceiling, but the floor is not learned
jointly. For an independent uniform next token, any normalized randomized
prediction has expected accuracy exactly `1/|C|`. This is the relevant limit
for the random-LM control, not a failure of convex optimization.

`Calibration.lean` proves these statements, with nonuniform stochastic
witnesses. It does not set a forecasting skill, CE-gain or language threshold.

## CCa7. Free generation and EOS

`generate` evaluates the next-token rule on the current visible prefix,
appends its own answer, and repeats until EOS or a fixed fuel bound. It
includes the EOS token in the emitted list. The maximum-probability readout
of a smoothed recovered rule is strictly correct, so the recovered model
and the true bank member emit identical complete sequences, including
termination. `learns_causal` combines actual finite learning, feasible
parameters, exact empirical CE minimality and this generation equality.

The nonconstant witness learns from the empty prefix, then emits a new
token and EOS while reading its own newly generated prefix. These results
do not replace free generation by teacher forcing. A bank of complete
causal algorithms may cover arbitrary lengths; the bounded-prefix fallback
of CCa3 covers only its declared finite horizon.

## CCa8. The condition cannot be assumed for a universal bank

For any unobserved input there are two binary functions that agree on every
observed input and disagree there. Consequently the bank of **all** functions
cannot have separating training signatures on a proper observation set.
`universal_not_separates` proves this explicitly. Universal capacity and
conditional structured learning are separate results, and neither flatness
nor exact empirical optimality supplies the missing identifying evidence.

No claim is made that the fixed historical random splits separate an
appropriate compact bank for all deterministic cases. The original random
table, random labels and random language controls must retain their negative
interpretation. Native optimizer convergence and approximate-CE stability
remain separate from these exact-real results.

## CCa9. Relation to the existing convex spectral model

`Compatibility.lean` embeds feasible channel outputs and coupled-moment
probability tables in the new probability interface. The calibrated loss
is convex on any convex restriction of these rows, with an explicit theorem
for the original operator/nuclear-budget domain. Thus those structured
domains remain available. Their old numerical CE values are not preserved:
the output readout has changed.

Universal support uses the enlarged prefix head rather than imposing one
tight fixed spectral budget on every task. Such a budget can exclude valid
tables, as CM9 and CM15 already show. A constrained spectral submodel still
needs the target to satisfy its budgets. Neither tight spectra nor automatic
selection of a compact identifying representation follows from this extension.

## CCa10. All catalog families and their original metrics

Each catalog task is presented to the GPT as finite tokens with a finite
evaluation horizon. A training observation is the supplied visible prefix
and its next supervised token. Previously generated or teacher-forced tokens
are included exactly when the original adapter makes them visible. The same
interface applies to all eleven catalog groups:

| Group | Cases | Model support and metric qualification |
| --- | ---: | --- |
| Arithmetic functions | 40 | Joint prefix interactions include every pair table, including cubic mixtures, inequalities, arbitrary label permutations and random tables; no unseen-label promise for the random control. |
| Modular/powered division | 14 | Entire tokenized pair and original vocabulary, including the three constant controls and excluded zero divisors as specified by the data. |
| Synthetic algorithms/control | 38 | Ordered prefixes and free generation with EOS support copying, sorting, multi-hop lookup, typed brackets, counting, scratchpads and carries; probabilistic output for the random-LM control. |
| Basis/parity/languages | 14 | Prefixes retain overwrite order and all bracket/block/count information; length probes must fit the declared horizon; noisy parity is not an exact-realizability claim. |
| Modular chains | 12 | Full ordered joint interactions support any finite composition length; intermediate answers are not supplied by the feature map. |
| Associative arithmetic | 8 | Complete visible memory and computed-row prefix; preserve the source's teacher-forced later rows and negative random-label control. |
| Differenced division | 2 | The actual encoded prefix and encoded target, with no inverse arithmetic feature or changed label. |
| Temporal recurrences | 7 | Encoded causal context and fixed source readout; no claim that tokens or trajectories from unseen orbits are identified. |
| Weather/series/noisy anchors | 22 | Calibrated conditional token distributions; keep measured targets, training-only codecs, chronological splits and the original skill/relative-CE/accuracy criteria. |
| Full-GPT probes | 6 | Whole tokenized vectors or binding/order prompts; generated numeric tokens are decoded with the original codec and invalid-output penalties. |
| Fashion-MNIST/Shakespeare | 2 | Patch-token classification or character next-token distributions; preserve 90% classification and CE 1.63 as distinct empirical targets. |
| Total | 165 | Common capacity/output interface, not 165 proved benchmark passes. |

`Readout.lean` proves that exact recovered generated sequences retain any
fixed original task scorer, including numeric decoding. For a fixed scalar
decoder it proves the exact conditional squared-error decomposition and
optimality of the conditional mean. Recovering the correct token leaves its
codec error against an unquantized teacher unchanged. The vector case applies
this scalar identity per coordinate; no vector-specific threshold is asserted.

The original codecs, data generators, Fashion labels, Shakespeare corpus,
weather measurements and each of the 165 task semantics are not reimplemented
or separately verified here. Their common finite-token model interface is
covered by the universal theorem. No empirical threshold is inferred from it.
Compact representation discovery, useful sample bounds, noise-robust learning
and practical training across the full catalog remain research work.

## Validation

All source modules are reachable from `Transformer.lean`. They contain no
unfinished proofs. Conditional statements have concrete inhabited examples;
the namespace audit checks all its public declarations for extra axioms.
Use native Lean directly; no Python or experiment run is required:

```sh
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ConvexCausalAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml
cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml -- --check
git diff --check
```

The existing native index generator only maintains the required `INDEX.md`;
all new implementation and proofs are Lean. Generated validation evidence is
kept under the ignored root `.kanban/artifacts/198/` and is not committed.
