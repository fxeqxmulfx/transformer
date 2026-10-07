# Verified tensor stack against original GPTMini softmax

Does the compact causal stack proved in Lean learn all six Basis recipes
under ordinary AdamW within the original GPTMini's measured training FLOPs?
This is stages 3 and 4 of [the active cycle](../../plan.md). All eighteen
softmax references have passed; matched convex comparisons and repairs continue.

| Arm | Embedding/attention | Training objective | Budget |
| --- | --- | --- | --- |
| `softmax-easy-*-seed{0,1,2}` | Original width-64, two-layer, four-head GPTMini; QKNorm/RoPE/XSA/softmax | Existing answer cross entropy | Unchanged substantial Basis recipe, stopped at 99% sequence accuracy |
| `softmax-hard-*-seed{0,1,2}` | Original width-128, six-layer GPTMini | Existing answer cross entropy | Unchanged hard Basis recipe; E4 selected at length 128 |
| `tensor-*-*-seed*` | Same NTC module interfaces; 52 free fields per token, two shared structured heads, learned absolute/relative positions | Actual complete branch/path/route/channel NLL, labels from raw training data | The corresponding measured first-success reference FLOPs; whole updates, with early success stopping disabled |
| `tensor-gain8-*-*-seed*` | The same true stack with all free potentials in fixed gain-eight coordinates | The actual gained complete NLL; same data targets and jointly convex free domain | The same pinned per-seed reference ceiling; original actual-potential initialization scale and unchanged AdamW/rate/schedule |
| `tensor-gain32-hard-recall-seed*` | The same stack in fixed gain-32 coordinates; fresh initialization | The same genuine complete NLL and unrestricted jointly convex domain | The same three hard-recall reference ceilings; separately charged repair attempts, with previous arms preserved |

All references use the existing 20,000/512/512 data splits from data seed
1, model seeds 0/1/2, original batch/rate/warmup/AdamW settings, float32 and
measured compiled CPU execution. Thread counts reproduce the existing Basis
small/large choices. The GPU is a GTX 1050 with 2 GB; no hardware change is
used to claim an arithmetic advantage. An unsuccessful reference must be
preserved and extended before it can define a success budget. Evaluation,
generation, setup and compilation costs must be recorded separately.

The candidate is `TensorStack` in the DSL, ported from
`Transformer.GPTMini.Convex.Structured` at `b0a43a8`. Its literal model
executes both prenorm residuals and the fixed-zero original ReLU2 FFN in
every layer. Only embedding, attention and positional processing change;
the residual width, final RMSNorm and tied dot-product readout remain.
All layers share head/token fields; this does not claim independent deep
trainable matrices. The public `integer_function(list[int]) -> list[int]`
preserves its input and appends one checked greedy token, including PAD
fallback for invalid inputs. Actual attention never sees labels, task
recipes, record counts or parsed routes.

There are `52*V + 3*C + 128` free scalars: 2,384 for depth, 28,816 for
recall and 3,721 for parity. Ten output-code axes and a unit anchor are
fixed geometry, and both FFN matrices are fixed zero. The loss is globally
convex jointly in the entire free parameter space over real arithmetic;
ordinary output CE and AdamW convergence are not covered. Relative/absolute
positions, Q, K and value potentials all learn together. The actual
inference contracts six-state histories and every causal key/value pair;
its log-space implementation uses quadratic positional arrays and small
channels, rather than enumerating paths or the implicit `4**9` bank.

Numerical checks compare the prefix contraction and gradients with literal
pair sums, and state marginals with explicit short-path sums. They check
actual embedding/RMS/residual/readout equivalence and causality at widths
64/128 and depths two/six, joint-loss convexity along a parameter segment,
the probability/NLL identity, raw order/overwrite/EOS labels, parameter
counts and the checked integer interface. Given finite Lean witness weights
decode depth/parity/latest-overwrite controls and remain finite at recall
gain `65*log(10**12)`; these controls are not trained checkpoints.

The implemented convention is `aten-reference-arithmetic-v1`. It traces
actual eager operator shapes on independent copies and then charges each
compiled learning update at that shape. Multiply/add is two operations;
scalar arithmetic, comparisons, selections and transcendentals are one.
Stable log-sum-exp and log-prefix sums have explicit formulas, and standard
AdamW includes both moments, decay and bias corrections. Reshapes, copies,
indexing and allocation have zero arithmetic cost. Every encountered
operator must have a rule; unsupported operators stop the measurement.
Detailed per-operator/per-phase profiles travel in the result JSON.

Floating and integer counts remain separate. The training ceiling charges
both, plus tensorized complete-label preparation; rebuilding those labels
after a resume is charged again. Common raw dataset generation, model
initialization, sampling, compilation and independent counting probes are
outside learning compute. Evaluation model calls, including every actual
generation call, are counted separately; validation metric aggregation is
not counted. Setup evaluation costs and wall/training time are reported.
These are reference arithmetic counts, not hardware instruction counts;
compiler fusion/reassociation, memory traffic and sorting implementations
are not a speed claim. Sorting has an explicit reference comparison
charge. The same convention applies to both architectures.

`FlopBudget` sets the exact arithmetic ceiling. A whole update runs only if
it fits; the unspent remainder is reported, with no dummy learning work to
fill it. The safety update limit is separate. The last minibatch training
objective and ordinary validation output CE/accuracy are recorded. Tests
verify manual matrix counts, unknown-operator rejection, full raw state
tables, no probe mutation, identical ordinary compiled softmax trajectories,
tail-batch accounting, exact budget stopping, sampler restoration and
checkpoint continuation including repeated preprocessing.

Full-shape counting probes from model seed 0 and data seed 1 cover every
recipe below, including the 32-row recall epoch tail. These are measured
per-update reference operations, not a first-success budget or a training
result. Both columns use the recipe's original batch, static training width
(64 for depth/recall and 19 for parity), precision and AdamW.

| Recipe | Batch | Softmax update, G operations | Tensor update, G operations | Gain-eight tensor update, G operations |
| --- | ---: | ---: | ---: | ---: |
| Easy depth, 64 × 2 | 16 | 0.744891 | 0.225785 | 0.225790 |
| Easy recall, 64 × 2 | 64 | 3.028276 | 0.961052 | 0.961110 |
| Easy parity, 64 × 2 | 16 | 0.196258 | 0.049685 | 0.049692 |
| Hard depth, 128 × 6 | 16 | 8.017926 | 3.160813 | 3.160821 |
| Hard recall, 128 × 6 | 64 | 32.120729 | 12.701124 | 12.701184 |
| Hard parity, 128 × 6 | 16 | 2.256004 | 0.851454 | 0.851464 |

All operator coverage checks passed on these shapes. The tensor training
loss computes the complete likelihood at the actual final-head input; it
executes all earlier residual blocks and their zero FFNs. The last head
computes its actual NLL rather than unused decoded logits/zero final FFN.
Evaluation executes the complete model. No skipped operation is counted
as executed work, and every actual repeated layer is charged. Auxiliary
label preparation is additional to the per-update table.

First completed references use lab revision `0700c49`. A success means
the original selection split reaches at least 99% sequence accuracy;
ordinary test and length extrapolation are reported separately. The
ceiling is the successful history row's cumulative training charge.

| Reference | First successful update | Training operations | Validation sequence accuracy | Test sequence accuracy | Length-128 sequence accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy depth, seed 0 | 200 | 148,978,162,800 | 1.000000 | 1.000000 | 0.603516 |
| Easy depth, seed 1 | 200 | 148,978,162,800 | 1.000000 | 0.998047 | 0.630859 |
| Easy depth, seed 2 | 200 | 148,978,162,800 | 0.998047 | 0.998047 | 0.904297 |
| Easy recall, seed 0 | 1,650 | 4,989,090,525,500 | 0.998047 | 1.000000 | — |
| Easy recall, seed 1 | 2,450 | 7,408,685,409,020 | 0.990234 | 0.988281 | — |
| Easy recall, seed 2 | 1,800 | 5,443,331,961,200 | 0.994141 | 0.982422 | — |
| Easy parity, seed 0 | 8,200 | 1,609,311,910,000 | 0.990234 | 0.972656 | — |
| Easy parity, seed 1 | 5,000 | 981,287,750,000 | 0.990234 | 0.974609 | — |
| Easy parity, seed 2 | 4,200 | 824,281,710,000 | 0.994141 | 0.980469 | — |
| Hard depth, seed 0 | not passed at 10,000 | 80,179,263,340,000 | 0.980469 | 1.000000 | 0.978516 |
| Hard depth, seed 0 continuation | 11,400 | 91,404,360,207,600 | 0.992188 | 1.000000 | 0.994141 |
| Hard depth, seed 1 | 1,400 | 11,225,096,867,600 | 0.992188 | 1.000000 | 0.988281 |
| Hard depth, seed 2 | 5,400 | 43,296,802,203,600 | 1.000000 | 1.000000 | 1.000000 |
| Hard recall, seed 0 | 3,350 | 107,443,940,040,500 | 0.992188 | 0.986328 | — |
| Hard recall, seed 1 | 2,650 | 84,991,530,151,660 | 0.990234 | 0.984375 | — |
| Hard recall, seed 2 | 2,400 | 76,977,398,143,040 | 0.996094 | 0.982422 | — |
| Hard parity, seed 0 | 4,200 | 9,475,216,186,800 | 0.990234 | 0.974609 | — |
| Hard parity, seed 1 | 7,200 | 16,243,227,748,800 | 0.990234 | 0.974609 | — |
| Hard parity, seed 2 | 8,800 | 19,852,833,915,200 | 0.996094 | 0.990234 | — |
The complete original attempts and pinned settings are preserved in
[reference_budgets.json](reference_budgets.json). All eighteen
original references passed. Hard depth seed 0 first reached a best length-128 validation
accuracy of 98.05% in 10,000 updates; its original result/checkpoint are
archived in the run directory and it continues to 20,000 updates at
unchanged AdamW/rate/batch/schedule. That continuation passes at 11,400
updates, supplying the eighteenth measured success ceiling. Both its
original failure and successful continuation are kept in the JSON.

Candidate arms are added only for a completed successful reference,
retain its data, batch, learning rate,
schedule, AdamW and thread count, and stop at its exact arithmetic ceiling.
The 100,000-update bound is a safety limit, not permission to exceed that
ceiling. Preserve every failure for stage 4 instead of treating capability
weights as evidence of learning.

Initial controlled candidate results:

| Candidate | Updates at ceiling | Charged training operations | Unspent ceiling | Best validation sequence accuracy | Test sequence accuracy | Finding |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tensor-easy-depth-seed0 | 659 | 148,793,847,818 | 184,314,982 | 0.185547 | 0.197266 | Only reject; complete NLL still falling |
| tensor-easy-parity-seed0 | 32,390 | 1,609,283,865,140 | 28,044,860 | 1.000000 | 1.000000 | Full parity/EOS generation passes; first validation pass at update 7,000 |
| tensor-hard-parity-seed0 | 11,128 | 9,474,985,814,904 | 230,371,896 | 1.000000 | 1.000000 | First validation pass at update 7,000 |
| tensor-easy-recall-seed0 | 5,199 | 4,988,858,443,969 | 232,081,531 | 1.000000 | 1.000000 | First validation pass at update 1,900 |

The depth checkpoint has the correct preferred transition on every
observed A/B/neutral row, but individual correct-transition probabilities
are only 0.32–0.40 and the state branch weight is 0.764. A diagnostic
uniform multiplication of its learned potentials by eight, keeping fixed
decoder/anchor geometry, changes the real order-control outputs from
`[15, 15]` to `[16, 15]` and attains 100% validation sequence accuracy.
This is a post-training diagnostic, not a fresh ordinary-AdamW result.
Stage 4 will test a mathematically verified common linear gain during
training from an independent initialization, including its arithmetic.
Gauge directions and a guarantee of AdamW convergence remain open.

The repaired block is `TensorGain`, from Lean's TensorGain at `c12df24`.
For fixed nonzero gain its function class is unchanged: inverse coordinates
recover every original model and explicit finite raw Basis solver. Its
real complete sample/batch loss remains jointly convex in all free
coordinates and is the negative log of the same actual inference joint.
The implementation uses gain eight in embeddings, absolute/relative
positions, chronology, initial/state/value/head potentials and complete
NLL. Coordinates initialize at `std/gain`, preserving the original actual
potentials at `std=.02`; no trained witness initializes an experiment.
Gain arithmetic is counted. Independent tests compare actual forward,
both losses and all gradients with an explicitly scaled original model,
check identical initial potentials and the unrestricted Jensen inequality.
Fresh matched learning proceeds at the pinned per-seed ceilings.
The complete gain-eight port passes 218 Python tests. All eight full-shape
counting probes, including both recall batch sizes, have complete operator
coverage. The added actual gain work is included in the table and every
candidate's whole-update ceiling.

Twelve gain-eight arms have completed their full ceilings at the
2026-10-07 15:11 UTC snapshot. All twelve have 100% best validation and test
sequence accuracy; completed depth arms also have 100% length-128 test
accuracy. Detailed source, actual charges, first-success observations and
unchanged split fingerprints are pinned in
[gain_comparison.json](gain_comparison.json). Running arms are excluded
from that completed comparison.

| Completed gain-eight arms | First successful update | First-success operations | Full-ceiling updates, seeds in order |
| --- | ---: | ---: | --- |
| Easy depth, seeds 0/1/2 | 600 | 135,475,524,680 | 659 / 659 / 659 |
| Easy parity, seeds 0/1/2 | 1,000 | 49,692,659,720 | 32,385 / 19,747 / 16,587 |
| Easy recall, seeds 0/2 | 300 | 288,366,336,124 | 5,198 / 5,672 |
| Hard depth, seed 1 | 800 | 2,528,658,313,080 | 3,551 |
| Hard parity, seeds 0/1/2 | 1,000 | 851,463,887,720 | 11,128 / 19,076 / 23,316 |

Hard recall is the remaining observed accuracy bottleneck. Read-only
diagnosis of every validation row at updates 3450/3500/3500 gives
76.76/76.56/76.95% sequence accuracy and 98.19/98.07/98.10% maximal-route
accuracy. All 74/79/78 misselected routes match a different key; none
selects a nonadjacent value or an outdated matching write. Effective
table-position biases span about seven logits. Removing only the value
partition from routing without retraining raises sequence accuracy to
77.73/77.93/78.52%, so this alone does not solve the bottleneck.

The separately charged gain-32 hard-recall repair retains the same
verified class, genuine complete loss, ordinary AdamW and actual initial
potentials. It tests whether stronger learned matching separation can
overcome the observed positional preference. Previous full-ceiling runs
continue and their costs remain recorded; no post-training scaling is
reported as successful fresh learning.

Lean's arbitrary-weight parity certificate is committed in `9b86323`.
Nine required transition rows, fifteen reachable value rows, initial and
learned branch confidence derive correctness for every valid raw parity
input when `deltaHead + deltaInitial + 19*deltaTransition + 5*deltaValue`
is below `1/11`. It covers the actual full tensor stack and both generated
answer/EOS calls. Numerical float64 row checks of all six completed
gain-eight parity checkpoints give budgets below `2.17e-7`; every needed
logit gap exceeds 20. These numerical estimates support the sufficient
condition; exact learned-weight and IEEE certification are separate checks.
