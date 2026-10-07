# Verified tensor stack against original GPTMini softmax

Does the compact causal stack proved in Lean learn all six Basis recipes
under ordinary AdamW within the original GPTMini's measured training FLOPs?
This is stage 3 of [the active cycle](../../plan.md). No new training
result or successful reference FLOP budget has been recorded yet.

| Arm | Embedding/attention | Training objective | Budget |
| --- | --- | --- | --- |
| `softmax-easy-*-seed{0,1,2}` | Original width-64, two-layer, four-head GPTMini; QKNorm/RoPE/XSA/softmax | Existing answer cross entropy | Unchanged substantial Basis recipe, stopped at 99% sequence accuracy |
| `softmax-hard-*-seed{0,1,2}` | Original width-128, six-layer GPTMini | Existing answer cross entropy | Unchanged hard Basis recipe; E4 selected at length 128 |
| Tensor candidate (queued) | Same NTC module interfaces; 52 free fields per token, two shared structured heads, learned absolute/relative positions | Actual complete branch/path/route/channel NLL, labels from raw training data | The corresponding measured first-success reference FLOPs; not assigned before measurement |

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

| Recipe | Batch | Softmax update, G operations | Tensor update, G operations |
| --- | ---: | ---: | ---: |
| Easy depth, 64 × 2 | 16 | 0.744891 | 0.225785 |
| Easy recall, 64 × 2 | 64 | 3.028276 | 0.961052 |
| Easy parity, 64 × 2 | 16 | 0.196258 | 0.049685 |
| Hard depth, 128 × 6 | 16 | 8.017926 | 3.160813 |
| Hard recall, 128 × 6 | 64 | 32.120729 | 12.701124 |
| Hard parity, 128 × 6 | 16 | 2.256004 | 0.851454 |

All operator coverage checks passed on these shapes. The tensor training
loss computes the complete likelihood at the actual final-head input; it
executes all earlier residual blocks and their zero FFNs. The last head
computes its actual NLL rather than unused decoded logits/zero final FFN.
Evaluation executes the complete model. No skipped operation is counted
as executed work, and every actual repeated layer is charged. Auxiliary
label preparation is additional to the per-update table.

Next: measure reference first-success training FLOPs, then define and run
candidate arms at those exact ceilings. Preserve every failure for stage 4
instead of treating capability weights as evidence of learning.
