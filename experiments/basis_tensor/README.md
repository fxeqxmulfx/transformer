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
compiled CPU execution. Thread counts reproduce the existing Basis
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

Next: add an arithmetic counter with explicit coverage, measure reference
first-success training FLOPs, then define and run exactly matched candidate
arms. Preserve every failure for stage 4 instead of treating capability
weights as evidence of learning.
