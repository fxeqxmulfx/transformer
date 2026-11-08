# Frozen complete local-learning pilot

Task #190, 2026-11-08. Frozen before candidate training or timing. This
implements the directional pairwise loss from task #192 as a complete
training rule, with controls for its different ingredients. It is a modified
Forward–Forward method, not a replication of Hinton's original algorithm.

## Cohort and budget

Use the four original arms (real SwiGLU, complex diagonal, radial complex
product and plain complex product), seeds 7/11, source initialization and
train/validation split. All methods start from scratch with identical weights
and zero moments. Match the original sampler prefix and FP32 model kernels.
Use the exact source AdamW, warmup, parameter tying and hyperparameters.
The backbone and its 143360 independent parameters are unchanged.

Run four methods per arm/seed, **32 runs**, for **4000 updates** each, with
complete train/validation evaluation at step 0 and every 100 updates.
The horizon was chosen before candidate outcomes: original first-99% steps
span 800–3000 on these eight cases. Continue after reaching a threshold.
Cap each method at 120 seconds of active setup/training/evaluation time and
the whole training cohort at 900 seconds. Exclude compilation and correctness
preflight from these budgets. Preserve caps, errors and unfinished cases.
No learning-rate search, prototype search, restart selection or extensions.

## Exact objectives and boundaries

The source graph is
`raw=E[a]+rotate90(E[b]); x0=RMS(raw); n1=RMS(x0);`
`u=product(A*n1); b=RMS(B*u); h=RMS(x0+b); scores=E*h`.
The gate/product function is the original one for each arm.

Write `P(scores,y)=sum_{c!=y} softplus(1+scores_c-scores_y)/96`.
Batch objectives average their rows. Define three objectives, each with
coefficient **1**, using parameters and cached features from the same state:

* **Output objective L2:** `P(E*h,y)` on the actual training minibatch.
* **Hidden objective L1:** `P(R1*RMS_hidden(u),y)` on that minibatch.
  The extra RMS is only on the auxiliary head; it never changes the backbone.
* **Vocabulary objective L0:** the mean over all 97 vocabulary entries v of
  `P(R0*RMS(E[v]),v)`. This trains the input embedding to retain token identity
  without requiring the linear embedding stage to predict the nonlinear
  task label. Token identities are known vocabulary metadata, not held-out
  answer labels. This costs 97 auxiliary rows every update, including all
  96 wrong-token comparisons per row.

R0 has 97 rows of width 128; R1 has 97 rows of the arm's hidden width.
Generate independent standard-normal entries using the source SplitMix64/
Box-Muller generator seeded with `seed XOR 0x666670726f746f30` for R0 and
`seed XOR 0x666670726f746f31` for R1. Normalize each row to unit Euclidean
norm in FP64, then cast to FP32. These arrays stay fixed, have no optimizer
state, and are generated without checkpoints, gradients or task structure.
Record hashes and charge creation, transposes and GPU storage. No extra
trainable head is introduced. Auxiliary forward activations remain FP32;
auxiliary RMS reductions/derivatives use FP64, with source epsilon, and cast
back to FP32. Normalization supports all 341/682 hidden coordinates.

| Method | Update rule |
| --- | --- |
| ce-bp | Exact source cross-entropy forward, full backward and AdamW. |
| pair-bp | L2's pairwise logit errors, full source backward and AdamW. |
| auxiliary-bp | Full gradient of L0+L1+L2, including every path into shared E and into A. |
| local-pair | A receives only the L1 derivative with n1 fixed. B receives only the L2 derivative with u and x0 fixed. E receives L0's vocabulary derivative plus L2's partial readout derivative with h fixed. |

All contributions are accumulated before **one** AdamW update. For complex
weights the source optimizer combines the signed real-expanded gradients
into independent coordinates exactly as before. In local-pair, no error
from L2 crosses into A or the input embedding, and no error from L1 crosses
into the embedding. Local nonlinear derivatives and outer-product GEMMs are
still performed and charged. This is not derivative-free training.

The four methods separate changing the terminal loss, adding auxiliary
objectives, and stopping cross-block credit. The full-gradient auxiliary
control is necessary: comparing local-pair only with pair-bp would change
both the objective set and the gradient paths.

## Checks before interpreting the pilot

Keep the shared source/PTX pin from `../single_update/build.rs`; the reused
pair-loss kernel must match task #192's PTX hash. Recheck the original
initial state, split and sampler hashes. The ce-bp trajectory must replay
the original saved metrics at every 100-step observation within 1e-12 CE
and exactly in correctness counts. Original input files remain read-only.

On all four arms and both seeds, at original steps 0 and 5000, use the first
eight training examples for independent FP64 central differences with step
1e-4 and two deterministic unit directions per tested parameter group.
Check local A, B, readout E and vocabulary E derivatives separately, their
shared-E sum, the total pair-bp derivative, and the total auxiliary-bp
derivative. Perturbations respect complex coordinate ties. Use fixed error
tolerance `2e-5 + 2e-3*max(abs(finite),abs(analytic))`; retain every error.
Check nonzero finite gradient energy in every parameter group, valid tied
parameters/moments after one complete update, and exact ce-bp/source update
identity. Prototype construction and loss/normalization tests are native.
Correctness probes use saved states only for auditing, never to initialize
or select a training candidate.

## Timing, memory and quality

Keep each arm/seed's four engines resident. Alternate methods in rotating
order in 100-update chunks. Synchronize before and after each measured
chunk. Charge host launches, all local forward/derivative work, gradient
accumulation and optimizer updates. Save every raw chunk time. Measure
setup, train and evaluation time separately; their sum is cumulative active
wall time for that method, excluding waiting for the other methods. Charge
shared data/sampler construction once to each method's standalone budget.
Exclude correctness audits, compilation/JIT warmup, report serialization and
checkpoint I/O, and identify these exclusions in the findings.

Report first observed 99% and 99.5% validation accuracy, and time to the
**paired ce-bp endpoint accuracy AND no-worse endpoint CE**. The latter
criterion is defined here before endpoint values are known. Include all
trajectories, endpoint and best accuracy/CE, failed thresholds, per-step time
and allocated GPU payload bytes. A faster update with worse quality is not
an efficiency win. Poll GPU process ownership before and after each matched
group and every 400 steps; foreign/unknown ownership invalidates clean-device
timing claims but does not discard the measured results.

All four methods evaluate the same source classifier with one encoder and
one 97-class readout. Auxiliary heads are never used for prediction. Scheduled
evaluation includes source CE/accuracy calculations; it is not a measurement
of serving latency. The retained source engine allocates training buffers
even when a local rule does not need all of them. Report actual buffer
payloads and auxiliary deltas, not hypothetical streamed-memory savings or
total driver memory. Save endpoint/capped states, raw histories, source and
PTX identities, and failures under the ignored persistent runs directory.
