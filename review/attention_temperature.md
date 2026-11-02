# Isolated attention temperature: same-assistant review

Task162 follows the historical GPTMini control with a single component change
in the current GPT architecture. `TemperatureResidual` retains the canonical
model and multiplies each normalized, rotated query head by `exp(log_gain)`.
The 24 new parameters start at zero without consuming random draws. Every
common initial weight is identical to the task160 reference. This intervention
does not replace Q/K RMS, its epsilon, the accurate backward, SwiGLU, fixed
residual coefficients, Peri normalization, centered XSA or its prefix guard.
It is a separate experimental block, with no production architecture flag.

The focused checks establish exact FP64 initial outputs and common-parameter
gradients, finite nonzero derivatives for the new gains, agreement with explicit
causal scores, finite-difference derivatives, and finite repeated-prefix
behavior. A fullgraph compiled CPU continuation matches the uninterrupted
model, complete optimizer, sampler and gradient history exactly. Recipes differ
from the fixed canonical controls only in the model block. Reports reject
learning, interruption and continuation-boundary states before final test.

The full CPU suite passes 510 lab tests in 829.930 seconds with 13 CUDA skips,
and 104 optimizer tests in 12.752 seconds with 14 skips. The existing unrelated
CUDA Muon allocation issue remains task157. The frozen protocol pins all lab,
optimizer and cohort sources, exact data fingerprints, common initialization,
runtime and historical reference files before the first learning update.

The two hard-recall seeds use only Schedule-Free AdamW and CUDA fullgraph
max-autotune. The initial boundary is 100,000 updates, with explicit continuation
for promising cases and stopping at the first observed 99% exact validation
sequences. These are reused public Basis rows, not a fresh benchmark across
studies. No final-test predictions choose the intervention. Two runs are a
screen; broader Basis and modular-division confirmation, direct GPT tests and
same-assistant review are required before a production change.

All review work is performed by the same assistant. The learning outcome
remains pending.

Actual-shape CUDA preparation passes for 32- and 64-example training batches
and the 64-example validation batch. It takes 224.290 seconds, peaks at
414.08 MiB allocated, and leaves the complete model, optimizer and sampler
frame bit-identical with zero learning updates. The snapshot and source hashes
are retained under `analysis/`. This measures compilation/preparation, not
isolated training throughput.

Score scales need conversion between the two architectures. Historical QKNorm
uses unit vectors and SDPA `scale=1.0`, so its learned alpha is the multiplier
of the rotated cosine. Current Q/K RMS followed by SDPA's default scaling has
a score magnitude bounded by `gain * sqrt(head_width)`; finite epsilon reduces
it further. Thus the initial candidate gain of one corresponds to the same
ideal scale `sqrt(32)` as the historical initialization, not alpha=1. The
candidate retains current finite-epsilon RMS rather than asserting equivalence
to clamped unit normalization.
