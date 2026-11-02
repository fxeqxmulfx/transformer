# Historical GPTMini control: same-assistant review

Task162 addresses the optimizer/architecture confounding in the solved Basis
history. The original GPTMini definition at bd63e509 is retained exactly at
width128, depth6 and four heads. New runs use only Schedule-Free AdamW with
the same implementation, rate, betas, decay, warmup, batches and seed convention
as task160. Model/data seeds0/1 match the completed current-model controls.
No production architecture change or new model flag is part of this study.

`BasisControl` reuses the first99% reserved-test contract while keeping the
historical supervised-position model interface. The task builder preserves
original data generators and fingerprints. Shared endpoint analysis is moved
without changing the current-GPT objective. Query conditioning is explicitly
restricted to the residual model that declares it; the historical model uses
the same Schedule-Free stepper with no coordinate-conditioning operation.

The focused tests compare the actual historical recipe and the task160 frozen
split hashes, full masked loss and every parameter gradient, exact compiled
CPU continuation across an epoch remainder, first-crossing stopping and final
test isolation. The first fixture incorrectly tried to change model and
benchmark in separate DSL replacements, creating an invalid intermediate
configuration. Atomic construction fixes that fixture and recipe; all10
then-focused checks pass in36.84 seconds. A further report-disclosure guard
is included in the complete suite. No learning ran before this correction.

The control has1,249,816 parameters versus1,249,024 for current GPT; both use
the same width, depth and heads. Matching this optimizer isolates a whole
architecture comparison. It cannot attribute any difference specifically to
QK temperature, Peri normalization, epsilon, activation or initialization.

The frozen protocol also hashes the committed current-model result, original
experiment and exact engine sources. Every new training update retains its
gradient norm and complete-state checkpoints; sampled first99% validation
chooses the endpoint. Both new final tests remain sealed during learning.
These are reused public Basis rows, not a fresh test distribution across the
entire research program. A new component intervention requires its own declared
comparison. No representational theorem is claimed falsified by a training gap.

The full CPU suite passes505 lab tests in1243.36 seconds with13 CUDA skips
and104 optimizer tests in13.26 seconds with14 skips. The existing unrelated
CUDA Muon allocation issue remains task157. Actual-shape CUDA preparation
passes at both32- and64-example batches, preserves the complete model,
optimizer and sampler frame exactly, and takes zero updates. It takes187.55
seconds with366.75MiB peak allocated memory; this is compilation/preparation,
not an isolated learning-throughput measurement.

The frozen GPU cohort started after that preparation while CPU verification
finished. Its source bytes are pinned in the protocol even though the first
run began before the implementation commit. Seed0 reaches99% at12,200 updates.
Seed1 remains unresolved; no new final test is open. Copied-state diagnostics
at seed0's endpoint and seed1 step22,350 are retained. At averages, both fit
all512 sampled training examples, while validation is507/512 versus0/512.
The gap is present at both Schedule-Free states. Averaged head temperatures
range3.99–9.59 in seed0 and4.34–10.52 in seed1; larger temperatures alone do
not explain success. This is exploratory evidence, not a causal diagnosis.
Both copied-state probes reproduce their complete per-class result hashes
exactly, including all aggregate scores and both sets of head temperatures.

The remaining learning outcome and paired final review are pending.
The same assistant performs this review; no delegated reviewer is used.
