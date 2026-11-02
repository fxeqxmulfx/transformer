# Basis comparison: same-assistant review

Task160 is complete:24 cases,21 first99% validation crossings and three
explicitly censored hard-recall endpoints. The conditional-gain candidate
improves large-model parity crossing time in both seeds, but regresses on
hard depth, small recall and small parity. Production GPT remains df1e9f7.
This assistant reviews the work; no delegated reviewer was used.

## Protocol and source correspondence

The study preserves the original Basis generators, labels, input formats,
20,000/512/512 splits, data seed1, batch-seed convention, learning rates,
training batches,50-update warmup and3,200-example observation cadence.
Small/easy means width64/depth2; large/hard means width128/depth6; all use
four heads and model seeds0/1. Original split fingerprints match exactly.

The actual current-GPT adapter and conditional-gain candidate have independent
source correspondence checks. Full-position and supervised-position losses
and every parameter gradient agree for depth, recall and parity. Parity uses
free generation including EOS; hard depth stops on length128 validation.
Every endpoint is current full state, with no retrospective best-state restore.
Schedule-Free AdamW, Python3.15.2, Torch2.15.0, FP32 CUDA fullgraph max-autotune,
math SDPA and evaluation batch64 are explicit deviations from old CPU AdamW
Basis. The source/data/runtime protocol stays byte-identical to18112982.

## Continuation and data isolation

All initial chunks completed before continuation1's validation-only decisions.
Canonical hard-recall seed1 rose from51.27% to82.52% token accuracy between
75,000 and100,000 updates, so it was extended within a200,000-update horizon.
The other three complete100,000-update states were retained and censored.
This was a recorded research decision, not a preregistered numeric threshold;
unobserved crossings are neither infinite times nor proofs of impossibility.

The extension was interrupted at102,750. An elevated process check confirmed
no trainer remained. Recovery6d0f7b34 retained the exact interrupted phase,
manifest, checkpoint and raw logs before restoring model, optimizer, sampler
and clocks. Compiled CPU interruption/resume agrees exactly with an
uninterrupted run, excluding elapsed clocks. Trailing unsaved logs are retained
separately. Stale phases, changed horizons and disclosed final tests are rejected.
The resumed run reached507/512 validation sequences at123,550 and stopped.
Only then did the phase close and final-test evaluation begin.

Training-subset probes at26,100 and100,000 on canonical seed0 reproduce all
reported counts. Both working and averaged weights have the same broad
memorization/validation gap. This rules out a simple weight-mode explanation
at those checkpoints; it does not identify a causal architectural defect.

## Results and independent audit

The final audit verifies all24 checkpoint hashes, finite saved model tensors,
650,300 consecutive finite nonzero gradient records, every retained history
aggregate and every first crossing. All endpoint validation prediction counts
agree between canonical CUDA records and independent CPU replay. The full
unreduced report is retained with its SHA256; storage reduction drops only
repeated history class tables and preserves every endpoint class table.

Large parity takes8,400/10,200 updates with gains versus14,000/14,000 for the
control. Small parity, small recall and hard depth are slower with gains.
Canonical hard-recall seed1 scores98.44% on final test after99.02% validation;
its three censored companions score0% exact test sequences at their endpoints.
Different horizons remain visible in the report.

Parity has only157/512 novel validation inputs and184/512 novel test inputs.
Large-parity novel test is96.74%/97.83% for the controls and92.93%/100% with
gains; faster sampled validation does not establish improved novel-input
accuracy in both seeds. Small-depth length128 test is also mixed:
canonical/gain63.67%/100% and99.80%/86.33%. All final scores are descriptive,
not used for stopping or architecture selection. No gain architecture is promoted.

## Verification and next comparison

The complete CPU suite at6d0f7b34 passes500 lab tests in790.99 seconds with13
CUDA skips and104 optimizer tests in12.09 seconds with14 skips. Five focused
continuation/recovery checks pass in30.04 seconds. Earlier source-matched CUDA
preparation checks cover all three large shapes and both recall batch sizes,
with fullgraph execution and unchanged model/optimizer state. They are not
isolated throughput measurements. The unrelated CUDA Muon allocation failures
remain task157; this cohort uses Schedule-Free AdamW only.
The final focused Basis suite passes17 tests in40.06 seconds. The independent
audit is retained as executable source beside its result and runs after the
normal report command's storage reduction.

The user's historical check confirms successful original GPTMini hard mode in
all three seeds. Hard-recall crossings were3,350/2,850/2,600 at bd63e509 and
2,950/2,150/2,750 in the b7d6e673 softmax repetition. Committed records and
source hashes are retained; the original ignored checkpoints are absent.
Optimizer and architecture both changed. Task162 therefore compares original
GPTMini under the required Schedule-Free AdamW before isolating learned QK
temperature, residual normalization, epsilon, activation or initialization.
Task161 retains the conditional-gain transfer question. These finite training
results do not refute a representational theorem or prove a depth lower bound.
