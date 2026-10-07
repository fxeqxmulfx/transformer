# Grokking progress measurements

Started 2026-10-07 at the user's request. Status: active; implement and
experimentally assess a causal observer on ordinary softmax transformers.
The previous [Basis/convex cycle](plan.md) is paused, unfinished. ANSR stays
stopped. Work solo and follow AGENTS.md.

## Objective

Make hidden progress during a memorization plateau measurable. Separate
current rule formation, cleanup and observed delayed generalization from
a prediction that training will succeed. Use unchanged ordinary AdamW.

## 1. Reject weak signals using existing histories

Apply a causal, pinned held-out log-loss slope to archived ordinary
transformer histories and available negative controls. Keep answer loss
separate from EOS loss where recorded. Record false positives, fit times,
accuracy crossing times, complete budgets and source fingerprints.
Loss improvement alone is insufficient: existing mod-193 sparsemax gives
a negative case with loss decline and no 99% generalization by 300,000.

Status: archived causal accounting completed for 13 histories, with source
hashes in `experiments/grokking_progress/archived_loss_baseline.json`.

## 2. Observe functional structure without altering training

Use a fixed common-scaling projection of current division logits, adapting
Nanda et al., arXiv:2301.05217v1, section 5.1. The main signal pools only
held-out logits to prevent training memorization from supplying projected
answers. Remove global bias, row shifts and the easy zero-quotient case
from energy. Require correctness in addition to invariant energy.

Reject constant logits, insufficient orbit coverage, future information
and symmetry-only wrong rules. Verify actual CPU/CUDA AdamW trajectories,
optimizer and sampler states, RNG and checkpoint resume before long runs.
Keep raw metrics separate from pinned phase thresholds.

The user also requested internal observations. Read actual answer-query
attention/FFN/residual features with temporary hooks; record held-out orbit
coherence, activation RMS, per-head attention entropy/routes and actual
parameter/gradient/update norms. Treat norm changes as scale-dependent
evidence, not a rule-success certificate. Preserve the already running
observer-v1 trajectory and read its saved checkpoints for new internals;
explicitly record observation versus training hardware.

Status: implemented. Twenty-three focused observer/report tests pass,
including unchanged actual CPU/CUDA AdamW checkpoints and resume,
temporary-hook cleanup, constant/wrong symmetric rules and a training-only
memorizer. The complete `./make.py test` suite passes: 233 tests on
2026-10-07, including available CUDA checks.

## 3. Run the complete ordinary-transformer budget

Use [grokking_progress](experiments/grokking_progress/README.md): GPTMini
softmax seeds 1, 2 and 3, plus the reference-transformer memorization
control. Every run retains the archived recipe's 150,000-update budget.
Record observer overhead separately. Compare current functional evidence
with later accuracy only for evaluation, not for constructing each signal.

Report lead times, false alarms and immediate-learning exclusions. Known
seed repetitions test instrumentation; they do not establish independent
predictive validity. If useful, add unseen seeds and tasks with different
symmetries before claiming a transferable detector. Do not resume the
convex-architecture search unless the user redirects work back to it.

Status: two full-budget runs active, two CUDA seed controls queued. Fresh
primary evidence: structure signal at 33,000 (14.39% held-out accuracy),
first 99% at 35,500, confirmed delayed generalization at 36,500. Internal
snapshots at 30,000 and 35,000 show strong changes in the second block
while overall weight norm changes only 0.2%. This is a partial-budget,
known-seed positive example. Complete controls and general predictive
validity are unfinished; do not mark this research direction complete.
