# Conditional native AdamW schedule preparation

Prepared on 2026-10-03, Europe/London, while the frozen sparsemax/softmax pair
continues with constant learning rate. No scientific schedule manifest or run
has been selected or launched. Finish both normalizer budgets, verify/review
and commit their complete results before selecting another calibration.

The complete softmax continuation already retains the required memorization
phase and delayed generalization, but fails persistence at 274,000 and
275,500 despite recovery. This supports a separately labeled investigation
of update size during the final window. It does not identify the cause.
If the fresh softmax control passes, use the unchanged independent-confirmation
gate; do not automatically launch the conditional adaptation.

The proposed controlled intervention changes only `learning_rate_schedule`
between `constant` and `cosine_tail`. Both tagged configurations carry the same
fixed annealing start, end and final factor. Preserve native AdamW, original
softmax GPTMini, corpus, seeds, initialization, betas, epsilon, all-parameter
decay amount, warmup, batching, diagnostics, cadence, full 300,000-update cap,
and every phase/persistence criterion. This is an adaptation to the Section 4
task in *Convexifying Transformers*, not a claim about the author's schedule.

The proposed scientific interval is completed updates 150,000–250,000. The
rate multiplier is `0.1 + 0.9 * (1 + cos(pi * progress)) / 2`, with progress
clamped to [0, 1]. It remains 0.1 over the complete final window. The original
ten-update warmup and all early rates remain exact. Target observations never
start or stop annealing. The native per-update decay also scales with learning
rate, so the intervention changes both effective update and decay amounts;
their contributions are not isolated.

| Completed updates before the next update | Constant rate | Proposed annealed rate |
| --- | ---: | ---: |
| 0 | 0 | 0 |
| 10 | 0.0003 | 0.0003 |
| 150,000 | 0.0003 | 0.0003 |
| 200,000 | 0.0003 | 0.000165 |
| 250,000 | 0.0003 | 0.00003 |
| 300,000 | 0.0003 | 0.00003 |

The [tagged trainer](../../scheduled_training.py) patches only the existing
rate function and source dictionary within its call. It reuses the unchanged
model factory, initialization, native optimizer, sampling, original loop,
checkpoint validation and diagnostics. Actual optimizer-group rates are
recorded in both full tensor diagnostics and every gradient record. The config
records the schedule before loading; changing it or its factor on resume fails
before checkpoint loading. The two tagged training sources are pinned in both
paths. The ordinary core cannot silently interpret an annealed archive.

The [portable validator](../../scheduled_integrity.py) adapts the repository's
original native-AdamW checks by replacing only their expected-rate calculations.
It retains exact complete history, exhaustive scoring, batches/exposure,
diagnostic cadence, neighbor support, native moment/parameter norms, gradient
agreement and learned-temperature checks. No stored rate or other record is
rewritten. The [archive wrapper](../../scheduled_report.py) uses these checks
and restores the original functions after success or failure; completed archives
verify without Torch, checkpoints, original run paths or manuscripts.

Five [actual CPU checks](schedule-training-validation.json) pass: a full-width
40-update constant trajectory is identical to the original core in model,
native buffers, sampling, gradients and diagnostics; a 30→40-update annealed
continuation matches 40 uninterrupted updates; its early trajectory and actual
rate anchors are checked; changed resume/invalid configs/exceptions are rejected;
and real complete negative archives verify without Torch. A forged rate with
both logs/CSV/checksums regenerated is rejected. Eight additional corrupted
non-rate fixtures are also rejected, including missing native buffers.
The tiny CPU extension is implementation evidence, not a scientific extension.

The actual mod-193 scientific initialization is also checked on CPU: 436,104
parameters, identical initial tensors and RNG in both tagged paths. Every
current 43 Python/nine Lean normalizer fingerprint remains unchanged. These
checks are not evidence of improved learning or stability. A future scientific
launcher, immutable paired manifest, all case/archive guards, recovery/time
comparison, and complete-budget curve review are still required before launch.
The all-six independent benchmark and later architecture/complementary gates
remain unchanged. Do not replace them with this preparation or the current
single exploratory normalizer pair.
