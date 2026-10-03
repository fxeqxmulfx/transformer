# Observed delayed generalization in the quarter-split calibration

The frozen mod-193 GPTMini / AdamW calibration with 25% training pairs now
has the required order of phases. Its longest qualifying pre-target plateau
spans updates 4,500–11,500: 29 consecutive exhaustive observations with
train ≥99% and held-out ≤10%. An earlier qualifying plateau spans
1,000–3,000 (nine observations), interrupted by a training-fit decline.
The first held-out 99% crossing occurs at update 20,000, with train/held-out
complete RHS accuracy 100%/99.5862%. All 20 consecutive joint target
observations from 20,000 through 24,750 pass. Long confirmation costs
678.96 training / 830.95 wall seconds from the run start.

This is measured delayed generalization under the unchanged prospective
criterion. It is an adaptation of *Convexifying Transformers*, Section 4,
on modular division and fixed-length novel operand pairs. Native AdamW
uses betas (0.9, 0.98), bias correction, all-parameter decay 0.1, learning
rate 0.001, warmup 10 and no clipping. Width 128, two layers, four heads,
436,104 parameters and seeds 0/0 match the frozen
[quarter-split plan](fraction25-plan.json). The thresholds and stronger
persistence requirement are explicit study choices.

The [portable prefix](first-long-confirmation-fraction25) preserves all
100 canonical observations through update 24,750 and all 24,750 pre-update
gradient records, with the original full-budget configuration, split and
source/manuscript fingerprints. The largest prefix gradient is 575.4626
at update 14,550 on a full 512-example batch. Accuracy was not measured
at that exact update; this value does not establish a collapse or its cause.
The [PNG](fraction25-phase-curves/phase-prefix.png) and
[PDF](fraction25-phase-curves/phase-prefix.pdf) show every canonical point
without smoothing and explicitly label the partial calibration.

[Offline validation](first-long-confirmation-fraction25-validation.json)
recomputes the assessment, supports, exposure, costs, gradient rates and
artifact hashes without PyTorch or original run paths. Six isolated corrupted
copies are rejected after recomputing artifact hashes: missing canonical or
gradient observations, incomplete exhaustive support, a negative gradient,
an amended criterion and absent actual memorization. The last corruption
also recomputes the assessment and first train target, so a summary-consistency
check alone cannot reject it. All 31 frozen source/manuscript files are
unchanged. The standalone positive-phase verifier is separate from the
historical verifier that explicitly requires an absent plateau.

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_phase_prefix
```

This prefix leaves the 150,000-update budget incomplete. The final window
100,000–150,000, six fresh independent confirmations, architecture comparison
and scientific complementary tasks remain outstanding. Event timing is
observed; eligibility as persistent-benchmark timing is unresolved. Finish
the frozen budget and retain every failure before applying the confirmation
gate. This single calibration does not establish repeatability, universal
optimizer stability, an internal algorithm or a causal advantage over its
half-split parent.
