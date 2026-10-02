# Early AdamW generalization under the unchanged phase gate

The frozen 50% AdamW calibration first reaches canonical held-out 99% at
update 1,250, with train/held-out complete RHS accuracy 100%/100%. The first
20-observation joint confirmation spans 1,250–6,000, costing 181.81 training /
194.91 wall seconds. At update 1,000 the scores are 99.5490%/97.4227%.
There is no required train ≥99% / held-out ≤10% memorization plateau before
the first crossing, so the recipe is phase-ineligible for the stable-grokking
benchmark regardless of its later persistence. Rapid generalization is a
separate observation.

This is the mod-97 division adaptation of *Convexifying Transformers*, Section
4, with softmax GPTMini, 50% exhaustive split, native AdamW betas (0.9, 0.98),
learning rate 0.001, decay 0.1, seeds 0/0 and unchanged full-budget criterion.
The [frozen optimizer-pair plan](optimizer-pair-plan.json) remains unchanged.

The [archived prefix](first-long-confirmation) preserves all 25 canonical
observations from zero through 6,000 and all 6,000 pre-update gradient-norm
records, with batch sizes, tail flags and learning rates. Its summary retains
the immutable source/manuscript/split fingerprints and complete partial
assessment. The largest observed prefix gradient is 28.4592 at update 3,793,
after an ordinary 512-example batch. No accuracy was measured at that exact
update; a large gradient alone does not identify a collapse or its cause.

The prefix is incomplete. Long confirmation does not certify the last
50,000-update tail, behavior between evaluations, or independent repeatability.
Finish both frozen 150,000-update budgets, retain every failure, verify full
portable archives, and review the curves before the next calibration selection.
A separately frozen train-fraction control can test a longer memorization
phase; no such new recipe has been selected or launched by this inspection.
The primary optimizer remains AdamW, as explicitly selected by the user.

Recompute the prefix summary and artifact checks without PyTorch or original
run paths:

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_first_confirmation
```
