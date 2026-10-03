# Frozen midpoint learning-rate calibration

The complete [original grid](../../baselines/amsgradw_stability_calibration_20261002/REPORT.md)
has no eligible recipe. At learning rate 0.0001, memorization precedes partial
generalization, but held-out accuracy never reaches 99%. At 0.0003,
generalization is early, the required memorization plateau is absent, and
late target failures remain. The next calibration tests 0.0002 between these
rates. Interpolation might balance these behaviors; it does not imply a
monotone response or predict success.

This is the same explicit GPTMini/raw AMSGradW adaptation of *Convexifying
Transformers*, Section 4, with only `learning_rate` changed relative to the
original short-final 0.001 control. Learning rate affects both the gradient
step and decoupled shrinkage `1 - learning_rate * weight_decay`. This experiment
does not independently identify their causal effects.

The [immutable plan](midpoint-plan.json) was frozen before the first update,
using the completed negative comparison's artifact fingerprint. It retains
the same 19 source hashes, two local manuscript hashes, environment, model,
optimizer, data split, and success criterion. The launcher has its own frozen
SHA256. Initialization/data seeds remain 0/0; this is calibration, not an
independent repeat. Width is 128, depth two, heads four, batch size 512 with
short-final batches, decay 0.1, warmup ten, prime 97, and train fraction 50%.

The entire 150,000-update budget is required regardless of intermediate
targets. Canonical evaluation and diagnostics are every 250 updates, with
immediate neighbor probes. Stable grokking requires the unchanged train≥99% /
held-out≤10% memorization plateau for five consecutive observations spanning
at least 1,000 updates before first held-out 99%, then 20 consecutive joint
99% observations, and joint 99% at every canonical observation in the final
50,000 updates. Rapid generalization without that plateau is a distinct result.
Missing event times retain null values and support zero; later failures remain
in the report. No target, cadence, or budget is changed after inspecting outcomes.

Launch or resume from the repository root:

```bash
.venv/bin/python -m experiments.synthetic_trainers.protocols.amsgradw_stability_20261002.run_midpoint --check-only
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.amsgradw_stability_20261002.run_midpoint
```

The launcher compares the live and committed manifests, its own fingerprint,
and current sources, split, environment, and manuscripts before training. The
default `stability --resume` CLI reconstructs the original four-recipe grid
and is therefore unsuitable for this custom single-recipe stage.
Live output is `experiments/runs/amsgradw_stability_20261002/calibration_midpoint_lr0002/`.
Preserve complete histories and checkpoint continuation after interruptions.

After completion, use `stability_report` to create the fresh individual archive
`baselines/amsgradw_stability_short_lr0002_seed0_data0_20261002`, then
`stability_comparison` for
`baselines/amsgradw_stability_midpoint_calibration_20261002`. Verify both without
PyTorch, visually inspect the PNG/PDF curves, retain the negative parent grid,
and commit the completed result with a measured handoff.

Only a complete passing calibration may freeze the six independent cases
(initialization seeds 4/5/6 crossed with data seeds 2/3). All six must pass
before the architecture benchmark proceeds. Failed results become calibration
evidence and are retained; further tuning needs a justified fresh frozen plan
with unchanged criteria. Architecture and complementary-mechanics studies
remain pending.
