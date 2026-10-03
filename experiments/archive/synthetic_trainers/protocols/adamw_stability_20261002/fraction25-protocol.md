# Mod-193 AdamW training-fraction control

The complete mod-97 optimizer pair and mod-193 half-split calibration are
negative under the unchanged phase and persistence gates. The first mod-193
held-out target occurs at 40,000 without a memorization plateau; seven final
window observations fail. This next calibration tests whether reducing the
observed training fraction creates the missing phase, without assuming that
it solves persistence.

The [frozen plan](fraction25-plan.json) records one complete 150,000-update
recipe, `adamw-mod193-fraction25-lr001`, with training fraction 25% instead of
50%. All other training configuration fields, source hashes, instrumentation,
environment, local manuscripts and criteria match the completed mod-193
parent. Native AdamW uses betas (0.9, 0.98), epsilon 1e-8, bias correction,
no maximum buffer, all-parameter decay 0.1, learning rate 0.001 and warmup 10.
GPTMini retains width 128, two layers, four heads and 436,104 parameters.
Both answer and EOS must be correct. Use initialization/data seeds 0/0,
short-final 512-example batching and exhaustive evaluations every 250 updates,
with all gradient norms and unchanged neighbor/tensor probes. Do not stop
at a target or drop failures.

The [prepared complete corpus and CPU smoke](fraction25-preparation.md)
check 9,264 train / 27,792 held-out equations, vocabulary 335, all legal pairs
and class coverages, nested training/common held-out sets and identical CPU
initial states. Epochs now have 18 full batches plus a 48-example tail;
full-budget exposure is 73,137,184 examples. These derived changes in epoch
length, sampling, exposure, held-out support and evaluation cost are explicit.
The native initial model and parameter count are preserved. This remains
novel operand-pair generalization at fixed lengths, not length transfer.

The prospective criterion is unchanged: train ≥99% / held-out ≤10% for five
consecutive evaluations spanning 1,000 updates before first held-out 99%;
later joint ≥99% for 20 consecutive evaluations; joint ≥99% at all 201
observations over updates 100,000–150,000. Only a full passing primary
calibration can freeze six new confirmations (model seeds 4/5/6 crossed with
data seeds 2/3). Every case must pass before architecture selection. None of
those later stages is complete or launched by this calibration freeze.

Two analysis-only changes since the completed parent are recorded: the
verified ordinary CSV optimization and readable comparison labels/layout.
All training/task sources remain identical. The strict launcher checks the
new analysis hashes, unchanged criterion/environment/papers, exact fraction
intervention, its own SHA256 and the complete verified negative parent archive.

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_fraction25
```

Add `--check-only` to verify without scientific training. Live output is
`experiments/runs/adamw_stability_20261002/calibration_mod193_fraction25_lr001/`.
Finish and verify the whole budget, archive every observation/log/CSV,
review PNG/PDF figures and commit the measured outcome. Keep this single
calibration distinct from independent confirmation, architecture improvement
and scientific complementary mechanics, which remain outstanding.

This is an explicit adaptation of *Convexifying Transformers*, Section 4.
The manuscript does not specify these fractions or the stronger persistence
rule. One calibration cannot establish repeatability, universal optimizer
stability, an internal algorithm or a causal speedup.
