# Modular division and delayed generalization

The local source is *Convexifying Transformers*, arXiv:2211.11052v1,
Section 4, the figures labeled `fig:grokking_p97_L1` and `fig:grokking_p97_L12`.
Its standard transformer fits mod-97 training data
around 1,000 updates and generalizes after more than 100,000. This trainer
reproduces that arithmetic setting and measures the delay to generalization.
It does not implement the paper's convex architecture or claim its speedup.

The reference architecture is copied from
[openai/grok at 3d64b1](https://github.com/openai/grok/tree/3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3),
with its MIT license preserved. The only source removal is an unused Lightning
import. The two-layer, four-head, width-128 model uses ReLU, post-layer
normalization, sinusoidal positions, untied embeddings, and no dropout. It
trains in float32. AdamW uses the author's betas (0.9, 0.98), epsilon 1e-8,
all-parameter weight decay, and ten-update warmup to a constant learning rate.
No gradient clipping or learning-rate annealing is used.

The native PyTorch AdamW implementation was independently compared with the
pinned author's `CustomAdamW` in its zero-noise, `to_zero` mode over 20 updates,
including warmup. Parameters matched in float32 and float64; the source hash
and settings are saved in [reference-optimizer-check.json](reference-optimizer-check.json).
This checks the update recurrence, without claiming bitwise reproduction of
an entire GPU training run across library versions.

Every equation is `EOS a / b = c EOS`, where `b` is nonzero and `c = a/b`
in the prime field. Both `c` and EOS are supervised. Causal masking prevents
the teacher-forced answer from entering its own prediction. The complete
9,312-equation domain and legacy NumPy shuffle exactly match the author data
generator; tests independently check every answer with modular inversion.
Composite moduli are rejected because division then needs a different domain.

The fixed default protocol uses 1,862 train equations and all remaining 7,450
equations as exhaustive held-out data, batch size 512, learning rate 0.001,
weight decay 1, and 150,000 updates. The local manuscript does not disclose
its exact train fraction and regularization. These choices are explicit
experimental settings, not recovered paper hyperparameters. The complete
budget runs regardless of held-out scores, with evaluation every 250 updates.
Train/held-out onset at 99% and two consecutive held-out observations define
the reported delay. This measures new operand pairs at the same prime.

```bash
.venv/bin/python -m experiments.synthetic_trainers.paper_reproduction.grokking \
  --model reference --optimizer adamw --seed 0 \
  --output experiments/runs/paper_reproduction/mod97-reference-seed0
```

The output must be fresh. `--resume` restores model, moments, shuffled epoch,
cursor, and batch RNG exactly from `checkpoint.pt`. The configuration must
match; increasing `--steps` is allowed but is recorded as a post hoc extension.
Outputs include a frozen plan, corpus fingerprints, source hashes, history,
final measurements, and resumable checkpoints every 5,000 updates.

`--model gptmini` replaces only the model with the repository's softmax GPTMini
and matrix initialization standard deviation 0.02. `--optimizer amsgradw`
uses the repository's raw AMSGrad update, betas (0.9, 0.999), no bias correction,
and decoupled decay. These are explicitly labeled adaptations. Comparisons
must retain identical data, observation points, budgets, and failed runs.

For all three controls across seeds 0/1/2, run the serial confirmation campaign:

```bash
.venv/bin/python -m experiments.synthetic_trainers.reproduction \
  --output experiments/runs/paper_reproduction/mod97-confirmation
```

The campaign freezes nine recipes before training. `--resume` skips verified
completed runs and continues checkpoints; source hashes and configuration must
match. `--adopt-completed` can incorporate an existing complete matching run
without retraining it. It rejects unfinished runs, different sources/configs,
and unrelated files in the directory. Fixed budgets cannot change on campaign
resume. Individual exploratory budget extensions use the separate trainer CLI.
`--train-fraction`, `--weight-decay`, and `--learning-rate` select a fresh,
explicit campaign protocol for calibration of details missing from the paper;
they cannot change an existing campaign on resume.

Paper reports also distinguish the first train threshold crossing from a
two-observation train fit. A separate memorization-phase diagnostic requires
two consecutive observations with train accuracy at least 99% and held-out
accuracy at most 10% before held-out onset. The 10% ceiling is an explicitly
chosen conservative diagnostic for mod 97, whose answer chance is 1/97; the
paper does not prescribe it. Reports retain later collapses, the fraction of
observations at target after confirmation, and final scores. A transient fit
or two early held-out successes alone do not establish persistent performance.
`--data-seed` permits an independent split for confirmation after calibration.

Reports record cumulative training and wall seconds at both the observed
sustained-target onset and its confirmation. These are scheduled observations,
without interpolation between evaluations. Training time excludes evaluation
and checkpoint writing; wall time includes those costs. Aggregate time-to-target
means include only runs reaching the target and report their support; failures
remain missing rather than becoming zero cost. Final accuracy and later
collapses must accompany timing comparisons. Historical archives retain their
original analysis; new reports include these measurements.
