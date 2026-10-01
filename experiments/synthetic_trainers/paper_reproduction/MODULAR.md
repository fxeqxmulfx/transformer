# Modular division and delayed generalization

The local source is *Convexifying Transformers*, arXiv:2211.11052v1,
Section 4, Figures 6–10. Its standard transformer fits mod-97 training data
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
