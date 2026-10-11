# Mod 97 dense size search

Find the smallest tested dense GPTMini that learns `a / b mod 97` within
10,000 complete training epochs, then repeat the search with Mesh GPT in
`/home/misha/rust_projects/grokking_abmod`. This starts from the successful
recipe in [mod97_grokking](../mod97_grokking): AdamW, learning rate 0.001,
weight decay 0.1, betas (0.9, 0.98), and ten warmup updates.

The exhaustive corpus has 9,312 equations, excluding zero denominators.
The fixed data seed 1 splits it into 4,656 training and 4,656 held-out rows.
A training epoch visits every training row once, with nine batches of 512
and a final batch of 48. Thus 100,000 updates is exactly 10,000 full epochs.
Both splits are evaluated exhaustively every 25 epochs. Runs use model
seed 1, batch-order seed 10,001, two layers and four heads. Only width varies
in the primary search. No run stops at a transient accuracy peak.

| Run | Width | Learned parameters | Execution |
| --- | ---: | ---: | --- |
| graph-w144 | 144 | 532,088 | Captured CUDA graphs |
| graph-w128 | 128 | 423,816 | Captured CUDA graphs |
| graph-w96 | 96 | 244,136 | Captured CUDA graphs |
| graph-w64 | 64 | 113,608 | Captured CUDA graphs |
| graph-w48 | 48 | 66,776 | Captured CUDA graphs |
| graph-w32 | 32 | 32,232 | Captured CUDA graphs |
| graph-w16 | 16 | 9,976 | Captured CUDA graphs |
| graph-w8 | 8 | 3,456 | Captured CUDA graphs |
| native-w128 | 128 | 423,816 | Native eager control |

The count includes the learned tied embedding and eight head-temperature
parameters, counting tied tensors once. RMSNorm has no learned scale.
CUDA graphs use capturable AdamW, which rounds differently from native
AdamW. The eager control checks the archived recipe; graph runs compare
sizes under the same execution mode. Results from the two modes are
reported separately.

A generalization event requires both training and held-out equation accuracy
at least 99% for 20 consecutive observations (475 epochs from first to
last). Delayed grokking also requires an earlier memorization plateau:
training at least 99% and held-out at most 10% for five consecutive
observations. Report the first 99% crossing separately. Stability at the
budget requires every observation in the final 1,000 epochs to meet 99% on
both splits. These are finite observations, not a guarantee between them.

The dense author-format benchmark predicts both the quotient and EOS and
has 239 vocabulary tokens. Mesh GPT currently predicts only the quotient
from four prompt tokens, uses a smaller vocabulary and a different seeded
split. Architecture searches keep their own data fixed; cross-architecture
results must state these differences.

From `python/`, validate or run using the locked environment:

```sh
uv run --locked ../make.py check experiments/mod97_size_search
uv run --locked ../make.py run experiments/mod97_size_search graph-w128
uv run --locked ../make.py report experiments/mod97_size_search graph-w128
```

Runs, full model/optimizer/sampler checkpoints, engine fingerprints and
exhaustive histories live in ignored `runs/<label>/`. Width refinements and
additional seeds will be added under new labels when the coarse bracket is
known. A reported minimum means the smallest successful tested configuration,
not a proof that every smaller architecture fails.

The fresh width-128 graph run has reached sustained 99% generalization;
its final-budget result is still pending. Smaller models have not run yet.

## Validation

The study checker accepts all nine configurations, and direct PyTorch model
construction verifies every parameter count in the table. The required
`uv run --locked ../make.py test` ran 196 tests in 905 seconds and reported
69 failures and three skips. The failures include exact historical CPU
fixture comparisons; the modular fixture was recorded with PyTorch
2.14.0+cu126 while the locked environment uses 2.14.1+cu126. An isolated
`test_engine.py` rerun also fails at update zero. This study changes no lab
source, test, optimizer, framework pin, or archived run, so the historical
reproduction issue remains a separate limitation of the current environment.
Fresh graph runs record their own framework, device and source fingerprints.
