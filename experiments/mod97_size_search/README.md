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
| graph-w32-wd02 | 32 | 32,232 | Captured CUDA graphs; decay 0.2 |
| graph-w24-wd02 | 24 | 19,568 | Captured CUDA graphs; decay 0.2 |
| graph-w16-wd02 | 16 | 9,976 | Captured CUDA graphs; decay 0.2 |
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
uv run --locked ../run.py check experiments/mod97_size_search
uv run --locked ../run.py run experiments/mod97_size_search graph-w128
uv run --locked ../run.py report experiments/mod97_size_search graph-w128
```

Runs, full model/optimizer/sampler checkpoints, engine fingerprints and
exhaustive histories live in ignored `runs/<label>/`. Width refinements and
additional seeds will be added under new labels when the coarse bracket is
known. A reported minimum means the smallest successful tested configuration,
not a proof that every smaller architecture fails.

## Fresh results

All completed runs below used exactly 100,000 updates, or 10,000 full epochs.

| Run | Final training / held-out | Sustained joint 99% onset / confirmation | Final 1,000 epochs stable | Delayed grokking |
| --- | --- | --- | --- | --- |
| graph-w128 | 100% / 100% | 3,675 / 4,150 | No; minimum held-out 96.28% | No strict near-chance plateau |
| graph-w64 | 100% / 100% | 1,375 / 1,850 | Yes; minimum held-out 99.87% | No strict near-chance plateau |
| graph-w32 | 99.81% / 96.41% | Not reached | No | Not reached within cap |
| graph-w16 | 61.96% / 37.26% | Not reached | No | Not reached within cap |
| graph-w32-wd02 | 100% / 100% | 7,675 / 8,150 | Yes; minimum held-out 99.98% | Yes |
| graph-w24-wd02 | 99.89% / 37.46% | Not reached | No | Not reached within cap |
| graph-w16-wd02 | 69.61% / 34.60% | Not reached | No | Not reached within cap |

Changing only weight decay from 0.1 to 0.2 brought the width-32 transition
inside the budget. Training first reached 99% at epoch 825. Five consecutive
near-chance held-out observations with fitted training began at epoch 1,050
and were confirmed at 1,150. This 32,232-parameter model is currently the
smallest tested delayed-grokking model. Widths 24 and 16 with decay 0.2
failed within the cap. Seven fresh dense runs are complete; the larger
Mesh GPT search is now active in the Rust repository. The remaining labels
in the configuration table are predefined candidates, not completed runs. Portable full learning curves, history hashes and the shared
stricter criteria are archived in the Rust repository's
`docs/grokking-results/summary.json`; the lab's built-in transition detector
uses a shorter two-observation patience and is reported separately.

## Validation

The study checker accepts all twelve configurations, and direct PyTorch model
construction verifies every parameter count in the table. The required
`uv run --locked ../run.py test` ran 196 tests in 905 seconds and reported
69 failures and three skips. A later required rerun took 511 seconds with
the same failure and skip counts. The final refinement rerun took 516
seconds, again with 69 failures and three skips. The failures include exact historical CPU
fixture comparisons; the modular fixture was recorded with PyTorch
2.14.0+cu126 while the locked environment uses 2.14.1+cu126. An isolated
`test_engine.py` rerun also fails at update zero. This study changes no lab
source, test, optimizer, framework pin, or archived run, so the historical
reproduction issue remains a separate limitation of the current environment.
Fresh graph runs record their own framework, device and source fingerprints.
