# Sparsemax on the basis

Where does sparsemax attention fail where softmax passes the calibrated
[basis](../basis/README.md), and is a failure removed by a neighboring rate?
This is steps 0 and 1 of [EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md).
The arms change only the attention weights. Both keep QKNorm, XSA, the
basis's recipes, splits, seeds and thread counts. Hard parity is the easy
parity run, as in the basis.

| Labels | Runs | Difference |
| --- | ---: | --- |
| `softmax-<mode>-<model>-<task>-seed<seed>` | 30 | The basis repeated under softmax |
| `sparsemax-<mode>-<model>-<task>-seed<seed>` | 30 | Euclidean simplex projection instead of softmax |
| `time-<weights>-small-<task>` | 6 | 300 updates at the easy recipe, without early stopping, observed every 100 |
| `sparsemax-<mode>-<model>-<task>-seed<seed>-lr<rate>` | 60 | The two adjacent rates of `RATES`; only failures whose softmax control passes are trained |

`Diagnostics` samples every observed update: `loss` is the
last training batch's mean supervised cross-entropy before that update, and
`temperatures` holds each head's `log_alpha` and `inverse_temperature`
(e^alpha) after it. Update 0 has no training batch. Diagnostics also
records the existing per-tensor gradient, update and moment norms.

## Found

Preparation on 2026-10-04 used torch 2.14.1 on the basis's Xeon E5-2680 v4
CPU. The lab sources are those of ea18d55; the run manifests keep their
hashes and this experiment's source. Sparsemax compiles in a full graph,
including its custom backward. Tests compare its weights and score
gradients at lengths 1, 17 and 64, and all logits and parameter gradients
of the small GPTMini at length 64, to eager evaluation within float32
rounding. A short compiled modular run under each normalizer records
identical observation metrics, model, optimizer and sampler state with and
without diagnostics; the sparsemax engine also checks interruption and
resumption against an uninterrupted run. `./make.py test` passes all 142
tests, including the CUDA graph and historical trainer checks.

Each timing run trains 300 updates on one physical core at the easy recipe.
The table divides `training_seconds` by 300, excluding compilation,
evaluation and diagnostic time:

| Task | Softmax, ms/update | Sparsemax, ms/update | Sparsemax / softmax |
| --- | ---: | ---: | ---: |
| depth | 22.580 | 37.949 | 1.681 |
| recall | 84.349 | 144.095 | 1.708 |
| parity | 6.691 | 8.281 | 1.238 |

All are below step 0's factor-of-two ceiling. The timing controls'
validation records match bd63e50's archived basis exactly at every common
observation: updates 0 and 200 for depth and parity, and 0, 100, 200 and 300
for recall. The full benchmark will check the rest of each trajectory.
The small-model comparison is complete; the large-model comparison is in progress.

### Small model

All 30 original small-model runs have finished. The 15 softmax controls
repeat all 661 non-timing observations of the archived basis exactly;
their model, optimizer and sampler checkpoints are bit-identical too.
The update of each pass, or the best selection sequence accuracy of a
failure, is below. Batch losses are the last sampled training batch before
the final update, not an exhaustive training-set evaluation.

| Mode | Task | Seed | Softmax | Sparsemax | Last batch loss, softmax | Last batch loss, sparsemax |
| --- | --- | ---: | --- | --- | ---: | ---: |
| easy | depth | 0 | passes at 200 | passes at 400 | 0.019239 | 0.002907 |
| easy | depth | 1 | passes at 200 | passes at 200 | 0.026029 | 0.032107 |
| easy | depth | 2 | passes at 200 | passes at 200 | 0.025272 | 0.021076 |
| easy | recall | 0 | passes at 1,650 | passes at 550 | 0.010832 | 0.305198 |
| easy | recall | 1 | passes at 2,450 | passes at 350 | 0.011766 | 0.843603 |
| easy | recall | 2 | passes at 1,800 | fails, 41.21% | 0.008468 | 0.183331 |
| easy | parity | 0 | passes at 8,200 | fails, 87.11% | 0.009457 | 0.064472 |
| easy | parity | 1 | passes at 5,000 | fails, 96.68% | 0.076804 | 0.015308 |
| easy | parity | 2 | passes at 4,200 | passes at 14,200 | 0.065896 | 0.000511 |
| hard | depth | 0 | fails, 44.34% | fails, 32.62% | 0.000196 | 0.000460 |
| hard | depth | 1 | fails, 34.96% | fails, 44.34% | 0.024446 | 0.004452 |
| hard | depth | 2 | fails, 31.05% | fails, 33.59% | 0.000307 | 0.000205 |
| hard | recall | 0 | fails, 0.00% | fails, 9.57% | 3.361794 | 1.167205 |
| hard | recall | 1 | fails, 0.00% | fails, 0.00% | 2.503449 | 2.310641 |
| hard | recall | 2 | fails, 0.00% | fails, 0.00% | 2.549058 | 3.820318 |

Sparsemax learns the easy depth from every seed, and recall much earlier
than softmax from seeds 0 and 1; from seed 2 it fails recall within 4,800
updates. Parity takes longer and fails from two seeds at its recipe.
Neither normalizer passes the small model's hard depth or hard recall.

H5 removes both parity failures: rate 1e-4 passes both. The larger neighbor
1e-3 passes seed 1, but not seed 0. The adjacent-rate checks for the recall
failure are queued with the remaining step 1 work.

| Parity seed | Recipe, 3e-4 | Neighbor, 1e-4 | Neighbor, 1e-3 |
| ---: | --- | --- | --- |
| 0 | fails, 87.11% | passes at 13,400 | fails, 95.31% |
| 1 | fails, 96.68% | passes at 9,800 | passes at 16,600 |

The full-precision numbers, final head scales and hashes of the source
records are in [small_results.json](small_results.json). Large-model
results and the final list of mechanism targets remain pending.

Run a pair with:

```sh
./make.py check experiments/basis_sparsemax
./make.py run experiments/basis_sparsemax softmax-easy-small-depth-seed0 sparsemax-easy-small-depth-seed0
./make.py report experiments/basis_sparsemax softmax-easy-small-depth-seed0 sparsemax-easy-small-depth-seed0
```
