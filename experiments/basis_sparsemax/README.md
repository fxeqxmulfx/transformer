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
No full benchmark result has been recorded yet.

Run a pair with:

```sh
./make.py check experiments/basis_sparsemax
./make.py run experiments/basis_sparsemax softmax-easy-small-depth-seed0 sparsemax-easy-small-depth-seed0
./make.py report experiments/basis_sparsemax softmax-easy-small-depth-seed0 sparsemax-easy-small-depth-seed0
```
