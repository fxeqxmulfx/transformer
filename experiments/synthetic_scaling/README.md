# GPTMini on noisy parity across widths, and on copy across sizes

Does GPTMini's test error descend twice as its width grows, at the sample
size where it can just fit noisy labels; and does a larger model copy
longer strings? These are the sweeps of `scaling.py` of the synthetic
trainers (its phases `dd-calibration`, `dd-widths` and `copy-scaling`), in
the lab's language: 27 runs.

## Runs

GPTMini with 8 heads trains under AMSGradW with betas (0.9, 0.999), epsilon
1e-8 and weight decay 0.1, on 64 rows per update, and is observed in chunks
of 64 rows. Every run is a memorization study at curve tolerance 0.02 on
data seed 1, and draws its batch order from its model seed, as the
historical trainer drew it.

| Labels | Runs | What they train |
| --- | --- | --- |
| `calibration-samples<n>` | 5 | parity of 16 bits with 20% of its training labels corrupted (noise seed 2), 3,000 updates at rate 3e-4 observed every 500, on n = 64, 256, 1,024, 4,096 and 16,384 training rows, each split a prefix of the next, with 128 validation and 256 test rows, tested at 32 and 64 bits; width 64 and 2 layers, from model seed 0 |
| `widths-width<width>-seed<seed>` | 18 | the same on 512 training rows, at widths 16, 32, 64, 128, 256 and 512, from model seeds 0, 1 and 2 |
| `copy-width<width>-depth<depth>` | 4 | binary strings of 1 to 32 symbols, 5,000 updates at rate 1e-4 observed every 1,000, on 16,384 training rows, 64 validation and 128 test, tested at 64 and 128 symbols, at widths 64 and 512 and depths 2 and 6, from model seed 0 |

512 is the geometric midpoint of the bracket the archived calibration found,
between the largest training split it fit (256) and the smallest larger one
it did not (1,024): `critical_sample_choice` of `scaling_calculations.py`.
16,384 is the power of two above the 12,118 rows a union bound requires for
every motif of 4 symbols to occur at every position of every length with
probability 0.95 (`motif_bound`): a coverage condition weaker than the
diversity condition of the RASP-Generalization Conjecture
(arXiv:2310.16028v1, Section 2), and no guarantee about training. Its source
cites Section 3 for that condition. Every run replays from CUDA graphs.

## Running

```sh
./make.py check experiments/synthetic_scaling              # the runs, and what differs between them
./make.py run experiments/synthetic_scaling [label ...]    # train every run, or the labeled ones
./make.py report experiments/synthetic_scaling [label ...] # what the runs recorded, as JSON
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again.

## Archived runs

The archived runs (RTX 3050, 2026-10-01, issued eagerly) are
[`baselines/amsgradw_softmax_scaling_20261002`](../archive/synthetic_trainers/baselines/amsgradw_softmax_scaling_20261002)
of the [synthetic trainers](../archive/synthetic_trainers), whose code is at
commit `5d64147`, under `experiments/synthetic_trainers/`. They found no
double descent in width: the mean test loss and error of the last models
over width descend twice nowhere, and the error of two seeds alone does, by
at most 0.07 around chance.

The calibration fit its noisy labels on 64 and 256 rows and not on 1,024 or
more. At 512 rows width 16 fit in no run, every larger width fit in every
run but one at width 512, and every width tested at sequence accuracy 0.48
to 0.57, the chance of parity. Copy tested at 0.96 at width 64 and depth 2
and 1.0 at the three larger sizes, and 0 at 64 and 128 symbols at every
size.

The runs replay from CUDA graphs here, so none is its archived run bit for
bit.
