# Random features and double descent

The local source is *Deep Double Descent*, arXiv:1912.02292v1, Appendix C,
Figures 14–15 (`papers/arXiv-1912.02292v1/rffs.tex`). The paper uses Fashion-MNIST,
a frozen Gaussian first layer of variance 1/d, complex activation exp(-i*x),
and a zero-initialized linear head trained with MSE by gradient flow. Its
interpolation peak follows n=d. Figure 15 fixes d=1,000 and varies sample count.

This trainer uses the official 60,000/10,000 Fashion-MNIST partitions and checks
all four compressed IDX files against the publisher's MD5 values. Every
prediction uses all 10,000 official test images. Training pools are uniform
samples without replacement, with nested prefixes within each seed. The
Gaussian matrix is independent of the data permutation. Width comparisons
reuse its column prefixes and apply the paper's 1/sqrt(d) scaling at each d.

Complex QR computes the minimum-norm least-squares limit of gradient flow
from zero. Underdetermined systems fit in the range of the adjoint design;
overdetermined systems project the targets onto the feature span. Tests compare
both with independent SVD solutions, check a null direction, and recover the
same head from the finite-time gradient-flow formula. No ridge or clipping
is added. Pivot, interpolation, and normal-equation residuals stay visible.

The manuscript does not specify pixel normalization, feature/data seeds,
finite training time, or how complex predictions become class labels. This
protocol records uint8/255 pixels, float64/complex128 arithmetic, argmax of the
real part, and the infinite-time limit. It reproduces the setting and curve
shape, with these explicit details; it does not claim identical figure values
or a reproduction of the paper's CNN/translation experiments.

The default campaign crosses n=1,000 and d=1,000 separately over 21 points,
including 950, 975, 1,000, 1,025, and 1,050, for seeds 0, 1, and 2: 123 fits.
Reports retain every run and pool fingerprint, means and sample SD, and finite
four-point double-descent witnesses with margin 0.02 for error and MSE. These
witnesses describe measured points rather than statistical significance.

```bash
.venv/bin/python -m experiments.synthetic_trainers.paper_reproduction.fashion_data \
  experiments/runs/paper_data/fashion-mnist
.venv/bin/python -m experiments.synthetic_trainers.paper_reproduction.rff_experiment \
  --device cuda --output experiments/runs/paper_reproduction/fashion-rff
```

The output must be fresh. `plan.json` freezes the dataset checksums, grid,
seeds, numerical protocol, and source hashes before fitting. Measurements are
saved after every fit, together with heads; all failed or unfavorable measured
points belong to the report.
