# Tensor diagnostics at selected late regressions

This post hoc inspection uses the complete four-recipe raw AMSGradW / GPTMini
calibration, an explicit adaptation of *Convexifying Transformers*, Section 4.
For each recipe, it selects the canonical observation with the lowest held-out
accuracy in the last 50,000 updates, and the latest earlier canonical
observation with train accuracy at least 99%. The reference condition concerns
only train fit: the 0.0001 reference has 67.16% held-out accuracy and never
reached the held-out target. These selected points are not independent repeats
or a representative summary of all updates.

Tensor norms come from unchanged raw diagnostic records. Gradients concern
the sampled batch before its update; parameter/update norms and temperatures
are measured after it; complete split accuracies are evaluated after it.
The two selected points are 250 updates apart, with many unmeasured gradients
between them. A larger gradient at a failing point may follow an earlier
regression; this inspection does not determine temporal cause.

| Recipe / observation | Update | Train / held-out | Gradient L2 | Global update/weight | Largest tensor update/weight | Embed squared-gradient share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-lr001 / Earlier train fit | 132,250 | 100.00% / 100.00% | 0.0978 | 0.0108% | 0.0434% | 35.34% |
| short-lr001 / Worst tail | 132,500 | 58.91% / 58.14% | 26.1051 | 1.3429% | 4.9384% | 62.64% |
| wrap-lr001 / Earlier train fit | 110,250 | 100.00% / 100.00% | 0.0845 | 0.0102% | 0.0464% | 44.36% |
| wrap-lr001 / Worst tail | 110,500 | 1.59% / 1.98% | 36.8381 | 4.0458% | 18.2370% | 53.78% |
| short-lr0003 / Earlier train fit | 102,750 | 100.00% / 100.00% | 0.2025 | 0.0083% | 0.0757% | 74.98% |
| short-lr0003 / Worst tail | 103,000 | 46.80% / 40.55% | 17.1718 | 0.4637% | 1.6074% | 70.60% |
| short-lr0001 / Earlier train fit | 149,750 | 100.00% / 67.16% | 0.3902 | 0.0024% | 0.0172% | 81.19% |
| short-lr0001 / Worst tail | 150,000 | 31.27% / 17.53% | 48.4098 | 0.2694% | 0.8589% | 71.52% |

At all four selected regressions the tied embedding/unembedding tensor has
the largest gradient norm, contributing 53.78%–71.52% of the summed squared
gradient norms. It also has the largest gradient at every selected reference;
its reference shares range from 35.34% to 81.19%. Dominance alone therefore
does not distinguish a failing update. The maximum tensor update ratios at
regressions occur in attention projection or QKV weights: 4.94% and 18.24%
for short/wrap 0.001, 1.61% for 0.0003, and 0.86% for 0.0001.

Inverse temperatures stay in similar ranges at each selected pair. The source
diagnostics retain every head value. At the worst
points their ranges are approximately 1.000–1.047, 1.001–1.017, 1.082–1.131,
and 1.527–1.578 respectively. This does not exclude unobserved changes between
samples or a role for attention. It does not identify temperature explosion
at these four recorded updates. Short-final diagnostics use 48-example batches
at both selected points; the wrap control uses 512. These batch-specific norms
should not be interpreted as estimates with identical sampling variability.

The measured associations motivate inspecting how actual updates compare to
submodule weight norms when designing a later control. They do not certify a
clipping, schedule, fraction, or architecture intervention. The running
0.0002 control, original targets, and independent-confirmation gate are unchanged.

The [inspection JSON](collapse-tensor-inspection.json) retains all per-tensor
norms, derived shares/ratios, selection rules, source archive fingerprints,
and the analysis SHA256. The [script](inspect_collapse_tensors.py) validates
the complete source comparison before analysis and checks the aggregate
gradient and share identities. It runs without PyTorch:

```bash
python3 -m experiments.synthetic_trainers.protocols.amsgradw_stability_20261002.inspect_collapse_tensors --output /tmp/amsgradw-collapse-tensor-inspection.json
```

Use a fresh output destination. All numbers refer to the full-budget original
grid, not the incomplete midpoint control.
