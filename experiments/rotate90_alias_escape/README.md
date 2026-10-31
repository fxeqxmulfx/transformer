# Escaping Rotate90 answer aliases

Why do narrow Schedule-Free models merge answers when Lean proves their architecture
can represent the complete powered-division table? This study tests the actual
gradient symmetry and interventions on six existing width-16 checkpoints.

The [hypotheses](hypotheses.json) and 50 definitions are recorded before follow-up
training. Source weights are pinned by SHA256. Every restart resets optimizer
histories and batch order, so the unchanged restart is the control for each
intervention. The original 175-run study and its checkpoints remain unchanged.

| Family | Cases | Difference from its control |
| --- | ---: | --- |
| Restart | 6 | Original query weights, new Schedule-Free state, 5000 updates |
| Reduced decay | 12 | Decay .1 or 0 instead of 1 |
| Frozen parameters | 12 | Freeze E or all branch matrices |
| Perturbed weights | 12 | Independent relative Gaussian noise .03 or .3 |
| Symmetry controls | 8 | Full-table, 200 updates: original, projected, perturbed projected, minibatch projected |

Tasks are P97/K2 and P193/K3, with source seeds 11, 17 and 23. The seed-11
checkpoints have approximate alias groups of sizes three and eight; projection
is used only in their symmetry controls. P193 seeds 17/23 are far from the
order-eight alias subspace. Checkpoint restarts otherwise preserve the learned E.
All new training uses Schedule-Free AdamW, Python 3.15, PyTorch 2.15 and
`torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)` on CUDA.

The Lean-proved exact result concerns full-table loss: when K is coprime to the
alias-group order, a symmetric embedding has a symmetric embedding gradient.
This is a condition on the data and current parameters, not a random-start
convergence assertion. Partial-table gradients and floating-point arithmetic
can break the symmetry. Both are measured explicitly below.

All 50 runs finished. None of the 42 half-table restarts reaches the prespecified
99% final heldout threshold in 5000 new updates. The eight full-table symmetry
controls measure a different question and have no heldout set.

Final heldout accuracy, in percent; each entry uses the Schedule-Free average:

| Task | Seed | Original | Restart | Decay .1 | Decay 0 | Freeze E | Freeze branch | Noise .03 | Noise .3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P97/K2 | 11 | 31.14 | 31.57 | 21.41 | 19.97 | 31.89 | 31.74 | 31.68 | 31.55 |
| P97/K2 | 17 | 31.59 | 31.14 | 22.10 | 20.90 | 31.68 | 31.10 | 31.08 | 31.55 |
| P97/K2 | 23 | 31.64 | 31.53 | 20.47 | 20.06 | 31.57 | 31.34 | 31.55 | 31.40 |
| P193/K3 | 11 | 11.79 | 11.79 | 8.15 | 8.01 | 11.87 | 11.63 | 11.66 | 11.65 |
| P193/K3 | 17 | 58.14 | 72.62 | 94.28 | 95.81 | 61.43 | 61.78 | 72.50 | 72.39 |
| P193/K3 | 23 | 45.09 | 74.31 | 98.26 | 98.39 | 57.24 | 61.13 | 74.45 | 73.80 |

Lower decay helps two P193 seeds substantially; it initially improves training
fit while harming heldout accuracy on the approximate quotient-code starts.
Freezing or perturbing weights does not yield a 99% solution in this budget.
These are finite-horizon findings. The subsequent user-requested
[duration/width study](../rotate90_alias_budget/README.md) tests whether those
failures persist with 20000 updates or a fresh width 64 model.

The actual full-table float64 gradient has transverse fraction 2.41e-16 at
P97/K2/order 3 and 3.15e-16 at P193/K3/order 8. The original half-table gradients
instead have fractions .223 and .220. Noncoprime exponent controls also break
the symmetry. Float64 diagnostics explicitly retain the original FP32 RMSNorm
epsilon, since PyTorch's default would otherwise change when casting.

After 200 compiled FP32 full-batch steps from projected starts, evaluation E's
relative projection residual is 7.31e-8 or 7.60e-8. The matched minibatch runs
have residuals .00888 or .01074. All eight symmetry controls predict the quotient
class correctly on 100% of pairs, while actual task accuracy remains about 34%
or 13%. Exact real-arithmetic symmetry and FP32 roundoff are distinguished.
CPU eager replay differs from compiled-CUDA argmax accuracy by up to .226
percentage points in the nearly tied projected controls; primary metrics remain
the recorded compiled-CUDA measurements.

[Results](results.json), [summary](summary.csv), [curves](curves.csv),
[source-checkpoint replays](sources.json), [gradient checks](gradient_checks.json)
and [frozen-tensor audit](intervention_audit.json)
retain every result, split fingerprint and provenance. All 12 frozen-parameter
interventions keep their selected tensors bitwise equal to the pinned sources. The exact theorem has
no random-start or partial-table convergence claim. Approximate aliases alone
do not establish a stable basin or permanent failure.

The pretraining definition SHA256 is
`b05f376e2b33d0209ce40a165651c5c4e390f5120be57a062a62919cde802741`;
the hypothesis SHA256 is
`39a41c1f645535daa3ae893560f268b11beed373bacf9291ef5d04b60a7ab46e`.

```sh
./run.py check experiments/rotate90_alias_escape
./run.py run experiments/rotate90_alias_escape
./run.py report experiments/rotate90_alias_escape
```

Replay the portable diagnostics from `python/` (after training):

```sh
uv run --locked python ../experiments/rotate90_alias_escape/analyze.py gradients
uv run --locked python ../experiments/rotate90_alias_escape/analyze.py escape
uv run --locked python ../experiments/rotate90_alias_escape/check_interventions.py
```
