# Convex exclusion gives one additional raw fix per mode

The complete task310 run and all 35 saved-model replays pass. Both 105-choice
policies retain 35/35 original primary successes and 35/35 auxiliary diagnostics
in full batch, minibatch and forward-only. Exact findings-commit independent
review remains pending. This is a seed1 comparison on the original suite used
during model development, not a fresh architecture-selection holdout.

Only the three Fibonacci endpoints change. Each policy explicitly inherits
the other 102 task/mode choices from reviewed task305. All 35 conventional
transformer endpoints, 336 depth outcomes, previous references, failed attempts
and inherited costs remain in the complete summary. Every source metric object
reproduces exactly before the new metrics are accepted.

| Mode | Task305 Fibonacci raw errors / CE | Task310 raw errors / CE | Task310 raw accuracy |
| --- | --- | --- | ---: |
| Full batch | 96 / 0.165498 | **95 / 0.164323** | **97.9804%** |
| Minibatch | 95 / 0.168674 | **94 / 0.167717** | **98.0017%** |
| Forward only | 95 / 0.171923 | **94 / 0.171430** | **98.0017%** |

The Fibonacci denominator is 4704 validation examples. Exact new CE values are
0.1643226369114892, 0.16771719805299373 and 0.17143042908302827. Pooled-field
correct counts remain 4701/4704 in every mode. Ordinary observed training CE
remains exactly 9.626594307035853e-9 and observed raw accuracy remains 4704/4704;
the entire observed metric object is unchanged.

| Mode | All35 raw correct / targets | Raw errors | Token-weighted raw accuracy |
| --- | ---: | ---: | ---: |
| Full batch | 279857 / 279952 | 95 | **99.9661%** |
| Minibatch | 279856 / 279952 | 96 | **99.9657%** |
| Forward only | 279858 / 279952 | 94 | **99.9664%** |

Both policies have identical totals. Minibatch retains two inherited
`arithmetic-sum-square` errors (4703/4705); all remaining errors are Fibonacci.
The aggregate weights large tasks more heavily, so it must be read beside
the per-task result rather than as a claim of uniformly near-perfect accuracy.
The raw totals use native counts where field metrics coexist and include all
35 primary denominators in every mode.

The independent raw transition audit finds exactly one changed validation
prediction, one fix and zero regressions per mode. There are 169 validation
queries with a new exclusion context, all raw-correct. The other 168 already
had the correct raw prediction. No training prediction changes. Thus the
large additional input-domain coverage translates into a small measured raw
accuracy gain; it does not remove the remaining 95/94/94 Fibonacci errors.

The prior identifies 11904 previously uncovered domain inputs and leaves 14112
ambiguous; the supplement identifies 12311 and leaves 14539 ambiguous. Neither
has an inconsistent candidate pool. These are complete-domain construction
counts, not validation counts. A conflict is certified when an observed point
of another class lies inside the augmented candidate hull. This implementation
does not test every possible intersection between two hulls. Candidate-pool
closure and convex target regions remain explicitly unproved assumptions;
neither follows from flatness or this benchmark result. Every unmodified row
retains the complete parent prediction.

Both complete 210-choice comparisons against task305 and task301 are preserved.
The new raw accuracy and CE improve on each reference in every changed mode.
Task296 source comparisons and the task298 lower-CE tradeoff remain in the
retained references. Task298 has CE 0.133659/0.132217/0.130133 and 105/107/105
raw errors: this new model improves raw decisions but does not dominate that
separate CE result. No held-out score selects the new contexts or confidence.

All complete states are sealed before new scoring. The head coefficients,
original records, order, repeats, short tails and mode metadata are unchanged.
The context rule adds zero gradients, optimizer updates or learned row updates.
All old and new complete complex products, raw-audit calls and reconstruction
passes are charged; zero new training is not zero verification work.

The endpoint buffer forecast is 13,129,615,360 bytes (about 12.23 GiB), and the
complete case reserves 1,778,118,179,328 scalar-work units, including three
construction passes. Metrics and the raw audit reserve 34,900,217,856 units
per endpoint. Actual peak RSS is 4,643,996 KiB, about 4.43 GiB, below the 64 GiB
process address limit. Preparation takes 46.41 seconds, all cases total 194.15s,
all replays 261.02s and summary 5.28s. The 56 GiB forecast cap and 16 GiB disk reserve
remain unchanged. These are reconstruction/evaluation times, not matched
training-compute comparisons.

Frozen producer: `ee4b6a44465ccf436e0a7dcf7e3753336a091dd7`; reviewed component
`9455f516`, Lean CM49 proof `26ec2faa`, and reviewed task305 source `5dd63c6e`.
Source fingerprint:
`0cbb1d45dd23c2f6e8ca0de908c2e6f0b420c15fd17225f15defabf92b1fb124`.
Inventory hash:
`955800d34a3dc56ddd4a1b5bd894c0283df167334441eeb8f624fd37dd2b4b70`.
Complete summary hash:
`5b43dafcca83d5ae48141ea8dd0394276b99aa382d9e6d072c8335a2094e9845`.
Report hash:
`f94d4c3f71580a1334ec0fe399cce6c75f5535ce5cd09c1e0e2c6d0fc4e6acdc`.
The 334-file manifest `.kanban/artifacts/310/completed.sha256` has hash
`6b1782a509e684c0153a7f1e31b15409df407b80c50c553abd2bda45b91b7b8f`.

Generated evidence remains ignored under
`/run/media/misha/storage/transformer-runs/raw-exclusion-compare/original-v1`,
with `report-v1.json` beside that directory. Frozen source, executable and
request remain in `freezes/raw-exclusion-compare-v1` under the same storage
root. Use the package README's native commands with that request and a fresh
output directory. Independent exact-commit review must prepare anew, rerun
all 35 cases and all 35 replays, reproduce every JSON and report byte-for-byte,
and rerun native/Lean/index/axiom and frozen-hash checks. Authored interpretation
and commands are committed; generated states, reports and logs are not.
