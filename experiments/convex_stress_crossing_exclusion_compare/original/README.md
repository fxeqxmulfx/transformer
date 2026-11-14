# Hull crossings lower CE slightly without fixing another raw answer

Task314 completes all 35 original cases and all 35 saved-model replays. Both
105-choice policies retain 35/35 original primary successes and 35/35 auxiliary
diagnostics in full batch, minibatch and forward only. Raw accuracy is unchanged
from reviewed task310. Exact findings-commit independent review follows this
record. This is the seed1 suite used during model development; earlier suite
outcomes informed the architecture, so it is not a fresh architecture holdout.

Only the three Fibonacci endpoints have new metrics. Each policy explicitly
inherits the other 102 task/mode choices from task310. All 35 conventional
transformer endpoints, 336 depth outcomes, prior references, failures and costs
remain in the complete summary. Source metric objects reproduce exactly before
new metrics are accepted. The conventional baseline retains 33/35 primary
successes; the separately recorded DiffGPT baseline has 35/35 on each of seeds
0, 1 and 2. They are distinct baselines.

| Mode | Fibonacci raw correct / targets | Raw accuracy | Task310 CE | Task314 CE |
| --- | ---: | ---: | ---: | ---: |
| Full batch | 4609 / 4704 | 97.9804% | 0.164322637 | 0.164271130 |
| Minibatch | 4610 / 4704 | 98.0017% | 0.167717198 | 0.167655642 |
| Forward only | 4610 / 4704 | 98.0017% | 0.171430429 | 0.171368873 |

Exact new CE values are 0.16427112972771699, 0.1676556422115525 and
0.17136887324158667. Pooled-field correct counts remain 4701/4704 in every
mode. The complete observed metric object is unchanged: raw accuracy is
4704/4704 and ordinary observed CE is 9.626594307035853e-9.

| Mode | All35 raw correct / targets | Raw errors | Token-weighted raw accuracy |
| --- | ---: | ---: | ---: |
| Full batch | 279857 / 279952 | 95 | 99.9661% |
| Minibatch | 279856 / 279952 | 96 | 99.9657% |
| Forward only | 279858 / 279952 | 94 | 99.9664% |

Both policies have identical totals. Minibatch retains two inherited
`arithmetic-sum-square` errors (4703/4705); every other raw error is Fibonacci.
These totals include all 35 primary denominators and use native raw counts
where field metrics coexist. Large tasks have greater weight in this aggregate;
the Fibonacci result remains necessary to assess the unresolved errors.

Every mode's independent raw transition audit reports zero changed predictions,
zero fixes and zero regressions, on both training and validation. The stronger
geometry covers 91 additional validation queries, all already raw-correct.
Their margins can change, explaining the small CE reduction without an accuracy
gain. Thus the larger geometric coverage does not solve any of the remaining
95/94/94 Fibonacci errors. More hull coverage alone has not supplied the missing
raw-label information in this experiment.

Across the complete input domain, the prior adds 6350 context rows and leaves
7762 ambiguous; the supplement adds 6540 and leaves 7999 ambiguous. Neither
candidate pool is inconsistent. These are construction counts, not validation
counts. Each accepted exclusion verifies both rational mixtures, their supports
and masses, the positive tagged query coefficient and exact coordinates. Every
other parent row is retained. Candidate-pool closure and convex target regions
remain explicit unproved assumptions; flatness provides feature bounds, not
an unconditional identification theorem for missing answers.

All 210 choices are compared separately with task310, task305 and task301.
Against task310 there are six CE changes (three modes in each policy), no raw
accuracy changes and no primary-pass changes. Task296 references and task298's
lower-CE tradeoff remain available. Task298 has CE 0.133659/0.132217/0.130133
with 105/107/105 Fibonacci raw errors, so task314 does not dominate its CE.
No held-out label or score chooses the new rule or confidence.

Every complete new state was sealed before scoring. Full head coefficients,
original records, repetitions, order, short tails and training modes are
unchanged. The new rule performs zero gradient batches, optimizer updates or
learned row updates. All complete complex products, raw-audit calls and repeated
constructions are charged; the absence of new training does not remove these
evaluation and verification costs.

The largest endpoint buffer forecast is 19,491,694,592 bytes (about 18.15 GiB).
The complete case forecast is 3,125,113,330,176 scalar-work units, including
three construction passes and 34,900,217,856 metric/audit units per endpoint.
The prospective allowance was raised to 4T before scoring because the
training-only forecast already exceeded the old 3T cap. The 56 GiB buffer cap,
64 GiB process address limit and 16 GiB free-disk reserve were retained.
Actual peak RSS is 6,112,288 KiB (about 5.83 GiB). Preparation takes 50.78s,
all cases total 242.24s, replays 321.30s and summary 6.90s. All process exits
are zero. These are reconstruction/evaluation times, not matched training costs.

Frozen producer: `9373e8539d09444b371b47088f5fd65072a21054`; reviewed component
`a6be8048`, Lean CM50 proof `2b17dc86`, and reviewed task310 source `df79e005`.
Source fingerprint:
`21ccf60c95e56a69cd6ce3254fde2f18fe73e010abbaebbe60873f0153ff8c42`.
Inventory hash:
`b48f62536a84dcf9ebf68dd5218533ea1def76795f5f86a39129c3106e40ad29`.
Complete summary hash:
`93e97d47cc2e741929114d5ae0122c8f2ee070c26a66ddc167f7e65687ebc0c0`.
Report hash:
`c94dc00aefa2086865705578baefddab9e24fe8d4661f63edf761a2f538321f3`.
The 334-file manifest `.kanban/artifacts/314/completed.sha256` has hash
`1fb54dfc771c0d8b92cabf4b5837895f4c927829d7dba75867ac99c450031713`.

Generated evidence remains ignored under
`/run/media/misha/storage/transformer-runs/crossing-exclusion-compare/original-v1`,
with `report-v1.json` beside that directory. Frozen source, executable, request
and protocol remain in `freezes/crossing-exclusion-compare-v1` under that storage
root. Besides the 999 Cargo dependency files, the freeze pins 21 external static
source inputs; all literal source inclusions were resolved before scoring.
Use the package README's native commands with that request and a fresh output
directory. Exact-commit review must prepare anew, run all 35 cases and all 35
replays, reproduce every JSON and the full report byte-for-byte, and pass native,
Lean, INDEX, axiom and frozen-hash checks. Only authored interpretation is committed.
