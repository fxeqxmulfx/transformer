# Original results with five shared count-profile gates

Five shared gates lower Fibonacci validation cross-entropy in all modes while
retaining 35/35 original primary and auxiliary successes in each mode and both
initializer policies. Raw accuracy improves in full batch but regresses in
minibatch and forward-only. This is a CE improvement with a retained raw-label
tradeoff, not an improvement on every metric. All 35 cases and 35 replays finish
with exit0, and every eligible CM45 parent metric reproduces exactly. Complete
independent reproduction follows the authored findings commit before approval.

| Fibonacci mode, 4704 validation records | CM45 errors | Five-gate errors | CM45 CE | Five-gate CE |
| --- | ---: | ---: | ---: | ---: |
| Full batch | 109 | 105 | 0.1401069095 | 0.1336592553 |
| Minibatch | 103 | 107 | 0.1419598638 | 0.1322171895 |
| Forward only | 100 | 105 | 0.1490438375 | 0.1301328768 |

Raw correct counts are 4599, 4597 and 4599, respectively. The original field
accuracy remains 4701/4704, or 99.9362%, in all three modes. Observed CE stays
9.626594307e-9. Each policy changes the same three Fibonacci choices and
inherits 102 reviewed choices. All other original task/mode metrics remain
explicit. No primary success is lost; the mini and forward raw regressions
are retained separately from that success decision.

The prior block-local result from task296 remains in full: its full/mini/forward
errors were 101/100/100 and CEs 0.1726145666/0.1751058800/0.1772798206. Five-gate
CE is lower than both previous variants in every mode, but its raw counts are
worse than the block-local variant in every mode. The best prior raw count
remains 100 errors. The complete task296 summary, all completed artifacts,
both 105-choice policies, every split metric, fresh model outcome and work
count are pinned and retained in the new summary/report. No held-out fallback
selects between them. Earlier original-suite results motivated this model
change, so the suite is not an untouched architecture-selection holdout.

The five fixed source-excluded class-count gates share coefficients across
source blocks. They preserve the original distinct complex features, CM45
anchor, radius 0.5 and 91-map bank. The prior component retains full step8;
mini and forward retain initialization. Their source-excluded diagnostic CEs
are 0.05749619, 0.06105835 and 0.06083512. These diagnostics are higher than the
block-local variant's, while validation CE is lower. This is consistent with
less local overfitting, but it does not establish a unique cause or identify
unknown labels. The diagnostic does not refit latent discovery after exclusion.

Across six prior/supplement components, work is 288 new gradient batches,
176 optimizer updates and 797 local-row updates. Full contributes 128 batches,
16 updates and 80 row updates; mini contributes 160 batches, 160 updates and
717 row updates; forward contributes zero gradients and updates. Gradient
record visits total 677376 and scoring visits 7366464, including repeated
records and the 96-record mini tail. Every inherited constructor and optimizer
cost remains visible. Sharing rows reduces local-row updates from 7887 in the
block-local variant; the gradient-batch count does not decrease.

All 35 conventional transformer endpoints and 336 depth outcomes remain,
without retraining. The frozen conventional Fibonacci endpoint has 95.1318%
raw accuracy and CE0.234544; the five-gate model has 97.7253-97.7679% raw accuracy
and CE0.130-0.134. Historical DiffGPT's recorded 35/35 on three seeds is a
separate result. This is not a matched-compute or new DiffGPT experiment.

Preparation takes 27.49 seconds, the original cases total 90.02 seconds and
their replays 128.92 seconds; summary generation takes 2.55 seconds. Peak case
RSS is 2025432 KiB and peak replay RSS 3164016 KiB, about 3.02 GiB. The eligible
case forecasts 22168327680 buffer bytes and 195987782784 scalar-work units.
Every process has the 64 GiB address-space limit. Timing describes this run,
not a speed comparison. The complete inventory has 333 hashed files, including
preparation outputs and the report; no failed numerical case was removed.

Reproduce producer `448f034d1e66d7a74234bd9719220ddb83394e4b`, component
`5e89c616`, generic Lean `c60816fa`, parent findings `f0a19727`, and the negative
local reference reviewed at `96aee0fc`. The frozen source fingerprint is
`a4046f038b9f32d9c5b7f512813cb929eb6eaeef35c936cd38cc2f2f79623bec`;
the executable hash is
`31a595b418536ad5f2da7714d3810b64ac70e5b6e6e6c001cce4d110cff380b7`.
The inventory hash is
`71eb6a2b265b097637c3e90624b165ecb7367a56a86d6f826c92bf2cb60f5500`;
the summary hash is
`e4b401d61429dc74e104489c62861777795ac5012b8129d29383e0a6354931f3`;
the report hash is
`8f5e57ccc959c1c01a4fca20e04d3cf39a0a52b5cc063be80a382780711ee9e5`.
The completed-manifest hash is
`07383d6017f2b46ce00398680a000687af0d037bc4e42f299edd03cb79f2d051`.

Evidence is retained under `transformer-runs/count-gates-compare/`; source,
request and executable are in `transformer-runs/freezes/count-gates-compare-v1/`.
The parent README documents regeneration. Exact-commit review must reproduce
fresh preparation, all 35 cases and replays, all 116 JSON artifacts and the
report, together with native/Lean/index/axiom and frozen-source checks.

The next question is whether convex margin constraints can preserve stronger
raw predictions while retaining confidence improvements. Such constraints
would stabilize a fixed reference model, not certify that its unseen labels
are correct. Ordinary CE, all original modes and the complex bounds remain
required; this experiment alone does not solve the remaining raw-label errors.
