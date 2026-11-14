# Raw accuracy improves with observed-region contexts

The complete task305 run improves both raw accuracy and ordinary CE relative
to the task296 source and task301 guarded reference. All 35 original seed1
cases and all 35 reconstruction replays pass. Both 105-choice policies retain
35/35 original primary successes and 35/35 all-split diagnostics in full batch,
minibatch and forward-only modes. Exact findings-commit independent review is
pending; the original run and its reconstruction replays are complete.

The frozen producer is `64efc9f82da630cc660692a15f8a204110677f83`, using reviewed
component `0fee1a9d` and CM48 proof `71d96c93`. Eligibility changes the Fibonacci
representation in three modes. Each policy retains the other 102 choices with
explicit inherited-metric attribution. All 35 conventional transformer
endpoints, 336 depth outcomes, earlier references and the failed task301 v1
attempt remain in the complete summary. Source metrics reproduce exactly.

The following counts are raw labels on 4,704 Fibonacci validation examples;
they are distinct from the original pooled-field success metric.

| Mode | Source296 errors / CE | Guarded301 errors / CE | New305 errors / CE | New raw accuracy |
| --- | --- | --- | --- | --- |
| Full batch | 101 / 0.172615 | 101 / 0.170831 | **96 / 0.165498** | **97.9592%** |
| Minibatch | 100 / 0.175106 | 100 / 0.173737 | **95 / 0.168674** | **97.9804%** |
| Forward only | 100 / 0.177280 | 100 / 0.176289 | **95 / 0.171923** | **97.9804%** |

Across all 35 primary splits there are 279,952 target tokens per mode. Their
token-weighted raw totals are 279,856 correct (99.9657%) in full batch,
279,855 (99.9654%) in minibatch and 279,857 (99.9661%) in forward-only. Both
policies give these same totals. The only additional raw errors outside
Fibonacci are two inherited minibatch errors on `arithmetic-sum-square`
(4703/4705). These totals weight large tasks more heavily; the per-task table
above exposes the remaining concentration of errors.

Each mode gains five correct answers, or 0.1063 percentage points. Exact new
CE values are 0.16549829994461673, 0.1686739651131691 and
0.17192274938116286, respectively. Pooled-field accuracy remains 4701/4704
(99.9362%) in all modes. Observed raw accuracy remains 4704/4704 and ordinary
training CE remains exactly `9.626594307035853e-9`.

The retained task298 five-gate result still has lower CE: 0.133659, 0.132217
and 0.130133, with 105, 107 and 105 raw errors. The new model improves raw
accuracy but does not dominate this separate CE tradeoff. All 210 choices
are also compared explicitly against task301, so improvement against the
weaker source CE cannot hide a regression against that stronger reference.

The prior covers 3,878 unseen inputs and the supplement 3,932 across the full
65,536-input domain in each mode. There are 93 and 94 separated ambiguous
blocks respectively, with no intersecting observed class hulls. The prior
retains 26,016 uncovered and 29,623 unsupported inputs; the supplement retains
26,850 and 29,055. These domain counts are not validation counts. Geometry
alone does not verify the unknown target's convex-region assumption. The
remaining 96/95/95 raw errors and the unverified assumption stay explicit.

New context construction adds zero gradient batches, optimizer updates or
learned row updates in every mode. The entire original complex head and all
training records remain unchanged. Geometry uses fixed rational mixtures of
actual source features; no arithmetic answer formula or held-out label enters
construction. Confidence is one, fixed before the original evaluation. All
complete models were saved on storage and sealed before any new metric.

The maximum live endpoint forecast is 7,494,148,096 bytes (about 6.98 GiB),
and the complete case reserves 491,044,824,576 scalar-work units, including
inherited costs and three construction passes. Actual peak RSS is 3,663,992
KiB (3.4943 GiB), below the 64 GiB process limit. Preparation took 44.15 seconds;
all case executions totalled 158.11 seconds, all replays 202.80 seconds and the
summary 3.27 seconds. These measured times include reconstruction and original
metric checks; the zero new-update count does not erase inherited training or
verification costs.

Reproduction inputs and outputs remain local and ignored:

- Frozen source, executable and request: `/run/media/misha/storage/transformer-runs/freezes/raw-regions-compare-v1`.
- Complete original output: `/run/media/misha/storage/transformer-runs/raw-regions-compare/original-v1`.
- Complete report: `/run/media/misha/storage/transformer-runs/raw-regions-compare/report-v1.json`.
- The 334-file manifest: `.kanban/artifacts/305/completed.sha256`.
- Component review: `.kanban/artifacts/304/review`; producer checks: `.kanban/artifacts/305/precommit`.

| Binding | SHA-256 |
| --- | --- |
| Source fingerprint | `aae77450f5dc67f2a92cc0b97ddad8b00c35e77aedb24af97aeca00f3092c9c1` |
| Executable | `90e9572f01862552d54abf4c10162324c44469352b1a5d8a815235a48b22a25d` |
| Inventory | `5dbf85022d80822f03a80ed164313be4fd965e5a2fd8dbc144d79ac8f516fd25` |
| Summary | `285f2d748866b833c25a4dc23785468718d7fe899949872287dca29ec990ee71` |
| Report | `b6025d78b63e47dba1bf140cc5a9b4fbe96eccbe8a5ae0074a4b0a331c5dd9d3` |
| Completed manifest | `b28d298a8cfacf8e801082be1ae2c2eec4204b4b926b520930af59ad0325c01d` |

Use the package README commands with the frozen executable and request and a
fresh output directory. The exact findings-commit review must independently
prepare and rerun all cases/replays, compare every JSON artifact and the report
byte-for-byte, and recheck frozen hashes. Generated evidence is preserved
outside worktrees. This is a seed1 result on a suite already used during model
development; it is not evidence for unseen tasks or a three-seed result.
