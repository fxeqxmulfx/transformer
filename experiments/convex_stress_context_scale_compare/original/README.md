# Original CM43 confidence results

Training-only confidence calibration reduces Fibonacci validation cross-entropy
from approximately0.27997472 to0.16604154 while retaining4574/4704 correct raw
labels (97.2364%) and4701/4704 correct field answers (99.9362%). Full-batch,
minibatch and forward-only endpoints have these results. Both initializer
policies retain35/35 primary and auxiliary successes in every mode. All35
original case processes and their numerical replays exit0.

| Mode | Raw accuracy | Field accuracy | CE before calibration | CE after calibration |
| --- | ---: | ---: | ---: | ---: |
| Full batch | 97.2364% | 99.9362% | 0.2799747191651211 | 0.16604154431811982 |
| Minibatch | 97.2364% | 99.9362% | 0.2799747191653473 | 0.16604154431817347 |
| Forward only | 97.2364% | 99.9362% | 0.2799747191650992 | 0.16604154431811502 |

Combined with the preceding reviewed CM42 change, raw accuracy rises from
66.3265% to97.2364% and CE falls from8.08163266 to0.16604154. This calibration
step changes confidence only: the130 remaining raw errors and three field
errors are unchanged. Original observed CE remains9.626594307035853e-9.
No optimizer update or gradient batch is added. These are original seed1
results, not a new three-seed result or an unseen-answer identification proof.

The retained conventional transformer has95.1318% raw accuracy,99.0434% field
accuracy and CE0.23454365089024853 on this Fibonacci endpoint. The calibrated
complex head now has higher measured raw and field accuracy and lower CE.
The transformer values are inherited from the reviewed baseline; this is
not a new transformer run or a matched-compute comparison.

All three distinct saved context heads select scale0.5 for both their prior
and supplemental components. The five-neighbor CM42 context maps stay intact.
Canonical training-only CE falls from0.14218948 to0.08339867 for the prior,
and from0.14729152 to0.08597368 for the supplement. All4704 original training
record IDs and their multiplicities remain. The query's observed input state
is excluded from its canonical neighborhood; discovery and neighborhood
selection still use all training data. Trained coefficients and held-out
targets never enter confidence selection. Actual inference scales the complex
features before the unchanged complete coefficient product.

Each initializer policy retains105 choices: three fresh changed endpoints and
102 explicitly inherited ones, or one fresh plus32 inherited per mode. The
complete summary retains all35 conventional transformer and336 depth outcomes,
all earlier constructor accounting, and the parent's CM42 forecasts and resource
artifacts. Original raw labels, field pooling and the99% primary success
threshold remain unchanged. No held-out policy selection is performed.

The measured maximum original RSS is780,884KiB, below the64GiB process cap.
The changed-case run takes29.30s and its original numerical replay30.25s,
each including all three modes. Preparation takes7.98s and summary generation
2.14s. The combined three-head buffer forecast is15,109,934,592bytes and
metric-work forecast13,693,833,216units. These CPU measurements use the optimized
test profile with debug assertions and include reconstruction and evaluation.
Earlier training and construction work remains charged to the parent chain.

The frozen producer is`123dfc9cf1ed2e8806701186a85b6d84992beded`. The reviewed
component is`0eb9fd7c`, following CM43 at`2dcaa38a`. Six adapter tests and the
six reviewed component tests pass, with formatting, clippy and direct Lean
checks. Lean has13885 build jobs and36 existing sorry declarations, with zero
results resting on sorry and zero extra axioms. The generated index has4788
modules and25056 theorems, with zero vacuous or placeholder declarations.

| Evidence | SHA256 |
| --- | --- |
| Source fingerprint | `dd3cd653cd9df1f08759017119cd425bd8d6cd3e4fed4ece6d2baa6269f58c57` |
| Frozen executable | `9da1ccee8d26b0c847064e1ec4b97aaa1645335dfe92adce3d654df7f06085fa` |
| Original inventory | `a40b905b2aa1adc93edea52dc8cd0e453b39be950cbeafd389dd6ab5a95dea70` |
| Original summary | `2de0d1d7959d8dba5dddff30a8035fbdc0bd5f2e2f14be8073971454954dd4e5` |

Generated evidence stays at
`/run/media/misha/storage/transformer-runs/context-scale-compare/original-v1`.
The frozen source, executable, request and hashes are at
`.kanban/artifacts/287/freeze-v1`; the completed manifest covers329 original
files including preparation evidence and the compact report. The parent
README documents native regeneration. Exact-commit review additionally runs
all35 cases and replays in a fresh directory and compares complete model,
selection, metric, summary and report bytes against these originals.
