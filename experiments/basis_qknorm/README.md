# Sparsemax without query/key normalization

Does the basis GPTMini need QKNorm when its learned per-head gain is retained?
The user asked this on 2026-10-05 while the pinned mod-193 confirmation runs
were already training. This CPU study continues H2 from
[EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md), alongside those unchanged runs.

Step 4's ScaledDot intervention removed both query/key L2 normalization and
the learned head gain, and started with much smaller scores. It matched 12
of 22 softmax passes, against 17 for QKNorm starting at one. That comparison
does not isolate the normalization. These variants separate the learned
gain and its starting value while keeping the original recipes, splits,
seeds, ordinary sparsemax backward and XSA.

| Arm | Weights | Query/key normalization | Learned head gain | Starting gain |
| --- | --- | --- | --- | --- |
| `softmax` | Softmax | L2 | yes | sqrt(head width), the original control |
| `qknorm-one` | Sparsemax | L2 | yes | 1 |
| `scaleddot` | Sparsemax | none | no | 1 over ScaledDot |
| `learned-dot-one` | Sparsemax | none | yes | 1 over ScaledDot |
| `learned-dot-matched` | Sparsemax | none | yes | initialization estimate, 9.765625 small / about 3.45267 large |

`LearnedScaledDot` computes `exp(alpha) * (q @ k.T) / sqrt(head width)`.
Its gain starts at one or at
`1 / (width * init_std**2 * sqrt(head width))`. The second value follows the
unit-RMS/independent-Normal initialization estimate: ScaledDot scores have
standard deviation about `width * init_std**2`, while cosine scores at
QKNorm gain one have about `1 / sqrt(head width)`. It is selected before
measurements, without fitting to validation labels or pass/fail outcomes.
The estimates do not make every row's scores or support identical.

At gain one the new block preserves ScaledDot's initial forward and every
non-gain parameter gradient exactly. Unlike QKNorm, it preserves query/key
norms in the scores, so their radial changes can change attention.
`learned-dot-one` against `scaleddot` tests the added trainable gain;
the two learned-dot starts test its initialization. `learned-dot-matched`
against `qknorm-one` compares normalized and raw query/key vectors with
the same learned head gain and approximately comparable initial score
dispersion, which will be measured rather than assumed.

All 150 labels are declared: five arms on the 15 small and 15 large basis
runs. The screen trains the two new arms and fresh softmax controls on
all 45 small runs, then the 45 large runs, plus six large hard-recall
controls under QKNorm-one and ScaledDot. All 96 runs use the same lab
sources. Fresh controls matter because separate multi-threaded trajectories
can diverge despite the same initialization. Other declared QKNorm-one
and ScaledDot controls are not evidence until trained. A complete repair
must match every contemporary softmax pass and retain actual sparsity;
record support and visible exact zeros per layer and head. Dense-only
passes do not count as a sparse-attention repair.

`./make.py check experiments/basis_qknorm` checks the definitions;
`run` trains selected labels and resumes an existing run. The staged
controller will preserve the source hashes and collect pass updates,
final batch losses, head gains and all initial/final per-head sparsity.

## Preparation

All 150 descriptions check. Five focused tests pass (25.430 seconds):
gain validation, identical initialization and gain-one forward/gradients,
independent analytic raw-dot and gain derivatives, fused causal softmax
and full-graph compiled sparsemax logits and every parameter gradient.
On the actual basis shapes, six further ScaledDot/learned-gain-one pairs
have bit-identical logits and every non-gain gradient; their gain gradients
are nonzero. All 90 control descriptions equal their original descriptions
with the same observer. No non-gain initial parameter changes between arms.

The table averages first-layer supervised query rows across four heads and
three seeds on the same fixed 256 validation examples per seed. Their mean
visible prefix has 45.24 positions. These are measured initial supports,
not independent-normal estimates at a full context length of 64.

| Size | Scores | Score standard deviation | Support positions | Visible exact zeros |
| --- | --- | ---: | ---: | ---: |
| small | QKNorm=1 | 0.24430 | 8.5798 | 81.03% |
| small | ScaledDot / learned-dot-one | 0.02406 | 35.4657 | 21.60% |
| small | learned-dot-matched | 0.23494 | 8.6963 | 80.78% |
| large | QKNorm=1 | 0.17152 | 10.7562 | 76.22% |
| large | ScaledDot / learned-dot-one | 0.04804 | 24.8788 | 45.00% |
| large | learned-dot-matched | 0.16587 | 10.9980 | 75.69% |

The estimated gain gives similar initial dispersion and support; it does
not make the rows identical. All 30 initial observations preserve model
state, modes, RNG, inputs and absent gradients. Complete observations,
control checks, gain gradients, program source and lab hashes are in
[preparation.json](preparation.json).
The tracked observations retain every per-head statistic; the complete
per-query routes remain in the hashed raw snapshot under
`runs/preparation/observations.jsonl`.

The full CPU lab gate checks 171 tests in 1172.811 seconds: 165 pass and six
CUDA-only checks are skipped with CUDA hidden from this checkout while
mod193 uses the GPU. The new block's focused analytic, fused and compilation
checks run on CPU.

The completed step-4 hard-recall runs give another reason to distinguish
initial scale from later behavior. Averaged over all 24 heads, ScaledDot's
query score standard deviation starts at 0.0443--0.0447 and ends at
1.393--2.886. Its initial support of 25.96--26.13 positions narrows to
3.16--3.26, with 15.98--19.90% singleton queries. QKNorm-one ends with
6.69--7.45 support positions and no singleton queries; two seeds pass and
one fails. Those are observations of separate multi-threaded runs, not
a proof that score growth causes failure. All six means, gains, losses
and the source artifact hash are in
[prior_score_growth.json](prior_score_growth.json).

## Small-model results

All 45 small runs have completed with the pinned sources and fixed validation
rows. Fresh softmax passes nine of its fifteen cells. Each new arm matches
six of those nine; neither passes the small-model repair criterion. These
are complete small-model results, with the 51 large runs still in progress.

Pass updates below are ordered by seed 0, 1, 2. A dash means no 99% sequence
accuracy within the declared budget. Hard depth selects length 128;
the other cells select the ordinary validation split.

| Mode/task | Softmax pass updates | Learned dot, gain one | Learned dot, estimated starting gain |
| --- | --- | --- | --- |
| easy depth | 200 / 200 / 200 | 400 / 800 / 200 | 200 / 200 / 200 |
| easy recall | 1650 / 2450 / 1800 | — / — / — | — / 750 / 4050 |
| easy parity | 8200 / 5000 / 4200 | 15800 / 13000 / 12200 | 20000 / — / — |
| hard depth | — / — / — | — / — / — | — / — / — |
| hard recall | — / — / — | — / — / — | — / — / — |

Removing QKNorm while retaining a learned gain permits sparsemax to solve
all three small parity seeds at gain one, though later than softmax.
It does not fix recall: best sequence accuracies are 91.02%, 68.55% and
86.13%. The estimated starting gain helps recall seeds 1 and 2, while seed 0
reaches 98.44% without passing. It instead loses parity seeds 1 and 2,
whose best accuracies are 71.09% and 77.93%. A similar initial score
dispersion and support therefore do not ensure the same training outcome.
QKNorm is not required for these successful sparsemax runs; simply removing
it is not a repair across the required cells.

These successes retain sparsity. Every final non-BOS head has visible
exact zeros: 120 of 120 heads in each new arm. Gain-one parity ends with
4.17--4.56 support positions and 39.93--45.11% visible exact zeros, averaged
over the eight heads of each run. The two passing estimated-gain recall
runs end with 2.79 and 3.82 support positions, and 89.33% and 85.40% zeros.
No matched pass is dense-only. All initial/final per-head statistics,
fixed-row trajectories, pass comparisons, training-batch losses, head-gain
trajectories, reduction source and raw file hashes are retained in
[small_results.json](small_results.json).

The trainable gain alone does not prevent later raw-score growth. In the
three failed gain-one easy-recall runs, supervised-query score standard
deviation rises from about 0.024 to 5.59, 10.10 and 0.91; support ends at
1.52, 1.60 and 2.35 positions. These means cover all eight heads on fixed
queries. Sharp rows also occur in the two passing estimated-gain recall
runs: 69.02% and 47.71% of their supervised rows are singletons at their
respective stopping steps. Singleton frequency alone does not distinguish
success from failure in these runs.

The 15 fresh softmax controls reproduce all 661 canonical observations
from step 1 exactly, excluding timing fields. Final model, optimizer,
sampler and best-model checkpoint fields also match exactly; timings and
the newly recorded attention state are excluded. All 225 raw reduction
hashes and all 78 runtime-source hashes verify. The complete comparison,
checkpoint hashes for all 45 runs and its program are retained in
[small_control_repetition.json](small_control_repetition.json).

Before committing these findings, `./make.py test` checks 171 tests again
in 818.584 seconds: 165 pass and the same six CUDA-only checks are skipped.
The runtime and experiment sources remain unchanged during the large runs.

The first controller completed all 45 training runs before its source
validation failed: it included three command-line files that Lab excludes
from runtime provenance. The corrected check validates the 78 recorded
runtime files against their pinned hashes. It collected the existing small
runs without retraining and continued with the fresh large runs.

This study stays in a separate checkout so the running mod-193 series keeps
its pinned lab code and descriptions.
