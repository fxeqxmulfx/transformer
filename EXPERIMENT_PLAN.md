# Project experiment plan: why sparsemax attention fails, and a repair

Updated on 2026-10-05 UTC. **Done: steps 0 to 6 and the requested QKNorm follow-up.** The investigation cycle
was started on 2026-10-04 at the user's request. On 2026-10-04 the user asked for
this plan: find out why sparsemax attention fails, and try to repair it, on
the basis benchmark. It replaces the plan of 2026-10-03 for the mod-193
normalizer pair, which the user had paused. That plan's forward/backward
intervention and its Lean step continue here as steps 3 and 6, and mod 193
returns in step 5 to confirm a repair. The steps run in order; each ends in a
commit, and this file marks it done. Detailed historical evidence remains in
[the synthetic trainer handoff](experiments/archive/synthetic_trainers/HANDOFF.md).

## Objective

Find what makes sparsemax attention fail where softmax succeeds, on
[`experiments/basis`](experiments/basis/README.md); repair it; prove in Lean
what the experiments support. A repair succeeds when sparsemax attention with
it passes the basis wherever softmax passes, and still attends sparsely.

## What is known

Sparsemax replaces softmax by the Euclidean projection of a row's scores onto
the simplex (arXiv:1602.02068v2, §2), whose weights are exactly zero
outside a support.

- Modular division ([`mod193_stability`](experiments/mod193_stability/README.md)):
  GPTMini, whose attention normalizes queries and keys (QKNorm) and removes
  from each position's output its own value (XSA), trained on 25% of the
  37,056 equations `x / y mod 193`. Under sparsemax it fits every training
  equation, memorizes for 206,000 updates and never generalizes: held-out
  accuracy ends at 9,465/27,792 = 34.06% and peaks at 35.48%. Under softmax
  it ends at 100% on both splits.
- Associative recall ([`mqar_sparsemax`](experiments/mqar_sparsemax/README.md)):
  a transformer of one head, scaled dot-product scores and no XSA. Sparsemax
  reached 99% in 11 of 12 archived runs, at lengths 64 to 256 and four rates;
  softmax in 2.
- Tiny Shakespeare ([`shakespeare_amsgradw`](experiments/shakespeare_amsgradw/README.md)):
  GPTMini again; best test cross-entropy 1.638935 ± 0.016796 under sparsemax
  against 1.625375 ± 0.001731 under softmax, over three seeds.
- Lean, commit `0cf0524` (`Transformer.GPTMini.Sparsemax`): when one visible
  score of a causal row exceeds every other by more than 1, the sparsemax row
  is that position's basis vector on a neighborhood
  (`sparseWeights_eventually_eq_basis`), so its derivative, and that of any
  loss of it, is zero (`sparseWeights_hasFDerivAt_zero`,
  `rowLoss_hasFDerivAt_zero`); a wrong route can be such a point, of positive
  loss (`wrong_route_positive_stationary_point`). These concern one row: a
  possible obstruction, not the cause. Lean also certifies the accuracy of
  all 37,056 final CPU prediction records of the mod-193 pair
  ([report](experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-mechanism-and-final-certificate.md)).

So sparsemax does not fail everywhere: it wins on recall without QKNorm and
XSA, nearly ties on language with them, and fails to generalize on modular
division with them. The basis tells tasks apart: depth, recall and parity,
which a transformer solves by different means, on two sizes of GPTMini, each
calibrated under softmax ([Found](experiments/basis/README.md#found), at
bd63e50). Its 30 runs take 20 minutes on 28 cores, the small model's 15 about
nine.

## Hypotheses

Steps 1 to 3 decide each by its criterion.

**H1. Locked routes.** A sparsemax row passes no gradient to the scores
outside its support, and none at all once a gap of more than 1 isolates one
position (the Lean results above). A route the task needs that starts outside
the support is found only through other rows. *Predicted:* in a failing run
the supports of fixed rows stop changing while the run is below target, and
on wrong recall answers the position holding the answer lies outside every
head's support. *Refuted if* the supports keep changing until the run ends,
or the answer's position is in a support on the wrong answers.

**H2. The starting scale.** QKNorm multiplies the cosine of a query and a key
by a learned e^α that starts at √(head width): 4 on the small model, about
5.7 on the large. A row's scores then spread about 1, whatever the
initialization, and sparsemax keeps few positions: were a row's 64 scores
independent standard normals, it would keep 3.3 on average, and one alone in
6% of rows. Scaled dot-product scores under the initialization at 0.02 spread
about 0.02² × width, 0.026 on the small model and 0.051 on the large; as
independent normals, a row of 64 keeps about 41 and 27. *Predicted:* at
update 0, QKNorm rows keep a few positions, and a smaller starting e^α or
scaled dot-product scores remove sparsemax's failures. *Refuted if* QKNorm
rows start wide, or neither change helps where sparsemax fails.

**H3. Counting needs ties.** Depth is definable by counting
(`Transformer.CRASP.definablePos_altPlusNeutral`), and parity of up to 16
bits can be read off the count of ones; a count is an average over
positions. A sparsemax row that keeps all T visible positions weighs each by
1/T plus its score's deviation from the row's mean, so averaging within a
relative error ε needs every score within ε/T of the mean; softmax needs
about ε, whatever T. At length 64 sparsemax needs ties 64 times finer.
*Predicted:* sparsemax fails or slows on depth and parity more than on
recall; where softmax passes, some heads average many positions nearly
uniformly, while under sparsemax the same layers keep few positions, or their
e^α falls far below its start. *Refuted if* sparsemax passes depth and parity
as softmax does.

**H4. XSA erases self-routes.** XSA subtracts from a head's output its
component along the position's own value. A sparsemax row whose support is
the position itself outputs exactly that value, so the head outputs zero
there, and with a gap above 1 the row passes no gradient to its query, keys
or values. Softmax gives one position all of a row's weight only in a row of
one position. *Predicted:* failing sparsemax runs have many such rows, and
dropping XSA helps sparsemax more than softmax. *Refuted if* such rows are
rare, or dropping XSA leaves sparsemax's outcome unchanged.

**H5. The recipe.** The basis recipes were set under softmax. A sparsemax
failure that a neighboring rate of `RATES` removes is the recipe's, and the
mechanism study leaves it.

Step 6 corrects H4's implementation-level wording: exact erasure requires
the self-value norm at least epsilon, and the locally-zero result uses a
strict norm bound above epsilon. A clipped nonzero value can survive a
strict self route; a Lean counterexample below makes that correction
explicit. It does not change the empirical rarity conclusion of step 2.

## 0. Run sparsemax on the basis

**Done on 2026-10-04.** [basis_sparsemax](experiments/basis_sparsemax/README.md)
defines the 60 benchmark runs and six short timing runs, with the existing
diagnostics recording training batch loss and head scales. Sparsemax's
projection, backward, model logits and every parameter gradient agree with
eager evaluation within rounding under full-graph compilation. Diagnostics
preserves one-thread compiled observation metrics and the complete training
state under both weights, and a one-thread sparsemax run resumes onto its
own records. On large depth and recall at two and four threads, measurements
leave the actual training state and their inputs exactly unchanged; separate
large multi-threaded runs can round differently even without diagnostics,
so exact trajectory repetition is not guaranteed there. The measured
sparsemax/softmax update-time ratios are
1.681 (depth), 1.708 (recall) and 1.238 (parity), below the factor-of-two
ceiling; no implementation repair was needed. The short softmax controls
repeat every common archived basis validation record exactly; step 1 checks
the full trajectories. The small model's 15 runs under each weight come
first, then the large model's, then neighboring rates for failures.

Write `experiments/basis_sparsemax` (`experiment.py`, `README.md` and a row
in `experiments/README.md`). Its runs are the basis's 30 benchmark runs under
each of the two weights, labeled
`<weights>-<mode>-<model>-<task>-seed<seed>`: `softmax` repeats the basis,
and `sparsemax` is `basis(substitute(model, Softmax, Sparsemax()), mode,
seed)`, both on the basis's thread counts. Both record `Diagnostics` at
every observation: the training batch's loss, which tells a failure to fit
from mod 193's failure to generalize, and the e^α of every head. First
check:

- that `Sparsemax`, a sort, a cumulative sum and a custom backward, compiles
  under `Compiled`, and that compiled and eager sparsemax give the same
  logits and gradients to rounding (a test in `python/tests`);
- that diagnostics leave a compiled run's records unchanged, and that the
  softmax runs repeat bd63e50's; if not, the two arms compare at this commit
  alone;
- the time of an update of each small task under both weights. If sparsemax
  does not compile, or costs more than twice softmax, fix that in the lab
  first.

## 1. Find where sparsemax fails

**Done on 2026-10-04.** All 60 original runs and 12 required adjacent-rate
checks are complete. The 15 small softmax controls repeat all 661 archived
non-timing observations and their model, optimizer and sampler checkpoints
exactly. Fourteen of the 15 large controls' full histories differ, with the
same pass/fail set, so the large comparison uses the current arms as step 0
allows. Separate large runs without diagnostics also drift; direct tests
verify that measurements leave the actual training state unchanged.

H5 removes both small parity failures at 1e-4, small easy recall seed 2 at
3e-4, and large easy recall seed 2 at either 3e-4 or 3e-3. There is no
persistent matched-seed small-model failure. On the large model, sparsemax
passes every depth and parity; hard depth passes earlier than softmax,
parity later. Only `(hard, large, recall)` remains a mechanism target:
softmax passes seeds 1 and 2 while sparsemax's best accuracies at the recipe
are 98.63% and 97.66%, and neither adjacent rate passes either seed. Step 2
measures this cell from all three seeds under both weights. The small-model
target ablation is empty. Tables, final training batch losses, head scales
and source hashes are in [basis_sparsemax](experiments/basis_sparsemax/README.md).

Train the small model's 15 runs under both weights, then the large model's
15. Record, as the basis's Found does, the update of each pass or the best
selection accuracy, and the last training loss. Where sparsemax fails from a
seed that softmax passes from, train that seed at the two rates of `RATES`
beside the recipe's (H5).

Predicted before the runs: by H3, depth and parity fail or slow; by
`mqar_sparsemax`, recall passes at least as often as under softmax, unless
H2 or H4 interferes.

The cells (mode, model, task) where sparsemax fails at every rate tried and
softmax passes are the targets of steps 2 to 4. If there are none, the basis
does not show the failure: record that, move step 2's measurement to mod 193,
and keep the basis to check that a repair costs nothing there.

## 2. Measure the mechanism

**Done on 2026-10-04 UTC.** The target is large hard recall.
`AttentionDiagnostics` retains fixed validation examples and previous
causal support masks across checkpoints. Focused tests check hand-made
statistics, latest-write query routes, CPU/CUDA live-state preservation,
compiler guards, one-thread compiled noninterference and resumed turnover.
The 24 initial QKNorm/scaled-dot measurements cover both sizes and weights
from three seeds; the six target runs retain every canonical measurement.
All 155 lab tests pass. The 24 initial runs are complete: first-layer
query support averages 3.43/3.34 positions under QKNorm on the small/large
models, against 35.47/24.88 under ScaledDot, with otherwise identical
initial parameters. This supports H2's initialization prediction, not
its causal or repair conclusion. The six target repeats are complete:
softmax passes all seeds, sparsemax none. The measured sparsemax trajectories
also fit poorly, unlike step 1's fitted near-passes; separate threaded runs
diverge despite identical initial metrics. Direct real-shape, four-thread
checks preserve the entire live training state around attention observation.
H1 is refuted by continued turnover through the budget and by answer
membership on 90--97% of final wrong queries. H4 is refuted by rare self-only
routes, excluding BOS (at most 1.87% averaged over layer/head pairs at any
observation). H3 has no persistent depth/parity failure after H5; parity
can still take longer. H2's causal prediction awaits the score/scale
interventions. There is no small-model target for the small factorial
ablation. Detailed trajectories, per-head statistics, qualifications and
source hashes are in [basis_sparsemax](experiments/basis_sparsemax/README.md).

Add to `Diagnostics` a measurement of attention (domain, infrastructure,
test). At every observation, an uncompiled forward of 256 fixed validation
rows, which draws no random numbers, records per layer and head:

- the size of the support, in positions and as a share of the visible ones;
- the share of rows that keep one position, and of rows that keep only
  their own;
- the share of rows with a gap above 1, which are locally constant;
- the turnover: the share of (row, position) pairs whose membership in the
  support changed since the previous observation;
- under softmax, the same for the sparsemax of its scores, and the
  normalized entropy of its weights;
- on recall, at each query, the weight on the position holding the answer,
  the largest over the layer's heads, and whether that position is in any
  support.

Tests check each statistic on hand-made rows, and that a run with the
measurement records the same losses as one without. Then:

1. At update 0, measure both models under QKNorm and under scaled
   dot-product scores (H2's estimate).
2. Train the target cells under both weights from seeds 0 to 2 with the
   measurement.
3. On the small model's target cells, train {QKNorm, ScaledDot} ×
   {XSA, none} × {softmax, sparsemax} from seeds 0 to 2 (H2, H4).

Decide each hypothesis by its criterion, and write what was found in the
experiment's README.

## 3. Separate forward routing from backward sensitivity

**Done on 2026-10-05 UTC.** `SurrogateWeights` and six mixed target
labels are implemented. Identical pairs use the ordinary block; mixed pairs
retain exact forward probabilities and ask autograd for the other map's
score Jacobian. Tests cover bit-identical diagonal logits and gradients,
independent `torch.func.vjp` references, inactive scores and compilation.
All six focused tests pass, as do all 161 lab tests. The previous 156 run
descriptions remain identical; the six mixed targets differ only in the
explicit forward/backward weights block. All six mixed targets completed
4,800 updates. Sparse forward with softmax backward fails from every seed
(best sequence accuracy 0%); softmax forward with sparse backward learns
much more but also misses the target (96.88%, 96.29%, 98.63%). Both ordinary
and surrogate backward fail with sparse routing, so the forward map remains
binding in this intervention; this does not confirm H3's counting or H4's
self-erasure mechanism, already refuted by their criteria. The inverse
surrogate's failures also leave a role for backward sensitivity or an
unsuitable surrogate. No gradient-only repair is found. The conditional
inactive-score-only case is not triggered: softmax backward does not rescue
sparse forward. H2's initialization prediction remains the next justified
repair test. Results, per-head statistics and source hashes are in
[basis_sparsemax](experiments/basis_sparsemax/README.md).

The intervention planned for mod 193, moved onto the basis:

| Forward weights | Backward score map | Purpose |
| --- | --- | --- |
| Softmax | Softmax Jacobian | Control |
| Sparsemax | Sparsemax support Jacobian | Candidate |
| Sparsemax | Softmax Jacobian | Keep sparse routing; change score gradients |
| Softmax | Sparsemax support Jacobian | Keep dense routing; change score gradients |

A new `Weights` block computes one normalizer forward and the other's
Jacobian backward, at the same scores: a declared surrogate gradient, not
the derivative of the loss. CPU tests: with both normalizers the same, it
equals the plain block bit for bit, logits and every gradient; mixed, its
weights are those of its forward normalizer, and its backward equals an
independent vector-Jacobian product of its backward normalizer
(`torch.func.vjp`), inactive entries included. Train the two mixed cases on
the target cells from seeds 0 to 2; steps 1 and 2 hold the diagonal ones.

If sparse forward with the softmax backward passes where sparsemax fails,
and dense forward with the sparsemax backward fails where softmax passes,
the backward map is implicated (H1). The swap changes both the support and
the size of score gradients; a focused case, the sparsemax backward with
softmax's gradients for the inactive scores alone, attributes the effect to
their zeros. If sparse forward fails under either
backward, the forward is implicated (H3, H4). Other patterns indicate an
interaction or an unsuitable surrogate.

## 4. Repair

**Done on 2026-10-05 UTC.** H2 justifies the two score interventions:
QKNorm initialized at scale one and ScaledDot, each with the ordinary
sparsemax backward and unchanged basis recipes. `QKNorm.initial_scale`
defaults to the original square-root-of-head-width initialization and keeps
the scale learned. Five focused tests pass: validation, old-description
compatibility, original rounding, identical non-score parameters and score
gradients with a learned scale. Sixty repair labels cover both candidates
on the small model's 15 runs and then the large model's 15, with fixed-row
attention measurements. All 166 lab tests pass. All 30 small-model runs are
complete: each candidate matches 6 of the 9 softmax passes, neither passes
the six hard runs, and every passing run retains exact visible zeros in
all eight heads after BOS. QKNorm-one fails parity seeds 0 and 2 and easy
recall seed 2; ScaledDot fails every easy recall seed. Thus neither is a
complete basis repair at these recipes. All 30 large-model runs are now
complete too: QKNorm-one matches 11 of 13 required large softmax passes,
ScaledDot 6 of 13. QKNorm-one passes the original persistent hard-recall
seed 2 at update 3,550 but not seed 1, and fails large parity seed 2.
ScaledDot fails every large parity and hard-recall seed. Every final head
of all 60 runs retains exact visible zeros after BOS. Across both sizes,
QKNorm-one matches 17 of 22 required softmax passes and has 19 total passes;
ScaledDot matches 12 of 22 and has 13 total passes. Neither is a complete
repair. QKNorm-one is the best attempted intervention for step 5, ranked
by sparse matched softmax passes, then persistent target passes, then all
sparse passes. H2's narrow-start prediction holds and changing the start
helps one persistent seed, but its failure-removal prediction is not met.
The complete source/description/cadence and 300 raw-file hash checks pass,
as do all 166 lab tests (938.589 seconds).
The matched-seed table, final losses, full per-head supports and source
hashes are in [basis_sparsemax](experiments/basis_sparsemax/README.md).
The failed surrogate is not screened as a repair; H4's rarity criterion and
H3's lack of persistent failures do not justify no-XSA or entmax trials here.

Try the candidates whose hypothesis survived, each a block or a `swap` or
`substitute` variant:

- (H2) QKNorm starting at e^α = 1, a new field of `QKNorm`; scaled
  dot-product scores;
- (H4) no XSA;
- (H1, H3) α-entmax at α = 1.5 (arXiv:1905.05702v2, §3.2 to 3.4): still
  exact zeros, but its support reaches 2 below the top score, not 1, and
  averaging within ε needs scores within about ε/√T of the mean, not ε/T;
  or α = 1 + sigmoid(a) learned per head, in (1, 2) (arXiv:1909.00015v2,
  §3 and 4), so that a head that counts can stay dense; there the heads
  first grew denser, and some sparser later (§5.1);
- (H1) sparse forward with the softmax backward, from step 3.

Screen each on the small model's 15 runs: in the easy mode a failure counts
against it, in the hard mode a pass counts for it. Then the large model's 15.
A repair succeeds if it passes every cell from every seed that softmax
passes from. Record its updates to pass beside softmax's, seed by seed, and
how sparse its attention ends: the mean support and the share of exact
zeros, per layer and head. A repair that passes only by making its rows
dense explains the failure; it does not repair sparse attention.

### QKNorm follow-up requested during step 5

On 2026-10-05 the user asked whether sparsemax needs QKNorm and requested
continued investigation. The completed ScaledDot screen removed both L2
normalization and the learned gain, and changed the initial score scale.
A supplementary CPU study, [basis_qknorm](experiments/basis_qknorm/README.md),
therefore removes L2 normalization while keeping a learned head gain.
It compares gain one against the initialization-only gain estimate
`1 / (width * init_std^2 * sqrt(head width))`, with fresh QKNorm-one,
ScaledDot and softmax controls in the large hard-recall target cell.
The estimated scale is fixed before measurements; actual initial score
dispersion and support must be measured, not assumed equal. All 150
descriptions check and five focused tests pass. Thirty initial models
preserve every non-gain parameter and the observer's live state; all 90
control descriptions equal their originals with the same observer. Six
actual-shape pairs preserve ScaledDot's initial logits and every non-gain
gradient exactly at gain one, while the gain receives nonzero gradients.
On first-layer queries, the estimated-gain raw-dot rows keep 8.70/11.00
positions on the small/large model, against QKNorm-one's 8.58/10.76,
with similar measured score dispersion. These are initialization checks,
not evidence that removing normalization repairs training. Their full
observations and source hashes are retained in
[preparation.json](experiments/basis_qknorm/preparation.json).

The full CPU lab gate checks 171 tests in 1172.811 seconds: 165 pass and six
CUDA-only checks are skipped while the GPU runs mod193. This follow-up starts
with independent derivative/compilation and initialization checks and
that full lab gate, then trains both new arms
and fresh softmax controls on the small model's 15 runs before the large
model's 15, plus the six QKNorm-one/ScaledDot target controls. All 96
runs use the same lab sources. It collects the same pass, batch-loss
and per-head sparsity evidence as step 4; untrained declared controls
support no conclusion. Preparation and runs
use a separate checkout until the pinned step-5 series ends. The existing
mod-193 jobs continue, and steps 5 and 6 remain required in their original
order. This supplements the completed screen rather than changing its
criteria or calling either old intervention a repair.

All 45 small follow-up runs are complete. Fresh softmax passes nine cells;
both learned raw-dot starts match six, with visible exact zeros in all
120 final non-BOS heads per arm. Gain one passes all three parity seeds
but loses all three easy-recall seeds. The estimated starting gain passes
two recall seeds but loses two parity seeds; recall seed 0 reaches 98.44%
without passing. Neither is a complete small-model repair. All 51 large
runs have now finished with unchanged sources and budgets. The full small reduction,
per-head statistics, gains, pass comparisons and raw file hashes are in
[small_results.json](experiments/basis_qknorm/small_results.json).
All 15 fresh small softmax controls repeat step 1's 661 canonical
observations and every non-timing model/optimizer/sampler/best checkpoint
field exactly, excluding the new attention-observer state. All 225 raw
reduction hashes verify; the independent comparison is retained in
[small_control_repetition.json](experiments/basis_qknorm/small_control_repetition.json).
The findings' required `./make.py test` gate checks 171 tests in 818.584
seconds: 165 pass and the same six CUDA-only checks are skipped. Runtime
and experiment sources remained unchanged during the large runs.

The complete 96-run screen is done. Each new arm matches 6 of 14 large
softmax passes and 12 of 23 across both sizes. All 360 final large-model
non-BOS heads per arm retain visible exact zeros. Neither new arm passes
large hard recall from any seed; current softmax passes 3/3, QKNorm-one
2/3 and ScaledDot 1/3. The earlier step-4 global counts use different
multi-threaded trajectories; only the three current target seeds compare
all five arms. There is no better candidate warranting another mod-193
transfer. [results.json](experiments/basis_qknorm/results.json) verifies
480 raw hashes and all runtime/source/description/fixed-row pins and links
the detailed large and small reductions. Before the complete-findings
commit, all 171 tests are checked again (808.529 seconds): 165 pass and six
CUDA-only tests are skipped. Steps 5 and 6 retain their original scope.
The source-validation mismatch after the completed small training was
corrected to follow Lab's 78-file runtime provenance, excluding its three
command-line files; all completed small sources match and none is retrained.
After all four pinned mod193 runs finished, the complete study was merged
into the main workspace. The full CPU/CUDA lab gate then passed all 171
tests without skips (911.172 seconds). Raw study runs are retained in the
main workspace's ignored `experiments/basis_qknorm/runs/` directory.

## 5. Confirm on mod 193

**Done on 2026-10-05 UTC.** Step 4 selects QKNorm-one as the best
partial attempt; neither screened intervention passes the complete basis
repair criterion. `mod193_stability` declares `repair-qknorm-one` beside
fresh `base` and `sparsemax` controls, with the same ordinary backward,
recipes and seeds, and fixed attention on 256 held-out examples at every
observation. A separate seed-1 label changes both model and data seeds and
is trained only if the first attempt confirms. The seven run descriptions
and CUDA live-state preservation are checked before the full training.
Confirmation uses 20 consecutive canonical evaluations with both
accuracies at least 99% and ends at 100% on both splits; neighbor probes
are excluded. Memorization and the 201 final-window checks remain separate
report columns, not extra confirmation requirements. All seven
descriptions check and the three actual-shape CUDA observer checks pass:
parameters, buffers, gradients, optimizer, sampler, CPU/CUDA RNG, modes,
hooks and inputs remain exactly unchanged at updates 0, 10 and 20. These
temporary short runs are setup checks, not generalization evidence.
Before/after descriptions and the check's source/output are retained in
[preparation.json](experiments/mod193_stability/preparation.json).
The full lab check passes all 166 tests (936.095 seconds) before the setup
commits and full runs.

All four scheduled lab runs completed 300,000 updates at the pinned sources.
The fresh softmax and ordinary sparsemax controls repeat all 1,201 archived
canonical observations exactly, excluding timing. Softmax ends at 100% on
both splits, with two final-window failures; ordinary sparsemax ends at
100%/34.06%, never confirms and fails all 201 final-window checks. QKNorm-one
confirms at update 121,000, ends at 100%/100%, and passes all 201 final-window
checks (worst held-out 99.62%). Its fresh model/data-seed-1 repeat confirms
at 186,250 and ends at 100%/100%; two final-window held-out checks fall below
99% (worst 98.93%). Both meet the original step-5 confirmation criterion,
but strict final-window persistence holds only for the first seed. Their
eight final heads retain visible exact zeros, averaging 36.52% and 46.56%
of visible non-BOS pairs respectively. This confirms the selected partial
intervention on mod193 from two seeds; it does not repair the complete basis
or establish general success or persistence on unseen seeds.

[confirmation_results.json](experiments/mod193_stability/confirmation_results.json)
retains all canonical/attention curves, initial/final per-head statistics,
component losses, gains, 32 raw file hashes and embedded reduction sources.
Each run has 1,201 canonical evaluations, 2,400 neighbor probes, 3,601
fixed-256-example attention records and 300,000 verified finite gradient
updates. [control_repetition.json](experiments/mod193_stability/control_repetition.json)
independently verifies both archived histories, all four canonical reductions,
matched descriptions, fresh seeds and all 78 runtime source hashes.
The complete-findings `./make.py test` gate passes all 166 tests, including
CUDA checks, in 886.968 seconds.

Train the best repair as a label of `experiments/mod193_stability` beside
`sparsemax`, changing nothing else, with step 2's measurement, for its
300,000 updates; the archived pair took 2.6 and 3.5 hours on the GPU. The
original table held archived runs of the historical trainers, so train
`base` and `sparsemax` afresh too. Record the archived
table's columns: memorized, confirmed, failures in the last 50,000 updates
and the worst held-out accuracy there, final accuracies. The repair is
confirmed if it generalizes as `base` does: confirmed for 20 evaluations,
ending at 100% on both splits. One seed cannot establish general
applicability: repeat a confirmed repair from fresh model and data seeds
before claiming it.

A preliminary CPU inspection of the archived pair's final checkpoints has
finished; its observations are under
`experiments/archive/synthetic_trainers/runs/adamw_stability_20261002/bootstrap/sparsemax-generalization-inspection-20261003/`
(not tracked). The inspector that wrote them is in commit `441f47b`, at
`experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/inspect_sparsemax_generalization.py`,
and left the tree with the other archive scripts; its SHA256 is the
`program_sha256` the observations record. It imports the removed synthetic
trainers and reads `experiments/runs/`, so it does not run as it is. Review
the source, independently check the observations, and archive them before
treating them as final.

This review is complete. The exact source from `441f47b`, original
observations, independent verification source/output and input hashes are
archived in [sparsemax-final-review](experiments/archive/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-final-review/README.md).
The independent program rebuilt every equation and strictly loaded the
unchanged final parameters in the lab at `d03f594`; all four fixed-weight
forward swaps and the selected local derivative probes match exactly,
with no rounding differences or optimizer updates. Both raw checkpoint
hashes and every embedded lab source hash match their pinned inputs.
This establishes the fixed-checkpoint observations, not a training cause
or the result of the new 300,000-update lab trials.

## 6. Prove the supported statements in Lean

**Done on 2026-10-05 UTC.** Five modules beside
`Transformer.GPTMini.Sparsemax` add 25 proved theorems, with examples for
every explicit hypothesis and no new `sorry`:

- [ClosedForm.lean](src/Transformer/GPTMini/Sparsemax/ClosedForm.lean)
  proves a normalized clipped candidate minimizes the existing variational
  objective, proves a threshold exists, and identifies the actual row with it.
- [SupportWindow.lean](src/Transformer/GPTMini/Sparsemax/SupportWindow.lean)
  proves H2's strict unit support window and the `exp(-alpha)` window for
  QKNorm's normalized similarities, including epsilon clipping.
- [Uniform.lean](src/Transformer/GPTMini/Sparsemax/Uniform.lean) proves H3's
  full-support weight formula, its exact relative-uniformity/score-tie
  equivalence at `epsilon/N`, and the resulting attention average.
- [SelfRoute.lean](src/Transformer/GPTMini/Sparsemax/SelfRoute.lean) proves
  H4's corrected locally-zero output and score/value/parameter derivative
  with strict score and epsilon-norm bounds; continuity of parameter maps
  is an explicit premise.
- [Clipping.lean](src/Transformer/GPTMini/Sparsemax/Clipping.lean) proves
  the residual below epsilon and a strict self-route counterexample:
  self-value norm `1/2`, epsilon `1`, output norm `3/8`.

Sources were reread beside the statements: arXiv:1602.02068v2, §2.2,
Proposition 1, and arXiv:2603.09078v1, §2, equation `xsa` and Algorithm 1,
plus the epsilon-clipped lab code at `73f8a0b`. All modules are reachable
from `src/Transformer.lean`. The full `lake build`, `./make.py audit`,
`./make.py index` and `./make.py forbidden` checks pass: 157 existing
`sorry`, zero resting on them, zero extra axioms, zero vacuous statements
and zero placeholders. The generated index has 1,836 modules and 7,556
theorems. New modules have no warnings. H1's intervention did not isolate
a whole-training cause, so no such theorem is claimed.

Prove only what the experiments support, under AGENTS.md: statements checked
against their sources, an example for every theorem with hypotheses, no axiom
or new `sorry`; then `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden`. Distinguish a real-arithmetic theorem, a
floating-point measurement and a certificate of supplied data; a theorem
about one row says nothing about a whole training run. Candidates, beside
`Transformer.GPTMini.Sparsemax`:

- (H2) every position of a sparsemax row's support scores within 1 of the
  row's maximum, by the closed form of arXiv:1602.02068v2, §2.2, so under
  QKNorm its cosine is within e^(−α) of the largest;
- (H3) with a full support the weights are 1/T plus each score's deviation
  from the mean, so a near-uniform average needs ties T times finer than
  under softmax;
- (H4) a row whose support is its own position, by a gap above 1, has an XSA
  output of zero on a neighborhood, hence zero derivative in its query, keys
  and values (from `sparseWeights_eventually_eq_basis`);
- (H1) what the intervention attributes.

## Cycle outcome

The prescribed cycle and the QKNorm follow-up are complete. The selected
initial-scale intervention confirms sparse Mod193 generalization on two
fresh model/data seeds, but the complete-basis repair criterion remains
unmet: QKNorm-one matched 17/22 step-4 softmax passes; each learned raw-dot
start matched 12/23 contemporary follow-up passes. Final-window persistence
holds for the first Mod193 repair seed and fails twice for the repeat.
These are qualified experimental conclusions; the row-level Lean results
do not prove a global training cause. The completion manifest is retained in
[cycle_audit.json](experiments/basis_sparsemax/cycle_audit.json).

## Mathematical follow-up: supervised score loss

**Done on 2026-10-05 UTC.** The loss in arXiv:1602.02068v2, §3.2,
`sparsemax_loss`, depends directly on the target score as well as on the
projection's quadratic potential. Its score gradient is `sparsemax(z) - y`
for a supplied routing target `y`. This objective can supply a corrective
signal on a wrong singleton even though every outer function of that
locally constant probability row has zero score derivative.

[RoutingLoss.lean](src/Transformer/GPTMini/Sparsemax/RoutingLoss.lean)
uses the existing causal variational projection. It proves the potential
identity, feasible-route lower bound, visible-target nonnegativity, and
full score derivative on strict singleton regions. Adding the score loss
to any outer probability loss supplies the same derivative multiplied by
its weight. On the previous counterexample `(2, 0)` with target slot 1,
the target-coordinate derivative is `-1`; a score-gradient step of size
`3/2` produces `(1/2, 3/2)`, the exact correct route and zero score loss.
No new `sorry` is used.

This repairs the supervised row counterexample, not the complete basis.
Attention routes are latent: the experiment's output labels do not directly
specify the desired position in every head. An auxiliary routing loss would
need supplied targets, a declared teacher or another source of score
supervision. No such training intervention has been run, and the previous
benchmark findings and completed-cycle manifest remain unchanged.

## Mathematical follow-up: constraints without routing targets

**Done on 2026-10-05 UTC.** Five modules add 34 proved theorems about
the existing causal variational projection, with inhabited examples for
every explicit hypothesis and no new `sorry`. These are derived results
from arXiv:1602.02068v2, §2.2 and §2.5, and the score/value code at
`73f8a0b`; they are not results of a new training intervention.

| Module | Constraint or result | Scope |
| --- | --- | --- |
| [NonSaturation.lean](src/Transformer/GPTMini/Sparsemax/NonSaturation.lean) | A singleton occurs exactly at a visible winner's unit gap; a top-two gap below one ensures two positive weights. | Exact zeros at other visible slots remain possible: `(2/5, 2/5, -2/5)` projects to `(1/2, 1/2, 0)`. |
| [ActiveDirection.lean](src/Transformer/GPTMini/Sparsemax/ActiveDirection.lean) | Transferring scores between two active slots transfers the same probability mass for small steps. | The score-to-weight map cannot have a zero full derivative; this does not assert existence of a full derivative at support boundaries. |
| [BoundedGain.lean](src/Transformer/GPTMini/Sparsemax/BoundedGain.lean) | QKNorm's persistent gain `g < 1/2` suffices; the new law `g = c/(1 + exp(-a))`, `0 < c < 1/2`, enforces it and has a positive derivative at every finite `a`. | Epsilon is nonnegative and at least two slots are visible. The guarantee concerns raw score derivatives, not query/key/gain parameter derivatives. |
| [OuterSensitivity.lean](src/Transformer/GPTMini/Sparsemax/OuterSensitivity.lean) | A task derivative distinguishes an active pair exactly when `output_gradient(value_j - value_k)` is nonzero. | This uses the ordinary task loss, with frozen values. Equal values cancel; distinct values can also lie in the gradient's kernel. |
| [ValuePlateau.lean](src/Transformer/GPTMini/Sparsemax/ValuePlateau.lean) | With values `(0, 0, 1)` and scalar output target `1/2`, squared error stays `1/4` on an open score neighborhood of the sparse bounded example. | Both this positive stationary point and a zero-error alternative obey `abs(score) <= 2/5`. The projection itself still has a nonzero active direction. |

The gain restriction removes singleton saturation without a teacher or
attention-position labels. It does not remove every flat outer loss or
prove convergence on Shakespeare or the basis. The value condition is a
conditional guarantee, not a rule proved enforceable for every task input.
No Python implementation or training run of the bounded gain has been
added. The original cycle remains complete with its qualified negative
complete-basis outcome and its pinned manifest.

Validation: the full `lake build`, `./make.py audit`, `./make.py index`
and `./make.py forbidden` pass. The generated index has 1,842 modules,
7,599 theorems and the same 157 existing `sorry`; there are zero results
resting on them, zero extra axioms, zero vacuous statements and zero
placeholders. The five new Lean modules have no warnings.
The required `./make.py test` also passes all 171 Python tests.

## Pure sparse follow-up: active value span

**Done on 2026-10-05 UTC.** Two further modules add ten proved theorems
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and the frozen linear value readout at `73f8a0b`.

[ValueSpan.lean](src/Transformer/GPTMini/Sparsemax/ValueSpan.lean)
proves a structural sufficient condition independent of the task target:
the differences of active values span the output space. Their span then
cannot lie in the kernel of a nonzero output derivative, so some active
pair supplies a nonzero raw score direction for the ordinary task loss.
Two distinct active scalars satisfy the condition. The earlier collapsed
values `(0, 0, 1)` on the actual two-active row are proved to violate it.

[SeparatedValues.lean](src/Transformer/GPTMini/Sparsemax/SeparatedValues.lean)
computes the actual derivative for ordinary scalar squared output error:
`2 * (output - target) * (value_j - value_k)`. A wrong scalar output and
distinct active values therefore exclude a zero score derivative. The
constructed values `(-4, 4, 1)` retain the previous initial output, target
`1/2`, scores `(2/5, 2/5, -2/5)` and weights `(1/2, 1/2, 0)`. The
active-pair derivative is `-8`; transferring scores by `1/16` gives
weights `(7/16, 9/16, 0)` and zero output error. Both score endpoints
stay strictly below the same absolute cap `12/25 < 1/2`.

This repairs the constructed output plateau after changing the values to
satisfy the separation premise. It does not repair the collapsed values
through a locally unchanged support, supply an enforcement rule for the
span in every head/input, prove query/key parameter accessibility of the
score direction, or establish whole-model convergence. The ordinary
output target is retained; no attention-position labels are introduced.
These additions are Lean results only. The Python code and previous
experimental outcomes are unchanged.

Validation: the full `lake build`, `./make.py audit`, `./make.py index`
and `./make.py forbidden` pass. The new modules have no warnings. The
index now has 1,844 modules and 7,609 theorems, with the same 157 existing
`sorry`, zero results resting on them, zero extra axioms, zero vacuous
statements and zero placeholders. Python is unchanged since the passing
171-test run recorded above.

## Pure sparse follow-up: enforcing the value span

**Done on 2026-10-05 UTC.** Three modules add sixteen proved theorems,
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and the frozen linear value readout at `73f8a0b`.

[BoundedCoordinates.lean](src/Transformer/GPTMini/Sparsemax/BoundedCoordinates.lean)
proves strict bounds, a positive derivative and a differentiable inverse
for independent score coordinates `c * (exp(a) - 1) / (exp(a) + 1)`.
These modify the scores; the weights remain the actual causal variational
sparsemax projection.

[AnchoredScores.lean](src/Transformer/GPTMini/Sparsemax/AnchoredScores.lean)
prepends `A` bounded anchor scores to ordinary scores below `-c`.
If `c > 0`, `2 * A * c < 1` and the anchors are visible, every anchor
has positive weight at every finite parameter assignment. Ordinary slots
can remain exactly inactive, including the checked weights `(1/2, 1/2, 0)`.

[AnchoredValues.lean](src/Transformer/GPTMini/Sparsemax/AnchoredValues.lean)
uses `d + 1` anchors in a `d`-dimensional output: a trainable common base
and that base plus each fixed basis direction multiplied by a positive
learned exponential scale. Ordinary values are arbitrary. All active value
differences automatically span the output, for arbitrary changes of the
base, scales, score parameters and ordinary values. Thus preservation
under finite parameter updates follows from the parameterization; the
span is no longer a hypothesis about learned values. A nonzero ordinary
output derivative always distinguishes two anchors and cannot disappear
in the raw score path.

This is an explicit anchored architecture modification, requiring `d + 1`
visible prefix slots and independent anchor scores. It does not prove that
the existing query/key architecture satisfies the same condition, or that
a zero output derivative or whole-model nonconvexity is repaired. No route
target, teacher, dense weight branch, Python implementation or new training
run is introduced. The completed benchmark cycle remains unchanged.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden`; the new modules have no warnings. Python is unchanged
since the passing 171-test run recorded above.

## Pure sparse follow-up: reaching trainable parameters

**Done on 2026-10-05 UTC.** Four modules and a readout evaluation lemma
add twenty-three proved theorems, with no new `sorry`, derived from
arXiv:1602.02068v2, §2.2 and §2.5, and the ordinary value sum at `73f8a0b`.

[AnchorTransfer.lean](src/Transformer/GPTMini/Sparsemax/AnchorTransfer.lean)
lifts an exact active-anchor score transfer through the bounded coordinate
inverse. The curve starts at the actual parameters, is differentiable,
and realizes the same raw score transfer on a neighborhood. Ordinary
scores and values stay frozen along this particular curve.

[TrainableAnchors.lean](src/Transformer/GPTMini/Sparsemax/TrainableAnchors.lean)
proves that the ordinary task derivative along this parameter curve is
`output_gradient(value_j - value_k)`. The scaled-basis anchors always
supply a separating pair for a nonzero output derivative. Thus the
earlier raw-score guarantee now reaches actual trainable anchor parameters,
for every finite assignment; no supplied attention route is required.

[AnchoredSquaredError.lean](src/Transformer/GPTMini/Sparsemax/AnchoredSquaredError.lean)
proves the actual derivative of squared distance to an ordinary output
target in a real inner-product space. It vanishes exactly at the target.
For the anchored architecture, every wrong finite output has a nonzero
trainable score direction and is not a local minimum of this row loss.
The local-minimum proof uses the differentiable parameter curve rather
than assuming a full sparsemax derivative at support boundaries.

[AnchoredCorrection.lean](src/Transformer/GPTMini/Sparsemax/AnchoredCorrection.lean)
checks a complete finite correction. Values are `(-1/16, 15/16, 7)` and
the ordinary output target remains `1/2`. Zero initial anchor parameters
give weights `(1/2, 1/2, 0)`, output `7/16` and loss `1/256`. The lifted
active-pair derivative is `-1/8`. A raw score transfer of `1/16` is exactly
realized by the finite parameters `(-log 3, log 3)`, yields weights
`(7/16, 9/16, 0)` and reaches zero ordinary output error.

The closure is for the explicit anchored row architecture. It does not
transfer automatically to the existing query/key factorization, rule out
gradient cancellations between rows sharing parameters, supply a uniform
gradient lower bound, or prove global learning on the basis or Shakespeare.
Fixed value hulls can still make a target unattainable. All results use
the actual causal variational sparsemax with exact zeros and ordinary
output losses. Python and the completed benchmark cycle are unchanged.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings. No new
`sorry`, extra axiom, result resting on a `sorry`, vacuous statement or
placeholder is introduced. Python is unchanged since its passing 171-test
run recorded above.

## Abandoned schedule pair

The frozen constant/cosine schedule pair is incomplete and will not be
resumed. Trainer PID 168237 (session 11503) and archive worker PID 168332
(session 60201), stopped with `SIGSTOP` at 07:48:53 UTC, were terminated with
`SIGKILL` at 08:03 UTC on 2026-10-03 by the user's decision, before the Python
code moved out of `experiments/`. Neither process ran again after the pause,
so the raw state below is exactly the paused state.

The constant case's last canonical observation is update 288,000 and its last
complete gradient record is 288,250. Its latest canonical train and held-out
accuracies are 100%, but failures at 274,000 and 275,500 already violate the
strict final-window condition. Only 153 of 201 final-window checks exist.
The durable checkpoint is update 285,000, SHA-256
`14706a5662e124119fef1caf882c6adae1ba8b8a8dfd7ed5cf14e6080a371380`,
rechecked after termination. The cosine case never started. Raw state is under
`experiments/archive/synthetic_trainers/runs/adamw_stability_20261002/schedule_mod193_fraction25_lr0003_budget300k/`
(`experiments/runs/` until 2026-10-03).
The incomplete pair supports no conclusion about either schedule. Its frozen
plan pins sources at their `experiments/` paths; replaying it requires a
checkout of commit `698d904`.
