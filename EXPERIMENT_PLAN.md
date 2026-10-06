# Project experiment plan: why sparsemax attention fails, and a repair

Updated on 2026-10-06 UTC. **Done: steps 0 to 6 and the requested QKNorm follow-up.** The investigation cycle
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

## Pure sparse follow-up: reaching query/key projections

**Done on 2026-10-05 UTC.** Five modules add twenty-five proved theorems,
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and the linear projections, normalization and value sum at `73f8a0b`.

[QKChart.lean](src/Transformer/GPTMini/Sparsemax/QKChart.lean)
constructs actual unit keys in an orthogonal unit frame. A desired score
`s` in `(-g, g)` is represented by
`(s/g) * query + sqrt(1 - (s/g)^2) * transverse`. The existing normalized
dot product, with temperature `log g` and epsilon at most one, is proved
equal to `s`. The vector chart is differentiable on its strict interval.

[QKAnchors.lean](src/Transformer/GPTMini/Sparsemax/QKAnchors.lean)
sets `g = c + 1 + sum exp ordinary`, which strictly dominates every
anchor and ordinary score at all finite parameters. The actual QKNorm
scores therefore equal the earlier anchored scores. Every key has norm
one; the key array is differentiable in the learned anchor parameters.
The actual sparse example retains weights `(1/2, 1/2, 0)`.

[QKTaskDirections.lean](src/Transformer/GPTMini/Sparsemax/QKTaskDirections.lean)
proves the full active-value span on these actual Q/K rows and excludes
a zero ordinary task derivative both in anchor parameters and in all
actual key-vector directions. The chain rule uses the genuine differentiable
key chart; no sparsemax derivative is assigned or assumed at support boundaries.

[QKSquaredError.lean](src/Transformer/GPTMini/Sparsemax/QKSquaredError.lean)
removes the nonzero output-gradient premise for ordinary squared output
error: any wrong output has a nonzero actual K direction and is not a
local minimum of the key-vector loss. The finite correction's loss
`1/256` to `0` is now verified through the existing QKNorm operator.

[QKProjection.lean](src/Transformer/GPTMini/Sparsemax/QKProjection.lean)
evaluates actual shared linear Q/K projection matrices on standard-basis
inputs. Each input selects one column, so the nonzero task direction
reaches the jointly trained projection parameters. Equal inputs are
proved to force equal scores under every shared projection, explaining
why arbitrary input embeddings do not inherit this freedom.

[QKProjectedError.lean](src/Transformer/GPTMini/Sparsemax/QKProjectedError.lean)
adds four further proved theorems. Wrong squared-error outputs have a
nonzero direction in the jointly learned Q/K projection matrices and
cannot be local minima of that projection loss. The complete finite
correction, now evaluated through the shared linear projections,
normalization and actual sparsemax, changes only the key matrix and
reduces ordinary output loss from `1/256` to `0`.

This closes the score-to-Q/K transfer for an explicit restricted row
architecture. It requires an orthogonal unit query frame, chart-parameterized
keys, a gain depending on this row's ordinary score parameters, `d + 1`
visible value anchors and independent input coordinates for the projection
result. These are modifications, not consequences of the existing unrestricted
transformer. Shared-row gradient cancellation, target attainability with fixed
values and global convergence remain open; no basis or Shakespeare training
claim is made. Python and the completed experiment cycle are unchanged.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Pure sparse follow-up: removing standard-basis inputs

**Done on 2026-10-05 UTC.** Six modules add forty proved theorems,
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and the shared projections and normalization at `73f8a0b`.

[InputDecoder.lean](src/Transformer/GPTMini/Sparsemax/InputDecoder.lean)
proves that a finite input family's linear independence is equivalent to
the existence of a continuous linear coordinate decoder. Its existence
comes from a proved left inverse of the synthesis map; no decoder oracle
or unproved accessibility assumption is used. Inputs need not be standard
basis vectors, orthogonal, or span the ambient space. The rank restriction
is explicit: the full family must have size at most the input width, and
a longer context cannot admit a full coordinate decoder.

[ProjectionLift.lean](src/Transformer/GPTMini/Sparsemax/ProjectionLift.lean)
constructs the actual matrix-column lift and proves its evaluation is a
right inverse. [ProjectionUpdate.lean](src/Transformer/GPTMini/Sparsemax/ProjectionUpdate.lean)
turns it into an affine update from an arbitrary current key matrix. The
update starts at that matrix, is differentiable, realizes every requested
key row, preserves the matrix's action on decoder-kernel inputs, and
composes correctly with subsequent updates. This closes the gap between
a specially constructed matrix and a local perturbation at actual parameters.

[QKIndependentDirections.lean](src/Transformer/GPTMini/Sparsemax/QKIndependentDirections.lean)
transports a nonzero ordinary output derivative into actual shared key
columns and joint Q/K columns on any independent input family.
[QKIndependentError.lean](src/Transformer/GPTMini/Sparsemax/QKIndependentError.lean)
proves that wrong squared-error outputs cannot have a zero joint matrix
derivative and cannot be local minima of that matrix loss. The latter
uses continuity of the affine lift and holds without assuming sparsemax
differentiable at inactive support boundaries.

[MixedInputQK.lean](src/Transformer/GPTMini/Sparsemax/MixedInputQK.lean)
supplies concrete inhabited premises with input vectors `(2, 1, 0)`,
`(1, 2, 0)` and `(0, 0, 1)`. Actual shared matrices and QKNorm realize the
anchored scores. A finite change of the key matrix lowers ordinary output
error from `1/256` to `0`, retaining the third weight exactly zero.

The unit query frame, chart-restricted projected keys, fixed values,
visible value anchors and single unrotated row remain explicit conditions.
Full-family independence is still stronger than arbitrary learned input
embeddings; long contexts, shared-row cancellation and whole-model
convergence remain open. No new basis or Shakespeare training claim is
made. The completed experiment cycle and Python implementation are unchanged.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Pure sparse follow-up: partial anchor access on long contexts

**Done on 2026-10-05 UTC.** Five modules add twenty-seven proved
theorems with no new `sorry`, derived from arXiv:1602.02068v2,
§2.2 and §2.5, and shared Q/K projections and QKNorm at `73f8a0b`.
The full-context independence and resulting context-width bound are
removed from the task-direction and local-minimum results below.

[PrefixInputs.lean](src/Transformer/GPTMini/Sparsemax/PrefixInputs.lean)
constructs dedicated anchor channels followed by arbitrary ordinary
embedding channels. Its proved continuous linear decoder identifies
each anchor and kills every ordinary input, regardless of repetitions,
rank or context length. A context longer than the input width is proved
dependent and still satisfies the partial decoder identities. This
architecture uses `A + B` input coordinates for `A + N` tokens, where
`N` is unrestricted. The ordinary embedding channels are not decoded.

[QKPrefixMatrix.lean](src/Transformer/GPTMini/Sparsemax/QKPrefixMatrix.lean)
builds a differentiable path through the actual current matrix that
changes anchor keys and preserves all ordinary keys. Because the ordinary
keys in the anchored family do not depend on anchor parameters, the
actual shared projection realizes the entire desired family along that
path. A special initial matrix with zero unseen components is not required.

[QKPrefixDirections.lean](src/Transformer/GPTMini/Sparsemax/QKPrefixDirections.lean)
proves that a nonzero ordinary output derivative survives in the shared
key matrix and joint Q/K matrices under only the partial decoder condition.
[QKPrefixError.lean](src/Transformer/GPTMini/Sparsemax/QKPrefixError.lean)
excludes zero joint matrix derivatives and erroneous local minima for
ordinary squared output error. The local-minimum proof uses the continuous
anchor path and does not assume full sparsemax differentiability at
inactive support boundaries.

[LongContextQK.lean](src/Transformer/GPTMini/Sparsemax/LongContextQK.lean)
provides explicit shared matrices for two anchors and arbitrarily many
repeated nonzero ordinary inputs, using input width three and head width
two. The actual normalized scores realize every finite anchor assignment.
The initial actual sparse row has weights `1/2` on both anchors and
exactly zero on every ordinary token. With anchor values zero and one,
ordinary values seven and output target zero, its incorrect output is
`1/2`. The general matrix theorems have concrete inhabited examples at
one hundred ordinary tokens, where full independence is proved impossible.

The partial decoder is an enforced input-channel modification, with a
proved implementation, rather than a consequence of arbitrary embeddings.
The current projected key family, unit query frame, active value anchors,
fixed values and single unrotated readout before XSA/output projection
remain explicit conditions. Shared-row cancellation, uniform gradient
bounds and whole-model convergence remain open. The completed experiment
cycle and Python implementation are unchanged; no new dataset training
claim is made.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Pure sparse follow-up: transfer to a shared multi-row objective

**Done on 2026-10-05 UTC.** Nine modules add fifty-eight proved theorems,
with no new `sorry`, derived from arXiv:1602.02068v2, §2.2 and §2.5,
and actual shared Q/K projections, QKNorm and value sums at `73f8a0b`.
This is a conditional transfer of the single-row result to the sum;
unconditional transfer is refuted by an actual shared-matrix counterexample.

[SupportSegment.lean](src/Transformer/GPTMini/Sparsemax/SupportSegment.lean)
proves that matching actual endpoint zero patterns make sparsemax affine
on the score segment, including inactive threshold ties. It constructs a
normalized threshold for the actual simplex projection and retains exact
inactive zeros; no attention targets, dense mixture or replacement
projection are introduced.

[ClippedKeySegment.lean](src/Transformer/GPTMini/Sparsemax/ClippedKeySegment.lean)
connects this score segment to one actual shared key-matrix segment,
with the query matrix fixed. The additional restriction is explicit:
projected keys at both endpoints have norm at most epsilon. Their whole
convex segment remains within that ball, where the actual QKNorm maximum
denominator is constant and normalized scores are affine for every row.
This replaces the preceding curved unit-key chart for this transfer;
arbitrary normalized unit-key segments need not be affine. The example
uses epsilon one. At the usual epsilon `1e-6`, these premises require
projected keys within that much smaller ball; they are not an automatic
property of the existing initialization or an experiment-backed repair.

[SharedRows.lean](src/Transformer/GPTMini/Sparsemax/SharedRows.lean)
defines actual multi-row outputs and their summed ordinary squared loss.
Different examples can have different inputs and frozen values, while all
use one pair of actual Q/K matrices. Endpoint norm and support conditions
give simultaneous affine outputs and exact inactive zeros without query
independence, an input decoder, row-specific parameters or a context-width
bound. [SharedRowLoss.lean](src/Transformer/GPTMini/Sparsemax/SharedRowLoss.lean)
proves convexity of the sum along that common matrix segment. A shared
endpoint fitting all ordinary targets gives exact loss `(1-t)^2 * L(0)`.
[SquaredSegment.lean](src/Transformer/GPTMini/Sparsemax/SquaredSegment.lean)
supplies the ordinary squared-loss and neighborhood arguments.

[SharedRowMinimum.lean](src/Transformer/GPTMini/Sparsemax/SharedRowMinimum.lean)
excludes a local minimum in jointly learned Q/K whenever a better
compatible shared key endpoint exists, even without an exact target fit.
Consequently, a joint local minimum is optimal against all clipped-key
endpoints with the same row supports and fixed Q. If a common admissible
endpoint fits the ordinary targets, positive-loss local minima are
excluded. [SharedRowDerivative.lean](src/Transformer/GPTMini/Sparsemax/SharedRowDerivative.lean)
proves that the actual common path has right derivative `-2 * L(0)`.
At positive loss, this rules out a zero full joint Q/K derivative without
assuming ambient sparsemax differentiability across support boundaries.

[SharedRowsExample.lean](src/Transformer/GPTMini/Sparsemax/SharedRowsExample.lean)
provides actual shared matrices for two different causal queries in a
four-position context. Both initial outputs are `1/2`, ordinary targets
are `11/16` and `5/16`, and the initial errors have opposite signs. One
shared key correction attains both: the sum drops from `9/128` to zero,
is `9/512` halfway along the actual matrix path, and has initial right
derivative `-9/64`. Both anchors stay active and ordinary weights stay zero.
Every conditional theorem has a concrete inhabited example.

[SharedRowCancellation.lean](src/Transformer/GPTMini/Sparsemax/SharedRowCancellation.lean)
refutes unconditional transfer. Two identical actual observations with
ordinary targets zero and one necessarily have the same output for every
shared parameter point. A certified sparse output `1/2` is a global and
local matrix minimum with summed loss `1/2`, despite both rows being wrong.
It has two active anchors and exact ordinary zeros, rather than a saturated
one-hot route. No common target fit exists. Ordinary target attainability
cannot be inferred from separate single-row correction theorems.

Inputs, values, gain and epsilon remain fixed, and readouts are unrotated
and before XSA/output projection. Joint attainability, compatible endpoint
supports and epsilon-clipped keys remain restrictions. Uniform gradient
bounds, unrestricted support changes and whole-model convergence remain
open. The completed experiment cycle and Python implementation are unchanged;
no new basis or Shakespeare training claim is made.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Convex embedding and attention foundation

**Done on 2026-10-05 UTC.** At the user's request, the next architectural
stage focuses on learned embeddings and attention; FFN and task-loss design
are deferred. Seven modules add fifty-five proved theorems without new
`sorry`, derived from sparsemax arXiv:1602.02068v2, Eq. (1) and Proposition 1.
This is a new restricted architecture, not a completed transformer repair
or an equivalence to the QKNorm implementation at `73f8a0b`.

[EmbeddingGram.lean](src/Transformer/GPTMini/Sparsemax/EmbeddingGram.lean)
learns one positive semidefinite Gram matrix on the query and key copies of
a finite vocabulary. Its full bounded domain is proved convex; every feasible
matrix has exact finite embedding coordinates. Query and key families both
remain trainable, and no fixed small feature width is assumed. Unnormalized
Gram scores replace QKNorm, RoPE and the learned multiplicative gain.

[GramRouting.lean](src/Transformer/GPTMini/Sparsemax/GramRouting.lean)
uses this single matrix across every specified context and row. Shared token
identities select affine content scores, so identical visible token codes
receive identical weights; occurrence information requires richer codes.
The joint embedding/causal-weight domain is convex with no prescribed support.
[GramRoutingEnergy.lean](src/Transformer/GPTMini/Sparsemax/GramRoutingEnergy.lean)
proves joint convexity and the exact Jensen gap of the squared projection
energy. Actual sparsemax conditionally minimizes it at any fixed learned
Gram. The score-square term is retained: it is variable during embedding
learning. A future objective on free attention rows can change their
conditional optimizer, so this energy alone is not an exact task-training
reformulation.

[GramSupport.lean](src/Transformer/GPTMini/Sparsemax/GramSupport.lean)
proves that a Gram entry cap below one half guarantees two positive weights
on every row with two visible positions, for every feasible learned matrix.
A scalar embedding example has an exact visible zero whereas the feasible
zero Gram has full support. No routing targets or frozen Q/K are needed.
[GramBoundary.lean](src/Transformer/GPTMini/Sparsemax/GramBoundary.lean)
proves that the unrestricted exact sparsemax graph remains nonconvex even
inside this bounded PSD domain. Omitting the score-square term has an actual
Jensen violation. Ordinary value mixing has a nonconvex exact output graph
even on an unchanged full two-slot support with bounded scores and values;
this obstruction precedes any task loss or FFN.

[NormalizedGram.lean](src/Transformer/GPTMini/Sparsemax/NormalizedGram.lean)
provides an exact positive restriction: require each specified Gram score
row to be nonnegative, sum to one and vanish at future positions. These are
linear constraints on the learned matrix. The normalized domain is proved
convex, actual variational sparsemax fixes every such row, and the exact
embedding/attention graph is convex. Supports can acquire or lose exact
zeros. Normalization is an architectural constraint; it is not inferred
from PSD or an ordinary norm bound, and arbitrary-context feasibility is
not assumed.

[NormalizedGramExamples.lean](src/Transformer/GPTMini/Sparsemax/NormalizedGramExamples.lean)
inhabits the exact domain at cap `3/8`. Two genuine one-feature embedding
families have different Q and K squared norms. The actual row changes from
`(1/4, 1/4, 1/4, 1/4)` to `(1/3, 1/3, 1/3, 0)`; its true Gram midpoint
gives `(7/24, 7/24, 7/24, 1/8)`, exactly the mean endpoint attention.
A positive two-by-two minor proves that this midpoint cannot be recovered
with only one feature, recording the cost of a fixed small embedding width.

The foundation covers Q/K embeddings and attention weights on the specified
finite contexts. Jointly learned value mixtures, arbitrary-context
normalization, a practical embedding-width bound, FFN and task loss remain
outside the guarantee. No new basis or Shakespeare training is claimed.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Exact convex coordinates for jointly learned values

**Done on 2026-10-05 UTC.** The user's joint-values extension adds five
modules and forty-six proved theorems without new `sorry`. FFN and task-loss
design remain deferred. This is an exact change of coordinates on an explicit
finite-context architecture, not a convexity claim in the original value
parameters or an unrestricted language-model guarantee.

[InvertibleGram.lean](src/Transformer/GPTMini/Sparsemax/InvertibleGram.lean)
uses one distinct-token context with every causal row and the normalized
learned Gram from the previous stage. A positive diagonal floor `eta` adds
linear constraints. The domain remains convex, actual sparsemax attention
is lower triangular, and its determinant is positive for every feasible
learned Gram. Off-diagonal supports can change. The entry cap is explicit;
the forced first causal row cannot satisfy the earlier sub-half cap.

[GramValues.lean](src/Transformer/GPTMini/Sparsemax/GramValues.lean)
replaces a learned shared value table `V` by its output table `Z = A(G) * V`.
It decodes `V = A(G) inverse * Z`, with both inverse identities, uniqueness,
injectivity and surjectivity proved for the actual sparsemax/value product.
Changing the Gram also admits an exact output-preserving value transport.
No original values are frozen, private row values or routing targets used.

[JointGramValues.lean](src/Transformer/GPTMini/Sparsemax/JointGramValues.lean)
proves convexity of the joint Gram/output domain and affinity of the actual
decoded forward map. Any future convex objective on outputs is convex in
these new learned coordinates. Every feasible original Gram/value pair is
recovered exactly, and every output table is attainable once the Gram domain
is inhabited. This is a bijective coordinate change, not a rank relaxation.
An output-only objective therefore leaves the Gram undetermined: learned
values can compensate any feasible attention change on this context.

[GramValueExampleGrams.lean](src/Transformer/GPTMini/Sparsemax/GramValueExampleGrams.lean)
and [JointGramValueExamples.lean](src/Transformer/GPTMini/Sparsemax/JointGramValueExamples.lean)
give two genuine Q/K embedding tables in the same cap-four, floor-one-half
domain. Actual attention changes from identity to `[[1, 0], [1/2, 1/2]]`,
and one value table shared across both rows changes from `(1, 0)` to `(0, 2)`.
Outputs change from `(1, 0)` to `(0, 1)`. At the true joint midpoint, actual
attention is `[[1, 0], [1/4, 3/4]]`, recovered values and outputs are both
`(1/2, 1/2)`, and the determinant is `3/4`. Literal mean original values
instead give second output `7/8`, witnessing the nonlinear value decoder.

The guarantee covers this one distinct-token context, before XSA and output
projection. Repeated-token or arbitrary multi-context value sharing is not
proved. Original value penalties or norm bounds and a fixed small embedding
width are not preserved, and no uniform inverse conditioning bound is proved.
No new basis or Shakespeare training is claimed.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Convex shared memory across causal data contexts

**Done on 2026-10-05 UTC.** The user's continuation adds eight modules and
sixty-eight proved theorems without new `sorry`. One learned dictionary
and one learned value table now serve arbitrarily many context queries,
including repeated tokens. The positive construction changes ordinary
input-token self-attention into attention to learned parameter memory.
Fixed causal probability codes from data mix learned query embeddings.
FFN and task-loss design remain deferred.

[MemoryGram.lean](src/Transformer/GPTMini/Sparsemax/MemoryGram.lean)
normalizes every learned dictionary score row linearly and bounds its
diagonal below by a floor greater than one half. Together with a bounded
PSD Gram, these constraints define a convex domain. Actual variational
sparsemax equals its normalized genuine Q/K scores. Unit row mass gives
strict diagonal dominance, and Gershgorin proves nonsingularity without
a triangular support restriction. Both directional supports can change.

[MemoryValues.lean](src/Transformer/GPTMini/Sparsemax/MemoryValues.lean)
uses the exact global coordinates `Z = B(G) * V`, where `B` is dictionary
attention. It decodes one table `V = B(G) inverse * Z`, independent of the
context. Both inverse identities and uniqueness are proved. Original values
and both embedding families remain variable in the convex joint domain.

[ContextMemory.lean](src/Transformer/GPTMini/Sparsemax/ContextMemory.lean)
proves that scores for a probability code `M` are actual inner products of
mixed learned Q and the common learned K. Actual sparsemax equals `M * B`.
Data codes are inputs, not teacher routes or attention targets. All learned
memory slots are visible parameters preceding the query; they do not contain
future observations. Codes are fixed during optimization, while query and
key embedding coordinates remain trainable through the Gram.

[PrefixMemoryCodes.lean](src/Transformer/GPTMini/Sparsemax/PrefixMemoryCodes.lean)
constructs such codes from arbitrary causal token prefixes. Zero-score
sparsemax is proved exactly uniform on every visible prefix, and its mass
is aggregated by token identity. Every length and repetition pattern gives
a probability code. Future-token changes leave codes unchanged, and absent
visible tokens receive exactly zero mass. This concrete encoder retains
prefix frequencies and loses word order; other fixed causal probability
codes can be supplied to the general memory results.

[SharedMemoryValues.lean](src/Transformer/GPTMini/Sparsemax/SharedMemoryValues.lean)
proves simultaneous exact original-value recovery and actual forward affinity
for all context rows. The shared decoded output is `M * Z`, computed through
the real sparsemax/value operation. Prefix-code forwards are causal even
outside the structural Gram domain.
[SharedMemoryGeometry.lean](src/Transformer/GPTMini/Sparsemax/SharedMemoryGeometry.lean)
proves that every future convex criterion on the whole context-output table
is convex in joint Gram/output coordinates. The exact prediction class is
the linear image `{M * Z}` and is convex; arbitrary context targets need not
be attainable. A common decoder compensates feasible Gram changes across
all contexts, so an output-only objective leaves the Gram undetermined.

[MemoryExampleGrams.lean](src/Transformer/GPTMini/Sparsemax/MemoryExampleGrams.lean)
gives a genuine feasible identity embedding Gram for every dictionary size.
Its two-slot witness changes Q/K squared norms and actual memory attention
from identity to `[[3/4,1/4],[1/4,3/4]]`, acquiring both off-diagonal supports.
[SharedMemoryExamples.lean](src/Transformer/GPTMini/Sparsemax/SharedMemoryExamples.lean)
uses actual repeated-token prefixes `(1,1,1)`, `(0,1,1)` and `(0,0,1)`.
One original value table changes from `(1,0)` to `(-1/2,3/2)`, while global
output coordinates change from `(1,0)` to `(0,1)`. The three actual context
outputs change from `(0,1/3,2/3)` to `(1,2/3,1/3)`. At the true midpoint,
one shared decoded table `(1/2,1/2)` gives all three outputs one half.
Literal mean original values instead give first-context output `11/16`.
Every actual output obeys `Y2 = 2 * Y1 - Y0`; the target triple `(0,0,1)`
is proved unattainable for every feasible Gram and common value table.

The guarantee covers the stated memory architecture and fixed data codes.
It does not recover unrestricted occurrence self-attention, trainable codes,
a fixed small embedding width, original value penalties or inverse norm bounds.
Outputs precede XSA and output projection. No new basis or Shakespeare training
is claimed, and the completed experiment cycle is unchanged.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Richer causal codes remove the three-target obstruction

**Done on 2026-10-05 UTC.** The user's challenge adds five modules and
thirty-eight proved theorems without new `sorry`. The previous exclusion
of `(0,0,1)` was specific to the rank-two token-frequency code. It remains
valid for that encoder; it is not a universal impossibility for shared
values or the convex memory architecture. Richer codes are derived from
observations alone, and the learned Gram, values and supports remain variable.

[PrefixFeatureCodes.lean](src/Transformer/GPTMini/Sparsemax/PrefixFeatureCodes.lean)
pushes causal zero-score sparsemax probabilities through arbitrary fixed
position/token features. Probability feasibility and future insensitivity
are proved, as is the actual shared-memory output formula. The same three
prefixes use six shared `(position, token)` slots; their code rows occupy
`(1,3,5)`, `(0,3,5)` and `(0,2,5)`, each with weights one third.

[PositionalMemoryTargets.lean](src/Transformer/GPTMini/Sparsemax/PositionalMemoryTargets.lean)
constructs a fixed right inverse `C` with `M * C = I`. For every finite
vector-valued target table `Y` and every feasible learned Gram, one common
original value table `V = B(G) inverse * C * Y` gives actual outputs `Y`.
Scalar global coordinates for `(a,b,c)` put `3*(a-b)`, `3*(c-b)` and `3*b`
in slots 1, 2 and 5. The exact actual prediction class is the entire target
space. From every current joint point, the update `Z += C * delta` produces
any desired output correction `delta`; no task loss or attention target is
needed for this forward-map statement. Explicit original values with only
slot 2 equal to three give the formerly excluded triple `(0,0,1)`.

[CodeMemoryOutputs.lean](src/Transformer/GPTMini/Sparsemax/CodeMemoryOutputs.lean)
proves a general categorical-code characterization. One global table extends
every target table constant on equal-code fibers. Actual shared-memory
attainment is equivalent to this consistency, for every feasible Gram.
The encoder is chosen from data before the target table is supplied.

[CausalPrefixKeys.lean](src/Transformer/GPTMini/Sparsemax/CausalPrefixKeys.lean)
encodes complete visible prefixes, masking every future position with `none`.
Equality of dictionary keys is proved equivalent to equality of query positions
and every visible token. Word order, repetitions and prefix length are retained.
The full target-independent dictionary has exactly `(V + 1)^T` slots for
vocabulary size `V` and window length `T`, including unused signatures.

[CausalMemoryUniversality.lean](src/Transformer/GPTMini/Sparsemax/CausalMemoryUniversality.lean)
proves that the exact actual prediction class is all output tables consistent
on identical observed causal prefixes. Distinct prefixes permit arbitrary
finite vector targets; repeated observations with contradictory targets remain
impossible for a deterministic causal model. The joint actual forward is causal,
the shared prediction class is convex, and the earlier convex-objective theorem
applies. Original values are decoded globally, independently of context.

The six-slot result removes the specific example obstruction at modest cost.
Complete-prefix universality pays an exponential dictionary cost and unbounded
feature width; it does not prove an efficient compact ordinary transformer.
Fixed data codes and the learned-memory architecture remain explicit. An
output-only objective still leaves the Gram undetermined. FFN, task-loss choice
and new training remain deferred; the completed experiment cycle is unchanged.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

## Compact observed-prefix kernel memory

**Done on 2026-10-06 UTC.** The requested compact mathematical construction
adds six modules and forty-two proved theorems without new `sorry`. Register
`P` distinct observed causal prefixes before target fitting and use `P`
learned memory slots. This replaces the exponential complete-prefix dictionary
by a data-relative memory. The learned Gram and the common value table remain
variable on the earlier exact convex joint domain.

[PrototypeKernelCodes.lean](src/Transformer/GPTMini/Sparsemax/PrototypeKernelCodes.lean)
defines the fixed data kernel
`k(q,j) = indicator(prefix(q) = prefix(j)) + 1 + dot(phi(q), phi(j))^2`.
Every entry is at least one, giving a positive normalization denominator
even on unseen queries. Its normalized profiles are probability codes.
Evaluation uses dot products and a normalized weighted sum of the global
output table, without enumerating all signatures or ordered feature pairs.

[PrototypeKernelFeatures.lean](src/Transformer/GPTMini/Sparsemax/PrototypeKernelFeatures.lean)
proves that the squared dot product is the Gram of ordered pair features.
On distinct registered observations, the training kernel is identity plus
the constant Gram plus this genuine pair-feature Gram. It is positive
definite and nonsingular for arbitrary real feature rows, even when those
rows are zero or rank deficient. The two-prototype example with scalar
features `(1,2)` gives kernel `[[3,5],[5,18]]` and determinant 29.

[PrototypeKernelTraining.lean](src/Transformer/GPTMini/Sparsemax/PrototypeKernelTraining.lean)
constructs the data-derived right inverse `C = kernel inverse * diagonal(row mass)`
of the normalized training codes `M`, proving `M * C = I`. Every vector
target table `Y` has global coordinates `Z = C * Y` and one common original
value table `V = B(G) inverse * C * Y`. Actual sparsemax/value outputs are
exactly `Y` for every feasible learned Gram. Any current point admits every
prototype-output correction through `Z += C * delta`; attainability is a
proved consequence of the encoder, not an assumed target-compatibility input.

[BoundedGramWidth.lean](src/Transformer/GPTMini/Sparsemax/BoundedGramWidth.lean)
retains the explicit dimension in Mathlib's positive spectral decomposition.
Every PSD Gram on `2P` query/key indices is recovered exactly with Q/K width
`2P`, preserving the squared norm cap. The same feature table realizes all
context scores and actual normalized attention. Context count does not
increase this width, and no nonconvex rank restriction is added to the domain.

[PrototypePrefixMemory.lean](src/Transformer/GPTMini/Sparsemax/PrototypePrefixMemory.lean)
constructs the kernel profiles from actual masked prefix signatures and
causal position/token features. Future changes in either queries or prototype
records leave codes and actual outputs unchanged. Its final theorem combines
arbitrary prototype-target recovery, one common original value table and
Q/K width `2P`, for every feasible learned Gram. The actual three-prefix
example fits `(0,0,1)` with three memory slots, versus six position/token
slots or twenty-seven complete-signature slots in the earlier constructions.

[PrototypeKernelGeometry.lean](src/Transformer/GPTMini/Sparsemax/PrototypeKernelGeometry.lean)
proves convexity of every future convex output criterion on the same joint
domain, together with the actual normalized-kernel forward formula. Matrix
rank gives a lower bound: this exact fixed-code affine chart needs at least
`R` memory slots to realize every independently specifiable vector target
table on `R` contexts. The construction attains that slot bound on its
registered distinct observations. This is a bound for this architecture,
not an impossibility for every transformer or structured target class.

The guarantees cover finite registered observations. Learned Gram parameters
still grow quadratically with `P`, the width bound grows linearly, and fixed
features and prototype selection remain explicit. Positive profiles and the
diagonal floor force dense actual query attention; memory supports can still
change. Output-only optimization still leaves the Gram undetermined under
the common value decoder. Unseen queries have a well-defined prediction,
without a generalization guarantee or an independently specifiable target.
FFN, task-loss choice and new training remain deferred; the completed sparsemax
experiment cycle is unchanged. A sample-independent compact construction for
structured text targets remains a mathematical question.

Validation: full `lake build`, `./make.py audit`, `./make.py index` and
`./make.py forbidden` pass; the new modules have no warnings, new `sorry`,
extra axioms, results resting on a `sorry`, vacuous statements or placeholders.
Python is unchanged since the recorded passing 171-test run.

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
