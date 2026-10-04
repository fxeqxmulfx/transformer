# Project experiment plan: why sparsemax attention fails, and a repair

Updated on 2026-10-04 UTC. **In progress: step 1. Step 0 is done.** The investigation cycle
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

## 0. Run sparsemax on the basis

**Done on 2026-10-04.** [basis_sparsemax](experiments/basis_sparsemax/README.md)
defines the 60 benchmark runs and six short timing runs, with the existing
diagnostics recording training batch loss and head scales. Sparsemax's
projection, backward, model logits and every parameter gradient agree with
eager evaluation within rounding under full-graph compilation. Diagnostics
preserves compiled observation metrics and the complete training state
under both weights, and a sparsemax run resumes onto its own records.
All 142 lab tests pass. The measured sparsemax/softmax update-time ratios are
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

## 5. Confirm on mod 193

Train the best repair as a label of `experiments/mod193_stability` beside
`sparsemax`, changing nothing else, with step 2's measurement, for its
300,000 updates; the archived pair took 2.6 and 3.5 hours on the GPU. The
lab has not trained `base` or `sparsemax` yet: their table holds archived
runs of the historical trainers, so train both too. Record the archived
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

## 6. Prove the supported statements in Lean

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
