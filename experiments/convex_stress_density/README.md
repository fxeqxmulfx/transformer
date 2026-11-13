# Convex density extraction from categorical subspaces

## Observable row balancing and missing axes

CM24 / Lean `75ca9276` proves that radial normalization of actual nonzero
feature rows admits the explicit uniform PSD trace-one density. Arbitrary
completion outside observed phase axes preserves ordinary observed CE and
the same latent spectral budgets. A separate theorem exposes the native
permutation family's constant-row capacity limitation. These statements do
not recover a hidden subspace or identify missing labels.

`feature_balance.rs` records measured energies, scales, missing rows and the
uniform matrix, independently auditing actual matrix readings and its spectrum.
It refuses near-zero observed rows and explicitly fills unobserved ones with
ones. The partial phase clock orders observed coordinates and assigns unused
ranks to missing entries. All original features, masks and returned permutations
are retained. Balancing does not preserve column centering or orthogonality;
the general interval core accepts these features without assuming either.

`balanced_search` preserves the complete CM21–23 report, including every
failed attempt and supplied earlier candidate. It proposes the uniform density
without an objective-optimality claim, adds declared interval variants, then
uses the independently certified CM23 refinements. Full input/state/work replay
and raw latent DFT/SVD/CE audits check new candidates. Five additional native
tests cover genuine unequal/zero feature rows, missing-axis phase collisions,
refused and capped slots, old CE ties, label renaming, tampering and replay.

On the existing 113-observation masked fixture, the balanced rank-one uniform
candidate already fits every observed and withheld cell; higher-rank candidates
can still be wrong. The retained earlier correct model wins the CE tie. Under
the smaller test policy, nine new seeds use 22 certified assignments and
7,458 interval iterations. This is fixture evidence. The original-data policy
and complete native replay harness are separately specified in
[`balanced_cohort`](balanced_cohort/README.md); the previous original3/9 result
remains until the frozen follow-up has run. All35/modes/transformer scope remains.

## Earlier extraction and refinement

Can the convex lift proved in CM21 separate Fourier modes when a supplied
singular subspace is an unknown rotation of the true mode basis?

Lean source: `DensityExtraction` at `1885957b`; manuscript
[`review/flat_spectrum.md`, CM21](../../review/flat_spectrum.md#cm21-convex-density-extraction-in-a-supplied-fourier-subspace).
The numerical input is a centered complex feature matrix `B` with `B*B=n I`
and two categorical row indices. No latent permutation, frequency, mode label
or held-out answer is passed to the extractor.

The variable is a Hermitian positive-semidefinite matrix `M`, with trace one
and affine readings `diag(B M B*)=1`. First maximize the real kernel entry at
the two rows; then maximize its imaginary part on the first optimum face.
There is no rank constraint. A bounded ADMM iteration alternates projection
onto PSD trace-one matrices and the affine readings. PSD projection uses a
Hermitian eigendecomposition and simplex projection of the eigenvalues;
the affine projector uses a normalized row Gram pseudoinverse with a declared
cutoff. The second stage uses the measured first-stage objective as its extra
affine target. This is a finite-precision implementation, not the exact
`LexMaximum` predicate or a proved convergence algorithm.

Every stage stores its matrix, affine state, dual state, row multipliers and
complete scalar trace, including capped/refused attempts. Acceptance checks
affine and splitting residuals, the dual step and the absolute primal-dual
gap. The numerical upper bound is
`lambda_max(C - A* y) + dot(y,b)` for PSD trace-one matrices satisfying `A M=b`.
An independent audit reconstructs these operators by polarization of actual
kernel readings, checks PSD/trace, recomputes the objective and bound, and
checks the returned phase column directly. Full replay repeats both stages.
These floating-point diagnostics are **not** interval or Lean certificates.

| Fixture | Purpose | Finding |
| --- | --- | --- |
| Three modes `1,3,9` modulo 13, complex rotated columns and permuted rows | Recover a character without giving its index to the solver | Both stages accept; the recovered column matches a true character within `1e-6`. |
| Two modes `1,12` modulo 13 | Real scores tie; the imaginary stage must select a mode | The first stage retains a half/half mixture; the second selects the positive-imaginary score. |
| Modulo-two collision | Check the boundary of the separation hypothesis | The PSD trace-one matrix with imaginary off-diagonal entries has constant readings. It must not be assumed diagonal. |
| Centered, re-orthonormalized noisy subspace | Distinguish normalization from the exact-subspace hypothesis | The affine equalities are inconsistent; the extractor explicitly refuses. |
| Capped/invalid input and modified artifacts | Check budgets, replay and refusal paths | Failures remain explicit; changing the matrix, dual, input, trace or audit is rejected. |

Twenty-six native tests cover these cases, the complex eigensolver/coordinate
conventions, categorical proposal/comparison and interval paths. Eight cover
the exact extractor, four cover the exact proposal path, six cover the interval
solver, three cover interval proposals and retained comparisons, and five
cover certified refinement, replay, categorical renaming and budget limits.
The standalone `fixtures-v1`
run accepts the rotated case after
46 and 41 iterations; its final diagonal error is `4.11e-9`. The real-tie
case takes 3 and 3 iterations with diagonal error `1.11e-15`. Both artifacts
replay and pass the raw audit. Peak RSS for this fixture run is 7,116 KiB;
the CLI also rejects an attempt to overwrite an existing artifact.
This is an extraction prototype, not an original-task result.
The exact equalities are brittle under subspace estimation error; normalization
alone does not fix that gap. A tolerance relaxation needs its own explicit
convex-domain argument and quantitative limits before being called a repair.

Dimensions are bounded by `n<=193`, rank `<=8`, and 16,384 iterations per
stage. The explicit extraction-buffer forecast is below 80 MiB; launch the
entire native process with the user's 64 GiB limit. Single-thread BLAS keeps
the run deterministic. Generated artifacts stay in the ignored root-linked
`runs/` directory. The frozen earlier producers and all35 short/long,
full/minibatch/forward-only and conventional-transformer comparisons remain
unchanged. No original-data protocol for this new constructor is frozen.

`search.rs` now constructs complex label sketches from observed cells in
first-occurrence class order, centers rows and columns, and proposes small
leading subspaces. Extraction sees only each feature matrix and rows 0 and 1.
The right phase vector is computed through the **same original sketch's
adjoint**, then conjugated. Clock ordering proposes latent permutations;
empirical categorical decoding refuses uncovered latent states. This discrete
postprocessing is a conditional candidate constructor, not a global convex
decoder-search oracle. There are no arithmetic input features or CE gradients.
Every candidate requires ordinary CE and the existing raw latent DFT/SVD
audit. All attempted subspaces, extraction traces, failures and model-audit
budget refusals remain in the report. A global iteration budget counts actual
work on completed calls and charges the entire reserved allowance on an error.

The categorical fixture uses an arbitrarily permuted 11-state table and an
arbitrary decoder that merges several latent states. With all 121 cells
observed, the new path returns the complete table correctly, using 364 total
ADMM iterations and four model audits. With the eight
cells at multiples of 17 removed, all six new subspace proposals are refused.
Their affine-consistency residuals range from `0.0166` to `1.15`, far above
the declared `1e-8` tolerance. Peak RSS for both categorical fixtures is
8,096 KiB under the 64 GiB process limit; both saved proposals replay.
The previous CM15 constructor still recovers the fixture; `comparison.rs`
retains it, every other supplied prior decoded vertex, and all new attempts,
then selects using observed CE alone. This comparison is limited to the
supplied decoded vertices; it does not replace the pending full35 comparison
with the other model families. Categorical output renaming, exact proposal
replay, full attempt/budget accounting and modified-artifact rejection pass.
The masked failure is evidence that exact amplitude constraints are too
brittle for estimated subspaces. Lean CM22 (`DensityBounds`) now proves the
convex interval domain and quantitative limits: if feature entries differ by
at most `delta` from a reference bounded by `R`, every density reading changes
by at most `kappa=d^2 delta (2R+delta)`. For a unitary-rotated Fourier reference
with separated differences, a relaxed feasible density, model tolerance
`epsilon>=kappa`, objective error `eta` and a positive real-score gap `gamma`,
the mass outside the best mode is at most
`(eta+2kappa+d^2(epsilon+kappa))/gamma`.
These are conditional bounds, not a certificate that observed labels provide
the reference subspace or its error. Approximate real-score ties remain
uncovered. The exact extractor remains available alongside the new interval
solver; the interval proposal report includes the entire exact search.

`interval.rs` implements the CM22 domain with a separate `model_tolerance`.
A bounded primal-dual proximal update couples the normalized diagonal-reading
operator with PSD trace-one projection. Its dual update uses the conjugate
of the interval indicator; the primal update uses the full adjoint operator.
The step sizes use the explicit Frobenius bound on that operator. Acceptance
checks raw interval violations, the objective gap and both scaled iterate
steps. The numerical dual upper bound is
`lambda_max(C-A* y)+support_box(y)`, independently rebuilt by polarization in
`interval_audit.rs`. Input, matrix, multipliers, extrapolated state, every
scalar iterate diagnostic and the raw audit are replayed. No convergence or
interval-arithmetic certificate is claimed.

This variant maximizes one real objective. Its accepted kernel column is an
empirical candidate and may be mixed; purity is recorded rather than imposed
as a nonconvex constraint. At a real-score tie the tests retain a half/half
mixture. A rank-one feature fixture scaled by `11/10` has diagonal error
`0.21`: the zero-tolerance run refuses, while a declared `0.21` interval
accepts with unchanged numerical thresholds. The interval core permits
unnormalized inputs, as CM22's perturbation bound does; it does not infer
centering, normalization or structural feature error. A separate centered,
re-orthonormalized noisy three-mode test is refused by the exact extractor and
accepted with a declared `0.1` interval; the raw interval violation is below
`1e-8`. A complex rotated and row-permuted exact three-mode test also recovers
an actual character within `1e-6`. These are controlled feature fixtures.

The standalone `interval-fixtures-v1` run accepts the exact case in 63
iterations and the mixed real-tie case in 167. The scaled feature fixture is
refused at its 64-iteration cap with zero model tolerance, and accepts in one
iteration with model tolerance `0.21`. All four artifacts pass raw audits and
replay; peak RSS is 6,748 KiB. Small negative apparent dual gaps are recorded
alongside the primal interval violations, and acceptance uses the absolute
gap. They are not presented as exact feasible optimality certificates.

`interval_search.rs` applies declared tolerances `0.03,0.1,0.3` to each exact
search subspace, retaining all earlier exact proposals and all new attempts.
It reconstructs the identical original categorical sketch for right-side
phases, uses the same empirical decoder and raw latent DFT/SVD/CE audits,
and scores candidates by observed CE. The default caps are 4,096 iterations
per attempt, 73,728 additional iterations globally and eight model audits.
Construction work is recorded separately for exact and interval proposals;
verification repeats that work and is included in the measured process time.
All supplied earlier decoded models remain in `compare`, with earlier-first
CE ties and full replay. These comparisons still cover supplied decoded
vertices, not the full35 collection of other model families.

On the masked fixture, the first 65,536-iteration global budget spent its
allowance before the final two slots; all 18 slots and their refusals remain
in `interval-sketches-v1`. A separate diagnostic restricted tolerance to
`0.3`, with 16,384 iterations per slot and a 98,304 global cap. It found a
later candidate in 1,189 iterations. Its kernel density was mixed, with
purity error `0.379`, and its decoded answers had 53/113 observed errors and
6/8 hidden errors. The errors on hidden cells are evaluated after selection;
only observed feasibility diagnostics motivated the additional solver work.

The revised 73,728 global budget covers all 18 masked-fixture slots, using
70,821 iterations. It returns that same poor candidate, with observed CE
`2.012067`; the earlier correct decoder has CE `1.543040` and stays selected.
The complete fixture retains its exact successful candidate, using 19,529
additional interval iterations. `interval-sketches-v2` preserves both complete
reports, raw audits and comparisons; the earlier reports and longer diagnostic
still replay. Peak RSS for both revised fixtures with verification is
51,812 KiB under the 64 GiB cap. No benchmark improvement is established.

This fixes a numerical-domain obstruction and budget starvation, but does
not identify the correct phases from masked labels. The accepted bad candidate
passes the latent flat-spectrum audit, illustrating the remaining limitation.
Observed-label agreement guides the phase/decoder recovery below.

## Certified observed-label refinement

CM23, proved in Lean first at `0bb5496f`, identifies actual unit-logit CE
excess with observed mismatch count. Feasible assignment dual potentials,
tight on the returned permutation, certify the minimum of each axis update.
Decoder count maxima cannot increase CE. Every new decoded phase table remains
in the same convex spectral prediction hull, including imperfect fits. This
certifies local updates; the discrete proposal search is not a global convex
optimization oracle and supplies no universal unseen-label guarantee.

`assignment_certificate.rs` returns integer row and column potentials.
An independent checker uses `i128` to verify every inequality, tight edge and
objective, including malformed extreme saved potentials without overflow.
`refinement.rs` alternates decoder, row, decoder, column and decoder steps,
keeping old decoder entries on count ties. It stops on exact observed fit,
repeated state or the declared round cap. Every state and assignment dual is
saved. Uncovered latent states are recorded and are not selectable endpoints.
`refinement_audit.rs` separately reconstructs actual labels, checks error
descent and coverage, rebuilds assignment costs, verifies decoder maxima,
and audits the selected endpoint's raw latent DFT/SVD and full-vocabulary CE.

`refinement_search.rs` retains the full exact/interval comparison, including
all failures and supplied earlier candidates. Defaults are 24 rounds per seed,
16 seeds and 256 assignments overall. Errors charge their full reserved
allowance; every skipped slot gets an explicit reason. Earlier CE ties retain
the earlier model. Integer construction work and independent audit work are
recorded separately; full replay repeats the work. No CE gradients are used.

The masked eleven-state fixture improves from **53/113 observed errors to
zero**, after two assignments and three decoder refits. Independent evaluation
of its 121 known fixture labels finds zero errors, including all eight withheld
cells. Construction uses 2,486 assignment-label comparisons and 242 assignment
relaxations; the independent checker scans 27,346 observation/assignment pairs
and verifies 242 dual inequalities. CE becomes `1.5430404724093327`, the ideal
unit-logit value for its full eleven-class vocabulary. The unrefined bad
candidate and every exact/interval failure remain in the report.

Without a prior model, the refined masked candidate is selected. With the
earlier correct decoder supplied, that decoder remains selected on the CE tie.
The complete fixture retains its existing exact success, requiring no new
assignments. All four comparisons in `refined-sketches-v1` replay from the
original interval inputs and saved parents. Peak RSS, including verification,
is 69,896 KiB under the 64 GiB cap. This is a fixture recovery result, not a new
original-task benchmark result or a full35/transformer comparison.

The five refinement checks include all 512 binary three-by-three assignment
matrices against exhaustive permutation minima; count/dual/state/work/input
tampering; output renaming; uncovered states; seed and assignment caps; and
full retained-parent replay. Frozen earlier cohorts are unchanged.

The separate [`cohort`](cohort) harness freezes this measured policy on nine
original pair tasks, including division and the blind long case. It retains
the earlier CM15 proposal and all new attempts, seals observed-only selection
before metrics, and independently replays every scored outcome under 64 GiB.
Its four synthetic harness tests preceded original discovery. All nine original
tasks now replay: 3/9 pass all splits, with no improvement over the earlier
decoder. All 54 exact and 162 interval attempts remain failed/capped, and
division has no eligible candidate. Peak RSS is 47,000 KiB. The cohort README
records every outcome; full35 and transformer integration remain pending.

From this directory, using native tools only:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CARGO_BUILD_JOBS=2
prlimit --as=68719476736:68719476736 -- cargo test --locked --offline
cargo fmt -- --check
prlimit --as=68719476736:68719476736 -- cargo clippy --locked --offline --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --locked --offline --bins --examples
prlimit --as=68719476736:68719476736 -- target/release/examples/fixtures runs/fixtures-v1
prlimit --as=68719476736:68719476736 -- target/release/convex-stress-density replay runs/fixtures-v1/rotated.input.json runs/fixtures-v1/rotated.artifact.json
prlimit --as=68719476736:68719476736 -- target/release/examples/sketches runs/sketches-v1
prlimit --as=68719476736:68719476736 -- target/release/convex-stress-density replay-proposal runs/sketches-v1/masked.input.json runs/sketches-v1/masked.proposal.json
prlimit --as=68719476736:68719476736 -- target/release/examples/intervals runs/interval-fixtures-v1
prlimit --as=68719476736:68719476736 -- target/release/convex-stress-density replay-interval runs/interval-fixtures-v1/perturbed_interval.input.json runs/interval-fixtures-v1/perturbed_interval.artifact.json
prlimit --as=68719476736:68719476736 -- target/release/examples/interval_sketches runs/sketches-v1 runs/interval-sketches-v2
prlimit --as=68719476736:68719476736 -- target/release/convex-stress-density replay-interval-proposal runs/interval-sketches-v2/masked.input.json runs/interval-sketches-v2/masked.proposal.json
prlimit --as=68719476736:68719476736 -- target/release/examples/refined_sketches run runs/interval-sketches-v2 runs/refined-sketches-v1
prlimit --as=68719476736:68719476736 -- target/release/examples/refined_sketches replay runs/interval-sketches-v2 runs/refined-sketches-v1
```

`extract INPUT.json NEW_ARTIFACT.json` also accepts externally supplied feature
inputs and refuses an existing destination. Replay compares the artifact with
the separately supplied original input. `examples/fixtures.rs` documents the
input schema and generates reproducible exact-geometry inputs only.
`propose INPUT.json NEW_PROPOSAL.json` constructs the categorical proposal;
`examples/sketches.rs` specifies its input schema and the declared fixture
budgets (two sketches, ranks 1/2/3, 256 iterations per stage, 2,048 total
iterations and four model audits). Original-data discovery must use a
separately frozen protocol, source, binary, input fingerprints and replay.
`interval INPUT.json NEW_ARTIFACT.json` and `replay-interval` provide the
corresponding interval interface, with the same no-overwrite and file-size
bounds. `examples/intervals.rs` specifies its input schema.
`propose-interval` and `replay-interval-proposal` use the input schema saved by
`examples/interval_sketches.rs`. Reproduce the earlier budget and the longer
diagnostic without changing code, using fresh destinations:

```sh
jq '.policy.total_iterations=65536' runs/interval-sketches-v2/masked.input.json > runs/masked-budget65536.input.json
prlimit --as=68719476736:68719476736 -- target/release/convex-stress-density propose-interval runs/masked-budget65536.input.json runs/masked-budget65536.proposal.json
jq '.policy.variants=[.policy.variants[2] | .iterations=16384] | .policy.total_iterations=98304' runs/interval-sketches-v2/masked.input.json > runs/masked-longer.input.json
prlimit --as=68719476736:68719476736 -- target/release/convex-stress-density propose-interval runs/masked-longer.input.json runs/masked-longer.proposal.json
```
