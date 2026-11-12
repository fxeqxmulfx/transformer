# Certified unseen answers for the paired decoder

Can training-only identification repair the wrong global CE minima of the
paired spectral model? Lean CM18 at `5c0fe9bc` comes first. Its
`accepted_decoding` proves a certificate sound for a fixed decoder on an odd
cyclic group whose fibers are exactly `{z, -z}`. The target must have unknown
row and column phases in this family. This structural assumption concerns
the unknown truth and cannot be tested on training samples alone.

`budget_wrong_minimum` proves why optimization alone is insufficient: even
with an accepted certificate, the unchanged spectral relaxation has a global
CE minimum that predicts the wrong unseen label. `certified_restriction`
repairs this by adding linear equalities for the certified output rows. It
preserves convexity and the optimal training CE under the stated assumptions.
The restriction is a changed prediction family; the equalities do not follow
from unconstrained minimization.

## Native interface and scope

| Component | What is computed |
| --- | --- |
| `Training` | Canonical observed cell/label pairs, retaining exposure counts across batches and refusing conflicting repetitions atomically |
| `Certificate` | Proposed row/column phases and the four indices of an observed rectangle; no held-out labels or trusted coverage flags |
| `check` | Exact integer checks of the rectangle, independent closure of Lean `Pinned`, and agreement at every reached training observation |
| `discover` | Bounded categorical proposals, sign choices and phase propagation, followed by the independent checker; selection by covered-cell count alone |
| `Verified::anchor` | A symmetric phase table at radius and nuclear budget `N`, with the certified decoded rows; it need not fit unreached training observations |
| `Solver` | The certified equalities in every row projection, with full-batch CE, persistent minibatch gradients, or derivative-free observed-fiber projections |
| `EndpointBuilder` | A checked spectral endpoint on the segment from the phase anchor to the projected iterate, preserving every certified row |

The input side, latent orbit count and full output vocabulary remain separate.
The decoder must assign distinct class names to different latent orbits;
classes outside its image remain part of the vocabulary. Neither proposals
nor checking read numerical values of input tokens or a task name. Output
renaming is supported by renaming the fixed decoder with the labels.

The checker implements the finite conditions of CM18. Its reach relation is
recomputed from the proposed phases and observations, not supplied by the
proposal. It checks the labels only at reached rows and columns, just as Lean
does. Answers at other cells are explicitly absent. A proposal that reaches
the entire table also checks every observed label. Reported answers are
conditional on the target family even when all observations fit.

The default proposal budgets are 4096 examined shared-column rectangles and
16384 row-difference candidates. Every attempted anchor/difference and its refusal
or acceptance status is retained, together with work counters and the stopping
reason. It retains the largest verified region found, stops early on
complete coverage, and reports exhausted or bounded search explicitly. A
search failure does not establish inconsistency or ambiguity. This is not
an executable extraction of the classical Lean `recover` specification, nor
a completeness or running-time theorem for native propagation.
The certificate search is discrete preprocessing. The resulting constrained
prediction domain is convex; no convex optimization formulation of this search
is claimed.

## Findings

Twenty native tests pass. They reproduce the folded held-out witness, the paired
three-cell ambiguity, and CM18's wrong global minimum. The latter satisfies
the original numerical spectral budgets and has the same observed CE as the
correct anchor, while assigning `2/3` of decoded mass to the wrong unseen label.
The checker identifies the correct label from the eight training cells.

The tests also reject forged phases, missing rectangle cells, wrong reached
labels and unsupported decoder fibers; check partial coverage separately
from full training fit; and verify categorical input permutations, output
renaming, repeated observations and every batch size of the fixture. An
independent enumeration checks every accepted missing answer against every
compatible phase completion for all 243 gauge-normalized three-state,
three-by-three phase assignments and each of their nine single-cell holdouts.
Malformed dimensions are refused before arithmetic or allocation can overflow.

All three modes exclude the wrong global minimum throughout the fixture updates.
On a partial certificate covering nine of sixteen cells, they also learn an
uncovered training cell that the phase anchor initially gets wrong. Endpoint
repair retains the certified rows, and an independent direct complex Fourier
sum checks the actual returned table's bounds. Complete solver states roundtrip
and replay identically in every mode; malformed batches and mismatched instance
states are refused without changing the solver.

The parent solver's uniform mixture can erase certified rows. The new endpoint
uses a checked phase anchor and at most 40 bisection rounds, auditing each returned
candidate after interpolation. A positive iterate weight is not guaranteed.
Certified rows are preserved exactly in this unique-orbit decoder. With partial
coverage, the anchor can disagree with uncovered training observations, so a
forward-only endpoint need not retain those additional observed-fiber constraints
at finite precision or finite iterations. Its measured training defect and CE
must be reported; they are not assumed to vanish. Forward-only performs no
gradient evaluations. Full/mini use the unchanged full-vocabulary CE, and mini
uses a persistent gradient table rather than an unbiased stochastic gradient.

These fixtures are separate from the original-data results below. The frozen
parent source and running cohorts are unchanged.

## Original results, 2026-11-12

The committed producer `d7164082` completed all 24 planned cases in
`runs/cohort-v1`, under source fingerprint
`e4e21f0f3eca67d10b979bd4ed9085df0fa7bd615b7799cf72172647e5b662f3`.
All 24 discovery results replay. The three returned models also reproduce every
update and pass the independent audit of all 192 nonconstant Fourier frequencies,
decoded probabilities, ordinary CE and original split metrics.

| Original task | Full-batch | Minibatch | Forward-only |
| --- | --- | --- | --- |
| Absolute difference | 100% held-out | 100% held-out | 100% held-out |
| Permuted addition, addition, subtraction, division, multiplication, sum-cube, sum-fourth-power | Uncertified | Uncertified | Uncertified |

Absolute difference has 4,704 training and 4,705 validation pairs. One proposed
certificate, with 5,082 phase tests, covers all 9,409 table cells. Every mode
answers all validation pairs correctly and attains the ordinary-CE lower bound
up to rounding. Its update-zero endpoint already does so: identification comes
from the training-only certificate, not from discovering these answers during
optimization. The frozen positive-update stop is first checked at update 19.
Full-batch evaluates 89,376 training gradients, minibatch 4,704 and forward-only
zero. Returned iterate weights are one; the largest observed one-hot defect is
`1.12e-16`.

Every other task reaches the 4,096-rectangle search cap without an accepted
certificate, in all three modes. These 21 outcomes are retained with every
attempt and its refusal status. They have no new prediction candidate and are
not counted as passes. Bounded refusal does not prove the target unrepresentable
or unidentifiable. This extension therefore passes **one of eight tasks in each
mode**, or three of 24 task/mode cases, and supplies no new division result.

The largest measured process RSS, including all training and replay processes,
is 134,556 KiB (about 131 MiB), below the enforced 64 GiB address-space cap.
The conservative explicit-buffer bound is 876,832,336 bytes. Frozen producer
hashes and a repeated summary check agree. Generated evidence remains local in
`runs/cohort-v1` and `.kanban/artifacts/201/cohort-v1`.

Training-only integration with earlier candidates remains pending the complete
frozen paired and decoded-parent evidence; those runs are still in progress.
Task #201 stays open for that comparison. This result is neither a complete
35-task short/long pass nor a new transformer comparison.

## Frozen original follow-up

`protocol.json` fixes the same eight original pair tasks and three training modes
as CM16, with 128 full-batch, 512 minibatch and 128 forward-only update caps. The
source constructors, splits, batches, shuffle seed and full output vocabulary
are reused. Proposal budgets are the defaults above and snapshot repair uses
12 bisection rounds. Positive-update stopping uses only training CE excess.
Held-out metrics are computed after the endpoint is fixed. A refused search
produces an explicit result with no model; partial certificates are retained.

The five cohort tests use only small fixtures. They exercise all three modes,
the original minibatch tail rule, refusals, full update replay and raw numerical
audits, and reject altered certificates, models, gradient states, traces and
metrics even when their file hashes are updated. Selection tests deliberately
choose a worse held-out result when its training CE wins. No original data is
built or evaluated by these cohort tests.

The replay repeats every training-only proposal and every update from the seeded
initial state, compares the saved state and endpoint, then independently audits
the saved coordinates with the CM16 raw auditor. Setup, snapshots and the fresh
full-gradient diagnostic have separate work ledgers. Integration requires the
complete frozen paired comparison and its decoded predecessor, retains them
whole, and compares candidates using training CE alone. The all-35 short/long
goal and the existing transformer comparison remain separate obligations.

## Reproduction and resources

New runs follow the user's 2026-11-12 limit of **64 GiB**, enforced as
68,719,476,736 bytes of process address space by `limited.sh`. Explicit buffers
are bounded before allocation. This upper limit is not a claimed peak usage.
Generated outputs belong in ignored root `runs/` and `.kanban/artifacts/201/`.
Use one native numerical test process and one BLAS thread:

```sh
cargo fmt --manifest-path experiments/convex_stress_certified/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash experiments/convex_stress_certified/limited.sh \
  cargo test --offline --locked --manifest-path experiments/convex_stress_certified/Cargo.toml \
  -- --test-threads=1
cargo clippy --offline --locked --manifest-path experiments/convex_stress_certified/Cargo.toml \
  --all-targets -- -D warnings
bash -n experiments/convex_stress_certified/limited.sh
bash -n experiments/convex_stress_certified/run_cohort.sh
```

After committing and freezing the producer binary and its SHA-256, run a fresh
ignored destination. The launcher applies the cap to discovery, training, replay
and summary. It preserves started cells and refuses to resume or overwrite them.

```sh
cargo build --release --offline --locked --manifest-path experiments/convex_stress_certified/Cargo.toml
CERTIFIED_PRODUCER_COMMIT=$(git rev-parse HEAD) \
CERTIFIED_BINARY=/absolute/path/to/frozen/convex-stress-certified-v1 \
  bash experiments/convex_stress_certified/run_cohort.sh \
  experiments/convex_stress_certified/runs/cohort-v1
# Once the complete unchanged parent evidence exists:
/absolute/path/to/frozen/convex-stress-certified-v1 integrate \
  experiments/convex_stress_certified/runs/cohort-v1 \
  experiments/convex_stress_paired/runs/cohort-v1 \
  experiments/convex_stress_decode/runs/cohort-v1
```

Spectral checks are floating-point SVD checks, not interval certificates.
No iterative convergence, numerical tolerance-to-recovery theorem or coverage
of all original stress tasks is claimed.
