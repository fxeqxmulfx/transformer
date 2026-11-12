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
16384 row-difference candidates. All work counters and the stopping reason
are returned. It retains the largest verified region found, stops early on
complete coverage, and reports exhausted or bounded search explicitly. A
search failure does not establish inconsistency or ambiguity. This is not
an executable extraction of the classical Lean `recover` specification, nor
a completeness or running-time theorem for native propagation.

## Findings

Ten native tests pass. They reproduce the folded held-out witness, the paired
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

No original-task proposal, training, held-out evaluation or stress success is
reported by these fixture checks. Solver integration and the frozen original
follow-up remain work under board task #201. In particular the parent solver's
uniform feasibility mixture is not a valid repair here: it can erase the
certified equalities. A feasible endpoint must preserve them as well as the
spectral constraints. The frozen parent source and running cohorts are unchanged.

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
```

Spectral checks are floating-point SVD checks, not interval certificates.
No iterative convergence, numerical tolerance-to-recovery theorem or coverage
of all original stress tasks is claimed.
