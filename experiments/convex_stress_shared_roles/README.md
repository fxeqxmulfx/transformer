# A complete kernel with shared input roles

Can the shared categorical encoder proved in CM40 supply a bounded complex
head without selecting an arbitrary latent direction? This Rust component
constructs the actual restricted kernel and trains its fixed complex features
with ordinary raw-vocabulary cross-entropy. The Lean source is
`ec5f839dd7584fc07765373132d255286f53f052`, task 271. The earlier training-only
diagnostic remains in `../convex_stress_role_sharing/`.

The learner compacts observed row and column names separately, groups observed
raw classes through the declared full-vocabulary output partition, and aligns
identical token IDs across the two input dictionaries. It retains every
original equation and the two original shift gauges, then adds every aligned
row-minus-column difference relative to the first alignment. The common offset
remains free. Token IDs supply equality and categorical cell indexing; no
arithmetic answer rule is used. A new certificate records this augmented matrix.
It is not represented as an unchanged CM35 certificate.

The finite-field checker independently replays the saved row operations. The
complete-basis audit checks all original and additional rows, the RREF nullity,
and an invertible free-coordinate matrix. All basis directions are retained.
Distinct observed answer groups must have distinct full vector codes before
raw class expansion. Missing alignments, incompatible groups, incomplete
elimination, exhausted budgets and unsupported queries remain explicit.

Original training records have separately bound unique IDs, order and
multiplicity. They must cover exactly the distinct equation dictionary, with
the same labels. Repeated inputs remain separate records. The Fourier head
uses the existing actual-range blocks, separate real and imaginary coefficients,
raw margin 24, observed-input states and first-matching-class defaults. Training
retains the unchanged CE derivative and projected table update; its encoding
and derivative now accept repeated records with their actual denominator.

| Mode | Updates | Record weighting |
| --- | --- | --- |
| Full | One projected update | Every original record |
| Mini | One epoch at the declared original batch size | Every record, including the actual final tail |
| Forward | Zero | Same feasible initializer; zero CE gradients |

The model's actual Fourier Gram matrices, singular values, conjugate transform,
coefficient budgets and complex logits are audited. Original-record CE is
computed separately with complex products, because the distinct equation
dictionary is not the training distribution. Saved histories replay through
an independent complex derivative and projected-state reconstruction.
An immutable checked endpoint supplies inference for prefixes of 2–4096
valid input tokens. It distinguishes matched support, unmatched active codes,
unknown axes and exact observed inputs.
The original suite's cell order is retained: the last token is the row and
the penultimate token is the column. An asymmetric raw-alias fixture checks
this order at short and long prefix lengths.

The head forecast already reserves its initializer, three endpoints, replay,
gradients and serialization. The component additionally charges the entire
matrix-certificate forecast and 128 MiB for bounded records, histories and
binding metadata. Their sum must fit 56 GiB before head allocation. Numeric
processes also run under the user's 64 GiB limit. Each head audit and training
forecast retains the 32-billion scalar-operation cap. Matrix construction and
certificate verification are separate costs and must be included in original
run timing; these forecasts are not measured RSS or an exact allocator proof.

Nine component checks cover 13 exhaustively enumerated small systems, full-basis
changes, different gauge anchors, a nonzero common role offset, strict new
coverage and characteristic-two refusal. They also cover incompatible group
codes, actual raw aliases, full/mini/forward histories, repeated original IDs
versus repeated inputs, short tails, all real/imaginary finite differences,
observed-table finite differences, long prefixes, changed certificates and
states, and resource refusals. Gradient differences are checked at a feasible
head with nonzero derivatives. Coefficient perturbations check the ambient
derivative; they do not assert feasibility inside the narrower affine defaults.

The Lean and native witnesses also retain an independent-role target that fits
the same observations but disagrees with a supported new answer. Global target
realizability and readout compatibility are not established by observed fit,
class separation or flatness. Raw aliases remain unidentified. Component tests
produce no new benchmark score. The last complete selected counts remain
34/35 full, 34/35 mini and 33/35 forward; the conventional transformer remains
33/35 primary. DiffGPT's recorded 35/35 for each of seeds 0, 1 and 2 remains a
separate historical result. Frozen comparisons 255, 261 and 267 and component
270 are unchanged.

From the repository root, build and check with native tools:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
manifest=experiments/convex_stress_shared_roles/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path "$manifest" -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path "$manifest" --all-targets -- -D warnings
LAKE_ARTIFACT_CACHE=false prlimit --as=68719476736:68719476736 -- \
  lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles
```

After committing and passing checks, freeze the component into a fresh path:

```sh
freeze=/run/media/misha/storage/transformer-runs/freezes/shared-roles-v1
bash experiments/convex_stress_shared_roles/freeze_source.sh \
  "$PWD" "$(git rev-parse HEAD)" "$freeze"
prlimit --as=68719476736:68719476736 -- \
  "$freeze/shared-roles-tests-v1" --test-threads=1
```

The source graph, locked dependencies, toolchain and test executable are
preserved for review. Generated outputs remain local on the storage disk.
`protocol.json` fixes this component's assumptions and limits. Original-data
training needs a separate frozen cohort and original-record binding. Any
benchmark improvement needs a separate complete 35-task comparison retaining
both prior initializer policies, all modes, baselines, costs and failures.
