# Conservative extension using a shared-role head

This component applies the proved CM39 gate to the actual CM40 augmented kernel.
The prior raw-kernel head predicts wherever it has support and wherever both
heads refuse. The shared-role head predicts only on newly supported inputs.
Both complex heads remain stored. This gives a concrete candidate for extending
coverage while preserving every prior supported prediction and the original
training objective. It has no original benchmark result.

The source ports `convex_stress_coverage_extension` at `08e4108b` and imports its
checked combined resource bounds. It replaces the CM35 supplemental certificate
with the CM40 certificate from `convex_stress_shared_roles` at `5d4eee84` and
retains explicit original record IDs. The complete augmented matrix is checked;
a restricted code is never presented as the complete unrestricted kernel.

The mathematical sources are CM39 in `review/flat_spectrum.md`, proved in
`ConvexDepth/CoverageExtension/Basic.lean` at `2ca6d2cf`, and CM40's
`CoupledMoments/SharedRoles/Identification.lean` at `ec5f839d`. The former proves
exact logit, batch-loss and derivative preservation by this gate. The latter
proves preservation of existing code matches when adding shared-role equations.
Its assumptions on the unknown target and declared readout remain explicit.
The component checks actual support inclusion over the entire finite input
range; it does not infer target correctness from flatness or observed fit.

| Prior supports input | Shared head supports input | Routed complex logits |
| --- | --- | --- |
| Yes | Either | Prior |
| No | Yes | Shared head |
| No | No | Prior |

Construction binds identical original data, dimensions and prime, both checked
certificates, exact support codes, observed dictionaries, actual Fourier
coefficients and all original records. Repeated inputs retain their separate
IDs and weights. Loading serialized state additionally requires trusted external
bindings. Both original optimization histories must be replayed by the later
comparison adapter; component feasibility alone does not establish provenance.

The derivative uses ordinary CE over the full raw vocabulary. A batch consists
of positions in the immutable original record list, with every repetition and
final tail retained. Its logits and prior derivative equal the existing head's
values; every supplemental coefficient has zero derivative. No new optimizer
step occurs. Finite differences refer to the ambient real and imaginary
coordinates, not feasibility of arbitrary perturbations under the native table
constraints.

Six native checks exercise strict coverage on a three-diagonal example with raw
aliases, every prior/joint-refusal logit, independent complex products, actual
full/minibatch/forward histories, nonzero gradients, repeated records and a 3+1
tail. They test every prior and supplemental coefficient, changed serialized
parameters and certificates, original identity/order mutations, combined
resource overflows and asymmetric raw-label predictions for long prefixes.
The prefix uses the last token as row and the penultimate token as column,
with supplied lengths from 2 through 4096 checked before indexing.

Both coefficient budgets and all stored heads are charged. The combined
conservative head, certificate and record forecast must fit 56 GiB, numerical
processes run under 64 GiB, and head audits and derivative batches retain a
32-billion-operation forecast limit. Forecasts are separate from measured RSS.
Original evaluation must also retain verification, routing and serialization
costs and both source constructors.

Build and verify from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
manifest=experiments/convex_stress_shared_roles/extension/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path "$manifest" -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path "$manifest" --all-targets -- -D warnings
```

After committing, freeze the clean producer source graph and test executable:

```sh
bash experiments/convex_stress_shared_roles/extension/freeze_source.sh \
  "$PWD" "$(git rev-parse HEAD)" \
  /run/media/misha/storage/transformer-runs/freezes/shared-role-extension-v1
```

The prospective original protocol is fixed in `protocol.json`: extend only an
already selected raw complete-kernel head, using its available same-prime,
same-mode shared-role endpoint with positive new coverage. This is a structural
extension with unchanged observed CE, not a held-out selection rule. A separate
adapter must retain both 105-choice arms, all 35 tasks, the unchanged baseline
records and all failure and cost evidence. Existing frozen comparisons and
components remain unchanged. The gate's extra support alone is not a full pass.
