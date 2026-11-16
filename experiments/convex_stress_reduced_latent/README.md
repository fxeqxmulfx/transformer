# Exact row elimination for the bounded complex decoder

Can the complete CM64 model retain all compatible programs and every unknown
decoder entry while avoiding the cost of storing anchored copies? This native
component follows the 24 CM65 Lean theorems at `1bb564d8` and the 11 CM66
projection theorems at `8d788115`. It supplies a smaller representation for a
future wider fixed input family. It has no task names, target formulas, data
generators or heldout labels at its model boundary.

| Part | Behavior |
| --- | --- |
| Variables | All shared masses and all raw-class coordinates of every unknown row |
| Anchored rows | Exact copies of their program mass at the observed raw class, zero elsewhere |
| Convex domain | Positive bounded total mass, nonnegative joint coordinates, common row masses; unchanged real feasible set |
| Complex head | The same rounded four-point block-Fourier coefficients as the dense component; an anchored block stores only its observed classes, with exact zeros elsewhere |
| Projection | CM66 selected proposal sums and weights `1 + number of known states`; unknown rows use their complete simplex solver |
| CE | Ordinary full-vocabulary CE from actual complete complex products, with every original record and its multiplicity |
| Policies | Genuine Forward, Full and Mini updates; the caller fixes epochs, rate and batch size before data fitting |
| Predictions | Independent complex products, outward intervals, exact dyadic ambiguity evaluation, genuine tie sets and explicit budget refusals |
| Persistence | Exact binary64 masses/free rows, complete dictionary, version/dimensions and a pinned SHA256; deterministic reconstruction of every head coefficient |

The implementation ports the numerical projection and scoring methods from
`convex_stress_bounded_latent` at `a3959a46`. The full dense component remains
unchanged as a reference. Reconstructed head coefficients and all class logits
are bit-identical on the exhaustive small controls. Summing selected derivatives
in reduced coordinates can change floating-point addition order: full training
trajectories are compared with explicit tolerances, not claimed bit-identical.
The mathematical CE derivative uses the proved Fourier Gram identity, as in
the dense component; finite differences independently check its reduced form.

Fifteen native tests cover all coefficients and all 64 queries of a nonconstant
two-program fixture, known and unknown rows, imaginary coordinates, all raw
classes, exact tie handling, adjacent floating values, and finite work refusals.
Twenty arbitrary off-domain proposals exercise nonuniform unknown rows, unequal
anchor counts, active mass bounds and independent dense KKT/objective comparisons.
Every reduced derivative is checked against finite differences and the full
derivative. Forward/full/minibatch training compares all executed update ranges,
CE and full parameter changes on 130 original records, including duplicates and
the `127 + 3` tail, at learning rates 3 and 1,000,000. Checkpoint controls retain
negative zero and a subnormal, and reject bad dimensions, truncation, trailing
bytes, nonfinite data, checksum changes and rehashed corruption. A coefficient
perturbed by 1e-3 and a missing sparse column fail the independent full-head audit.
Both-disk controls include a shared filesystem, overflow, refusal before file
creation and preservation of an existing checkpoint.

CM68, proved first at `84a455b7`, adds checked UInt16 storage for every original
state and raw label. One reusable row is decoded during iteration; original
indices, duplicate records and short tails stay intact. The expanded APIs remain
available. The packed controls reproduce complete domains, all complex coefficient
bits, CE, gradients, full/minibatch/forward update traces and checkpoint bytes.
Invalid words, shape changes, index ranges and exhausted work budgets are refused.
The wider-family controls independently rederive the original state/dictionary
hashes and repeat the same comparisons for a noninjective three-term decoder.
Packing adds decoding work, so charged operation counts intentionally differ.

The legacy memory formula remains unchanged for frozen probe audits. Packed
admission replaces only `N * (8P + 64)` observation bytes by
`N * (2P + 64) + 8P`, including one decoded row; every unknown raw-class
coordinate and all head/projection buffers remain charged. For the first probe's
193-class constant control, the complete forecast changes from 122,579,278,464
to 89,281,189,304 bytes. This is a forecast, not measured original-run peak RAM
or accuracy. The complete support remains 299,551 candidates.

A separate synthetic storage control has 128 programs, 196 states and 193 raw
classes, with all states anchored at three labels. It retains all 9,683,968
logical real/imaginary head coordinates, stores 150,528 of them, and reconstructs
the rest as exact zeros. Only 128 learned coordinates remain because this
particular control has no unknown rows. Its complete checkpoint is 101,464 bytes.
These are component storage counts, not a measured saving on the current stress
cohort or a new accuracy result. Cases with unknown rows keep every class entry.

Admission accounts for original encoded records, domain indices, complete free
rows, projection/sort scratch, four sparse heads, codec buffers and 2 GiB slack.
The buffer ceiling is 96,000,000,000 bytes inside the user limit of
110,000,000,000 bytes. A shape can be refused before allocation. Work is capped
at 40 trillion charged scalar operations; exact products additionally use an
8,192-bit ceiling and an explicit query limit. The run wrapper applies the
process cap. The guarded `write_file` path checks both disks before creating a
file, every 64 MiB and on completion. It reserves the caller's remaining cohort
forecast, at least one additional checkpoint for a repeat, and 2/16 GiB of
workspace/storage slack. Shared filesystems add the requirements. It refuses
overwrites and preserves partial evidence on failure. The lower-level stream
codec creates no files; callers using it directly must provide equivalent disk
admission and pass the complete cohort forecast to their output writer.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/reduced-latent-v1
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_reduced_latent/Cargo.toml -j2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_reduced_latent/Cargo.toml --all-targets -j2 -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_reduced_latent/Cargo.toml -j2
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ObservedRowAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ObservedProjectionAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/PackedRecordAxioms.lean
```

The component never certifies target realizability or exact-real numerical
projection. Feasibility and KKT tolerances remain explicit. The frozen task351
producer and its repeat are unchanged. A new input family, original-data
training, prospective freeze and complete independent heldout comparison are
still required before any new catalog pass can be credited.
