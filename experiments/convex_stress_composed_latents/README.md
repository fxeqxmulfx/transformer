# Fixed input compositions for the complete convex decoder

Can a complete finite family of branching and integer compositions supply
compatible input features for the five training sets missed by the previous
family? The output decoder, shared masses, ordinary CE and full complex head
remain the CM64/65/66 model. CM67 at `f1783d04` proves inclusion of every old
feasible tensor for arbitrary added input maps. Its thirteen axiom checks were
rerun before this implementation. The feature family is an explicit structural
prior, separate from the flat-spectrum capacity bound.

| Branch | Complete enumeration | Count |
| --- | --- | ---: |
| Old prefix | Every candidate from the verified task354 family, in the same order | 299,551 |
| Four terms | All four distinct monomials whose coordinate exponents are in `[0,3]`; first coefficient one, other coefficients independently in `[-2,-1,1,2]` | 116,480 |
| Conditional inputs | All 15 predicates and every ordered pair of 178 signed leaves | 475,260 |
| Residue compositions | All 4,767 original pair programs, five integer controls and moduli 2, 3 and 4 | 71,505 |
| Integer digits | Low and high base-vocabulary digits of each of five controls | 10 |
| Total | No deduplication, fitted beam, top-k or surviving-subset shortcut | 962,806 |

The five controls are the input integers `a`, `b`, `a+b`, `ab` and `abs(a-b)`.
A branch tests whether a control modulo two or three is below each possible
nontrivial threshold. Leaves are all 49 old Laurent monomials and the 40
two-term sums over `[a^-1,b^-1,1,b,a]`, with both signs. Every ordered leaf pair,
including equal leaves, is included. Residue composition replaces a program's
remainder modulo a small integer with a control's remainder, then reduces the
result modulo the vocabulary. Digit extraction uses ordinary integers before
the final reduction. These operations and all bounds are fixed before labels.

The model receives only the vocabulary and ordered original rows `[a,b,label]`.
It receives no task identifier, target expression, seed, oracle, supplied raw
decoder or heldout labels. Every compatible candidate has its own complete
decoder block. The selection and materialization engine is reused directly
from `convex_stress_wide_latents` at `e5ef378a`, with this crate's fixed family.
Every rejected candidate retains a two-record collision; every surviving state
and every dictionary entry is audited. Every unknown raw-class coordinate
remains free. The recent CM69 projection symmetry does not replace these rows.

Primary composition uses signed modular sums and ordinary integer operators.
The independent interpreter uses the existing direct-power/Fermat leaf path,
unsigned repeated sums, local residue identities and bounded subtraction for
integer quotients. It never uses the primary power table. The controls check
all candidates on all nine input pairs at prime three, boundary inputs at
larger primes, and independently ranked complete grammar membership. Pairwise
collision tests validate selection without using its first-observation table.
Label renaming, repeated records, contradictory inputs, forged subsets, original
row order, noninjective decoding and free padded states remain covered.

The materialization control compares expanded and packed records in full-batch,
minibatch and forward-only modes, including a singleton tail. It compares every
complex coefficient bit, saved parameter bit, CE and update trace. A constant
1,021-class fixture audits the whole bank and explicitly refuses its oversized
full model. A separate arithmetic forecast charges the largest original
constant probe with every candidate retained. Resource failures never truncate
the family or count as an accuracy pass.

The probe protocol pins the independently verified task354 manifest and its
verification digest. It reads only those original training extracts and keeps
the complete 54-case inventory, including six unavailable source definitions.
All 48 available pools must be attempted in their original order. Reports
preserve every empty support or model-memory refusal. No heldout split is read,
no model is fitted by the probe, and no accuracy credit follows from support.
Any subsequent model comparison needs its own committed producer, complete
training seal, raw prediction replay and independent source rebuild/repeat.

The 40-trillion-operation cap applies to each complete probe. Full model
admission retains the existing 96 GB core-buffer limit inside 110 GB process
RAM. The complete selection report cap is 128 MiB. Both original and repeat
outputs are reserved on both disks, with 2 GiB workspace and 16 GiB storage
slack. A disk or process failure stops the cohort and preserves partial results.
The frozen task351 and task355 comparisons remain unchanged.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/composed-latents-v1
export CARGO_BUILD_JOBS=2
manifest=experiments/convex_stress_composed_latents/Cargo.toml
cargo fmt --manifest-path "$manifest" --check
prlimit --as=51539607552:51539607552 -- cargo test --offline --locked --manifest-path "$manifest" -- --test-threads=2
prlimit --as=51539607552:51539607552 -- cargo clippy --offline --locked --manifest-path "$manifest" --all-targets -- -D warnings
prlimit --as=51539607552:51539607552 -- cargo build --offline --locked --release --manifest-path "$manifest"
LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2 lake env lean -j2 scripts/FamilyExtensionAxioms.lean
# Commit clean source before reading any original training pool:
bash experiments/convex_stress_composed_latents/freeze.sh NEW_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-composed-latents"
bash NEW_FREEZE/source/experiments/convex_stress_composed_latents/probe.sh FROZEN_BINARY WORKSPACE STORAGE NEW_ORIGINAL VERIFIED_WIDE_PROBE WIDE_VERIFICATION_JSON PRODUCER_COMMIT
# Independently rebuild the frozen source in another fresh target directory,
# repeat all 48 pools, and verify all 102 deterministic artifacts:
bash NEW_FREEZE/source/experiments/convex_stress_composed_latents/verify.sh NEW_ORIGINAL NEW_REPEAT NEW_VERIFICATION_JSON
```

The complete original training-only probe and its independently rebuilt repeat
finished on 2026-11-16 at producer `cbde1e2d9fc10813f254982c1f52a9bad5884c18`.
All 102 deterministic artifacts match exactly. All 48 available original
training sets have nonempty compatible support. The old full-model memory
forecast admits 46 and refuses the two large constant controls; those refusals
and the six unavailable source definitions remain in the 54-case inventory.
Both constant controls retain all 962,806 candidates. The 97-class control has
5,112,868 unknown rows, and the 193-class control has 7,324,366. These counts
motivate separate compact storage, not removal of candidates or free classes.

The verified manifest SHA256 is
`86e01a783bfe712e096ce6e6dd3175ec1003d8a27d4062f1d668024879b87f60`;
the verification receipt SHA256 is
`3067b3e4bd17d4666c1c321ed5758557cf5578cc62e679f18e02f67db8fa880c`.
Frozen source and generated evidence remain under storage `composed-latents/`
and `freezes/composed-latents-v1/`. The probe reads zero heldout pools, performs
zero model updates and measures zero accuracy. Heldout accuracy of this added
family remains unmeasured until a separate complete model comparison.
The old zero-block embedding is a representational inclusion; initializing
every added candidate with positive mass can change actual predictions.

All nine library controls and the executable boundary control pass, as do
formatting, strict Clippy and the release build. The complete 193-class
constant probe with 18,528 original rows is charged at 38,699,598,835,200
operations in both original and repeated probes, below the declared 40-trillion
cap. This charged work does not assert that the complete model's buffers will
be admitted or that a fitted model reaches any heldout target.
