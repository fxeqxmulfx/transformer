# Original arithmetic data for the expanded stress tests

The GPT catalog runner can now reconstruct all 54 arithmetic and powered
division tasks. The original registry contains the SplitMix64 seeds and
sampling order for the random answer table, the four swapped intervals
and the random half-set. `dataset::build_catalog` ports those generators
in Rust. All twelve additional train and validation tensor hashes match
the historical records, and the other 48 datasets match the frozen port
row for row. Run `cargo +stable test --offline --locked` in this directory
to check the complete catalog and the historical freeze.
[Original task generators](https://github.com/fxeqxmulfx/grokking/blob/308912f2d6b736e4a061e72be8c32c0e72ac687a/experiments/training/src/training/domain/tasks.py)

## Task348 data freeze

Task348 prepares the54 arithmetic and powered-division cases from the original
165-case catalog. It preserves22 original35 controls and32 additional cases.
This executable prepares data only; it trains no model and claims no accuracy.
The existing transformer and complex-model results remain separate references.

| Data outcome | Original35 cases | Additional cases | Total |
| --- | ---: | ---: | ---: |
| Native port with both historical fingerprints verified |22|26|48|
| Explicit unavailable-oracle refusal |0|6|6|
| Complete attempted cohort |22|32|54|

Every supported train and validation tensor matches the previously recorded
SHA256, including original row order, supervision masks, repeated labels and
disjoint half-table membership. Division excludes zero denominators; other
functions retain them. The three power-zero controls remain constant-one data,
including a zero numerator. Integer floor, absolute distance and max retain
their original meanings. The old22 cases also match the independently written,
reviewed native oracle in `convex_stress/src/vendor/arithmetic_data.rs`.

At the time of the freeze, the original registry path under `rust_projects/grokking` was absent locally.
The pinned DiffGPT task inventory retains exact expressions for the supported
functions, and the catalog supplies the parity-cubic definition. Six helpers'
fixed data are unavailable: the random table, four swapped interval sets and
the random half-set. They retain all original fingerprints and GPT outcomes
with an explicit refusal. No seed or replacement table is invented. These
refusals do not establish that a learned model cannot solve the cases.

`references/cases.json` and `references/definitions.json` are required source
inputs extracted from DiffGPT commit `5a37c515639f23037d191b0659a5a0853984328d`.
They contain the original specifications, recorded hashes and historical
metrics, not outputs of this port. Preparation checks their semantic equality
to the complete pinned source files and checks all file hashes in the protocol.
The whole inputs are copied into each generated evidence directory.

The model interface `Training` contains only vocabulary size and ordered raw
training rows. The oracle, case names, formulas, seed and validation answers
are confined to preparation. A future model must freeze its complete producer
and seal all endpoints before scoring; this data port supplies no new
identification or generalization theorem. Existing Lean fixed-feature bounds
and conditional CE claims are unchanged. New mathematical claims require Lean
proofs before implementation. Model runs must retain ordinary CE and all three
training modes with real minibatch multiplicities.

Five native checks cover every historical fingerprint, all22 old-oracle
comparisons, precise tensor masks, unsupported source definitions, altered
labels/criteria, disjointness, integer/constant controls and full source
reconstruction after saved labels and their reported hashes are changed.
The complete driver rebuilds all48 datasets independently and preserves all
six refusals. Verification reconstructs the source data again, not merely
two equal manifests. Dataset files are under1MiB; the conservative1GiB buffer
allowance and64GiB address-space cap apply.

Producer `cf05378c053a7a376c164579f54682d966b3aaae` is now frozen and fully
reproduced. A fresh offline target passes all five tests and clippy and builds
the identical executable, SHA256
`88ba4f5e02b18627884ebbb8e01dcedd53843c39055f8f2fb92a5c3eb88c9ad1`.
All52 artifacts match byte for byte:48 datasets, three complete source inputs
and the full54-case inventory. Verification independently reconstructs every
source row again. The relative manifest SHA256 is
`30be56ecda6b77edaea1958f91e5e15197c348be551d9b673633a2e55d5492bc`.
Original/repeat take4.43/5.03 seconds with9748/10480KiB peak RSS; the final
source recount takes1.73 seconds and10736KiB. All exit0 with no swaps.
Evidence remains under `/run/media/misha/storage/transformer-runs/arithmetic-data/`
and `freezes/arithmetic-data-v1` on that disk. This proves data reproduction,
with26 additional cases ready for model tests and six explicit refusals.
Verified model coverage remains77/165 on seed1.

Recreate the three whole reference inputs with native Git tooling:

```sh
mkdir -p "$INPUTS"
git -C /home/misha/python_projects/diff-gpt show \
  5a37c515639f23037d191b0659a5a0853984328d:research/current_grokking/plan.json > "$INPUTS/plan.json"
git -C /home/misha/python_projects/diff-gpt show \
  5a37c515639f23037d191b0659a5a0853984328d:research/current_grokking/results/arithmetic_seed1.json > "$INPUTS/arithmetic_seed1.json"
git -C /home/misha/python_projects/diff-gpt show \
  5a37c515639f23037d191b0659a5a0853984328d:research/grokking_task_inventory.json > "$INPUTS/grokking_task_inventory.json"
```

`cases.json` is produced by joining arithmetic entries of the first file with
the complete matching `.runs[]` entry of the second, retaining objects
`{case: CASE, history: RUN}` in plan order. `definitions.json` is the third
file's `.arithmetic_function_catalog` object. Their committed hashes bind the
exact required inputs. Reports, generated rows and logs remain ignored on
the storage disk.

```sh
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/arithmetic-data-v1
cargo fmt --manifest-path experiments/convex_stress_arithmetic_data/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_arithmetic_data/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --all-targets --offline --locked \
  --manifest-path experiments/convex_stress_arithmetic_data/Cargo.toml -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_arithmetic_data/Cargo.toml
# Commit the source first, then use fresh evidence paths.
bash experiments/convex_stress_arithmetic_data/freeze.sh "$FREEZE" \
  "$CARGO_TARGET_DIR/release/convex-stress-arithmetic-data"
prlimit --as=68719476736:68719476736 -- "$FREEZE/convex-stress-arithmetic-data" \
  prepare "$INPUTS" "$ORIGINAL" "$(cat "$FREEZE/producer.commit")"
# Independently rebuild in a fresh Cargo target and require identical hashes.
prlimit --as=68719476736:68719476736 -- "$REBUILT" repeat "$ORIGINAL" "$REPEAT"
prlimit --as=68719476736:68719476736 -- "$REBUILT" verify "$ORIGINAL" "$REPEAT" "$VERIFIED_JSON"
```
