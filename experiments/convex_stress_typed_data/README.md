# Original two- and three-type bracket datasets

This data-only Rust adapter prepares `suite-dyck-2` and `suite-dyck-3` from
the165-case stress catalog reviewed at `07a39ac2`. It advances the user's99%
per-task goal by freezing the next two original cases. It does not construct
or score a model. The reviewed four-case100% result at `3fde00b5`, original35
comparisons and pending signed-kernel tasks317/319 remain unchanged.

The source definitions are `TypedDyck` in `python/src/lab/domain/tasks.py`,
the synthetic suite in `experiments/synthetic_amsgradw/experiment.py`, and
`benchmarks/synthetic/prefixes.py` in task318's preserved snapshot. Eight
reference hashes are pinned in `reference-sources.sha256`; all match that
snapshot. Python remains read-only reference material and is never executed.
The RNG, canonical JSON/tensor encoding and neutral-gap helpers reuse native
`convex_stress_sequences/src/wire.rs` from the reviewed earlier adapter.

| Case | Raw vocabulary | Original training | Longer probes |
| --- | --- | --- | --- |
| Two bracket types |38 tokens |128 sequences, lengths12–16 |32 and64 |
| Three bracket types |40 tokens |128 sequences, lengths12–16 |32 and64 |

Both use data seed1, maximum generation balance8 and neutral fraction0.25.
Each retains32 validation and64 test examples, clean training, all original
long validation/test splits and novel-validation subsets. Original selection
is novel validation when nonempty. Every supervised prefix is scored; BOS is
ignored. This is not a free-generation task. Original95% exact success stays
separate from the new99% raw-token and99% exact-sequence requirements.

The port preserves draw order, including conditional draws in balanced-word
construction, forced first two opening types, later random type choices,
different-type closing swaps, random-word draws, neutral gaps and pair reversal.
Targets and transition masks are created only in this data adapter. The future
model receives visible prefixes and observed labels without designated bracket
pairs or task identity.

Five native checks cover every original split using an independent adjacent-pair
reduction oracle, matched-pair count preservation and type corruption, persistent
invalid prefixes, stream-prefix determinism, bounds and pinned-input mutations.
The original test target totals are842 and850; longer tests have1984 and4032.
These known totals support source conformance but do not replace missing
historical per-example hashes/checkpoints. New split/tensor fingerprints are
explicitly source-derived; historical artifact identity is not claimed.

The inventory embeds the historical GPT rows without changing their statuses,
metrics or targets. Both were original budget misses: test raw accuracy was
787/842 and782/850, with42/64 and38/64 exact sequences. They use the historical
model commit `01f18f86787511279fe52fe41883495fce0e3105`; these older endpoints
are not a new equal-budget transformer run. The unchanged task324 report is
required by its exact content hash and retained as a complete separate baseline.
Every new model result stays null.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/typed-prefix-data
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_typed_data/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_typed_data/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_typed_data/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_typed_data/Cargo.toml
sha256sum -c experiments/convex_stress_typed_data/reference-sources.sha256
experiments/convex_stress_typed_data/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-typed-data"
prlimit --as=68719476736:68719476736 -- FRESH_FREEZE/convex-stress-typed-data prepare FROZEN_HISTORY TASK324_REPORT FRESH_INVENTORY.json
```

The history is task318's pinned DiffGPT `progress_snapshot.json`; the task324
report hash is `1607addda8c8eafa98c7999b5b8e7e16c3a1dd1396b8cb8fd6c0b889907c8a97`.
The command rejects changed input bytes and an existing output. Keep generated
inventories, copied baselines and source freezes on storage outside the worktree.
Task326 must audit the existing CM53 proof, then implement a generic nested-state
program bank with training-learned raw decoders and evaluate a separately frozen
full/mini/forward comparison. No new accuracy is inferred from preparing data.
