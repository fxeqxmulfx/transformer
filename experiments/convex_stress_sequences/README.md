# Additional GPT sequence stress tests

This experiment extends the reviewed 165-case inventory at `07a39ac2` with
MQAR, two-hop lookup, Dyck prefixes and alternating blocks. The user's target
is at least 99% held-out raw-token and exact-sequence accuracy per task. Retain
each original 95% success flag, ordinary cross-entropy and longer probes as
separate measurements. The first change implements and checks the native data
adapter; no new model result is claimed yet.

The four definitions and split sizes are fixed by
`../convex_stress_catalog/protocol.json`. The original reference sources are
pinned by that package's `REFERENCE_SOURCES.sha256` and preserved under
`stress-catalog/inputs-v1` on the storage disk. Historical Python is read only.
MQAR and alternating-block helpers port the already checked Rust generators;
lookup and Dyck port `synthetic/retrieval.py` and `synthetic/prefixes.py`.
Random draw order, source identity JSON, numeric token layout and supervision
masks follow those sources. A zero MQAR overwrite count is omitted from its
RNG identity, as the source requires.

| Case | Input lengths | Longer probes | Training / validation / test examples |
| --- | --- | --- | --- |
| MQAR | 24 | 48, 96 | 128 / 32 / 64 |
| Two-hop lookup | 24 | 48, 96 | 128 / 32 / 64 |
| Dyck prefixes | 12–16 | 32, 64 | 128 / 32 / 64 |
| Alternating blocks | 8–16 | 32, 64 | 128 / 32 / 64 |

Every longer probe has 32 validation and 64 test examples. Retain clean
training rows and each nonempty novel-validation subset. Empty novel subsets
are recorded explicitly; primary selection uses original validation when its
novel subset is empty. All four are prefix-labeling tasks, so the input is
supplied and only designated targets are scored. They are not free-generation
tasks. Later generative cases must preserve their own generation protocol.

Four native tests pass. Reused generators reproduce ten frozen original35
split fingerprints across key-value binding, depth-two and depth-four controls,
including the length-128 probe. Independent token-table and prefix-count
oracles check every new example, target and change mask. Lookup's paired
examples retain token counts while changing answers. Checks also cover novel
input exclusion, deterministic regeneration, size refusals and scoring: a
199/200-token fixture has 99.5% raw accuracy but only 50% exact sequences.

The individual historical runs' fingerprints and checkpoints are unavailable
at their recorded paths. The retained snapshot contains their metrics but not
those fingerprint values. New MQAR/lookup/Dyck/blocks hashes are therefore
reported as source-derived fingerprints, without claiming byte-for-byte
reproduction of unavailable historical artifacts. Conformance to the frozen
source, existing35 golden hashes and independent oracles is recorded separately.
Native tensor hashes cover every produced split and support later replay.

The adapter accepts no predictor or model coefficients. Task names and answer
rules stay within its data modules. The subsequent comparison must construct
models from training records only, preserve full/minibatch/forward modes and
complete complex products, seal states before evaluation, and retain every
bounded miss or refusal. Extend Lean before claiming a new model guarantee.
The original35 comparisons and their conventional/DiffGPT baselines remain.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_sequences/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_sequences/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_sequences/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo run --release --offline --locked \
  --manifest-path experiments/convex_stress_sequences/Cargo.toml -- prepare FRESH_METADATA.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
```

Generated metadata, data, models and reports stay ignored on the storage disk.
