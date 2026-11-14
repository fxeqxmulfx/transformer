# Additional GPT sequence stress tests

This experiment extends the reviewed 165-case inventory at `07a39ac2` with
MQAR, two-hop lookup, Dyck prefixes and alternating blocks. The user's target
is at least 99% held-out raw-token and exact-sequence accuracy per task. Retain
each original 95% success flag, ordinary cross-entropy and longer probes as
separate measurements. The native data adapter was committed at `a3283cd7`.
The comparison at `7a2e2ba4` connects the unchanged CFt4/CFt5 causal-pair
component as a baseline for the expansion. Its four-case run and a fresh
independent run now reproduce all23 JSON artifacts byte-for-byte. None of
these four additional cases reaches its original success or the new99% goal.

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

Task names and answer rules stay within data modules. `adapter::request` accepts
only the vocabulary and training records. Its context limit is the maximum
training input length. The protocol fixes the existing feature order and all
budgets before scoring. This is a measurement of the existing component, not
the full selected original35 ensemble. CFt4/CFt5 and CD8 already prove its
conditional representation, complete complex initializer and fixed-feature
convex CE domain. No new guarantee or unseen-label identification is asserted.

| Mode | Unchanged component policy |
| --- | --- |
| Full | 16 projected CE steps, all supervised training prefixes, rate1 |
| Mini | 256 cyclic steps of64 supervised prefixes, rate0.1 |
| Forward | Conjugate Fourier initializer, zero CE gradients or head updates |

Minibatches wrap in original prefix order; they are not short-tail epoch
batches. The complete real/imaginary head and all vocabulary logits are used.
Discovery and initialization refusals retain all three unavailable outcomes;
an iterative refusal retains independently available modes. Unknown feature
states initially select class0, and absent queries have zero logits. Known
feature-state coverage is not evidence that the selected feature was correct.

The CLI writes and fsyncs complete training states and a SHA256 seal before
any held-out model scoring. `run.sh` completes this stage for all four cases
before scoring any of them. Every split reports original success, raw99,
exact99 and joint99 independently, plus ordinary full-vocabulary CE. Primary,
test and all-short/long-probe attainment are retained separately. A second
class-major complex multiplication checks every scored prediction; replay
reconstructs original discovery, both iterative trajectories, work counts,
actual complex states and all metric objects. Fresh-file writes refuse
overwriting prior results. Process exits, timing, RSS and bounded refusals
remain on disk. The run uses a64GiB address-space cap and the existing56GiB
component buffer gate, with a separate2B scoring-work forecast per endpoint.

Eight native tests pass: the four data checks above and four adapter checks
cover training-only construction, invariance to metadata/held-out changes,
actual full/mini/forward trajectories and independent products, cyclic batch
wrap, conflicting-label refusals, altered coefficients/work, fresh artifacts,
and independent original/raw/exact targets. The original35 comparisons and
their conventional/DiffGPT baselines remain frozen. The separate historical165
GPT snapshot is retained verbatim: its two-layer width64 model used batches
of32 examples and a different optimization budget. These are not matched
compute comparisons or replays of those unavailable historical checkpoints.

The frozen seed1 comparison retains all12 case/mode outcomes: two scored
MQAR endpoints and ten bounded refusals, with zero process failures.

| Case | Full | Mini | Forward |
| --- | --- | --- | --- |
| MQAR | Test raw1/128; exact0/64; CE23.8125000063 | Existing2B training-work cap | Test raw1/128; exact0/64; CE23.8125000062 |
| Two-hop lookup | Candidate trace cap | Candidate trace cap | Candidate trace cap |
| Dyck prefixes | No consistent candidate | No consistent candidate | No consistent candidate |
| Alternating blocks | No consistent candidate | No consistent candidate | No consistent candidate |

MQAR full and forward fit all256 training targets, with CE about6.15e-9.
Their primary validation is0/64 tokens and0/32 exact sequences. Length48 test
is0/128 tokens; length96 test is1/128, both0/64 exact sequences. The ordinary
test has54 known feature states:53 are wrong. All74 unknown-state targets are
also wrong. This directly separates spurious selected features from missing
state coverage; filling only unseen states would leave53 errors. Full training
performs16 gradient batches and4096 gradient examples. Forward has zero
gradient batches, derivative terms and coefficient updates. The minibatch
forecast refusal is not an out-of-memory event.

Lookup exhausts250000 candidate attempts after514056 observation visits.
For Dyck and blocks, all13041 single/pair candidates fail on the first query
group. These observations motivate shared query-relative retrieval programs
and richer generic prefix summaries, not merely a larger search budget. Any
copy decoder is an explicit structural prior beyond flatness. Extend the
Lean account before claiming that it determines unseen raw labels; retain
this failed baseline while testing any repair.

The original run's largest measured RSS is53388KiB (about52.1MiB), well below
64GiB; its model binding forecasts about24.8GiB including worst-case copies.
Artifacts live on the storage disk in `stress-sequences/original-v1` and
`stress-sequences/fresh-review-v1`. The source/executable freeze is
`freezes/stress-sequences-v1` (373 tracked inputs, including Cargo's actual
compiled dependency paths). Both complete reports have SHA256
`65620f0757ae3c57ece3d23e3ed6fb464e05484f9b055dc335e9694f09f7ff7b`.
These measurements add four evaluated cases;126 of the130 additional catalog
cases remain unmeasured by this model. Original35 results are unchanged.

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

# Run only after committing. Freeze source, protocol and release executable.
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_sequences/Cargo.toml
bash experiments/convex_stress_sequences/freeze.sh \
  /run/media/misha/storage/transformer-runs/freezes/FRESH_FREEZE \
  "$CARGO_TARGET_DIR/release/convex-stress-sequences"
bash experiments/convex_stress_sequences/run.sh \
  /run/media/misha/storage/transformer-runs/freezes/FRESH_FREEZE/convex-stress-sequences \
  /run/media/misha/storage/transformer-runs/stress-catalog/inputs-v1/diff-gpt/research/current_grokking/results/progress_snapshot.json \
  /run/media/misha/storage/transformer-runs/stress-sequences/FRESH_RUN
```

Generated metadata, data, models and reports stay ignored on the storage disk.
