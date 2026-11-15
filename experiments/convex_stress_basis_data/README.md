# Remaining Basis data

Task345 prepares the eight remaining Basis cases from the original165-case
catalog. This is a Rust data port; no model is fitted and no accuracy is added.
The frozen diff-gpt plan/adapters at5a37c515 and the Basis/domain references at
07a39ac2 specify the distributions. Python files are read as references only;
`reference-sources.sha256` binds the exact tracked sources, all unchanged from
that Basis reference commit. Source files, dependency closure and executable
are frozen before complete inventories are prepared.

| Cases | Original pools and behavior | Available historical evidence |
| --- | --- | --- |
| Hard recall |20000/512/512 rows, lengths48–64,16 writes to8 keys,8 queries; every rewrite changes the current value | One historical test:512 rows,4096 targets,4082 transitions |
| L1–L6 | Each has20000/512/512 rows, training lengths32–64,512-row validation/test probes at128; no neutral symbols; selection at128 | All six were pending; no historical measured fingerprints or denominators |
| Noisy parity |512/128/256 disjoint inputs of1–16 bits; training corruption rate0.2, fixed noise seed2; heldout labels clean | Historical status pending |

Language labels describe membership after every prefix, including the
single-block case's distinct corruption rule. Hard recall samples eight
distinct keys, eight additional writes, then shuffles writes and samples each
rewrite uniformly excluding the previous value. The native stream retains the
original random pair reversal and exact draw order. The input-only novelty key
is the full word for prefix tasks and the prompt for generated parity.

Parity's default32 reserved symbols give68 raw output classes; the suite's
explicit64-symbol variant is different. Noisy labels are addressed by the
original prompt and answer position, never by the current clean/noisy label.
EOS stays structural. A corrupted parity answer also changes its teacher-forced
input before EOS, exactly as the source does. `train_clean` remains a diagnostic
split and must never enter model discovery, fitting, or selection.

Seven native checks exercise full original pool sizes, independent recounting
of every generated label, reverse-scanned latest writes, corrupt masks and
earlier-value counterexamples, source fingerprint compatibility, all noisy
pool disjointness, independently serialized noise addresses, exhaustion of a
tiny domain, and own-output generation. Compatibility uses six immutable
fingerprints from the original35 inputs: depth2/4 at ordinary and128 lengths,
easy recall and parity16. These references are not regenerated from this port.

The inventory retains every old status, including historical `passed` even
when a later final score was lower. Only hard recall has a historical final
test to reconcile. Pending cases remain pending references, with no invented
historical scores. Original success and the new raw/exact criteria are all0.99
for this cohort. The entire344 report, original35 comparison and source history
are separately copied byte for byte; they are never replaced by scalar summaries.

Memory stays bounded before construction: each source pool has at most20000
rows and two million positions, at most six splits are live for one case, and
the fixed largest full dataset has1542144 position slots. Prefix recounting
uses one temporary prefix, while one case's serialization is discarded before
the next. History input is capped at16MiB; retained reports at1GiB each are
hashed/copied separately. An8GiB buffer allowance is conservative for these
fixed shapes; every execution still has the user's64GiB address-space cap.
This data preparation does not solve the existing model's200000-record cap.
Task346 must preserve the complete observations with proved aggregation or
bounded streaming before training the larger language pools.

Reproduce without Python, using fresh output paths:

```sh
cargo fmt --manifest-path experiments/convex_stress_basis_data/Cargo.toml --check
CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/basis-data \
  CARGO_BUILD_JOBS=2 prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked --manifest-path experiments/convex_stress_basis_data/Cargo.toml
CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/basis-data \
  CARGO_BUILD_JOBS=2 prlimit --as=68719476736:68719476736 -- \
  cargo clippy --all-targets --offline --locked \
  --manifest-path experiments/convex_stress_basis_data/Cargo.toml -- -D warnings
CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/basis-data \
  CARGO_BUILD_JOBS=2 prlimit --as=68719476736:68719476736 -- \
  cargo build --release --offline --locked --manifest-path experiments/convex_stress_basis_data/Cargo.toml
```

After committing, `freeze.sh FRESH_FREEZE BUILT_BINARY` captures the producer.
Invoke `convex-stress-basis-data prepare HISTORY BASELINE344_ROOT FRESH_ROOT
COMMIT`, where HISTORY is the pinned `progress_snapshot.json`. Independently
build into a fresh target directory, require an identical binary, then invoke
`convex-stress-basis-data repeat ORIGINAL_ROOT FRESH_ROOT`. Compare all four
retained JSON files. `check-history HISTORY FRESH_JSON` separately reconstructs
the full pools and reconciles available original denominators. All commands
must use `prlimit --as=68719476736:68719476736 --`.
