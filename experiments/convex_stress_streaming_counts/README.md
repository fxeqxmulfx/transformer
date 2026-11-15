# Streamed count-program discovery

Task347 asks whether the depth3 failure retained by task344 comes from storing
both latent banks under a 262144-class cap. The repair keeps one anchor bank
indexed and streams opposite candidates through every known observed count
constraint. The old experiment and its complete refusal history stay intact.
Search receives only causal training inputs and observed labels. A task name,
target AST, generator seed or heldout answer is never a search input.

CM62 (`44a0d5d9`) proves exact joins in either orientation, threshold clipping,
streamed existence relative to fixed anchors, complete-signature congruence,
and size-aware reuse of cached refusals. Its 17 theorems pass the full Lean build
and direct standard-axiom audit. The index scanner correction `c2e63bee` fixes
an identifier-suffix false positive without changing any Lean claim.

| Component | Declared behavior |
| --- | --- |
| Shallow search | Preserve the task344 depth1/2 implementation and outputs |
| Anchor | Prefer a complete bank; if both truncate, retain the left bank and explicitly report conditional coverage |
| Query | Use a complete retained bank when available; otherwise enumerate the entire same bounded grammar without a query-class retention cap |
| Selection | First fitting branch, then first fitting query in the declared enumeration; retain at most 32 matching anchors ordered by actual root node count and AST |
| Cache | At most 65536 full signatures, storing the least already-tested syntax cost; retest cheaper arrivals; clear on capacity and repeat any evicted work |
| Bounds | Same 40T search work, 56GiB admitted buffers, 64GiB process address space; cache storage is added before search |

All known constraints are intersected. Unknown labels and inactive gates add
none. Final programs must fit every original raw-label record and agree with
the independent input interpreter. A complete stream against truncated anchors
still does not prove exhaustive grammar coverage. Discrete program discovery
remains outside the downstream CM57 complex head's convex ordinary CE claim.

Six native tests compare both index orientations to independent exhaustive
interpretation, with 73 anchors, negative offsets, both label orientations and
padded bitsets. They check inactive gates, work/storage refusals, exact shallow
compatibility, complete small-grammar signatures and least syntax costs, a fit
beyond an eight-class truncation, cache eviction, and the actual 127-node cap:
a 121-node equivalent child is refused while the one-node arrival fits.

```sh
cargo fmt --manifest-path experiments/convex_stress_streaming_counts/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_streaming_counts/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --all-targets --offline --locked \
  --manifest-path experiments/convex_stress_streaming_counts/Cargo.toml -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --bin discover --offline --locked \
  --manifest-path experiments/convex_stress_streaming_counts/Cargo.toml
prlimit --as=68719476736:68719476736 -- \
  experiments/convex_stress_streaming_counts/target/release/discover \
  suite-crasp-depth-3 FRESH_TRAINING_ONLY_JSON
```

Before a development probe, commit its source and run `bash
experiments/convex_stress_streaming_counts/freeze.sh FRESH_SNAPSHOT
experiments/convex_stress_streaming_counts/target/release/discover` from the
clean worktree. The snapshot binds all local Rust dependencies, literal include
files, Lean sources and audits, protocol source, binary and compiler hashes.
Run its saved `discover` with `/usr/bin/time -v` and the address-space cap,
keeping JSON, logs and resources outside the worktree on the storage disk.

The training-only probe and complete frozen 13-case heldout comparison remain
pending. Verified catalog coverage stays 77/165, the separately reproduced
cohort union on seed 1. This implementation alone adds no accuracy credit.
