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

The training-only probe completed from committed `ee289380` and the snapshot
`/run/media/misha/storage/transformer-runs/freezes/streaming-counts-training-v1`.
Its results are kept under `streaming-counts/training-v1` on the same disk.
It finds 12 programs that fit every one of the 1370 original observed labels,
with agreement between the primary and independent interpreters. The fit is
on branch 12 after 17223763 indexed queries over 883 closed prefixes. The first
11 joins exhaust without a fit; the successful branch streams past the old
right-bank cap while retaining a capped 262144-class left anchor. Its success
is a checked training fit, not an exhaustive search or a generalization proof.

The cache skips 8722818 previously tested queries but correctly retests 858393
cheaper equivalent arrivals. Charged work is 37393912927376, below the unchanged
40T cap. Wall time is 16:59.56 and peak RSS is 1596700KiB (about 1.52GiB), under
the 64GiB process cap. No target AST, formula seed or heldout answer was used.
These 12 programs have not yet been scored on heldout data or credited as a
new catalog success. The frozen 13-case heldout comparison is next. Verified
catalog coverage stays 77/165, the separately reproduced cohort union on seed 1.
This implementation alone adds no accuracy credit.

The complete driver reuses task344's actual joint complex head, ordinary CE,
forward-only initializer, full-batch schedule and 127-record minibatches. It
retains the same categorical control. All 13 original cases and all six
endpoints must be trained and sealed before the first heldout query. Scoring
and replay each reconstruct the complete discovery and every CE update.
Every raw vector, state, score, replay and earlier report enters a manifest;
the verifier rehashes both independently produced manifests. Original success
stays at 95% exact sequences; the requested target requires at least 99% raw
and exact accuracy on every original heldout split in all three modes.

All of task344's history, the independently verified task346 report and manifest,
the earlier history file, data345 inventory and original35 reference are copied
whole and checked against fixed hashes. Catalog accounting uses case identities:
newly passing cases and regressions are both listed. Repeating an earlier case
does not add another success. The transformer reference is retained unchanged.

The 13-case forecast admits at most 263010390589440 scalar work units and
47561448432 buffer bytes per composed case. The complete pair needs about
96.7GiB of disk including two full raw replays, state allowances and retained
histories. `disk.rs` bounds the unchanged JSON schema using the pinned float
formatter's 24-byte buffer, actual escaped split names, all vocabulary logits,
program residuals, original prefixes and result arrays. Serialization tests and
actual file-size checks enforce this bound. All raw evidence remains stored.
The native suite now has 12 checks, including complete six-endpoint construction,
real minibatch tails, exact reconstruction, original rollouts, integer threshold
boundaries and regression accounting.

After committing the complete producer, use fresh paths:

```sh
prlimit --as=68719476736:68719476736 -- cargo build --release --bins --offline --locked \
  --manifest-path experiments/convex_stress_streaming_counts/Cargo.toml
bash experiments/convex_stress_streaming_counts/freeze.sh \
  /run/media/misha/storage/transformer-runs/freezes/streaming-count-programs-v1 \
  experiments/convex_stress_streaming_counts/target/release/convex-stress-streaming-counts
bash experiments/convex_stress_streaming_counts/run.sh \
  /run/media/misha/storage/transformer-runs/freezes/streaming-count-programs-v1/convex-stress-streaming-counts \
  /run/media/misha/storage/transformer-runs/streaming-counts/original-v1 original "$(git rev-parse HEAD)" \
  /run/media/misha/storage/transformer-runs/basis-programs/original-v1 \
  /run/media/misha/storage/transformer-runs/basis-programs/verified-v1.json
```

Rebuild the same producer with a fresh `CARGO_TARGET_DIR`, require identical
binary hashes, then call `run.sh REBUILT_BINARY FRESH_REPEAT repeat ORIGINAL`.
Run `REBUILT_BINARY verify ORIGINAL REPEAT FRESH_VERIFIED_JSON` with the verified
output outside both run directories. Reports are provisional until that passes.
