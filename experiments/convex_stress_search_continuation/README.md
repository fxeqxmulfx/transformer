# Continue observed-compatible count search

Can training-only search improve its retained programs by continuing after its
first completed fit? Task347 stops at its first fitting streamed query and keeps
12 programs of56--64 AST nodes on the1370-observation depth3 training pool. All
fit observed labels, but that does not identify their unseen answers. This
separate task350 probe investigates search quality without reading heldout data.

| Item | Continuation protocol |
| --- | --- |
| Constraints | Unchanged CM62 exact threshold joins, both index orientations, original observed-prefix closure and count bounds |
| Cache | Unchanged negative-result cache: only no-larger complete observed signatures can reuse a refusal; smaller arrivals and evicted entries are retested |
| Successful candidates | Continue the stream, retain up to32 distinct ASTs ordered by actual node count then syntax; do not merge different successful programs using only observed signatures |
| Budget stop | Keep previously completed fitting roots and record the stop reason and incomplete coverage; unexpected interpreter, shape and size errors remain fatal |
| Limits | Same40T search operation budget,262144 anchor classes,65536 negative-cache entries,127 AST nodes and64 branches; reserve final interpreter checks inside the work budget |
| Final validation | Both interpreters must agree on every original observation; each retained program must induce a consistent raw-label map, then the existing joint-decoder compatibility check runs |

Lean was checked first. CM62 `countFit_left_bounds`, `countFit_right_bounds` and
`cached_refusal_preserved` justify the unchanged logical tests. The streaming
module builds and its axiom audit contains only the standard axioms. No new
pruning or equivalence rule is introduced. Ranking is a bounded native scheduling
policy, not a new Lean theorem of global minimum syntax or unseen-label accuracy.
CM63's conflicting-completion example remains applicable. Smaller syntax is an
experimental prior; a simpler observed fit can still predict the wrong answer.

This probe consumes only the source training pool. Discovery receives vocabulary,
causal input prefixes, original observed labels and fixed resource settings; no
task identity, target AST, generator seed or heldout answer reaches the search.
It reports candidate counts, sizes, first-fit work, final work, interrupted
coverage and independent fit checks. It produces no cross-entropy/accuracy score,
changes no frozen347/349 sources, and adds no catalog coverage. The decoder
compatibility check's separate work/buffer forecast is retained in its report.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/search-continuation-v1
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_search_continuation/Cargo.toml -j2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_search_continuation/Cargo.toml --all-targets -j2 -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_search_continuation/Cargo.toml -j2
# Commit and freeze before the training-only measurement:
bash experiments/convex_stress_search_continuation/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-search-continuation"
/usr/bin/time -v -o FRESH_RESOURCES prlimit --as=68719476736:68719476736 -- FROZEN_BINARY suite-crasp-depth-3 FRESH_JSON
```

Independently rebuild the frozen source and repeat into a fresh file. Compare
the full JSON reports, preserving work stops as well as fitted candidates.
Keep generated evidence in persistent ignored storage. A later model comparison
must freeze its own protocol and retain the full13-case/six-endpoint cohort.
