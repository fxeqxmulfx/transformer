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

The frozen producer is `f5fb5f43bcb34bc088211950ec132fb51d5473b1`, stored at
`/run/media/misha/storage/transformer-runs/freezes/search-continuation-v1`.
Original and independently rebuilt runs both exit zero and produce identical
complete JSON reports, SHA256
`07617c92e2814bb9c0e93aad5bda8653ccac675f5609ef5c8fee8f555ec012f5`.
The reports and measured resources are retained in `search-continuation/` under
the same persistent storage root. Both runs use a tighter 8 GiB address-space
cap and peak near 1.52 GiB RSS; wall times are 19:36.52 and 19:58.66.

| Search outcome | Independently repeated finding |
| --- | --- |
| Observed fit | All 1370 labels over 883 closed prefixes fit; both interpreters and the separate joint decoder check agree |
| Successful queries | Five queries offer 60 completed fitting roots; the declared ranking retains 32 distinct ASTs |
| Smallest program | 56 nodes at first fit and at the final stop; no smaller program was found |
| Final charged work | 39999999938512, including the 11678318592 reserved final audit, below the unchanged 40T limit |
| Stop | Work cap reached inside branch 12, with truncated anchors and incomplete remaining search; completed fits are preserved |
| Heldout evidence | Zero heldout pools read and zero accuracy measurements; no catalog credit |

The independent build passes all four native controls and strict Clippy. Its
different checkout path changes executable bytes, so only the full behavioral
report is asserted byte-identical; both executable hashes are preserved in
task350's review evidence. The enlarged retained set is a candidate for a later
frozen comparison. This probe does not establish that it improves raw accuracy,
exact accuracy or cross-entropy.
