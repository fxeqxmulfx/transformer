# Four original parity scratchpad datasets

Task333 ports the four `suite-parity-*` cases in `stress_tests_for_gpt.md` to
Rust before task334 extends the learned model. The source is the pinned
synthetic `bits.py`, `base.py`, `splits.py`, vocabulary and original experiment
definitions. Existing Python is reference material only; no Python executes.

| Case | Original answer | Vocabulary |
| --- | --- | --- |
| none | Final parity, EOS | 100 |
| running | Initial EVEN, state after every bit, EOS | 100 |
| running-hints | Initial EVEN, original index and state after every bit, EOS | 229 |
| ones-hints | Initial EVEN, original index and state after each one-bit, EOS | 229 |

All formats share exactly the same underlying bit problems. The split seed
omits scratchpad, hints, symbols and number limit. Every sample consumes a
128-bit format seed before drawing bits, even without hints. That seed's
little-endian integer limbs initialize a separate hint RNG. Two examples share
one drawn length and are randomly reversed, preserving stream prefixes for
odd requested counts. Unhinted fingerprint identities omit number controls.
Hinted identities include them. Original source constraints and upper bounds
are checked before allocating a split.

Seed1 retains 128 training, 32 validation and 64 test examples at lengths8–16,
clean training, length32/64 probes and prompt-based novel validation:11 splits
per case. All targets use the original causal shift and include EOS. Sparse
one-bit scratchpads copy each original index, without renumbering it. Their
generation limit still comes from the full prompt length. Numbers are reserved
after the64 symbol slots, so hinted vocabularies exceed the old328 model's126
class bound. Supporting that vocabulary is a separate model change in334.

The six native checks verify all12 historical test token and transition
denominators, independently recomputed prefix counts, format alignment, pair
ordering, causal masks, novelty, initial EVEN, sparse hints, EOS and bounded
own-output generation. They also exercise invalid layouts and immutable pins.
Historical per-example hashes are unavailable: source-derived hashes and
matching aggregate counts do not certify past byte identity. Each new inventory
records original-style split hashes, padded tensor hashes and full record hashes.

The inventory embeds the complete reviewed328 report unchanged, retaining all
12 additional passes, earlier refusals and historical GPT results. It adds four
data definitions and **zero model evaluations**. The original35 suite remains
separate;118 additional cases still await new-model evaluation. The original95%
exact criterion and requested99% raw/exact goals remain separate. A data/scorer
oracle is never a model feature or a learned decoder.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
cargo fmt --manifest-path experiments/convex_stress_parity_data/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_parity_data/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_parity_data/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_parity_data/Cargo.toml
experiments/convex_stress_parity_data/freeze.sh "$FREEZE" "$CARGO_TARGET_DIR/release/convex-stress-parity-data"
prlimit --as=68719476736:68719476736 -- "$FREEZE/convex-stress-parity-data" prepare "$HISTORY" "$BASELINE328" "$FRESH_INVENTORY"
```

`HISTORY` is task318's pinned `progress_snapshot.json`; `BASELINE328` is
`generative-programs/original-v2/report.json`. Use fresh storage output paths,
freeze committed sources before production and repeat at a separate path.
Compare complete inventory bytes and preserve evidence outside the worktree.
Direct `lake build`, `lake env lean -j 2 scripts/Axioms.lean`, native INDEX
`--check` and forbidden-source scans remain required. This data port changes
no Lean theorem. Generated inventories, logs and binaries remain local.
