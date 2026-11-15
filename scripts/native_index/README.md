# Native Lean index generator

Run from the repository root:

```sh
cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml
```

This Rust port of `scripts/index.py` at `22e6bbe2` runs `lake build` and
`lake env lean scripts/Sorries.lean` with `LAKE_ARTIFACT_CACHE=false`, then
generates `INDEX.md`. It preserves the previous scanner's declaration, namespace,
status and line-number conventions. An unplaced build-reported `sorry` is an
error. No Python program is invoked. `-- --check` checks without writing, and
`-- --output PATH` writes a comparison copy.

Validation: `cargo test --offline --manifest-path scripts/native_index/Cargo.toml`,
`cargo fmt --manifest-path scripts/native_index/Cargo.toml --check`, and
`cargo clippy --offline --manifest-path scripts/native_index/Cargo.toml -- -D warnings`.
The first generated index was compared with the committed index: only its
generator attribution changed.

The concluding-`True` check now recognizes the final proposition token rather
than an identifier suffix. Regression fixtures retain detection of `True`,
quantified `True` and `_root_.True`, without misclassifying `lastTrue`, Unicode
identifier suffixes or a separately qualified `N.True`. This corrects the CM62
`double_negated_observed` index entry; its Lean statement and proof are unchanged.
