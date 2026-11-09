# Geometry persistence during continued training

Can a bound using the recorded AdamW updates certify the useful geometry
through the low-rate hold, and can old prediction certificates survive finite
parameter changes? The [frozen protocol](PROTOCOL.md) tests all three existing
eight-seed cohorts, every hold update, and 192 fixed endpoint windows.

The study is in progress. It separates observed-path guarantees from a
predictive theorem about unknown future gradients, and separates readout
energy concentration from full nonlinear prediction certificates.

Run from the root with the pinned native dependencies already available:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target"
cargo test --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin geometry-persistence
cargo clippy --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin geometry-persistence --tests -- -D warnings
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin geometry-persistence
"$CARGO_TARGET_DIR/release/geometry-persistence" \
  experiments/geometry_persistence/runs/reproduction \
  experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01 \
  experiments/geometry_dynamics/runs/support-01
"$CARGO_TARGET_DIR/release/geometry-persistence" verify experiments/geometry_persistence/runs/reproduction
```

The output directory must be new. The artifact verifier checks all 192
windows and rereads all 24 complete traces. The binary includes the pinned
Kernite source adapters but this postprocessing performs no GPU work.
