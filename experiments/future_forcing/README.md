# Future AdamW forcing from current state

Can the current moments and known schedule certify geometry before later
updates are read? The [frozen protocol](PROTOCOL.md) compares a warm-state
moment cone with the existing uniform bound on 144 windows, including
initialization, early learning, the hold, and late cooldown.

The study is in progress. Its exact-real theorem permits arbitrary future
gradients; the native FP32 replay checks separately declared numerical
guards. Neither establishes task alignment or reliable discovery.

## Reproduce

Run from the repository root with the three preserved source cohorts. Construction
and verification are separate processes; the first freezes every forecast before
the second opens trajectory files. Output directories must be new.

```sh
CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target" cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin future-forcing
.lake/geometry-gap-target/release/future-forcing experiments/future_forcing/runs/forecast-01 experiments/class_geometry_wsd/runs/audit-02 experiments/class_geometry_lr/runs/lr01-01 experiments/geometry_gap/runs/settle-01
.lake/geometry-gap-target/release/future-forcing verify experiments/future_forcing/runs/forecast-01 > experiments/future_forcing/runs/forecast-01/verification.json
CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target" cargo test --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin future-forcing
CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target" cargo clippy --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin future-forcing --tests -- -D warnings
```

Construction hashes its source, inputs, and each forecast. Verification checks
those hashes again, checks every declared prefix, and retains failures. The
source ledger includes the training support and checkpoint states; neither later
gradients nor observed update magnitudes enter a forecast.
