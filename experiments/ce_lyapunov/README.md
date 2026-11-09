# CE damping with momentum

Can a quantitative property of the actual partial CE gradient absorb
momentum, and which full-model terms break the resulting potential bound?
The [frozen protocol](PROTOCOL.md) fixes 144 checkpoint replays and separates
the partial-model theorem from the full tied AdamW diagnosis.

The study is in progress. Reliable discovery and long useful future control
remain open; no replay result is a guarantee about unobserved gradients.

## Native commands

Build `ce-lyapunov` from `experiments/class_geometry_wsd/Cargo.toml` with
`cargo build --offline --locked --release --bin ce-lyapunov`. Use no feature
for `lr001`, `--features lr01` for `lr01`, and `--features settle` for
`settle`; preserve each executable before building the next. Set
`CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target"` from the repository root.
Each executable takes `NEW_OUTPUT SOURCE_COHORT`, then `verify OUTPUT`.
Verification rehashes all source and evidence files and reconstructs every
diagnosis from saved arrays without executing GPU updates.
