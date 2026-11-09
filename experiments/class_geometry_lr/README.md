# Higher learning rates with the main-model AdamW/WSD recipe

The user suggested a higher learning rate during task #200. The
[protocol](PROTOCOL.md), frozen at `637d3a1e` before candidate measurements,
compares peak rates 0.003 and 0.01 with the completed 0.001 arm. Each arm uses
all eight paired seeds, 10000 updates and all 31 geometry probes. Only the
peak changes: beta2=.95, decay=.1, independent-coordinate clipping at 1,
WSD fractions and final LR1e-8 remain fixed. No architecture change or early
success stopping. The constant-rate task #199 study remains separate.

| Arm | Native build feature | Warmup/stable peak | Final LR |
| --- | --- | ---: | ---: |
| Main-model baseline | None | 0.001 | 1e-8 |
| Three-times peak | `lr003` | 0.003 | 1e-8 |
| Ten-times peak | `lr01` | 0.01 | 1e-8 |

The shared native source is in `../class_geometry_wsd`; the feature arms are
mutually exclusive, embed this protocol, record their actual peak, and include
that value in their source fingerprint. Kernels and all other update rules
are shared unchanged. Native tests, strict clippy and formatting pass for both
new builds. GPU preflight and complete paired outcomes are pending.

From the root, using a CUDA-enabled context, run arms sequentially after the
0.001 cohort completes. Each invocation requires a fresh output directory:

```sh
for arm in lr003 lr01; do
  export CARGO_TARGET_DIR="$PWD/.lake/class-geometry-$arm-target"
  cargo test --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features "$arm"
  cargo clippy --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features "$arm" --all-targets -- -D warnings
  cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features "$arm" --bins
  "$CARGO_TARGET_DIR/release/rotate90-complex-kernite" preflight
  "$CARGO_TARGET_DIR/release/rotate90-complex-kernite" \
    /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
    experiments/class_geometry_wsd/runs/audit-02/report.json \
    "experiments/class_geometry_lr/runs/$arm-01"
  "$CARGO_TARGET_DIR/release/verify" "experiments/class_geometry_lr/runs/$arm-01" \
    experiments/class_geometry_wsd/runs/audit-02/report.json
  "$CARGO_TARGET_DIR/release/analyze" "experiments/class_geometry_lr/runs/$arm-01"
  "$CARGO_TARGET_DIR/release/compare" experiments/class_geometry_wsd/runs/audit-02 \
    "experiments/class_geometry_lr/runs/$arm-01"
done
```

Generated evidence remains ignored under root `experiments/class_geometry_lr/runs/`
and `.kanban/artifacts/201/`. Preserve failures, all seeds and every probe.
