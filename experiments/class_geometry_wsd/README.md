# Current main-model AdamW/WSD and class geometry

The user requested the optimizer from `/home/misha/python_projects/gpt`.
The [frozen protocol](PROTOCOL.md) pins its selected main-model native kernels
and recipe, then compares all eight task #199 trajectories with matched
model, initialization, data, sampler, 10000-update budget and geometry probes.

| Setting | Task #199 baseline | GPT main-model transfer |
| --- | --- | --- |
| AdamW betas | 0.9, 0.98 | 0.9, 0.95 |
| Weight decay | 1 | 0.1 |
| Independent gradient clipping | None | Global L2 norm 1, after tie contraction |
| Peak LR | 0.001 | 0.001 |
| Warmup | 10 updates, one-based | 200 updates, zero-based; first LR zero |
| Terminal schedule | Constant | Last 20% cosine decay to 1e-8 |

The actual current FP32 kernels are copied unchanged. The adapter supplies
the existing model's ownership map and moment buffers. Its model has no
learned frequencies, so the main model's separate frequency group is absent.
The schedule formula is unchanged; only its budget allowlist admits 10000.
This is a full-recipe comparison, not an isolated test of WSD's causal effect.

Before candidate training, all 16 GPU preflight cases match the scalar FP32
parameter and moment reference exactly, across clipped/unclipped gradients
and schedule boundaries. Independent FP64 norm error is at most `1.50e-8`
relative to max(1,norm), within the fixed `2e-6` allowance. Native tests and
strict clippy pass. Complete training outcomes are pending.

The first attempt completed seeds 7,11,1,3,5,13. Seeds 17 and 19 stopped
before any update because their initialization comparison included derived
JSON ratios parsed without exact floating-point roundtrip support. All three
initial evidence files (`state.bin`, `margins.bin`, `queries.json`) are byte
identical to #199 for both failures. The repair enables exact JSON float
parsing and compares integer coverage counts, with a regression that still
rejects changed counts. It reuses the six completed seed directories without
modification and runs only the two missing seeds in `audit-02`. The combined
clock includes all 687.506760975 seconds of `audit-01`, preserving the original
1800-second cap. Both attempts and their separate source fingerprints remain.

From the root, using native tools and a CUDA-enabled execution context:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/class-geometry-wsd-target"
cargo test --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --bins
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" preflight
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/class_geometry/runs/audit-02/report.json \
  experiments/class_geometry_wsd/runs/audit-01
"$CARGO_TARGET_DIR/release/verify" experiments/class_geometry_wsd/runs/audit-01 \
  experiments/class_geometry/runs/audit-02/report.json
"$CARGO_TARGET_DIR/release/analyze" experiments/class_geometry_wsd/runs/audit-01
```

Use a fresh output directory. Evidence is ignored and preserved under root
`experiments/class_geometry_wsd/runs/` and `.kanban/artifacts/200/`.
