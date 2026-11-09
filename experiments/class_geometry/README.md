# Discovery and persistence of class-visible geometry

Does the source optimizer build the 32 class contrasts among observed answers,
and does that explain useful compressed geometry across whole trajectories?
The [protocol](PROTOCOL.md), initially frozen at `9692d9a0`, fixes two original
seeds and six replication seeds. Implementation tolerances are recorded before
the first run. This is a mechanism test; success is not assumed.

| Comparison | Fixed scope |
| --- | --- |
| Original replay | Seeds 7,11; all 10000 updates; original complete state hashes |
| Replication | Seeds 1,3,5,13,17,19; identical source and budget |
| Class components | Active-answer contrasts, unused-class contrasts, between-group means |
| Update terms | Decay, isotropic readout, adaptive correction, tied encoder, memory, rounding; total squared increment retains all cross terms |
| Geometry | Hidden/readout-aware bases at ranks 0,32,33,128; 31 probes per seed, including 21 late probes |

All updates and every prefix receive finite energy checks. Query labels never
select a basis; current-state geometry retains #198's numerical allowances.
The source files reuse #198 at `8e719f2b`, with rank 33 added as a declared
control. No training equation changes. The independent source evaluator is
the existing FP64 implementation from `flat_approx`.

Native tests, strict clippy and formatting pass before the first run. Results,
Lean derivations and independent verification will be recorded after the
fixed cohort completes, including failed hypotheses and incomplete slots.

The first startup attempt (`audit-01`) is retained. No update ran: checks
caught a pre-final-RMS versus normalized-feature adapter error in seven seeds,
and an overly strict JSON-number equality check in the eighth. The repair
uses the evaluator's recorded final-RMS denominators and restores #198's
exact-correct-count / `1e-12` loss comparison for serialized history.
A native regression checks the normalized features against independent logits.
Certificate allowances, seeds, ranks, optimizer, probe schedule and cap are
unchanged. The corrected complete attempt uses `audit-02`.

From the root, with CUDA available to the execution context:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/class-geometry-target"
cargo test --offline --locked --release --manifest-path experiments/class_geometry/Cargo.toml
cargo fmt --manifest-path experiments/class_geometry/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/class_geometry/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/class_geometry/Cargo.toml
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/readout_geometry/runs/audit-01/report.json \
  experiments/class_geometry/runs/audit-02
```

Use a fresh output directory. Generated evidence remains ignored and local;
it is preserved under root `experiments/class_geometry/runs/` and check logs
under root `.kanban/artifacts/199/`.
