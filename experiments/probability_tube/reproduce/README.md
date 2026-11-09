# Native Rotate90 reproduction

This runner recreates the parent experiment's three selected optimizer
cohorts directly from the recovered original trainer. The
[bootstrap protocol](PROTOCOL.md) preserves its initialization, split,
sampler, forward/backward kernels and main-model optimizer adapter. Every
attempt has separate execution provenance; the historical reports are not
replaced or imitated.

Place the exact sources in the repository root's ignored checkouts:

| Checkout | Repository | Revision |
| --- | --- | --- |
| `.lake/reproduction/grokking` | `https://github.com/fxeqxmulfx/grokking` | `b70811e0e0e7ebdc4a8a2e48044c180332f7c40d` |
| `.lake/reproduction/kgen` | `https://github.com/fxeqxmulfx/kgen` | `a1b6bc2c956e1a5f35960e26b209e1b6c213096c` |
| `.lake/reproduction/diff-gpt` | `https://github.com/fxeqxmulfx/diff-gpt` | `6bb9d03a7184cb0a43bfaf240977639213e2cbeb` |

Build-time checks validate the trainer digest, clean pinned Kernite sources
and all four GPT source digests. Only an output-directory copy of the trainer
receives the same read-only instrumentation used by the frozen studies.
The original model PTX digest is also checked at execution. The `preflight`
command checks source provenance, the optimizer's independent FP32 reference
and the existing FP64 forward/gradient oracle without training a trajectory.
The cohort run repeats the forward/gradient checks for every seed and checks
all geometry probes and instrumented versus ordinary update bytes.

Run `cargo fetch --locked --manifest-path experiments/probability_tube/reproduce/Cargo.toml`
once if the registry dependencies are not cached. From the repository root,
use native tools. Repeat the checks and build for
each of the three configurations: no feature, `--features lr01`, and
`--features settle`. Preserve a separate executable after each build:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/probability-reproduction-target"
cargo test --offline --locked --release --manifest-path experiments/probability_tube/reproduce/Cargo.toml
cargo clippy --offline --locked --release --all-targets --manifest-path experiments/probability_tube/reproduce/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/reproduce/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/probability_tube/reproduce/Cargo.toml
mkdir -p .lake/probability-reproduction-bins
cp "$CARGO_TARGET_DIR/release/rotate90-complex-kernite" .lake/probability-reproduction-bins/lr001
.lake/probability-reproduction-bins/lr001 preflight
.lake/probability-reproduction-bins/lr001 run experiments/probability_tube/runs/lr001
```

Use the names `lr01` and `settle` for the corresponding executables and
output roots. Run cohorts sequentially on the GPU. Each root contains new
`attempt-NN` directories. A later attempt charges all earlier recorded
attempt durations and never overwrites them; a still-running attempt blocks
a concurrent launch. All seeds and requested checkpoint slots remain in
the report even after errors or caps. The later forecast constructor and
verifier are separate from this trajectory runner.

All three configurations pass 21 native tests, strict clippy, and the GPU
source/optimizer preflight on the GTX 1050. Optimizer comparisons have zero
ULP differences in all 16/16/32 declared cases; the source PTX digest matches
the original pin. The source forward maximum absolute error is 1.32e-6;
both full-embedding directional checks are below 5.39e-8. These are preflight
checks. Training findings will be recorded in the parent README after the
frozen runs and audits. Generated logs are in `.kanban/artifacts/81/`.
