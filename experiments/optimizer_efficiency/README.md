# Optimizer efficiency with full backpropagation

The user skipped further no-gradient/local-learning work. This study keeps
the original CE objective and full backpropagation, first measuring the
source AdamW cost before specifying an implementation optimization.

## Source profile

The [profile protocol](PROFILE_PROTOCOL.md) was frozen at `ecc98fa3` before
measurement. All 24 original states pass: four arms, seeds 7/11 and original
steps 0/5000/10000. Each restored state supplies fixed-gradient throughput
probes of 256 AdamW updates and eight-step actual CE training workloads.
Three warmup and nine measured rounds alternate workload order. All 576
samples, including 432 measured samples, and final state hashes are retained.
Every restored workload reproduces its final state; all 73 input files remain
unchanged. The cohort finishes in 21.93 seconds under its 600-second cap.

Ranges below span the six state-specific medians per architecture. Timings
include host launches and synchronized completion. The ratio compares two
workloads; it is not an additive phase decomposition or an Amdahl bound.

| Arm | Update-only wall time, µs/update | Complete wall time, µs/step | Update / complete |
| --- | ---: | ---: | ---: |
| Real SwiGLU | 47.64–48.84 | 1857.83–1930.02 | 2.47–2.61% |
| Complex diagonal | 87.95–108.82 | 2288.56–2395.80 | 3.67–4.73% |
| Complex product radial | 96.36–108.90 | 1938.11–2400.44 | 4.01–5.52% |
| Complex product plain | 105.68–108.42 | 2300.78–2404.38 | 4.43–4.71% |

CUDA-event spans are retained separately and show similar ratios. They can
include queue gaps and should not be read as pure arithmetic execution time.
The wrapper follows the [CUDA driver event API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html);
nine empty brackets are retained as a measurement floor without subtraction.

All 48 ownership queries succeeded and observed this process, GNOME Shell,
and another `phase-mhc-transformer` training process. These are shared-device
observations; **none supports an exclusive-device speed claim**. No process
was stopped and no favorable cases or rounds were selected.

The real source stores 1146880 bytes of optimizer moments. Each complex arm
stores 2194432 bytes because first moments have signed aliases and second
moments have unsigned aliases. The same 143360 independent real coordinates
need only 1146880 moment bytes. This gives a concrete candidate independent
of the contaminated timings: store moments once per independent coordinate,
rebuild only parameter aliases, and precompute uniform AdamW scalars. Freeze
and validate that candidate separately before evaluating any performance gain.

## Source profile reproduction

The shared build pins the original trainer, its PTX and Kernite revision.
The current executable profiles only the source; it does not change its
forward, backward or optimizer equations. Three Rust tests, warning-free
Clippy and the native repository checks pass. No Lean statement changes.

```sh
cargo test --offline --locked --manifest-path experiments/optimizer_efficiency/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/optimizer_efficiency/Cargo.toml -- -D warnings
cargo fmt --check --manifest-path experiments/optimizer_efficiency/Cargo.toml
rustfmt --edition 2024 --check experiments/optimizer_efficiency/src/instrumentation.rs
cargo build --offline --locked --release --manifest-path experiments/optimizer_efficiency/Cargo.toml
mkdir -p experiments/optimizer_efficiency/runs
experiments/optimizer_efficiency/target/release/rotate90-complex-kernite \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/optimizer_efficiency/runs/profile
jq -f experiments/optimizer_efficiency/profile-summary.jq \
  experiments/optimizer_efficiency/runs/profile/report.json \
  > experiments/optimizer_efficiency/runs/profile/summary.json
```

Outputs must be new directories outside the input checkpoint tree. The
executed evidence is retained in root `experiments/optimizer_efficiency/runs/profile-01/`;
logs and the original profile executable are in `.kanban/artifacts/191/`.
The compiled profile source digest is
`e0a9a0968ff6823f83eba8d74232d85fccfffdf39af81f88f898bbaf0ca45d21`.
Generated reports, logs and binaries remain ignored and uncommitted.
