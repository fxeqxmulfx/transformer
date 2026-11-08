# Optimizer efficiency with full backpropagation

Compact complex AdamW moments save **1047552 bytes (47.74% of moment storage)**
while retaining the original CE objective and full backpropagation. They
match the source bitwise at every checked state. All 16 fresh runs finish
with identical paired quality. Shared-device timings include improvements
and slowdowns, so this establishes a memory saving, not a reliable speedup.
The user skipped further no-gradient/local-learning work.

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
need only 1146880 moment bytes. This motivated the separately frozen candidate
below: store moments once per independent coordinate, rebuild parameter
aliases, and precompute uniform AdamW scalars.

## Compact-moment comparison

The [candidate protocol](PROTOCOL.md) was frozen at `0f17cdf9` before outcomes.
The source control calls its original CE, full backward and AdamW. The
candidate retains their FP32 arithmetic order, including epsilon placement,
while storing complex moments only at independent owners and precomputing
`lr / bc1` and `1 - lr * decay` on the host. Real SwiGLU tests only the scalar
change. This experiment does not separate the two complex changes' effects.
Canonical export preserves the source's initial positive-zero first moments,
then reconstructs signed first-moment and unsigned second-moment aliases.

The correctness gate passes all 24 saved-state next steps and their metrics,
160 initial fixed-gradient warmup updates, and eight zero/tiny-gradient cases.
All 576 paired benchmark workloads finish with identical canonical state
hashes. All 1152 timing samples, including 864 measured samples, are retained;
implementation and workload order alternate. No case, sample or run failed,
was capped, or was selected for favorable timing. All 73 inputs are unchanged.
The paired benchmark takes 33.05 seconds; training takes 135.45 cohort seconds.

These are ranges of the six per-state median **source/candidate** ratios:
values above one favor the candidate. They are observations on a shared GPU.
Raw wall/event durations, nearest-rank p90 and every paired ratio are retained.

| Arm | Update-only wall ratio | Complete-step wall ratio |
| --- | ---: | ---: |
| Real SwiGLU | 0.982–1.002 | 0.998–1.040 |
| Complex diagonal | 1.283–1.883 | 0.791–1.024 |
| Complex product radial | 0.915–1.775 | 0.988–1.548 |
| Complex product plain | 0.914–1.338 | 0.972–1.548 |

Fresh seeds 29/31 supply four architecture pairs, 4000 updates per method:
16 completed runs, 64000 full backward passes and 656 scheduled observations.
Parameters, both moments, train metrics and validation metrics match at all
328 paired observations, including initialization. Endpoints and first
threshold steps are therefore common to each pair. Thresholds are observed
on the fixed 100-update evaluation grid; all histories remain available.

| Arm | Seed | Final validation % | Final CE | First 99% / 99.5% step | First source-endpoint quality step | Training ratio | Matched-quality active ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 29 | 99.872 | 0.01222 | 2800 / 2800 | 4000 | 0.958 | 0.962 |
| Real SwiGLU | 31 | 99.830 | 0.01780 | 2700 / 2900 | 4000 | 0.946 | 0.950 |
| Complex diagonal | 29 | 99.745 | 0.01267 | 1600 / 1900 | 2800 | 0.958 | 0.946 |
| Complex diagonal | 31 | 99.639 | 0.01546 | 1600 / 1800 | 2400 | 1.053 | 1.012 |
| Complex product radial | 29 | 99.639 | 0.02799 | 900 / 1200 | 1500 | 0.974 | 0.947 |
| Complex product radial | 31 | 99.596 | 0.03017 | 1100 / 1500 | 1500 | 1.008 | 1.011 |
| Complex product plain | 29 | 99.596 | 0.03504 | 1100 / 1200 | 1200 | 1.050 | 1.012 |
| Complex product plain | 31 | 99.447 | 0.03882 | 900 / 1300 | 1300 | 1.014 | 1.004 |

Both ratios divide source time by candidate time. Matched quality requires
at least the source endpoint's validation correct count and no worse CE;
the comparator is the source's first attainment of that same joint target,
not automatically all 4000 source updates. Active time charges setup,
training, evaluation and canonical conversion/hashing. These identical
trajectories establish no convergence improvement. They retain the full-CE
quality above 99% without a new accuracy repair or loss change.

Each complex model's moment storage falls from 2194432 to 1146880 bytes.
Its resident training buffer payload falls from 28469904 to 27422352 bytes,
a 3.68% reduction. Real storage is unchanged. The benchmark uses a longer
sampler prefix, hence larger total buffers; its paired saving is the same.
These are allocated buffer payloads, not total driver memory. Setup still
temporarily allocates the source moments before replacing them, so no reduced
peak setup memory is claimed. Replacement is charged to setup and canonical
expansion/hash overhead to scheduled evaluation.

All 144 comparison ownership queries succeeded. Every snapshot includes
GNOME Shell; foreign `phase-mhc-transformer` compute jobs also appear during
the cohort. No benchmark case or training group qualifies as exclusive.
The update-only improvement does not establish faster complete training:
four of eight matched-quality active ratios are below one. Keep the compact
representation as a validated storage optimization; an exclusive-device
profile would be needed before claiming speed. Larger end-to-end gains
remain a question for forward/backward or inference work, with full
backpropagation retained.

## Source profile reproduction

The shared build pins the original trainer, its PTX and Kernite revision.
Without `compare`, the executable profiles only the source. Five Rust tests,
warning-free Clippy, formatting and the native repository checks pass.
No Lean statement changes.

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

## Candidate reproduction

After the build commands above, use a new output directory:

```sh
experiments/optimizer_efficiency/target/release/rotate90-complex-kernite compare \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/optimizer_efficiency/runs/compare
jq -e -f experiments/optimizer_efficiency/verify.jq \
  experiments/optimizer_efficiency/runs/compare/report.json
jq -f experiments/optimizer_efficiency/summary.jq \
  experiments/optimizer_efficiency/runs/compare/report.json \
  > experiments/optimizer_efficiency/runs/compare/summary.json
```

`verify.jq` checks the retained cohort, ordering, paired states/metrics, cost
counts and actual memory without trusting only top-level pass flags.
Executed evidence and all 16 endpoint checkpoints are retained in root
`experiments/optimizer_efficiency/runs/compare-01/`. The executed binary,
checksum, toolchain/check logs and checkpoint hashes are preserved in
`.kanban/artifacts/191/`. The compiled candidate source digest is
`da4ed2267bb9c0f4500696ff94adfc6cc88e8f485f33b0cbd1d10868304d546d`;
candidate PTX digest is
`b80bc818262bddd3ad28b4971dce0eca68bdac6b416b40e34926ccdd15bddeff`.
