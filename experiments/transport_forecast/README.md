# Training geometry across a certified update

Can the flat-spectrum development's signed training transport support a
pre-update certificate for the actual complex-product model, and does it
become informative before the model reaches 99% accuracy? The conditional
bridge works: all 26 probes pass with zero false certificates. Rank-32
endpoint coverage is **96.22% / 97.45%** for seeds 7/11. Its first 50%
coverage occurs at the same sampled step as each seed's first 99% accuracy;
it does not provide an earlier majority-coverage signal of that transition.

The [protocol](PROTOCOL.md) was committed at `07c328f1` before candidate
measurements. Both original plain-product trajectories were replayed without
changing the source CE/backpropagation, AdamW, data, sampler or parameters.
All six complete parameter/moment hashes at 0, 5000 and 10000 match exactly,
as do the six corresponding next-update hashes from task #196. All declared
cases finished in **162.08 seconds**, below the fixed 1200-second cap.
The step-10000 next-update probes remain outside the original training budget.

This experiment is for `(a+b)^3 mod 97`. It connects the existing conditional
training-geometry theory to a finite update. It does not establish that neural
weight decay minimizes a nuclear norm, predict eventual optimizer discovery,
or settle the original flat-spectrum task `a^3+ab`.

## What is connected

For an old query state `x` and competitor-minus-target readout `w`, signed
training reconstruction gives `old_gap <= P+R`: `P` uses training scores,
and `R` bounds the residual using a training basis and the explicit
orthogonality defect. The proposed rounded readout `w'`, old feature tangent
`v` and the tighter feature radius `r` give

```
new_gap <= P + R + <w'-w,x> + <w',v> + ||w'||*r + numerical_allowance.
```

The readout/tangent mixed term is retained. Alternatively, if the direct
directional forecast provides endpoint `U`, the transported bound is

```
training_bridge = U + (P+R-old_gap).
```

The parenthesis is nonnegative in exact arithmetic. This identity exposes
how much training-to-query uncertainty remains; replacing this bound by `U`
would bypass the training-geometry question. Fifteen new Lean theorems prove
the finite expansion, radius bound, endpoint transport, margins, probability
and CE bounds, and the actual four-normalization source adapters. Every
conditional theorem has an explicit example. A separate nonzero witness
has simultaneous readout motion, feature motion, reconstruction residual
and forecast remainder. See the [derivation](../../review/flat_spectrum.md#training-to-forecast-bridge).

## Fixed comparisons and information boundary

| Comparison | Information and purpose |
| --- | --- |
| Old geometry | Signed training-state reconstruction before the update, including execution allowances |
| Combined bridge | Old geometry plus signed update and combined Gram/local-RMS radius |
| Directional bridge | Old geometry plus the drift interval from the signed final-RMS forecast |
| Direct forecast | Task #196 endpoint without training-reconstruction slack; separate comparator |
| Rank 0 / 32 / 128 | Center baseline / primary compressed basis / reconstruction-limit control |

Each answer orbit uses a basis fitted only to its old training states.
Query projections determine reconstruction coefficients; actual residuals
and computed orthogonality defects are retained. Training queries use the
opposite alternating reference fold. Validation queries use all eligible
training references. All groups have references; none abstains in this cohort.
Candidate answers come from observed training labels and the proved cubic
input routes. This uses known task symmetry without assuming learned
equivariance. The constructor never reads held-out target labels or future
states/logits/errors. Every class endpoint and candidate label is frozen to
a hashed binary before the updated-network forward evaluation.

The forecasts keep the fixed 1e-3 Euclidean feature allowance, 1e-3 per-logit
allowance and scaled 1e-8 enclosure/strictness tolerance. These are audited
execution assumptions; the whole Rust/BLAS/CUDA execution is not formally
verified. No allowance, rank or probe schedule was adjusted after outcomes.

## Temporal result

Accuracy is the old source's validation accuracy; bridge coverage certifies
the next actual source update. Each entry uses all 4705 validation queries.
Repeated queries and checkpoints are not independent statistical samples.

| Step | Seed 7 accuracy | Seed 7 rank-32 bridge | Seed 11 accuracy | Seed 11 rank-32 bridge |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.06% | 0% | 0.94% | 0% |
| 100 | 32.79% | 0% | 32.71% | 0% |
| 250 | 54.71% | 0.02% | 53.24% | 0.26% |
| 500 | 85.10% | 4.36% | 87.10% | 2.30% |
| 750 | 98.47% | 13.54% | 97.22% | 2.17% |
| 1000 | 98.47% | 31.71% | 99.36% | 67.44% |
| 1500 | 99.68% | 89.71% | 98.89% | 68.93% |
| 2000 | 99.72% | 91.88% | 97.32% | 83.34% |
| 3000 | 99.74% | 89.37% | 99.68% | 93.86% |
| 4000 | 99.83% | 98.38% | 99.66% | 88.01% |
| 5000 | 99.68% | 46.10% | 99.64% | 92.94% |
| 7500 | 99.68% | 92.16% | 97.75% | 75.79% |
| 10000 | 99.66% | 96.22% | 99.68% | 97.45% |

Rank-32 coverage first exceeds 50% at 1500/1000 and 90% at 2000/3000
for seeds 7/11; neither reaches 99% coverage. The first sampled 99% accuracy
is 1500/1000. There are positive certificates before those crossings, but
the broad coverage comes later. A certificate cannot precede the correctness
it certifies. These are one-step guarantees, not forecasts of the eventual
learning transition; crossings between declared probes remain unresolved.

Coverage is not monotone. Seed 7 falls from 98.38% at 4000 to 46.10% at
5000 while accuracy stays above 99%. At 5000, its old rank-32 geometry
already covers only 47.76%, so this failure is mainly in the compressed
training representation rather than the new forecast radius. Seed 11 also
has later accuracy and certificate regressions. All remain in the table.

## Endpoint ablations and remaining gap

| Step-10000 condition | Seed 7 | Seed 11 |
| --- | ---: | ---: |
| Actual new accuracy | 99.66% (4689) | 99.68% (4690) |
| Direct directional forecast | 99.64% | 99.68% |
| Old rank-32 geometry | 96.58% | 97.39% |
| Combined rank-32 bridge | 96.22% (4527) | 97.45% (4585) |
| Directional rank-32 bridge | 96.22% (4527) | 97.45% (4585) |
| Directional rank-0 bridge | 12.67% | 23.27% |
| Directional rank-128 bridge | 99.60% | 99.66% |

At rank 32 the mean future validation CE ceilings are 0.66435/0.62394;
at rank 128 they are 0.02729/0.02553. All queries contribute to these means.
The full-rank control approaches re-expression of the original classifier;
its near-complete coverage does not demonstrate a compressed learning
mechanism. Rank clamping and every group's spectrum are retained in the
frozen metadata. There is a separate basis per answer orbit, not one global
32-dimensional representation of the whole task.

The bridge therefore establishes a useful local conditional guarantee.
What remains is control of the training reconstruction and its movement
that explains the transition and the regressions. The earlier componentwise
nuclear-norm bound's zero coverage is unchanged by this result.

## Audit, cost and reproduction

All 26 cases have zero false certificates, class/CE enclosure violations,
layer-ball violations and execution-allowance violations. Maximum executed
feature discrepancy is 8.48e-5, and maximum readout execution discrepancy
is 1.13e-5. The old-geometry FP64 rounding excess is at most 2.55e-13,
inside the unchanged scaled tolerance. The original independent tangent
check passes at all probes (maximum scaled error 1.77e-7).

The independent verifier rereads all 20 input hashes and all 78 frozen
bridge/forecast/state binaries, reconstructs every reported margin, CE and
certificate flag, and compares them with the separately recorded source
outcomes. It also evaluates the saved future parameters with the independent
batched FP64 model from `flat_approx`, checking every future bridge endpoint.
The Lean full build and axiom audits pass: 36 existing sorry, zero dependent
results and zero extra axioms. Generated INDEX has 21738 theorems and zero
vacuous/placeholder declarations.

All 23 native tests and strict Clippy pass. Independent verification covers
2,201,706 query/method reductions and 142,376,988 future FP64 class gaps.

Mean time per 9409-query probe: forecast preparation 1.136 seconds, geometry
0.461, readout/bridge 0.385, persistence 1.694 and audit 1.560. The source
minibatch update takes 1.157 milliseconds, a different workload. These are
multi-method diagnostic costs; no inference or training speedup is claimed.
Replay between probes averages 0.826 seconds. Retained evidence occupies
about 5.2 GiB at root `experiments/transport_forecast/runs/audit-01/`;
logs, verification and generated summaries live in `.kanban/artifacts/197/`.
All generated outputs remain ignored. The measured harness fingerprint is
`6c8c192353d75d1525a7ac4694846524d92b5b6fe059d98b90e6f81bb1c949d5`.

Run from the repository root with the pinned original trainer/Kernite and
task #196 evidence retained. Choose a new output directory on every replay.

```sh
cargo test --offline --locked --release --manifest-path experiments/transport_forecast/Cargo.toml
cargo clippy --offline --locked --release --manifest-path experiments/transport_forecast/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/transport_forecast/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/transport_forecast/Cargo.toml

experiments/transport_forecast/target/release/rotate90-complex-kernite \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/forecast_tightening/runs/audit-01/report.json \
  experiments/transport_forecast/runs/reproduction
experiments/transport_forecast/target/release/verify experiments/transport_forecast/runs/reproduction
jq -f experiments/transport_forecast/summary.jq \
  experiments/transport_forecast/runs/reproduction/report.json > /tmp/transport-forecast-summary.json

LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/TransportForecastAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --locked --release --manifest-path scripts/native_index/Cargo.toml
if rg -n 'native_decide|^axiom |linter\..* false|set_option autoImplicit' src; then exit 1; else test "$?" -eq 1; fi
```
