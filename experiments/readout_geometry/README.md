# Observable geometry and actual AdamW contributions

Does extending flat-spectrum geometry to directions visible in class-score
differences close the compressed-certificate gap, and what actually drives
the proposed update? At the same rank 32, endpoint coverage improves from
**96.22% / 97.45% to 99.21% / 99.53%** for seeds 7/11. Both trajectories
first exceed 50% coverage at step 500, before their first sampled 99%
accuracy at 1500/1000. All 26 probes pass with zero false certificates.

The [protocol](PROTOCOL.md) was committed at `db7b55b3` before candidate
measurements. This study extends [task #197](../transport_forecast/README.md)
without changing its source training or forecasts. The task is
`(a+b)^3 mod 97`; the original `a^3+ab` obstruction remains separate.
The comparison uses existing trajectories and a previously examined split,
so it is a temporal diagnostic, not an independent generalization test.

## The extension

For class readout rows `E_c`, center across classes: `C_c=E_c-mean(E)`.
The error geometry is `sum_c <C_c,e>² = eᵀ(CᵀC)e`. Zero energy means every
class score receives the same shift, even when the hidden error is nonzero.
Every pairwise score error is at most the square root of twice this energy.
Nineteen new Lean theorems prove the quadratic identity, exact invisible-error
characterization, sharp contrast/margin bounds, nonzero witnesses, and signed
finite update attribution including the mixed term and nonlinear remainder.
See the [derivation](../../review/flat_spectrum.md#observable-spectral-geometry).

| Basis | Construction from the old state | Fixed controls |
| --- | --- | --- |
| Hidden variance | Leading right singular vectors of centered training states; unchanged #197 baseline | Requested ranks 0, 32, 128 |
| Readout-aware | Project class-centered readouts into the training span, then take leading right singular vectors | Same ranks, `1e-10*s_max` cutoff, actual residuals and orthogonality defects |

The basis never sees query labels, query scores or future outcomes. Query
states determine reconstruction coefficients, while observed training labels
and checked cubic routes supply candidate answers. Training queries use
opposite reference folds; validation queries use all available training
references. No group abstains. All 97 competing classes remain in the bounds.
Every endpoint and attribution table is frozen before the updated-network
forward evaluation. The forecast still uses the fixed `1e-3` feature and
per-logit allowances and scaled `1e-8` tolerance from #197. The new theory
does not insert observed future error into a forecast.

## Coverage over both trajectories

Each cell below is directional-bridge coverage of all 4705 validation queries.
The complete old/new accuracy, loss, all ranks, both splits and both forecast
methods are in the reproducible report; `summary.jq` extracts all 26 rows.

| Step | Seed 7 hidden | Seed 7 readout | Seed 11 hidden | Seed 11 readout |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0% | 0% | 0% | 0% |
| 100 | 0% | 0.53% | 0% | 0.49% |
| 250 | 0.02% | 35.94% | 0.26% | 38.89% |
| 500 | 4.36% | 66.80% | 2.30% | 67.86% |
| 750 | 13.54% | 90.27% | 2.17% | 85.99% |
| 1000 | 31.71% | 90.56% | 67.44% | 98.45% |
| 1500 | 89.71% | 99.23% | 68.93% | 97.60% |
| 2000 | 91.88% | 99.17% | 83.34% | 93.96% |
| 3000 | 89.37% | 99.34% | 93.86% | 99.34% |
| 4000 | 98.38% | 99.57% | 88.01% | 99.02% |
| 5000 | 46.10% | 97.47% | 92.94% | 99.38% |
| 7500 | 92.16% | 99.00% | 75.79% | 90.78% |
| 10000 | 96.22% | 99.21% | 97.45% | 99.53% |

The first 90% crossing moves from 2000/3000 to 750/1000. The first 99%
crossing is 1500/3000; hidden rank 32 never reaches 99% on this schedule.
First old-accuracy 99% crossings remain 1500/1000. Useful compressed
coverage therefore appears earlier, while 99% coverage still lags seed 11's
accuracy transition. The later regressions remain; coverage is not monotone.
These are current-step certificates, not predictions of eventual learning.

| Endpoint validation outcome | Seed 7 | Seed 11 |
| --- | ---: | ---: |
| Correct after the proposed update | 4689 / 4705 | 4690 / 4705 |
| Hidden rank 32 certified | 4527 | 4585 |
| Readout rank 32 certified | 4668 | 4683 |
| Direct forecast certified | 4688 | 4690 |
| Remaining correct but uncertified by readout rank 32 | 21 | 7 |
| Incorrect, necessarily uncertified | 16 | 15 |

Across the two endpoints this removes 239 of the old bridge's 266 missing
certificates relative to the direct forecast: **89.85% of that gap**.
The remaining direct-forecast gap is 20/7 queries, or 0.43/0.15 percentage
points. Model predictions and accuracy are unchanged. The earlier 99.27%
figure and its 328 incorrect / 225 correct uncertified queries concern the
different, larger #196 cohort; they must not be merged with these counts.

Rank zero is byte-identical across both families. At requested rank 128,
readout endpoint coverage is 99.57%/99.66%, versus hidden 99.60%/99.66%:
there is no uniform improvement at the reconstruction limit. Across all
26 probes, rank-32 certified query occurrences rise from 62385 to 91673
out of 122330; rank-128 counts fall from 98972 to 98893. Repeated queries
are not independent samples. Requested readout ranks 32/128 achieve
24–32 / 24–96 across all reference folds. Every endpoint validation fold
achieves exactly rank 32 in the primary comparison.

At the endpoints, the mean retained centered-readout energy rises from
45.70%/46.06% to 75.64%/74.47%, while retained hidden-state energy falls
from 74.60%/73.06% to 46.20%/48.56% (weighted by validation query count).
This identifies hidden-variance basis selection as substantial avoidable
slack. It does not establish a flat learned spectrum. The smallest retained
training singular value is 0.01428, maximum condition ratio 1379.08,
maximum reconstruction coefficient norm 94.23, and maximum retained
orthogonality penalty `5.03e-13`; none is silently set to zero.

## What the actual update contributes

The embedding increment is decomposed into decay, full-training readout
credit, tied-encoder credit, sampling difference, corrected moment memory,
and rounding, using the actual coordinate-specific preconditioners. A
seventh parameter component contains the projection-matrix update. Each
component changes both the readout and the encoder feature tangent. The
eighth score term is the total mixed readout/feature product. Nonlinear
feature error remains in the separately certified forecast remainder.

For each query, attribution fixes its old worst competitor and measures
target-minus-competitor margin change. Positive terms help this fixed margin.
These are signed directional contributions to one actual update, not
independently trained counterfactuals. All-class certificates remain valid
when the worst competitor changes. Full-training derivative checks and
byte-identical instrumented/ordinary updates audit the source connection.

Mean validation margin contributions at step 10000:

| Contribution | Seed 7 | Seed 11 |
| --- | ---: | ---: |
| Decay | -0.007912 | -0.008351 |
| Readout credit | +0.005841 | +0.004479 |
| Encoder credit | +0.013404 | +0.004496 |
| Sampling | -0.000555 | +0.000594 |
| Moment memory | -0.009234 | -0.000044 |
| Parameter rounding | +0.000000103 | +0.000000098 |
| Projection parameters | +0.009439 | +0.013702 |
| Mixed readout/feature term | -0.00000491 | -0.00000872 |
| Total predicted change | +0.010977 | +0.014869 |

Readout credit helps 93.43%/95.18% of endpoint validation fixed-competitor
margins, but the total predicted change helps only 58.43%/78.24%. Their
signs agree on 57.22%/77.07% of queries. Across the 26 probes, sign agreement
ranges from 44.12% to 77.24% on validation and 42.26% to 78.21% on training.
All component means and sign fractions for both complete splits are retained.

Four validation probes have a negative mean total predicted change even
though mean readout credit is positive. For seed 11 at step 7500, readout
and encoder credit contribute +0.0513 and +0.3256, while sampling, memory
and projection changes contribute -0.1432, -0.2112 and -0.2261; the total
is -0.2100. Endpoint validation CE also increases slightly in both actual
updates despite positive mean fixed-competitor margin changes. Neither a
readout derivative nor an average margin is a substitute for the full
adaptive update and all-class CE.

This closes the algebraic attribution gap and gives earlier useful geometry.
It leaves a specific learning question: why the coupled dynamics discover
and maintain the class-visible training span across many updates. No
discovery, monotonicity or convergence theorem follows from this audit.

## Checks and reproducibility

All 26 slots finish in **368.15 seconds**, below the fixed 1200-second cap.
All 20 pinned input files remain unchanged, six original parameter/moment
checkpoint hashes match, and six #196 next-update hashes match. Independent
verification additionally matches all 26 #197 next-state hashes and every
hidden-baseline bound byte for byte. Both basis families have zero false
certificates and enclosure violations, including independent FP64 evaluation.

Maximum parameter/tangent/logit attribution closure errors are `1.79e-16`,
`4.46e-15`, and `5.38e-15`. The independent full-training partial readout
gradient error is at most `7.99e-9`. Feature and logit execution discrepancies
remain below `8.48e-5` and `1.13e-5`, within the unchanged `1e-3` allowances.
The independent verifier checks 4403412 query/method reductions, 284753976
future FP64 class gaps, and 244634 attribution query totals. It also checks
the attribution summary reductions and identical rank-zero controls.

Native tests, formatting and strict clippy pass. The full Lean build has no
new warnings; all nineteen theorem axiom checks use only standard axioms.
The global audit remains 36 preexisting `sorry`, zero dependencies on them,
and zero extra axioms. The native generator produces INDEX.md with zero
vacuous statements or placeholder definitions. The complete CUDA/BLAS
execution itself is empirically audited, not formally verified.

Average per-probe seconds: geometry 1.973, attribution 4.766, forecast 1.104,
readout/bridge 0.648, persistence 2.315 and audit 2.015. These are diagnostic
costs on the shared RTX 3050; no inference or training speedup is claimed.
Source modules are reused from #197 at `b52f0d84`, the original update
decomposition from `single_update`, and the pinned exact Rust source trainer.

Run from the repository root; CUDA execution may require the authorized
host context. Keep outputs local and use a fresh output directory:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/readout-geometry-target"
cargo test --offline --locked --release --manifest-path experiments/readout_geometry/Cargo.toml
cargo fmt --manifest-path experiments/readout_geometry/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/readout_geometry/Cargo.toml --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/readout_geometry/Cargo.toml
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/forecast_tightening/runs/audit-01/report.json \
  experiments/readout_geometry/runs/audit-02
"$CARGO_TARGET_DIR/release/verify" experiments/readout_geometry/runs/audit-02 \
  experiments/transport_forecast/runs/audit-01/report.json
jq -f experiments/readout_geometry/summary.jq experiments/readout_geometry/runs/audit-02/report.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ObservableGeometryAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --locked --release --manifest-path scripts/native_index/Cargo.toml
```

The measured harness fingerprint is
`cd27fa8c2f751f050161671edbce0225fdafd33920b14adcf754002babf768d9`.
All binary forecasts, states, attribution/query tables and reports are
preserved under root `experiments/readout_geometry/runs/audit-01/`.
Check logs and the measured/verifier executables remain under root
`.kanban/artifacts/198/`. Generated evidence is ignored and uncommitted.
