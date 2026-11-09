# Tighter pre-update forecasts

Can computable bounds close the gap between task #188's 90.38% certificate
coverage and the same trained states' 99.56% correctness? Yes, on this fixed
cohort: combined Gram/RMS bounds reach 99.21%; adding a signed final-readout
interval reaches **74727/75280 = 99.27%**, with zero false certificates.
The classifier, CE/full-backpropagation update and AdamW trajectory are unchanged.
The remaining 553 uncertified observations comprise 328 wrong decisions and
225 correct decisions whose intervals remain inconclusive.

The [protocol](PROTOCOL.md) was committed at `f783d620` before candidate
measurements. All 24 cases finished in 103.71 seconds, inside the 900-second
cap: four arms, seeds 7/11, states 0/5000/10000, all 9409 queries per case.
Step 10000 is a post-budget one-update probe. The trained subset comprises
the 16 noninitial states, with 4705 validation observations each. Repeated
queries/states are not independent samples or a fresh-seed confirmation.

## What differs

Every method shares the original old feature and directional tangent. Only
the radius propagation or final interval changes. The complex product retains
the source `1/sqrt(2)` and all mixed terms. Each final center, radius and gap
interval is persisted before evaluating the updated network. Nominal nonlinear
values compute defects and denominators; they never replace the propagated
old-value-plus-tangent center.

| Method | Change from the original forecast | Trained validation certificates | Coverage | Mean margin width | Mean CE width |
| --- | --- | ---: | ---: | ---: | ---: |
| Legacy | Original Frobenius and `2/s` RMS bounds | 68040 | 90.38% | 4.94750 | 0.992150 |
| Gram | Bound linear amplification through a Gram Frobenius norm | 73845 | 98.09% | 1.30708 | 0.134268 |
| Local RMS | Regularized secant bound, with the original global fallback | 73967 | 98.26% | 1.32485 | 0.194967 |
| Combined | Both preceding changes | 74685 | 99.21% | 0.35973 | 0.032675 |
| Directional | Intersect combined with signed pre-final-RMS readout bounds | 74727 | 99.27% | 0.35428 | 0.029704 |
| Directional raw | Direct readout component before intersection, separately audited | 74727 | 99.27% | 0.42884 | 0.032090 |

Gram and RMS each explain a substantial improvement; their combination adds
more than either alone. The final intersection adds 42 correctness certificates
to the combined method. Although raw and intersected intervals certify the
same correct decisions here, intersection matters for loss directions and
interval width. The final mean margin interval is 13.96 times narrower than
legacy; the CE interval is 33.40 times narrower. These ratios describe this
cohort's intervals, not runtime speedups.

The [FT1–FT3 derivations](../../review/flat_spectrum.md#forecast-tightening)
have 18 Lean theorems with explicit witnesses: the regularized secant identity,
finite rectangular Gram/operator bounds, signed positive-denominator quotients,
and interval intersection/class ordering. They prove real-arithmetic bounds.
The whole Rust/BLAS/CUDA execution is not formally verified. The unchanged
1e-3 feature and 1e-3-per-logit allowances are checked assumptions; the scaled
1e-8 enclosure/strictness tolerance is also unchanged. Strictness uses only old
values and frozen endpoints, which dominate enclosed future values in valid
intervals. No tolerance or method was tuned after reading results.

## Coverage, including the difficult cases

The difficult subset is each split's old-margin bottom decile, selected before
the source update: 471 queries, with canonical-index tie breaking. Training and
validation splits and subsets are those of task #188.

| Observations | Count | Actually correct | Legacy certified | Directional certified |
| --- | ---: | ---: | ---: | ---: |
| All states, train | 112896 | 75618 | 69886 | 75454 |
| All states, validation | 112920 | 75328 | 68416 | 75103 |
| All states, difficult train | 11304 | 7490 | 6342 | 7356 |
| All states, difficult validation | 11304 | 7208 | 5555 | 7021 |
| Trained states, train | 75264 | 75218 | 69487 | 75054 |
| Trained states, validation | 75280 | 74952 | 68040 | 74727 |
| Trained states, difficult train | 7536 | 7490 | 6342 | 7356 |
| Trained states, difficult validation | 7536 | 7208 | 5555 | 7021 |

Initial states have low accuracy, so including them gives 66.51% directional
validation coverage. They are retained, not discarded as failures. On trained
difficult validation, coverage rises from 73.71% to 93.17%, against 95.65%
actual correctness. The summary generator retains every ablation for each
of these eight subsets and every individual case.

Each arm below has 18820 trained validation observations across both seeds
and noninitial states. This compares certification of already trained models;
it is not a new architecture-selection experiment.

| Arm | Actually correct | Legacy coverage | Combined coverage | Directional coverage | Correct but still uncertified |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 18720 | 92.69% | 99.28% | 99.35% | 23 |
| Complex diagonal | 18785 | 98.56% | 99.77% | 99.79% | 5 |
| Complex product radial | 18690 | 71.51% | 98.15% | 98.28% | 193 |
| Complex product plain | 18757 | 98.76% | 99.64% | 99.64% | 4 |

Both seeds improve: seed 7 goes from 95.94% to 99.63%, seed 11 from 84.82%
to 98.90%. Radial seed 11 at step 5000 remains the limiting case: 4483 of
4705 decisions certified, versus 4651 actually correct. Its original bound
certified only 564. This one case accounts for 168 of the remaining 225 correct
but uncertified trained validation observations; no new fit was attempted.

## Loss directions and layer inflation

| Method | Trained validation CE decreases certified | CE increases certified | Inconclusive |
| --- | ---: | ---: | ---: |
| Legacy | 0 | 0 | 75280 |
| Gram | 435 | 70 | 74775 |
| Local RMS | 886 | 145 | 74249 |
| Combined | 10135 | 2025 | 63120 |
| Directional | 10166 | 2090 | 63024 |
| Directional raw | 5849 | 1345 | 68086 |

The directional intersection also certifies 16263 margin improvements and
6202 regressions on trained validation. Across all 225816 train/validation
observations, it certifies 35684 CE decreases and 5807 increases. Most loss
directions remain inconclusive; certification does not imply that every
update improves every query, or that mean margin growth controls mean CE.

The following are means over all 150544 trained train/validation observations.
Errors are measured only after the complete forecast freeze. The final
directional method uses the combined pre-final ball; it has no additional
whole-feature radius.

| Stage | Actual error | Legacy radius | Gram radius | Local RMS radius | Combined radius |
| --- | ---: | ---: | ---: | ---: | ---: |
| First linear | 0.0001356 | 0.0005182 | 0.0002717 | 0.0003832 | 0.0002235 |
| Product | 0.0000944 | 0.0006114 | 0.0003961 | 0.0004932 | 0.0003541 |
| Second linear | 0.0000313 | 0.0024743 | 0.0005974 | 0.0020106 | 0.0005377 |
| Branch RMS | 0.0061695 | 0.5102259 | 0.1291544 | 0.2282916 | 0.0625467 |
| Residual | 0.0061588 | 0.5102544 | 0.1291829 | 0.2283201 | 0.0625751 |
| Final RMS | 0.0119480 | 1.4693158 | 0.3765592 | 0.3863965 | 0.1011875 |

The second linear map and subsequent normalization still amplify conservative
scalar radii. Average linear coefficients fall from 4.429/3.517 to 1.554/1.268;
the branch RMS coefficient falls from 413.105 to 206.953. Final radius divided
by actual error, using pooled means, falls from 122.98 to 8.47. The retained
report includes every layer's distributions, nominal/mixed defects and effective
amplification; product amplification is an effective scalar ratio, not a full
direction-preserving representation. The final nominal defect alone averages
0.009214, so bound transport is not the only remaining contribution.

## Audit and cost

All 24 source next-state hashes exactly match task #188. All 74 input files
remain unchanged. All layer balls, 21678336 class-pair intervals per track,
and all margin/CE intervals pass, including the raw directional component.
There are zero false certificates and zero execution-allowance violations.
Maximum old/new feature discrepancies are 5.84e-5/6.32e-5; maximum executed
logit discrepancy is 9.07e-6. Independent tangent scaled error is at most
5.47e-9, and all independent source gradient/forward checks pass.

Mean times per complete 9409-query state: preparation 1.258 seconds, comprising
Gram work 0.145, propagation 0.991 and readout intervals 0.122; FP64 parameter
conversion/differencing is also charged. Persistence takes 1.257 seconds and
post-freeze independent audit plus query JSON serialization 1.263 seconds.
The actual source update takes 1.287 milliseconds for its original minibatch,
a different workload. All six tracks run together, so these are harness costs,
not isolated per-method benchmarks. GPU snapshots retain the desktop graphics
process; exclusive-device timing is not established. No inference or training
speedup follows from this experiment.

Retained forecast numeric buffers occupy 317.29–440.33 MiB by length and
330.97–454.01 MiB by capacity per case. These exclude weights, temporary
preparation buffers, allocator metadata and the later audit graph; they are
not process peak memory. Each frozen binary is 107488448 bytes (102.51 MiB),
2.40 GiB across the cohort. JSON reports and GPU snapshots add storage.

The next efficiency question is whether a smaller, label-free serving bound
can save actual inference work after charging preparation and fallback. That
is the separate task #189. This result supplies stronger mathematical bounds;
its current multi-track audit is not an efficient serving implementation.

## Reproduction and retained evidence

Run from the repository root with native tools. The build pins the original
trainer hash `14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`,
Kernite revision `a1b6bc2c956e1a5f35960e26b209e1b6c213096c`, original PTX and
Cargo lockfile. `ROTATE90_SOURCE` can locate the identical source elsewhere;
Kernite paths are declared in Cargo.toml and the shared build script.

```sh
cargo test --offline --locked --manifest-path experiments/forecast_tightening/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/forecast_tightening/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/forecast_tightening/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/forecast_tightening/Cargo.toml

# Regenerate the reference ledger from the retained original forecast report.
jq -r '.cases | sort_by(.arm,.seed,.step)[] | [.arm,.seed,.step,.audit.new_state_sha256] | @tsv' \
  .kanban/artifacts/188/forecast-report.json > /tmp/forecast-states-188.tsv

# Choose a new output directory: existing evidence is never overwritten.
experiments/forecast_tightening/target/release/rotate90-complex-kernite \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  /tmp/forecast-states-188.tsv experiments/forecast_tightening/runs/reproduction

bash experiments/forecast_tightening/verify.sh experiments/forecast_tightening/runs/reproduction
jq -f experiments/forecast_tightening/summary.jq \
  experiments/forecast_tightening/runs/reproduction/report.json > /tmp/forecast-summary.json

LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ForecastTighteningAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --locked --release --manifest-path scripts/native_index/Cargo.toml
if rg -n 'native_decide|^axiom |linter\..* false|set_option autoImplicit' src; then exit 1; else test "$?" -eq 1; fi
```

All 18 native tests and strict Clippy pass. Lean's full build and axiom checks
pass: 36 existing `sorry`, zero results resting on them, zero extra axioms.
The generated INDEX has 21723 theorems, zero vacuous and zero placeholder
declarations. Native tests cover nonzero source-step forecasts, legacy tangent
identity, cancellation, Gram layout/rank witnesses, near-zero RMS/fallback,
and all numerator signs. `verify.sh` independently rereads all input hashes,
24 frozen binary hashes and all serialized query intervals/certificates.

Evidence lives at root `experiments/forecast_tightening/runs/audit-01/` and
`.kanban/artifacts/196/`, outside the disposable worktree. The measured source
fingerprint is `271350f7a2f91637264de8a825b24b6829e6e5a9b757f9988b600c924f40f85b`;
the task #188 reference TSV fingerprint is
`64858e9fd68c19c548f99ca1ac3df644b553687ec2681195305a0e826359bf42`.
Generated reports, binaries, summaries and logs remain ignored and uncommitted.
