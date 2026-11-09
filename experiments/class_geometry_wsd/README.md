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
strict clippy pass. The completed cohort passes all eight trajectories, all
80000 updates and 248 geometry probes in 920.89 seconds of combined execution,
including the initialization-only failures described below. The independent
verifier passes 18667456 margin checks with zero false certificates.

## Paired findings

The requested full recipe improves the weakest final accuracies, reaching
99% in 8/8 seeds versus 6/8 with the previous recipe. Final accuracy still
decreases in five individual seeds. Rank-32 coverage increases in two seeds
and decreases in six; no new-recipe seed reaches 99% coverage at any sampled
probe. The full recipe therefore does not close the compressed-certificate gap.

Entries are validation percentages; differences are percentage points against
the paired task #199 baseline. Late minima include every one of the 21 fixed
probes from step 5000 through 10000, not just the cooldown.

| Seed | Final accuracy | Accuracy change | Final rank-32 coverage | Coverage change | Minimum late rank-32 coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 99.64 | -0.02 | 97.90 | -1.38 | 91.46 |
| 11 | 99.38 | -0.30 | 98.02 | -1.53 | 93.22 |
| 1 | 99.64 | +0.02 | 98.64 | -0.51 | 94.69 |
| 3 | 99.57 | +0.60 | 98.07 | +1.02 | 92.18 |
| 5 | 99.68 | -0.09 | 97.79 | -0.30 | 93.52 |
| 13 | 99.47 | -0.15 | 98.21 | -0.91 | 92.88 |
| 17 | 99.53 | +1.42 | 97.26 | +1.28 | 89.95 |
| 19 | 99.62 | -0.17 | 97.87 | -1.87 | 94.20 |

Minimum late rank-32 coverage decreases in every paired seed. Seven of eight
retain 90% throughout that sampled window, versus all eight previously.
Two seeds retain 99% accuracy at every late probe (previously zero), but zero
retain 99% coverage. First sampled 99% accuracy occurs at 2000–4000 updates,
later than the previous 1000–1500; all crossings and regressions are in the
native comparison output. This study checks current-state certificates and
does not forecast an unevaluated next update.

Active-contrast energy share finishes at 71.51–73.97%, close to the previous
70.16–75.87% range. Its mean signed change, in percentage points, comprises
isotropic readout +20.10, adaptive readout +94.61, tied encoder -72.69, memory
-1.97, total quadratic -0.20 and negligible rounding. Pure-decay radial share
credit is zero. The same opposing signs persist, but concentration does not
guarantee aligned query margins. Global top-32 readout tail energy remains
22.27–24.95%, and active/full rank-32 overlap is 0.865–0.952. The exact-real
CG1–CG6 theorems apply with the recorded positive clipping scalar; they do
not assert discovery or convergence for this optimizer.

Clipping activates on 424–469 updates per seed. At all 240 actual probe
updates, parameters and both moment buffers match the independent scalar
FP32 implementation exactly; maximum independent FP64 norm discrepancy is
`7.08e-8`. Ordinary/instrumented backward-update comparisons are byte exact.
All paired initialization/data/sampler hashes match. Maximum parameter closure
is `6.62e-24`, scaled energy closure `3.59e-14`, and share closure `7.55e-15`.
Independent feature/logit errors stay below `3.70e-5`/`1.91e-5`, within the
unchanged `1e-3` allowances. No numerical tolerance was tuned.

Rank-32 folds achieve 20–32 directions, with minimum retained training
singular value `9.36e-4`, maximum coefficient norm 1116.96 and orthogonality
penalty `1.37e-11`. The analyzer retains every fold and rank control. These
are measured bounds with execution allowances, not a floating-point theorem.

## Preserved execution history

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

The six original completed seeds were measured at `bfd64cc8`, source hash
`af0ba06257b9ad7b2f6cd2d0d92ea1815bbc7fa55ab79b354f92cf9c63abf68d`.
The completion uses `2a914b44`, source hash
`93e998da00cb5f033868e6a421ff297de2a0283b24aaf8a2126c3fc33f67e9c5`.
The final `audit-02` report records each seed's source and links the six
unchanged `audit-01` seed directories. Both measured executables are retained.
Native tests, strict clippy, formatting and independent verification pass.

## Reproduce

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
  experiments/class_geometry_wsd/runs/audit-03
"$CARGO_TARGET_DIR/release/verify" experiments/class_geometry_wsd/runs/audit-03 \
  experiments/class_geometry/runs/audit-02/report.json
"$CARGO_TARGET_DIR/release/analyze" experiments/class_geometry_wsd/runs/audit-03
"$CARGO_TARGET_DIR/release/compare" experiments/class_geometry/runs/audit-02 \
  experiments/class_geometry_wsd/runs/audit-03
```

Use a fresh output directory. Evidence is ignored and preserved under root
`experiments/class_geometry_wsd/runs/` and `.kanban/artifacts/200/`.
To reproduce the retained completion specifically, append the immutable
`experiments/class_geometry_wsd/runs/audit-01` directory as a fourth training
argument and use another fresh output directory. Its report hash is pinned;
the runner reuses complete seeds and charges its prior runtime to the same cap.
