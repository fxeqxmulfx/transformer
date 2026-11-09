# Higher learning rates with the main-model AdamW/WSD recipe

The requested 0.01 cohort is complete: eight seeds, 80000 updates and 248
geometry probes in 877.96 seconds. Compared with 0.001, final validation
accuracy improves in every paired seed, from a mean of 99.567% to 99.631%.
Mean final rank-32 certificate coverage changes only from 97.970% to 98.015%.
Persistence deteriorates: each seed's minimum late coverage falls from
89.95–94.69% to 16.07–22.61%. Increasing the peak does not close the geometry
gap in this fixed task and budget.

The user suggested a higher learning rate during task #200. The
[protocol](PROTOCOL.md), frozen at `637d3a1e` before candidate measurements,
declared peak rates 0.003 and 0.01 against the completed 0.001 arm. Each arm
declared eight paired seeds, 10000 updates and 31 geometry probes. Only the
peak changes: beta2=.95, decay=.1, independent-coordinate clipping at 1,
WSD fractions and final LR1e-8 remain fixed. No architecture change or early
success stopping. The constant-rate task #199 study remains separate.

The later [user priority change](USER_PRIORITY.md) started 0.01 immediately.
The 0.003 sweep is preserved with four complete seeds, one partial seed and
three unrun seeds; it does not supply a complete-cohort comparison. The
requested 0.01 cohort completed all eight fixed seeds within its original
1800-second cap, with no numerical failures or changed allowances.

| Arm | Native build feature | Warmup/stable peak | Final LR |
| --- | --- | ---: | ---: |
| Main-model baseline | None | 0.001 | 1e-8 |
| Three-times peak | `lr003` | 0.003 | 1e-8 |
| Ten-times peak | `lr01` | 0.01 | 1e-8 |

The shared native source is in `../class_geometry_wsd`; the feature arms are
mutually exclusive, embed this protocol, record their actual peak, and include
that value in their source fingerprint. Kernels and all other update rules
are shared unchanged. Native tests, strict clippy and formatting pass for both
new builds. Both 16-case GPU preflights match the FP32 reference exactly.

## Paired outcomes

All entries are validation percentages. The late minimum uses all 21 probes
from step 5000 through 10000, including the stable interval and cooldown.
Certificates describe the current evaluated state; they do not forecast a
future unevaluated update.

| Seed | Accuracy .001 | Accuracy .01 | Rank-32 coverage .001 | Rank-32 coverage .01 | Late minimum .001 | Late minimum .01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 99.639 | 99.660 | 97.896 | 97.577 | 91.456 | 19.490 |
| 11 | 99.384 | 99.532 | 98.023 | 98.130 | 93.220 | 16.344 |
| 1 | 99.639 | 99.660 | 98.640 | 97.938 | 94.687 | 20.255 |
| 3 | 99.575 | 99.639 | 98.066 | 98.406 | 92.179 | 22.614 |
| 5 | 99.681 | 99.745 | 97.790 | 98.470 | 93.518 | 19.936 |
| 13 | 99.469 | 99.596 | 98.215 | 98.151 | 92.880 | 20.850 |
| 17 | 99.532 | 99.554 | 97.258 | 97.407 | 89.947 | 17.450 |
| 19 | 99.617 | 99.660 | 97.875 | 98.045 | 94.198 | 16.068 |

Final rank-32 coverage improves in five pairs and falls in three. No seed
reaches 99% coverage at any of the 31 probes, with either peak. At the eight
0.01 endpoints, the 37640 seed-query evaluations contain 139 incorrect
predictions and 608 correct but uncertified predictions; the paired 0.001
counts are 163 and 601. Thus 24 fewer errors produce only 17 additional
certificates. These counts concern this eight-endpoint cohort, separately
from the earlier 24-state certificate audit.

All eight 0.01 seeds first reach 90% sampled accuracy at step 250, versus
1000–1500 for 0.001. But first 99% sampled accuracy is later in every pair,
at 3000–9000 versus 2000–4000. First 90% rank-32 coverage occurs only during
cooldown, at 8500–9250 versus 1500–2000 for 0.001. Coverage at cooldown start
(step 8000) is just 16.34–33.50%. The apparent endpoint recovery therefore
hides poor coverage through much of training. The native comparison retains
all 50/90/99 crossings and counts of coverage regressions, plus ranks 33
and 128 at the endpoints.

Zero of eight 0.01 seeds retain 90% coverage at every late probe, compared
with seven at 0.001. Zero retain 99% accuracy throughout that window,
compared with two at 0.001. Minimum late accuracy falls in every pair to
96.09–97.92%. Endpoint improvement does not imply persistent geometry.

## Geometry and verification

Final active-class energy share is 63.59–70.28%, lower in every pair than
the 0.001 range of 71.51–73.97%. The global top-32 readout tail retains
22.80–26.86% of energy, and active/full rank-32 overlap is 0.747–0.867.
Mean cumulative share contributions, in percentage points, are isotropic
readout +60.57, adaptive readout +573.84, tied encoder -474.19, moment memory
-120.76 and total quadratic -6.14. These are signed contributions summed
over 10000 steps, so individual magnitudes can exceed 100 points. Pure-decay
radial share credit is zero, and rounding is negligible. The large opposing
terms remain essential; retaining only favorable readout terms would give
the wrong account of the trajectory.

The independent verifier passes all 80000 updates, 248 geometry probes,
18667456 query margins and eight paired initial states, with zero false
certificates or bound violations. All external input hashes are unchanged.
All 240 actual optimizer probe updates match the scalar FP32 parameter and
moment reference at 0 ulps; maximum independent norm discrepancy is
`7.08e-8`. Clipping activates on 132–139 updates per seed, versus 424–469 at
0.001. Maximum parameter, scaled energy and share closure errors are
`6.62e-24`, `1.78e-13` and `4.12e-14`. Feature/logit errors are at most
`3.50e-5`/`2.08e-5`, within the unchanged `1e-3` allowances.

Across readout rank-32 folds, achieved rank is 20–32, the smallest retained
training singular value is `9.71e-5`, the largest coefficient norm is
5913.59 and the largest orthogonality penalty is `7.85e-11`. These measured
checks include numerical allowances; they are not floating-point theorems.
The CG1–CG6 exact-real identities from task #199 still apply, but a proof
that training discovers and maintains aligned margins remains open. The
observed recovery during cooldown identifies schedule phase as relevant to
that question; this experiment does not isolate cooldown from elapsed time.

The measured executable uses source at `2a914b44` with fingerprint
`a00eab5a04b17486e47c7042325af87f0924b6604da5200a6412a3388d0f5fa5`.
The complete-cohort verifier includes the priority amendment at `f00c4514`.
Native tests (19 producer and three verifier tests), strict clippy and
formatting pass. The measured executable and verification binaries are
preserved with the reports outside the temporary worktree.

## Interrupted 0.003 arm

Only these four seeds completed before the user prioritized 0.01:

| Seed | Final accuracy | Final rank-32 coverage | Minimum late rank-32 coverage |
| --- | ---: | ---: | ---: |
| 7 | 99.660 | 98.385 | 74.240 |
| 11 | 99.575 | 98.002 | 62.210 |
| 1 | 99.660 | 98.746 | 76.854 |
| 3 | 99.617 | 97.875 | 67.290 |

Seed 5 stops with its last saved probe at step 7250; seeds 13,17,19 are
unrun. Their outcomes are retained as incomplete, with no eight-seed
conclusion. The optional limited verifier passes 40000 updates, 124 probes
and 9333728 margins for the four completed seeds. It explicitly reports
that the interrupted producer did not finalize its external-input ledger.
The strict complete-cohort verifier rejects this interrupted arm, as intended.
The original producer report and all partial files remain preserved; see
the [priority amendment](USER_PRIORITY.md) for the exact audit commands.

## Reproduce

From the root, using a CUDA-enabled context and the completed 0.001 baseline,
run the requested 0.01 arm into a fresh output directory:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/class-geometry-lr01-target"
cargo test --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features lr01
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features lr01 --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features lr01 --bins
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" preflight
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/class_geometry_wsd/runs/audit-02/report.json \
  experiments/class_geometry_lr/runs/lr01-02
"$CARGO_TARGET_DIR/release/verify" experiments/class_geometry_lr/runs/lr01-02 \
  experiments/class_geometry_wsd/runs/audit-02/report.json
"$CARGO_TARGET_DIR/release/analyze" experiments/class_geometry_lr/runs/lr01-02
"$CARGO_TARGET_DIR/release/compare" experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-02
```

Generated evidence remains ignored under root `experiments/class_geometry_lr/runs/`
and `.kanban/artifacts/201/`. Preserve failures, all seeds and every probe.
