# Geometry persistence during continued training

The early-settling schedule now has a useful observed-path maintenance
certificate: all eight seeds keep **more than 60% active-class readout energy
through all 4000 moving hold updates**. Rank-33 prediction certificates also
survive finite nonlinear parameter changes: coverage is **98.510% after 250
updates** and **98.045% after 1000**. A single 4000-update jump remains mostly
uncertified, at 0.244%. Discovery and future-gradient control remain open.

The [protocol](PROTOCOL.md) was frozen at `6dd2eb7c`; the measured native
harness is `9960c497`. All 24 complete traces and 192 declared endpoint
windows pass, with zero false certificates or bound violations. These are
the existing three eight-seed cohorts; no training or trajectory changed.
The endpoint parameters and realized updates are known to the constructor.
This is conditional maintenance, not prediction from only the initial state.

Ten Lean theorems formalize the damped forcing radius, projected AdamW
forcing, norm and energy-share bounds, and finite-window margin transport.
Their explicit forcing hypotheses describe an observed path; they do not
establish a favorable future forcing budget from the starting optimizer state.
The [DP1–DP3 derivation](../../review/flat_spectrum.md#observed-finite-window-persistence)
also explains why readout concentration is insufficient for prediction.

## Readout geometry throughout the hold

Each seed starts at state 4000 and receives all 4000 updates through state
8000. The active/unused partition comes only from observed training labels.
The bound separates common decay from the full executed forcing and retains
all adaptive, tied-encoder, moment, and rounding contributions. The table
gives the range of per-seed minima over every hold update.

| Schedule | Projected-forcing share lower bound | Whole-embedding forcing lower bound | Observed minimum active share | Full windows certified above 60% |
| --- | ---: | ---: | ---: | ---: |
| Main WSD, .001 plateau | 0% | 0% | 69.832–72.658% | 0/8 |
| Main WSD, .01 plateau | 0% | 0% | 47.750–49.711% | 0/8 |
| Early settling, .0001 hold | **61.763–65.877%** | 58.496–61.915% | 67.316–72.219% | **8/8** |

All 32000 early-settling updates move; none is a frozen checkpoint. The
projected bound exceeds 60% at all of them. The whole-embedding bound does
so at 30080 updates and all 4000 updates in only three seeds. Neither bound
certifies a full hold above two thirds. The higher-rate controls exhaust the
sufficient radius even though substantial active energy remains observed.

For early settling, the final active-mode radius is 0.881–0.987; the whole
embedding radius is 1.346–1.546. They are centered at the **decayed anchor**.
Task #203's gradient-independent 56.58–56.98 motion bound is centered at the
original anchor, so these numbers are not an identical-center improvement
ratio. More importantly, the new radius consumes observed forcing budgets;
the old bound was available for arbitrary future gradients. Small observed
motion alone does not prove those future budgets will remain small.

## Prediction certificates after many updates

The full model's nonlinear feature movement is enclosed by the unchanged
Gram/RMS and signed-readout forecast. Every old certificate comes from the
stored task #203 envelope. Every rival stays in the transported bound.
No endpoint evaluation or query label is used to construct the future bound.

| Early-settling window | Validation observations | Actually correct | Direct nonlinear forecast | Transport rank 32 | Transport rank 33 | Transport rank 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 updates | 112920 | 99.588% | 99.580% | 98.486% | **98.510%** | 99.394% |
| 1000 updates | 150560 | 99.591% | 99.327% | 98.012% | **98.045%** | 98.759% |
| 4000 updates | 37640 | 99.586% | 8.053% | 0.231% | **0.244%** | 1.012% |

At 250 updates, 111238 of 111274 positive old rank-33 certificates survive
(99.968%). At 1000 updates, 147617 of 148350 survive (99.506%). Those
retention percentages use the old-certified subset; the table's coverage
uses all validation queries. All emitted positive transport bounds happen
to have positive old bounds here; no newly positive case is counted as
retained. Across individual early-settling windows, rank-33 coverage ranges
98.172–98.767% at 250 updates, 97.683–98.512% at 1000, and 0–0.553% at 4000.

There are three 250-update anchors (5000, 6000, 7000), four 1000-update
anchors (4000, 5000, 6000, 7000), and one 4000-update anchor (4000).
These repeated queries/windows are correlated, with eight training seeds,
not 192 independent replications. Endpoint certification says nothing about
query predictions at every intervening update. A four-window comparison
does not by itself prove one original certificate survives continuously.

| Control schedule | Window | Direct forecast coverage | Transport rank-33 coverage |
| --- | ---: | ---: | ---: |
| Main WSD .001 | 250 | 2.573% | 0.195% |
| Main WSD .001 | 1000 or 4000 | 0% | 0% |
| Main WSD .01 | 250, 1000, or 4000 | 0% | 0% |

Failed sufficient conditions are retained. Even in early settling, the
4000-update forecast radius becomes too loose while actual correctness
remains high. This locates a remaining obstacle in finite nonlinear bound
transport; it is not evidence that the model forgets its predictions.
The direct forecast is a separate full-model control, not a low-rank
geometry certificate or a new training-accuracy improvement.

## Evidence and limits

Independent rereading verifies **173426688 class-pair upper bounds** and
**7226112 emitted minimum-margin bounds**, including every original GPU
minimum margin. All 96000 hold prefixes pass. The fixed 1e-3 feature,
1e-3-per-logit, and scaled 1e-8 enclosure/strictness allowances are unchanged.
This audits the native arithmetic; Lean proves the real-arithmetic results,
not the full Rust/BLAS/FP32 execution.

Construction takes 398.33 seconds and independent verification 121.87,
520.20 combined, below the frozen 1800-second cap. The three old bases are
reused; their original preparation cost is not included. No inference or
training speedup follows. All outcomes and bounds remain under ignored
`runs/persistence-01/`; summaries are regenerated with `summary.jq`.
The measured source SHA-256 is
`0beac7b3962c0067be936aecc5658e3ef17cd8a5ce4d9553ca939fda51d643a9`.

The two-panel [local figure](../../.kanban/artifacts/204/persistence.svg)
shows the minimum across seeds at every update and the three endpoint
horizons. Its sources are `trajectories.jq` and `persistence.gp`. All reports,
plots and the measured executable survive worktree removal in root ignored
storage. Native checks pass 19 tests plus clippy/fmt; all ten new Lean
theorems use only standard axioms. The full build and global audit remain at
36 existing `sorry`, zero results resting on them, and zero extra axioms.

What remains unproved is that AdamW reliably creates favorable alignment
and controls future forcing from initialization. Readout energy concentration
also does not imply a flat nonzero spectrum. This result supplies useful
observed-path persistence and finite-window prediction certificates without
claiming either missing implication.

## Reproduction

Run from the root with the pinned native dependencies already available:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target"
cargo test --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin geometry-persistence
cargo clippy --offline --locked --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin geometry-persistence --tests -- -D warnings
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --bin geometry-persistence
"$CARGO_TARGET_DIR/release/geometry-persistence" \
  experiments/geometry_persistence/runs/reproduction \
  experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01 \
  experiments/geometry_dynamics/runs/support-01
"$CARGO_TARGET_DIR/release/geometry-persistence" verify experiments/geometry_persistence/runs/reproduction
jq -f experiments/geometry_persistence/summary.jq \
  experiments/geometry_persistence/runs/reproduction/report.json > /tmp/persistence-summary.json

LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/PersistenceAxioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml
```

The output directory must be new. The artifact verifier checks all 192
windows and rereads all 24 complete traces. The binary includes the pinned
Kernite source adapters but this postprocessing performs no GPU work.

Generate the figure from the verified run:

```sh
jq -nr -f experiments/geometry_persistence/trajectories.jq \
  experiments/geometry_persistence/runs/reproduction/*-trace.jsonl > /tmp/persistence-trajectories.tsv
jq -r '[.forecasts[]|select(.cohort=="settle" and .split=="validation")]|group_by(.horizon)[]|[.[0].horizon,([.[]|select(.method=="direct_directional")][0].coverage*100),([.[]|select(.method=="transport33")][0].coverage*100),(100*.[0].correct/.[0].count)]|@tsv' \
  /tmp/persistence-summary.json > /tmp/persistence-forecasts.tsv
gnuplot -e "data='/tmp/persistence-trajectories.tsv'; forecast='/tmp/persistence-forecasts.tsv'; output='/tmp/persistence.svg'" \
  experiments/geometry_persistence/persistence.gp
```
