# User priority change during the learning-rate comparison

On 2026-11-09 at 12:27:53 UTC the user requested `0.01` and `start it`.
This supersedes the original protocol's order/completion of both arms.
The 0.003 process was stopped and the already prepared 0.01 cohort started
immediately, with no concurrent GPU training.

At interruption, 0.003 had four complete seeds (7,11,1,3), a partial seed 5
with its last saved geometry probe at step 7250, and three unrun seeds
(13,17,19). Preserve all files. Do not count these missing outcomes as
successful measurements or draw an eight-seed conclusion from that arm.
The complete original producer report is archived as
`report-before-user-priority.json`; a native tool annotates terminal status
and preserves every measured numeric value with exact JSON float parsing.

The interrupted producer had not finalized its external-input ledger.
The optional verifier mode `--completed-only` audits the four complete seed
directories, trace hashes, input pairing, optimizer checks, energies and FP64
query margins. Its output explicitly limits scope and records the missing
finalized input ledger. The ordinary complete-cohort verifier remains strict.
Partial seed 5 and unrun seeds receive no independent completion claim.

The 0.01 arm keeps all eight original seeds, all 10000 updates, all 31 probes,
the fixed 1800-second cap and every numerical allowance. Its comparison to
the completed 0.001 cohort remains a full paired learning-rate comparison.
No other recipe or model setting changes.

To reproduce the annotation on a copy of the stopped, unannotated producer
directory, build the native tools with the `lr003` feature and run
`record_interruption STOPPED_DIRECTORY '2026-11-09 12:27:53 UTC'`. The tool
refuses an already annotated directory or a different slot inventory.
To audit the preserved completed runs from the repository root:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/class-geometry-lr003-target"
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features lr003 --bin verify --bin record_interruption
"$CARGO_TARGET_DIR/release/verify" experiments/class_geometry_lr/runs/lr003-01 \
  experiments/class_geometry_wsd/runs/audit-02/report.json --completed-only
```

This audit passes 40000 updates, 124 probes, 9333728 query margins and all
four paired initial states. It does not change the interrupted arm's status.
