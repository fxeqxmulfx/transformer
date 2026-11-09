# Inference efficiency draft

Saved at the user's request to commit the current work and stop. The trainer
is `grokking` commit `b70811e0e0e7ebdc4a8a2e48044c180332f7c40d`; its source
matches the existing pinned hash exactly.

The [original protocol](PROTOCOL.md) and [tightening extension](TIGHTENING_PROTOCOL.md)
declare source inference, cached weight transposes, and two label-free feature
reuse methods. Draft native adapters, bounds, audits and timing code are present.
The cohort entry point and provenance/report aggregation remain unfinished;
the executable returns an explicit draft error. No GPU inference comparison
has been run, and there are no performance or reuse-coverage findings.

CPU checks use:

```sh
cargo test --offline --locked --manifest-path experiments/inference_efficiency/Cargo.toml
cargo fmt --manifest-path experiments/inference_efficiency/Cargo.toml --check
```

Unused-code warnings are expected while the cohort runner is disconnected.
Generated build products and check logs remain local and uncommitted.
