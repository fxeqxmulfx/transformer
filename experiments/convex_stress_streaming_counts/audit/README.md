# Read-only raw-output audit

This separate crate audits the frozen `1371ae4f` experiment. It is not linked
into that producer and never searches for a program, trains a model, selects
parameters, changes predictions or rewrites experiment artifacts.

It requires all 13 original training seals and an existing case measurement.
It authenticates the frozen executable, protocol, original source metadata and
complete saved model before reading each raw teacher-forced or generated query.
An independent complex product checks every full vocabulary vector and winner.
Scalar recounts check CE, original masks, transitions, raw/exact denominators,
unseen labels, actual own-output prefixes, EOS stopping and extra tokens.
The 95% original-success and 99% requested thresholds are recomputed using
integer counts. Refused endpoints retain their errors and receive no credit.

The streaming reader is ported from task346's `c8cc576d` audit, with its weighted
training groups replaced by every original ungrouped query in task343's schema.
Two regression tests cover prefix and generative layouts, genuine failures on
early EOS and extra outputs, coordinate/categorical unseen-label differences,
altered aggregate counts, missing splits and changed raw CE after rehashing.

```sh
cargo fmt --manifest-path experiments/convex_stress_streaming_counts/audit/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_streaming_counts/audit/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --all-targets --offline --locked \
  --manifest-path experiments/convex_stress_streaming_counts/audit/Cargo.toml -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_streaming_counts/audit/Cargo.toml
prlimit --as=68719476736:68719476736 -- \
  experiments/convex_stress_streaming_counts/audit/target/release/convex-streaming-raw-audit \
  RUN FROZEN_BINARY PRODUCER_COMMIT CASE FRESH_EXTERNAL_JSON
```

Store the audit result outside the immutable run directory. Its report binds
the state, metrics and raw-file digests and lists raw99/exact99 separately.
The producer's independent-pair verifier still compares every artifact and
full training update; this audit additionally rederives the original scores.
