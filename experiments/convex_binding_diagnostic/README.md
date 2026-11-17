# Stored values and query binding

Do the remaining original MQAR errors come from predicting outside the two
stored values, or from choosing the wrong stored value? This read-only
diagnostic follows task400 and uses all eight c620019a inputs and all sixteen
f5d317a6 final stage8 endpoints. It performs zero updates and selects no model.
All original examples, labels, masks, classes and logits remain unchanged.

| Inputs | Count | Geometry / starting variant / mode |
| --- | ---: | --- |
| initial c620019a | 8 | old or Hilbert / cautious or refined / Mini or forward-only Mini |
| equal-update continuation | 8 | the same complete input grid |
| fixed-capacity content growth | 8 | the same complete input grid |

The model receives only the original constant-object prefix. Task roles enter
the analysis afterward. The frozen original generator places two distinct
keys at positions1 and3, their values at2 and4, and the supervised query at
one of positions5 through11. Values may repeat. Prefix shape and the original
target are validated against that generator; no replacement labels or
synthetic examples enter the measured reports.

Full-vocabulary predictions are partitioned into correct, the other distinct
stored value, or outside both stored values. An auxiliary diagnostic compares
the two stored-value logits on distinct-value examples, using the original
class tie order and reporting the true-minus-other margin. This candidate
restriction is not the original model, accuracy or success. Duplicate-value
examples remain in every original metric and in the error partition; they
are explicitly counted and excluded only from two-choice denominators.

Complete CE is also decomposed numerically into negative log probability mass
on the unique stored values plus conditional CE within those values. The mass
denominator includes all100 classes. Duplicates form a singleton with zero
conditional CE. Each record must reproduce full CE within2e-12. Reports retain
the original raw report verbatim, including its raw-logit digest, and cross-check
independently accumulated counts, CE and logits. Fixed breakdowns cover both
selected pairs and every query position. Empty denominators are explicit nulls.

The scope is one original case, `gpt-probe-key-value-binding`, seed1. These are
post-hoc descriptive measurements, not a learned decoder, a causal query test,
an unseen-answer theorem or a new stress-test success. The original transformer
baseline is unchanged. No outcome is inferred before the complete reports run.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/binding-diagnostic
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1
cargo test --offline --locked --manifest-path experiments/convex_binding_diagnostic/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_binding_diagnostic/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_binding_diagnostic/Cargo.toml --check
cargo build --release --offline --locked --manifest-path experiments/convex_binding_diagnostic/Cargo.toml
```

Before evaluation, archive the producing commit and freeze the executable,
protocol, complete input pins, plan, wrappers and summary generator with a
SHA256 manifest. Generated evidence stays on storage outside the worktree.

```sh
bash experiments/convex_binding_diagnostic/run.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/binding-diagnostic-v1 \
 /run/media/misha/storage/transformer-runs/content-growth/original-v1 \
 /run/media/misha/storage/transformer-runs/binding-diagnostic/original-v1
bash experiments/convex_binding_diagnostic/review.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/binding-diagnostic-v1 \
 /run/media/misha/storage/transformer-runs/content-growth/original-v1 \
 /run/media/misha/storage/transformer-runs/binding-diagnostic/original-v1 \
 /run/media/misha/storage/transformer-runs/reviews/task402-v1
jq -s -f experiments/convex_binding_diagnostic/summarize.jq \
 /run/media/misha/storage/transformer-runs/binding-diagnostic/original-v1/*.json \
 > /run/media/misha/storage/transformer-runs/binding-diagnostic/summary-v1.json
```

One32GiB native worker respects110GB aggregate RAM and free-space thresholds
of2GiB on the workspace and16GiB on storage. Low disk space stops execution.
Review repeats all24 reports byte for byte and refuses changed metadata,
scores, final endpoints and intermediate endpoints on copies. Native controls
cover independent full/candidate accuracy, duplicate values, complete CE,
class ties, every query position, and malformed prefixes without relabelling.
