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

## Findings

Producer `3a509853` reproduces all 24 complete original score reports exactly.
The table gives descriptive means across the four starting-variant/mode cells
in each group. These runs share seed 1 and the same examples; they are not four
independent seeds. Each entry is novel / test percent. Stored-value prediction
means the full-vocabulary winner is one of the two stored values. Choice is
the auxiliary comparison on examples with two distinct values.

| Input | Geometry | Original raw % | Stored-value prediction % | Two-value choice % |
| --- | --- | ---: | ---: | ---: |
| initial | old | 23.73 / 23.63 | 46.48 / 43.60 | 48.80 / 50.61 |
| continue | old | 23.19 / 21.73 | 44.38 / 40.67 | 50.10 / 51.58 |
| grow | old | 41.26 / 42.43 | 80.13 / 78.71 | 49.90 / 51.79 |
| initial | Hilbert | 23.58 / 23.73 | 45.02 / 42.68 | 48.59 / 51.58 |
| continue | Hilbert | 21.39 / 23.24 | 42.87 / 42.24 | 50.55 / 51.23 |
| grow | Hilbert | 38.67 / 36.13 | 70.61 / 67.63 | 52.91 / 48.98 |

Growth substantially improves stored-value prediction while unseen two-value
choice stays close to half. Old-geometry growth chooses correctly between the
two values on 82.46% of training examples, versus 49.90% novel and 51.79% test.
Hilbert growth has 84.98% training choice, versus 52.91% novel and 48.98% test.
Thus training fit does not transfer to choosing the value attached to the
query. The original accuracy improvement is real, but it has not established
generalized query binding. These observational results do not prove that the
query has no influence or that a specific architectural change will fix it.

The CE decomposition makes the old-geometry gain more precise:

| Split / input | Full CE | Negative log stored-value mass | Conditional stored-value CE |
| --- | ---: | ---: | ---: |
| novel / initial | 2.695 | 1.852 | 0.843 |
| novel / grow | 1.795 | 0.906 | 0.889 |
| test / initial | 2.691 | 1.883 | 0.808 |
| test / grow | 1.778 | 0.943 | 0.834 |

Here the full CE reduction comes from concentrating probability on stored
values; conditional CE actually worsens. The average complete probability
mass on stored values rises from 17.93% to 45.99% on novel examples and from
17.76% to 44.83% on test examples. This probability mass is distinct from the
frequency of predicting a stored value in the first table.

The fixed breakdowns also show a position preference. On novel examples,
old-geometry growth's choice accuracy is 67.63% when the first pair is queried
and 33.27% when the second is queried. Raw accuracy is 56.43% and 26.90%, while
stored-value prediction is nearly equal at 80.32% and 79.94%. Hilbert growth
has choice accuracy 67.63% and 39.11% for those same two groups. These are
conditional observations, not a query-swap intervention.

At query position 5, before any filler, old-geometry growth predicts a stored
value on every novel example but achieves only 54.01% two-value choice. At
position 11, stored-value prediction falls to 69.74% and choice is 50.00%.
Hilbert growth has 99.10% / 55.86% at position 5 and 52.63% / 50.46% at
position 11. Longer-prefix value retention remains another limitation; poor
query binding is already visible at the shortest original prefix.

All 2,048 training examples remain, including 65 duplicate-value examples.
Each held-out split has 512 examples, including 14 duplicate-value novel
examples and 23 duplicate-value test examples. The original `validation`
report equals `validation/novel`; it is retained as supplied, not counted as
another independent dataset. No original success or 99% flag changes.

The frozen source fingerprint is
`4379f93ff2eb53216f60ca21541a2c230c846c794a81628c6572580004787cd1`,
and executable SHA256 is
`21e4810c1c1967d55dc60b29b671c4f9ca41e2d50a82f1e2d25e59f37afddadd`.
Fresh review repeats all 24 diagnostic files byte for byte and refuses all
four deliberate changes to copied inputs. Restored inputs verify again.
The maximum per-record CE decomposition discrepancy is 8.882e-16.
The six native controls, clippy and formatting pass, including from a fresh
extraction of the frozen producing source. Evidence and exit codes are under
`.kanban/artifacts/402`, with complete reports under
`transformer-runs/binding-diagnostic/original-v1` and review copies under
`transformer-runs/reviews/task402-v1`. Both disks remain above their free-space
thresholds and no original model, label, example or transformer is changed.

The next architectural question is how to share learned content matching
across unseen key/value combinations. Increasing the number of plain blocks
has chiefly improved value retention here. CM110's independent state-width
obstruction remains valid, but does not establish that width alone repairs
this observed binding failure. Generalization and 99% remain unresolved.
