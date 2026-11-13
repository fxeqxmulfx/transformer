# Original-task categorical phase follow-up

Does generic constant-strip and repeated-phase discovery improve the same
nine original pair tasks after the balanced density repair? This harness ports
the complete balanced cohort at `4510ee7d`, preserving its original generator,
training observations, held-out splits, vocabulary, metrics and success rules.
The new constructor receives only training cell/label pairs and a common policy.

| Retained or new component | Frozen policy |
| --- | --- |
| Complete balanced parent | Rebuild every original decode proposal, exact and interval attempt, certified refinement and balanced proposal with unchanged policies. Retain the full report and earlier proposal, including all failures. |
| Categorical strips | At least eight observed labels agree in a row or column. Keep the base view and up to three strip views in first-occurrence label order. Unobserved axes and padding are explicit hypotheses. |
| Phase proposals | Eight categorical sketches per view,64 power steps each. Quantize angles in the input-size and observed-class-count domains. Keep all codes, matrices and power states. |
| Free-axis refinement | At most32 seeds,24 rounds and1024 total axis updates; scan every possible phase with independent integer checks. Require observed coverage of every base latent state. |
| Selection | Every earlier candidate precedes every new candidate on observed CE ties within1e-12. Seal the choice and artifact digests before held-out evaluation. |

Every returned endpoint receives a raw full-table audit of every latent Fourier
matrix, its SVD budgets and ordinary full-vocabulary CE. The constant-strip lift
also checks actual full-table errors against the observed interior partition.
Replay reconstructs the original data, all earlier families, new discovery,
every state and work ledger, then repeats the raw audits and all split metrics.
Rehashing a modified artifact does not bypass those checks.

The launcher records every process failure, continues the remaining planned
tasks, refuses existing cells and requires all nine outcomes for its summary.
It preserves all27 references in full batch, minibatch and forward-only modes;
these refer to nine forward constructions and are not three optimizer runs.
The full35 short/long goal, all105 original choices, stronger earlier families
and the conventional transformer comparison remain required for integration.

The single-model and aggregate search forecasts are checked separately; the
combined parent/new report must stay within48GiB. Each numerical process has
the user's64GiB address-space limit, with one BLAS thread, serial cases and at
least16GiB free disk. Each artifact has a512MiB cap. All partial and completed
generated evidence remains ignored in the root experiment's runs directory.

```sh
cd experiments/convex_stress_tagged/cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked --release
cd ../../..
TAGGED_PRODUCER_COMMIT=<committed-producer> \
TAGGED_BINARY=<frozen-binary> \
  bash experiments/convex_stress_tagged/cohort/run_cohort.sh \
  experiments/convex_stress_tagged/runs/original-v1
```

Freeze the tested committed source, complete source fingerprint, binary hash,
protocol and launcher before preparing original data. The core's masked
recovery is fixture evidence; no original result is established by this harness
definition or the CM25 proof. The preceding balanced follow-up passes3/9.

Four harness checks pass before the original freeze. They verify the exact
policy and frozen shared-data fingerprint, a complete fixture with raw metric
replay, every earlier candidate's order and winning CE tie, refusal/input/size
bounds, and preservation of all27 mode references when all nine processes fail.
The replay rejects a modified selection, a rehashed parent ledger and a
rehashed new-search ledger.
