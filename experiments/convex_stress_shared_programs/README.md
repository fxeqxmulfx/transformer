# Shared causal copy programs

Can a shared input program repair retrieval on new symbols without a task
answer function? This native Rust experiment implements CM52 in
`review/flat_spectrum.md` and
`Transformer.Grokking.FlatSpectrum.ConvexDepth.SharedPrograms`, reviewed at
`62b675fc789469d6457dd7d0f41d68c7ee80abea` (task321). It preserves the failed
task320 four-case baseline, historical GPT references and the separate original35
ensemble. The user's target is at least99% **per task**, for both raw tokens and
exact sequences. Ordinary CE and original95% success remain separate fields.

The fixed library contains relative lags and causal successor walks. A walk
starts from the last visible symbol and makes1–4 hops, each over an earlier
completed match at offset1–4. First/latest occurrence and an optional predecessor
marker from the entire vocabulary are enumerated. No task identity, designated
key/value token, generator formula or held-out answer enters the model. Missing
edges return class0 with an explicit diagnostic flag. Training retains every
program that agrees with every supervised training prefix, recording the first
contradiction for each rejected program. An empty library or cap is a refusal.

The additional prior is a shared copy decoder over the entire raw vocabulary,
including labels never observed in training. The normalized DFT code and full
conjugate identity head are fixed at margin24. The learned coefficients are a
simplex mixture of retained programs; all real/imaginary head products and raw
classes are computed. The realified column budget64V is conservative relative to
CM52's2|margin|V bound. A numerical SVD and independent class-major complex
products check the instantiated tensor; they are not an exact real certificate.
The DFT phases are a basis, not an arithmetic answer rule.

| Mode | Actual training | Initialization |
| --- | --- | --- |
| Full |16 complete-data projected ordinary-CE gradients, learning rate1 | Uniform retained mixture |
| Mini |4 epochs in original prefix order, batch32, real short tails, learning rate0.1 | Same mixture |
| Forward | No gradients, projections or optimizer updates | Same mixture |

On the perfectly compatible face all training logits coincide. Full/minibatch
therefore exercise the real CE derivative without identifying a better program;
roundoff changes are counted separately from attempted steps. Generalization
comes from the explicit shared-program/copy prior and observations. CM52 proves
accepted consensus correct only when the unknown target is realizable in the
bank. The experiment never asserts this premise from flatness or training fit.
It records unanimity, ambiguity, missing defaults, unseen-label coverage and
roundoff-sensitive predictions separately from measured accuracy.

Ten native tests pass. Seven component tests cover exhaustive short causal walks against an independent
interpreter, recovery of a training-unseen symbol, real complex products,
full-vocabulary finite-difference gradients at competing mixtures, weighted
minibatch/tail equivalence, exact trajectory replay, and mutation/cap failures.
Three adapter checks additionally cover the training-only boundary, longer inputs,
per-prediction metrics, ambiguity/default accounting and all-mode refusals.
Original cohort scores are pending at this source freeze; a toy witness is not a
new stress-test result. Copy-only programs need not fit Dyck or blocks labels
absent from their input; a generic prefix-summary decoder is separate work.

The unchanged `convex_stress_sequences` adapter supplies all original train,
clean/novel validation, test and longer splits of MQAR, two-hop lookup, Dyck and
three alternating blocks. Individual historical dataset hashes/checkpoints are
unavailable; source-derived fingerprints do not claim historical artifact
identity. The task320 report is required verbatim with SHA256
`65620f0757ae3c57ece3d23e3ed6fb464e05484f9b055dc335e9694f09f7ff7b`.
It contains the historical GPT rows, whose original statuses are preserved.

Every new training state is sealed before any held-out scoring. The scorer
retains every supervised prediction and raw CE, separately recomputes program
interpretation and complete complex products, and independently checks aggregate
counts. Logits must agree within1e-8; predictions with margin above1e-7 must also
agree. Smaller margins are explicitly counted as roundoff-sensitive. Replays
reconstruct discovery, every training step, full metrics and all refusals.
All12 mode outcomes remain in the denominator. The original35 comparison314 and
pending signed-kernel integration317/319 are unchanged; no union pass count is
inferred from this component.

Commands from a clean committed checkout, using storage paths outside a worktree:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
cargo fmt --manifest-path experiments/convex_stress_shared_programs/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_shared_programs/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_shared_programs/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_shared_programs/Cargo.toml
experiments/convex_stress_shared_programs/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-shared-programs"
experiments/convex_stress_shared_programs/run.sh FRESH_FREEZE/convex-stress-shared-programs BASELINE_REPORT FRESH_RUN
```

`run.sh` enforces64GiB virtual memory and serial execution, retains process exits
and `/usr/bin/time -v` evidence, then produces the complete report only after all
four independent replays. Source/binary/protocol freezes include the compiled
Cargo dependency graph and literal include closure. Search, initialization,
training and scoring each have explicit2B work guards; buffers have a56GiB guard.
Forecasts are conservative scalar-work units, not measured FLOPs; repeated
checks and replays incur additional work and are included in process timing.
Generated evidence stays local outside this worktree and is not committed.
