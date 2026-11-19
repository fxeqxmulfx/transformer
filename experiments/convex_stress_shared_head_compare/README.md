# Complete shared-head comparison

Task330 asks whether the reviewed329 shared-feature head improves raw-label
accuracy and ordinary cross-entropy on the complete original35-task suite.
The target is99% per task, with full batch, minibatch and forward-only results
reported separately. The reviewed319 suite remains34/35 raw99 in every mode;
no new heldout result is claimed by this prospective specification.

The implementation must retain both105-choice policies, every unchanged choice,
all old endpoints, conventional/depth/DiffGPT references and failed attempts.
Eligibility is the already selected signed-kernel representation, fixed before
new scores. Both prior/supplement roles retain their original input routing.
For each role, a block is covered exactly when an original training record
routes there. Covered inputs use the complete329 head; uncovered inputs use
the complete saved319 predictor. The gate uses no task name or heldout labels.
All new scores remain visible even when they regress.

Lean first: `scripts/SharedHeadFallbackAxioms.lean` audits CM38 in
`review/convex/moments.md` and `ConvexDepth/Fallback/{Basic,Domain,Batch}`.
The finite disjoint coordinate sum implements the exact selected logits;
fixing the entire parent block is an affine constraint. Both stored coefficient
budgets are charged. Ordinary CE is convex in every candidate coefficient.
Every original training record is covered, so the routed objective and its
derivative agree with the actual shared-feature objective, including repeated
record IDs and a shorter final minibatch. The audit uses non-Dirac rows and
an unsupported query with nonzero parent logits. All seven printed declarations
use only `propext`, `Classical.choice`, and `Quot.sound`.

These existing results suffice; no new theorem, assumption or `sorry` is added.
Coverage does not prove unseen-label correctness. The two roles are trained
on their own original records; their summed standalone objectives are convex.
The actual composite predictor's observed CE must also be reported, without
claiming that independent role updates minimize a routing-conditioned objective.
Full and minibatch change all real/imaginary head coordinates; forward-only
retains the initializer with zero CE gradients. Flat spectrum supplies the
feature/coefficient bound, not the missing labels.

Before any heldout score, freeze the complete adapter, source closure, binary,
inputs, eligibility, shape-derived construction/training/metric/audit costs
and disk-sealed coefficients. All processes use at most64GiB. Preserve exact
old/new raw logits, native counts, field readouts, ordinary CE changes,
fixes/regressions, and task298's lower-CE tradeoffs. A fresh complete run and
saved-state replay must reproduce all artifacts. Generated outputs stay local
on storage; source, protocols and authored interpretation are committed.

The native adapter imports the unchanged original generator and metric code
through the hash-bound319 dependency. Both original metric paths are evaluated,
including their separate field calls. Every original single-target record then
gets a paired old/new logit audit and an independent inverse-Fourier audit of
the new complete tensor. JSONL keeps every class logit and raw target, not only
errors. Actual metric counters must equal the prospective layout counts.
The new feature gate never supplies a label. The complete original role model
is reconstructed from its saved state; the new head repeats all initialization
and CE updates and must equal the reviewed329 training inspection exactly.

Before scoring, the inventory forecasts the complete original case from
dimensions and layouts. Reserve two old construction passes, three complete
new training passes, all scoring/audit calls, all tensors and simultaneous JSON
buffers. The reviewed six-role bounds total1,340,280,954,000 old units and
1,565,412,163,584 new units. Original layouts have9408 records per endpoint,
18816 calls per old/new metric path, and three9408-call audit paths:65856 calls
at the conservative1,789,280-unit bound. The full case forecast is
7,730,302,869,792 units, below the prospective10T cap. These are conservative
kernel-operation accounting units; file hashing/serialization and wall time/RSS
are separately retained. The process address limit is64GiB, and explicit
buffers may not exceed56GiB. Nothing is fitted or scored by these calculations.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-head-compare
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
cargo fmt --manifest-path experiments/convex_stress_shared_head_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_shared_head_compare/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_shared_head_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_shared_head_compare/Cargo.toml
prlimit --as=68719476736:68719476736 -- lake build
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/SharedHeadFallbackAxioms.lean
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/Axioms.lean
prlimit --as=68719476736:68719476736 -- scripts/native_index/target/release/lean-index --check
experiments/convex_stress_shared_head_compare/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-shared-head-compare"
FRESH_FREEZE/source/experiments/convex_stress_shared_head_compare/run_original.sh FRESH_FREEZE/convex-stress-shared-head-compare FRESH_RUN_ROOT REQUEST_JSON
```

The request names `producer_commit`, `parent_root`, `parent_manifest`,
`component_root`, and `diff_gpt_history`. Every input has a hard-coded reviewed
SHA256 in `protocol.json`:331 completed319 artifacts, all six329 outcomes,
and the separate historical DiffGPT result. Use the same request and frozen
driver in a second fresh run directory and compare every JSON/JSONL byte hash.
The driver records every case and replay exit code and continues after case
failures. A failed case stays in the35-task denominator; a complete summary is
not replaced by the old successful one. Neither driver nor inputs may be edited
while running. All generated evidence lives outside the disposable worktree.

Before the first original run, seven adapter checks pass: real unsupported
fallback with nonzero logits, both roles in all three modes, complete-logit
replay and coefficient-mutation rejection, field-call forecasts unchanged by
heldout labels, repeated training-record coverage, and complete105-choice
reporting that retains raw/CE regressions behind successful field readouts.
The first fixture accidentally covered every block; that failed assertion is
retained in local evidence. An asymmetric synthetic observation band supplies
unmatched blocks and tests the required fallback. A test-only ownership error
was also fixed before the passing build. No original model or score was used
to choose either correction. Release tests, strict clippy, full Lean build,
seven standard-axiom declarations, global audit, INDEX and forbidden checks pass.

The v1 producer55d0cd4b freezes1256 inputs. Its original preparation verifies
all331319 artifacts, then refuses before fitting/scoring because the protocol
confuses the rendered report's7632e96c hash with the complete summary's
fd8f7ff3 hash. The same complete-summary hash occurs in the independently
reproduced319 run and its reviewed manifest. Version2 corrects this input pin
only; the model, selection, forecasts and scoring are unchanged. All four
failed preparation/driver logs and resource records remain hash-bound in the
protocol and final report. This refusal supplies no new accuracy measurement.

The corrected producer3653b58e freezes1256 inputs and completes all35 cases,
all35 saved-state replays and the summary. Its complete raw logit files replay
exactly. Both105-choice policies retain35/35 original primary/all-split successes,
but only34/35 tasks reach raw99, native exact99 and all-heldout both99 in each
mode. All34 unchanged tasks retain their earlier metrics. The fresh Fibonacci
head regresses against319 in every mode:

| Mode | New raw correct /4704 | New raw accuracy | New validation CE |319 raw correct |319 validation CE |
| --- | --- | --- | --- | --- | --- |
| Full |4607|97.9379%|0.151124497113|4608|0.115829730335|
| Mini |4607|97.9379%|0.151167264799|4611|0.101307444436|
| Forward |4608|97.9592%|0.151180896772|4611|0.101307444436|

Original field readout remains0.999362244898 in every mode; it does not change
the raw99 result. Composite observed CE is0.001310457498 full,
0.001330829691 mini and0.001336682346 forward, compared with319's9.63e-9.
Full makes256 new updates across both roles, mini608, forward0. Full fixes14
old errors and introduces15, mini fixes13 and introduces17, forward fixes16
and introduces19. All previous successes, failures and lower-CE tradeoffs
remain explicit in the complete snapshot and four older-reference comparisons.

All4704 validation inputs use blocks with training records; no validation
prediction invokes the new empty-block fallback. The forward result's96 errors
include91 whose true raw label occurs in that routed block's training labels,
three absent labels in a two-class block and two in single-class blocks.
`error_diagnostic.jq` reproduces this diagnostic from the complete saved model
and scored JSONL. Block coverage therefore does not identify the answer.
The result also shows that unfreezing a complete head, or reducing its observed
CE, does not by itself improve unseen raw labels. It does not prove that the
entire model class cannot express a better predictor. A next experiment should
test whether training-only sharing of decision geometry between latent blocks
improves their individually fitted boundaries, with the corresponding linear
parameterization audited in Lean before construction. No arithmetic rule or
heldout label may be installed as a decoder.

Every endpoint retains7,056,384 old/new real coefficient slots and a charged
combined radius bound764288. The independent inverse-Fourier audit's maximum
logit residual is1.82e-12. The complete run takes9m21.72s, peaks at6,339,272KiB
RSS (6.05GiB), and exits0. Its150 JSON and6 JSONL artifacts live under
`/run/media/misha/storage/transformer-runs/shared-head-compare/original-v2`.
The complete summary SHA256 is
`774b6beabafe008a5cf41904d89faadda9bd20c313b735db0bb54fd8c501f9c6`.
Independent review at `ec89d684dab72693c9a77fbe55dd7b20c1772e78` repeats all
native and Lean checks and reproduces all156 JSON/JSONL artifacts byte for byte.
The rebuilt binary also equals the frozen executable. The fresh run plus
replay takes9m20.70s, peaks at6,339,236KiB RSS and exits0; the error diagnostic
also reproduces exactly. Task330 is approved as a comparison. This experiment
does not attain the user's99% target and remains a failed improvement.
