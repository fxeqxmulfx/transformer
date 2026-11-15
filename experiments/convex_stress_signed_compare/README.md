# Bounded signed contexts in the original stress comparison

Can the reviewed signed-context extension lift the remaining raw-label errors
above the user's 99% per-task target? This Rust adapter integrates task316's
component at `99a4265262c72c9c93e37ad29e3f2b4c947dce8b` into the complete
task314 comparison reviewed at `b69d8ea0`. It retains all 35 original seed-1
tasks, both initializer policies with 105 choices each, all conventional
transformer endpoints, 336 depth outcomes, and every earlier CE/raw tradeoff.
The separately reviewed six expanded sequence cases remain unchanged.

Lean comes first. CM51 in `review/flat_spectrum.md` and
`ConvexDepth/SignedContext/{Basic,Witness}` was reread and rechecked before this
adapter was implemented. Its 22 declarations bound arbitrary fixed signed
combinations of the actual complex source features. Normalization limits their
absolute mass to one; observed Dirac rows retain the original objective and
all complete-head derivatives. The existing result applies without an extension
to the theorem. Query-reference residuals and realizability remain explicit
unproved premises, and native flags remain false. The current manuscript's
later CM52/CM53 results do not remove those premises for this kernel bank.

The component is unchanged. Its fixed bank contains the existing context rule
and 48 polynomial/Gaussian kernels over ordinal input coordinates. It fits only
observed data, excludes entire source states in its diagnostics, normalizes
signed rows, and selects using ordinary full-vocabulary CE with a fixed canonical
head. Final predictions use the actual complete trained complex head from the
original source. No task name, answer formula, validation label or validation
score selects a kernel. The ordinal kernel geometry is an additional modeling
assumption; flatness supplies a bound, not an answer to a missing query.

| Mode | Original trained head | New training-only diagnostic |
| --- | --- | --- |
| Full batch | Complete original state and history retained | Best fixed vertex, then 64 conditional-gradient steps |
| Minibatch | Original records, repeated IDs, batches and tails retained | Best vertex, then 8 epochs with the inherited batch size and real short tails |
| Forward only | Original endpoint with zero training updates | Best vertex, no gradients or optimizer updates |

The diagnostic objective is convex while its head and bank are fixed. This
does not claim joint convexity of a learned feature map and a learned prediction
head. Diagnostic CE is distinct from ordinary observed CE of the retained head.
Every observed source row and its raw metrics/CE must reproduce the parent
exactly. The original routing is retained: use the prior wherever supported;
use the supplement only when the prior refuses and the supplement supports.
Every eligible representation is applied without selecting the best held-out
endpoint. All regressions remain in the report.

The adapter is ported from `convex_stress_crossing_exclusion_compare` at
`9373e853`. Its complete CM50 source states, original mode, batch, reference,
record IDs and labels are checked before construction. New saved states retain
the full prior/supplement chains, signed fits and diagnostics. The parent
projection adds a complete CM50 reference and costs while preserving CM49,
CM48, guarded CE and task298's lower-CE/higher-error outcomes. The earlier
failed attempt remains recorded. New eligibility depends only on the selected
representation and its current source hash, never stale nested metadata.

All source models are sealed to disk before any new metrics for that case.
Only one endpoint is live at a time; scoring reconstructs and checks its hash.
Replay independently reconstructs, loads and verifies complete heads, signed
rows, original records, histories and metric objects. Every fresh split checks
the parent metrics exactly and independently counts raw changes, fixes,
regressions, and errors on queries that use signed features. The field readout
retains its original success flags. New raw exact-sequence goals use the
pre-readout `native_correct / examples`, guarded by the original single-target
field layout; ordinary sequence tasks retain their exact-sequence metric.
Every choice reports raw99/exact99 separately on all splits. Aggregate token
accuracy never replaces the per-task requirement.

The prospective bounds use the six already reviewed training-only inspections,
SHA256 `46f05261a1f87ac80b0dcf20021d1314975a9a4dae052c985b8c702b77bfcd8a`.
Their total construction forecast is 1,340,280,954,000 units per pass. The frozen
original input layout requires 56,448 metric/audit calls per endpoint: raw and
field metrics for both models, plus both independent raw audit products. At
1,789,280 units per call this is 101,001,277,440 units per endpoint. Three complete
construction passes and all metrics total 4,323,846,694,320 units. Reserving
128B per endpoint gives 4,404,842,862,000 units, within the **5T case allowance
fixed before original scoring**. Component limits remain 256B, buffers 56 GiB,
and process address space 64 GiB. The largest
paired endpoint buffer forecast is 22,586,405,888 bytes (about 21.04 GiB).
At least 16 GiB free storage is required. These are conservative work forecasts,
not measured FLOPs or proofs about the allocator. Repeated construction,
diagnostic optimization, original records and all metric/audit calls are charged.

All 22 adapter checks pass, using actual full/minibatch/forward source histories,
45 repeated records and a five-record tail. They exercise changed signed rows,
negative weights, complete complex metrics, held-out independence, disk sealing,
source/state mutations, both policies, historical regressions, native count
semantics and prospective limits. The added regression check verifies the
complete original field layout without predicting any new answer. The separately reviewed component supplies
the independent kernel solves, gradients and normalization checks. No original
signed-model accuracy is inferred from synthetic checks or the Lean bound.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/signed-compare
export TMPDIR=/run/media/misha/storage/transformer-runs/signed-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
cargo fmt --manifest-path experiments/convex_stress_signed_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_signed_compare/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_signed_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_signed_compare/Cargo.toml
prlimit --as=68719476736:68719476736 -- "$CARGO_TARGET_DIR/release/convex-stress-signed-compare" forecast REVIEWED316_INSPECTION.jsonl
experiments/convex_stress_signed_compare/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-signed-compare"
jq --arg producer "$(cat FRESH_FREEZE/producer.commit)" '.producer_commit=$producer' experiments/convex_stress_signed_compare/request.template.json > FRESH_FREEZE/request.json
experiments/convex_stress_signed_compare/run_original.sh FRESH_FREEZE/convex-stress-signed-compare FRESH_FREEZE/request.json FRESH_RUN
jq -f experiments/convex_stress_signed_compare/report.jq FRESH_RUN/summary.json
```

The request supplies this producer commit; absolute task314 source directory,
completed manifest and exact review files; task316 component review files;
task301 guarded reference directory, manifest and review files; and its earlier
failed-attempt directory/manifest. All identities are pinned by `protocol.json`.
The frozen source closure and executable must precede any original signed-model
score. Generated results stay on storage outside the temporary worktree.
The template also pins the failed first signed attempt and its complete manifest.
It records the current local evidence locations; replace its producer
field from the freeze as shown above. The native forecast reproduces the exact
4,404,842,862,000-unit reservation and 22,586,405,888-byte bound without scoring
any new model. It uses the reviewed inspection plus original generator input
layout; no held-out label or score selects a model or changes its parameters.
The full 13932-job Lean build, global 36sorry/0rests/0extra audit, native INDEX,
formatting, clippy, release and forbidden-source checks pass. No Lean source or
generated INDEX content changes in this adapter.
Task317's adapter review precedes task319's original evaluation. The initial
allowance correction and subsequent measurements are recorded below.

The initial task319 run at `1bbbe27f` attempted all35 cases. Its 34 unchanged
cases ran and replayed successfully. Fibonacci refused the old 64B metric
allowance before saving a new model or scoring a held-out answer, so a complete
success summary was refused. The failure exposes an incomplete prospective
check: it reserved a cap without verifying the actual raw/field/audit call count.
The corrected forecast above shares the runtime counting function and checks
each endpoint before scoring. The 49 kernels, selection, coefficients, data,
5T case cap and memory cap are unchanged. All324 files of the failed attempt
remain pinned and its complete35-case failure record is carried into new reports.
The failing process peaked at3,210,804 KiB RSS; it was not an OOM. An earlier
driver invocation failed because the shell file lacked an executable bit; that
log is preserved and both documented shell entry points are now executable.

## Frozen original findings

Producer `9c7370a593c078a994b06a12449d00b940d26fd0` was independently reviewed
and frozen in `signed-compare-v3` before the corrected `original-v2` run.
All35 cases and all35 saved-state replays finish successfully. The three fresh
Fibonacci endpoints reconstruct their complete states, original metrics and new
metrics exactly on replay. The other34 cases retain their reviewed314 states
and metrics, explicitly marked as inherited; they are not newly trained models.

| Fibonacci mode | Parent raw correct /4704 | New raw correct /4704 | New raw and native exact | Parent CE → new CE |
| --- | --- | --- | --- | --- |
| Full batch | 4609 | 4608 | 97.959184% | 0.164271130 → 0.115829730 |
| Minibatch | 4610 | 4611 | 98.022959% | 0.167655642 → 0.101307444 |
| Forward only | 4610 | 4611 | 98.022959% | 0.171368873 → 0.101307444 |

Both initializer policies keep35/35 original success flags in every mode and
on every original diagnostic split. Their per-task raw99 and native exact99
counts are34/35 in every mode, also34/35 when requiring every held-out split.
Fibonacci is the only miss. Its original decoded field accuracy is99.936224%
in all modes and remains a success; that readout is separate from the raw goal.
The new model therefore does **not** meet the user's99% per-task target.

The independent raw transition audit finds2 fixes and3 regressions in full
batch, and14 fixes and13 regressions in each of mini and forward. Every mode
uses signed features on360 validation inputs. Those inputs contain91 of the
96 full-batch errors and88 of the93 mini/forward errors; five errors remain
on unchanged routes. The substantial CE decrease has barely improved raw
decisions and worsens the full-batch count by one. The complete original
training metrics remain exactly unchanged, including CE9.626594307035853e-9.
No new target-realizability, query-residual or unconditional correctness
certificate is inferred from the lower CE.

Across all279952 original primary targets, full batch scores279856, mini279857
and forward279859 correctly, in both policies. These pooled accuracies near
99.966% do not replace the per-task miss. Mini retains two earlier sum-square
errors in addition to its93 Fibonacci errors. Every older raw/CE tradeoff,
the35 conventional transformer endpoints,336 depth outcomes, stronger guarded
reference, task298 lower-CE/higher-error result, and both prior refused attempts
remain in the report. The separate recorded DiffGPT reference is unchanged;
this is not a new matched-budget transformer experiment.

The new diagnostics execute128 full-batch gradient batches with10 effective
updates across the two roles,160 minibatch batches with160 nonzero trial
updates, and zero forward gradients/updates. Retained minibatch weights still
come from the best whole-training diagnostic epoch, including the initial
vertex when later epochs do not improve it. All original gradient histories
are retained separately. Head coefficients and the49-map bank are unchanged.

The actual complete-case forecast equals4,323,846,694,320 work units, below the
5T cap. Paired endpoint buffers are bounded at22,586,405,888 bytes. The original
run plus replay takes14m46.79s; Fibonacci construction/scoring takes4m24.46s and
its replay6m37.53s. Peak RSS is7,019,032 KiB (6.69 GiB), with no OOM. Rebuilds,
failed preparation, source checks and independent repeats are additional work.
All22 native checks and the full Lean/global axiom/INDEX audits pass.

The generated report has SHA256
`7632e96c882483211d31a2fc882990ffbe0d3fd77144424b353dc1bed2b18ebc`.
The331-artifact original manifest is
`9d483213ca4b6bff59cc1c96583bce10b98a3241938a35db0135c3e1de10f54f`;
118 JSON files cover the complete prepared inputs, states, outcomes and summary.
Evidence lives at `/run/media/misha/storage/transformer-runs/signed-compare/`
and the1077-input freeze under `.../freezes/signed-compare-v3`.
Independent review at `b99c845b` reproduced all118 JSON files and the complete
report byte-for-byte. All22 native checks and the direct Lean/global axiom/INDEX
checks passed. The fresh full run took14m50.38s, peaked at7,023,100KiB RSS
(6.70GiB), and exited0 under64GiB. Task319 is approved; the reviewed result
still misses the per-task99% target on Fibonacci. Review logs and the board
record are preserved under the freeze's `findings-review` directory.

The next raw-label repair, tasks329–330, will test a fresh bounded complex head
on shared input features, with ordinary CE applied to the actual observed
prediction logits. It must audit the existing Lean contract first and preserve
this result as a baseline. Tasks327–328 separately add the six original
copy/reverse/sort cases with repeated/unique symbols and true EOS generation.
The six reviewed retrieval/prefix cases remain100% in all modes/splits; no
combined41-task model or score is established, and124 additional catalog cases
remain unscored.
