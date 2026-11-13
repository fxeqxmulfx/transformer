# Compare refined complex heads on the original stress suite

Do the CM31/CM32 heads improve the complete original comparison when selection
still uses ordinary raw-label training cross-entropy? This native adapter imports
saved models; it performs no discovery or training. Seven native tests cover
actual complex-state import and the comparison contract. No original benchmark
has run through this adapter yet, so the reviewed scores remain 34/35 full,
34/35 mini and 33/35 forward. The conventional transformer remains 33/35.

DiffGPT's separately recorded selected `phase-score-lerope` model passes 35/35
on each of seeds 0, 1 and 2, totaling 105/105. Its committed
`research/main_model/model.json` records 75 short and 30 long successes and zero
audit failures. That is the existing reference, not a fresh replay or the
conventional transformer endpoint in this comparison. This adapter retains our
original seed-1 data and does not claim a new three-seed result.

## Immutable inputs and scope

| Imported evidence | Frozen producer | Use |
| --- | --- | --- |
| Complete causal comparison, including quotient and earlier parents | `b516af90` | Both initializer arms, 105 task/mode choices each |
| Refined quotient training stage | `4f210282` | All 23 pair tasks and their 55-prime source catalogues |
| Preserved causal training stage 237 | `b516af90` | Exact reviewed summary/inventory already imported into the causal parent |

`protocol.json` pins the complete source fingerprints, executable hashes and
causal stage hashes. A request additionally pins both complete summaries and
inventories, the independent review summary bytes, exit file and reviewed commit
file. Preparation replays the foreign summaries with their frozen executables,
copies their complete metadata, and inventories every file. The same checks run
when rebuilding the final summary. Frozen 232/237/243 sources and binaries stay
unchanged.

The complete parent retains all 35 tasks in both initializer policies, all three
training modes, 336 depth outcomes and 35 conventional transformer endpoints.
Every training stage, refusal, failed process and partial resource artifact
remains visible. A task without a refined proposal retains its previous choices.
A failed import or metric process also retains all six previous task/mode/arm
choices, its partial files and all already charged construction costs.

The inherited native generator rebuilds the original inputs and data hashes.
For a refined candidate, training order, raw labels, vocabulary and original
minibatch size must match the saved discovery and refined stage. Actual complex
coefficients, initialization, CE trajectory, audits and mode refusals are checked
before any offer reaches the selector. Independent numerical checks reject
altered coefficients even after file hashes are recomputed.

## Selection and declared success

The two initializer policies remain separate. For each task and mode, the old
selected endpoint is considered first, followed by refined candidates in
increasing prime order. Replacement requires an ordinary raw training CE
decrease greater than `1e-12`; no held-out score or certificate breaks a tie.
Offers and selected references are sealed before metrics are evaluated.

Declared readout compatibility is only a candidate eligibility check. It does
not change model features, representatives, updates or raw-label CE. Original
field metrics still sum probabilities over their existing modulo-97 groups,
including unequal group sizes. No arithmetic answer rule enters construction.
Every available candidate is evaluated, including ineligible candidates; only
eligible candidates can replace the parent.

`primary_pass` uses the original primary split's sequence accuracy threshold of
99%. `full35_pass` requires that primary success for every retained task/mode
choice in an arm. The per-mode counts expose all 35 cases separately. Auxiliary
split scores, `all_splits_pass` and `all_split_diagnostic_pass` remain diagnostics;
an auxiliary failure does not revoke primary success. Historical parent fields
remain byte-for-byte in the complete imported parent.

All original splits are scored from actual complex forward multiplication.
Each prediction records whether its categorical axes and latent support were
observed, and separately whether its exact input belongs to the training
dictionary. A supported default is still an unobserved input; neither mask proves
its target label. The fixture deliberately has a correct pooled answer and an
incorrect unseen raw alias, alongside a failed auxiliary split.

CM31/CM32 prove fixed-feature representation, bounds and feasible updates. They
do not establish arbitrary unseen raw-label identification, target readout
compatibility, global discovery convexity or optimizer convergence. The adapter
imports the already reviewed model and makes no additional Lean claim.

## Resources and verification

The new metric allowance is prospectively 32 billion estimated scalar operations
per available endpoint across all original splits. It is distinct from the old
quotient's 2-billion training allowance and is a refusal threshold, not measured
FLOPs. Malformed or out-of-domain metric inputs fail before model indexing.
All numerical processes run serially with single-threaded BLAS and a 64 GiB
address-space cap. The runner keeps a 16 GiB disk reserve. Preparation,
evaluation, numerical replay and summary resource records remain separate.
Every prior discovery/training cost is retained and charged once across both
arms, including costs from failed imports. This is not matched-compute training.

The seven tests exercise full 35-task fixtures, actual unequal-group pooling,
unknown aliases and axes, independent initializer choices, CE ties and prime
order, missing modes/models, real per-mode preflight refusals, parent-scope and
review mutations, process failures, altered metrics/masks and rehashed numerical
corruption. These are implementation checks, not original benchmark passes.

```sh
cd experiments/convex_stress_refined_compare
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked -- --test-threads=1
prlimit --as=68719476736:68719476736 -- \
  cargo clippy --offline --locked --all-targets -- -D warnings
bash -n run_original.sh freeze_original.sh
cargo build --offline --locked --release
```

Original execution requires approved tasks 232, 234, 237, 243, 244 and 245.
Task246 tracks that separate original run and its complete evidence review.
Run `bash freeze_original.sh REPO PRODUCER PARENT_ROOT PARENT_BINARY
PARENT_REVIEW STAGE_ROOT STAGE_BINARY STAGE_REVIEW NEW_REQUEST` to bind the
reviewed inputs. Each review directory supplies `process.exit`, `reviewed.commit`
and `summary.json`. Preserve the committed source fingerprint, executable,
request, protocol and hashes outside the worktree before running. Set
`REFINED_COMPARE_BINARY` to that frozen executable, then run
`bash run_original.sh REQUEST FRESH_ROOT`. All generated evidence stays ignored
under the root's experiment `runs/` directory. Independently rerun every complete
case with `replay-case ROOT TASK` and reproduce `summarize ROOT` exactly before
approving the separate original evaluation.
