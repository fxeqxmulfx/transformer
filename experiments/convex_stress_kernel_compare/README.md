# Compare complete-kernel complex heads on the original stress suite

Do CM34's complete vector codes improve the original stress comparison when
selection still uses ordinary raw-label training cross-entropy? This Rust
adapter imports saved discovery and head states without constructing or
training a new candidate. Eight native tests cover the comparison contract.
Original execution has not started: the reviewed primary counts remain 34/35
full, 34/35 mini and 33/35 forward, with the conventional transformer at 33/35.

The separate DiffGPT reference records 35/35 on each of seeds 0, 1 and 2:
105/105, comprising 75 short and 30 long successes, for the selected
`phase-score-lerope` model in `research/main_model/model.json`. These are its
recorded results, not a new numerical replay. This comparison keeps the original
seed-1 data and conventional transformer endpoints; it makes no new three-seed
claim.

## Immutable evidence

| Input | Frozen producer | Scope |
| --- | --- | --- |
| Complete refined comparison | `a0046702` | Both 105-choice initializer arms and every earlier parent |
| Complete-kernel training | `166bebc3` | All 23 original pair tasks, 55 primes each, full/mini/forward |
| Preserved causal training | `b516af90` | Reviewed task237 summary and inventory inside the parent |
| Preserved refined training | `4f210282` | Reviewed task243 summary and inventory inside the parent |

`protocol.json` pins sources, executables, producers and the two completed
earlier training stages. A request additionally pins the complete parent and
kernel summaries, inventories, review summary bytes, review exit file and
reviewed commit file. Preparation and final summarization replay both foreign
summaries with their frozen executables and compare the exact bytes. All
source files, resources and failed processes are retained in complete artifact
inventories. Comparison results never replace or rewrite the training evidence.

The original parent includes 35 tasks, three modes, two independent initializer
policies, 336 depth outcomes and 35 conventional transformer endpoints. Every
kernel stage and its discovery/head errors, refusals and costs remain in the
imported summary. A task with no eligible candidate retains its parent choices.
A failed import or metric process also retains all six prior choices and every
partial artifact. All discovery/training costs are charged once across the two
arms, even when the imported candidate is unavailable or a later process fails.

## Actual models and comparison rule

The inherited native data generator verifies the original input fingerprints.
Original observation order, raw labels, vocabulary, minibatch size and policies
must match each kernel stage. Failed construction cases also retain this input
binding. Every saved prime is in the fixed catalogue and has a unique canonical
file. The independent checker verifies the original equations, complete basis,
actual vector codes, complex coefficients, initialization, ordinary CE trajectory
and mode refusals before any candidate offer. Changing a coefficient or removing
a basis direction still fails after its enclosing file hashes are recomputed.

For each task and mode, each initializer arm considers its own parent endpoint
first, followed by kernel candidates in increasing prime order. Replacement
requires a raw-label training CE decrease greater than `1e-12`. The selector
uses no held-out score or certificate tie-breaker. Offers and selected references
are saved before the first held-out metric is evaluated.

Declared readout compatibility is an eligibility test on the actual vector-code
class fibers. It does not change features, the representative rule, optimization
or raw-label CE. All available heads are evaluated, including ineligible ones.
The original field metric continues to sum probabilities over its modulo-97
groups, including unequal group sizes. No arithmetic answer rule is added to
construction.

The original primary split's sequence accuracy threshold of 99% determines
`primary_pass`. Each per-mode count covers all 35 tasks; `full35_pass` requires
primary success for all 105 task/mode choices in that initializer arm. Auxiliary
split failures remain diagnostics and do not revoke primary success. Historical
parent summaries remain exact in `complete_parent`.

Predictions use the saved complex head and retain three separate masks: both
axes encoded, the full vector code matching an observed class, and the exact
input belonging to the training dictionary. Unknown axes and unmatched active
codes have distinct counts. A matched default still need not identify an unseen
raw label. The main fixture uses a two-dimensional kernel, a correct pooled
answer with an incorrect raw alias, an unobserved exact input, and a failed
auxiliary split. It exercises real full/mini/forward coefficients and 55 actual
kernel constructions; nonselected fixture heads have explicitly simulated
execution errors, rather than invented trained states.

CM34 proves complete-code matching under explicit additive realizability and
readout compatibility hypotheses, along with bounded fixed-feature complex
heads. The executable does not establish those hypotheses for the unknown
target. Observed low CE and complete-basis checks do not prove arbitrary unseen
answers, discovery convexity or optimizer convergence. This adapter adds no
Lean claim.

## Resources and reproduction

The prospective metric cap is 32 billion estimated scalar operations per
endpoint across all original splits. It is a refusal threshold, not measured
FLOPs. The inherited construction caps and actual measured costs remain
unchanged. Malformed or out-of-domain metric inputs fail before model indexing.
All numerical processes use a 64 GiB address-space limit and single-threaded
BLAS settings. A 16 GiB disk reserve is checked before preparation and each case.
An unavailable case records exit125 and zero new evaluation, then the runner
continues to later cases and the complete summary. A shell fixture tests both
that transition and refusal before preparation. Preparation, evaluation, replay
and summary resource records are separate. This is not matched-compute training.

The eight tests cover actual multidimensional and aliased heads, full synthetic
35-task/105-choice scope, original pooling and primary success, CE tie order,
independent parent arms, missing models, per-mode refusals, process failures,
review/input/scope mutations, altered metrics and masks, rehashed coefficients
and missing basis directions, and low-disk continuation. Synthetic passes do
not constitute original benchmark results.

```sh
cd experiments/convex_stress_kernel_compare
prlimit --as=68719476736:68719476736 -- cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked -- --test-threads=1
prlimit --as=68719476736:68719476736 -- \
  cargo clippy --offline --locked --all-targets -- -D warnings
bash -n run_original.sh freeze_original.sh
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release
```

Original execution requires approved tasks 232, 234, 237, 243, 244, 245, 246,
250, 251, 252 and 253. In particular, both complete kernel training252 and the
complete refined parent246 must finish their reviews before metrics start.
Freeze the reviewed executable, protocol, scripts and source inventory outside
the worktree. Run `bash freeze_original.sh REPO PRODUCER PARENT_ROOT
PARENT_BINARY PARENT_REVIEW STAGE_ROOT STAGE_BINARY STAGE_REVIEW NEW_REQUEST`
to bind the reviewed evidence. Each review directory supplies `process.exit`,
`reviewed.commit` and `summary.json`. Set `KERNEL_COMPARE_BINARY` to the frozen
executable and run `bash run_original.sh REQUEST FRESH_ROOT`.

Keep generated outputs ignored under the root's experiment `runs/` directory.
Independently rerun every completed case using `replay-case ROOT TASK` and
reproduce `summarize ROOT` byte-for-byte before reviewing the separate original
evaluation. Preserve the declared primary decision even if an auxiliary split
fails, and report any regressions or process failures with the complete parents.
