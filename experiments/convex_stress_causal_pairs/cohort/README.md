# Complete original comparison adapter

This native adapter tests the CFt4/CD8 causal-pair component against the
complete reviewed quotient comparison. It preserves both initializer arms,
all 105 task/mode choices per arm, all 35 conventional transformer endpoints,
336 depth outcomes, earlier failures and every inherited cost. It has passed
eight synthetic adapter tests; no original causal-pair benchmark has run yet.
Task232 must finish and pass independent review before the original run.

| Stage | Fixed behavior |
| --- | --- |
| Inputs | Rebuild the original generators and verify every frozen split fingerprint; use unchanged supervised raw labels |
| Eligibility | At least one supervised training prefix has length greater than two; this input-shape rule reads no task identifier or held-out answer |
| Construction | Use the component's declared feature order, search caps, complex head and full/mini/forward policies |
| Selection | Keep each initializer's parent endpoint first; replace only for raw training CE improvement greater than `1e-12` |
| Evaluation | Seal references first, then run unchanged complete-split metrics, original greedy rollout and EOS behavior |
| Unknown inputs | Retain a Boolean coverage mask for every teacher-forced and generated prediction call; coverage is not a correctness certificate |
| Refusals | Retain shape, storage, discovery, head, training, audit and process failures, including partial artifacts and available modes |
| Accounting | Charge a new construction once across both policy arms; retain prior costs and measured process time/RSS |

The shape rule restricts these new proposals to causal-context tasks. The
categorical two-token tasks keep their reviewed parent models. Every task
still appears in the comparison and the process ledger. The component has no
task identifier, answer formula, readout hint or held-out input. The training
context is the longest observed supervised prefix; it does not constrain
prediction to that length. The model's fixed prediction limit is 4096 tokens.

The adapter reuses the original native generators and metrics through
`convex-stress-relabel`; artifact I/O and its inherited source fingerprint
come from the quotient cohort at `49105730`. Local component, adapter, lockfile,
protocol and launcher bytes are also embedded in the source fingerprint. The
complete process runs under a 64 GiB address-space cap. The metric forecast
has a two-billion-operation gate; a failed evaluation leaves the attempted
model and selection artifacts and preserves the parent choices in the final
comparison. Partial failures never become omitted tasks.

Preparation and final summary re-run the frozen quotient parent's summary
command and require byte identity with the independent review. They also
hash every retained parent file. Per-case execution checks the pinned parent
summary, inventory, binary, review and local snapshots. Actual causal replay
regenerates original inputs, independently checks discovery and CE steps,
then recomputes original metrics and every coverage mask from saved complex
heads. The complete report retains successful and failed process resources.

The synthetic generation fixture needs a pair of features, emits the raw
label followed by EOS, and passes all its unchanged metrics in all three
modes. A new uncovered prefix retains 100% training accuracy but falls to
75% test sequence accuracy and EOS rate, with nonzero unknown-call masks.
Other checks retain all 35 cases when 33 processes fail, preserve a missing
forward parent, reject changed states/metrics/masks/costs/review pins, and
verify that held-out answers and task names cannot change discovery inputs
or the training-CE selector. These are fixtures, not original benchmark scores.

From this directory:

```sh
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --locked -- --test-threads=1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --locked -p convex-stress-causal-pairs -- --test-threads=1
cargo clippy --locked --all-targets -- -D warnings
bash -n run_original.sh freeze_original.sh
cargo build --locked --release
```

After tasks232–234 are approved, freeze a committed producer and its release
binary under the root's ignored `.kanban/artifacts/`, retaining source,
binary and protocol hashes before any original construction. From the
repository root, `bash experiments/convex_stress_causal_pairs/cohort/freeze_original.sh
REPOSITORY_ROOT PRODUCER_COMMIT NEW_REQUEST` writes a request pinned to the
reviewed quotient parent. Set `CAUSAL_PAIRS_BINARY` to that frozen executable
and run `bash experiments/convex_stress_causal_pairs/cohort/run_original.sh
NEW_REQUEST FRESH_RUN_ROOT`. The launcher requires all three board approvals
and runs all 35 cases and their replay serially. Generated outputs stay local
and ignored. An independent review must replay every completed case again,
retain failures, regenerate `summarize-original ROOT` byte-for-byte, and run
the native and Lean checks at the submitted commit before any pass claim.
