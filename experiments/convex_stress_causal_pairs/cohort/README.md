# Complete original comparison adapter

This native adapter tests the CFt4/CD8 causal-pair component against the
complete reviewed quotient comparison. It preserves both initializer arms,
all 105 task/mode choices per arm, all 35 conventional transformer endpoints,
336 depth outcomes, earlier failures and every inherited cost. It has passed
15 synthetic adapter tests; no original causal-pair benchmark has run yet.
Task232 must finish and pass independent review before comparison and held-out
evaluation. Task236 adds training-only staging while that parent is running.

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

The staging commands construct the same fixed proposals from observed prefixes
and raw training labels. The numerical constructor takes only that configuration
and its input refusal, with no parent or held-out argument. The stage binds the
original suite, all 35 input records, configuration, committed producer, executable
and source fingerprint. It retains every actual complex state, training CE,
full/mini/forward outcome, refusal, failed replay and partial process artifact.
It produces no parent selection or held-out metric. Split fingerprints in its
inventory bind the eventual comparison data; they are not training features.

An eventual comparison can import a sealed stage only with the same frozen
producer and executable, the original reviewed quotient parent and the existing
comparison approvals. Import independently replays the actual staged model before
training-CE selection; held-out evaluation still follows the selection seal.
Staged construction time and derivative work are charged once across both arms,
including successfully staged models whose later import or metric evaluation
fails. Partial failed-stage costs remain in measured resources. For imported
cases, `total_seconds` adds the original construction time to the current import
and evaluation time; resource files measure each process separately. The nested
stage ledger preserves construction counts, while the comparison ledger counts
completed import/evaluation cases. No parent outcome or earlier cost is dropped.

The synthetic generation fixture needs a pair of features, emits the raw
label followed by EOS, and passes all its unchanged metrics in all three
modes. A new uncovered prefix retains 100% training accuracy but falls to
75% test sequence accuracy and EOS rate, with nonzero unknown-call masks.
Other checks retain all 35 cases when 33 processes fail, preserve a missing
forward parent, reject changed states/metrics/masks/costs/review pins, and
verify that held-out answers and task names cannot change discovery inputs
or the training-CE selector. These are fixtures, not original benchmark scores.
Seven staging checks also remove the comparison parent entirely, compare fresh
and staged models/choices/metrics/masks, and reject altered numerical states even
after all local hashes are recomputed. Input, binary, source, protocol, timing,
resource and partial-artifact mutations fail their respective bindings. Failed
imports preserve both complete parent arms and all staged costs.

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
bash -n run_original.sh freeze_original.sh run_training.sh freeze_training.sh
prlimit --as=68719476736:68719476736 -- cargo build --locked --release
```

Freeze a committed producer and its release binary under the root's ignored
`.kanban/artifacts/`, retaining source, binary and both protocol hashes before
any original construction. Preserve the earlier prospective-v1 binary. Set
`CAUSAL_PAIRS_BINARY` to the new frozen executable for every command below.
Use absolute run roots, and create their parent directory first. Both launchers
retain a 16 GiB disk reserve and the 64 GiB address-space cap.

After tasks233,235,236 are approved, from the repository root:

```sh
bash experiments/convex_stress_causal_pairs/cohort/freeze_training.sh \
  REPOSITORY_ROOT PRODUCER_COMMIT NEW_TRAINING_REQUEST
bash experiments/convex_stress_causal_pairs/cohort/run_training.sh \
  NEW_TRAINING_REQUEST FRESH_TRAINING_ROOT
```

This runs all 35 original constructions and their independent replay serially,
then seals `summarize-training ROOT`. Keep this complete directory unchanged.
After tasks232–234 are approved, import it with the same producer and executable:

```sh
bash experiments/convex_stress_causal_pairs/cohort/freeze_original.sh \
  REPOSITORY_ROOT PRODUCER_COMMIT NEW_REQUEST SEALED_TRAINING_ROOT
bash experiments/convex_stress_causal_pairs/cohort/run_original.sh \
  NEW_REQUEST FRESH_RUN_ROOT
```

Omitting `SEALED_TRAINING_ROOT` retains fresh construction in the comparison
run. Staging never bypasses the original parent gates. Generated outputs stay
local and ignored. An independent review must replay every completed case
again, retain failures, regenerate the applicable summary byte-for-byte, and
run native and Lean checks at the submitted commit before any pass claim.
