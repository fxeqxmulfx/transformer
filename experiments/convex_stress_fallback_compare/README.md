# Complete comparison of fixed support fallback

Can the CM38 fallback preserve useful saved answers outside a selected causal
head's observed support on the complete original stress suite? This adapter
imports the reviewed native core at `027237cd` and the complete grouped-kernel
comparison from task261. Its original benchmark execution is a separate step
(task267); the fixture results below are not benchmark passes.

| Arm | Model choice | Prediction |
| --- | --- | --- |
| Original minimum-CE initializer | Exact task261 choices | Exact retained metrics and models |
| Original first-small-CE initializer | Exact task261 choices | Exact retained metrics and models |
| Transformed minimum-CE initializer | Wrap each importable selected causal head | Candidate on observed support; its frozen predecessor elsewhere |
| Transformed first-small-CE initializer | Same rule within this initializer | Same fixed support rule |

Every arm retains all 35 tasks and 105 task/mode choices. The original 35
conventional transformer endpoints and 336 depth outcomes remain in the complete
ancestor evidence. Primary sequence accuracy of at least 99% is success;
auxiliary splits remain diagnostics. The recorded DiffGPT 35/35 on seeds 0, 1
and 2 remains a separate historical reference.

The adapter does not change ordinary-CE head selection or use held-out scores
to decide whether to wrap a model. The structural transformation is declared in
`protocol.json`, then sealed before its metrics. Exact training-logit equality
and raw CE are checked on every original record, preserving order, multiplicity
and labels. Full, cyclic minibatch and forward histories come from the same
saved causal stage and are replayed numerically. There are no new optimizer
updates or discovery operations. Replaying old histories does perform their
verification computations; the process resource records retain this cost.

The complete parent, stage, source fingerprints, exact executables, review
records, data fingerprints and original checkpoint hashes are bound. A saved
predecessor path only locates bytes already identified by the ancestor choice;
both absolute and checkpoint-root-relative paths retain that hash binding.
It never chooses a model by task name. The current core accepts typed Causal
and Counts checkpoints. Missing checkpoints, unsupported families, prediction
domains and work limits remain explicit refusals. Failed processes retain all
six original choices for that task and their partial files.

Both actual heads and their storage are charged. Causal models retain the
original 4,096-token prediction domain even when the training feature library
is shorter. Counts retain their saved count-value domain. The metric forecast
charges both heads and the original pooled-field and greedy calls. Each split
retains the supported mask, parent-route mask and exact-observed-input mask in
prediction order. These expose supported unseen aliases as well as unsupported
inputs. CM38 proves preservation outside support; it does not prove that all
supported unseen answers are correct.

The numerical checks exercise actual imported histories and heads, both full
initializer policies, all 35 synthetic task slots, long prefixes, unchanged
raw CE, primary success despite auxiliary failure, checkpoint and rehashed
parameter/support/input corruption, process failures, missing predecessors,
exact Counts loading and disk-refusal continuation. The immutable parent arm
and every ancestor cost survive these cases. Original outcomes are pending.

The source ports comparison bookkeeping from `d341fdf8` and supervised-prefix
assembly and causal history replay from `b516af90`. It uses the CM38 bounded
complex construction at `16e34b2f`; the routed feature matrix is not asserted
to be unitary. Flat component codes supply a feature bound, not unseen labels.

Run the native checks and build on the storage disk:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/fallback-compare
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path experiments/convex_stress_fallback_compare/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_fallback_compare/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_fallback_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release \
  --manifest-path experiments/convex_stress_fallback_compare/Cargo.toml
LAKE_ARTIFACT_CACHE=false prlimit --as=68719476736:68719476736 -- lake build
LAKE_ARTIFACT_CACHE=false prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/Axioms.lean
```

Before original execution, use `freeze_source.sh REPO PRODUCER NEW_DIRECTORY`
to archive the committed local dependency graph, borrowed original modules,
locked metadata and compiler versions. Copy the exact release executable into
the freeze directory and record its SHA256 and `fingerprint` output. After all
board dependencies have passed review, use `freeze_original.sh REPO PRODUCER
PARENT_ROOT PARENT_BINARY PARENT_REVIEW STAGE_ROOT STAGE_BINARY STAGE_REVIEW
CHECKPOINT_ROOT NEW_REQUEST` to freeze immutable inputs. The original Counts
checkpoints currently reside in `experiments/convex_stress_hull/runs/suite-v1`;
their exact saved hashes decide which bytes are acceptable.

Set `FALLBACK_COMPARE_BINARY` to the frozen executable and run `run_original.sh
REQUEST FRESH_ROOT` on the storage disk. The script enforces a 64 GiB address
space limit and 16 GiB disk reserve, retains every run/replay exit and resource
record, then rebuilds the complete summary. It cannot start before task261 and
the native adapter review pass. Preserve all generated evidence outside the
worktree. Commit source, protocol and authored interpretation only.
