# Recover the four interrupted full-suite fits

Can the unchanged full-suite protocol be completed after the host OOM without
discarding its earlier evidence? This follow-up belongs to board task #204.
Lean CM5–CM8 already covers the original model domains and prediction hull;
the recovery adds no new model family or mathematical claim.

The original `suite-v1` has 311 terminal records from 315 planned candidate
slots, and its 236 returned models have successful independent audits. Four
training processes were terminated without saved solver states. Their partial
traces remain in the original directories and in task #177's interruption
archive. Recovery starts these four cases afresh in a separate directory.

| Training mode | Original task | Candidate |
| --- | --- | --- |
| Forward-only | `recurrence-fibonacci-k1-c2-lgt` | `full_rank` |
| Minibatch | `basis-recall-easy` | `counts` |
| Minibatch | `recurrence-fibonacci-k1-c2-lgt` | `full_rank` |
| Minibatch | `recurrence-fibonacci-k1-c2-lgt` | `rank_one` |

`protocol.json` binds the original producer `ff368fd5`, its binary and source
fingerprints, original metadata and interruption manifests, numerical libraries
and the repaired independent auditors. It changes no learning rule, initialization,
seed, batch, tail, update cap, stopping rule, loss or evaluation. New processes
use the user's **64 GiB** address-space cap, one numerical process at a time and
one BLAS thread. The producer retains its original eight frequency workers.
The process cap bounds failure; it is not a mathematical upper bound on every
sparse allocation or a guarantee that a run will finish within memory.

Before training, the harness verifies the exact 311 retained cells and four gaps,
all retained result digests, model/state bindings, replay identities and successful
independent audits. It freezes hashes of this evidence, including the interrupted
traces. Started attempts cannot be overwritten or resumed. Every attempt retains
its exit, stdout, stderr and `/usr/bin/time` resources. Training failures remain
recorded while the other planned attempts continue. Every returned endpoint is
replayed and independently audited before its evidence is sealed.

After all four succeed, a separate `view/` links the 311 original results and
four new results. The unchanged native selector recomputes training CE, makes
all 105 choices over the original 35 tasks and three modes, and replays that
selection. All prior model failures and ineligible cases remain in this view.
Completion of the inventory does not imply a benchmark pass. A missing or failed
recovery prevents a complete comparison from being reported.

## Checks and execution

The native inventory check accepts a 311-cell metadata fixture and refuses ten
mutations: missing, duplicate or extra/unplanned identities; nonterminal status;
changed source, protocol, data, model binding, vocabulary or training mode.
Preparation also rejects an existing destination and a missing committed harness
identity. These tests construct no original examples and run no learner.

```sh
bash -n experiments/convex_stress_hull/recovery/run.sh
bash -n experiments/convex_stress_hull/recovery/test.sh
bash experiments/convex_stress_hull/recovery/test.sh
bash experiments/convex_stress_hull/recovery/run.sh preflight
```

Run from the root checkout. Copy the committed `recovery/` sources to a durable
ignored artifact directory before preparation: the inventory pins their paths
and hashes, so they must survive worktree removal. The old producer and auditor
binaries are reused from the exact artifact paths in the protocol. Their source
commits and earlier reproduction commands are documented in the parent README.
No original output is copied over or edited.

```sh
HULL_RECOVERY_COMMIT=$(git rev-parse HEAD) \
  bash .kanban/artifacts/204/harness/run.sh prepare \
  experiments/convex_stress_hull/runs/recovery-v1
bash .kanban/artifacts/204/harness/run.sh run \
  experiments/convex_stress_hull/runs/recovery-v1
bash .kanban/artifacts/204/harness/run.sh verify \
  experiments/convex_stress_hull/runs/recovery-v1
```

`finish ROOT` is available if all four attempts and audits completed but no view
has been created. It refuses an existing view. `verify ROOT` checks the retained
and new evidence hashes and repeats the original selection without overwriting
the frozen comparison. Generated evidence remains ignored under the root runs
and `.kanban/artifacts/204/`; authored findings will be recorded here.
