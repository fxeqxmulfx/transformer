# Recover the four interrupted full-suite fits

The frozen recovery now completes and independently audits all four fresh
attempts. Its separate comparison contains all 315 planned slots and all 105
training-CE choices, with no replacement of the original 311 records or their
interrupted traces. The unchanged selector and its replay both finish normally.
Inventory completion does not produce a full benchmark pass. Passes use the
original 99% exact-sequence threshold:

| Mode | Primary split passes | Every split passes | No selected model |
| --- | ---: | ---: | ---: |
| Full batch | 22/35 | 21/35 | 0 |
| Minibatch | 22/35 | 21/35 | 0 |
| Forward only | 21/35 | 20/35 | 3 |

The final inventory retains 240 returned models, three fit failures and 72
ineligible slots. These are the original 236 audited models plus the four
separately audited recoveries. All cases, including failures, remain available
to the later depth/decoder/transformer comparisons; this is the unchanged
baseline rather than a completed integration of those follow-ups.

| Fresh attempt | Updates | Training CE | CE excess | Validation sequence accuracy |
| --- | ---: | ---: | ---: | ---: |
| Forward recurrence, full rank | 128 | 5.109154052 | 0.557286995 | 3.6352% |
| Minibatch Basis recall, counts | 512 | 0.024742101 | 0.000211685 | 100% |
| Minibatch recurrence, full rank | 512 | 4.916851861 | 0.364984804 | 0.0850% |
| Minibatch recurrence, rank one | 512 | 4.617273734 | 0.065406676 | 99.7662% |

Basis recall also reaches 100% on its test and novel-input splits. The recurrence
rank-one fit reaches 100% training classification but retains positive CE
excess. The fixed training-CE selector therefore chooses the earlier count
candidate for that task: its training CE is 4.551972796 and its validation
sequence accuracy is zero. This exposes a selection failure as well as the
other model failures; the better validation score was not used to change the
choice or extend the frozen runs.

Fresh training takes 733, 14,770, 2,868 and 2,751 seconds respectively. Peak
training RSS is 15,914,720 KiB, about 15.2 GiB, under the 64 GiB process cap.
Producer and audit binaries, native library hashes, all four endpoint/state
digests, replay seals and the complete comparison hashes remain rooted in
`runs/recovery-v1` and `.kanban/artifacts/204`. The harness was frozen at
`a79c64b7` before these fresh attempts. The metadata guards still accept the
valid 311-cell fixture and reject all ten mutations plus both unsafe
preparation requests.

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
and `.kanban/artifacts/204/`; the findings above describe that retained evidence.
