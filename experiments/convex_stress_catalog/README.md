# Extend the GPT stress suite to the per-task 99% goal

The user's 2026-11-14 request extends the current convex-model study to
`stress_tests_for_gpt.md` and targets at least 99% held-out accuracy per task.
This native inventory reconciles all 165 catalog cases with the original plan,
the historical GPT snapshot and the existing 35-task manifest. It identifies
130 additional cases. No model is trained or evaluated by this inventory.

As of 2026-11-16, the latest independently verified
[`convex_stress_probe_compare`](../convex_stress_probe_compare/README.md) adds
conditional amplitudes and order discrimination. Both reach 100% raw and exact
accuracy on every original split in forward-only and minibatch modes; three
older controls also remain at 100%, and routing keeps its teacher-data refusal.
All 66 artifacts match an independent repeat with full raw-score replay.
The historical 77-case union therefore becomes 79/165 under the user's current
two-mode requirement, across separate seed 1 cohorts. The older three-mode total
remains 77. Full batch is optional for new work, the aggregate RAM limit is 110 GB,
and shortage on either disk stops a run. Earlier experiment descriptions below
retain their historical protocols and counts.

The catalog and original plan are pinned to DiffGPT commit
`5a37c515639f23037d191b0659a5a0853984328d`. `protocol.json` pins the SHA-256 of
each input and keeps the historical model identity separately. The historical
snapshot has 48 target crossings, 78 budget misses, 29 pending cases, nine
user-requested skips and one stopped case. Only 40 of the 48 crossings retain
their original target at the final observation. These are older GPT outcomes;
every expanded-model result in the generated inventory starts as null.

Raw-token accuracy, exact-sequence accuracy and ordinary cross-entropy remain
separate. For a sequence case, the new 99% goal requires both raw and exact
sequence accuracy to reach 99% on its held-out selection split. Each original
metric and success flag is retained independently, including the historical
95% sequence thresholds. Numeric error, forecasting skill and language-model
CE targets are not converted to an accuracy percentage. Aggregate accuracy
does not substitute for the per-task goal.

The current 35-task results and their conventional and recorded DiffGPT
baselines remain intact. Their original success flag is 35/35 in every mode;
Fibonacci's approximately 98% raw accuracy remains below the new raw target.
The signed-kernel component at `99a42652` improves training-only diagnostics,
but has no new held-out accuracy result yet. Its separate original35 comparison
and this expansion are distinct experiments.

The later reviewed task334 measures16 of the130 additional cases: six prefix
cases, six copy/reverse/sort cases, and four parity/scratchpad variants. All16
reach100% raw and exact sequence accuracy on every tested short/long split in
full, minibatch and forward modes. Its independent100-artifact repeat is exact;
the12 earlier controls and every prior refusal are retained. There are114
additional cases still unmeasured. See
[`convex_stress_prefix_programs`](../convex_stress_prefix_programs/README.md)
for the frozen source, metrics, costs and explicit unseen-copy-prior limitation.
This is a separate expanded cohort, not a unified51-case model evaluation.

| First additional case | Original data | Longer probes | Original / new accuracy targets |
| --- | --- | --- | --- |
| MQAR | Four writes, two queries, 64 keys and 64 values; length 24 | 48 and 96 | 95% exact / 99% raw and exact |
| Two-hop lookup | Four cyclic links, two queries, 64 identities; length 24 | 48 and 96 | 95% exact / 99% raw and exact |
| Dyck prefixes | One bracket type, neutral letters, bounded generation balance; lengths 12–16 | 32 and 64 | 95% exact / 99% raw and exact |
| Alternating blocks | Three alternating blocks with neutral letters; lengths 8–16 | 32 and 64 | 95% exact / 99% raw and exact |

Each case retains 128 training, 32 validation and 64 test examples, plus
32-example validation and 64-example test probes at each longer length.
Retain clean training rows and novel-input subsets. Selection uses the original
novel-validation split when nonempty. These four cases label supplied input
prefixes; every supervised token and every complete sequence is scored.
Generative cases added later must use their original free-generation scoring
and termination rules. The exact first-cohort definitions are fixed in the
protocol, read beside the original domain defaults and source generators.

The next native data port must reproduce the RNG, draw order, vocabulary,
supervised masks, split identities and fingerprints before a model run. Read
historical Python sources only as reference. Task IDs and answer formulas
belong in generators and scorers; they must never choose predictions in the
learned model. Train full batch, minibatch and forward only with ordinary CE,
preserve original repeated records and short tails, and separately count all
construction, optimization and evaluation work. Keep the 64 GiB process limit
and store generated data, states and reports on the storage disk.

Three native checks validate the complete frozen inputs and reject missing or
duplicate cases, changed targets, changed history, unknown existing cases,
duplicate catalog rows and incorrect source hashes. The inventory preserves
all original case parameters, historical records, source sections, recorded
test-split names and the existing35 metadata. It records controls explicitly
without assigning them invented passes. Original skipped cases retain that
status and none belongs to the first cohort.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2
export STRESS_REFERENCE_ROOT=/home/misha/python_projects/diff-gpt
cargo fmt --manifest-path experiments/convex_stress_catalog/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_catalog/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_catalog/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo run --release --offline --locked \
  --manifest-path experiments/convex_stress_catalog/Cargo.toml -- \
  stress_tests_for_gpt.md "$STRESS_REFERENCE_ROOT/research/current_grokking/plan.json" \
  experiments/convex_stress/suite.json \
  "$STRESS_REFERENCE_ROOT/research/current_grokking/results/progress_snapshot.json" \
  FRESH_INVENTORY.json
```

The executable refuses an existing output path. Inputs can instead be read
from the preserved source snapshot on storage; their four content hashes
remain mandatory. The historical reference files are available from the pinned
DiffGPT commit at the paths above. Commit the generator, protocol, dependency
lock, source identities and these authored findings; keep generated inventories
and copied historical reports ignored outside the worktree.
