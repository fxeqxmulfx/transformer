# Fixed support fallback closes the earlier forward-only gap

The complete task267 run finishes all 35 original cases and their 35 numerical
replays with zero process failures. Both initializer policies now retain 35/35
primary successes and 35/35 auxiliary diagnostics in full batch, minibatch and
forward-only. The unchanged task261 parent had 34/35 in forward-only. Exact
findings-commit independent review is pending. This earlier fallback branch is
separate from the newer raw-region and convex-exclusion comparisons.

| Mode | Parent primary successes | Fallback primary successes | Fallback auxiliary successes |
| --- | ---: | ---: | ---: |
| Full batch | 35/35 | 35/35 | 35/35 |
| Minibatch | 35/35 | 35/35 | 35/35 |
| Forward only | 34/35 | 35/35 | 35/35 |

Each policy changes exactly one selected endpoint: forward-only
`associative-divisor-n2-l1-w0`. Its raw and primary validation accuracy rises
from 2508/4656 (53.8660%) to 4656/4656 (100%). Ordinary validation CE falls
from 11.072164958079812 to 0.005310253135586406. The original evaluator's
observed training loss remains exactly 9.626592856193383e-9, with 100% observed
accuracy. There are no selected primary regressions or reported metric
regressions. The complete earlier metrics and every selected reference remain
in the original summary.

The validation set has no exact observed input. Its 2508 supported calls use
the candidate; 2148 unsupported calls use its frozen predecessor. All 4656
training calls remain on supported, exactly observed inputs. This measured
result supports the fixed routing rule on this dataset. It does not prove
that all supported unseen aliases are correct. All masks retain the original
prediction-call order and are independently reconstructed by replay.

All 210 policy/task/mode import outcomes remain explicit: 176 unchanged,
32 wrapped, and two refused. Of the 32 wrappers, two are evaluated and 30
refuse because an original metric input exceeds their common prediction
domain. The two import refusals are forward-only `gpt-probe-branch-amplitudes`,
one in each policy, because the exact predecessor checkpoint is missing or
unsupported. These refusals retain the previously selected models and their
original results. The 35/35 result applies to the complete retained portfolio;
it is not a claim that every proposed wrapper was evaluated successfully.

The complete summary preserves all 35 conventional transformer endpoints and
336 depth outcomes. The conventional baseline passes 33/35; its failures are
`arithmetic-sum-pow4` and `basis-recall-easy`. Historical DiffGPT's recorded
35/35 on each of seeds 0, 1 and 2 remains a separate reference. This
comparison uses the original seed1 tasks, including every short/long split.
It adds no discovery work or gradient updates. Full and minibatch histories
remain their original saved histories; forward-only adds no training. All
inherited training, reconstruction and metric costs remain charged.

Preparation takes 1:16:28, all case executions total 520.87 seconds, all
replays 525.81 seconds, and full summary reconstruction 1:16:33. Preparation
and summary recursively verify the complete parent comparisons and staged
models. Peak RSS is 5,120,164 KiB, about 4.883 GiB, under the 64 GiB process
address limit. The runner retains a 16 GiB disk reserve. These verification
times do not establish a training-speed or matched-compute advantage.

Frozen producer: `3852e0b0bf83afc9ec3ec1b4b83bc6eefa964abd`.
The source fingerprint is
`71c6b081a43c630ca44cb8b1239dca2df0bc1d482b0ab21fb71f0152196f44ec`.
The executable hash is
`808f75ce5881e115ebe2e7e18d48d0bd178d815fdfa7ac3493024a20a9c354b1`;
the inventory hash is
`936b557b798d07e6396af25640160a6e00425f823163c4a1c85f22ec106f3530`;
the complete summary hash is
`7f658904aee34f8b94c5ba53ed3dbca04a41d82addd257e5aa7b8fe2fd3b49db`.
The 428-file manifest `.kanban/artifacts/267/completed.sha256` has hash
`2b4f2d113bb110bf5a7478b809abeb6365c877ed98f3cc1a37f2a85760247848`.

Generated evidence remains ignored on the storage disk under
`transformer-runs/convex_stress_fallback_compare/original-v1`. Frozen source
and executable evidence is under `freezes/fallback-compare-v2`; the bound
original request and executable are under `freezes/fallback-compare-original-v1`.
Use the parent README's native commands to reproduce the complete comparison.
For independent review, rerun every original case through `replay-case`, then
`summarize` the original directory into a separate file and compare it exactly:

```sh
while IFS= read -r task; do
  prlimit --as=68719476736:68719476736 -- FROZEN_EXECUTABLE \
    replay-case ORIGINAL_ROOT "$task" > "FRESH_REVIEW/$task.replay.json"
  cmp "ORIGINAL_ROOT/$task.replay.json" "FRESH_REVIEW/$task.replay.json"
done < <(jq -r '.tasks[]' ORIGINAL_ROOT/inventory.json)
prlimit --as=68719476736:68719476736 -- FROZEN_EXECUTABLE \
  summarize ORIGINAL_ROOT > FRESH_REVIEW/summary.json
cmp ORIGINAL_ROOT/summary.json FRESH_REVIEW/summary.json
```

Review also reruns native core/adapter checks, the complete Lean build, axiom
and generated-index checks, and all frozen source/parent/output hashes at the
submitted findings commit. No generated report or checkpoint is committed.
