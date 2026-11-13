# Original causal-pair comparison

Does importing the preserved causal-pair stage improve the complete reviewed
quotient comparison? The frozen task245 run imports producer
`b516af90296f7f26aaeb8d6af2d9ad32e0e12c2b` without retraining or new discovery.
All 35 original cases and their first independent actual-state replays finish
successfully. Fresh review at the findings commit is a separate step.

Both initializer policies retain all 105 task/mode choices. Selection still
requires an ordinary raw-training-CE decrease greater than `1e-12`, with the
parent first. Each task's declared primary 99% criterion determines success;
auxiliary split failures do not revoke it. The old `full35_pass` field in this
frozen executable uses the all-split diagnostic. `report.jq` exposes both
interpretations explicitly without changing any stored flag or selection.

| Mode | Parent primary successes | New primary successes | New all-split successes |
| --- | ---: | ---: | ---: |
| Full-batch | 34/35 | 34/35 | 34/35 |
| Minibatch | 34/35 | 34/35 | 34/35 |
| Forward-only | 33/35 | 33/35 | 33/35 |

The unchanged totals conceal a gain and a regression. Each policy selects 17
new endpoints. Forward-only branch amplitudes gains its previously missing
model and reaches 100% primary sequence accuracy. Forward-only associative
divisor drops from 100% to `53.8659793814433%`, despite training CE decreasing
from approximately `0.01151048` to `9.62659e-9`. Its validation loss increases
from `0.01151048159022136` to `11.072164958079812`; 2,148 validation prediction
calls have an unknown prefix. This is the sole selected primary regression,
accuracy decrease and loss increase. The exact call masks remain in the saved
report; an unknown-prefix count is not by itself an error attribution.

The shifted Boolean AND replacements retain primary success and also pass
the previously failing auxiliary split. Fibonacci remains unsuccessful in
all modes; this causal stage excludes its two-token input shape. The current
forward-only failures are Fibonacci and associative divisor.

There are 19 evaluated standalone endpoints and eight iterative work-cap
refusals on nine eligible tasks. The two key-value-binding candidates reach
`86.1328125%` on their primary novel-input split, but neither is selected:
their training loss does not beat the corresponding parent. They are retained
candidate failures, not selected regressions. The remaining 26 task records
retain 23 two-token exclusions, two depth input-storage refusals and one recall
feature-cache refusal. No task or available mode is omitted.

All 35 conventional-transformer endpoints and 336 depth outcomes remain in the
complete parent chain. The conventional baseline still has 33/35 primary
successes. DiffGPT's recorded phase-score-lerope result of 35/35 for each of
seeds 0, 1 and 2 remains a separate historical reference; this comparison is
the unchanged seed1 cohort, not a three-seed replication.

Staged construction is charged once across both policies: `5.358890285`
seconds and 1,840 gradient batches, with zero new discovery or training in
this import. Import, inference, repeated parent validation, hashing and replay
have their own resource records. The largest measured process uses 1,756,124
KiB RSS, under the 64 GiB address-space cap. This is not matched-compute
evidence against the transformer.

The result supports a specific limitation of training-CE selection: fitting
all observed prefixes does not make a default on an unknown prefix correct.
It does not prove the CM35 model succeeds on Fibonacci or that a coverage-based
fallback will repair associative divisor. Those require separately declared
models and complete comparisons. The frozen selector and every failed outcome
are preserved.

Generated evidence resides on the storage disk at
`/run/media/misha/storage/transformer-runs/convex_stress_causal_pairs/original-v1`.
The source fingerprint is
`f3a1ca48ff6e8a0e91db4cf7443187886381e74a74e72c08005b331d6c4d8b54`;
the executable SHA256 is
`fbb6ee1566a7191ff293ae2f8c9e052a1f8e1ec434c438a55338ac42e435abc5`.
The inventory SHA256 is
`3fa7e52e233a4c60d1ef08e6411a7eb2154aaf3929c91de4544114dd32ef3e23`;
the complete summary SHA256 is
`fa07679de40f3662f2ae6177a40f0f8fa67b8504dfa9a83b74a8b8b4884c2c14`.

From the repository root, regenerate the compact interpretation and replay
the saved complete summary with the same frozen executable:

```sh
run_root=/run/media/misha/storage/transformer-runs/convex_stress_causal_pairs/original-v1
freeze="$PWD/.kanban/artifacts/245/freeze-v1"
evidence_root=/run/media/misha/storage/transformer-runs/convex_stress_causal_pairs/recheck
mkdir -p "$evidence_root"
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_causal_pairs/cohort/original/report.jq \
  "$run_root/summary.json" > "$evidence_root/analysis.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r task; do
  prlimit --as=68719476736:68719476736 -- "$freeze/causal-original-v1" \
    replay-original-case "$run_root" "$task" > "$evidence_root/$task.replay.json"
  cmp "$evidence_root/$task.replay.json" "$run_root/$task.replay.json"
done < <(jq -r '.tasks[]' "$run_root/inventory.json")
prlimit --as=68719476736:68719476736 -- "$freeze/causal-original-v1" \
  summarize-original "$run_root" > "$evidence_root/summary.json"
cmp "$evidence_root/summary.json" "$run_root/summary.json"
```

For a fresh import, set `CAUSAL_PAIRS_BINARY` to that frozen executable and
invoke its preserved `run_original.sh REQUEST FRESH_ROOT` using the frozen
request and a new directory on the storage disk. The reviewed parent/stage
artifacts and board gates are required. New measured timings and file paths
produce a new summary digest; exact-byte replay applies to each saved run.
The [adapter README](../README.md) retains the original protocol and native
test commands. This authored report supplements that frozen document.
