# Original refined complex comparison

Does the reviewed CM31/CM32 stage improve the complete causal comparison?
Task246's frozen import completed all 35 original cases and all 35 independent
actual-state replays without process failures. Fresh review at the findings
commit remains a separate step.

Both initializer policies retain their exact 105 parent choices. None of the
18 refined endpoints lowers the selected model's observed raw CE by more than
the declared `1e-12` threshold, so no replacement occurs. Primary accuracy of
at least 99% remains the success criterion; auxiliary splits are diagnostics.

| Training mode | Selected primary success, each policy | All-split diagnostic success |
| --- | ---: | ---: |
| Full batch | 34/35 | 34/35 |
| Minibatch | 34/35 | 34/35 |
| Forward only | 33/35 | 33/35 |

There is no selected gain or regression. Fibonacci remains unsuccessful in all
three modes; associative divisor remains the second forward-only failure.
The six available refined models, each at prime97, all pass their original
primary and auxiliary splits in all modes: cube-plus, subtraction, addition,
square-sum, half-sum and permuted-addition. Their raw training CE is
`3.6241303291717486e-9`. These 18 standalone successes do not add a selected
benchmark pass. All unavailable staged outcomes and their costs remain in the
complete 23-case, 1,265-stage construction evidence.

The run imports the exact reviewed stage without new training or discovery.
Its retained construction cost is `185.750755602` seconds and 120 gradient
batches, charged once across both initializer policies. Original metric
execution, hashing, foreign-summary verification and replay have separate
resource records. The largest measured process used 2,361,828 KiB RSS under
the 64 GiB address-space cap. The full 35 conventional transformer endpoints
and 336 depth outcomes remain unchanged. The conventional transformer retains
33/35 primary successes; DiffGPT's recorded 35/35 for each seed0/1/2 remains a
historical reference rather than a fresh three-seed run here.

The result confirms that these particular refined endpoints do not change the
current parent-first selector. It does not establish the future complete-kernel,
grouped-kernel or support-fallback results. Those models and comparison protocols
remain separate, with their original refusals, selection rules and review gates.

Generated evidence is preserved at
`/run/media/misha/storage/transformer-runs/convex_stress_refined_compare/original-v1`.
The producer is `a00467028b30dec33e570e6659b6e61c2dd3b4dd`;
source fingerprint
`8c5407c70cf6047f4504198a702e59c83c4e4fca8f059b53f2db05a8a21ef8c9`;
executable SHA256
`303e590e65235b345867a575b8052422dd97cfd11f15ad612f2cf7f31fa1145b`;
inventory SHA256
`fb3371f91ad32361a7b0df849897b743ce81b756ebff715ec41ff2e38283d576`;
complete summary SHA256
`dce5ed27d3f9f2738a14a798077c53db2bb0fb04a0e116b50e3e47df365b63b3`.

Regenerate the interpretation and independently replay every case and the
complete summary with the frozen executable:

```sh
run_root=/run/media/misha/storage/transformer-runs/convex_stress_refined_compare/original-v1
freeze="$PWD/.kanban/artifacts/246/freeze-v1"
evidence_root=/run/media/misha/storage/transformer-runs/convex_stress_refined_compare/recheck
mkdir -p "$evidence_root"
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_refined_compare/original/report.jq \
  "$run_root/summary.json" > "$evidence_root/analysis.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r task; do
  prlimit --as=68719476736:68719476736 -- "$freeze/refined-compare-v1" \
    replay-case "$run_root" "$task" > "$evidence_root/$task.replay.json"
  cmp "$evidence_root/$task.replay.json" "$run_root/$task.replay.json"
done < <(jq -r '.tasks[]' "$run_root/inventory.json")
prlimit --as=68719476736:68719476736 -- "$freeze/refined-compare-v1" \
  summarize "$run_root" > "$evidence_root/summary.json"
cmp "$evidence_root/summary.json" "$run_root/summary.json"
```

For a fresh import, use the frozen request, protocol and shell scripts with
`REFINED_COMPARE_BINARY` pointing to the frozen executable and a new destination
on the storage disk. Keep all logs, summaries and resource evidence local and
outside temporary worktrees. The existing adapter README and source are frozen;
this authored findings directory does not alter their fingerprint.
