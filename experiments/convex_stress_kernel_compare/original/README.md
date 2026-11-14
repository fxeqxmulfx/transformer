# Original complete-kernel comparison

Task 255 evaluates the reviewed task 252 raw-kernel heads against the complete
task 246 parent. All 35 original seed-1 cases and their independent numerical
replays completed with zero process failures. Primary sequence accuracy of at
least 99% remains the success criterion; auxiliary splits remain diagnostics.
Fresh review at the findings commit is a separate step.

| Training mode | Primary passes, each initializer policy | All-split diagnostic passes |
| --- | ---: | ---: |
| Full batch | 35/35 | 35/35 |
| Minibatch | 35/35 | 35/35 |
| Forward only | 34/35 | 34/35 |

Both policies retain their exact 105 choices. The parent-first raw-CE selector
changes three choices per policy: Fibonacci in each mode selects the actual
prime-97 complete-kernel head. These are three newly successful choices, with
no primary or sequence-accuracy regression. The other 102 choices per policy retain
their reviewed parent metrics. Forward-only associative divisor remains at
53.8659793814433% and is the sole failed task/mode choice per policy. This raw
comparison has no predecessor fallback or shared-role extension; their later
results remain separate experiments.

Fibonacci reaches `0.9987244897959183` primary field accuracy in each mode
(4,698/4,704). Observed raw CE is `9.626594307035853e-9`. Its 9,408 validation
prediction calls include 9,396 supported calls, six unknown-axis calls and
six unmatched active calls. These are call counts, not unique examples or a
proof of arbitrary unseen-label identification.

There are three validation-loss increases per policy, all on Fibonacci:
5.5518670574 (full/forward) or 5.5506739925 (minibatch) rises to
8.0918367443. Native raw-token accuracy is 66.2840%, alongside the 99.8724%
primary field accuracy. The unchanged benchmark pools logits into field
classes, while retaining raw-token CE and accuracy as diagnostics. The primary
success does not establish improved held-out raw-token CE.

The run evaluates 159 actual endpoints; 111 satisfy the declared offer gate.
Only the three Fibonacci endpoints change selected heads. The complete
1,265-stage, three-mode ledger also retains 3,636 refused mode outcomes:
2,964 feature/state-cap refusals, 660 unavailable-kernel outcomes, six
memory/audit-work refusals and six minibatch scalar-work refusals. These are
retained construction options, not failed original-case processes. No refused
model becomes an evaluated or selected head.

There is no new discovery or optimization in this comparison. Its reviewed
construction retains 792.72834577 seconds of discovery, 1,992.023800371 seconds
of head construction and 599 gradient batches, charged once across policies.
All earlier costs remain in the complete parent. Original import, metric,
replay, hashing and recursive-summary costs have separate resource records;
the final summary alone took 1:12:57. Peak original process RSS was
3,937,836 KiB under the 64 GiB address-space cap.

All 35 conventional transformer endpoints and 336 depth outcomes remain
unchanged. The conventional transformer retains 33/35 primary successes.
DiffGPT's recorded 35/35 on each of seeds 0, 1 and 2 remains a separate
historical reference, while this experiment uses original seed 1.

Generated evidence is preserved at
`/run/media/misha/storage/transformer-runs/convex_stress_kernel_compare/original-v1`.
Producer: `d51aa4229b6178210aff831ebcffa2361164a950`;
source fingerprint:
`06df40a70bc3c52b2ca4abacba3ad3100a4003688f525b2164f7b136714c7d8d`;
executable SHA256:
`7476d0ee117484de115d3b8dddac515fc32c316a85ef34ea09312e5452506c26`;
inventory SHA256:
`56764686b3fb1c4be31f4414a9c837ea14c88b1f1414dc38d6bf6d9da4adfce8`;
complete summary SHA256:
`98e9a77c89c1dcd71b4eb9a1d0d1af8feccc98c36c55011fe4ecedba22a9e0dd`.

Regenerate the interpretation and independently replay every original case:

```sh
run_root=/run/media/misha/storage/transformer-runs/convex_stress_kernel_compare/original-v1
freeze="$PWD/.kanban/artifacts/255/freeze-v1"
evidence_root=/run/media/misha/storage/transformer-runs/convex_stress_kernel_compare/recheck
mkdir -p "$evidence_root"
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_kernel_compare/original/report.jq \
  "$run_root/summary.json" > "$evidence_root/report.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r task; do
  prlimit --as=68719476736:68719476736 -- "$freeze/kernel-compare-v1" \
    replay-case "$run_root" "$task" > "$evidence_root/$task.replay.json"
  cmp "$evidence_root/$task.replay.json" "$run_root/$task.replay.json"
done < <(jq -r '.tasks[]' "$run_root/inventory.json")
prlimit --as=68719476736:68719476736 -- "$freeze/kernel-compare-v1" \
  summarize "$run_root" > "$evidence_root/summary.json"
cmp "$evidence_root/summary.json" "$run_root/summary.json"
```

For a new import, use the frozen request and runner with
`KERNEL_COMPARE_BINARY` pointing to the frozen executable and a fresh storage
destination. Keep generated JSON, tensors, logs and archives local. The frozen
adapter sources and original metrics are unchanged by this findings directory.
