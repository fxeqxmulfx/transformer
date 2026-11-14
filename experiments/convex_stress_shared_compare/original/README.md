# Original shared-role and predecessor comparison

Task 277 attempted all 35 original tasks with the frozen task 276 adapter.
Twenty-eight cases completed and reproduced their numerical replay. Seven
cases failed before metrics because some original long inputs exceeded a
saved count predecessor's domain. Those failures remain in the denominator;
their earlier successful parent scores are not substituted.

| Training mode | Primary passes, each initializer policy | All-split diagnostic passes | Process failures |
| --- | ---: | ---: | ---: |
| Full batch | 28/35 | 28/35 | 7 |
| Minibatch | 28/35 | 28/35 | 7 |
| Forward only | 28/35 | 28/35 | 7 |

The two policies retain 105 choices each. Primary sequence accuracy of at
least 99% is the success rule. Every completed case passes it in every mode.
The complete comparison is nevertheless unsuccessful because seven cases
have no scores. Fresh review at the findings commit is a separate step.

The failed cases are `suite-boolean-and-shift`, `suite-boolean-and-random`,
`gpt-probe-branch-amplitudes`, `gpt-probe-linear-transport`, `basis-parity16`,
`basis-parity20` and `basis-parity24`. Each retained process exits 1 with
`original metric input exceeds common prediction domain`. The old wrapper
restricts the whole model to its saved predecessor's prediction limit.
CM41 and tasks 279/280 correct this prospectively by retaining the candidate
outside that predecessor's domain. Their new comparison is separate from
these immutable results.

Fibonacci's selected prime-97 shared-role extension reaches
`0.9993622448979592` primary accuracy (99.936%) in all three modes and both
policies. Its observed raw CE is `9.626594307035853e-9`. Among 9,408 validation
prediction calls, prior support covers 9,396 and shared support covers 9,402;
six calls use the supplement. Six calls still lack shared support. These are
call counts, not distinct sequences or an unseen-answer correctness proof.

Forward-only associative divisor reaches 100% in both policies, up from the
reviewed parent's 53.866%. Its candidate supports 2,508 of 4,656 validation
calls; the actual saved predecessor supplies the other 2,148. Full-batch and
minibatch associative-divisor outcomes remain inherited reviewed successes.

Per policy, full batch and minibatch each contain one fresh changed-model
metric and 27 inherited reviewed metrics. Forward only contains two fresh
metrics and 26 inherited ones. The three Fibonacci states are shared across
the policies; associative divisor has two separately bound wrappers. The
report retains five distinct fresh endpoint records. Its change table records
four newly successful task/mode choices and 21 process-failure regressions per
policy; these regressions have no measured accuracy. Selection uses observed
raw CE, fixed prime order and structural support, never held-out scores.

There is no new optimization or feature discovery during this comparison.
Raw-kernel construction retains 792.72834577 seconds of discovery,
1,992.023800371 seconds of head construction and 599 gradient batches.
Shared-role construction retains 28.351324534 seconds of discovery,
81.512768323 seconds of head construction and 91 gradient batches. Both are
charged once across the policies, with all earlier costs retained. Completed
cases record 1,113.104896363 seconds of verification and 3.895533768 seconds of
metrics. Failed-process costs remain in their resource records. The largest
original process used 3,723,900 KiB RSS under the 64 GiB address-space cap.

All 35 conventional transformer endpoints and 336 depth outcomes remain.
The conventional transformer retains 33/35 primary successes. DiffGPT's
recorded 35/35 on each seed 0, 1 and 2 is a separate historical reference;
this experiment uses the original seed 1 only.

Generated evidence remains on the storage disk at
`/run/media/misha/storage/transformer-runs/convex_stress_shared_compare/original-v1`.
The producer is `15761f51e6c2f02b2fccbd6a2bf2bb809f645bed`;
source fingerprint
`78fb5d50d7314aecf5404c0791b37d2a6e399cddc1b803e82dc04b0f0fb66bc5`;
executable SHA256
`7f5a47bf45e89573a034e9351aa5d2642ce314d00f9cd3678e4329290c661025`;
inventory SHA256
`8e1d94880ee04234c2de37e9b5e2df2f922883ecf865affcbd337c7432633ce1`;
complete summary SHA256
`ebd1e6a4092986ab74edd4f97f138945691b51d314458eebde277be5f954a086`.

Regenerate the interpretation and replay all completed cases with the frozen
binary. Failed cases retain their original exits, logs and partial artifacts:

```sh
run_root=/run/media/misha/storage/transformer-runs/convex_stress_shared_compare/original-v1
freeze="$PWD/.kanban/artifacts/277/freeze-v1"
evidence_root=/run/media/misha/storage/transformer-runs/convex_stress_shared_compare/recheck
mkdir -p "$evidence_root"
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_shared_compare/original/report.jq \
  "$run_root/summary.json" > "$evidence_root/report.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r task; do
  if test "$(cat "$run_root/$task.run.exit")" = 0; then
    prlimit --as=68719476736:68719476736 -- "$freeze/shared-compare-v1" \
      replay-case "$run_root" "$task" > "$evidence_root/$task.replay.json"
    cmp "$evidence_root/$task.replay.json" "$run_root/$task.replay.json"
  fi
done < <(jq -r '.tasks[]' "$run_root/inventory.json")
prlimit --as=68719476736:68719476736 -- "$freeze/shared-compare-v1" \
  summarize "$run_root" > "$evidence_root/summary.json"
cmp "$evidence_root/summary.json" "$run_root/summary.json"
```

For a new import, use the frozen request and runner with
`SHARED_COMPARE_BINARY` pointing to the frozen executable and a fresh storage
destination. Keep generated summaries, tensors, archives and logs local.
