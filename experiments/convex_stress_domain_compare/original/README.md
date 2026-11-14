# Original comparison with partial-parent routing

This experiment evaluates the reviewed task 280 construction on the original
35-task, seed-1 stress suite. The full-batch, minibatch and forward-only modes
retain both independent 105-choice initializer policies. Primary sequence
accuracy of at least 99% is the success rule; auxiliary splits are diagnostics.
All 35 cases and their original independent numerical replays completed with
zero process failures. Both policies pass every primary and auxiliary check.
Fresh review at the findings commit is a separate step.

| Training mode | Primary passes, each policy | All-split diagnostic passes | Fresh / inherited selected metrics, each policy |
| --- | ---: | ---: | ---: |
| Full batch | 35/35 | 35/35 | 3 / 32 |
| Minibatch | 35/35 | 35/35 | 8 / 27 |
| Forward only | 35/35 | 35/35 | 8 / 27 |

The reviewed 246 parent had 34/35 full-batch, 34/35 minibatch and 33/35
forward-only successes. Each policy now has 19 changed choices, four newly
successful task/mode choices, and no primary or accuracy regression. The
three Fibonacci states are shared across policies, so the run retains 35
distinct fresh endpoint records for 38 changed policy choices. All other
selected metrics retain their explicit reviewed-parent provenance.

Fibonacci's selected prime-97 shared-role extension reaches
`0.9993622448979592` primary accuracy in every mode. Its observed raw CE is
`9.626594307035853e-9`. Forward-only associative divisor reaches 100%, up from
53.866%; its other modes retain their reviewed successes. The seven task 277
preflight failures now have passing original outcomes, with no failed case
silently replaced by an ancestor score.

There are three validation-loss regressions per policy, all on Fibonacci:
raw validation CE rises from 5.5518670574 (full/forward) or 5.5506739925
(minibatch) to 8.0816326627. The original Fibonacci primary metric pools
predictions into field classes; raw token CE is a separate retained diagnostic.
Fibonacci's native raw-token accuracy is 66.3265% (3,120/4,704), alongside
99.9362% primary field accuracy (4,701/4,704). The primary successes therefore
do not establish improved held-out raw CE or
unconditional identification of the original token labels.

The fixed construction retains the reviewed 246 parent, raw-kernel training
from 252, shared-role training from 274 and causal training from 237. The
original parent-first raw-CE selector and fixed prime ordering remain. A
selected raw head may gain the same-field shared-role extension; a selected
causal head uses the checked saved predecessor on unsupported inputs inside
that predecessor's domain. Elsewhere the candidate remains available through
length 4096. No held-out label or score chooses the model or route, and this
comparison performs no new optimization or feature discovery.

CM41 at `baf7c2e4` proves the actual partial-parent feature construction,
component bounds, convex CE on the affine domain freezing parent coefficients,
and equality of supported normalized batch objectives and derivatives. The
native wrapper at `4f6ebe46` retains the actual complex tensors and both head
budgets. Fixed features and fixed routing are essential to that convexity
statement. Discrete feature discovery is outside this convex optimization
claim, and the theorem does not guarantee an arbitrary unseen answer.

The original task 277 failure evidence remains separate. Its preflight bound
restricted a whole wrapper to the saved predecessor's domain, including the
`prompt length + generation limit` forecast. Actual prediction-call masks are
the evidence for whether a parent is unavailable during scoring. The report
records support, parent availability, actual parent use and unsupported calls
routed to the candidate without a parent. These counts must not be confused
with correctness or with unique examples across overlapping splits.

Across the complete original run, the actual parent-unavailable count is zero,
and no unsupported call uses the candidate because its parent is unavailable.
Thus recovery of the seven task 277 failures is a preflight/domain-validation
fix, not evidence of extra parent-unavailable coverage on these examples. The
original rollout queries before appending its next token; at a full generation
horizon the last queried prefix is one shorter than the old final-length
forecast. The native CM41 fixtures separately exercise genuinely unavailable
parents through length 4096. Actual available-parent fallback is used on
2,148 associative-divisor validation calls per policy, 4,296 calls in total.

Unchanged selected metrics retain explicit reviewed-parent provenance. Changed
actual models are evaluated afresh after their selection and saved tensors are
sealed, then reconstructed for independent numerical replay. All 35 conventional
transformer endpoints and 336 depth outcomes remain. The conventional transformer
has 33/35 primary successes; DiffGPT's recorded 35/35 on each of seeds 0, 1 and 2
remains a separate historical reference.

Construction costs remain charged once across policies: raw-kernel discovery
792.72834577 seconds, head construction 1,992.023800371 seconds and 599 gradient
batches; shared-role discovery 28.351324534 seconds, head construction
81.512768323 seconds and 91 gradient batches. All earlier costs remain in the
complete parent. This comparison records 1,151.7906422 seconds of case
verification and 12.8411804 seconds of changed-model metrics, with zero new
optimizer updates. Hashing, summary generation and independent replay have
separate resource records. Peak original process RSS was 4,941,812 KiB, below
the 64 GiB address-space cap; buffer, metric-work and disk caps remain fixed.

Generated evidence stays at
`/run/media/misha/storage/transformer-runs/convex_stress_domain_compare/original-v1`.
The producer is `3a4434fa67d2d24d8d6d1b88abfffdd024c9a051`;
source fingerprint
`63cbf0c5c3ae9ea63b58707d7f073e04e926df521631f9d2388af16981dab6d0`;
executable SHA256
`0fdea4648bc59b923f85ab93e1aaf437dbb1e0cc3a23812704dbb8ca1b56c4ef`;
inventory SHA256
`7a540150674e7ddf9f21e0e1b1c327f90b6a6c6b1fad967dd9ef6db341b7c633`;
complete summary SHA256
`10565c46d2ce592c7c7eda3cea1b683018115440e9602c0947ee87f37b2fbb1b`.
The frozen model, protocol and all source parents are unchanged by this
authored findings directory.

Regenerate the interpretation and replay the completed original cases:

```sh
run_root=/run/media/misha/storage/transformer-runs/convex_stress_domain_compare/original-v1
freeze="$PWD/.kanban/artifacts/281/freeze-v1"
evidence_root=/run/media/misha/storage/transformer-runs/convex_stress_domain_compare/recheck
mkdir -p "$evidence_root"
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_domain_compare/original/report.jq \
  "$run_root/summary.json" > "$evidence_root/report.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while IFS= read -r task; do
  if test "$(cat "$run_root/$task.run.exit")" = 0; then
    prlimit --as=68719476736:68719476736 -- "$freeze/domain-compare-v1" \
      replay-case "$run_root" "$task" > "$evidence_root/$task.replay.json"
    cmp "$evidence_root/$task.replay.json" "$run_root/$task.replay.json"
  fi
done < <(jq -r '.tasks[]' "$run_root/inventory.json")
prlimit --as=68719476736:68719476736 -- "$freeze/domain-compare-v1" \
  summarize "$run_root" > "$evidence_root/summary.json"
cmp "$evidence_root/summary.json" "$run_root/summary.json"
```

For a new import, use the frozen request and runner with
`DOMAIN_COMPARE_BINARY` pointing to the frozen executable and a new storage
destination. Keep generated summaries, tensors, logs and archives local.
