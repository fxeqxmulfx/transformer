# Original CM44 decoder confidence results

The frozen producer `d5ec25b695f7f34c9a88dfcdabb08c5e61808cc9` completed all
35 original seed-1 cases and all 35 fresh model/metric replays. All 35 primary
and auxiliary task successes remain in full-batch, minibatch and forward-only
modes, under both initializer policies. No process failed and no measured
raw accuracy or CE regressed.

The comparison changes 35 saved heads across 13 tasks. These are the exact
selected identified, consequence and fiber phase models, not newly discovered
answers. All original source metrics reproduced their reviewed values before
the new heads were scored. Every transformed head selected margin 24 using
original training CE, with zero training errors and zero new optimizer updates.

| Original vocabulary | Changed heads | Previous validation CE | New validation CE | Raw accuracy |
| --- | ---: | ---: | ---: | ---: |
| 97 | 25 | 3.5922701555 | 0.00000000362413 | 100%, unchanged |
| 193 | 10 | 4.2715538054 | 0.00000000724826 | 100%, unchanged |

There are 13 fresh full-batch endpoints, nine fresh minibatch endpoints and
13 fresh forward-only endpoints per policy. The other 70 of each policy's
105 choices remain explicitly inherited. Four minibatch choices use other
saved representations and are retained. The complete 35 conventional
transformer endpoints, 336 depth outcomes and previous construction costs
remain present. These baselines were retained, not retrained in this run.

The old categorical predictor used logit 1 for its already correct raw label.
CM44 realizes a larger margin with the actual complete conjugate complex head
and charges every real and imaginary coefficient to its explicit budget. The
CE improvement is a confidence correction; it does not identify additional
unseen raw labels. The old latent-moment constraint and the new complete-head
budget remain distinct in the accounting. All logits and CE were recomputed
from actual complex products, with ordinary full-vocabulary denominators.

The inherited Fibonacci endpoint remains at 4574/4704 raw answers correct
(97.2364%), 4701/4704 field answers correct (99.9362%), and CE approximately
0.16604154. Its remaining 130 raw errors are untouched by this change. Earlier
non-phase representations with higher CE also remain explicit; the result is
not a claim of near-zero CE or perfect raw labels across all 35 tasks.

Preparation took 133.80 seconds and peaked at 8,493,192 KiB (8.10 GiB) while
importing large historical source containers. The 35 case runs totaled 58.29
seconds; their 35 replays totaled 63.39 seconds. A case run peaked at 624,088
KiB (0.60 GiB), and summary generation took 2.15 seconds. Every process had
a 64 GiB address-space limit. The aggregate per-head buffer forecast is
22,942,683,328 bytes and aggregate construction work is 39,278,779,328 scalar
operations; these sums span separate sequential cases, not a single live
allocation or one process's work limit. No matched-compute speed claim is made.

Eight adapter tests, six reviewed component tests, formatting, Clippy with
warnings denied, the full Lean build, axiom audit and generated INDEX check
pass. There remain 36 pre-existing unfinished statements, zero declarations
resting on them, zero extra axioms and no new unfinished proof. The independent
exact-commit review reruns these checks and a separate complete preparation,
35-case evaluation and replay, comparing every JSON artifact and compact report.

Reproducibility inputs and generated evidence remain local and ignored:

- Frozen executable, request, source inventory and hashes:
  `.kanban/artifacts/290/freeze-v1`, linked to storage `freezes/decoder-compare-v1`.
- Original outputs: storage `decoder-compare/original-v1`.
- Compact generated report: storage `decoder-compare/report-v1.json`.
- Complete evidence manifest: `.kanban/artifacts/290/completed.sha256`, 373 files.
- Source fingerprint: `df7902344a99a7d407e18ba334f35e5821168660f4788cd69a114f2f85104bb9`.
- Executable SHA-256: `b6cc48aa6dc5867cd5a617a1cc73697e6665b9c4f48f29c915f9621dae26820b`.
- Inventory SHA-256: `45225555f92551941b3ae329d55c085b31dc6aa21462346476230b6f25cbffa5`.
- Summary SHA-256: `6e998d454cb5d6c51e780c1fab60fb0dd1757e62c02b4d2a65c7ddca24160d34`.
- Report SHA-256: `96ffa6cd95ac915bf6563cab0e3e00313a24b67a8912bc3b5ae74285397508fa`.

The parent is reviewed task 287 at `c5ba15c6`, the native component is reviewed
task 289 at `9ddda0f8`, and CM44 is proved at `a8f0020a`. The prospective
protocol and commands are in the experiment's parent README. All failures,
unchanged outcomes and original success definitions remain in scope.
