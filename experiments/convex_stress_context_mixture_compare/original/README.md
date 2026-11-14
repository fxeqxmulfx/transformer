# Original CM45 context-mixture results

The frozen producer `714d6d6998ae87e44c21808b9f056922c85f5a49` completed all
35 original seed-1 cases and all 35 model/metric replays with zero process
failures. Both initializer policies retain 35/35 primary and auxiliary
successes in full-batch, minibatch and forward-only modes. Raw-label accuracy
and ordinary CE improve on every changed endpoint. No endpoint regresses on
primary success, aggregate raw accuracy or CE relative to task290.

Eligibility selected the three inherited CM43 Fibonacci context heads by
their current representation, reference and actual model hash. The other
102 choices per policy remain explicitly inherited. All 35 conventional
transformer endpoints, 336 depth outcomes, historical DiffGPT results and
prior constructor costs remain in the comparison. These baseline models were
not retrained. Every changed source reproduced its old metrics exactly before
the new model was evaluated.

| Fibonacci endpoint | Correct raw labels / 4704 | Raw accuracy | Raw errors | Validation CE |
| --- | ---: | ---: | ---: | ---: |
| Previous CM43/44 parent | 4574 | 97.2364% | 130 | 0.16604154 |
| CM45 full-batch | 4595 | 97.6828% | 109 | 0.14010691 |
| CM45 minibatch | 4601 | 97.8104% | 103 | 0.14195986 |
| CM45 forward-only | 4604 | 97.8741% | 100 | 0.14904384 |

All three preserve 4701/4704 field answers correct (99.9362%) and the original
observed CE of approximately 9.62659e-9. Forward-only removes 30 of the
previous 130 raw errors without derivatives or optimizer updates. Full-batch
achieves the lowest CE, while forward-only has the highest raw accuracy. The
same CE/argmax tradeoff exposed by the synthetic fixture remains visible;
lower CE is not treated as proof of a better raw-label count. At least 100 raw
errors remain, and the three wrong field answers are unchanged.

The fixed bank has 91 maps. Both prior and supplement initially select the
17-neighbor map with axis weights (4,1), inverse squared-distance power one
and confidence scale 0.5. Forward retains this vertex. Full-batch retains a
five-map mixture after its sixth step; its training-only diagnostic CE falls
from 0.08339867 to 0.06427663 on the prior. Minibatch retains a four-map
mixture after the first epoch, with prior diagnostic CE 0.06984945. The source
coefficients and observed rows remain unchanged; inference multiplies actual
mixed complex features by the complete original tensor.

Full-batch executes 64 gradient batches on each of the prior and supplement,
with six nonzero updates each. Minibatch executes eight epochs at the original
batch size 512: ten batches per epoch, including the 96-record tail. It performs
80 updates per component. All attempted work remains charged even when later
epochs are discarded. In total there are 288 new gradient batches, 172 nonzero
optimizer updates, 677376 gradient-record visits and 7310016 forward-score
record visits. These totals count each model once, separately from reuse under
both initializer policies. Forward-only contributes zero gradients and updates.

The diagnostic uses original raw training records and excludes the queried
observed source state, with the ordinary full-vocabulary CE denominator. It
does not refit latent discovery. Its fixed canonical head selects a mixture
that is transferred to the unchanged trained head. The ordinal locality bias
is explicit; there is no task-specific answer formula. CM45 proves convexity
in the mixture with bank/head fixed and retains the spectral feature and
complete coefficient bounds. It does not certify unseen labels or convexity
when both factors change together.

Preparation took 15.43 seconds and peaked at 725248 KiB. The 35 runs totaled
66.50 seconds, peaking at 1196484 KiB (1.14 GiB); the 35 replays totaled
93.45 seconds, peaking at 2090912 KiB (1.99 GiB). Summary generation took
2.10 seconds. Every process had a 64 GiB address-space limit. The changed
case forecast is 18639830784 buffer bytes and 145037791104 scalar work units
across all six prior/supplement models and both old/new metrics. These are
conservative native forecasts, not a matched-compute speed comparison.

Eight adapter tests and eight reviewed component tests pass, together with
formatting, Clippy with warnings denied, full Lean build, axiom/source audit
and generated INDEX checks. Lean has 36 pre-existing unfinished statements,
zero declarations resting on them and zero extra axioms. The exact-commit
review repeats the checks and a separate complete preparation, all 35 cases
and replays, then compares every JSON artifact and compact report.

Reproducibility inputs and evidence remain ignored on storage:

- Freeze: `.kanban/artifacts/293/freeze-v1`, linked to storage
  `freezes/context-mixture-compare-v1`.
- Outputs: storage `context-mixture-compare/original-v1`.
- Compact report: storage `context-mixture-compare/report-v1.json`.
- Evidence manifest: `.kanban/artifacts/293/completed.sha256`, 332 files.
- Source fingerprint: `1e09a72866e403a9a817d6bc2ab3b0de4e6da98709ab1d77e6f8f5e4110567d6`.
- Binary SHA-256: `4554ecd4481ead11cae16057dc951b945ee1b61ea8f39058936d4045ddfde8f6`.
- Inventory SHA-256: `e444daa0a959d112d11aa6c59b136f3fdb68e5b835807cfd108f613303c38ce0`.
- Summary SHA-256: `f47a43e7ae3799f421cb974b2a0298fc5d4c57399852cfd45a3db1b286c9a9c7`.
- Report SHA-256: `bc69c641f1514238a332ad4acebdd1b97967b0ab9e7a679bda7342976192031c`.

The reviewed component is task292 at `14b8ad6e`, backed by CM45 in task291
at `2f0dc9ba`. The parent comparison is task290 at `347f29e1`, and the exact
context tensors come from reviewed task287 at `c5ba15c6`. The experiment's
parent README records the prospective protocol and reproduction commands.
