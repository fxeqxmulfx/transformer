# Original CM42 raw-label results

The fixed context decoder improves Fibonacci raw-token accuracy from
3120/4704 (66.3265%) to 4574/4704 (97.2364%) and ordinary validation CE from
8.08163266 to approximately 0.27997472 in full-batch, minibatch and forward-only
modes. Primary field accuracy remains 4701/4704 (99.9362%). Both initializer
policies retain 35/35 primary and auxiliary passes in every mode, with zero
process failures. All 35 case processes and original independent replays exit 0.

| Mode | Raw accuracy before | Raw accuracy after | CE before | CE after | Primary field accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full batch | 66.3265% | 97.2364% | 8.08163266 | 0.2799747191651211 | 99.9362% |
| Minibatch | 66.3265% | 97.2364% | 8.08163266 | 0.2799747191653473 | 99.9362% |
| Forward only | 66.3265% | 97.2364% | 8.08163266 | 0.2799747191650992 | 99.9362% |

The unchanged original success criterion remains decisive. Raw-label errors
fall by 1454, with 130 still present. Balanced raw accuracy reaches 92.5133%.
Original observed raw CE remains 9.626594307035853e-9, and no optimizer update
or gradient batch is added. This is the original seed 1 experiment; it is not
a new three-seed result or a proof of all unseen raw answers.

Three distinct saved pair heads are transformed, shared by the two initializer
policies. Each policy has three fresh changed choices and 102 explicitly
inherited choices: one fresh Fibonacci endpoint plus 32 inherited tasks per
mode. All 35 conventional transformer endpoints and 336 depth outcomes remain
in the complete summary. Their measurements and all earlier constructor costs
are retained. No choice between initializer policies is made.

The retained conventional transformer's Fibonacci endpoint has 95.1318% raw
accuracy, validation CE 0.23454365089024853 and 99.0434% primary field accuracy.
The new context head has higher raw and field accuracy; the transformer has
lower CE. These transformer measurements are inherited from the reviewed
baseline, without a new transformer run or a matched-compute claim.

Both the prior and supplemental head select five uniformly weighted neighbors
using the fixed canonical training-only CE diagnostic. The complete 4704
training record IDs retain their multiplicity. Excluding the query state gives
canonical CE 0.14218948 for the prior and 0.14729152 for the supplement; the
original default gives 8.16326532 and 8.16836736 respectively. Neither trained
coefficients nor held-out labels enter that selection. Actual predictions use
the unchanged saved coefficients and interpolated complex features, with the
original prior/supplement support gate and fallback behavior.

The ordinal metric supplies an explicit locality bias. The complete latent
discovery remains fitted to all training data during the leave-one-state-out
diagnostic. Consequently this diagnostic is not a refitted cross-validation
certificate, and the remaining unseen raw errors are not ruled out by CM42.
Flatness supplies the source feature/head bound. The numerical results do not
turn that bound into an unconditional identifiability theorem.

All three changed heads together have a 15,102,337,536-byte buffer forecast
and 13,642,352,640-unit metric-work forecast. Maximum measured original RSS is
1,662,128 KiB during parent preparation, below the 64 GiB cap. The changed-case
run takes 23.52 s and its original numerical replay 24.58 s; each includes all
three modes. Preparation takes 27.05 s and summary generation 2.05 s. These are
local CPU measurements using the optimized test build profile, not matched
training times against the historical transformer.

Frozen producer: `cb71deb9956f5bd16943e1ceb63f329354383a11`.
The native component is the reviewed `c62022c5`, following CM42 at `81f4e46e`.
Six adapter tests plus eight component tests, formatting and clippy pass.
The full Lean build has 13882 jobs, with 36 existing sorry declarations,
zero results resting on sorry and zero extra axioms. The generated index
remains 4785 modules and 25041 theorems, with zero vacuous/placeholders.

| Evidence | SHA256 |
| --- | --- |
| Source fingerprint | `7d85ddbe8a4548e6656626bab48069ca3159747fc2f2cd03b0d87409b611ec56` |
| Frozen executable | `9316667dd6e2aa5fcddfe7f02c0b74968da1ad2a7ae4ffcd2a0433123f136b25` |
| Original inventory | `52c82e6804f66d3e00dd66314ce194e61dcb71379258f51bec51c27e3632948f` |
| Original summary | `e8adcd6bf3ee3db8ec465731c70215a920ccdf7942bbd383d9adead1f11ef964` |

Generated artifacts remain on storage at
`/run/media/misha/storage/transformer-runs/context-compare/original-v1`.
The request, binary and source inventories are frozen at
`.kanban/artifacts/284/freeze-v1`; the completed manifest covers 329 original
files including the compact report and preparation evidence. Regenerate the
report with the source `../report.jq` and reproduce the run using the commands
in the parent README. An exact-commit review must additionally prepare a fresh
directory, reproduce all 35 processes/replays, and compare inventories, model
artifacts, complete metrics, summary and report bytes with the frozen original.
