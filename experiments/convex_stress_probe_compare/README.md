# Complete causal probe comparison

Does the fixed 176-program causal model extend the complete convex complex
decoder to the original GPT mechanism probes at 99% raw accuracy? This runner
uses `convex_stress_probe_model` at `ba159ae6`, whose arbitrary finite-program
Lean contract was checked before implementation. It keeps all compatible
programs, every unknown state/class coordinate and all original supervised IDs.
It trains the actual CM70 complex head with ordinary CE. The structural input
features are declared separately from the spectral bound.

| Protocol | Fixed behavior |
| --- | --- |
| Source |All six task361 entries, pinned to independently reproduced inventorySHA256`12fd36f6076b922aa959ab0b9c0a9ace11d34a0df3f0579e5445ac107e3f9033` |
| Accepted data |Branch amplitudes, conditional amplitudes, linear transport, order discrimination and key-value binding; every original split |
| Routing |Retain its exact unquantized-teacher refusal; no substitute numeric teacher or silent omitted case |
| Existing baseline |Complete original transformer protocol/history remains in the copied data inventory; verification includes its original final-test reference |
| Policies |Forward-only and minibatch only, batch127 with actual tails; no full-batch requirement |
| Training boundary |Original training tokens, masks and raw labels only; no task identity, formula, numeric teacher, validation or test labels |
| Endpoints |Complete binary64 decoder checkpoint, full logical complex-head digest, inverse and column-L1 audits, CE and actual update records |
| Barrier |All six outcome slots and both required endpoints for each supported case seal before any heldout prediction |
| Raw output |Every predictor call records the exact visible-prefix SHA256 and all raw-class binary64 logits in order |
| Replay |A trace-backed predictor independently reruns original teacher-forced CE, free generation, raw/exact counts and unquantized numeric MSE |
| Verification |Recompute all raw scores again, then require both independent runs to match every input, stage, checkpoint, score, trace and replay receipt; missing/extra files fail |
| Resources |4GiB process address-space cap, aggregate110000000000-byte RAM limit, both disks, bounded files and prospective full original/repeat disk forecast |

Raw-token99%, exact-sequence99% and original numeric success are separate
fields. The numeric criterion remains MSE≤0.001 against the unquantized source
teacher, with the original penalty for invalid or missing components. The
order probe has only its original train/validation split; no test split is
invented. Key-value binding retains its original validation/novel split.
No output is relabeled or canonicalized for the raw-label accuracy.

Preparation copies and revalidates the complete data inventory. It forecasts
the disk upper bound assuming all176 features remain and every decoder row is
unknown, before fitting. Full owning dataset capacities, strings, nested teacher
rows and BTreeMap node allowances are charged alongside a256MiB allowance for
bounded metadata/readers/serialization and1MiB scratch. The model budget then
counts its original packed observations, dictionary and nested live training
buffers. Only one policy's parameters and head remain live at a time. Loading
an endpoint counts its second dictionary and parameters before reconstructing
the full head; the redundant dictionary is dropped after exact comparison.

The production watchdog checks aggregate live RSS and both filesystems every
two seconds. Each process also reserves its full4GiB future growth when checking
the110GB ceiling. Files are capped at1GiB and complete runs at16GiB; before
starting, the actual shape forecast must fit those caps. Both runs plus a16GiB
storage reserve and2GiB workspace reserve must fit. These are disk allocation
bounds, not a110GB disk quota. Shortage stops this process and retains partial
evidence. No existing artifact is overwritten, removed or automatically retried.

Native controls use six explicitly synthetic entries, including an empty
support and unavailable source. They exercise both actual training policies,
the complete seal, all original synthetic dense/generation/numeric splits,
every raw call, independent repeated endpoints and raw replay. Production has
no synthetic fixture switch, and its loader rejects the test inventories.
Separate controls test altered checkpoints, missing modes/replays, truncated
families, changed policy/refusal, extra artifacts, wrong prefix order, signed
zero, nonfinite logits, both disks, RAM and bounded non-overwriting writes.
Changing heldout labels, teachers and task metadata leaves training endpoints
identical. A changed accuracy summary with a rewritten replay receipt still
fails the final verifier's independent raw-score computation. These controls
do not count as original stress-test accuracy.

All eight controls pass in default and release profiles, with strict Clippy,
formatting and shell checks. The full Lean build passes13995 jobs; the global
audit retains36 existing direct sorries, zero results resting on a sorry and
zero extra axioms. The unchanged generated index checks4898 modules and25678
theorems, with zero vacuous or placeholder entries. The current manuscript
was rechecked atCM70 (`cc6931a4`); no new Lean claim is introduced by this runner.

Reproduce checks without Python:

```sh
cargo test --offline --locked --manifest-path experiments/convex_stress_probe_compare/Cargo.toml -- --test-threads=1
cargo test --offline --locked --release --manifest-path experiments/convex_stress_probe_compare/Cargo.toml -- --test-threads=1
cargo fmt --manifest-path experiments/convex_stress_probe_compare/Cargo.toml --check
cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_stress_probe_compare/Cargo.toml -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/convex_stress_probe_compare/Cargo.toml
```

Before original fitting, freeze the clean committed source, required data,
producer commit and executable hashes. Independently rebuild the same frozen
source for the repeat. Do not reuse a partially completed output directory:

```sh
bash experiments/convex_stress_probe_compare/run.sh "$original_binary" "$workspace" "$original_output" "$original_data" "$producer_commit"
bash experiments/convex_stress_probe_compare/run.sh "$review_binary" "$workspace" "$repeat_output" "$independently_regenerated_data" "$producer_commit"
"$review_binary" verify "$workspace" "$original_output" "$repeat_output" "$verification_output"
```

Generated artifacts stay outside the checkout. The frozen original and
independent repeat are now complete and verified. They establish this cohort
on seed 1; they do not establish an extra seed, unconditional unseen-label
identification or a unified model covering the complete catalog.

## Verified original results, 2026-11-16

Producer `1c67b444ce763c9b7674c2ef4b15c12b68583e1e` was frozen with 7290
source paths and independently prepared original/review data. A fresh build
passed all 16 model/runner controls in both profiles and strict Clippy. All 66
artifacts match across the complete runs. Both runs replayed every raw call;
the final verifier independently recomputed the scores again.

All five available probes achieve **100% raw-token and exact-sequence accuracy
in both forward-only and minibatch modes on every original split**. Each mode
scores 52,720/52,720 heldout label occurrences and 3568/3568 complete heldout
examples. These totals retain the original overlapping validation/novel subsets;
they are not counts of independent samples. All original numeric/classification
success criteria also hold. Routing remains an explicit source-data refusal.

| Original probe | Retained programs | New raw / exact, both modes | Recorded transformer raw / exact |
| --- | ---: | ---: | ---: |
| Branch amplitudes |9|100% /100%|100% /100% test |
| Conditional amplitudes |1|100% /100%|99.929% /97.656% test |
| Linear transport |18|100% /100%|100% /100% test |
| Order discrimination |4|100% /100%|94.355% /94.355% validation |
| Key-value binding |2|100% /100%|99.609% /99.609% test |
| Routing mixture |Source refused|Unmeasured|99.173% /87.109% test |

The transformer measurements are the preserved original snapshot, without
retraining or changing its success flags. The new model has an explicit fixed
causal feature prior. Conditional amplitudes retain exactly the five-chunk
contiguous feature with input context position 0. Its raw decoder labels come
from training. The prior is not supplied by a spectral bound, and the result
does not certify target realizability for arbitrary future inputs.

Forward CE lies between 0.0225552 and 0.0326712 across these tasks. Minibatch CE
rounds to zero in the current float64 evaluator; finite logits do not give
mathematically zero CE. Maximum unquantized numeric MSE is approximately
4.26×10⁻¹⁵, far below the unchanged 0.001 criterion. The full original run took
101.40 s and its independent repeat 99.31 s; peak RSS was 68.3 MiB. Each complete run
occupies about 281 MB. The prospective disk upper bound was 1,509,314,313 bytes per
run, separate from the 110 GB aggregate RAM limit and both-disk reserves.

Conditional amplitudes and order discrimination add two catalog cases to the
historically verified 77 under the current **forward+minibatch** requirement:
79/165 across separate seed 1 cohorts, with 86 unmeasured or not passing. The
older three-mode total remains 77; no full-batch result is claimed for the two
additions. The three other measured probes are retained controls. Older
bounded/reduced/composed comparisons are separate and gain no credit here.

Verification SHA256:
`9bd2c2d9489c8421beecb3b29148e148fe9f1a831cc49d1eb875db1bc4924d3f`.
Training seal SHA256:
`5b3bf514de5559665ec937ccac13e968808b18725b990fb12852addb378c3d57`.
Original executable SHA256:
`b5611711855760bcb39c4691e6bcb5bb02268de21296ddb21028b35a28c099df`.
Independent executable SHA256:
`b266829b53ff109bbb8ef83c16b8e2cc0d722a53816ac449f0806b83d1d5fde2`.
The full generated evidence remains under
`/run/media/misha/storage/transformer-runs/probe-compare/`.
Regenerate the count/CE summary with:

```sh
jq -f experiments/convex_stress_probe_compare/analyze.jq \
  /run/media/misha/storage/transformer-runs/probe-compare/verified-v1.json
```
