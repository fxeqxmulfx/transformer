# Complete three-term input family for the bounded complex decoder

Does a larger fixed input family admit more original training sets while keeping
every compatible candidate and the complete convex decoder? CM67, proved first
in `f1783d04`, embeds every old feasible tensor at unchanged complex logits,
ordinary CE and spectral budget. This component uses the CM65/66 reduced model
from `bff278c4`. The input family is a structural prior, separate from the
flat-spectrum head-capacity bound.

| Part | Fixed behavior |
| --- | --- |
| Old prefix | Every one of the 4,767 original pair programs, in the same order |
| Added family | All triples of distinct Laurent monomials with coordinate exponents in `[-3,3]`; first coefficient 1, other coefficients independently in `[-2,-1,1,2]` |
| Candidate count | 294,784 additions; 299,551 total, generated before inspecting labels |
| Selection | Retain every consistent candidate; preserve a two-record conflict witness for every eliminated block |
| Interpreter audit | Independent direct powers and Fermat inversion, with repeated unsigned additions for triples; check every retained state on every original row and every conflict witness |
| Decoder | Every shared mass and every unknown raw-class entry remain variables; observed rows use the proved affine reconstruction |
| Confidence and policies | Same total-mass interval `[1,32]`, initial mass 8, full 4 epochs, mini 1 epoch with batch 127, forward 0 updates, learning rate 1,000,000 |
| Probe | Training compatibility and complete buffer forecasts; no fitting updates or heldout scoring |
| Resource admission | 40 trillion charged operations, 96 GB buffer ceiling within the 110 GB process cap, both-disk checks reserving the original and repeat |

The probe consumes only the vocabulary and ordered original rows `[a,b,label]`.
It receives no task identity, target formula, generator seed, oracle or heldout
labels. Its shell adapter copies the independently verified training extracts
from the previous probe; it never deserializes the original heldout datasets.
The full 54-case inventory preserves the six unavailable sources. Existing
35-task, 77-case and transformer summaries remain unchanged.

Selection streams one candidate's dictionary at a time. It hashes every padded
state, including unknown entries, and every retained program's state on every
original record. The resulting counts forecast the entire reduced model,
including expanded original records, free rows, projection buffers and complete
complex heads. This forecast can refuse a model even though its streamed
training audit fits easily. The materializing API recomputes the complete
candidate partition, then verifies both hashes and all counts against the
constructed reduced domain. A forged surviving subset is rejected.

The separate `encode_packed` API follows the CM68 proof at `84a455b7`. It preserves
the original selection and audit, including any expanded-record memory refusal,
while computing packed-buffer admission separately. It checks every stored
UInt16 word after decoding against the original program-major state digest,
then reconstructs and checks the complete dictionary. No candidate, unknown
coordinate or original record is removed. Controls compare every complex
coefficient bit, CE, parameters and update traces against expanded encoding
in all three modes; reordered observations and forged audits are rejected.
This API is for a prospectively frozen comparison. The existing task354 probe
continues to use its original expanded-record forecast.

Eight native controls pass. They enumerate the complete family independently,
check both interpreters for every candidate and all nine input pairs at prime 3,
and sample boundary inputs at primes 17, 193 and 1021. An independent pairwise
collision definition checks the full partition. Label renaming, duplicates,
conflicting repeats and forged selection records are covered. A noninjective
synthetic three-term decoder is materialized and trained in all three modes,
including a `4 + 4 + 1` minibatch tail. A 1,021-class constant fixture retains
all candidates while explicitly refusing its oversized model buffers. These
tests measure no original-task accuracy and supply no target-realizability
certificate.

The frozen original producer `2b55e14f` and an independently rebuilt repeat
completed all 48 available training extracts. All 102 deterministic artifacts
match exactly, including the ordered training inputs, complete selections,
state/dictionary audits and summary. The manifest SHA256 is
`5e5f40d4cd3c8a756d552d0012839daca09c53bffb60a8c80a8f695adf83e674`;
verification SHA256 is
`9806e609908625c357a8463d3ebd26a565c46abc04adabb7c177b7bbdc0f6447`.

| Original-cohort outcome | Count |
| --- | ---: |
| Original inventory, including unavailable sources | 54 |
| Available training extracts, all independently repeated | 48 |
| Nonempty compatible support | 43 (previous bank: 41) |
| Models admitted by the original expanded-record forecast | 42 |
| Empty compatible support | 5 |
| Source unavailable | 6 |
| Training updates / heldout pools / accuracy measurements | 0 / 0 / 0 |

The additional compatible cases are `arithmetic-cubic-mixed` and
`arithmetic-quadratic`, each with one surviving candidate. The remaining empty
supports are `arithmetic-floor-product`, `arithmetic-mixed`,
`arithmetic-parity-cubic-blind`, `arithmetic-parity-cubic`, and
`arithmetic-quadratic-linear`. The 193-class constant control retains every
candidate but exceeds the original model-memory forecast. Its refusal remains
in the probe; CM68's subsequently reviewed packed path supplies a separate
89,281,189,304-byte comparison forecast. Neither compatibility nor refined
memory admission earns an accuracy pass. The full comparison has its own
protocol, producer freeze, training seal, raw replay and independent repeat.

The executable control also verifies the complete output partition, absence of
expanded record allocation, refusal of extra input fields and preservation of
an existing report. It exercises the actual guarded file-writing path.

The process checks workspace and storage space before each case and every
report write. It reserves both complete cohorts plus 2 GiB workspace and
16 GiB storage slack, adding requirements on a shared filesystem. Reports use
the guarded writer from the reduced component, including checks during writing
and on completion. Existing outputs are never overwritten. A disk or process
failure stops the cohort and retains partial evidence; a complete empty-support
or model-buffer refusal remains an explicit result in the complete manifest.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/wide-latents-v1
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_wide_latents/Cargo.toml -j2 -- --test-threads=2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_wide_latents/Cargo.toml --all-targets -j2 -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_wide_latents/Cargo.toml -j2
LAKE_ARTIFACT_CACHE=false lake env lean scripts/FamilyExtensionAxioms.lean
# Commit, then freeze before any original-data probe:
bash experiments/convex_stress_wide_latents/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-wide-latents"
bash FRESH_FREEZE/source/experiments/convex_stress_wide_latents/probe.sh FROZEN_BINARY WORKSPACE STORAGE FRESH_OUTPUT OLD_PROBE OLD_VERIFIED_JSON PRODUCER_COMMIT
# Independently build the frozen source into another fresh target directory,
# repeat the entire probe into FRESH_REPEAT, then rehash all 102 artifacts:
bash FRESH_FREEZE/source/experiments/convex_stress_wide_latents/verify.sh FRESH_OUTPUT FRESH_REPEAT NEW_VERIFICATION_JSON
```

The old-family embedding is a representational inclusion, not the positive-mass
initializer used by the new policy. It therefore does not promise identical
training trajectories or improved heldout accuracy. The existing task351
comparison stays frozen. A later complete comparison of this wider family
requires its own prospective freeze, all training outcomes sealed before any
heldout scoring, and an independent repeat before catalog credit.
