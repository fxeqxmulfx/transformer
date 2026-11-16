# Reduced wide-family comparison on the original stress tasks

Does the complete 299,551-program input family improve raw heldout accuracy
when trained through the CM65/66 reduced joint decoder? This runner ports the
54-case comparison at `ca6a4977`, uses the independently reviewed reduced
component at `bff278c4`, and uses the fixed wider bank at `2b55e14f`. CM65/66/67
were proved before these changes. CM68 (`84a455b7`) proves the subsequent
lossless UInt16 observation representation before its native implementation.
Flat spectrum bounds the complete complex head; it does not identify unseen
labels or certify that the chosen family contains the target.

| Part | Behavior |
| --- | --- |
| Input family | Every compatible member of the fixed 299,551-program bank; no task identity, target formula, seed, oracle or heldout label enters the model |
| Parameters | All shared masses and every raw-class entry of every unknown decoder row; observed rows use CM65's exact affine reconstruction |
| Observations | Every original program state and label stored as a checked UInt16 word, with one decoded row during iteration; original IDs and multiplicities preserved |
| Full batch | Four projected CE updates, learning rate 1,000,000 |
| Minibatch | One original-order epoch, batches of 127 including the final short batch |
| Forward only | Initial total mass eight, no updates |
| Domain | Total mass in [1,32], shared nonnegative decoder rows; CM66 weighted projection |
| Checkpoints | RDLT6401 preserves every free binary64 bit and every dictionary entry; both the entire file hash and body hash are bound |
| Head | Reconstruct every complete block-Fourier coefficient with the reduced component's fixed arithmetic, then independently audit the head before scoring |
| Scoring | Complete original train and validation denominators, all class logits, raw labels, ordinary CE, original sequence-success rule, genuine ties and explicit refusals |
| Reproduction | Seal every training endpoint before any heldout scoring; replay every saved raw row, then independently rebuild and repeat the whole cohort |
| Resources | Complete 96 GB core-buffer forecast inside a 110 GB process cap; reserve both runs and check both disks before and during writes |

Memory refusals preserve the complete selected support, its audit, and all
original case/record counts. They receive no accuracy credit. Missing source
cases, empty support, training failures and prediction failures remain explicit.
I/O or disk exhaustion stops execution and retains partial evidence. The
checkpoint forecast charges every free class coordinate and all three modes;
only reconstructible observed copies and deterministic head storage are omitted.

The immutable probe audit keeps its legacy expanded-record memory forecast and
refusal. Separate `comparison_buffer_bytes` and `comparison_refusal` fields
record complete packed-buffer admission. The first probe's 193-class constant
control keeps all 299,551 candidates; its forecast changes from 122,579,278,464
to 89,281,189,304 bytes, including scratch for one decoded row. This admits the
model under the 96 GB core limit; actual fitting, work limits, peak RAM and
accuracy still require the original run. Oversized unknown tensors remain
explicit refusals. Reports count both legacy and current memory refusals.

Thirteen native controls pass: bit-exact coordinates including negative zero and
a subnormal; every reconstructed real and imaginary coefficient; changed
headers, coordinates, footer, dictionary, file hash and trailing data; genuine
full/minibatch/forward updates on 130 records with duplicates and a 127+3 tail;
raw 99/100 versus 98/100 labels and CE; predictions independent of scoring
labels; rehashed raw corruption; numerical refusal denominators; incomplete
training barriers and changed seals; free unseen completions with identical observed CE and genuine ties; complete oversized-support accounting;
prospective production pins; and disk shortages on either filesystem, shared
filesystem accounting, overflow, bounded writes and overwrite refusal.

The production protocol now pins the verified task354 manifest
`5e5f40d4cd3c8a756d552d0012839daca09c53bffb60a8c80a8f695adf83e674`
and verification
`9806e609908625c357a8463d3ebd26a565c46abc04adabb7c177b7bbdc0f6447`.
All 102 probe artifacts match its independent rebuild/repeat. The packed path
at `e5ef378a` has also passed a fresh source/build review: 15 core, eight family
and thirteen comparison controls. Original model fitting still requires the
prospective freeze below. These prerequisites add no stress-task pass or
catalog credit. The frozen task351 and task354 producers remain unchanged.

Run the native checks from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/reduced-compare-v1
cargo test --offline --locked --manifest-path experiments/convex_stress_reduced_compare/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_stress_reduced_compare/Cargo.toml -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/convex_stress_reduced_compare/Cargo.toml
```

The packed-path review and task354 verification are complete, and their exact
digests are bound in `protocol.json`. Preserve the original probe audit and
report refined comparison admission separately. Rerun the checks and commit
before freezing. `freeze.sh NEW_FREEZE RELEASE_BINARY` saves the native and Lean source
closure. Use the copied executable and scripts for both original and repeat:

```sh
bash FREEZE/source/experiments/convex_stress_reduced_compare/run.sh \
  FREEZE/convex-stress-reduced-compare WORKSPACE NEW_ORIGINAL original \
  ORIGINAL_DATA VERIFIED_WIDE_PROBE RETAINED_SUMMARIES PRODUCER_COMMIT
bash FREEZE/source/experiments/convex_stress_reduced_compare/run.sh \
  INDEPENDENTLY_REBUILT_BINARY WORKSPACE NEW_REPEAT repeat NEW_ORIGINAL
INDEPENDENTLY_REBUILT_BINARY verify WORKSPACE NEW_ORIGINAL NEW_REPEAT NEW_VERIFICATION_JSON
```

All generated evidence stays outside Git. Historical transformer and original
35/catalog77 summaries are pinned references; this is not a new transformer
training run or three-seed claim. The subsequent authored findings must report
both changes from those references and all outstanding failures.
