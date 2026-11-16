# Complete composed-family comparison on the original stress tasks

Does the fixed 962,806-program input family improve raw heldout accuracy with
the complete CM65/66 convex decoder and ordinary cross-entropy? This adapter
uses the composition family at `cbde1e2d` and the comparison implementation at
`f5015cec`. CM67 proves inclusion of the old model class before the family
extension, and CM68 proves lossless packing of original observations. These
results provide no target-realizability or unseen-label guarantee. Flat
spectrum supplies the complex-head capacity bound.

| Part | Behavior |
| --- | --- |
| Input family | All 962,806 fixed candidates are tested; every compatible candidate survives with its original ID |
| Model | Every shared mass and arbitrary unknown raw-class coordinate remains free; observed rows use the exact CM65 reconstruction |
| Data | Every original record, order and multiplicity is retained through checked UInt16 observations |
| Full batch | Four projected ordinary CE updates, learning rate 1,000,000 |
| Minibatch | One original-order epoch with batches of 127, including the final short batch |
| Forward only | Initial total mass eight, zero updates |
| Domain | Total mass in [1,32], nonnegative decoder rows with shared mass, CM66 weighted projection |
| Persistence | Exact binary64 checkpoints and independently audited complete real and imaginary head coefficients |
| Scoring | Every original train and validation label, all logits, ordinary CE, exact maximizers, raw99, exact99 and the original sequence-success rule |
| Comparison | All 54 inventory cases, including six unavailable sources and every explicit model or numerical refusal |
| Resources | Complete 96 GB core-buffer admission under the 110 GB process RAM cap; both disks must fit the original and independent repeat |

The numerical model accepts only the original training records or an input
pair. Task names, target formulas, seeds, oracles and heldout labels stay outside
that boundary. A training seal binds every endpoint or refusal for every case
before any heldout pool is scored. All saved raw answers are replayed. An
independent source rebuild must repeat every deterministic artifact before
new accuracy is credited. Historical transformer, original35 and catalog77
summaries remain immutable references.

The adapter reuses the prior model, resource admission, checkpoints, scoring
and complete-cohort summary modules directly through Rust source paths. The
new report reader deserializes the complete probe into strict typed fields
without building and cloning a generic JSON tree. Its 128 MiB limit matches
the composition producer. It checks all training-only counters, work counts,
family and dictionary digests, audit dimensions, candidate order and the
complete disjoint retained/conflicting partition. Each file also remains bound
by the independently verified probe manifest.

Fifteen native controls pass. They exercise actual full/minibatch/forward CE,
130 original records with duplicates and a 127+3 tail, every parameter and
complex-coefficient bit, arbitrary unseen completions and genuine ties,
raw99 thresholds, changed scoring labels, rehashed corruption, incomplete
training seals, complete memory refusals and shortages on either disk. The
new reader also round-trips a complete report padded above the former 64 MiB
limit, rejects an oversized report, and rejects tainted metadata, changed
file hashes, unknown fields and forged candidate partitions.

Increasing the family can increase memory cost. A synthetic admission control
with every new candidate and only three unknown padded states per candidate
already exceeds the core limit at 193 classes, even with packed observations.
It remains a refusal with the complete support and denominators. This size
control is not a measured original-data support, peak RAM or accuracy result.

Production is disabled while task358's full training-only probe and independent
repeat are unfinished. The two probe reference hashes in `protocol.json` are
unset, and both the production entry points and `freeze.sh` reject this state.
The five previously empty training supports are nonempty in the first probe's
completed reports, but those partial findings supply no heldout accuracy.
The verified historical coverage remains 77/165 across separate seed-1 cohorts.

Run the native component checks from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/composed-compare-v1
cargo test --offline --locked --manifest-path experiments/convex_stress_composed_compare/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_stress_composed_compare/Cargo.toml -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/convex_stress_composed_compare/Cargo.toml
```

After task358's complete independent verification, pin its exact manifest and
verification hashes, enable the finalized protocol, rerun the checks and commit.
Create a new freeze from that clean commit before any original fitting:

```sh
bash experiments/convex_stress_composed_compare/freeze.sh NEW_FREEZE RELEASE_BINARY
bash FREEZE/source/experiments/convex_stress_composed_compare/run.sh \
  FREEZE/convex-stress-composed-compare WORKSPACE NEW_ORIGINAL original \
  ORIGINAL_DATA VERIFIED_COMPOSITION_PROBE RETAINED_SUMMARIES PRODUCER_COMMIT
bash FREEZE/source/experiments/convex_stress_composed_compare/run.sh \
  INDEPENDENTLY_REBUILT_BINARY WORKSPACE NEW_REPEAT repeat NEW_ORIGINAL
INDEPENDENTLY_REBUILT_BINARY verify WORKSPACE NEW_ORIGINAL NEW_REPEAT NEW_VERIFICATION_JSON
```

The freeze includes the directly reused comparison sources, all native
dependencies and the Lean source closure. Keep generated evidence outside Git.
An I/O failure or shortage on either disk stops execution with partial evidence
intact. The task351, task355 and task358 frozen producers remain unchanged.
