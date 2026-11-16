# Lossless storage for the complete convex latent model

Can repeated parameter and complex-head values reduce memory while preserving
every logical coordinate and the original numerical model? Lean CM70 was
committed first at `cc6931a4`. It proves exact row reconstruction, full-domain
coverage, complete complex-head and indexed-CE preservation, projection-distance
identity and conditional head-storage bounds. The existing full model and
frozen comparisons remain the numerical references.

The current Rust component implements parameter storage, direct compact
initialization, the complete complex head and actual projected ordinary CE
training in full-batch, minibatch and forward-only modes. Numerical audits,
interval certificates, exact dyadic fallback and compatible streaming
checkpoints are retained. The input-aware budget now counts the complete
dictionary, packed observations and an explicit caller-owned storage reserve
alongside the local training buffers. Compact dictionary construction and
validation use actual capacities instead of the prior dense-model estimate.
The separate complete runners inventory their external allocations and enforce
process admission; an input forecast alone does not admit a larger original
model. The original compact comparison remains queued, while the separate
[`probe comparison`](../convex_stress_probe_compare) has independently verified
five available original probes. This component has no original-data production
entry point and its synthetic controls add no accuracy credit.

| Property | Current behavior |
| --- | --- |
| Repeated row | One value when every original binary64 word agrees |
| Nonuniform row | Complete explicit class row; arbitrary unknown completions remain representable |
| Row identity | Every original row, its order and class count survive; equal rows are not merged |
| Numeric words | Preserve signed zero, subnormal words and NaN payloads as raw storage; model-level finite-value validation remains required |
| Parameter iteration | Visit every original logical class entry in order, including repeated values |
| Transformation | Apply a zipped operation at every original coordinate; rows can move in either direction between shared and explicit storage |
| Allocation | Bound metadata, actual capacities, one scratch row and old-plus-new buffers during moving growth; checked cloning requires an explicit limit |
| Parameters | Preserve every mass, arbitrary unknown row, anchored expansion, original feasibility sums and the full distance with known-row multiplicities |
| Complex head | Share an eight-word column only when all four real and four imaginary coefficient words agree; retain every exception and the original logical active-class sequence |
| Prediction | Preserve forward and reverse contractions, interval enclosures, exact dyadic ties, raw winners and ordinary CE |
| Shared computation | After Lean CM71, identical complete columns may share their ordered forward accumulator; expand every original class before CE or argmax |
| Projection | Preserve weighted proposals, every unknown gradient coordinate, all scalar root iterations and the full numerical KKT audit |
| Training | Preserve the original dictionary check, every original record, duplicates, batch tails, CE guards and step metrics in all three modes |
| Checkpoints | Stream every logical value in the exact prior RDLT6401 format; admit the complete read dictionary, compact output, scratch and fallback before allocation |
| Dictionary | Retain the original labels, sorted unknown IDs, offsets, known multiplicities and logical head bound; validate every canonical index without cloning the dictionary |
| Observations | Retain every original UInt16 state and label, callback order, duplicate ID and batch tail; admit actual capacities and one expanded row |
| Combined budget | Subtract full input capacities and caller-owned storage before each local training call; retain at least 2 GiB runtime reserve within the 110 GB process ceiling |

The parameter codec specializes CM70's general baseline/exception scheme:
a nonuniform row uses a full explicit fallback even when some of its values
repeat. This preserves all coordinates while keeping lookup and growth simple.
The head uses the more general exception representation for each four-state
Fourier block. A baseline class outside the block's observed labels is preferred
when one exists; any other distinct column remains explicitly stored. Known-only
blocks retain their sorted observed classes, and any block with an unknown row
still represents all raw classes, including zero-valued and repeated columns.
Neither form adds a uniformity constraint to the feasible model domain.

Forty-six native controls pass. Mixed shared and explicit rows retain every word,
including duplicate original rows and two distinct zero words. Stateful zipped
operations see every coordinate, and a floating sum matches the full original
sequence bit for bit. A sum of a thousand copies of 0.1 also demonstrates why
replacing the original additions by multiplication would change the result.
Malformed dimensions, incomplete rows, oversized metadata, tight storage caps,
clone refusal and a growth peak exceeding the cap are rejected. A failed growth
leaves already stored rows unchanged.

The original codec control uses an actual two-program, nonconstant observed dictionary,
uniform and nonuniform unknown rows, a duplicate record and a subnormal entry.
It reconstructs every parameter bit, every real and imaginary coefficient and
all 16 possible query logits through the prior full implementation; a nonzero
imaginary coefficient is checked explicitly. Further controls build the actual
compact head on a two-program, eight-state dictionary with original duplicate
records, known-only blocks and mixed shared/nonuniform unknown rows. Every
coefficient and all 64 queries match the complete prior model bit for bit,
including CE, certificates, enclosure widths and logical work charges.

Additional controls span 2, 3, 4, 5, 7, 97, 193 and 1024 raw classes; zero
program mass; arbitrary negative proposals; signed zeros and subnormal products;
opposite unknown answers and genuine exact ties. Invalid parameters, queries,
support indices and nonfinite coefficients are rejected. A tight allocation
control admits the eventual head but refuses moving growth when old and new
buffers cannot fit together; prior blocks remain unchanged.

In the actual 128-state, 1024-class all-unknown uniform control, the compact
head stores 256 coefficient words versus 262144 in the prior representation.
Each output head and parameter container occupies less than 8 KiB, excluding
inputs and construction/query scratch. All 1024 classes remain in every query
and the audit charges all 262144 logical coefficients. This measured synthetic
storage reduction is not an original-model memory or accuracy result.
Actual training controls cover 48 combinations: full-batch, minibatch and
forward-only policies; learning rates 3 and 1000000; uniform or mixed nonuniform
initial unknown rows; expanded or packed records; and 128 or 130 original
records. Minibatches retain size 127 with singleton or three-record tails.
Final parameters, complete complex coefficients, all step metrics, CE, logical
operation counts and all 64 query answers agree bit for bit with the prior
complete reduced model. A separate full/minibatch replay compares every
intermediate parameter, gradient and head. Neither original records nor
arbitrary unknown coordinates are removed.

Following the user's 2026-11-16 priority, forward-only now processes its CE
passes in batches of the configured size127, retaining zero gradient updates.
Every original row still contributes in its original order, and division by
the complete record count happens only after the final addition. A separate
control with different losses in a127-row batch and its three-row tail checks
exact CE bits and logical work for batch sizes1,7,127 and65536, on both expanded
and packed inputs. It also demonstrates that equally averaging the two batch
means would give the wrong result. Existing full-batch and minibatch numerical
controls remain unchanged; the new comparison treats full batch as optional.

Projection controls also cover arbitrary proposals, all three multiplier
boundary cases, fully anchored programs and mixtures of uniform/nonuniform
programs. The complete objective and numerical KKT residual match exactly.
An entirely uniform program needs only each row's final prefix sum; it is still
computed by the original repeated additions. A 1000-class control detects the
rounding change that multiplication would introduce. Mixed programs keep the
complete sorted values and every prefix entry, including their uniform rows.
An oversized fallback is refused while the original proposal remains intact.

Training admission reserves dictionary reconstruction, original packed-record
expansion, all step metadata and class scratch before initialization. It then
subtracts every simultaneously live parameter, gradient, proposal, projection
and old/new head before building another. Projection separately charges all
plans, class scratch and simultaneous multiplier/mass/KKT vectors. Controls
reject both a cap too small for initialization and one that fits the initial
model but cannot fit its update buffers. Original caller-owned inputs and
runtime/process reserves must also be supplied by the complete runner.

`admission::Budget` takes an explicit process ceiling and external storage
reservation. Its packing, dictionary and training methods account for every
owned input capacity before delegating to the checked local constructors. The
core allowance is at most 96 GB, with at least 2 GiB left for runtime storage.
Caller-owned candidates, interpreters, original input records and persistence
buffers must be counted in the external reservation. Expanded slices do not
reveal their owning vector's spare capacity, so this combined training entry
point accepts the owned packed representation. Component APIs remain available
for comparison controls with explicit caller accounting.

New controls compare dictionary indices and work charges with the prior
implementation, including spare label capacity. Malformed public dictionary
fields are rejected before indexing. Packed construction preserves every word
and callback invocation, and visits preserve original IDs, duplicates and
singleton tails. Tight caps refuse before invoking a state callback; conversion
also reserves its temporary label vector. A 524,288-row all-unknown dictionary
actually constructs below 32 MiB while the old dense-model estimate refuses it.
Every unknown coordinate remains present logically; this is a dictionary-only
measurement, not a trained original model.

The combined budget reproduces all three packed training policies bit for bit
and refuses a case when external storage exhausts the room that the local
trainer alone would accept. Input forecasts for both 962,806-candidate constant
cases retain all 4,656/18,528 original records and all 5,112,868/7,324,366 unknown
rows. With an illustrative 1 GiB external reservation, each leaves more than
50 GB of the core allowance for the model. This does not establish that every
head, arbitrary fallback or update fits; their actual simultaneous allocations
still determine admission. The prospective runner must inventory its own
external data and account for other live processes before launching.

Checkpoint controls compare every serialized byte and SHA256 digest against
the prior implementation, read each format with both implementations and
reconstruct all coefficient words and 64 query products. Signed zero and the
least subnormal survive. The wire format retains all logical coordinates;
reading the uniform 128-state/1024-class stream needs less than 24 KiB of
admitted output and scratch, while its input stream exceeds 1 MB. Caller-owned
input bytes are separate. A nonuniform stream of the same shape is refused at
that small cap and loads losslessly with sufficient room.

Readers reject corrupt dimensions, dictionary counts, hashes, trailing or
truncated data and infeasible/nonfinite parameters even after rehashing. The
declared unknown-row count is checked before allocating derived dictionary
indices. File writes reuse the prior guard for both disks, independent-repeat
reserve, exact length and refusal to overwrite; injected write/flush failures
propagate as errors. Loading now uses compact dictionary admission and validates
all original indices, independently of the old dense-model estimate.

The component storage limit is at most 96 GB. Standalone parameter/head
constructors charge their output and scratch; projection and training charge
their nested live buffers as described above. A complete original-data runner
must use the combined input budget, charge its remaining caller-owned data and
persistence buffers, enforce the 110000000000-byte process RAM cap,
preserve the original operation order and ordinary CE policies, and stop when
either disk cannot fit the complete original and independent repeated outputs.

Run from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/compact-latent-v1
cargo test --offline --locked --manifest-path experiments/convex_stress_compact_latent/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_stress_compact_latent/Cargo.toml -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/convex_stress_compact_latent/Cargo.toml
```

Keep task351, task355, task358 and task359 sources and outputs unchanged.
The final compact producer needs its own complete checks, immutable source
freeze, pre-heldout training seal and independent original-cohort repeat.

## Ordered column sharing, CM71

Lean commit `91281572` proves equality of ordered accumulators for arbitrary
deterministic word operations when every active event and complete payload
agrees. It proves a sufficient criterion for CM70's actual storage: take the
union of all explicit column exceptions and all active classes of known-only
blocks. All classes outside this union have the same complete column and skip
the same blocks. Evaluate one such class, preserving every operation in its
original order, and copy its final word to the other outside classes.

The Rust change applies this only to `Head::logits`. Every raw output class is
materialized before ordinary CE; its class summation order is unchanged.
Reverse contractions, interval enclosures, exact dyadics, full norm audits,
gradients, projections and checkpoint format remain their original algorithms.
The full coefficient tensor remains accessible. Every new head rebuilds its
plan; arbitrary nonuniform coefficients remain representable. When fewer than
two outside classes remain, evaluation uses the complete original loop. The
conservative union can miss other equal columns, especially when the stored
baseline is an unusual value; no maximal grouping or universal speedup is claimed.

The plan adds two UInt16 lists whose total length equals the raw class count,
plus container metadata. Construction first drops the Fourier scratch row,
then admits the complete existing head, a class marker vector and both lists
together. `allocation_bytes` includes actual plan capacities. No per-query plan
allocation is needed. Logical `Work` remains the original complete charge;
`evaluated_columns` separately reports the physical forward columns.

Six additional controls cover global support unions, nonuniform fallback,
signed zeros, subnormals, imaginary coefficients, complete output/CE equality,
metadata rejection and simultaneous allocation refusal. Actual 97-class
training keeps a nonuniform explicit class and a shared complement. Full,
minibatch and forward-only controls compare every final parameter/coefficient,
step metric, checkpoint byte and work charge with the prior complete model at
learning rates 3 and 1000000. Separate minibatch replay compares intermediate
gradients, projected parameters and rebuilt heads, including the 127+3 tail.

An explicitly selected synthetic benchmark compares the same complete tensor
with and without its computation plan, checking every output word and CE first.
It uses 128/512/2048 programs, eight states, 97/193 classes, 16 queries and seven
alternating timing rounds. Both filesystem guards and aggregate 110 GB RAM
admission are checked before each bounded timing block. Run it separately:

```sh
export SHARED_COLUMNS_WORKSPACE=/home/misha/lean_projects/transformer
export SHARED_COLUMNS_STORAGE=/run/media/misha/storage/transformer-runs
prlimit --as=4294967296:4294967296 -- cargo test --offline --locked --release \
  --manifest-path experiments/convex_stress_compact_latent/Cargo.toml \
  benchmark_complete_column_sharing -- --ignored --nocapture --test-threads=1
```

On the current CPU, median forward contraction was 23.4–24.2 times faster at
97 classes and 41.0–43.6 times faster at 193 classes. The largest case evaluates
4096 columns per Fourier term instead of 395264 while returning all 193 logits.
The complete bounded benchmark finished in 4.11 s; peak RSS, including the Cargo
launcher, was 34612 KiB. Generated logs remain in `.kanban/artifacts/363/` outside
the worktree. These timings occurred alongside the older live comparisons.

This is component timing, not a new original fit or an end-to-end speedup.
Frozen producers for tasks 351, 355, 360 and 362 remain unchanged. Any future
comparison using CM71 needs a new committed source freeze, complete training
seal, independent rebuild/repeat and raw replay before accuracy credit.

The fixed `examples/phase_profile.rs` measures the remaining costs separately
before further optimization. Four synthetic fixtures use 128/512 programs,
97/193 classes, eight states and all 130 records with a 127+3 minibatch tail.
Three repetitions report median construction, loss, gradient, proposal,
projection, interval/exact prediction and complete forward/minibatch times.
Endpoint digests bind every parameter, complex coefficient, CE and step word;
these and logical work counts permit comparison after a representation change.
Both filesystem guards and a 4 GiB reservation within aggregate 110 GB RAM are
checked before each timing block. No original data or task accuracy is involved.

At CM71, the 512-program, 193-class fixture took approximately 21.1 ms to build
the head, 33.0 ms for projection, 4.46 ms for interval prediction and 36.4 ms for
exact prediction. Complete forward/minibatch runs took 63.7/217.8 ms. The earlier
forward-contraction factor therefore cannot describe complete training speed.
These initial timings were collected alongside the older live comparisons;
generated evidence is local in `.kanban/artifacts/364/phase-before-v1.jsonl`.

```sh
cargo build --offline --locked --release \
  --manifest-path experiments/convex_stress_compact_latent/Cargo.toml \
  --example phase_profile -j2
# Use the two SHARED_COLUMNS_* paths above and the configured Cargo target.
prlimit --as=4294967296:4294967296 -- \
  "$CARGO_TARGET_DIR/release/examples/phase_profile"
```
