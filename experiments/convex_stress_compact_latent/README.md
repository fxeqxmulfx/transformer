# Lossless storage for the complete convex latent model

Can repeated parameter and complex-head values reduce memory while preserving
every logical coordinate and the original numerical model? Lean CM70 was
committed first at `cc6931a4`. It proves exact row reconstruction, full-domain
coverage, complete complex-head and indexed-CE preservation, projection-distance
identity and conditional head-storage bounds. The existing full model and
frozen comparisons remain the numerical references.

The current Rust component implements parameter-row storage. Complete head
storage, optimizer integration, whole-model memory admission, checkpoints and
the new original-cohort comparison remain to be implemented and verified.
There is no new accuracy result or original-data production entry point here.

| Property | Current behavior |
| --- | --- |
| Repeated row | One value when every original binary64 word agrees |
| Nonuniform row | Complete explicit class row; arbitrary unknown completions remain representable |
| Row identity | Every original row, its order and class count survive; equal rows are not merged |
| Numeric words | Preserve signed zero, subnormal words and NaN payloads as raw storage; model-level finite-value validation remains required |
| Iteration | Visit every original logical class entry in order, including repeated values |
| Transformation | Apply a zipped operation at every original coordinate; rows can move in either direction between shared and explicit storage |
| Allocation | Bound metadata, actual capacities, one scratch row and old-plus-new buffers during moving growth; checked cloning requires an explicit limit |

The parameter codec specializes CM70's general baseline/exception scheme:
a nonuniform row uses a full explicit fallback even when some of its values
repeat. This preserves all coordinates while keeping lookup and growth simple.
The later head implementation can use the more general exception representation.
Neither form adds a uniformity constraint to the feasible model domain.

Six native controls pass. Mixed shared and explicit rows retain every word,
including duplicate original rows and two distinct zero words. Stateful zipped
operations see every coordinate, and a floating sum matches the full original
sequence bit for bit. A sum of a thousand copies of 0.1 also demonstrates why
replacing the original additions by multiplication would change the result.
Malformed dimensions, incomplete rows, oversized metadata, tight storage caps,
clone refusal and a growth peak exceeding the cap are rejected. A failed growth
leaves already stored rows unchanged.

The model control uses an actual two-program, nonconstant observed dictionary,
uniform and nonuniform unknown rows, a duplicate record and a subnormal entry.
It reconstructs every parameter bit, every real and imaginary coefficient and
all 16 possible query logits through the prior full implementation; a nonzero
imaginary coefficient is checked explicitly. This establishes a codec boundary
result. Actual full/minibatch/forward training with compressed storage still
requires the complete integration and independent comparison.

The storage limit is local to an output row table and its construction scratch,
with a maximum of 96 GB. Input tables, other model tensors, dictionaries and
heads remain separately owned by the caller. A future complete runner must
charge their simultaneous peak under the 110000000000-byte process RAM cap,
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
