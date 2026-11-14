# Conservative coverage extension of complex heads

Can a declared-readout decoder extend a raw decoder without overwriting any
of its supported predictions? CM39 in `review/flat_spectrum.md`, proved at
`2ca6d2cf`, supplies the exact gate and CE identities. This Rust component uses
the actual saved CM34 and CM35 Fourier heads. Original evaluation requires a
separate complete-comparison adapter; no original benchmark score is produced here.

The component accepts two complete certificates with identical original raw
observations, dimensions and field. The certificates' resource policies need
not match. Their actual vector codes, observed dictionaries, Fourier phases,
complex coefficients, conjugate transforms, column budgets and numerical Gram
and singular-value audits must pass. Every original record is bound separately,
including repeated records and its order. A trusted external binding is required
when loading serialized state; rebuilding its support and resource accounting
prevents the stored metadata from authorizing its own changes.

| Prior support | Supplemental support | Actual logits |
| --- | --- | --- |
| Yes | Either | Prior |
| No | Yes | Supplement |
| No | No | Prior |

The complete categorical range is checked for CM36/37 coverage inclusion.
Prediction uses the stored complex products and retains distinct unknown-axis,
unmatched-active-code and exact-observed-input masks. It adds no task-specific
answer rule. Prefix inference reads the original last two categorical tokens
within the declared length2–4096 domain. Invalid inputs fail before indexing.

Both heads remain stored and charged. The coefficient budget is the sum of
their radii; coefficient/table/code scalars, both conservative component memory
forecasts, both certificate forecasts and records are counted. Combined buffers
must fit56 GiB, and numerical processes additionally run under64 GiB. Combined
head audits and each requested derivative batch have32-billion scalar-operation
forecast caps. Field-certificate verification retains its original bounded
checkers and policies. Complete resource records must include verification,
support enumeration, serialization and hashing; those costs are not head
optimization. These forecasts are separate from measured RSS and runtime.

For every original record-ID batch, the routed logits equal the prior logits
bit for bit. Ordinary full-vocabulary CE and its derivative in the real and
imaginary prior coefficients therefore retain the original denominator, even
with repeated inputs or a shorter tail. The derivative formula ports
`convex_stress_kernel/src/training.rs` at `2ca6d2cf`, removing its rejection of
repeated categorical cells while retaining unique original record IDs. No
supplemental derivative or new update schedule is introduced. This component
checks feasible tensors; an original adapter must also bind and replay their
saved full/minibatch/forward optimization histories. The ambient derivative
does not assert feasibility of arbitrary perturbations under the narrower
native table/default constraints.

Six native checks use a genuine complete two-dimensional raw kernel and a
one-dimensional grouped kernel over the same observed data. They exercise a
strict new prediction, prior preservation, complex versus realified products,
both budgets, distinct refusal masks, full/mini/zero-update states, changed
overlapping logits, serialization/certificate mutations and resource limits.
Finite differences at a feasible head with nonzero CE derivatives test every
prior real/imaginary coefficient, a repeated record, a3+1 minibatch split and
the frozen supplemental derivative. Long prefixes cover2,65 and4096 tokens.

The supported new answer is not certified correct by this gate. The existing
selected primary counts remain34/35 full,34/35 mini and33/35 forward until a
complete original comparison supplies new evidence. The conventional
transformer remains33/35 primary. DiffGPT's recorded35/35 on each of seeds0/1/2
is a separate historical result. Frozen255/261/267 protocols are unchanged.

`protocol.json` declares a future separate transformed arm: extend only a
selected raw complete-kernel head using its available same-prime, same-mode
grouped counterpart, with unchanged observed raw CE and positive added coverage.
Keep every old arm and all failures, and report all35 primary outcomes and any
regression. This is a structural prior, not selection by held-out accuracy or a
strict training-loss improvement. Component fixtures cannot establish a full pass.

From the repository root, with a fresh target directory on the storage disk:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/coverage-extension
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
manifest=experiments/convex_stress_coverage_extension/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path "$manifest" -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path "$manifest" --all-targets -- -D warnings
```

After committing and checking the source, freeze it into a fresh directory:

```sh
freeze=/run/media/misha/storage/transformer-runs/freezes/coverage-extension-v1
bash experiments/convex_stress_coverage_extension/freeze_source.sh \
  "$PWD" "$(git rev-parse HEAD)" "$freeze"
prlimit --as=68719476736:68719476736 -- \
  "$freeze/coverage-extension-tests-v1" --test-threads=1
```

Keep generated evidence ignored outside temporary worktrees. The source graph,
locked dependencies, toolchain and compiled test executable support component
review. An original adapter requires its own frozen inference executable,
approved parent evidence, immutable models and complete independent replay.
