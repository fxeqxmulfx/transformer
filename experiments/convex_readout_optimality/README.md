# Complete conditional readout optimality

Does the poor original recall fit still admit a better readout on the already
learned states? CM103 at `6383026a` proves a training-only gap for ordinary
full-class CE. This Rust diagnostic measures its floating-point analogue on
twelve immutable endpoints. It changes no trained model and reads no score
files. Only original training records enter coefficients and probes.

| Input cohort | Content sources | Training modes | Readout updates |
| --- | --- | --- | --- |
| `original-v1` | cautious, refined | Mini, forward-only Mini | 512 radial |
| `long-v1` | cautious, refined | Mini, forward-only Mini | 2048 radial |
| `projected-v1` | cautious, refined | Mini, forward-only Mini | 512 Euclidean |

`inputs.json` is a required reproducibility input: it pins every reviewed
producer fingerprint, binary, endpoint and content hash. The adapter checks
these against original metadata, protocols and receipts before reading the
complete model. All 2048 training examples, original supervised masks, repeated
records, 100 classes, 129 components and both complex coordinates remain.
The unchanged original generator constructs its splits; the diagnostic uses
only `train`. The transformer baseline and existing experiment sources remain
unchanged.

The coefficient is the mean of `native_weight * (p_c - indicator_c) * state`.
Native weights already include the Lean capacity scale B. This is an
unconjugated complex coefficient: its real derivative is its real part and its
imaginary derivative is minus its imaginary part. The exact support minimum
is attained by a negative conjugate phase in a largest-magnitude coordinate.
Every original coordinate remains stored, including any zero minimizing value.

The gap is the current complete linear cost plus the sum of complete coefficient
maximum magnitudes. CM103 proves that actual CE minus this gap bounds all
conditional readout losses in exact real arithmetic. Reported floating-point
values have **no rounding certificate**. A small conditional gap would not
prove a raw-accuracy obstruction, a global factor optimum or unseen-label
identification.

The analytic residual is a read-only diagnostic of both saved training modes;
it is not a forward-only training update. All training records stream in
minibatches of 32. Eight fixed interpolations toward the explicit feasible
minimizer are evaluated using actual model logits. No interpolation is accepted,
checkpointed or evaluated on validation/test labels. Reports include complete
coefficient and minimizer hashes, actual probe-logit hashes, CE costs and source
identities. No small coefficient, class, component, duplicate or tail is removed.

Native controls compare all 48 real/imaginary readout coordinates of a complex
model with actual CE finite differences, including duplicate records and a
short tail; test feasible competitors and an actual negative direction; test
256 complex support rows and 3840 feasible competitors; and reproduce CM103's
zero and positive gap witnesses. These checks do not certify roundoff bounds.

Run from the repository with native tooling:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/readout-optimality
export CARGO_BUILD_JOBS=2
cargo test --offline --locked --manifest-path experiments/convex_readout_optimality/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_readout_optimality/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_readout_optimality/Cargo.toml --check
cargo build --release --offline --locked --manifest-path experiments/convex_readout_optimality/Cargo.toml
```

Before running, preserve the producing commit with `git archive --format=tar
HEAD | gzip -6 > source.tar.gz`, copy the release binary, `protocol.json` and
`inputs.json` to a new directory on the storage disk, and record their SHA256
hashes and the binary's `plan` output. Then run a frozen copy:

```sh
bash experiments/convex_readout_optimality/run.sh \
  /home/misha/lean_projects/transformer /run/media/misha/storage \
  /run/media/misha/storage/transformer-runs/rotary-recurrence \
  /run/media/misha/storage/transformer-runs/freezes/readout-optimality-v1 \
  /run/media/misha/storage/transformer-runs/diagnostics/readout-optimality-v1
```

The output directory must be new. Repeat into another new directory and compare
all twelve reports byte for byte. The wrapper admits one worker under a32GiB
address-space cap; native checks enforce110GB aggregate RAM and both disk
thresholds. A space failure stops execution without cleanup. Generated evidence
stays outside the worktree. Original recall failure remains the current finding;
this new diagnostic has not yet produced measurements.
