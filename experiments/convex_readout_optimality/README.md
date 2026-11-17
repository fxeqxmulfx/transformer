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
stays outside the worktree.

Producer `2e1e1aa6` completes all twelve diagnostics. Each retains 2048 supervised
records, 64 minibatches, 100 classes, 129 components and 51,600 real coordinates.
The twelve original endpoints remain byte-identical. Training correct counts
match their original reports exactly; independently accumulated CE differs by
at most 6.67e-16. No model is updated or newly evaluated on held-out labels.

| Cohort | Source/mode | Current CE | Estimated gap | Best fixed probe eta | Probe CE | Training correct, before → probe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| radial512 | cautious Mini | 2.069413 | 0.516121 | 0.25 | 2.021902 | 1455 → 1375 |
| radial512 | cautious forward | 2.058971 | 0.536401 | 0.0625 | 2.032595 | 1459 → 1506 |
| radial512 | refined Mini | 2.094261 | 0.532555 | 0.25 | 2.067526 | 1437 → 1305 |
| radial512 | refined forward | 2.107671 | 0.549955 | 0.0625 | 2.081332 | 1393 → 1434 |
| radial2048 | cautious Mini | 1.920944 | 0.273263 | 0.0625 | 1.908081 | 1598 → 1617 |
| radial2048 | cautious forward | 1.911646 | 0.283142 | 0.0625 | 1.899469 | 1629 → 1632 |
| radial2048 | refined Mini | 1.948229 | 0.283913 | 0.0625 | 1.936016 | 1595 → 1620 |
| radial2048 | refined forward | 1.961521 | 0.305953 | 0.0625 | 1.949316 | 1569 → 1612 |
| Euclidean512 | cautious Mini | 2.042003 | 0.492518 | 0.25 | 2.003725 | 1390 → 1357 |
| Euclidean512 | cautious forward | 2.032235 | 0.509158 | 0.0625 | 2.007436 | 1408 → 1438 |
| Euclidean512 | refined Mini | 2.071083 | 0.514688 | 0.0625 | 2.046180 | 1368 → 1428 |
| Euclidean512 | refined forward | 2.085518 | 0.530963 | 0.0625 | 2.060469 | 1324 → 1377 |

Every endpoint has an actually evaluated feasible probe that lowers training CE,
by0.01218–0.04751. Thus none is treated as an optimum. The lowest-CE probe raises
training correct counts in nine cases and lowers them in three; CE improvement
does not imply raw-accuracy improvement. All eight probes remain in each report,
including worse ones. These unaccepted training-only probes earn no task success
or new held-out accuracy credit.

With current states **and mixture weights fixed**, estimated conditional CE
lower bounds are1.52257–1.56171 for radial512,1.62850–1.66432 for radial2048 and
1.52308–1.55640 for Euclidean512. These floating estimates are not certified
capacity obstructions, especially not raw-accuracy bounds. They also do not
cover reallocation of mixture weights or changing learned content. The next
Lean step is a joint weighted-readout capacity domain that permits capacity
to move among complete learned components before interpreting a readout limit
as a state-representation failure.

Actual complete logits agree exactly with the independent state contraction.
Maximum support-identity error is4.45e-16, change-identity error3.07e-14 and
row-budget excess2.23e-16. Each diagnostic performs20,480 CE evaluations,
245,760 across the cohort, with zero accepted updates. Generated reports are at
`/run/media/misha/storage/transformer-runs/diagnostics/readout-optimality-v1`;
source fingerprint is
`3ef357564512402401981c245fb702b65e8f022108258b7777f4e2c4496a7004`.
