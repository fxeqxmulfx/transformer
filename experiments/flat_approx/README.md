# Approximate spectral recovery on frozen cubic checkpoints

Can the new quantitative recovery bound explain the saved classifiers for
`(a+b)^3 mod 97`? The fixed full-character candidate does not: **none of the
24 checkpoints has a held-out query with a positive spectral budget**, even
when execution error is optimistically set to zero. The source models still
classify well. No model, optimizer, checkpoint or training run was changed.

The [protocol](PROTOCOL.md) was fixed before measuring results. All four arms,
both seeds and all three stored steps (0, 5000, 10000) are retained in
[report.json](report.json). Probes and the Fourier decoder use only training
data. Validation is the previously examined split, not a new test set.

## Final checkpoints

Every row has 4705 held-out inputs. “Direct budget” is the fraction whose
measured combined contrast error is smaller than the ideal raw gap. “Spectral
budget” substitutes the componentwise nuclear-norm recovery bound. Both are
FP64 diagnostics at zero assumed execution error, not certified executions.

| Arm | Seed | Source accuracy | Residual only | Direct budget positive | Spectral budget positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | 99.43% | 1.28% | 97.32% | 0 |
| Real SwiGLU | 11 | 99.17% | 0.98% | 97.51% | 0 |
| Complex diagonal | 7 | 99.83% | 1.21% | 97.24% | 0 |
| Complex diagonal | 11 | 99.74% | 1.13% | 96.75% | 0 |
| Complex radial product | 7 | 99.68% | 1.64% | 97.34% | 0 |
| Complex radial product | 11 | 98.79% | 1.85% | 97.41% | 0 |
| Complex plain product | 7 | 99.66% | 1.53% | 98.30% | 0 |
| Complex plain product | 11 | 99.68% | 1.74% | 98.79% | 0 |

All 16 trained checkpoints' training-fitted ideal decoders classify all 4705
held-out inputs. This decoder is a diagnostic fitted to source scores using
the known sum variable; it is not the network's forward computation. In
contrast, the source uses raw residues and its unchanged learned parameters.

## Where the proposed bound fails

The masks admit the declared eight perfect matchings and all three-edge
conditioning paths. Their estimated certificate slacks are 0.18034 and 0.15569.
The dominant costs come later. At the final checkpoints, averaged over the 48
character features, the signed objective excess is 82.57–131.39 and the observed
noise nuclear norm is 398.73–415.43. The resulting entry bounds average
11383.95–14018.79, while the largest measured held-out entry error is only 3.384.
After applying the decoder, even the best held-out spectral budget in each
final case is below -120000; ideal raw gaps range from about 3.38 to 8.93.

Thus this bound is not close to a numerical precision threshold. The character
probes themselves also have substantial error (mean held-out squared error
0.770–0.840). These facts do not identify which different representation the
network learns or show that every spectral certificate would fail.

The user's follow-up prompted an additional Lean extension:
[answer equivalence](../../review/flat_spectrum.md#answer-equivalence). Cubic
answers identify sums `s`, `35s`, and `61s`; full additive-character coordinates
distinguish them. The new fiber-readout theorem proves that errors balanced
inside answer fibers leave all scores unchanged, and bounds the combined
contrast after summation. This supplies a sound alternative theoretical target.
It does not establish that the measured probe errors lie in those invisible
directions. That source-alignment and conditioning analysis remains open.

## Validation and reproduction

All 24 state/history hashes and both split reconstructions pass. Batched FP64
source evaluation reproduces every stored GPU train/validation correctness
count. Its largest logit difference from the frozen scalar FP64 oracle on the
declared 32-point sample is `2.265e-14`. The report contains denominators,
alignment errors, all feature-level terms and immutable checkpoint identities.

Four native tests check rectangular GEMM, complex SVD against known singular
values, a known regression, every source arm against the independent scalar
oracle, and the frozen split/mask properties. Formatting and strict Clippy
checks pass. Following style-only corrections, all numerical report fields
reproduce exactly; only the source hash and elapsed time change. The final
diagnostic took 7.20 seconds, the initial pass 6.91 seconds, excluding compilation.
No training was performed. The fixed five-minute diagnostic cap was respected.

From the repository root:

```sh
cargo test --offline --release --manifest-path experiments/flat_approx/Cargo.toml
cargo fmt --manifest-path experiments/flat_approx/Cargo.toml --check
cargo clippy --offline --release --manifest-path experiments/flat_approx/Cargo.toml --all-targets -- -D warnings
cargo run --offline --release --manifest-path experiments/flat_approx/Cargo.toml -- /path/to/rotate90_complex/runs/cubic-adamw /tmp/flat-approx-report.json
```

The diagnostic links system OpenBLAS and uses four CPU threads. The frozen
oracle is byte-identical to native experiment b70811e's `reference.rs`, SHA256
`e5aeb985c308583f08db96ed347a53c288a51106dc4a6d9ea677bd18e3ec27de`.
The report binds the protocol and compiled source inputs by SHA256. These are
same-assistant checks, not independent review. Successful numerical replay does
not replace a proved FP32 execution-error certificate.
