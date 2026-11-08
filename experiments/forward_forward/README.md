# Directional Forward–Forward loss

Can positive/negative local learning use the direction learned by normalized
complex blocks, while sharing one encoder evaluation across negative labels?
This diagnostic implements and checks a changed local loss. It does not yet
train the entire model with Forward–Forward or measure an efficiency gain.
The follow-up [complete training pilot](../forward_training/README.md)
adds explicit objectives for every block and compares four update rules.

The [derivation](../../review/flat_spectrum.md#forwardforward-reformulation)
and [frozen protocol](PROTOCOL.md) distinguish this variant from Hinton,
[arXiv:2212.13345](https://arxiv.org/abs/2212.13345), §§2–3.3. Original paper
copies are kept in the ignored `papers/arXiv-2212.13345/` directory.

## Reformulation

Encode x once and score each class by `s_c=Re<h,e_c>`. The corrected-distance
goodness `(||e_c||²-||h-e_c||²)/2` has exactly these class differences.
This retains angular information after RMS and preserves the source argmax.
Putting the goodness into ordinary softmax gives CE again, so the candidate
instead uses

`L_local = sum_{c != y} softplus(1+s_c-s_y)/96`.

For wrong classes, `q_c=sigmoid(1+s_c-s_y)/96`; `q_y=-sum q_c`.
The local head derivatives are `dL/dh=sum q_c e_c` and `dL/de_c=q_c h`.
Batch averaging is included in the GPU implementation. All 96 competitors
share the same feature vector. No negative-label sampling is involved.
Unlike multiclass CE, these competitor weights need not give a collinear
gradient. The loss threshold that guarantees every margin is positive is
`softplus(1)/96`, not a percentage accuracy.

The local derivative treats the incoming features as fixed. The source E
is also the input embedding, so updating its readout contribution can affect
future features. A complete trainer still needs explicit losses/prototypes
at each block, detached boundaries and accumulation for shared parameters.
The native probe exposes this distinction rather than treating a partial
readout gradient as the full model gradient. Complex multiplication and RMS
are unchanged.

## Fixed comparisons and findings

| Check | Control | Candidate |
| --- | --- | --- |
| Loss and logit derivatives | Independent stable FP64 formulas | One Kernite pairwise-loss kernel |
| Partial readout derivative | FP64 outer products of executed features and independently computed logit derivatives | Original GPU GEMM with the new local error signal |
| One readout update | Frozen incoming features | Re-evaluate the actual tied network after the same rounded E update |
| Source fidelity | Original saved CE, accuracy, input hashes and compiled PTX | Replayed unchanged source forward; original state restored after each probe |

All **24** original states pass: four arms, seeds 7/11, checkpoints
0/5000/10000. All **225816** examples and **21904152** logit-gradient
coordinates satisfy the fixed tolerances. An additional 1056 synthetic
rows exercise batch sizes 1/32/511/512 and extreme/shifted logits. All
73 original input files remain unchanged, and every saved metric replays.

Maximum absolute errors on source inputs are 4.36e-7 for per-example loss,
1.26e-9 for batch-scaled logit gradients, and 1.89e-8 for partial readout
gradients. Edge-input maxima are 3.33e-6 and 1.90e-8 for loss and gradient.
No tolerance was enlarged after outcomes.

The fixed 1e-3 SGD readout step decreases both computed losses in all 24
probes. Actual tied-network batch-loss changes are below; both meanings of
the change, full precision and every case remain in the report.

| Arm | Seed | Initial state | Step 5000 | Step 10000 |
| --- | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | -1.36e-4 | -7.84e-13 | -4.97e-10 |
| Complex diagonal | 7 | -1.35e-4 | -1.22e-12 | -2.01e-12 |
| Complex product radial | 7 | -1.35e-4 | -1.82e-11 | -3.16e-11 |
| Complex product plain | 7 | -1.35e-4 | -1.24e-10 | -8.21e-12 |
| Real SwiGLU | 11 | -1.33e-4 | -4.90e-11 | -6.63e-10 |
| Complex diagonal | 11 | -1.39e-4 | -1.50e-12 | -6.32e-13 |
| Complex product radial | 11 | -1.25e-4 | -4.54e-9 | -3.91e-9 |
| Complex product plain | 11 | -1.26e-4 | -9.13e-12 | -4.74e-13 |

The trained-state changes are tiny and do not establish a practically useful
improvement or robust descent guarantee. These are single minibatch probes,
not validation improvements, AdamW comparisons or training from scratch.
The 90.38% in task #188 remains certificate coverage, not an accuracy target
or an error corrected by this experiment. No speedup was measured here.

Nineteen Lean theorems prove the goodness identities, normalization facts,
actual pair-loss directional derivative, a per-query correctness threshold
and a local/global derivative counterexample. They do not verify CUDA
roundoff or prove convergence. Seven Rust tests cover independent finite
differences, complex conjugation, stable logits and changed multiclass
gradient direction; GPU agreement is checked separately above.

## Reproduce

The shared build script checks the original trainer source, Kernite revision
and compiled original kernels; it generates instrumentation only in ignored
Cargo output. Dependencies and paths match `../single_update/Cargo.toml`.
No Python is used.

```sh
cargo test --offline --release --manifest-path experiments/forward_forward/Cargo.toml
cargo clippy --offline --release --all-targets --manifest-path experiments/forward_forward/Cargo.toml -- -D warnings
cargo fmt --check --manifest-path experiments/forward_forward/Cargo.toml
rustfmt --edition 2024 --check experiments/forward_forward/src/instrumentation.rs
cargo build --offline --release --manifest-path experiments/forward_forward/Cargo.toml
experiments/forward_forward/target/release/rotate90-complex-kernite \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/forward_forward/report.json
jq -f experiments/forward_forward/summary.jq experiments/forward_forward/report.json \
  > experiments/forward_forward/summary.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ForwardForwardAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
```

The executable refuses to overwrite a report or write inside the original
checkpoint directory. Use a fresh report filename for another audit. Logs,
reports and the final executable are preserved under `.kanban/artifacts/192/`;
generated evidence is not committed. `INDEX.md` retains its explicit exception.
