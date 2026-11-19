# ALiBi, KERPLE and −log(1 + r) in the main model

## Question

Can a distance bias replace LeRoPE in gpt's main research model if it is added
to the logit after SSMax's gain? Does KERPLE's learned logarithmic bias find
−log(1 + r)? This repository proves that −log(1 + r) is the one bias that
reaches the recall threshold at every distance at once
(`Transformer.Grokking.FlatSpectrum.Recall.LogBias`, PB5). The rest of the
model stays as in the control: SSMax, Peri-RMS q/k, the squared-imaginary
score, phase-mHC, the complex FFN, precision and AdamW/WSD.

[`PROTOCOL.md`](PROTOCOL.md) holds the frozen plan.

## Runs

| arm | position encoding | parameters added |
| --- | --- | --- |
| control (gpt `phase-score-lerope`, recorded) | LeRoPE | d/2 frequencies |
| `alibi` | −m_h r after the gain, slopes 2^(−8(h+1)/H) | none |
| `kerple-log` | −r1 ln(1 + r2 r) after the gain | 2 per head, shared by all layers |
| `kerple-power` | −r1 r^r2 (r2 ≤ 2) after the gain | 2 per head, shared by all layers |
| `log-bias` | −ln(1 + r) after the gain | none |
| `nope` | none | none |

All arms use the same suite: 35 tasks with seeds 0–2, the same budgets and
the same stopping rule (99% exact on the selected split). For a given seed,
every arm starts from the control's weights, except for KERPLE's own
parameters.

## Where the bias goes

KERPLE's Eq. 1 (arXiv:2205.09921v2 §4) puts the bias inside the 1/√d of the
logit. Its released code adds the bias after that scaling instead. Here the
bias goes after SSMax's gain ln(t + 1)/D, as in the released code and as
`biasedLogScaleAttention` places it in Lean. If the bias went inside the gain,
the factor ln(t + 1)/D would scale it, and its strength would change with the
query position.

## Build and commands

The build reads two pinned clones in the root checkout's `.lake/reproduction`.
A worktree reaches them through a link of the same name:

- `.lake/reproduction/gpt`: gpt at `5a37c515`, sparse
  (`research/complex_geometry_suite`, `research/main_model`);
- `.lake/reproduction/kgen`: kgen at `a1b6bc2c` (Kernite 0.3.2, the version
  the control's toolchain 1.99.1 accepts).

```
git clone --no-checkout ~/python_projects/gpt .lake/reproduction/gpt
git -C .lake/reproduction/gpt sparse-checkout set research/complex_geometry_suite research/main_model
git -C .lake/reproduction/gpt checkout 5a37c515
git clone ~/rust_projects/kgen .lake/reproduction/kgen
git -C .lake/reproduction/kgen checkout a1b6bc2c
```

Unchanged modules are compiled from the gpt clone. Four files are copies with
documented changes: `engine.rs`, `layout.rs`, `optimizer.rs` and `runner.rs`.
Two kernel files are new: `biased_forward.rs` (one change to the control's
forward) and `bias.rs` (KERPLE's table, gradient and clamps).

```
cargo build --offline --release
cargo build --offline --release --features fp32-affine-reference --target-dir target/fp32
mkdir runs/cohort-1-bin
cp target/release/alibi-kerple target/release/main-phase-score-native runs/cohort-1-bin/
CONTROL=<gpt>/runs/main-model-phase-score-01/phase-score-lerope
./target/fp32/release/alibi-kerple validate $CONTROL runs/validation-fp32-1
./runs/cohort-1-bin/alibi-kerple validate $CONTROL runs/validation-2
./runs/cohort-1-bin/alibi-kerple suite runs/cohort-1 short
./runs/cohort-1-bin/alibi-kerple suite runs/cohort-1 long
./runs/cohort-1-bin/alibi-kerple analyze runs/cohort-1 $CONTROL
```

The cohort runs from a copy of the binaries. The suite starts each cell as a
new process of its own executable, so a rebuild in `target/` cannot change a
cohort under way. The copy that was validated is the copy that trains.

The second binary, `main-phase-score-native`, is gpt's own `main.rs`, built
here. It keeps the control package's name because Kernite mangles each kernel
under its crate's name. Under any other name the PTX, and so its hash,
differs from the cohort's. The `fp32-affine-reference` build is gpt's
numerical diagnostic: every affine map is FP32 in both directions, and it
refuses to train.

## Validation

Both validations passed before the first cell. Each wrote a
`validation.json` in `runs/`, which git ignores, and the numbers below come
from them. `cargo test --offline --release -- --include-ignored` passed its 8
tests. Among them:
- a checkpoint resumes every arm with identical updates;
- the archive refuses a changed source;
- the largest suite shapes fit and have finite gradients.

`validation-2` ran the production build, on the frozen binaries in
`runs/cohort-1-bin`.

- **Kernels.** With a zero table, `biased_forward` equals the control's
  forward bit for bit at D = 16, 32 and 128. With a random table b:
  - its probabilities are within 1.4e-7 of the control's times e^b,
    renormalized;
  - its outputs are within a relative 1.7e-7;
  - its imaginary parts are identical.

  KERPLE's kernels against f64:
  - the tables are within a relative 1.5e-7 (log) and 5.7e-7 (power);
  - the distance sums are within 5.7e-8, and distance 0 receives nothing;
  - the parameter gradients are within 2.2e-7;
  - a second layer's contribution doubles the gradient exactly;
  - the clamps are exact.
- **Initialization.** In four cases, the control arm run by this binary writes
  the recorded cell's `initial.f32` byte for byte. Every arm shares all
  parameters before the trailing group with it.
- **Control binary.** Its PTX (`17c689bd…`), suite and plan hashes are the
  cohort's. Of the 104 source files it archives, 103 equal the cohort's byte
  for byte. The other is `research/main_model/model.json`, a description of
  the model. The cohort archived it from gpt's working tree, and no gpt
  commit holds that version. No code reads it, and gpt's own `profile.rs`
  exempts it when comparing sources with this cohort.
- **Replays.** This binary's runner, on the control arm, reproduces four
  recorded cells bit for bit:
  - every observation (18, 48, 8 and 1219 of them);
  - the endpoint, the final splits and the initial parameters;
  - 99% at the cohort's epochs: 17, 47, 1.4 and 1218.

`validation-fp32-1` ran gpt's `fp32-affine-reference` build. It checks
KERPLE's gradient in the whole model against central differences of the loss.
The step is 1e-2, on the first batch of seed 0, in a two-layer and a six-layer
case, both `kerple-log` and `kerple-power`.
- The directional derivative is within 3.4e-4 of the backward's (limit 2%).
- The worst error of a coordinate at least 5% of the largest is 0.24%
  (limit 5%).
- Kernels and initialization pass there too.

In the production build the same check gives directional ratios
0.978–1.025 and coordinate errors up to 58%. That is the rounding of the FP16
forward GEMMs, so the production build reports the check but does not gate on
it.

`validation-1` failed two checks, both fixed before the cohort:

- The control was built as the bin `control`. Kernite mangles each kernel
  under its crate's name, so the PTX hash differed.
- The gradient criterion was gated in production precision.

## Findings

Pending.
