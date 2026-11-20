# ReRoPE and Leaky ReRoPE in the main model

## Question

What happens when gpt's main research model trains with ReRoPE's or Leaky
ReRoPE's position map in place of LeRoPE's? Su (kexue.fm/archives/9708) clamps
RoPE's relative position at a window w, or lets it grow beyond w with slope
1/k. He computes two attention matrices and uses the map only at inference.
Here the model trains with the map, and each score is computed once. Lean
proves that this one pass is ReRoPE's attention for the control's score
(`Transformer.RoPE.ReRoPE.rerope_phase_attend_eq_onlineMerge`,
`leaky_phase_attend_eq_onlineMerge`).

Does the model learn the fixed suite as fast, and does it keep its accuracy on
the length and position-shift splits? What does the one pass cost? The rest
of the model stays as in the control:

- SSMax and Peri-RMS q and k;
- the squared-imaginary score;
- phase-mHC and the complex FFN;
- precision and AdamW/WSD;
- LeRoPE's learned frequencies.

[`PROTOCOL.md`](PROTOCOL.md) holds the frozen plan.

## Runs

| arm | relative position of a query t and a key j, r = t − j | parameters added |
| --- | --- | --- |
| control (gpt `phase-score-lerope`, recorded) | r | none |
| `rerope` | r if r < 16, else 16 | none |
| `leaky-rerope` | r if r < 16, else 16 + (r − 16)/16 | none |

All arms use the same suite: 35 tasks with seeds 0–2, the same budgets and
the same stopping rule (99% exact on the selected split). For a given seed,
every arm starts from the control's weights.

The window is w = 16 and the slope k = 16:

- **Window.** Su finds the best w between 1/4 and 1/2 of the training
  length. Here w is a quarter of 64, the training length of the depth and
  recall tasks.
- **Slope.** k = 16 is his best slope for w at a quarter of the training
  length.

Of the 35 tasks, 25 never exceed 16 tokens, so the arms train them exactly as
the control does. Ten tasks reach beyond the window:

- the depth tasks (training length 64, evaluated to 128);
- recall (64);
- the two Boolean tasks (19, with length splits to 67);
- parity 16, 20 and 24;
- linear transport (66);
- branch amplitudes (98).

## One attention

Each key's score is computed once, from the rotations of its region:

- **Inside the window** (t − j < w), from the query rotated by t and the key
  by j, as in the control.
- **Beyond it**, from the query rotated by w and the key unrotated (ReRoPE),
  or from the query rotated by (t − w)/k + w and the key by j/k (Leaky
  ReRoPE).

A common rotation keeps the score (`phaseScore_rotary`), so only the
difference of the two rotations counts. The softmax and the value sum are the
control's, over all keys at once.

Implementation:

- **Forward.** `rerope_forward.rs` is the control's forward kernel. It also
  receives the far rotations, and it takes q and k from the pair's region.
- **Far rotations.** The far rotations of q and k are a second rotation of
  the normalized q and k. Their tables are made by the control's learned
  table kernel from the far positions, so LeRoPE's frequencies apply to both
  regions.
- **Backward.** `rerope_backward.rs` routes each pair's adjoint to the near
  or far copy of q and k. The far frequency gradient joins the near one
  (`far_rotary.rs`), and the far adjoint is rotated back onto the normalized
  q and k.

Inside the window every operation is the control's, in its order. A batch
with no sequence longer than w runs the control's kernels.

What the learned frequencies see changes with the map. In LeRoPE's logit
model, a band's frequency gradient under a position map ψ is LeRoPE's
Eq. (6) with ψ(s − t) in place of the offset s − t
(`Transformer.RoPE.LeReRoPE.hasDerivAt_positioned_loss`):

- **ReRoPE.** Under the clamp, the far pairs reach the frequency only
  through their summed phasor, at the single offset w
  (`positionedSurrogate_rectified`).
- **Leaky ReRoPE.** It keeps every far offset, at the phase
  (w + (δ − w)/k)θ.

That model omits the control's squared-imaginary term and its gain. The
kernels here differentiate through both rotations instead, and the
validation checks them against central differences.

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

Unchanged modules, `layout.rs` and `optimizer.rs` among them, are compiled
from the gpt clone. Two files are copies with documented changes:
`engine.rs` and `runner.rs`. Three kernel files are new:

- `rerope_forward.rs`: the control's forward, with the pair's region
  choosing q and k;
- `rerope_backward.rs`: the control's attention backward and score
  corrections, with each pair's adjoint routed to its region;
- `far_rotary.rs`: the far rotation, its adjoint and the far frequency
  gradient.

```
cargo build --offline --release
cargo build --offline --release --features fp32-affine-reference --target-dir target/fp32
mkdir runs/cohort-1-bin
cp target/release/rerope target/release/main-phase-score-native runs/cohort-1-bin/
CONTROL=<gpt>/runs/main-model-phase-score-01/phase-score-lerope
./target/fp32/release/rerope validate $CONTROL runs/validation-fp32-1
./runs/cohort-1-bin/rerope validate $CONTROL runs/validation-1
./runs/cohort-1-bin/rerope suite runs/cohort-1 short
./runs/cohort-1-bin/rerope suite runs/cohort-1 long
./runs/cohort-1-bin/rerope analyze runs/cohort-1 $CONTROL
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
- a checkpoint resumes both arms with identical updates, at a window of one
  token, so that every batch has far pairs;
- the archive refuses a changed source;
- the largest suite shapes fit both arms and have finite gradients.

`validation-1` ran the production build, on the frozen binaries in
`runs/cohort-1-bin`.

- **Kernels within the window.** With a window as long as the sequence, the
  routed forward and backward equal the control's bit for bit at D = 16, 32
  and 128. This covers:
  - the output, probabilities and imaginary parts;
  - the adjoints of q, k and v, the score adjoint, intermediate sums and
    score parts.

  The far adjoint, filled with NaN before, is zero after.
- **The attention block beyond the window.** The first layer's block runs at
  length 40, where 300 of each sequence's 820 causal pairs are far, at D =
  16, 32 and 128. Against an f64 oracle that rotates each pair by its
  relative position alone:
  - the outputs are within a relative 3.8e-7;
  - the adjoints of q, k and v, of the learned frequencies and of the score
    coordinates are within 1.0e-6 of the oracle's central differences;
  - a random directional derivative is within 1.4e-5.

  The control passes the same check with its own map, within 3.8e-7,
  2.4e-6 and 5.8e-5, which checks the oracle.
- **Initialization.** In six cases, both arms and the control arm write the
  recorded cell's `initial.f32` byte for byte.
- **Control binary.** Its PTX (`17c689bd…`), suite and plan hashes are the
  cohort's. Of the 104 source files it archives, 103 equal the cohort's byte
  for byte. The other is `research/main_model/model.json`, which the cohort
  archived from gpt's working tree. No code reads it (see
  [`alibi_kerple`](../alibi_kerple/README.md#validation)).
- **Replays.** This binary's runner reproduces recorded control cells bit
  for bit: every observation, the endpoint, the final splits and the initial
  parameters.
  - The control arm, four cells: 18, 48, 8 and 1219 observations, 99% at the
    cohort's epochs 17, 47, 1.4 and 1218.
  - Both arms, the three cells whose sequences all fit in the window:
    `arithmetic-add` (48 observations, 99% at epoch 47),
    `associative-divisor-n2-l1-w0` (33, at 32) and
    `gpt-probe-key-value-binding` (8, at 7).

`validation-fp32-1` ran gpt's `fp32-affine-reference` build. Kernels, blocks
and initialization give the same numbers there. It also checks the learned
frequencies' gradient in the whole model against central differences of the
loss. The step is 1e-2, on the first batch of seed 0, in a two-layer case at
length 19 and a six-layer case at length 64, for the control and both arms.
- The directional derivative is within 0.25% of the backward's (limit 2%).
- The worst error of a coordinate at least 5% of the largest is 1.35% for the
  arms and 3.3% for the control (limit 5%).

In the production build the same check gives directional ratios 0.993–1.28
(the control's: 0.995 and 1.13) and coordinate errors up to 470%. That is
the rounding of the FP16 forward GEMMs. In the two-layer case, the measured
loss differences are off by 1.3–2.1e-5 for all three, against 1.1–2.4e-7 in
the reference build. The arms' gradients there are the smallest (norms
0.0037 and 0.0041, against the control's 0.0080), so their ratios move most.
The production build reports the check but does not gate on it.

## Findings

Pending.
