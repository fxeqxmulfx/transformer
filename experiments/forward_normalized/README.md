# Standalone normalized hidden-gradient training

Can task #194's favorable current-gradient direction become useful training
without borrowing AdamW's step length? **This fixed rule does not repair
training.** Its endpoint validation accuracy is 2.23–11.63%, versus
99.45–99.87% for CE backpropagation. Compared with original local-pair,
endpoint validation CE is worse in all eight cases and accuracy in six.

The [protocol](PROTOCOL.md) was committed at `6df6c833` before outcomes.
All **24 from-scratch runs completed 4000 updates**: four original arms,
fresh seeds 17/23, and three rules. There was no radius search, selected
restart, cap, failure or extension. These seeds were not used in the preceding
direction diagnostic. All original complex products and the classifier remain
unchanged, with 143360 independent real parameters.

## The single intervention

The candidate keeps task #190's local objectives, fixed random prototypes
and boundaries. L1 supplies A's gradient, L2 supplies B's, and tied E receives
the direct vocabulary-L0 and readout-L2 contributions before any update.
Only A's optimizer changes:

`a_new = FP32(a - alpha(t)*norm(a)*g/norm(g))`,
where `alpha(t)=0.01*min(t/10,1)` and `g` is the current local A gradient
pulled back through the complex signed expansion. Norms and pullback use
FP64; new independent owners are rounded once and their aliases rebuilt.
Zero norms skip A and are counted. No such skip occurred during training.

A uses no moments, coordinate preconditioner or decay. E/B keep the exact
source AdamW. The GPU computes the norms, scalar factor and A update; it
does not first compute an A AdamW step or transfer parameters to the CPU
to implement the update. The radius depends only on current A and the
fixed update clock. No validation metric, terminal gradient or line search
controls it. Local Jacobians and gradient matrix products remain necessary.

| Rule | Credit and optimizer |
| --- | --- |
| `ce-bp` | Original source CE, full backward, original AdamW. |
| `local-pair` | Original local L0/L1/L2 credit and AdamW. |
| `local-normalized-a` | Identical local credit; normalized current-gradient A, source AdamW on E/B. |

The intervention removes A's momentum, adaptive coordinate scaling and
decay together. It does not identify their individual effects. A 1% relative
step is a frozen design choice, not an established optimal calibration.

## Results

Each cell gives endpoint **validation accuracy % / CE**, retaining all
architectures, seeds and late regressions.

| Arm | Seed | CE BP | Original local | Normalized A local |
| --- | ---: | ---: | ---: | ---: |
| Real SwiGLU | 17 | 99.66 / 0.02040 | 2.91 / 7.60210 | 2.30 / 8.83211 |
| Complex diagonal | 17 | 99.77 / 0.01431 | 4.36 / 6.65286 | 3.08 / 7.89289 |
| Complex product radial | 17 | 99.45 / 0.03592 | 11.01 / 5.56831 | 9.18 / 6.71961 |
| Complex product plain | 17 | 99.57 / 0.03171 | 10.46 / 5.45497 | 8.82 / 6.76450 |
| Real SwiGLU | 23 | 99.87 / 0.01878 | 2.57 / 7.63762 | 2.23 / 8.88769 |
| Complex diagonal | 23 | 99.79 / 0.01101 | 4.06 / 6.72149 | 3.17 / 7.88323 |
| Complex product radial | 23 | 99.74 / 0.02060 | 10.33 / 5.59243 | 10.99 / 6.30009 |
| Complex product plain | 23 | 99.70 / 0.01895 | 10.01 / 5.53778 | 11.63 / 6.10365 |

| Rule | Ever observed ≥99% | Ever observed ≥99.5% | Matched paired CE endpoint accuracy and CE | Active seconds per run |
| --- | ---: | ---: | ---: | ---: |
| CE BP | 8/8 | 8/8 | 8/8 | 3.35–4.51 |
| Original local | 0/8 | 0/8 | 0/8 | 3.25–4.59 |
| Normalized A local | 0/8 | 0/8 | 0/8 | 3.51–4.70 |

Thresholds are first observations, not sustained guarantees. Radial CE seed
17 falls below 99.5% again by its endpoint. Neither local rule ever attains
its paired CE endpoint's accuracy and loss together. There is no time-to-
matched-quality efficiency result for either local rule in this budget.

Relative to original local-pair, the candidate adds 0.56587–1.30952 to
endpoint validation CE. Its two accuracy improvements are the seed-23
products, gaining 31 and 76 correct answers out of 4705. The other six
lose 16–86 correct answers. Thus the two accuracy gains do not improve both
endpoint criteria. Training accuracy rises in all four product cases but
falls in the four real/diagonal cases; the complete train/validation histories
remain in the report. Training fit remains incomplete, and the training-to-
validation gap remains large.

## What follows

The previous [one-step diagnostic](../forward_step/README.md) remains valid
at its saved states and AdamW-matched radii. It did not promise success for
this new radius or for an entire from-scratch trajectory. These results reject
this particular normalized local rule as a repair within 4000 updates; they
do not show that current gradients, all normalized optimizers or all local
losses are ineffective. They also do not justify changing complex multiplication.

Before another optimizer or loss search, freeze a control with **the same A
optimizer and full-gradient credit**. Keeping the auxiliary objective set
fixed while crossing local/full credit with original/normalized A updates
would distinguish an optimizer/step-policy failure from an effect of stopping
credit. Full-gradient controls are diagnostic training costs, not part of
a proposed forward-only method. This study alone cannot assign the failure
solely to the fixed hidden target, the 1% radius or the absent cross-block
paths. No further training was run here.

## Correctness and cost

All **12 new state-update checks** pass: eight fresh initializations and
four original seed-7 step-5000 audit states. A coordinates have zero measured
absolute error against the independent CPU reference. E/B parameters and
both moments match original local updates bitwise; A moments stay bitwise
unchanged. All **12 edge cases** and **16** fresh local derivative checks
pass, with maximum derivative error 3.70e-8. No tolerance changed.

The original local implementation's **224** directional checks and **16**
CE/source next-step identities also pass again. All **73 input files** remain
unchanged. All **984** scheduled state/metric checks pass; initial metrics
match across all three rules in every group. Fresh-seed CE calls the unchanged
source step directly; no pre-existing fresh-seed trajectory is claimed as a
replay. Fourteen Rust tests, formatting, warning-free Clippy, Lean build,
axiom audit and native index verification pass. No Lean statement changed.

The cohort takes 115.86 seconds. Training executes 96000 backbone forward
batches, 32000 terminal backwards, and 64000 each of output-local, hidden-local
and vocabulary Jacobians. The candidate adds 64000 norm-reduction/scalar
kernels and 32000 A writes; all rules together launch 128000 source AdamW
kernels. Scheduled evaluation adds 19680 forward batches. These counts cover
training/evaluation; the separate correctness preflight is excluded. All 960
training chunks and their synchronized wall times are retained.

Raw training time per update is 0.783–1.047 ms for CE, 0.758–1.067 ms for
original local, and 0.823–1.094 ms for normalized A. Setup takes 16.13–22.34 ms
per run; total scheduled evaluation and state checks take 0.190–0.318 seconds.
Active time includes setup, training and those checks. It excludes compilation/
JIT warmup, preflight, other methods' turns, ownership polling and report/
checkpoint I/O. All 96 ownership queries succeeded and show this process and
GNOME Shell. Under the frozen all-process ownership rule, **zero groups qualify
for clean-device timing claims**. No speedup is established.

| Rule | Real GPU payload, MiB | Complex GPU payload, MiB |
| --- | ---: | ---: |
| CE BP | 19.825 | 27.151 |
| Original local | 21.504 | 30.415 |
| Normalized A local | 21.546 | 30.435 |

The candidate adds 43680 real-arm / 21856 complex-arm bytes of norm scratch.
The retained source A moment buffers and global-backward buffers are still
allocated. These are actual buffer payloads, excluding driver/context/module
overhead; the implementation demonstrates no memory saving.

## Reproduction and evidence

Use the same pinned original trainer, checkpoint inputs and Kernite checkout
as `forward_training`. The build and runtime verify the original source/PTX
and prior local implementation. No Python is used.

```sh
cargo test --offline --locked --manifest-path experiments/forward_normalized/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/forward_normalized/Cargo.toml -- -D warnings
cargo fmt --check --manifest-path experiments/forward_normalized/Cargo.toml
rustfmt --edition 2024 --check experiments/forward_normalized/src/instrumentation.rs
cargo build --offline --locked --release --manifest-path experiments/forward_normalized/Cargo.toml
mkdir -p experiments/forward_normalized/runs
experiments/forward_normalized/target/release/rotate90-complex-kernite \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/forward_normalized/runs/pilot
jq -f experiments/forward_normalized/summary.jq \
  experiments/forward_normalized/runs/pilot/report.json \
  > experiments/forward_normalized/runs/pilot/summary.json
```

The executable refuses to overwrite an output directory or write inside the
checkpoint input. Executed evidence is in root
`experiments/forward_normalized/runs/pilot-01/`; logs and the executable are
preserved under `.kanban/artifacts/195/`. They are ignored and uncommitted.
Compiled-source SHA:
`54edcbfcf0723f6ad2a326c378f20a9335d43b36b45918d25c830c2cc9b70158`.
New normalization PTX SHA:
`0d79bd7f56a06ade36e096cb5ad748e4eac4942fb68d9a6d24dad93e8e8cf313`.
