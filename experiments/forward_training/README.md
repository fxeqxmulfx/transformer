# Complete local Forward–Forward pilot

Can directional pairwise local learning train the original tied complex
network from scratch, with less total work at the same validation quality?
This is a modified Forward–Forward method, not a replication of Hinton's
label-conditioned algorithm. The [loss diagnostic](../forward_forward/README.md)
established local derivatives; this experiment tests a complete update rule.
**This candidate fails:** local learning ends at 2.23–10.50% validation
accuracy, versus 99.28–99.87% for the original CE control. It establishes no
efficiency gain. This conclusion concerns this fixed rule and pilot budget,
not all possible local-learning algorithms.

The [protocol](PROTOCOL.md) was committed at `34da0951` before any candidate
training or timing. It fixes four architectures, seeds 7/11, four methods,
4000 updates, source initialization/data/sampler, and source AdamW. Every
model has the original 143360 independent trainable real coordinates.
There is no hyperparameter search, checkpoint warm start or budget extension.

## Objectives and controls

Let `P(s,y)=mean_{c!=y} softplus(1+s_c-s_y)`. L2 applies P to the original
classifier. L1 applies P to a fixed random class head on normalized hidden
products. L0 applies P to a separate fixed random vocabulary head on the
97 normalized embedding rows, predicting their token identities. All three
coefficients are one. Prototype seeds, normalization and boundaries are
specified in the protocol. All 96 competitors share their feature vector;
L0 also costs 97 vocabulary rows each update.

| Method | Objective and gradient paths |
| --- | --- |
| `ce-bp` | Original CE with full backpropagation. |
| `pair-bp` | L2 with full backpropagation; isolates the terminal loss change. |
| `auxiliary-bp` | L0+L1+L2 with full backpropagation; isolates added objectives. |
| `local-pair` | A receives only L1, B only L2, E the direct L0 and L2 readout contributions; incoming features are fixed. |

Every method accumulates contributions before one original AdamW update,
including the source's signed complex-coordinate tying. Local derivatives
remain necessary. Only cross-block credit is removed. Auxiliary heads have
no trainable parameters or optimizer moments, but their creation, matrix
products, normalization and storage are charged. They are unused at inference.

## Findings

All **32 runs** completed 4000 updates; no cap, failure, restart or extension
occurred. The 1312 full-split evaluations and all 1280 measured training
chunks are retained. Each cell below gives **validation accuracy % / CE**
at the endpoint, including late regressions.

| Arm | Seed | CE BP | Pair BP | Auxiliary BP | Local pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | 99.87 / 0.00916 | 99.79 / 0.05196 | 76.49 / 1.19854 | 2.36 / 7.87976 |
| Complex diagonal | 7 | 99.85 / 0.00942 | 99.83 / 0.03831 | 94.75 / 0.56007 | 4.23 / 6.66695 |
| Complex product radial | 7 | 99.66 / 0.02548 | 96.07 / 0.18434 | 99.51 / 0.11960 | 9.39 / 5.66565 |
| Complex product plain | 7 | 99.83 / 0.01201 | 99.11 / 0.06840 | 98.21 / 0.18245 | 10.50 / 5.50648 |
| Real SwiGLU | 11 | 99.79 / 0.02134 | 96.24 / 0.25691 | 78.83 / 1.13061 | 2.23 / 7.62636 |
| Complex diagonal | 11 | 99.57 / 0.04973 | 99.57 / 0.05141 | 96.60 / 0.47576 | 4.68 / 6.59353 |
| Complex product radial | 11 | 99.28 / 0.07095 | 99.38 / 0.05287 | 98.94 / 0.13702 | 9.46 / 5.51754 |
| Complex product plain | 11 | 99.66 / 0.02601 | 97.85 / 0.10473 | 99.60 / 0.11483 | 9.52 / 5.49034 |

| Method | Ever observed ≥99% | Ever observed ≥99.5% | Matched paired CE endpoint accuracy **and** CE | Active seconds per run, range |
| --- | ---: | ---: | ---: | ---: |
| CE BP | 8/8 | 8/8 | 8/8 | 4.82–8.63 |
| Pair BP | 8/8 | 8/8 | 2/8 | 4.76–8.65 |
| Auxiliary BP | 4/8 | 4/8 | 0/8 | 7.80–14.47 |
| Local pair | 0/8 | 0/8 | 0/8 | 4.73–8.78 |

Thresholds are first observations, not sustained guarantees. Three pair-BP
runs fall below 99% again by the endpoint. The two joint matches are
diagonal seed 11 at step 2200 versus CE's step 2000, and radial seed 11 at
1300 versus 1200. Their observed active times are 3.112 versus 2.911 seconds
and 1.844 versus 1.737 seconds. The comparator is CE's **first** attainment
of its own endpoint criterion, not automatically its full 4000-step time.

The local method's training accuracy is only 37.31–72.94%, with a further
large validation gap. Adding these auxiliary objectives already harms
quality with full backpropagation; removing cross-block credit makes it
substantially worse in every matched case. This isolates an additional
cost of the stopped paths for these objectives. It does not establish which
individual head or boundary causes the failure. Complex multiplication and
the original classifier were unchanged, so these results do not justify
replacing the complex product.

## Timing and memory limits

All **96** GPU ownership snapshots succeeded. Other `pair-adamw-transformer`
compute processes and the desktop graphics process were present during the
cohort. Every group is therefore marked `clean_device_timing=false` under
the frozen rule. The raw measurements remain available, but support no
clean-device speed comparison. No timing-only repeat was selected after
seeing the quality failure.

Active time includes source/prototype setup, training and all scheduled
evaluations. It excludes preflight, compilation/JIT warmup, other methods'
turns, ownership polling and report/checkpoint I/O. The cohort took 248.31
seconds of wall time. Setup ranges from 16.64 to 31.76 ms; full scheduled
evaluation costs 0.50–1.32 seconds per run. Observed training milliseconds
per update range from 1.037–1.831 for CE, 1.032–1.826 for pair BP,
1.790–3.289 for auxiliary BP and 1.009–1.872 for local pair. These ranges
mix architectures, seeds and device interference; they are not speedups.

| Method | Real GPU buffer payload, MiB | Complex GPU buffer payload, MiB |
| --- | ---: | ---: |
| CE BP / Pair BP | 19.825 | 27.151 |
| Auxiliary BP | 21.837 | 31.081 |
| Local pair | 21.504 | 30.415 |

Payload includes the sampler and retained original training buffers, even
where the local method does not use them. It excludes CUDA context/module
overhead. This implementation uses more memory with local heads; it does
not implement a streamed or minimal-storage local trainer.

## Correctness and next decision

All 224 independent FP64 directional finite differences pass at original
steps 0/5000 across four arms and two seeds. Maximum absolute error is
3.31e-8 against the predeclared tolerance. Readout E, vocabulary E, their
sum, local A/B, and both complete BP objectives are checked separately.
All 48 gradient-energy/update checks pass, including signed parameter/first
moment ties and unsigned second-moment ties. All 16 CE next updates match
the source bit for bit. Auxiliary RMS edge cases include zero, tiny, large
and final-coordinate-only inputs at widths 128/341/682; none violates the
fixed tolerance.

All **328** CE reports reproduce original correctness counts exactly and
both split losses within 1e-12. The 73 input files are unchanged. Eleven
Rust tests, formatting and warning-free Clippy pass. No Lean statement
changed; the existing 19 loss theorems prove local facts, not convergence
of this training rule.

The next research step is a diagnostic of how each local optimizer
displacement aligns with the terminal loss gradient, especially the two
contributions to tied E. That full gradient would be an explicitly charged
diagnostic, not free training information. Use the result to specify a new
rule whose block objectives are consistent with the final classifier, then
freeze a new comparison. Do not extend this failed pilot or infer that
another loss name alone restores useful cross-block credit. The older
90.38% result remains certificate coverage, not classifier accuracy.

The follow-up [alignment study](../forward_alignment/README.md) now performs
that diagnostic on all initial and endpoint states. It identifies hidden
block direction and finite-step effects as the next target; it does not
extend this training pilot.

## Reproduce

The shared native build script pins original trainer source
`14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`, original
PTX and the Kernite revision. The pair-loss PTX is pinned to task #192.
The original trainer/checkpoints and Kernite checkout use the same local
paths as `../single_update/`; generated source copies stay in Cargo output.
No Python is used.

```sh
cargo test --offline --locked --manifest-path experiments/forward_training/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/forward_training/Cargo.toml -- -D warnings
cargo fmt --check --manifest-path experiments/forward_training/Cargo.toml
rustfmt --edition 2024 --check experiments/forward_training/src/instrumentation.rs
cargo build --offline --locked --release --manifest-path experiments/forward_training/Cargo.toml
mkdir -p experiments/forward_training/runs
experiments/forward_training/target/release/rotate90-complex-kernite \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/forward_training/runs/pilot
jq -f experiments/forward_training/summary.jq \
  experiments/forward_training/runs/pilot/report.json \
  > experiments/forward_training/runs/pilot/summary.json
```

The executable refuses to overwrite evidence or write within the original
checkpoint directory. Use a fresh directory name for any further audit.
Reports include every observation, chunk time, threshold, endpoint state
hash, input verification and GPU ownership snapshot. Binary endpoint states
are stored alongside each group's result. Generated evidence stays ignored;
commit source, the frozen protocol and authored findings only.

The executed evidence is preserved in
`experiments/forward_training/runs/pilot-01/` in the primary checkout,
with check logs in `.kanban/artifacts/190/`. Its compiled source digest
is `0f63edd42b7aa9968345bdd7b3815c9d3c9b098d378944449a18f174c2f37aa8`.
