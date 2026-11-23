# Inkling 2 Mini on the GPT stress tests

How well does Inkling's three-pass text architecture learn the original
short and long tasks when trained from scratch? The release-topology run keeps
the published 24 layers, 128 experts, routing fan-out and attention head counts
at the original GPT task widths. An eight-layer comparison measures a smaller
instance. Each configuration stays fixed across tasks and trains fresh weights
on seeds 0, 1 and 2.
[Canonical GPT protocol](../../rule.md)

## Model

The released text model stores 24 layers in four blocks of six. Its first
and last blocks run once. Its two middle blocks share weights across three
passes. Each later pass restarts from the prelude. Each middle layer reads
four messages from the preceding pass's block history and gates them into
its query, key, value and expert inputs. The low-rank gate learns two tanh
projections and initially admits zero message. Lean proves that closed gates
recover the first pass and that any number of causal replays preserves prefix
causality. [Release configuration](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/config.json)
[Reference forward](https://github.com/huggingface/transformers/blob/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py)
[Lean proof](../../src/Transformer/InklingLoop.lean)

| Component | Released text model | Stress-test model |
| --- | --- | --- |
| Stored layers and applications per token | 24 and 48 | 8 and 16 |
| Layers per block | Five local, one global | One local, one global |
| Hidden width | 2,304 | Original task width, 64 or 128 |
| Attention | 24 query and 8 KV heads, head width 128 | Original task's 1 or 4 query heads, 1 KV head, head width equal to model width divided by query heads |
| Routed experts | Top 8 of 128 | Top 2 of 8 |
| Expert intermediate width | Half model width | Half model width |
| Message gate rank | 64 | Model width divided by 16 |
| Training and vocabulary | Pretrained multimodal weights, production mixed precision | Fresh FP32 text model with the task's symbol vocabulary and tied embeddings |

The eight-layer comparison in the table has model ID
`inkling2-text-8layer-8expert-top2-3pass-tied-fp32-v1`. The
`release-topology` build has ID `inkling2-text-24layer-128expert-top8-3pass-tied-fp32-v1`.
It keeps all six layers per block, the original five-local/one-global
attention pattern, 24 query heads, eight KV heads and gate rank 64. Head
width is the nearest integer to the task width divided by 18, reproducing
head width 128 at published width 2,304. It groups the selected expert
computations by expert and accumulates gradients by supervised-token count
over smaller chunks of each original optimizer batch. A dense numerical
reference checks the outputs and all gradients of the grouped calculation.

The Rust text model includes attention over block residuals, grouped-query
causal attention with a learned sink, content-dependent relative bias, static
and dynamic four-tap causal convolutions, gated attention output and
softcapped SwiGLU experts. The outer readout shares the trainable embedding
matrix and applies the release's logit softcap of 30. Each embedding table
starts with the same Gaussian coordinates as Phasor at the corresponding
width, vocabulary and seed. Dense embeddings enter the inner block. Labels
enter the outer cross-entropy and scorer. LibTorch 2.9 runs native CPU tensor
operations and autograd. Kernite 0.6.0 runs the CPU AdamW update. The model
uses fresh FP32 parameters throughout. These measurements describe this
scaled architecture.

## Results

The eight-layer Boolean checkpoints reached the 99% primary threshold below.
Each row comes from an independently reloaded checkpoint. Training uses a
different initialization on each seed. The primary denominator is seven
or eight novel examples. The independent test and length probes each
contain 64 examples. All scores use greedy free generation through EOS.

| Case | Seed | First Inkling update | First Phasor update | Inkling primary | Independent test | Length 32 | Length 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Boolean AND, shifted | 0 | 132 | 68 | 7/7 | 64/64 | 61/64 | 37/64 |
| Boolean AND, shifted | 1 | 92 | 76 | 7/7 | 27/64 | 29/64 | 35/64 |
| Boolean AND, shifted | 2 | 104 | 84 | 7/7 | 49/64 | 40/64 | 29/64 |
| Boolean AND, random | 0 | 108 | 84 | 8/8 | 58/64 | 35/64 | 28/64 |
| Boolean AND, random | 1 | 92 | 76 | 8/8 | 33/64 | 10/64 | 5/64 |
| Boolean AND, random | 2 | 104 | 84 | 8/8 | 50/64 | 37/64 | 28/64 |

Length transfer varied by seed. On shifted Boolean AND at length 32, Inkling
improved from Phasor's 6/64 to 61/64 on seed 0. On seed 2, Phasor got
64/64 and Inkling got 40/64.

On seed 0 the first positive scheduled observation gave 52/4,705 on
modular addition at update 19, 79/4,705 on absolute difference at update
19, and 44/4,704 on decoded Fibonacci at update 10. All three are early
observations within 100,000-update budgets. Phasor's main comparator
independently replayed all 75 short and 30 long threshold crossings.
The two models used different compute backends. [Phasor cohort](../phasor_value_transport/README.md#canonical-gpt-stress-tiers)

With the released topology, seed 0 scored 78/4,705 on absolute difference
at update 19 and 52/4,704 on decoded Fibonacci at update 10. The 24-layer
models had 81,116,992 and 81,137,344 trainable parameters respectively.
On Boolean shift the same topology had 21,481,088 parameters and scored
0/7 on each seed at update 40. These observations precede the original
budgets. Each architecture has its own cohort counters and checkpoints.

## Reproduction

Use the existing pinned GPT checkout at `.lake/reproduction/gpt`, commit
`5a37c515639f23037d191b0659a5a0853984328d`. The data loader checks
the original split fingerprints. The runner imports the pinned scorer and
WSD schedule. The original minibatch sampler shuffles once per epoch from
seed (+10,000) and retains the last short batch. AdamW uses betas
0.9 and 0.95, epsilon (10^{-8}), decay 0.1 and global norm clip 1.
WSD warms up for 2% of the original budget and starts its cosine decay at
80%. Scheduled observations always score the entire designated validation
split. The first 99% crossing stops the run. A cutoff saves all parameters,
both moments, the observation history and a protocol-bound digest. Resuming
reconstructs the sampler position. A separate process reloads each endpoint
and scores all held-out splits.

From this directory, set `LIBTORCH` to the extracted native 2.9.0 CPU
distribution. The Linux x86-64 archive used here has SHA-256
`22de42abde933be46ce843467930bd0190b72271bfa2c11f84db95591a9834f1`.
[Native LibTorch archive](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.0%2Bcpu.zip)

```sh
export LIBTORCH=/absolute/path/to/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset LIBTORCH_USE_PYTORCH
cargo +stable test --release --locked --offline -- --test-threads=1
cargo +stable clippy --release --locked --offline --all-targets -- -D warnings
cargo +stable build --release --locked --offline
INKLING_JOBS=6 INKLING_THREADS=2 ./cohort.sh run runs/canonical
./cohort.sh report runs/canonical
```

Build the 24-layer topology in its own target directory and give its
checkpoints a separate output directory.

```sh
export CARGO_TARGET_DIR="$PWD/target/release-topology"
cargo +stable test --release --locked --offline --features release-topology -- --test-threads=1
cargo +stable build --release --locked --offline --features release-topology
export INKLING_BIN="$CARGO_TARGET_DIR/release/inkling-stress"
INKLING_JOBS=2 INKLING_THREADS=2 ./cohort.sh run runs/release-topology 1
./cohort.sh report runs/release-topology
```

The optional final argument limits the cohort to a multiple of the original
observation interval. Omitting it advances the saved states through the full
original budgets or first threshold crossing.

A single cell can also be advanced and replayed directly.

```sh
target/release/inkling-stress run suite-boolean-and-shift 0 runs/canonical 4
target/release/inkling-stress run suite-boolean-and-shift 0 runs/canonical 320
target/release/inkling-stress replay suite-boolean-and-shift 0 runs/canonical
```

Tests cover open and closed gates across three passes, causality, fixed-input
label invariance, finite differences through shared weights, scalar AdamW
agreement, and the original field and free-generation scorer. On Boolean
shift, saving at update 4 and resuming to update 8 reproduced every byte
of the uninterrupted optimizer and model checkpoint and all observations.
