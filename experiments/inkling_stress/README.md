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

| Component | Released text model | Eight-layer comparison |
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

The eight-layer model reached the long-tier Basis Recall Easy threshold
in 550–650 updates. Phasor required 2,700–4,300 updates on the same
three seeds. Each cell below used the same training minibatches and the
original scheduled observations. The models ran on different hardware.

| Seed | First Inkling update | First Phasor update | Inkling primary | Inkling independent test |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 650 | 4,300 | 508/512 | 509/512 |
| 1 | 550 | 2,700 | 512/512 | 512/512 |
| 2 | 550 | 3,350 | 512/512 | 512/512 |

Inserting additional filler after the table keeps the original test
queries and answers fixed. The columns below give exact sequences out
of 512. Both models used the same rows at the 64-token gap. The
zero-gap fingerprints match the independent test.
[Gap invariant](../../src/Transformer/Basis/RecallParsed.lean)

| Seed | Inkling gap 0 | Gap 32 | Gap 64 | Gap 128 | Phasor gap 64 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 509 | 487 | 141 | 35 | 504 |
| 1 | 512 | 511 | 510 | 509 | 319 |
| 2 | 512 | 512 | 512 | 512 | 359 |

On short-tier Basis Depth Easy, the eight-layer checkpoints first passed
at updates 250, 500 and 500. The original length-128 test was regenerated
and its fingerprint verified before extending it to length 256. Phasor
and Inkling received exactly the same test rows at each length.

| Seed | Inkling primary | Independent test | Inkling length 128 | Inkling length 256 | Phasor length 256 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 512/512 | 509/512 | 491/512 | 252/512 | 428/512 |
| 1 | 512/512 | 512/512 | 512/512 | 478/512 | 341/512 |
| 2 | 512/512 | 512/512 | 512/512 | 464/512 | 496/512 |

All three eight-layer checkpoints passed Basis Depth Hard. Each scored
512/512 on the original length-128 test. At length 256 only seed 2
reached 99%, so its test length doubled again. It scored 481/512 at
length 512 with the same three passes and weights.

| Seed | First Inkling update | First Phasor update | Inkling primary /512 | Independent test /512 | Length 256 /512 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 750 | 1,000 | 511 | 511 | 496 |
| 1 | 1,250 | 2,250 | 510 | 511 | 502 |
| 2 | 1,000 | 1,000 | 512 | 512 | 512 |

On the identical length-256 tests, main Phasor scored 509, 512 and 507
out of 512 in seed order. At length 512 it scored 342, 512 and 293.

On seed 0, the eight-layer model reached 256/256 on both validation
and the independent test of branch amplitudes at update 256. Main
Phasor first passed validation at update 416 with 254/256. Both
architectures first passed linear transport at update 224. Inkling
scored 255/256 on validation and 256/256 on the independent test.
These canonical scores require generating the complete answer through EOS.

On long-tier Basis Parity 16, the eight-layer seed-0 checkpoint first
passed at update 2,500 with 156/157 novel sequences and 512/512 on the
independent test. Main Phasor reached 157/157 at update 1,250 and scored
512/512 on that test. Doubling the input to 32 bits gave the following
results on the same 256 distinct prompts. Each correct generation had
to emit both the parity label and EOS. The first Phasor checkpoint
scored 234/256. [Parity and EOS specification](../../src/Transformer/Basis/Parity.lean)

| Inkling passes at evaluation | Correct /256 | Gained vs. three passes | Lost vs. three passes |
| ---: | ---: | ---: | ---: |
| 3 (training depth) | 127 | 0 | 0 |
| 5 | 179 | 76 | 24 |
| 10 | 231 | 111 | 7 |
| 20 | 231 | 111 | 7 |
| 40 | 231 | 111 | 7 |

At every tested depth above three, the same checkpoint recovered the
one missed novel validation example and retained all 512 independent
test answers at the source length. On the doubled inputs, ten passes
improved accuracy by 104/256 and fell short of the 99% target.
At 20 bits, three passes scored 228/256. Ten and forty passes each scored
255/256, gaining 28 and losing one of the originally correct rows.

On seed 0 the first positive scheduled observation gave 52/4,705 on
modular addition at update 19, 79/4,705 on absolute difference at update
19, and 44/4,704 on decoded Fibonacci at update 10. All three are early
observations within 100,000-update budgets. Phasor's main comparator
independently replayed all 75 short and 30 long threshold crossings.
The two models used different compute backends. [Phasor cohort](../phasor_value_transport/README.md#canonical-gpt-stress-tiers)

The 24-layer topology reached the threshold earlier than the eight-layer
comparison on all six Boolean runs. Each checkpoint was separately reloaded.

| Boolean task | Seed | First 24-layer update | Primary | Independent test | Length 32 | Length 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shifted | 0 | 112 | 7/7 | 64/64 | 56/64 | 33/64 |
| Shifted | 1 | 80 | 7/7 | 64/64 | 17/64 | 3/64 |
| Shifted | 2 | 92 | 7/7 | 64/64 | 52/64 | 34/64 |
| Random | 0 | 92 | 8/8 | 58/64 | 39/64 | 28/64 |
| Random | 1 | 84 | 8/8 | 64/64 | 7/64 | 0/64 |
| Random | 2 | 88 | 8/8 | 52/64 | 42/64 | 0/64 |

This model used 21,481,088 parameters for Boolean shift. At 40 updates
it still scored 0/7 on each seed. On absolute difference the 81,116,992-
parameter model went from 78/4,705 at update 19 to 79/4,705 at update 76
on seed 0. On decoded Fibonacci the 81,137,344-parameter model went from
52/4,704 at update 10 to 53/4,704 at update 40 on seed 0. The original
100,000-update budgets are still open for these two cells.

We varied the evaluation depth at fixed weights after three-pass training.
For the 24-layer Boolean-shift checkpoints, one, two, three, five, ten,
twenty and forty passes each gave 7/7 on novel validation and 64/64 on
the independent test on seeds 0 and 1. Seed 2 scored 6/7 and 62/64 with
one pass, then 7/7 and 64/64 at every tested budget from two through forty.
On eight-layer Basis Recall Easy, one pass gave 485, 466 and 512 of 512
on novel validation and 498, 472 and 512 on the independent test.
Two passes recovered the three-pass scores on every seed. These counts
held at five, ten, twenty and forty passes. Lean gives a sufficient
condition for an entire hidden prefix to remain fixed.
[Shared-pass proof](../../src/Transformer/InklingLoop.lean)

The same depth comparison at a 128-token recall gap gave 28, 430 and
512 exact sequences out of 512 at one pass. Three passes gave 35, 509
and 512. Ten and forty passes gave the same three counts. Every budget
used the same weights and extended examples. Two identical totals allow
an equal number of gains and losses. Retaining every previously correct
answer requires zero losses on the matched rows.
[Paired exact-count identity](../../src/Transformer/InklingLoop.lean)

Pairing the 512 exact-sequence decisions against the three-pass run
revealed an exchange on seed 0. At forty passes the model recovered one
previously missed row and lost one previously correct row. Its score
stayed at 35/512. Seeds 1 and 2 kept every previously correct row. The
scorer records each outcome after dense prediction or after free generation
and the original termination check, and verifies its sum against the
original aggregate score.

| Seed | Three passes correct | One pass gained / lost | Forty passes gained / lost |
| ---: | ---: | ---: | ---: |
| 0 | 35 | 9 / 16 | 1 / 1 |
| 1 | 509 | 0 / 79 | 0 / 0 |
| 2 | 512 | 0 / 0 | 0 / 0 |

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
INKLING_ROLLING=1 INKLING_JOBS=6 INKLING_THREADS=2 ./cohort.sh run runs/canonical
./cohort.sh report runs/canonical
```

Build the 24-layer topology in its own target directory and give its
checkpoints a separate output directory.

```sh
export CARGO_TARGET_DIR="$PWD/target/release-topology"
cargo +stable test --release --locked --offline --features release-topology -- --test-threads=1
cargo +stable build --release --locked --offline --features release-topology
export INKLING_BIN="$CARGO_TARGET_DIR/release/inkling-stress"
INKLING_ROLLING=1 INKLING_JOBS=2 INKLING_THREADS=2 ./cohort.sh run runs/release-topology
./cohort.sh report runs/release-topology
```

The optional final argument limits the cohort to a multiple of the original
observation interval. Omitting it advances the saved states through the full
original budgets or first threshold crossing.

`INKLING_ROLLING=1` keeps the latest weights, both AdamW moments, and every
scheduled observation for each cell. It replaces its preceding rolling file
after the new weights and their digest have been published. Regular snapshots
use the `at-STEP.safetensors` filename and can be kept for depth sweeps.
On Boolean shift, rolling through updates 4, 8 and 12 produced the same
checkpoint bytes and observations as training directly to update 12.

A single cell can also be advanced and replayed directly.

```sh
target/release/inkling-stress run suite-boolean-and-shift 0 runs/canonical 4
target/release/inkling-stress run suite-boolean-and-shift 0 runs/canonical 320
target/release/inkling-stress replay suite-boolean-and-shift 0 runs/canonical
target/release/inkling-stress depth suite-boolean-and-shift 0 runs/canonical 1,2,3,5,10,20,40
target/release/inkling-stress extend basis-depth-easy 0 runs/canonical depth-length 128,256
for passes in 3 5 10 20 40; do
    target/release/inkling-stress extend basis-parity16 0 runs/canonical parity-length 32 "$passes"
done
target/release/inkling-stress extend basis-recall-easy 0 runs/canonical recall-gap 0,32,64,128
target/release/inkling-stress extend basis-recall-easy 0 runs/canonical recall-gap 128 40
```

Tests cover open and closed gates across three passes, causality, fixed-input
label invariance, finite differences through shared weights, scalar AdamW
agreement, and the original field and free-generation scorer. On Boolean
shift, saving at update 4 and resuming to update 8 reproduced every byte
of the uninterrupted optimizer and model checkpoint and all observations.
