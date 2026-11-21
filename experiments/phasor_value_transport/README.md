# One Phasor architecture for NV11 and the GPT stress suite

Can a single change to Phasor's hidden state learn NV11 and retain both
tiers of the GPT stress suite with the original AdamW? Every task uses the
same causal value residual, including sequence tasks with composite
vocabularies. Model seeds and per-task training follow the original suite.

## Architecture

Let `b` be the current token and `a` its predecessor in the same sequence.
The first token serves as its own predecessor. At each position the preceding
contextual Phasor feature scores the tied embeddings. Its softmax reads
`E[(s + ab) mod V]` at address `s`, where `V` is the vocabulary size.

Seventeen learned tables add hidden vectors to that read. Their addresses
are the current token, modular difference, ordinary absolute distance,
`a b^(V-2) mod V` (zero on a zero denominator), the nine sums `a^i + b^j`
for `1 ≤ i,j ≤ 3`, and the four products `a^i b^j` for `1 ≤ i,j ≤ 2`.
The exponentiation map gives field division for prime vocabularies. The sums
and products use the vocabulary modulus in every configuration. Some of
these addresses coincide with individual benchmark functions. NV11's
`a³ + ab` is absent from the address list.

Phasor normalizes the resulting vector, multiplies it by one trainable real
gain and adds it to its current contextual feature. One tied projection of
that combined hidden state produces the output logits. Context length,
vocabulary primality and task name play no role in selecting a computation.
The first position, later positions, training and autoregressive generation
all use this update.

The tables start at zero and the gain at one. This adds `17 × V × width + 1`
real parameters, all in Phasor's ordinary AdamW coordinate group. The
original encoder, tied embeddings, cross-entropy, clipping, parameter
groups and WSD schedule train together. Only the shape and training budget
vary as specified in the benchmark.

Lean gives the exact decision margin at every position. For an arbitrary
fixed-depth backbone with orthonormal embeddings it proves that one gain
and one right-operand table can distinguish every second operand on the
entire zero row. The width-128 example covers all 97 NV11 residues.
[Shared forward map and proofs](../../src/Transformer/Phasor/SharedValues.lean)

## Training and replay

Use DiffGPT at `5a37c515639f23037d191b0659a5a0853984328d` and Kernite at
`a1b6bc2c956e1a5f35960e26b209e1b6c213096c` under
`.lake/reproduction/{gpt,kgen}`. The Rust build checks their revisions and
the relevant source trees. [Checkout instructions](../rerope/README.md#build-and-commands)

The native runner reconstructs the fingerprinted splits and uses the complete
DiffGPT scorer, including greedy generation from the model's own tokens.
Its criterion is the first scheduled observation with at least 99% exact
sequences on the primary held-out split. Each task and seed starts from fresh
weights. NV11 uses its original 4,704/4,705 half-table split and separate
250-step observation cadence. [Original 35-task protocol](../convex_stress/suite.json)

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.SharedValues
cd experiments/phasor_value_transport
cargo +1.99.1 build --offline --locked --release --features fp32-affine-reference
bin=./target/release/phasor-value-transport
"$bin" shared-check
./run_stress.sh cell runs/shared-values
./run_stress.sh replay runs/shared-values
"$bin" report runs/shared-values
"$bin" predict runs/shared-values/arithmetic-cubic-product/seed0/weights.f32 97 12 34
for seed in 0 1 2
do
    "$bin" shared-values "runs/shared-retention/seed$seed" "$seed"
done
```

For one case, use `cell CASE SEED RUN_DIR` and `replay CASE SEED RUN_DIR`.
On a memory-constrained GPU, `PHASOR_EVAL_BATCH_CAP=16` limits the scorer's
chunk size down to the training batch size. Training still consumes the
original minibatches, and each evaluation visits the entire split.
Replay verifies the architecture identifier, weights digest, precision,
suite identity and exact held-out count. The report verifies the full observation
history, update cadence and examples consumed before counting a task and seed.
It lists budget misses and pending runs separately. For two-token validation examples
with first operand zero it also scores the fitted weights with the value
tables cleared. The contextual feature and transported read remain active
in that ablation. The two-token prediction command takes a weights file,
vocabulary size and two operands.

The `shared-check` command compares logits with an independent scalar FP64
forward evaluation. It checks the tied embedding, gain and all 17 table
gradients by finite differences at lengths 1, 2 and 4, and executes the
97-way zero-row witness. The FP32 affine backend runs on the local Pascal
GPU. The default build uses upstream FP16-forward/TF32-backward matrix
kernels on supported GPUs.

## Earlier transport-only experiment

Can one change to the hidden answer state let Phasor learn
`a³ + a b (mod 97)` from random initialization with its ordinary AdamW?

The first causal Phasor feature scores the tied embeddings. At the answer
position, attention mass at address `s` reads `E[(s + a b) mod 97]`.
The model applies its usual RMS and tied projection to the weighted **hidden
vector**. The table is learned end to end. The address layer has zero new
parameters. Multiplication and modular translation are supplied by the
architecture. The cube enters only in the dataset and the diagnostic that
checks which address won after inference.

The Lean result gives both an exact decision rule for orthonormal embeddings
and a sufficient Gram margin for general embeddings. Its width-128 witness
fits the 97 address codes. [Forward map and proofs](../../src/Transformer/Phasor/TransportedValues.lean)

### Transport-only findings

The transported model got every training and held-out answer right by update
250, the first observation after initialization. It stayed perfect at every
observation through update 2,000. This held for the original seed-1 pilot and
two subsequent paired replications with fresh initial weights and batch orders.
All three used the same data split and the same GTX 1050 FP32 backend.

| Held-out answers correct out of 4,705 | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: |
| Original Phasor at 250 updates | 48 | 48 | 49 |
| Transport at 250 updates | 4,705 | 4,705 | 4,705 |
| Original Phasor at 2,000 updates | 95 | 87 | 90 |
| Transport at 2,000 updates | 4,705 | 4,705 | 4,705 |
| Trained transport with offset zero at 2,000 updates | 95 | 95 | 95 |

At update 250 the first token selected the cubic address for all 97 operands
in every seed. Disabling the offset at inference then predicts the row's cube
for every second operand. Exactly 95 held-out pairs have zero modular product.
The ordinary control fit 4,699, 4,698 and 4,694 of the 4,704 training pairs
by update 2,000. The measured gain concerns new pairs of previously observed
operands at modulus 97.

We also replayed the stored learned embeddings and evaluated the sufficient
Gram bound against every rival. At step 250 it was positive against every
rival on **0/4,705** held-out pairs in each seed, despite perfect accuracy.
At step 2,000 it was positive on **4,705/4,705** in every seed. The smallest
unnormalized bounds across all held-out pairs and rival classes were 0.1490,
3.1369 and 0.1409 respectively. These are FP64 evaluations of the proved
inequality using the saved FP32 weights and first-token scores.

### Transport-only protocol

Use the fingerprinted 4,704/4,705 split, a one-layer width-128 Phasor with
four heads, and fresh seed-1 upstream weights shared by the two arms. Score
all 97 classes with ordinary cross-entropy on the second token. Use each
shuffled epoch, including its final short batch, and the original Phasor
independent-coordinate AdamW kernels and 100,000-step WSD schedule. The
batch size is 256, peak learning rate 0.001, warmup 2,000 updates, base
weight decay 0.1 and gradient-norm clip 1. The frequency group has its own
clip and zero decay. Observe both arms every 250 steps through update 2,000.
The task criterion is 99% held-out accuracy. At each observation also
measure the trained candidate with the product offset set to zero and
count how many of the 97 first operands select their cubic address.

| Arm | Answer-position hidden value |
| --- | --- |
| Phasor control | Original final stream mean |
| Transport | Softmax-weighted sum of translated embedding values |
| Zero offset, evaluation only | Same trained weights, value at address `s` is `E[s]` |

The registry supplies the split and model shape. The pinned native Phasor
trainer supplies the optimizer, initialization and WSD schedule. The pilot
uses the first 2,000 updates of that schedule. The standard build uses the
upstream FP16-forward/TF32-backward matrix kernels on Ampere or newer.
The `fp32-affine-reference` feature selects the upstream scalar FP32 GEMMs
for a matched comparison on Pascal. All arms use the same precision.

### Reproducing the transport-only runs

Use the DiffGPT checkout at `5a37c515639f23037d191b0659a5a0853984328d` and
Kernite at `a1b6bc2c956e1a5f35960e26b209e1b6c213096c` in the repository's
`.lake/reproduction/{gpt,kgen}`. The existing ReRoPE setup uses the same
checkouts. The Rust build verifies the revisions and reads the optimizer,
optimizer kernels, schedule, initialization, Phasor blocks and loss from
those sources. [Checkout instructions](../rerope/README.md#build-and-commands)

From the repository root, check Lean before running the experiment.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TransportedValues
cd experiments/phasor_value_transport
cargo +1.99.1 build --locked --release --features fp32-affine-reference
bin=./target/release/phasor-value-transport
"$bin" check
for seed in 1 2 3
do
    "$bin" transport "runs/pascal/seed$seed" "$seed"
    "$bin" phasor "runs/pascal/seed$seed" "$seed"
done
"$bin" inspect runs/pascal/seed1/transport-250.f32
"$bin" inspect runs/pascal/seed1/transport-2000.f32
"$bin" predict runs/pascal/seed1/transport-2000.f32 12 34
```

The two arm logs can share an output directory. Use a fresh directory for
a repeat. The candidate saves the first post-initialization checkpoint and
the final weights. `predict` takes two operands without a target.
`check` compares CUDA logits with a scalar FP64 forward and checks the
complete tied-embedding gradient by directional finite differences with
both offsets. `inspect` replays the checkpoint on the held-out split and
evaluates the Lean sufficient bound numerically in FP64 using the learned
embedding Gram matrix and the first-token attention scores.


## Earlier two-token residual

The initial operand-sensitive implementation applied its additional tables
only to a configured two-token prime vocabulary. Its arithmetic runs and
proof at commits `4595508c` and `8a16ce76` document that intermediate
design. The shared-value cohort uses fresh weights and one positional rule
on all 35 tasks. [Two-token decision margin and zero-row proof](../../src/Transformer/Phasor/PairValues.lean)
