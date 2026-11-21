# Transported embedding values for NV11

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

## Findings

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

## Protocol

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

## Reproduction

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

## Operand-sensitive pair values

The transported head uses the first operand and the product. When the first
operand is zero, the head produces the same scores for every second operand.
The canonical addition and absolute-difference validation splits each contain
51 different answers on that row. A model with that collision can get at
most 4,655 of the 4,705 examples right. The 99% threshold requires 4,658.

The new head adds learned width-128 vectors to the transported vector before
the final RMS and tied projection. One table reads the second operand
directly. The other tables use the modular difference, ordinary absolute
distance, field quotient (zero for denominator zero), the nine sums
`a^i + b^j` for `1 ≤ i,j ≤ 3`, and the four products `a^i b^j` for
`1 ≤ i,j ≤ 2`. Every pair task receives all 17 tables. A table maps an
address to a trainable hidden vector. Some addresses coincide with individual
benchmark functions. The NV11 target `a³ + ab` is absent from the addresses.
The cubic address distribution and tied embeddings are still learned.

The residual tables start at zero, recovering the previous transported head
exactly at initialization. Lean proves that a winner survives precisely when
the residual's adverse projected gap is smaller than the transport advantage.
It also constructs one width-128 table that separates all 97 possible second
operands on the zero row, with the transport branch active. The full-model
proof obtains the mass bounds from the actual first-position softmax at every
depth. The field layer
applies to position one (zero-based) in a model configured for exactly two
tokens and a prime vocabulary. Every other position or configuration uses the
ordinary causal Phasor. [Forward map and proofs](../../src/Transformer/Phasor/PairValues.lean)

### Training and replay

The pair layer adds 211,072 parameters at modulus 97 and 419,968 at modulus
193. They join the existing AdamW group for ordinary real coordinates before
the LeRoPE frequency group. Initialization, minibatch order, warmup, learning
rates, weight decay, gradient clipping, cross-entropy and the tied projection
use the same implementations as Phasor. Training and greedy generation both
call the modified engine's ordinary forward method. Inactive configurations
allocate zero additional parameters.

The stress runner reconstructs the fingerprinted data and uses the pinned
DiffGPT complete-split scorer, including autoregressive decoding on generation
tasks. Each case and model seed starts from fresh weights. The suite criterion
is the first scheduled observation with at least 99% exact sequences on the
primary held-out split. NV11 is a separate half-table experiment. [Stress
protocol and task definitions](../convex_stress/suite.json)

From the repository root with the pinned source checkouts described above:

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.PairValues
cd experiments/phasor_value_transport
cargo +1.99.1 build --offline --locked --release --features fp32-affine-reference
bin=./target/release/phasor-value-transport
"$bin" pair-check
./run_stress.sh cell runs/pair-values
./run_stress.sh replay runs/pair-values
"$bin" predict runs/pair-values/arithmetic-cubic-product/seed0/weights.f32 97 12 34
for seed in 0 1 2
do
    "$bin" pair-values "runs/pair-retention/seed$seed" "$seed"
done
```

For a single task, use `cell CASE SEED RUN_DIR` and `replay CASE SEED RUN_DIR`.
Replay checks the weights digest, precision, suite identity and exact held-out
count. It also measures the zero row with the trained tables and with the
tables cleared. Prediction uses the two operands and the field size.
The `pair-check` command exercises both supervised positions, repeated
addresses and a one-token prefix. It compares nonzero table values with a
scalar FP64 forward calculation, checks all 17 table gradients and the tied
embedding by finite differences, and executes the Lean zero-row witness.

## One architecture at every position

The shared-value Phasor applies one hidden update to arithmetic pairs, long
sequences, prime and composite vocabularies. At position `i`, let `b` be the
current token and `a` its predecessor within that sequence. At the first
position use the current token for both. The preceding Phasor feature scores
the tied vocabulary embeddings. Attention reads the learned values translated
by `ab` modulo the vocabulary size. The 17 learned tables above add their
vectors to that read. With composite vocabulary size `V` the fourth table
uses the total map `a b^(V-2) mod V`, with zero for a zero denominator.

The normalized result is multiplied by one learned scalar and added to the
current contextual Phasor feature. One tied projection of that combined
hidden vector gives the prediction. The same computation runs at every
training and generation position. All configurations initialize the gain at
one and the tables at zero. Each task trains its own weights with the original
AdamW, as specified by the GPT stress suite.

Lean gives the exact decision margin and proves that, at any fixed depth,
one gain and one right-operand table separate the whole zero row
simultaneously at width 128 and modulus 97. [Shared forward map and proof](../../src/Transformer/Phasor/SharedValues.lean)
