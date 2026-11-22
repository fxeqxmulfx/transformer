# One Phasor architecture for NV11 and the GPT stress suite

One prefix-and-successor Phasor passed every original short and long GPT
stress task and four additional catalog tasks in seeds 0, 1 and 2. Each
task trained its own weights using the same architecture and original AdamW.
Every NV11 seed classified all 9,409 pairs correctly at update 1,250.

## Selected prefix and successor model

The default Phasor applies one causal update at every token. Its prefix state
is the XOR of preceding token IDs. Its key/value memory records the latest
completed successor of each token. The linear-sum table reads the prefix
state plus the current token modulo the vocabulary size, and the original
right-operand table reads the retrieved successor. An unwritten key reads
itself. Both reads enter the same normalized hidden residual with one learned
gain, initialized to 0.5. The parameter count and the original AdamW are
the same as for the pair-address incumbent.

| Original tier | Passing runs | Paired epoch ratio to incumbent | Faster / tied / slower | First 99% updates |
| --- | ---: | ---: | ---: | ---: |
| Short | 75/75 | 0.6090 | 66 / 5 / 4 | 36–1,250 |
| Long | 30/30 | 0.1657 | 25 / 1 / 4 | 228–9,980 |

The ratio is the geometric mean across matched task and seed pairs. Every
saved score replayed with the public scorer. Easy recall took 400 updates
in every seed, down from 900, 3,500 and 7,500. Fibonacci recurrence took
9,980, 7,480 and 9,560, up from 7,550, 5,560 and 8,090. The five other
regressions occurred on associative divisor seed 1, easy depth seed 0,
hard depth seed 2 and branch amplitudes seeds 1 and 2.

The expanded benchmark has 29 short tasks and ten long tasks. All 87 short
and 30 long task/seed runs reached 99% and replayed independently.

Fresh key/value-binding runs on the GTX 1050 took 5.41, 5.32 and 3.95 s
for the incumbent, and 3.51, 3.50 and 3.50 s for the selected model.
The models reached 99% at 704 / 704 / 512 and 448 / 448 / 448 updates,
respectively. Each timing included setup, training and scoring on an
otherwise idle compute device. The arm run first alternated by seed.
Every timed run reproduced its earlier checkpoint byte for byte. To repeat
the timing, run `cell gpt-probe-key-value-binding SEED FRESH_DIR` with
each build and `PHASOR_EVAL_BATCH_CAP=16`.

For the epoch comparison, first generate the incumbent's paired report with
the baseline reproduction commands below.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.SuccessorValues
cd experiments/phasor_value_transport
cargo +1.99.1 build --release --offline --locked --features fp32-affine-reference
bin=./target/release/phasor-value-transport
"$bin" shared-check
PHASOR_EVAL_BATCH_CAP=16 ./run_stress.sh cell runs/prefix-successor-half extended
./run_stress.sh replay runs/prefix-successor-half extended
"$bin" extended-report runs/prefix-successor-half > runs/prefix-successor-half/report.json
jq -s -f compare.jq runs/shared-quarter/paired-report.json runs/prefix-successor-half/report.json
```

## Tied successor displacement

Can a successor read carry the identity of a retrieved token before the
network has learned that token's value table? For every position, let `b`
be the current token and `r` its latest completed successor, falling back
to `b`. Add `s × (E[r] − E[b])` inside the residual RMS using the trained
tied embedding `E`. The two trial strengths are 1 and 0.25. Each applies
the same rule at all lengths with the incumbent's parameters and AdamW.

Lean proves that moving the two coefficients of the embedding read implements
this hidden update exactly. Before normalization the retrieved-versus-key
score margin increases by `s × ‖E[r] − E[b]‖²`. Whenever the read returns its
key, including the first two positions of every prompt, all logits agree with
the incumbent for every parameter setting. These identities do not imply a
held-out accuracy gain. [Forward map and proofs](../../src/Transformer/Phasor/SuccessorDisplacement.lean)

The seed-0 screen uses the incumbent's first 99% crossing as the cutoff for
easy recall, key/value binding, blocks, Dyck and 20-bit parity. The two fresh
recall probes use 1,000 updates for MQAR and 750 for two-hop lookup. A
candidate that takes longer on a retention screen is set aside. A surviving
candidate needs all existing task/seed pairs, a per-tier geometric mean
epoch ratio at most one, and a verified gain to become the default.

## Pair-address baseline

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

The tables start at zero and the gain at 0.25 for every task. This adds `17 × V × width + 1`
real parameters, all in Phasor's ordinary AdamW coordinate group. The
original encoder, tied embeddings, cross-entropy, clipping, parameter
groups and WSD schedule train together. Phasor uses a peak learning rate
of 0.001 and a warmup lasting 2% of each task's update budget. The GPT suite
specifies the width, head count, depth, minibatch, budget and observation
cadence for each task.

Lean gives the exact decision margin at every position. For an arbitrary
fixed-depth backbone with orthonormal embeddings it proves that one gain
and one right-operand table can distinguish every second operand on the
entire zero row. The width-128 example covers all 97 NV11 residues. Lean also
bounds the initial displacement of the contextual feature by `sqrt(width)/4`.
The change in any two-class margin is bounded by that quantity times the
distance between the two tied embeddings.
[Shared forward map and proofs](../../src/Transformer/Phasor/SharedValues.lean)

### Causal prefix trials

One existing power-sum table can read the entire preceding prefix. For each
sequence start a state at zero. Reading a token updates the state by bitwise
XOR or ordinary addition on its token ID. At position `i`, the selected
table reads `(state^u + token[i]^v) mod V`. At the first position its left
operand is the first token. Every position, vocabulary and task uses the
same fold and exponents. The other sixteen tables, contextual Phasor
feature, gain, readout and AdamW retain their definitions.

Lean proves the scan composes over arbitrary consecutive chunks. In the XOR
scan each bit of the state gives the parity of that bit's occurrences in the
entire preceding prefix. The tokenizer, vocabulary, and sequence length are
arbitrary. The trained table sees the result after the specified power and
vocabulary reduction, which can map different states to the same address.
Lean also proves exact logit agreement with the pair residual at the first
two positions for every setting of the weights and all nine power-sum slots.
[Prefix representation and logit theorems](../../src/Transformer/Phasor/PrefixValues.lean)

The successor-read trial combines the XOR-prefix sum with a causal
key/value memory. After seeing a transition `x → y`, the memory associates
`x` with `y`. At each position the current token queries the most recent
matching transition whose two endpoints precede that position. An unwritten
key reads itself. This token addresses the existing right-operand table.
Lean proves exact latest-write retrieval at any distance, even after
overwrites, and proves that the lookup commutes with every injective token
renaming. These are properties of the internal address. Learned logits and
held-out accuracy are measured separately.
[Successor read and lookup theorems](../../src/Transformer/Phasor/SuccessorValues.lean)

The half-gain trial initializes this same learned residual at 0.5 throughout
the model. Lean bounds the change in any logit at fixed weights by
`|g − h| × ‖embedding[class]‖ × sqrt(width)` when the gain changes from
`h` to `g`, at every position. Build this trial with `prefix-successor-half`.

For promotion we compare all three seeds with the incumbent. Every original
pass must survive. We compare the geometric mean of paired training epochs
to the first scheduled 99% crossing separately for short and long, retaining
the individual crossings so regressions are visible. An improvement must
retain or improve both tier averages. Catalog additions are measured separately.

Use Cargo features to compile each trial as one fixed architecture. The
`prefix-xor` and `prefix-sum` trials change the linear sum table.
`prefix-xor-cubic` uses the sum of cubes and `prefix-successor` combines
the linear XOR-prefix table with the successor read. Use a fresh result
directory for each compiled architecture. The runner verifies the model
identity before accepting a saved result.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.SuccessorValues
cd experiments/phasor_value_transport
cargo +1.99.1 build --offline --locked --release --features fp32-affine-reference,prefix-xor
bin=./target/release/phasor-value-transport
"$bin" shared-check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen basis-parity24 1 runs/prefix-xor 2000
PHASOR_EVAL_BATCH_CAP=16 "$bin" cell suite-blocks 1 runs/prefix-xor
"$bin" probes suite-blocks 1 runs/prefix-xor
```

A screen stops at its stated scheduled observation. A successful screen
saves the usual first-crossing checkpoint. An unfinished screen keeps its
observation log and a full cell restarts it from the original seed. The
catalog adapter checks the original plan digest, uses native source splits,
and evaluates novel validation prompts where supplied. `probes` replays
the selected checkpoint and scores the additional source test splits.
The tier admission rule is 99% held-out exact sequences in each of seeds
0, 1 and 2. The slowest fresh run, including setup and scoring, places a new
case in short at up to 60 seconds or long at up to 300 seconds on the
measuring GPU. Historical membership of the original 35 stays fixed.

### Additions from the 165-case GPT catalog

The quarter-gain successor model qualified three new short tasks on the
GTX 1050 with FP32 affines. Each cell below used a fresh model, the source
training split and an independent checkpoint replay. The synthetic catalog
used a 95% target for blocks and Dyck. Phasor's admission threshold is 99%.

| New case | First 99% updates, seeds 0 / 1 / 2 | Held-out exact counts | Slowest fresh cell |
| --- | --- | --- | ---: |
| Three alternating blocks, with neutral symbols | 500 / 750 / 500 | 28/28 in each seed | 6.86 s |
| One-type Dyck bracket-prefix labeling | 250 / 250 / 250 | 32/32 in each seed | 2.45 s |
| Sixth power of a modular sum | 1,250 / 1,250 / 1,250 | 4,698 / 4,668 / 4,680 of 4,705 | 10.55 s |

On the separate ordinary test set, blocks scored 63, 62 and 57 of 64
sequences. On length 64 it scored 19, 15 and 17 of 64. Dyck scored 64/64
on the ordinary test in all seeds and 19, 18 and 18 of 64 at length 64.
The admissions describe the declared held-out selection split.

The selected half-gain model added `language-l1`, the one-block language
with no neutral tokens. It trained on lengths 32–64. At update 250 each
seed classified all 512 length-128 validation sequences and all 512 separate
length-128 test sequences correctly. The slowest fresh cell took 40.64 s,
qualifying this fourth addition for short.

The selected binary lists the four additions and reports the three seeds for each.

```sh
"$bin" extensions
"$bin" catalog-report runs/prefix-successor-half suite-blocks suite-dyck arithmetic-sum-pow6 language-l1
```

The runner lists the frozen original tiers and the extension tiers
separately. `probes` on a Basis parity checkpoint also runs the original
scratchpad-free catalog's 32- and 64-bit test prompts through the fitted
weights. The command expands the inference buffer for these longer inputs.
[Frozen extension membership](extensions.tsv)

### NV11 and length transfer with the half-gain successor

The single half-gain successor model reached 99% held-out on NV11 at 750
updates in every seed. At update 1,250 all three checkpoints classified
every one of the 9,409 input pairs correctly. An independent forward pass
with masked targets found minimum held-out margins of 1.155, 0.356 and
0.846 in seed order. All three remained exact when replayed at update 3,000.

| Model at its first 99% Basis checkpoint | 16-, 20-, 24-bit training tasks | New 32-bit prompts | New 64-bit prompts |
| --- | --- | --- | --- |
| Pair-address incumbent | 5,750–39,250 updates | 21–36/64 exact | 29–39/64 exact |
| Half-gain successor | 250 updates in all nine runs | 64/64 in every run | 64/64 in every run |

All nine saved models in each row generated parity and EOS on the same
64 held-out catalog prompts per length. Training ended at 16, 20 or 24 bits
according to the Basis task. These two transfer sets never entered training
or stopping. On the first three additions, sixth-power modular sum took
250 updates in all seeds, Dyck took 250
in all seeds, and blocks took 750, 1,000 and 1,250.

The linear prefix-sum pilot scored 214/306 on 24-bit parity at 2,000
updates in seed 1. The XOR-only and cubic-slot XOR pilots needed 640 and
736 updates on branch amplitudes seed 0, against 416 for the incumbent.
These results informed the successor trial. Additional quarter-gain
screens on small-sample catalog tasks reached 1/32 on associative recall
at 4,000 updates with the successor read, and 30/136 on division modulo 17
at 3,000 with the XOR-only model. Those cases remain outside the extension.

### Choice of initial gain

The first uniform architecture used an initial gain of one. Hard depth seed 0
classified all 512 length-128 sequences correctly at update 1,000. Seed 1
peaked at 490/512 in its first 10,000 updates. Reducing the initial gain to
0.25 gave that seed 512/512 at update 1,250. All three NV11 pilots also
crossed 99% with the smaller gain. These published splits informed the choice
of initialization. The reported quarter-gain suite starts each task and seed
from fresh weights.

## NV11 results for the pair-address baseline

Each model reached the original 99% held-out threshold at update 1,500.
Continuing the same AdamW trajectory produced one checkpoint per seed that
classified all 4,704 training pairs and all 4,705 held-out pairs correctly.
An exhaustive forward replay with masked targets measured the smallest
target logit minus its highest-scoring rival on each split.

| Seed | Held-out correct at 1,500 | First observed all 9,409 correct | Minimum training margin | Minimum held-out margin |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4,700/4,705 | 2,500 updates | 12.3811 | 3.6761 |
| 1 | 4,703/4,705 | 2,000 updates | 13.5084 | 1.5141 |
| 2 | 4,701/4,705 | 2,000 updates | 10.9942 | 1.9265 |

At update 2,000 seed 0 still missed the held-out pair `(23, 13)`.
Its target was 50 and class 63 led by 1.5026 logits. The first exact
checkpoint for this seed assigns the correct class to that pair as well.
At update 5,000 all three checkpoints again classified the complete
modulus-97 table correctly, with minimum held-out margins of 10.9882,
9.1783 and 11.6655 in seed order.

## GPT stress results for the pair-address baseline

One architecture, `phasor-causal-value-residual-v2`, reached the original
99% held-out exact-sequence threshold on every task in all three seeds.
The table gives the range of first scheduled crossings across each tier.

| Tier | Tasks | Passing task and seed runs | Updates to first crossing |
| --- | ---: | ---: | ---: |
| Short | 25 | 75/75 | 56–1,292 |
| Long | 10 | 30/30 | 570–39,250 |

Hard depth trains on lengths 32–64. At length 128 the three models got
508, 512 and 512 of 512 sequences correct at updates 750, 1,250 and 1,000.
The latest crossing was 24-bit parity seed 1, with 304/306 exact generated
answers at update 39,250. Recall seed 2 reached 507/512 at update 7,500.
The saved weights reproduce the exact held-out counts for all 105 GPT runs
and the three separate NV11 runs.

At the GPT stopping checkpoints, each addition seed classified 50/51
held-out pairs with first operand zero correctly. Each absolute-difference
seed got 51/51. Clearing the 17 learned tables gave addition 50/51 in all
three seeds and absolute difference 50/51, 48/51 and 50/51.

Every measured run used scalar FP32 affines on a GeForce GTX 1050. The
upstream mixed-precision configuration received a separate compilation check.

## Reproducing the pair-address baseline

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
cargo +1.99.1 build --offline --locked --release --features fp32-affine-reference,pair-v2
bin=./target/release/phasor-value-transport
"$bin" shared-check
PHASOR_EVAL_BATCH_CAP=16 ./run_stress.sh cell runs/shared-quarter
./run_stress.sh replay runs/shared-quarter
"$bin" report runs/shared-quarter > runs/shared-quarter/paired-report.json
"$bin" predict runs/shared-quarter/arithmetic-cubic-product/seed0/weights.f32 97 12 34
for seed in 0 1 2
do
    "$bin" shared-values "runs/shared-quarter-5000/seed$seed" "$seed" 5000
    "$bin" nv11-score "runs/shared-quarter-5000/seed$seed/shared-values-first-exact.f32"
done
```

For one case, use `cell CASE SEED RUN_DIR` and `replay CASE SEED RUN_DIR`.
Concurrent requests for the same cell wait for its training process and
verify the saved checkpoint when it finishes.
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

The optional final argument of `shared-values` sets the observation window
within the original 100,000-step schedule. It saves the first checkpoint
that classifies the entire NV11 table correctly. `nv11-score` takes a
shared-value checkpoint, masks all training targets in the forward pass
and measures the minimum correct-class margin on each split.

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
