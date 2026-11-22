# One Phasor architecture for NV11 and the GPT stress suite

## Learned stream readout

The inner model takes a sequence of embedding vectors. For a tokenized
benchmark, an input adapter looks up the vectors. Phasor's causal backbone
produces four hidden streams at each position. Each hidden coordinate learns four weights for reading those
streams before the original RMS normalization and tied output projection.
The weights start at one quarter, with zero trainable offsets. At width
128 the readout adds 512 parameters shared across all positions. Token
arithmetic, occurrence counts and symbolic lookup tables are excluded.

Lean proves that permuting the vocabulary and its embedding rows permutes
every output logit at every depth. It also gives the exact gradient of the
readout and shows that opposite stream contents, which cancel in the
original mean, can become visible to the classifier.
[Learned stream readout](../../src/Transformer/Phasor/LearnedStreamRead.lean)

The embedding-input formalization gives the backbone a type with no
vocabulary or target space. For arbitrary real input vectors it proves
that states at each position depend only on the vector prefix. Lookup
followed by this backbone and the tied classifier recovers every logit
of the token-level forward, including for distinct IDs that alias the
same vector. Training targets are arguments of the outer loss.
[Embedding-input interface](../../src/Transformer/Phasor/EmbeddingInput.lean)

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.LearnedStreamRead
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference
export PHASOR_EVAL_BATCH_CAP=16
bin=./target/release/phasor-value-transport
"$bin" check
"$bin" cell suite-boolean-and-shift 0 runs/learned-streams
"$bin" probes suite-boolean-and-shift 0 runs/learned-streams
jq -nc '[range(3) as $i | [range(64) as $j | (($i + $j) / 64)]]' > runs/learned-streams/vectors.json
"$bin" embedding-forward suite-boolean-and-shift 0 runs/learned-streams runs/learned-streams/vectors.json
"$bin" nv11-train runs/learned-nv11/seed0 0 3000
"$bin" nv11-score runs/learned-nv11/seed0/nv11-train-3000.f32
```

For the original fixed-mean Phasor control, add `original-stream-mean` to
the Cargo features and use a separate run directory. Both executables
use the same initialization for the backbone, per-task AdamW schedule,
training split and checkpoint selection rule.

`vectors.json` is a JSON array with one finite width-dimensional vector per
position. The command returns the hidden vectors and the tied classifier's
logits for each position. The case, seed and run directory identify the
trained checkpoint. The example supplies three width-64 vectors.

Changing targets altered the loss and left all 370 test logits bit-identical.
Two token IDs assigned the same vector also gave identical logits. On a
trained depth-one C-RASP checkpoint, eight vectors outside the embedding
table produced the same first three hidden rows and logits as the three-row
input alone. The training path reproduced the original checkpoint and
observation bytes on shifted AND, depth-one C-RASP and modular addition.

The FP32 kernels agreed with an independent FP64 calculation on 24,304
forward values and gradient components to within `2.3e-7` absolute error.
All 18 end-to-end finite differences passed. At widths 64, 128 and 256,
all 1,332 logits matched the original mean bit for bit at initialization
and permuted bit for bit under a shuffled vocabulary with nonzero offsets.

On depth-one C-RASP, seed 2 reached 32/32 validation at update 250 with
both readouts. Independent reloads of both scored 64/64 on the source
test, 56/64 at length 32 and 41/64 at length 64. On NV11 the learned
readout reached 4,696/4,704 training pairs but only 86/4,705 held-out
pairs at update 3,000 in seed 0.

### Long-tier addition from the embedding-input model

All three independently trained stream-read checkpoints solved the GPT
language-L2 source validation at length 128. Each checkpoint also scored
512/512 on both the original-length test and the independent length-128
test after reload. Training used lengths 32–64. The slowest training run
took 66.44 seconds on the GTX 1050, which places L2 in the long tier.
The original fixed-mean control passed in the same numbers of updates.

| Seed | First 512/512 validation update | All twelve-letter words, out of 4,096 | Eight letters repeated 32 times, out of 256 |
| ---: | ---: | ---: | ---: |
| 0 | 250 | 3,840 | 255 |
| 1 | 500 | 4,096 | 239 |
| 2 | 250 | 3,838 | 222 |

On the stretched inputs the mean control scored 255 / 229 / 222.
Use the learned-stream binary above to reproduce the first crossings
and the separate whole-sequence transfer scores.

```sh
for seed in 0 1 2; do
    PHASOR_EVAL_BATCH_CAP=16 "$bin" cell language-l2 "$seed" runs/learned-streams
    PHASOR_EVAL_BATCH_CAP=4 "$bin" probes language-l2 "$seed" runs/learned-streams extended
done
"$bin" catalog-report runs/learned-streams language-l2
```

## Bounded recurrent readout

Each position supplies its contextual embedding features to three learned
linear maps. A sigmoid gate retains part of the preceding memory, and the
remaining weight goes to a learned tanh proposal. A second sigmoid gate
reads the resulting state. A fourth linear map adds that read to the
current hidden vector before the tied classifier. All four maps are
shared across positions and tasks are trained separately.

Lean proves that the magnitude of every memory coordinate stays at most
one at any sequence length. The difference between two states equals their
initial difference times the product of the intervening retention gates.
The output map starts at zero, so the untrained model reproduces the
original Phasor features for every embedding
sequence. The proof also gives the gradient from a current read and all
later reads. [Bounded recurrent readout](../../src/Transformer/Phasor/BoundedMemoryRead.lean)

The FP32 implementation agreed with independent FP64 logits within
`3.5e-7` at widths 64 and 128. Both widths passed finite differences
for the tied embedding, a backbone router and each of the four new
projections. Only the output projection received a readout gradient at
initialization. On arbitrary dense inputs the three tested directions
agreed with finite differences and the first three positions were
bit-identical with either suffix. Width 128 adds 16,384 learned weights.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.BoundedMemoryRead
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,bounded-memory-read
bin=./target/release/phasor-value-transport
"$bin" memory-check
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen suite-blocks 0 runs/bounded-memory 2000
"$bin" probes suite-blocks 0 runs/bounded-memory
"$bin" nv11-train runs/bounded-nv11/seed0 0 3000
"$bin" nv11-score runs/bounded-nv11/seed0/nv11-train-3000.f32
```

## Causal product at the embedding input

This candidate changes the input to the first transformer layer. Let
`z[t]` be the RMS-normalized input embedding and let the preceding
vector at the start of a sequence be zero. Four learned matrices give
`u[t] = z[t] + O ((A z[t]) ⊙ (B z[t] + C z[t−1]))`. The four Phasor
streams start from copies of `u[t]`. All positions use the same maps,
with 32 product coordinates and a zero-initialized output projection.

Lean gives the exact cross-position interaction when two neighboring
vectors change and the gradient that the next position contributes to
the preceding input. It also proves that zero initialization preserves
every original layer state and that suffix changes preserve every
earlier state at any depth.
[Causal product input](../../src/Transformer/Phasor/CausalProductInput.lean)

## Compact complex feed-forward branch

This candidate keeps the embedding width, attention heads, depth, four
streams and learned stream readout. Inside each transformer layer the
complex-product feed-forward branch uses one complex unit per real input
channel. At model widths 64, 128 and 256 the original branch has 86, 171
and 342 complex units. The width-128 candidate therefore saves 33,024
expanded weights and the same number of real multiply-accumulates per
token in the three feed-forward projections of each layer.

Lean proves that the raw contribution of any group of units is additive
before normalization. Zero-padding the compact branch reproduces its
states in the wider model for every layer, position and embedding input.
[Compact feed-forward proofs](../../src/Transformer/Phasor/CompactFFN.lean)

## Representation-only causal memory

How much of the closed-relation head's learning speed and length transfer can
Phasor recover from its causal hidden states? All four arms use the original
backbone. Let `h[t]` be its RMS-normalized mean of four streams and `E` its
tied embedding. Both learned heads return logits `E (h[t] + O r[t])`.

| Head | Read at position `t` |
| --- | --- |
| Original Phasor | Projects `h[t]` directly with `E`. |
| Closed relation | Uses the historical 17 trainable tables, token addresses and closed-relation gain. |
| Transition attention | Scores each key `K h[i]` with `Q h[t]` by a dot product divided by √32. Softmax over `i + 1 < t` weights the values `V h[i+1]`. An empty memory returns zero. |
| Recurrent prefix | Computes `s[t] = tanh(A h[t]) ⊙ s[t−1] + tanh(B h[t])` and reads `r[t] = sigmoid(C h[t]) ⊙ s[t]`. The initial state is zero in each sequence. |

The two learned heads have 32 memory coordinates, four bias-free projections
and the same parameter count. Their weights start from independent Gaussian
draws with variance equal to the reciprocal of the input dimension. A separate
random stream preserves the paired backbone initialization and minibatch order.
Masked cross-entropy trains the backbone, tied embedding and head jointly
through the pinned independent-coordinate AdamW optimizer.

The benchmark uses seeds 0, 1 and 2, the original splits, shuffled epochs and
short final batches. All arms take 8,000 updates. WSD retains its 30,000-update
horizon for binding and 100,000 for the other cases. Complete held-out scoring
runs at initialization and every 64 updates for binding or 250 for the other
tasks. The first 99% score is the first **observed** crossing. Training continues
to the common budget even after a crossing. A fresh engine loads the update-8,000
weights and scores the held-out split and each available test split. Parity
requires greedy generation of the answer bit and EOS. Binding and lookup
require a correct prediction at every labeled query in the sequence. NV11
scores the answer at the second operand.

On each trained memory checkpoint the scorer takes a longest held-out
sequence and replaces every token after each possible prefix in turn. It
records the largest absolute change over every shared-prefix logit with
all labels masked. The separate `memory-check` command also compares whole
sequences in a batch against their isolated, unpadded prefixes.

The build expects the pinned DiffGPT checkout under `.lake/reproduction/gpt`.
Run these commands from the repository root. [DiffGPT checkout instructions](../rerope/README.md#build-and-commands)

```sh
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference
./target/release/phasor-value-transport memory-check
./run_memory_benchmark.sh runs/hidden-memory-8000
```

The script writes each cell's observations, checkpoint and scores under its
head, case and seed. Repeating it verifies the saved weights, observation
cadence, first crossing and replayed scores. For one cell, use
`memory-bench HEAD CASE SEED UPDATES RUN_DIR` with the same binary.

### Measurements at the stop

Eight of the planned 48 cells completed 8,000 updates. All eight are original
Phasor controls. The third lookup seed stopped after its 3,500-update observation.
The short learned-head runs below establish that the heads train and replay.
The fraction of the closed head's convergence and length-transfer gains recovered
across three seeds is still undetermined.

| Task | Independent parameters, original | Closed relation | Transition | Recurrent |
| --- | ---: | ---: | ---: | ---: |
| Parity, up to 20 bits | 78,472 | 152,457 | 86,664 | 86,664 |
| Key/value binding and two-hop lookup, each | 80,520 | 189,321 | 88,712 | 88,712 |
| NV11 | 135,504 | 346,577 | 151,888 | 151,888 |

Each learned head adds 8,192 independent coordinates at width 64 and 16,384
at width 128. The totals count each tied complex coordinate once. Each cell's
JSON also gives the number of stored FP32 entries in the expanded layout.

Counts and updates below are in seed order 0 / 1 / 2. Each numerator counts
whole correct sequences. “Miss” means every scheduled score through 8,000
updates was below 99%.

| Original Phasor | Held-out denominator | Correct at initialization | First 99% update | Correct at 8,000 | Longer-input correct at 8,000, each out of 64 |
| --- | ---: | --- | --- | --- | --- |
| Parity | 284 | 0 / 0 / 0 | 6,000 / miss / miss | 247 / 267 / 274 | 32 bits 44 / 22 / 41. 64 bits 38 / 25 / 39. |
| Key/value binding | 512 | 0 / 0 / 0 | 448 / 512 / 512 | 507 / 512 / 512 | Source provides an ordinary-length test only. |
| Two-hop lookup | 32 | 0 / 0 / 0 | miss / miss / stopped | 2 / 1 / — | Length 48 scored 0 / 1 / —. Length 96 scored 0 / 0 / —. |

The archived closed-relation runs used the same splits, initialization seeds,
minibatches and observation cadence. They saved the first crossing. The fresh
64-update binding run below reproduced the archived seed-0 weights byte for byte.

| Closed relation, archived | Held-out denominator | Correct at initialization | First 99% update |
| --- | ---: | --- | --- |
| Parity | 284 | 0 / 0 / 0 | 250 / 250 / 250 |
| Key/value binding | 512 | 501 / 508 / 505 | 64 / 0 / 64 |
| Two-hop lookup | 32 | 32 / 32 / 32 | 0 / 0 / 0 |
| NV11 | 4,705 | 47 / 48 / 48 | 500 / 500 / 500 |

| Fresh seed-0 pilot | Updates | Initial held-out | Final held-out | First 99% | Longer-input correct, each out of 64 |
| --- | ---: | ---: | ---: | --- | --- |
| Recurrent, parity | 250 | 0/284 | 194/284 | Miss through 250 | 39 at 32 bits, 29 at 64 bits |
| Transition, binding | 64 | 0/512 | 5/512 | Miss through 64 | — |
| Closed relation, binding | 64 | 501/512 | 507/512 | 64 | — |
| Transition, two-hop lookup | 250 | 0/32 | 0/32 | Miss through 250 | 0 at 48, 0 at 96 |
| Recurrent, two-hop lookup | 250 | 0/32 | 0/32 | Miss through 250 | 0 at 48, 0 at 96 |

At update 250 original Phasor scored 145/284 on parity in seed 0. The
recurrent head gained 49 correct answers at that observation. NV11 has two
input tokens, so its strict transition memory is empty at both positions.
Both learned heads showed zero prefix-logit change in the width-64 and
width-128 checks. The maximum error against the independent FP64 forward was
`6.88e-7`, and all 24 masked-loss directional derivatives passed. Trained
transition-binding and recurrent-lookup checkpoints also gave zero logit
change across 11 and 23 suffix replacements respectively.

```sh
bin=./target/release/phasor-value-transport
export PHASOR_EVAL_BATCH_CAP=16
"$bin" memory-bench recurrent basis-parity20 0 250 runs/hidden-memory-pilots
for head in transition closed; do
    "$bin" memory-bench "$head" gpt-probe-key-value-binding 0 64 runs/hidden-memory-pilots
done
for head in transition recurrent; do
    "$bin" memory-bench "$head" suite-lookup 0 250 runs/hidden-memory-pilots
done
```

## Historical benchmark with hand-written features

The closed-relation Phasor passed all **159 GPT task/seed runs** across 41 short
and twelve long tasks on Kernite v0.6.0. Each task used its own weights and
the original AdamW. The same architecture classified all 9,409 NV11 pairs
correctly at update 750 in every seed. The newest short task is
`(a + b)^48 mod 97`, which all three seeds passed at update 250. The newest
long task is the binary block language L3.

The GPT suite also includes the official eight-needle text evaluation at up
to 128K and at 1M context. The result is a mean match ratio on each published
split. [MRCR V2 stress test](../mrcr_v2/README.md)

### Closed-relation architecture on Kernite v0.6.0

Each completed triple writes its first token and third token as an edge
labelled by the middle token. At any position, relations that form a
permutation of their written tokens can propose a two-step destination.
When the largest relations with a moving destination agree, their common
destination supplies the read address and doubles the learned residual gain.
Other positions use the latest ordinary successor and the ordinary gain.
The same rule runs at every position of every task and uses the predecessor's
17 tables and tied embedding. [Closed-relation read and gain](../../src/Transformer/Phasor/ClosedRelations.lean)

| Matched runs against the 0.75-gain successor | Passes | Epoch ratio to 99% | Elapsed-time ratio | Fewer / equal / more updates |
| --- | ---: | ---: | ---: | ---: |
| Short | 114/114 | 0.9835 | 0.9862 | 5 / 106 / 3 |
| Long | 33/33 | 0.9035 | 0.9038 | 2 / 31 / 0 |

These ratios are geometric means on the same GPU, Kernite release and
FP32 affine configuration. The short tier retained the one existing pass
at initialization. Depth-one C-RASP fell from 750 / 9,500 / 250 updates
to 250 / 1,000 / 250. Three short runs took longer in updates. Hard-depth
seed 2 rose from 1,000 to 1,250, branch-amplitudes seed 0 from 416 to 512,
and blocks seed 0 from 500 to 750.

Two-hop lookup adds three short runs beyond the comparison on the
fourteen-extension registry at `6bc2b20c`.
Every checkpoint scored 32/32 whole sequences on novel validation inputs
at initialization, then 64/64 on each of the ordinary, length-48 and
length-96 test sets. On NV11 all three seeds reached 99% at update 500
and classified all 4,704 training and 4,705 held-out pairs correctly at
both 750 and 3,000 updates.

Both power-of-sum source tasks passed at update 250 in every seed. The
longest cell took 2.37 seconds. The predecessor produced byte-identical
weights on all six runs. The checkpoint replay audit covers 123 short
and 36 long GPT runs and three NV11 runs, for 162 verified checkpoints.
[Source case definitions](../convex_stress_arithmetic_data/references/definitions.json)

| Function modulo 97 | Held-out pairs, out of 4,705, seeds 0 / 1 / 2 | Held-out first operand zero, out of 51, seeds 0 / 1 / 2 |
| --- | ---: | ---: |
| `(a + b)^32` | 4,703 / 4,698 / 4,704 | 51 / 51 / 51 |
| `(a + b)^48` | 4,684 / 4,672 / 4,698 | 50 / 50 / 50 |

The closed-relation model reached 99% on the source L3 length-128 novel
validation split at updates 750 / 1,000 / 500. The slowest run took
114.73 seconds. Each checkpoint passed 512/512 on the original-length test.
The independent length-128 test scored 509 / 512 / 509 out of 512.
For all 4,096 twelve-letter words, the whole-sequence scores were
4,096 / 4,096 / 4,023. The 257-token repeated-letter probe scored
256 / 256 / 255 out of 256. Reproduce these four tests with
`PHASOR_EVAL_BATCH_CAP=4 "$bin" probes language-l3 "$seed" "$out" extended`.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.ClosedRelations
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,closed-relation-boost
export PHASOR_EVAL_BATCH_CAP=16
bin=./target/release/phasor-value-transport
out=runs/v060-closed
mkdir -p "$out"
git show 6000fa8e:experiments/phasor_value_transport/extensions.tsv > "$out/extensions.tsv"
"$bin" shared-check
./run_stress.sh cell "$out" "$out/extensions.tsv"
"$bin" extended-report "$out" "$out/extensions.tsv" > "$out/report.json"
./run_stress.sh replay "$out" "$out/extensions.tsv" > "$out/replays.jsonl"
jq -es --slurpfile report "$out/report.json" -f verify_replays.jq "$out/replays.jsonl"
for seed in 0 1 2; do "$bin" probes suite-lookup "$seed" "$out"; done
```

## Predecessor with successor displacement and gain 0.75

Every position uses the same causal prefix scan and the same 17 learned
tables. Let `b` be the current token and `r` its most recently observed
successor in the completed prefix, with `r = b` for an unwritten key.
The residual adds `E[r] − E[b]` before RMS normalization, scales the
normalized vector by the learned gain and adds it to the backbone feature.
The half-gain model and this model have the same parameter count.
[Tied read and score identities](../../src/Transformer/Phasor/SuccessorDisplacement.lean)

| Matched tier | Passing task/seed runs | Epoch ratio to half-gain predecessor | Faster / tied / slower |
| --- | ---: | ---: | ---: |
| Short, including seven catalog additions | 96/96 | 0.6223 | 67 / 23 / 6 |
| Short, including the first nine short catalog additions | 102/102 | 0.6144 | 70 / 26 / 6 |
| Short, including noisy parity and modular sum to the 24th power | 108/108 | 0.6314 | 70 / 32 / 6 |
| Short, including the square-product and quadratic-residue tasks | 114/114 | 0.6401 | 73 / 35 / 6 |
| Original long tasks | 30/30 | 0.7550 | 17 / 9 / 4 |
| Long, including depth-one C-RASP | 33/33 | 0.8049 | 18 / 9 / 6 |

Each ratio is the geometric mean of examples consumed to the first 99%
crossing for pairs that required training on both models. Key/value binding
seed 1 newly passed at initialization. The other two binding seeds took 64
updates, down from 448. Easy recall fell from 400 to 150 in all seeds.
NV11 first crossed 99% at 500 updates in each seed, then became exhaustive
at 750. All three were still exhaustive at 3,000.

Twelve pairs needed more updates. Hard depth rose from 750 to 1,000 in seed 0
and 1,250 to 1,750 in seed 1. Branch amplitudes rose from 352 to 416 and
320 to 352 in those seeds. Dyck seeds 0 and 2 rose from 250 to 500.
Associative divisor went from 646 / 684 / 570 to 988 / 1,064 / 627.
Fibonacci seed 1 went from 7,480 to 12,920. Depth-one C-RASP seeds 0 and 1
went from 500 to 750 and 1,000 to 9,500. Its third seed fell from 1,000 to 250.

Fresh v0.6.0 training on the same GTX 1050 also reduced elapsed time.
Easy recall took 15.15–15.28 seconds with the selected build and
33.96–34.06 seconds with the half-gain build. Key/value binding took
0.32–0.77 seconds and 3.49–3.52 seconds respectively. The build run first
alternated across the six pairs. Each timed cell reproduced its model's
earlier weights and observation history byte for byte.

For a paired comparison, use a fresh directory for each model. The
following commands reproduce the 147 GPT runs and three NV11 runs from
the fourteen-extension comparison.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.SuccessorDisplacement
cd experiments/phasor_value_transport
export PHASOR_EVAL_BATCH_CAP=16
mkdir -p runs
git show 6bc2b20c:experiments/phasor_value_transport/extensions.tsv > runs/fourteen-extensions.tsv
for arm in half selected; do
    features=fp32-affine-reference,successor-gain-three-quarters
    if [ "$arm" = half ]; then features=fp32-affine-reference,prefix-successor-half; fi
    cargo +stable build --release --locked --features "$features"
    bin=./target/release/phasor-value-transport
    "$bin" shared-check
    ./run_stress.sh cell "runs/v060-$arm" runs/fourteen-extensions.tsv
    "$bin" extended-report "runs/v060-$arm" runs/fourteen-extensions.tsv > "runs/v060-$arm/report.json"
    ./run_stress.sh replay "runs/v060-$arm" runs/fourteen-extensions.tsv > "runs/v060-$arm/replays.jsonl"
    jq -es --slurpfile report "runs/v060-$arm/report.json" -f verify_replays.jq "runs/v060-$arm/replays.jsonl"
done
git show 849aa3be:experiments/phasor_value_transport/extensions.tsv > runs/seven-extensions.tsv
jq -s --rawfile extensions runs/seven-extensions.tsv -f compare.jq \
    runs/v060-half/report.json runs/v060-selected/report.json
jq -s --rawfile extensions runs/fourteen-extensions.tsv -f compare.jq \
    runs/v060-half/report.json runs/v060-selected/report.json
```

## Kernite v0.6.0

The Phasor runner uses the published v0.6.0 CUDA and macro crates with
Rust 1.100.0. The original DiffGPT trainer and AdamW are pinned to
`5a37c515`. Fresh results record the Kernite version from Cargo.lock.
Checkpoint replay reports both the training and scoring versions.
The replay reloads the weights, verifies their SHA-256 digest and scores
the original held-out split. The verifier matches the architecture, digest,
training version and exact sequence count for all three seeds of every
reported task.

With a full successor displacement and initial gain 0.75, the v0.6.0
GPU forward pass agreed with the independent FP64 oracle to within
8.23 × 10⁻⁷ across all four tested shapes. All 76 directional derivatives
passed. The same compiler and architecture on Kernite 0.3.2 printed
identical oracle results. The six matched training runs produced byte-identical
observation histories and weights across the two versions. Key/value binding
crossed 99% at updates 64, 0 and 64. NV11 crossed at 500 in every seed.
Elapsed times for each matched NV11 run were within 0.02 seconds.

Under v0.6.0 every seed reached exact NV11 at update 750 and retained it at
3,000. Independent reloads of all six snapshots classified every one of
the 4,704 training pairs and 4,705 held-out pairs correctly. The previous
half-gain architecture first reached this score at 1,250 updates.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.DisplacementGain
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,successor-gain-three-quarters
bin=./target/release/phasor-value-transport
"$bin" shared-check
"$bin" shared-values runs/kernite-060-exact/seed0 0 3000
"$bin" nv11-score runs/kernite-060-exact/seed0/shared-values-first-exact.f32
PHASOR_EVAL_BATCH_CAP=16 "$bin" cell gpt-probe-key-value-binding 0 runs/kernite-060-gain075
"$bin" replay gpt-probe-key-value-binding 0 runs/kernite-060-gain075
```

The runner now builds the list of rows that use each table entry during
backpropagation. Forward-only scoring reads the table addresses directly.
Lean proves that summing gradients over these rows gives the adjoint for
any set of active rows, including repeated addresses. [Table read adjoint](../../src/Transformer/Phasor/PairValues.lean)

On the GTX 1050 this reduced fresh cell time by 0.56% as a geometric mean
over modular addition and 20-bit parity, three seeds each. All six elapsed
times fell, by 0.19–0.95%. The observation histories and checkpoint weights
were byte-identical. To reproduce, compare `cell arithmetic-add` and
`cell basis-parity20` with `successor-hit-gain-seven-quarters`, FP32 affines
and `PHASOR_EVAL_BATCH_CAP=16` against commit `cafcd74c`. Use fresh directories
for each build and alternate their order between seeds.

A separate timing trial replaced the vocabulary remainder in the forward and
backward transport kernels with a comparison against `p − offset`. Using
`prefix-successor-half`, the same six task/seed pairs preserved every
training observation and checkpoint byte.
The comparison took 0.30% longer on the GTX 1050 as a geometric mean, with
five of six runs slower. The selected kernel keeps the remainder instruction.
[Circular address identity](../../src/Transformer/Phasor/TransportedValues.lean)

Fusing the residual's RMS normalization and addition gave byte-identical
weights and observation histories in 18 fresh, matched v0.6.0 pairs. The
geometric mean time ratio was 0.9967 across nine short pairs and 1.0001
across nine long pairs. Addition ran 0.65% faster, 20-bit parity 0.04%
faster, language L1 0.31% slower and easy associative recall 0.12% slower.
Eight of the 18 pairs ran faster. The selected implementation keeps the
two launches. The timing used alternating run order, three seeds per task,
two rounds for addition and parity, and one for language and recall.
`shared-check 64`, `shared-check 128` and `shared-check 256` exercise the
vocabulary-tied residual and its adjoint at each supported width.
[Coordinate identity for RMS normalization](../../src/Transformer/Phasor/SharedValues.lean)

The host address trial caches the first three power residues of each token.
All thirteen power tables can reuse these values across every input pair.
For each sum the host compares against the first wrapping value, and for
the difference it compares the two operands. Lean proves that the resulting
word indices agree with the original formulas for every vocabulary size.
[Power and difference address identities](../../src/Transformer/Phasor/PowerAddresses.lean)

The trial matched the direct formulas for 991,457 table addresses across
nine vocabulary sizes. Twelve fresh paired cells reproduced the selected
weights and training histories byte for byte. Relative to direct evaluation,
short runs were 0.33% faster and long runs 0.35% slower as geometric
means. Addition was 0.62% faster in all three seeds, while 20-bit parity
was 0.96% slower across its three seeds. The selected host implementation
continues to evaluate the per-pair formulas.

The dense pair-address trial precomputes the entire static address vector
for each ordered token pair when the vocabulary fits in a bounded host
table. Each causal read copies that vector and applies its usual changes
for the current prefix. Lean proves that the flat index recovers both
tokens and that every updated state agrees with direct evaluation over any
sequence, including repeated reads and changes to an earlier address.
[Dense address lookup and scan equivalence](../../src/Transformer/Phasor/PowerAddresses.lean)

The tied-gradient trial combines the backbone, output-projection and value
contributions to each embedding coordinate in one GPU launch. It retains
the two original FP32 additions in their original order. Lean proves that
the combined gradient gives the directional derivative of both uses of
the embedding on every finite set of active rows.
[Tied embedding adjoint](../../src/Transformer/Phasor/TiedGradients.lean)

The fused backward launch reproduced all 97 directional checks and every
checkpoint and observation byte in twelve matched cells. Four cells ran
faster. The geometric mean elapsed time rose by 0.18% on the short tier
and 0.34% on the long tier. The selected implementation keeps the two
additions in separate launches. To reproduce, apply the patch at
`2e58f96b`, use `fp32-affine-reference,single-write-gain`, and run
`time_cells.sh` on addition, 20-bit parity, language L1 and depth-one C-RASP.
[Tied gradient source change](trials/tied-embedding-gradients.patch)

The packed-transfer trial sends table addresses and gain evidence in one
host-to-device copy. The backward pass packs the offset directory and
its row indices in a second buffer. Lean proves that both forward reads
and every ordered gradient reduction agree with separate arrays, including
empty groups, repeated owners and unused capacity after a shorter batch.
[Packed forward and transpose layout](../../src/Transformer/Phasor/PackedReadBuffers.lean)

The transport-normalization trial writes two coefficients when a vocabulary
mass is normalized. The original mass supplies the softmax derivative.
The second includes the displacement from the current token to its read
address and supplies the embedding multiply. Lean proves that each output
coordinate agrees with the two-pass calculation, even at a circular wrap
or a self read, and that changes in the normalized mass pass through the
displacement unchanged.
[Transport coefficient identities](../../src/Transformer/Phasor/TransportCoefficients.lean)

The combined kernel reproduced all 97 directional checks and every
checkpoint and observation byte in twelve fresh v0.6.0 pairs. Five cells
ran faster. The short-tier elapsed-time ratio was 0.9999 and the long-tier
ratio was 1.0044. The default keeps the two-pass computation. To repeat,
apply the patch at `47e0ca4d`, use `fp32-affine-reference,single-write-gain`,
and run `time_cells.sh` on Dyck, 20-bit parity, language L1 and depth-one
C-RASP. [Coefficient timing source](trials/emit-transport-coefficients.patch)

The RMS and power-address timing trials can be rebuilt from their archived
source changes. Build and save the FP32 control at `58bd7352` for RMS or
`8f7ca6ef` for cached powers. In that checkout apply the corresponding
patch with `git apply`, then build and save the second binary using the
same features. The paired runner trains fresh cells, alternates build order
and checks every byte of each checkpoint and observation history.
[RMS source change](trials/fused-residual-rms.patch) · [Cached powers source change](trials/cached-power-addresses.patch)

```sh
PHASOR_EVAL_BATCH_CAP=16 ./time_cells.sh BEFORE_BIN AFTER_BIN runs/timings \
    arithmetic-add basis-parity20 language-l1 basis-recall-easy
```

## Half-gain prefix and successor model

The half-gain predecessor applies one causal update at every token. Its prefix state
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

At the seven-addition comparison the benchmark had 32 short tasks and ten
long tasks. All 96 short and 30 long task/seed runs reached 99% and replayed
independently.

Fresh key/value-binding runs on the GTX 1050 took 5.41, 5.32 and 3.95 s
for the incumbent, and 3.51, 3.50 and 3.50 s for the half-gain model.
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
cargo +stable build --release --locked --features fp32-affine-reference,prefix-successor-half
bin=./target/release/phasor-value-transport
"$bin" shared-check
PHASOR_EVAL_BATCH_CAP=16 ./run_stress.sh cell runs/prefix-successor-half extended
./run_stress.sh replay runs/prefix-successor-half extended
"$bin" extended-report runs/prefix-successor-half > runs/prefix-successor-half/report.json
jq -s -f compare.jq runs/shared-quarter/paired-report.json runs/prefix-successor-half/report.json
```

To include all previously admitted tasks in a promotion comparison, pass a
frozen extension registry from the baseline commit. Both reports must contain
the same held-out sizes and three passing seeds per case.

The report counts a first crossing at update zero as a pass at initialization.
It lists gains, retentions and losses of these passes by tier. The geometric
mean uses the pairs where both models trained. The speed criterion requires
a mean at most one in each such tier and retention of every baseline pass
at initialization. A tier composed entirely of initialization comparisons
has no geometric mean.

Fresh reports also retain the GPU, Kernite version, affine precision and
source fingerprint for each timed cell. When every pair in a tier matches
on these fields, `compare.jq` reports its geometric mean elapsed-time ratio
and counts of faster, tied and slower cells. This timing includes evaluation
at initialization. Keep the evaluation batch cap and GPU workload fixed
when training both cohorts.

```sh
git show dd5e0f14:experiments/phasor_value_transport/extensions.tsv > runs/baseline-extensions.tsv
jq -s --rawfile extensions runs/baseline-extensions.tsv -f compare.jq CONTROL_REPORT CANDIDATE_REPORT
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

| Seed-0 held-out screen | Incumbent | Quarter displacement | Full displacement |
| --- | ---: | ---: | ---: |
| Easy recall, first 99% update | 400 | 400 | 350 |
| Key/value binding, first 99% update | 448 | 448 | 384 |
| Blocks, first 99% update | 750 | 500 | 500 |
| Dyck and 20-bit parity, first 99% update each | 250 | 250 | 250 |
| MQAR, exact sequences at 1,000 updates | 0/32 | 3/32 | 25/32 |
| Two-hop lookup, exact sequences at 750 updates | 0/32 | 0/32 | 0/32 |

The full displacement retained all 120 GPT passes on the cohort with five
extensions. The geometric-mean epoch ratio to the incumbent was 1.0122
across the 90 short runs and 0.9906 across the 30 long runs. Blocks seed 2
rose from 1,250 to 9,750 updates. The half-gain model without displacement
won that comparison. All three full-displacement NV11 seeds reached exact
classification at update 1,250 and remained exact at 3,000.

At 5,000 MQAR
updates it scored 31, 30 and 31 of 32 sequences in seeds 0, 1 and 2. Seed 0
still scored 31/32 at 15,000. The frozen successor address itself matched
all 64 MQAR targets and all 4,096 easy-recall targets on their novel validation
splits. For two-hop lookup one address matched 0/64 targets, and following
it twice matched 21/64, or 4/32 whole sequences. To reproduce these address
counts, run `successor-alignment CASE 1` and `successor-alignment CASE 2`
using the binary built below. This diagnostic uses no model weights.

Build each arm separately. The native oracle compares the logits with an
independent scalar FP64 computation and checks 76 directional derivatives,
including the input and output paths through the tied embedding.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.SuccessorDisplacement
cd experiments/phasor_value_transport
for arm in successor-displacement successor-displacement-quarter; do
    cargo +stable build --release --locked --features "fp32-affine-reference,$arm"
    ./target/release/phasor-value-transport shared-check
    PHASOR_EVAL_BATCH_CAP=16 ./target/release/phasor-value-transport screen suite-mqar 0 "runs/$arm" 1000
done
```

## Gain of the tied successor read

The trial keeps the full tied displacement and initializes the learned residual
gain at 0.75. For fixed other parameters, Lean proves that interpolating
two gains interpolates every logit. The absolute change in a two-class
margin is bounded by the gain change in magnitude, the distance between
the two embeddings and the square root of the hidden width, multiplied
together. The trial compares first observed crossings and exhaustive NV11
checkpoints under the original optimizer. [Gain interpolation and margin bound](../../src/Transformer/Phasor/DisplacementGain.lean)

A second trial doubles the learned gain only when the memory read returns
a different token. It starts the shared gain at 0.5. The condition depends
on the input prefix, and an injective token renaming preserves it. Lean
bounds the change in the two-class margin at every position and proves
exact score agreement on self-reads, including both positions of NV11.

On Kernite v0.6.0, the doubled-gain model classified all 32 novel MQAR
sequences at initialization in each of the three seeds. Each independently
reloaded checkpoint scored 64/64 on the ordinary test, 64/64 at length 48
and 64/64 at length 96. The slowest fresh cell took 0.25 seconds. Seed 0
also reached 99% on easy recall and key/value binding at initialization.
It scored 31/32 on Dyck at the 250-update retention cutoff. The incumbent
keeps its place in the expanded suite.

The smaller multiplier 1.75 also gave seed 0 an exact 32/32 MQAR score at
initialization and 31/32 on Dyck at 250 updates. To repeat this screen use
`successor-hit-gain-seven-quarters` and a fresh output directory. For fixed
weights Lean proves the 1.75 logit is a quarter of the multiplier-1 logit
plus three quarters of the multiplier-2 logit at every token.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.DisplacementGain
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,successor-hit-gain
bin=./target/release/phasor-value-transport
"$bin" shared-check
export PHASOR_EVAL_BATCH_CAP=16
for seed in 0 1 2; do
    "$bin" cell suite-mqar "$seed" runs/kernite-060-hit-gain
    "$bin" probes suite-mqar "$seed" runs/kernite-060-hit-gain
done
"$bin" screen suite-dyck 0 runs/kernite-060-hit-gain 250
```

## Agreement among previous successors

The agreement trial doubles the hidden residual gain when every completed
write to the current key names the same successor and that successor differs
from the key. Conflicting writes restore the constant gain of 0.5 at
initialization. The table and tied-displacement reads still use the latest
successor. During training AdamW updates the single gain shared by every row.

Lean proves that a last-value register and one conflict bit per key detect
exactly the histories with two distinct values. An injective token renaming
preserves the resulting gain. On any conflicting history every class logit
agrees with the constant-gain displacement model at the same weights.
[Causal agreement proofs](../../src/Transformer/Phasor/SuccessorAgreement.lean)

With `fp32-affine-reference,successor-agreement`, all three seeds scored
32/32 on novel MQAR sequences at initialization and 64/64 on each of
the ordinary, length-48 and length-96 tests after reload. Seed 0 again
scored 31/32 on Dyck at 250 updates. Changing the ambiguity rule preserved
recall and left the Dyck deficit at one sequence.

## Independent gain on changed successor reads

This trial trains a base gain and a separate scalar increment. A self-read
uses the base. A changed successor uses the sum. Both scalars start at 0.5,
so at initialization every logit matches the doubled-multiplier trial.
AdamW can then adjust the two amplitudes independently. Lean proves that
any prescribed pair of amplitudes has unique parameters whenever the input
contains both kinds of read. [Independent-gain theorem](../../src/Transformer/Phasor/DisplacementGain.lean)

The `successor-bonus` build passed all 80 directional derivatives, including
the separate scalar. At update 500, blocks seed 0 had learned a base gain of
0.55031 and a bonus of 0.54894. Its checkpoint replayed at 28/28. On MQAR it
scored 32/32 at initialization, and on Dyck it scored 31/32 at the 250-update
cutoff. This trial was set aside.

## Centered prefix frequencies

The separate centered-frequency trial reads the average tied embedding of
all tokens strictly before the previous position and subtracts the current
token's embedding. It adds half this vector before the residual RMS of the
full-displacement model with base gain 0.75. Empty histories contribute
zero. The rule uses every earlier occurrence, including repetitions, and
introduces no trained parameters.

Lean proves that the same update can be applied to the vocabulary
coefficients of the embedding read. Constant histories contribute zero
when they equal the current token, and all logits on two-token inputs
agree with the 0.75-gain model. [Centered-frequency derivation](../../src/Transformer/Phasor/PrefixFrequency.lean)

With `fp32-affine-reference,prefix-frequency` on v0.6.0, the four forward
checks agreed with the FP64 calculation to within 8.23 × 10⁻⁷. All 76
directional derivatives passed. At update 500 the NV11 seed-0 run had
byte-identical weights and training observations to the 0.75-gain control.
Its checkpoint replayed at 4,703/4,705. On seed 0, the frequency model
scored 31/32 on Dyck at 250 updates, 31/32 on MQAR at 1,000 and 0/32 on
the histogram-with-BOS task at 1,000. It was set aside.

## Two reads through a shared relation

The relation-path trial treats every completed triple `(key, relation, value)`
as a write. At each position it follows two edges with the same relation
from the current token, using the latest completed writes in the prefix.
Each relation has one learned scalar, initially zero. A complete path adds
that scalar times the difference between the endpoint and current embeddings
inside the existing residual. A missing edge contributes zero. At
initialization the trial has the same logits as the 0.75-gain model.
[Causal path and derivative proofs](../../src/Transformer/Phasor/RelationalPaths.lean)

The v0.6.0 build with `fp32-affine-reference,relation-paths` passed 100
directional derivatives and matched the independent FP64 logits to within
8.51 × 10⁻⁷. The length-12 check includes two relations that reach the
same endpoint. On two-hop lookup at 1,000 updates, seed 0 scored 21/32
with learned relations and 1/32 with the 0.75-gain control. The relation
model scored 32/32 on Dyck at 250 updates and 31/32 on MQAR at 1,000.
Reproduce with `shared-check` followed by `screen TASK 0 FRESH_DIR CUTOFF`.

On the extended lookup screen, its seed-0 score peaked at 24/32 at update
3,250 and was 17/32 at 10,000. A second variant initializes the relation
amplitudes to one and divides each row's contribution by the number of
relations leading to a different token. For a nonempty set of agreeing
reads, the initial update is exactly one displacement to their common
endpoint. Lean bounds the mean by the largest individual weighted
displacement, regardless of how many paths the prefix contains.
[Mean relation read](../../src/Transformer/Phasor/RelationalPaths.lean)

The `relation-path-mean` build passed all 101 numerical derivatives,
including an individually perturbed relation in a two-relation read.
Its derivative was 0.741261 analytically and 0.741482 by centered finite
differences. Seed 0 scored 23/32 on lookup and 30/32 on MQAR at update
1,000, and 31/32 on Dyck at update 250. This variant missed the threshold
on all three screens.

The support-ranked trial counts, for each relation, the distinct source
tokens from which the prefix table allows two consecutive reads. At a
query it averages the moving paths with the highest count. Ties share
the update. This rank uses only completed triples and treats every token
as a possible relation or endpoint. A unique maximizer supplies one full
tied displacement at initialization. [Relation support proofs](../../src/Transformer/Phasor/RelationSupport.lean)

The `relation-support` build checked its incremental counts after every
write on all 6,561 eight-token prompts over three symbols. It passed 123
directional derivatives. For an isolated strict winner the learned amplitude
gave 1.354837 analytically and 1.354814 numerically, and the losing
amplitude gave zero by both methods. On seed 0, lookup reached 26/32 by
1,000 updates and its best score through 5,000 remained 26/32. Blocks
passed at 500 updates (28/28), Dyck scored 31/32 at 250 and MQAR scored
30/32 at 1,000. These results leave the selected architecture in place.

## One address from a closed relation

This trial changes the address shared by the successor table and the tied
embedding displacement. Every completed triple still writes an edge. A
relation qualifies when all of its written sources lead to another written
source and it has as many distinct destinations as sources. Lean proves
that these two counts characterize a permutation of the written domain.
At each query the reader follows each qualifying relation twice, then takes
the moving endpoints from the relations with the most written sources.
If those endpoints agree, their common value becomes the address. An empty
vote or a disagreement leaves the ordinary successor address in place.
The model has the same 17 tables and learned gain as the selected model.
[Closed relation and address proofs](../../src/Transformer/Phasor/ClosedRelations.lean)

The source-row storage trial keeps a sorted list of relation labels and
destinations for each written source. Lean proves that replacing a value
or inserting a new label preserves strict order, and that the resulting
lookup matches the flat table after every finite write stream.
[Sorted relation rows](../../src/Transformer/Phasor/RelationRows.lean)

Sorted rows reproduced all 78 directional checks and the weights and
observation histories of twelve fresh matched cells on Kernite v0.6.0.
Six cells ran faster. Across all seeds the geometric mean time rose by
0.14% on the short tier and 0.24% on the long tier. The default keeps the
flat table. To reproduce, apply the patch at `4ff019bf` and run
`time_cells.sh` on Dyck, 20-bit parity, language L1 and depth-one C-RASP
with the default FP32 build on both sides.
[Sorted relation row source](trials/sorted-relation-rows.patch)

The query, the first endpoint and the second endpoint are distinct whenever
a closed two-step read moves. Thus fewer than three distinct stored edges
force the fallback for every relation and every query. The same argument
applies to a single relation with at most two written sources. A three-cycle
attains the bound.
[Minimum support of a closed read](../../src/Transformer/Phasor/ClosedReadBounds.lean)

Checking the bound before inspecting a relation reproduced all 78
directional checks and every checkpoint and observation byte in twelve
paired cells. Three cells ran faster. Time rose by 0.14% across the short
tier and 0.52% across the long tier as geometric means. Apply the patch at
`6bc2b20c` and use `fp32-affine-reference,closed-relation-boost` with
`time_cells.sh` on addition, 20-bit parity, language L1 and depth-one C-RASP.
The selected reader continues to inspect the closed relation directly.
[Minimum-support timing trial](trials/closed-read-minimum.patch)

With `fp32-affine-reference,closed-relation`, the two-step address matched
all 64 labelled positions on the 32 novel lookup sequences. The full model
scored 31 / 30 / 30 sequences at 5,000 updates. Its 78 directional
derivatives included the winning table row (0.549078 analytically,
0.548959 numerically) and the replaced row (zero by both methods).
`closed-alignment suite-lookup` reports address matches on every source split.
The same command on MQAR found no closed reads on novel validation. Its
ordinary successor matched all 64 labelled positions.

The gain trial doubles the common residual gain only when the closed
relations resolve one moving endpoint. Lean gives the exact change in the
endpoint's logit margin. If the normalized read points toward the endpoint
over every rival, one shared multiplier can overcome a finite set of
adverse gaps while retaining comparisons already won on ordinary reads.

With `closed-relation-boost` on v0.6.0, all three untrained seeds scored
32/32 on the novel lookup validation set and 128/128 on training. Reloading
each checkpoint gave 64/64 on the ordinary test, 64/64 at length 48 and
64/64 at length 96. The training-length prompts have 24 tokens. All three
seeds reproduced the selected NV11 checkpoints byte for byte at the first
exhaustive observation (750 updates) and at 3,000 updates. Dyck reached
32/32 at 500 / 250 / 500, matching the selected model. On MQAR, seed 0
scored 31/32 at 1,000 updates.

The single-write trial also doubles the gain when a moving ordinary
successor comes from exactly one completed write of its key. Seeing the
same key twice disables this ordinary-read multiplier, even when both
writes contain the same value. The counter stops at two. Lean proves that
this identifies the unique transition and its destination at any prefix
length and that an injective renaming of the input tokens preserves the
gain. [Single-write proofs](../../src/Transformer/Phasor/SingleWrite.lean)

The `single-write-gain` build passed 97 directional derivatives, including
prefixes with one write, two identical writes and two conflicting writes.
All three seeds scored 32/32 at initialization on both MQAR and lookup.
Each of the six checkpoints then scored 64/64 on its original test and
64/64 on both length-48 and length-96 tests. All three also scored 512/512
at initialization on both easy recall and key/value binding. The Dyck
crossings remained 500 / 250 / 500. Blocks took 750 / 500 / 500, and
depth-one C-RASP took 1,250 / 1,000 / 500.

On NV11, the closed-gain and single-write models both first classified
all 9,409 pairs at update 750 in each seed. At 750 and 3,000, all twelve
snapshots were byte-identical to the corresponding 0.75-gain successor
snapshot. Reloading each file independently reproduced 4,704/4,704 on
training and 4,705/4,705 on held-out pairs. The `shared-values` and
`nv11-score` commands in the Kernite v0.6.0 section reproduce the result
with `closed-relation-boost` or `single-write-gain`.

The exclusive singleton-gain trial learns a separate increment for a moving
successor supplied by one completed write. A closed-relation vote still
doubles the base gain and takes priority if both events occur. All other
positions use the base gain. The model also uses the inclusive change-count
table. Lean proves that a zero increment recovers closed-only gains and
an increment equal to the base recovers the single-write gains. It derives
both scalar gradients for arbitrary batches and shows that the two kinds
of moving read can receive independently prescribed amplitudes.
[Exclusive singleton-gain proofs](../../src/Transformer/Phasor/SingleWrite.lean)

The GPU passed 153 directional checks. A separately supervised singleton
gave identical base and bonus derivatives of 1.454884. The isolated closed
vote gave a zero bonus derivative. On depth-one C-RASP, seed 0 scored
31/32 at update 1,500. The inclusive control had already reached 32/32
at update 500 and the closed-relation default at 250. We ended the trial
at that screen. To reproduce it, apply the patch at `7f4f78bb`, build with
`fp32-affine-reference,exclusive-singleton-gain` and run `shared-check`
and `screen suite-crasp-depth-1 0 RUN_DIR 1500`.
[Exclusive singleton-gain source](trials/exclusive-singleton-gain.patch)

The context-backoff trial reads the last completed triple starting with
the previous and current tokens. A resolved closed relation has priority.
If neither read supplies a value, the model uses the ordinary successor
or, for an unwritten key, the current token. The same address selects the
learned table row and the embedding displacement. Lean proves that each
contextual value actually followed the current token in the prefix, that
agreeing writes preserve the ordinary successor, and that injectively
renaming the vocabulary preserves the read.
[Causal context-backoff proofs](../../src/Transformer/Phasor/ContextBackoff.lean)

The trial passed 118 directional checks. It retained 32/32 novel sequences
at initialization in every seed on MQAR and lookup, including 64/64 on
the separate tests at lengths 48 and 96. Dyck still crossed at 500 / 250 /
500 updates. At 1,500 updates on depth-one C-RASP, seed 0 scored 31/32.
The single-write control had already passed at 1,250. In the same seed
the trial scored 25/32 on typed Dyck and depth-two C-RASP at 2,500,
and 0/32 on repeated-token copy and 11/32 on mode at 1,000. These are
screen cutoffs. The experiment added no passing case. Apply the trial
source at `d5c9b861` and build with `fp32-affine-reference,context-backoff`
to reproduce the screens.
[Context-backoff source change](trials/context-backoff.patch)

## Counting completed transitions

The transition-count trial shifts the squared-product table address by the
number of unequal adjacent tokens strictly before the current read. For
position `i`, it counts edges ending no later than `i−1`. It uses the
incumbent's 17 learned tables and its original gain initialization. The
counter resets at the start of each sequence and uses token equality across
all vocabularies. Lean proves the chunk-composition law, invariance under
injective relabeling, and exact agreement of all logits whenever the observed
prefix is constant. [Transition-count proofs](../../src/Transformer/Phasor/TransitionValues.lean)

With this counter and the half-gain successor, each seed reached 512/512
on the language-L2 validation split at 250 updates. The training lengths
were 32–64. Each reloaded checkpoint scored 512/512 on the separate
length-128 test. The slowest fresh cell took 40.63 seconds on Kernite 0.3.2.
The incumbent's seed 2 scored 341/512 at update 1,750. The transition-count
model scored 31/32 on Dyck seed 0 at the incumbent's 250-update cutoff in
a v0.6.0 rerun, so language-L2 stays outside the selected suite.

To repeat the measurement, build with `fp32-affine-reference,transition-count`
and run `cell language-l2 SEED RUN_DIR` followed by `probes language-l2 SEED RUN_DIR`
for seeds 0, 1 and 2 with `PHASOR_EVAL_BATCH_CAP=16`.

The independent transition-table trial adds an eighteenth learned vector
table to the single-write model. At each position it reads the number of
completed changes of token identity modulo vocabulary size. The table
starts at zero and is added to the hidden value before RMS. Lean
characterizes exactly which residuals it can fit on any finite collection
of prefixes. Prefixes in the same bucket must require the same additive
correction. Its gradient sums the upstream vectors for every row that
read that bucket, including after the count wraps.
[Transition table representation and adjoint](../../src/Transformer/Phasor/TransitionTable.lean)

With `fp32-affine-reference,transition-table`, all three language-L2 seeds
reached 512/512 on length-128 validation at update 250 after training on
lengths 32–64. Each reloaded checkpoint scored 512/512 on the separate
original-length and length-128 tests. The slowest cell took 40.53 seconds.
On the identical seed-2 screen
the selected 0.75-gain successor and the closed-gain model each scored
341/512 at update 750. The transition-table GPU passed 124 directional
checks. At a prefix with 15 completed changes and 12 buckets, only bucket
3 of the new table received a gradient.

Across all 1,024 binary words of ten letters, the same checkpoints
classified every prefix correctly on 1,008 / 992 / 992 words. A second
probe starts with each of the 64 six-letter words and appends alternating
letters up to 128 tokens including the start symbol. It scored
63 / 62 / 62 whole sequences. Both probes run after training through
`probes language-l2 SEED RUN_DIR`. Every label was checked against the
source prefix oracle. On NV11 this architecture first reached 9,409/9,409
at update 750 and scored identically at update 3,000 in each seed,
including independent reloads of all six checkpoints.

The saturated transition-table trial gives the last address to all change
counts at or above that address. It uses the same number of parameters.
Lean proves that capped counters from adjacent segments can be merged, that
all counts below the last address remain distinct, and that modular and
capped tables produce identical reads and gradients on batches below capacity.
[Capped counter and table identities](../../src/Transformer/Phasor/SaturatedTransitions.lean)

Fresh capped-counter training produced the same L2 weights and observation
histories as the modular counter in every seed. It scored 512/512 on each
source test, 1,008 / 992 / 992 on the ten-letter words and 63 / 62 / 62 on
the long alternating tails. The FP32 oracle checked 124 directions and
verified that the gradient at 15 changes entered bucket 11 of 12.
To repeat, apply the patch at `10192529` and build with
`fp32-affine-reference,saturated-transitions`. Run `probes language-l2`
on each trained checkpoint for both additional evaluation sets.
[Capped counter source](trials/saturated-transition-table.patch)

The inclusive transition-table trial counts changes up to and including
the current token. On binary words with a distinct start token, that
counter gives the exact number of blocks before saturation. Lean proves
that one learned scalar per count and one per current token can then
separate each binary block language with its target below the cap.
[Inclusive counts and additive readout](../../src/Transformer/Phasor/InclusiveTransitions.lean)

The second-change trial adds a learned count vector only after two changes
of token identity. It uses the closed-relation gain at every position.
Any prefix made from two constant runs receives zero correction, regardless
of either run's length or of the table weights. Lean proves that the read
can still separate each binary block language with at least two blocks and
a target below saturation. The gradient for each table entry sums only
the rows that reached its active count.
[Second-change representation and gradient](../../src/Transformer/Phasor/InclusiveTransitions.lean)

With `fp32-affine-reference,inclusive-transitions`, all three L2 seeds
reached 512/512 at update 250 and repeated that score on both source test
splits. Under the same source protocol, the closed-relation default also
reached 512/512 at update 250 in seeds 0 and 1. In seed 2 it scored
453/512 at 250 and 341/512 at every observation from 500 through 2,250.
Counting the current edge improved the two additional whole-sequence
tests over the lagged count in every seed.

| L2 test | Previous modular count, seeds 0 / 1 / 2 | Inclusive capped count, seeds 0 / 1 / 2 |
| --- | ---: | ---: |
| Every ten-letter word, out of 1,024 | 1,008 / 992 / 992 | 1,022 / 1,016 / 1,008 |
| Alternating tails at length 128, out of 64 | 63 / 62 / 62 | 64 / 64 / 63 |

The two extra tests crossed 99% in seeds 0 and 1. Elapsed training time
rose by 0.43% as a geometric mean across these three L2 pairs. The
FP32 forward agreed with the scalar reference within 8.77 × 10⁻⁷, and
all 146 directional derivatives passed. A six-token fixture distinguishes
the last edge from its predecessor, and a longer fixture checks the
absorbing bucket. Reproduce with `shared-check`, `cell language-l2 SEED RUN_DIR`
and `probes language-l2 SEED RUN_DIR` in the same build.

Enumerating all nonempty words through ten letters identified the shortest
L2 mistakes in seeds 0 / 1 / 2. With the lagged table they were
`abbbbb` / `abbbb` / `abbbb`. With the inclusive table they were
`abbbbbbbb` / `abbbbbb` / `abbbbb`. Each word was independently rejected
when it was the entire input and all output labels were masked.

The inclusive architecture also first classified all 9,409 NV11 pairs
correctly at update 750 in all three seeds. Independent reloads of those
checkpoints and the three update-3,000 checkpoints reproduced 4,704/4,704
training and 4,705/4,705 held-out pairs. The smallest held-out logit
margin across seeds was 0.578 at update 750 and 10.024 at update 3,000.
Use `shared-values` and `nv11-score` with the inclusive feature flag.

On L3, the inclusive count passed **every prefix of every ten-letter word**
and all 64 alternating-tail sequences at length 128 in all three seeds.
All six checkpoints in the comparison also scored 512/512 on each of the
original test splits.

| L3 measure | Previous modular count, seeds 0 / 1 / 2 | Inclusive capped count, seeds 0 / 1 / 2 |
| --- | ---: | ---: |
| Updates to 99% on length-128 validation | 500 / 500 / 750 | 500 / 500 / 500 |
| Every ten-letter word, out of 1,024 | 939 / 975 / 1,020 | 1,024 / 1,024 / 1,024 |
| Alternating tails at length 128, out of 64 | 56 / 57 / 64 | 64 / 64 / 64 |

Elapsed time fell by 10.30% as a geometric mean across the three L3 pairs.
The slowest inclusive run took 65.23 seconds, within the long-tier limit.
Against the closed-relation default on L3, the inclusive runs used
30.66% fewer updates and 25.59% less elapsed time as geometric means.
Use the same commands with `language-l3` and the two feature flags in the
table comparison. The 159-run historical result used the closed-relation architecture.

The longer transfer test evaluates every twelve-letter binary word and
all 256 eight-letter words after repeating each letter 32 times. The
second set has 256 payload letters per prompt. Lean proves that every
copy has the same target as the corresponding original prefix, even when
the repetition counts differ across letters.
[Prefix labels under repetitions](../../src/Transformer/Phasor/StutterTransfer.lean)

At twelve letters, the inclusive L3 model classified every prefix of all
4,096 words correctly in each seed. On the 257-token prompts, seed 2
missed one sequence. These tests use the same six checkpoints as the
ten-letter comparison above.

| L3 whole sequences correct | Previous modular count, seeds 0 / 1 / 2 | Inclusive capped count, seeds 0 / 1 / 2 |
| --- | ---: | ---: |
| Every twelve-letter word, out of 4,096 | 3,737 / 3,872 / 4,080 | 4,096 / 4,096 / 4,096 |
| Eight letters repeated 32 times each, out of 256 | 251 / 198 / 256 | 256 / 256 / 255 |

Reproduce from either build with
`PHASOR_EVAL_BATCH_CAP=4 "$bin" probes language-l3 "$seed" "$run_dir" extended`.
The scorer emits the SHA-256 of the saved weights alongside the results.

For a missed repeated-letter sequence, the scorer now reports the first
misclassified prefix in probe order. It checks the expanded prefix by itself
and then checks its original letters, with all output labels masked. The
inclusive seed-2 error occurred at payload letter 232 of `abaaaaaa` after
expansion. The model rejected that prefix in isolation and accepted the
eight original letters, as required by L3. The lagged seed-0 witness was
`abbaaaaa` at letter 248 and also failed without expansion. Its seed-1
witness was `aaaaabaa` at letter 252 and passed without expansion.

On the source L4 task, the same inclusive architecture reached 512/512 on
length-128 validation at update 250 in every seed. The slowest run took
40.53 seconds, within the short-tier limit. The checkpoints were trained
separately on L4 using the original optimizer. At the same update the
closed-relation default scored 109 / 116 / 124 out of 512. Training it to
update 750 gave 230 / 153 / 447. Every scheduled default observation
through that cutoff was below the source's 99% threshold.

| L4 whole sequences correct | Total | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: | ---: |
| Original-length test | 512 | 511 | 512 | 512 |
| Length-128 test | 512 | 512 | 512 | 512 |
| Every ten-letter word | 1,024 | 991 | 1,024 | 1,021 |
| Alternating tails at length 128 | 64 | 64 | 64 | 64 |
| Every twelve-letter word | 4,096 | 3,946 | 4,096 | 4,081 |
| Eight letters repeated 32 times each | 256 | 252 | 256 | 256 |

Use `cell language-l4 SEED RUN_DIR` and `probes language-l4 SEED RUN_DIR`
with the inclusive feature flag to reproduce the training and transfer tests.
The `extended` probe with `PHASOR_EVAL_BATCH_CAP=4` adds the last two rows.
In seed 0 its first repeated-letter failure was `abbbabbb` at payload
letter 254. The eight-letter prefix also failed on its own.

The modulated transition-table trial reads a capped count entry `x` and a
current-token entry `y` at every position. Each hidden coordinate contributes
`x + y + x*y` before normalization. Initializing the token table to zero
recovers the inclusive count read for any learned count table. Lean proves
that summing these coordinates can represent any scalar function of the two
keys using one coordinate per token. It also gives the exact finite-batch
gradient, including repeated keys and masked rows.
[Two-key representation and gradient](../../src/Transformer/Phasor/ModulatedTransitions.lean)

The modulated model passed 156 directional checks, including zero gradients
when the other factor cancels a coordinate. It reached 512/512 on L2 and
L3 validation at the same updates as the inclusive table in all three seeds.
L2 ten-letter scores were 1,020 / 1,008 / 1,016 out of 1,024, down in seeds
0 and 1 and up in seed 2. All L3 seeds passed every ten-letter word and
every length-128 alternating tail. MQAR again passed at initialization and
scored 64/64 at all three test lengths. Dyck took 500 / 250 / 750 updates,
against 500 / 250 / 500 for the inclusive table. We retained the inclusive
table for further testing. To repeat this comparison, apply the patch at
`978d5c7f`, build with `fp32-affine-reference,modulated-transitions`, and
run `shared-check`, `cell` and `probes` on the four source tasks.
[Modulated transition source](trials/modulated-transition-table.patch)

The geometric-memory trial learns one additional value for each token and
hidden coordinate. It scans every input sequence in order. Each coordinate
retains a fixed fraction of its preceding state and adds the current learned
value. The retentions cycle through 1, 1/2, −1 and −1/2. All values start
at zero. Lean proves the composition law across arbitrary sequence splits,
a bound on state magnitude, and the reverse-scan gradient for every token,
including repeated occurrences and masked outputs.
[Causal geometric memory and its adjoint](../../src/Transformer/Phasor/GeometricMemory.lean)

Adding geometric memory to the lagged transition table passed 135 GPU
gradient checks. It recovered all 64 MQAR test sequences at each of the
three test lengths in every seed. Dyck took 500 / 500 / 500 updates,
compared with 500 / 250 / 500 for the transition table alone. On seed 0,
typed Dyck and C-RASP depths two and three reached 23/32, 23/32 and
27/31 at the 2,500-update cutoff. Copy-repeat reached 0/32 at update
1,000. This trial supplied no new passing case on those screens. To
repeat them, apply the patch at `8e47e999`, build with
`fp32-affine-reference,geometric-memory` and run `shared-check` followed
by `screen TASK SEED RUN_DIR CUTOFF`.
[Geometric memory trial source](trials/geometric-token-memory.patch)

The affine-projection memory trial learns a slope and intercept for every
token and coordinate pair. At each position it projects the preceding
pair of hidden values onto the current token's line. Lean proves that
repeating any token preserves the auxiliary state, while changing token
order can change it. Zero intercepts recover the previous residual for
every input. The proof also derives the reverse update for the slopes,
intercepts and states.
[Affine projection memory](../../src/Transformer/Phasor/AffineProjectionMemory.lean)

The GPU passed 159 gradient checks and agreed with the independent FP64
forward within 1.10 × 10⁻⁶. Two differently ordered prompts produced
hidden states (0.5, −0.5) and (1, 0). Each L2 seed reached 512/512 at
update 250 and repeated that score on both official test splits. The
three checkpoints passed 1,016 / 1,008 / 992 of the 1,024 ten-letter
words and 64 / 63 / 62 of the 64 alternating tails. The inclusive
table scored higher on the ten-letter test in every seed. Affine training
was 0.40% faster as a geometric mean on L2. On typed Dyck, seed 0
scored 26/32 at the 2,500-update cutoff. We set this trial aside.
To reproduce, apply the patch at `462565fa`, build with
`fp32-affine-reference,affine-projection-memory` and use `shared-check`,
`cell language-l2`, `probes language-l2` and `screen suite-dyck-2`.
[Affine projection source](trials/affine-projection-memory.patch)

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

The native build reads the update budgets from the pinned GPT catalog. Its
3,000- and 10,000-update cases use the same schedule as the original 35-task
cohort. On hard recall, warmup lasts 60 updates and cooldown starts at
2,400. A comparison at every step of all five declared budgets reproduced
all 163,010 learning rates bit for bit, including the step after each budget.

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
cargo +stable build --locked --release --features fp32-affine-reference,prefix-xor
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

The native catalog runner also reads the frozen copy, reverse, sort,
typed-Dyck, counting, addition, frequency and remaining synthetic datasets.
The data crates supply the original selection split and generation limits.
On v0.6.0 the half-gain model scored 0/32 on unique-token copy at update
1,000, 22/32 on two-type Dyck at 250 and 1/31 on counting at 250 (seed 0).
Copy used greedy generation from the model's own previous tokens. All three
fell below the admission threshold.

```sh
cargo +stable build --locked --release --features fp32-affine-reference,prefix-successor-half
bin=./target/release/phasor-value-transport
export PHASOR_EVAL_BATCH_CAP=16
"$bin" screen suite-copy-unique 0 runs/catalog-half 1000
"$bin" screen suite-dyck-2 0 runs/catalog-half 250
"$bin" screen suite-count 0 runs/catalog-half 250
"$bin" catalog-report runs/catalog-half suite-histogram-bos suite-mode-none suite-most_freq
```

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

The half-gain model added `language-l1`, the one-block language
with no neutral tokens. It trained on lengths 32–64. At update 250 each
seed classified all 512 length-128 validation sequences and all 512 separate
length-128 test sequences correctly. The slowest fresh cell took 40.64 s,
qualifying this fourth addition for short.

The fifth short admission, `arithmetic-sum-pow8`, reached 99% at update 250
in all three seeds. Independent replays scored 4,673, 4,667 and 4,669 of
4,705 held-out pairs, including all 51 pairs in the held-out zero row in
every seed. The slowest fresh cell took 2.41 seconds. The source provides
one held-out arithmetic split.

The sixth short admission, `arithmetic-sum-interval`, predicts whether
`(a + b) mod 97` falls in 0–47. On Kernite v0.6.0 every seed classified
all 4,705 held-out pairs, including the 51 zero-row pairs, at update 250.
All three saved checkpoints replayed exactly. The slowest fresh cell took
2.34 seconds. The source provides one held-out split.

On `arithmetic-sum-pow12`, the seventh short admission, the three seeds
classified 4,662, 4,662 and 4,671 of 4,705 held-out pairs at update 250.
Independent replay recovered each count and all 51 zero-row pairs in every
seed. The slowest v0.6.0 cell took 2.36 seconds.

The eighth short admission is the GPT catalog's scratchpad-free parity task.
The training inputs contain 8–16 bits. With the selected 0.75-gain model,
all three seeds scored 32/32 on novel validation sequences at updates
500 / 250 / 500. Replaying the checkpoints gave 64 / 63 / 63 of 64 on the
ordinary test, 64/64 in every seed at 32 bits and 64 / 63 / 64 at 64 bits.
The slowest fresh cell took 5.51 seconds. The half-gain predecessor needed
5,250 / 1,000 / 750 updates on the same task.

The ninth short admission is the sixteenth power of a modular sum. Every
seed reached 99% at update 250. The saved checkpoints replayed at 4,705 /
4,703 / 4,705 of 4,705 held-out pairs, including all 51 zero-row pairs per
seed. The slowest selected-model cell took 2.35 seconds.

Depth-one C-RASP is the first long catalog addition. Each of the 32 novel
validation sequences requires a correct label at every supervised prefix.
The selected model passed all 32 at updates 750 / 9,500 / 250, and every
checkpoint reproduced that score in a fresh process. The slowest run took
84.68 seconds. Training lengths were 8–16 tokens. On separate 64-sequence
splits the checkpoints scored 61 / 64 / 62 at the original lengths,
60 / 63 / 54 at length 32 and 59 / 60 / 37 at length 64.
[C-RASP source protocol](../convex_stress_remaining_synthetic_data/README.md)

The noisy parity task corrupts training labels with a fixed 20% noise
kernel. All three seeds learned from the 512 observed examples and reached
128/128 on clean validation at update 250. All three fit 409/512 observed
training labels. Each independently reloaded model scored 256/256 on the
separate clean test. The slowest cell took 5.43 seconds.
[Noisy parity source protocol](../convex_stress_basis_data/README.md)

On the 24th power of the modular sum, all three seeds reached 4,704/4,705
on validation at update 250, including 51/51 on the held-out zero row.
Every checkpoint reproduced these counts, and the slowest run took
2.32 seconds. The half-gain predecessor crossed 99% at the same update
with 4,660/4,705 in each seed.

On `(a² + ab) mod 97`, all three seeds crossed 99% at 500 updates. The
half-gain predecessor took 750 in each seed. The selected checkpoints
replayed at 4,704 / 4,698 / 4,703 of 4,705, including every held-out zero
row. The slowest cell took 4.39 seconds. On the source predicate that
`(a + b) mod 97` is a nonzero quadratic residue, both models scored
4,705/4,705 at update 250 in every seed. The slowest selected cell took
2.34 seconds. A separate screen of `(a² + ab + b²) mod 97` scored
520/4,705 in seed 0 at 1,500 updates.

The selected binary lists all fourteen additions and reports each seed.

```sh
"$bin" extensions
cut -f2 extensions.tsv | xargs "$bin" catalog-report runs/v060-selected
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

Use DiffGPT at `5a37c515639f23037d191b0659a5a0853984328d` under
`.lake/reproduction/gpt`. The build checks the historical trainer revision
and its source tree. Cargo locks the published Kernite v0.6.0 crates. The
commands below use stable Rust 1.100.0. Run `cargo fetch --locked` once for
offline builds. [DiffGPT checkout instructions](../rerope/README.md#build-and-commands)

The native runner reconstructs the fingerprinted splits and uses the complete
DiffGPT scorer, including greedy generation from the model's own tokens.
Its criterion is the first scheduled observation with at least 99% exact
sequences on the primary held-out split. Each task and seed starts from fresh
weights. NV11 uses its original 4,704/4,705 half-table split and separate
250-step observation cadence. [Original 35-task protocol](../convex_stress/suite.json)

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.SharedValues
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,pair-v2
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
suite identity and exact held-out count. A replay of a previously completed
screen includes its original observation, update and pass status. A cutoff
before the first crossing returns the recorded score at that cutoff.
The report verifies the full observation
history, update cadence and examples consumed before counting a task and seed.
It lists budget misses and pending runs separately. For two-token validation examples
with first operand zero it also scores the fitted weights with the value
tables cleared. The contextual feature and transported read remain active
in that ablation. The two-token prediction command takes a weights file,
vocabulary size and two operands.

The optional final argument of `shared-values` sets the observation window
within the original 100,000-step schedule. Each saved observation records
the SHA-256 of its exact weight bytes and the shared architecture identifier.
`nv11-score` returns the file digest, scoring architecture and Kernite
version alongside the exhaustive scores. Match these fields against the
training observation to verify a saved checkpoint. The trainer also saves
the first checkpoint
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

The historical experiment used DiffGPT at
`5a37c515639f23037d191b0659a5a0853984328d` and Kernite 0.3.2 at
`a1b6bc2c956e1a5f35960e26b209e1b6c213096c`. To repeat it with that
backend, use transformer commit `18cda3d8` and the two local checkouts.
The current build reads the original optimizer, schedule, initialization,
Phasor blocks and loss from the same DiffGPT sources and uses Kernite v0.6.0.
[Historical checkout instructions](../rerope/README.md#build-and-commands)

From the repository root, check Lean before running the experiment.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TransportedValues
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,pair-v2
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
