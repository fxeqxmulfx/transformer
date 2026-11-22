# Embedding-input Phasor on the GPT stress suite and NV11

**Triple transport is the default Phasor architecture.** It solved all
4,704 training and 4,705 held-out NV11 pairs on seeds 0, 1 and 2 at update
250. The same model is trained separately on each GPT task, starting from
the original random embeddings with ordinary AdamW. [Triple transport](#triple-transport) · [Model and scoring rules](../../rule.md)

A partial audit of the default verified **46/75 short runs and 6/30 long
runs** alongside all three exact NV11 checkpoints. [Default GPT results](#triple-transport-on-the-canonical-gpt-tiers)

The input-phase comparison reached 75/75 short and 30/30 long runs. Its
best NV11 result at updates 250 and 3,000 was 90/4,705 held-out pairs.
[Input-phase GPT results](#canonical-gpt-stress-tiers)

To train and independently score the default model on NV11 and both GPT
tiers, run the following from the repository root. The combined report
gives a count for each tier. The NV11 check requires exact recovery of
every training and held-out pair.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference
bin=./target/release/phasor-value-transport
out=runs/main-triple
export PHASOR_EVAL_BATCH_CAP=16
"$bin" check
./run_stress.sh cell "$out"
"$bin" report "$out" > "$out/report.json"
./run_stress.sh replay "$out" > "$out/replays.jsonl"
for seed in 0 1 2; do
    "$bin" nv11-score "$out/arithmetic-cubic-product/seed$seed/weights.f32"
done > "$out/nv11-scores.jsonl"
jq -es --slurpfile report "$out/report.json" --slurpfile nv11 "$out/nv11-scores.jsonl" \
    -f verify_replays.jq "$out/replays.jsonl"
```

A compact Fourier read used 49,985 read parameters and reached at most
100/4,705 held-out NV11 answers over three 10,000-update trials.
[Fourier measurements and ablations](#learned-fourier-transport)

## Canonical GPT stress tiers

The original suite has 25 short and ten long tasks. Each task trains fresh
weights for seeds 0, 1 and 2. A run passes at the first scheduled
observation with at least 99% exact sequences on the primary held-out
split. An independent process then reloads the weights and scores the same
split. The following commands run the input-phase Phasor on all 105 cells.
It learns its phase increments from normalized random input embeddings
and uses the same causal read and AdamW schedule rules on every task.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.InputPhaseRead
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,input-phase-read
export PHASOR_EVAL_BATCH_CAP=16
out=runs/input-phase-gpt
bin=./target/release/phasor-value-transport
"$bin" check
./run_stress.sh --gpt-only cell "$out"
"$bin" gpt-report "$out" > "$out/gpt-report.json"
./run_stress.sh --gpt-only replay "$out" > "$out/gpt-replays.jsonl"
jq -es --slurpfile report "$out/gpt-report.json" -f verify_replays.jq "$out/gpt-replays.jsonl"
./run_stress.sh --gpt-only probes "$out" > "$out/gpt-tests.jsonl"
jq -es --slurpfile report "$out/gpt-report.json" -f verify_replays.jq "$out/gpt-tests.jsonl"
```

For an extended cohort, pass the same two-column tier and case TSV to
the `run_stress.sh` commands and to `gpt-report` after the run directory.
The report counts the original and additional tasks separately. Each
`gpt-tests.jsonl` row identifies the checkpoint, its first passing update
and its primary score. Its `test_splits` object gives exact counts and
denominators for the available independent test and transfer splits.
For a partially completed extension cohort, the following command requires
a replay for every reported passing seed and reports pending and exhausted
seeds separately. It checks the first passing update as well as the model,
weights and exact counts.

```sh
"$bin" gpt-report "$out" extensions.tsv > "$out/gpt-report.json"
jq -r '.tasks[] as $t | $t.seeds[] | select(.status == "pass") | [$t.case,.seed] | @tsv' \
    "$out/gpt-report.json" |
    while read -r task seed; do "$bin" probes "$task" "$seed" "$out"; done > "$out/gpt-tests.jsonl"
jq -es --argjson passing_only true --slurpfile report "$out/gpt-report.json" \
    -f verify_replays.jq "$out/gpt-tests.jsonl"
```

For a broader catalog case scored by exact sequence accuracy or numeric MSE,
`catalog-target CASE SEED SEARCH_UPDATES RUN_DIR` uses the target and
post-crossing window from the frozen source plan. For example, a case with
a 95% target and 500 additional updates saves the first 95% checkpoint
and continues to the first scheduled observation at least 500 updates
later, capped by the source budget. A search window can end before the
full budget if no crossing has occurred. Repeating the command reloads
and scores the first and final weights, including available test splits.
Training uses Phasor's original WSD AdamW throughout.

For the four numeric mechanism probes, the target is MSE at most 0.001
on freely generated outputs. The scorer compares each decoded component
with the original unquantized binary64 teacher and charges squared error
64 for a missing or invalid component. The report includes exact sequences
and valid numeric coverage alongside the source metric. The independent
audit runs the original scalar scorer on the checkpoint, requesting logits
for one prefix at a time.

```sh
"$bin" catalog-inventory > "$out/catalog-source-inventory.json"
"$bin" catalog-target gpt-probe-conditional-amplitudes 0 4500 "$out"
weights="$out/phasor-learned-input-phase-32-v1/gpt-probe-conditional-amplitudes/seed0-4500/declared-target/weights.f32"
"$bin" numeric-audit gpt-probe-conditional-amplitudes "$weights"
```

The inventory checks native source adapters for every catalog entry and
records its declared metric, target, budget and cohort. A completed
canonical run and availability under the broader source protocol are
separate fields in the respective reports.

Of the 108 catalog cases outside the original tiers, the 21 registered
extensions and NV11, 56 have authenticated data and a supported source
metric in this runner. For 28 more sequence-accuracy cases the required
data adapter is unavailable. The other 24 need a different source scorer
or an authenticated numeric teacher. The inventory names each case and
gives its reason.

To report measured source coverage, replay each saved search with the same
executable. The summary counts the longest measured search for each case
and seed. It includes first and final scores on every available split and
distinguishes retention, loss of the target, a full-budget miss and an
earlier cutoff. Available cases with no measurements remain visible.

```sh
model=phasor-learned-input-phase-32-v1
rg --files --no-ignore --hidden -g result.json "$out/$model" |
    rg '/declared-target/result\.json$' | sort |
    while IFS= read -r record; do
        read -r task seed limit < <(jq -r '[.protocol.case,.protocol.seed,.protocol.updates] | @tsv' "$record")
        "$bin" catalog-target "$task" "$seed" "$limit" "$out"
    done > "$out/catalog-replays.jsonl"
jq -es --slurpfile inventory "$out/catalog-source-inventory.json" \
    -f catalog_results.jq "$out/catalog-replays.jsonl" > "$out/catalog-results.json"
```

On branch amplitudes at seed 0, the first MSE crossing came at update
1,000. Free generation produced 255/256 exact validation sequences and
MSE 0.0004882813 on 8,192 components. At update 1,500 the independent
test and validation each scored 256/256 with MSE approximately
2.22 × 10⁻¹⁶. This case also belongs to the canonical tiers.
Routing mixture has no Phasor MSE result because the reconstructed
unquantized teacher differs from the frozen fingerprint.

The catalog data adapter authenticates all 54 original arithmetic cases,
including the fixed random table, four swapped intervals and random
half-set. Each reconstructed train and validation tensor matches its
original hash before examples reach the experiment.
[Arithmetic data and source registry](../convex_stress_arithmetic_data/README.md)

On modular addition at seed 0, the broader plan's 250-update cadence
first observed 99% at update 1,000 (4,704/4,705). The retention checkpoint
at 1,500 scored 4,705/4,705. Both weight digests matched independently
trained fixed-budget runs ending at the respective update. The canonical
addition case observes every 19 updates.

The input-phase model passed **75/75 short runs and 30/30 long runs**.
All 105 first-crossing checkpoints independently reproduced their held-out
counts in Kernite v0.6.0. Every canonical case had a supported scorer and
every seed crossed within the original update budget. The short tier
ranged from 68 updates on shifted AND to 2,014 on the modular square of
the sum. The long tier ranged from 589 on associative division to 15,800
on the Fibonacci recurrence.

On the GTX 1050, median elapsed time to the first passing observation was
10.25 seconds for short runs and 30.92 seconds for long runs. All 30 long
runs finished within five minutes. The three hard-depth seeds were the
only original short runs over one minute, taking 162 / 212 / 114 seconds.
Tier membership follows the frozen suite.

| Task | Held-out sequences | First crossing, seeds 0 / 1 / 2 | Correct at crossing, seeds 0 / 1 / 2 |
| --- | ---: | --- | --- |
| Hard depth, length 128 | 512 | 1,500 / 2,000 / 1,000 | 511 / 512 / 512 |
| Easy key–value recall | 512 | 2,350 / 1,450 / 900 | 509 / 508 / 509 |
| Blinded absolute difference | 4,705 | 4,750 / 6,460 / 4,123 | 4,659 / 4,659 / 4,660 |
| Fourth power of the modular sum | 4,705 | 3,648 / 4,275 / 3,743 | 4,659 / 4,663 / 4,665 |
| Fibonacci recurrence, first differences | 4,704 | 12,360 / 15,800 / 14,570 | 4,657 / 4,659 / 4,662 |

Eleven tasks also supplied an independent test set. Of the 33 checkpoints
trained on these tasks, 25 reached 99% on that test. Across all 78
available test and transfer scores, 53 reached 99%. Hard depth scored
512/512 at length 128 on every independent test after training on lengths
32–64. On shifted AND, all three seeds scored 7/7 on the primary split.
The independent test scores were 64 / 27 / 64 of 64 and length-64 scores
were 0 / 35 / 64 of 64. The per-seed `gpt-tests.jsonl` above includes
every first crossing, denominator and available test split.

At length 256, all three hard-depth checkpoints still classified every
prefix correctly on all 512 test sequences. Increasing hard depth to
length 512 gave 341 / 342 / 348 correct sequences out of 512. Easy depth scored
507 / 512 / 295 out of 512 at length 256. Both tasks trained on
lengths 32–64. The depth probe rebuilds the original length-128 test and
checks its tensor fingerprint before generating the requested length.
On the long-tier easy recall task, inserting 64 filler tokens between the
writes and reads reduced exact sequences to 230 / 280 / 171 of 512.
The same checkpoints answered 3,722 / 3,813 / 3,603 of the 4,096
individual queries correctly.

```sh
for seed in 0 1 2; do
    PHASOR_EVAL_BATCH_CAP=2 "$bin" probes basis-depth-hard "$seed" "$out" depth-length 256
    PHASOR_EVAL_BATCH_CAP=2 "$bin" probes basis-depth-easy "$seed" "$out" depth-length 256
    PHASOR_EVAL_BATCH_CAP=2 "$bin" probes basis-recall-easy "$seed" "$out" recall-gap 64
done
```

On the separate NV11 regression, the same input-phase architecture
scored 52 / 50 / 53 of 4,705 held-out pairs at update 250 and
83 / 90 / 89 at update 3,000. The triple-transport reference solved
all 4,705 at update 250 on every seed.

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
cargo +stable build --release --locked --features fp32-affine-reference,learned-stream-read
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

For the original fixed-mean Phasor control, replace `learned-stream-read`
with `original-stream-mean` and use a separate run directory. Both executables
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

On fifteen source tasks the learned stream readout and original fixed
mean each produced 45 independently replayed first-crossing checkpoints.
The table counts all three fixed seeds for every named task. The fixed
mean also passed all three L4 seeds, giving 48 verified checkpoints over
these sixteen tasks.

| Source tier | Tasks passing all three seeds | Verified per architecture | Geometric mean time to first pass, learned / mean |
| --- | --- | ---: | ---: |
| Short | Shifted AND, random AND, modular addition, subtraction, multiplication, maximum, key/value binding, easy depth, blocks, Dyck, language L1 | 33 | 1.0321 |
| Long | Absolute difference, depth-one C-RASP, languages L2 and L3 | 12 | 1.0365 |

This timing comparison covers the named paired passes. On modular absolute
difference, the stream model crossed at 3,116 / 1,577 / 2,109 updates and
the original mean at 3,496 / 1,634 / 2,014. On 16-bit parity, all six
screens missed 99% through 3,500 updates. On 20-bit parity the six
screens also missed 99% through 3,500.

On depth-one C-RASP, seed 2 reached 32/32 validation at update 250 with
both readouts. Independent reloads of both scored 64/64 on the source
test, 56/64 at length 32 and 41/64 at length 64. On NV11 the learned
readout reached 4,696 / 4,692 / 4,703 of 4,704 training pairs at update
3,000. Reloading those weights gave 86 / 74 / 84 of 4,705 held-out pairs.

Seed 0 also ran the full original 100,000-update AdamW schedule on
`a³ + ab mod 97`. It first fit all 4,704 training pairs at update 3,250.
The best held-out count among 401 observations was 96/4,705 at update
15,250. Reloading the final checkpoint reproduced all 4,704 training
pairs and 90/4,705 held-out pairs (1.91%). This run failed the exact NV11
test.

```sh
"$bin" nv11-train runs/learned-nv11-full/seed0 0 100000
"$bin" nv11-score runs/learned-nv11-full/seed0/nv11-train-100000.f32
```

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

### Language L3 and L4

The learned stream readout also passed the existing long-tier L3 task
on all three seeds. On L4 it passed seeds 0 and 2 at updates 750 and
1,000. Seed 1 scored 355/512 at update 2,250.

The fixed-mean embedding-input control qualified L4 for the long tier.
Its slowest first crossing took 212.59 seconds. Each reloaded checkpoint
scored 512/512 on the source test and the independent length-128 test,
and 256/256 on the stretched eight-letter words. Training used lengths
32–64. The table gives the validation counts and the harder exhaustive
twelve-letter scores for both three-seed passes.

| Task and model | First ≥99% update, seeds 0 / 1 / 2 | Validation, out of 512 | Twelve-letter test, out of 4,096 |
| --- | --- | --- | --- |
| L3, learned stream read | 1,000 / 1,500 / 500 | 512 / 508 / 510 | 4,096 / 4,024 / 3,516 |
| L4, fixed mean | 1,000 / 2,000 / 1,000 | 511 / 512 / 512 | 4,096 / 4,016 / 4,096 |

For L4, build with `fp32-affine-reference,original-stream-mean` and run
the L2 commands above with `language-l4` and a fresh run directory.

### Short-tier addition from the maximum task

On GPT's `arithmetic-max`, the learned stream readout classified the
larger of two inputs over the 97-symbol vocabulary. All three training
runs fit 4,704/4,704 pairs. The first 99% checkpoints reloaded with
4,658 / 4,661 / 4,659 correct on the 4,705 held-out pairs, at updates
4,750 / 2,500 / 3,250. The slowest run took 37.11 seconds. The source
provides one held-out split for this task. The fixed mean scored
4,659 / 4,658 / 4,659 at updates 4,500 / 2,500 / 2,250.

```sh
for seed in 0 1 2; do
    PHASOR_EVAL_BATCH_CAP=16 "$bin" screen arithmetic-max "$seed" runs/learned-streams 6500
    "$bin" probes arithmetic-max "$seed" runs/learned-streams
done
"$bin" catalog-report runs/learned-streams arithmetic-max
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

All three bounded checkpoints reached 28/28 on novel GPT blocks at
update 750. Reloaded checkpoints scored as follows in seed order.

| Architecture | First 28/28 update | Source test, out of 64 | Length 64, out of 64 |
| --- | ---: | ---: | ---: |
| Bounded recurrent read | 750 / 750 / 750 | 63 / 63 / 64 | 20 / 19 / 24 |
| Learned stream read | 1,500 / 1,000 / 1,000 | 63 / 60 / 63 | 15 / 12 / 22 |
| Original mean | 750 / 750 / 1,000 | 60 / 57 / 63 | 14 / 10 / 26 |

Depth-one C-RASP seed 0 reached 31/32 by update 2,250 with the bounded
read. Both controls had reached 32/32 at update 750. We stopped the
broader bounded trial after this regression. On NV11 at update 3,000
the bounded checkpoints classified 95 / 86 / 101 of 4,705 held-out
pairs correctly. Extending seed 0 to 12,000 updates gave 85/4,705 for
the bounded read and 89/4,705 for the matched original mean. Their
training counts were 4,699/4,704 and 4,702/4,704.

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

On two six-position sequences an independent FP64 calculation agreed with
the GPU input residual within `1.25e-6` at widths 64 and 128. All twelve
finite differences across the embedding, router and four projections
passed. Isolating each prefix produced identical logits. Three additional
directions checked gradients with respect to dense vectors outside the
embedding table.

Five seed-0 comparisons reproduced the first validation pass after
checkpoint reload. The causal product used fewer updates on blocks and
absolute difference, and more on addition, binding and depth-one C-RASP.

| Task | Causal product, first ≥99% update | Learned stream control |
| --- | ---: | ---: |
| Blocks | 750 | 1,500 |
| Absolute difference | 2,926 | 3,116 |
| Modular addition | 912 | 874 |
| Key/value binding | 512 | 448 |
| Depth-one C-RASP | 1,250 | 750 |

At length 64 the product scored 24/64 on blocks and 58/64 on C-RASP.
The stream control scored 15/64 and 60/64. Three independently reloaded
product checkpoints on NV11 at update 3,000 scored 99 / 79 / 86 out of
4,705 held-out pairs. The broader trial stopped after the mixed results.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.CausalProductInput
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,causal-product-input
bin=./target/release/phasor-value-transport
"$bin" memory-check
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen suite-blocks 0 runs/causal-product 1500
"$bin" probes suite-blocks 0 runs/causal-product
"$bin" nv11-train runs/product-nv11/seed0 0 3000
"$bin" nv11-score runs/product-nv11/seed0/nv11-train-3000.f32
```

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

The compact initializer draws the same full-width weights and retains the
corresponding complex coordinates in every layer. On two-layer models
of widths 64, 128 and 256, the compact executable and its zero-padded
full-width counterpart produced bit-identical final hidden states and
logits on arbitrary embedding vectors. Their input gradients and every
parameter gradient of masked cross-entropy also matched exactly. At
width 128 the two-layer model removes 33,024 independently optimized
weights. The optimizer and its schedule are shared with the full model.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.CompactFFN
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,compact-ffn
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" cell arithmetic-add 0 runs/compact-streams
"$bin" probes arithmetic-add 0 runs/compact-streams
```

For a fixed-work timing comparison, build each executable from the same
checkout and run six pairs per task. Each seed runs twice, with the
execution order reversed. The timer includes process startup, training
and scheduled validation. It checks the source fingerprint, GPU, precision
settings, Kernite version and number of examples at every observation.
Choose a cutoff before either architecture reaches the target.

At 380 updates, two independently started series each compared six pairs
per task on the GTX 1050. Compact took less elapsed time in all 24 pairs.
At a 19-update cutoff on addition, the six-pair ratio was 1.0001.

| Task | Compact / full elapsed time, first series | Second series |
| --- | ---: | ---: |
| Modular addition | 0.9755 | 0.9750 |
| Absolute difference | 0.9764 | 0.9755 |

The ratios are geometric means of paired elapsed times. Addition reached
its first pass at 912 / 950 / 969 updates with compact and 874 / 950 / 931
with the full model. On depth-one
C-RASP all six checkpoints independently replayed at 32/32. Compact
crossed at updates 250 / 1,250 / 250 and the full model at 750 / 1,500 / 250.
At length 64 they scored 55 / 56 / 40 and 60 / 61 / 41 out of 64,
respectively.

```sh
# From experiments/phasor_value_transport
CARGO_TARGET_DIR=target/wide cargo +stable build --release --locked --features fp32-affine-reference,learned-stream-read
CARGO_TARGET_DIR=target/compact cargo +stable build --release --locked --features fp32-affine-reference,compact-ffn
PHASOR_EVAL_BATCH_CAP=16 ./time_screens.sh target/wide/release/phasor-value-transport \
    target/compact/release/phasor-value-transport runs/compact-timing 380 arithmetic-add arithmetic-abs-diff
```

## Learned phase memory

At each position the backbone produces a contextual hidden vector `h[t]`
with the learned stream readout. A shared learned matrix maps this vector
to 32 phase increments. Each sequence starts at zero phase and adds its
increments through the current position. Two learned matrices project the
sine and the centered cosine of these phases back into the hidden vector.
Both output matrices start at zero. The tied classifier reads the resulting
vector. The projection dimensions and operations are the same on every task.

Lean shows that an earlier phase difference keeps the same chord length
after any common sequence of increments. It also derives the reverse sum
that sends each read's gradient to earlier increments. The entire
embedding-input backbone remains causal at every depth.
[Phase memory proofs](../../src/Transformer/Phasor/PhasePrefixRead.lean)

At width 128 the phase projections add 12,288 learned weights to the
stream model. On three sequences of length 257, independent scalar FP64
sums checked 49,344 output values and 24,672 increment gradients. The
largest errors were `4.54e-6` and `2.38e-5`. Starting from the backbone's
contextual vectors, the FP64 phase read and classifier reproduced the GPU
logits within `4.22e-7` at widths 64 and 128.
All twelve masked-loss gradient checks passed, including separate directions
for the sine and cosine output matrices. Finite differences used three step
sizes to check convergence. Isolated prefixes matched exactly. Three further
directions checked derivatives with respect to arbitrary input vectors.

On the original long-tier 16-bit Basis parity task, all three first-crossing
checkpoints scored 157/157 on validation and 512/512 on the independent
source test after reload. Evaluation generated the answer bit and end token.
Neither the learned stream readout nor the fixed-mean control reached 99%
on any seed through the matched 3,500-update screen.

| Seed | Phase model, first 157/157 update | Stream / mean validation at 3,500, out of 157 | 32-bit test, out of 64 | 64-bit test, out of 64 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,500 | 154 / 143 | 64 | 15 |
| 1 | 750 | 154 / 155 | 45 | 34 |
| 2 | 1,250 | 151 / 130 | 61 | 32 |

Extending the same 16-bit training runs to 8,000 updates gave the mean
control three crossings at 5,250, 7,250 and 8,000. The learned stream
control crossed at 7,250 and 8,000 on seeds 0 and 2. Seed 1 still scored
154/157 at 8,000. The phase model's slowest crossing was at 1,500.

Training the same phase architecture on the original long-tier 20-bit
task gave 284/284 on all three validation splits at updates 7,000, 750
and 1,000. The saved checkpoints generated 512, 512 and 511 correct
answers out of 512 source test examples. Their 64-bit transfer scores
were 53, 16 and 0 out of 64. Seed 0 first crossed after the initial
3,500-update screen.

The same phase architecture, trained separately on modular addition,
passed seed 0 at update 874, matching the stream control. On depth-one
C-RASP it scored 31/32 at the 1,500-update cutoff. The stream control
had passed that seed at update 750.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.PhasePrefixRead
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,phase-prefix-read
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen basis-parity16 0 runs/phase-streams 3500
"$bin" probes basis-parity16 0 runs/phase-streams
```

## Input phase memory

This variant projects each RMS-normalized input embedding into the 32 phase
increments. The sine and centered cosine of their running sums join the
contextual stream vector after the final layer. The weights and optimizer
have the same shapes as the contextual phase model.

Lean proves that a segment contributes the same phase increment in every
preceding context. Swapping two input segments preserves the phase at their
end, and repeating any embedding adds its learned increment once per copy.
The contextual branch retains its sensitivity to sequence order. Both
branches are causal, and their gradient contributions add at the input.
[Input phase proofs](../../src/Transformer/Phasor/InputPhaseRead.lean)

On the original short task `arithmetic-square-sum`, the target is
$(a^2+b^2)\bmod 97$. The input-phase model reached 99% at updates
1,102 / 1,159 / 1,121, scoring 4,661 / 4,663 / 4,678 of 4,705
held-out pairs. The learned stream took 1,235 / 1,292 / 1,178 updates,
the fixed mean 1,235 / 1,292 / 1,216 and the bilinear phase model
1,178 / 1,235 / 1,178. All four architectures passed on all seeds.

Two independent timing series compared the input-phase and learned
stream models through their first passing observation on the GTX 1050.
Each series trained each seed twice in opposite execution orders. The
input-phase model finished first in all twelve pairs. Its geometric
mean elapsed time was 93.77% and 94.01% of the stream model's time in
the two series. Each of the 24 timed checkpoints was reloaded and
rescored outside the timer. The timer includes process startup, training
and source validation. To repeat one six-pair series, build both
executables from this checkout with Kernite v0.6.0.

```sh
# From experiments/phasor_value_transport
CARGO_TARGET_DIR=target/stream cargo +stable build --release --locked --features fp32-affine-reference,learned-stream-read
CARGO_TARGET_DIR=target/input cargo +stable build --release --locked --features fp32-affine-reference,input-phase-read
PHASOR_EVAL_BATCH_CAP=16 ./time_screens.sh --first-pass target/stream/release/phasor-value-transport \
    target/input/release/phasor-value-transport runs/square-time 1520 arithmetic-square-sum
```

The native forward projects the input RMS once per position. Its reverse
pass adds the phase gradient to the backbone gradient before differentiating
that RMS. At widths 64 and 128, an independent FP64 calculation starting
from the normalized inputs and contextual vectors reproduced the GPU logits
within `8.31e-7`. All twelve directional checks passed. Swapping two
unequal segments of arbitrary dense vectors changed the isolated endpoint
phase read by at most `2.99e-7` and its reordered input gradient by at
most `2.99e-8`.

On 16-bit Basis parity, all three input-phase checkpoints generated the
correct bit and end token for all 157 validation and 512 source test
examples. All three solved the 32-bit catalog probe. Moving the increment
projection to the input improved 64-bit accuracy on every paired seed.
Each catalog probe has 64 examples, so seed 0's 51/64 misses the 99% target.

| Seed | Input phase, first 157/157 update | 32-bit correct | 64-bit correct | Contextual phase, 64-bit correct |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,750 | 64 | 51 | 15 |
| 1 | 1,000 | 64 | 64 | 34 |
| 2 | 1,000 | 64 | 64 | 32 |

Training the same architecture separately on the original long-tier 20-
and 24-bit tasks gave perfect validation, source test and 32- and 64-bit
catalog scores on all three seeds. The catalog probes are the same 64
fixed prompts at each length for every checkpoint.

| Training task | First passing updates, seeds 0 / 1 / 2 | Validation per seed | Source test per seed | 32-bit / 64-bit per seed |
| --- | --- | ---: | ---: | ---: |
| Basis parity, 20 bits | 1,000 / 1,250 / 1,000 | 284/284 | 512/512 | 64/64 and 64/64 |
| Basis parity, 24 bits | 1,000 / 1,000 / 1,000 | 306/306 | 512/512 | 64/64 and 64/64 |

At the first passing checkpoint we separately zeroed the learned output
weights on the sine channels, the centered cosine channels, and both.
Every other weight, embedding and prompt stayed fixed. Each cell below
gives exact sequences for seeds 0 / 1 / 2.

| Task and split | Trained model | Sine removed | Cosine removed | Both removed |
| --- | --- | --- | --- | --- |
| Modular addition, 4,705 held-out pairs | 4,677 / 4,689 / 4,665 | 4,681 / 4,664 / 4,654 | 4,665 / 4,655 / 4,633 | 4,656 / 4,621 / 4,610 |
| 20-bit parity, 284 novel validation prompts | 284 / 284 / 284 | 284 / 224 / 284 | 151 / 284 / 133 | 151 / 136 / 133 |
| 20-bit parity, 64 length-64 prompts | 64 / 64 / 64 | 55 / 38 / 32 | 39 / 64 / 25 | 35 / 25 / 25 |

The phase read contributed to modular addition on each seed. Parity
seeds 0 and 2 relied on cosine for the training-length validation split,
while seed 1 relied on sine. The apparently dispensable sine projection
on seeds 0 and 2 still contributed to length-64 transfer. Removing both
projections reproduced the trained stream backbone on every scored split.
On arbitrary dense inputs it also reproduced every hidden coordinate and
logit bit for bit. These interventions measure reliance after training.
Testing whether a simpler model can learn the same computations requires
retraining.

The contextual phase control on 24-bit parity crossed at updates 500 and
750 on seeds 1 and 2. Seed 0 scored 184/306 even at 8,000 updates.
The input-phase model also passed two original short tasks on all seeds.
Modular addition crossed at updates 874/950/893 with 4,677/4,689/4,665
correct out of 4,705. Key–value binding crossed at 704/576/576 with
507/508/507 correct out of 512 validation examples. Each binding
checkpoint scored 506/512 on the independent test.

Depth-one C-RASP reached 30/32, 32/32 and 32/32 through update 1,500.
On NV11 at 3,000 updates, exact checkpoint replay gave 4,704/4,704/4,701
correct training pairs out of 4,704 and 83/90/89 held out of 4,705.

The optional length extension generates 256 distinct test prompts each
at 128 and 256 bits using the pinned scratchpad-free parity generator.
A separate implementation counts the bits and checks every answer and
end token. The scorer includes a fingerprint for each split. These two
supplementary splits are outside the published 32- and 64-bit catalog.
The first-crossing checkpoints below are the same ones used above.

| Training length | 128-bit correct, seeds 0 / 1 / 2 | 256-bit correct, seeds 0 / 1 / 2 |
| ---: | ---: | ---: |
| 16 | 0 / 256 / 256 | 248 / 148 / 240 |
| 20 | 198 / 256 / 255 | 3 / 0 / 256 |
| 24 | 126 / 41 / 256 | 55 / 250 / 238 |

All rows have 256 examples per seed and length. None of these training
lengths achieved a three-seed 99% crossing on either longer test.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.InputPhaseRead
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,input-phase-read
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen basis-parity16 0 runs/input-phase 3500
"$bin" probes basis-parity16 0 runs/input-phase
PHASOR_EVAL_BATCH_CAP=4 "$bin" probes basis-parity16 0 runs/input-phase extended
for name in arithmetic-add basis-parity20; do
    for seed in 0 1 2; do
        PHASOR_EVAL_BATCH_CAP=16 "$bin" phase-ablate "$name" "$seed" runs/input-phase
    done
done
```

## Affine phase memory

Each normalized input embedding supplies two learned scalars per channel,
an angle and a write. The channel starts at the complex value one. It
rotates the previous value by the angle, adds the write to the real
coordinate, and exposes the imaginary and centered real coordinates to
the same two learned output maps. The projected residual joins the final
contextual stream. The write matrix starts at zero.

Lean derives an equivalent prefix-sum algorithm and the derivative with
respect to angle, write and previous state. Zero writes recover the input
phase feature for arbitrary embeddings. Active writes can distinguish the
order of two inputs, and the distance between two states survives any
common suffix. [Affine phase proofs](../../src/Transformer/Phasor/AffinePhaseRead.lean)

The native scan uses three real accumulators per channel. A separate FP64
oracle applies each rotation and write directly, and differentiates each
output separately through every earlier step. At length 257 the largest
errors over 49,344 reads and 49,344 derivatives were `2.20e-5` and
`7.16e-5`. With zero writes and active output projections, the input-phase
control reproduced every hidden value and logit exactly at widths 64 and
128. Its embedding gradients agreed within `5.97e-8`. With active writes,
swapping unequal dense input segments changed the isolated endpoint read
by 1.31 and 1.95 at the two widths. All fourteen masked-loss parameter
directions passed, including separate angle and write directions.

At the matched 1,500-update depth-one C-RASP cutoff, the affine model
reached 32/32 on all three seeds at updates 750/500/250. The input-phase
control reached 30/32 on seed 0 and passed at 500/250 on the others.
The affine checkpoints scored 64/63/63 out of 64 on the source test
and 42/48/53 out of 64 on the length-64 transfer set.

On 20-bit parity the affine model passed all three seeds at 1,500/1,750/1,500
updates. Every checkpoint scored 284/284 on validation, 512/512 on the
source test and 64/64 at 32 bits. At 64 bits the scores were 64/41/64.
The input-phase control passed in fewer updates on each seed and scored
64/64 throughout. Short key–value binding also crossed on all three seeds
at 640/704/704 updates. Its independent test scores were 502/505/505 out
of 512. On canonical NV11 at 3,000 updates the affine model scored
4,702/4,700/4,701 out of 4,704 training pairs and 90/83/89 out of 4,705
held-out pairs. Every score in this comparison came from saved checkpoints
except the input-phase C-RASP cutoff that missed its threshold.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.AffinePhaseRead
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,affine-phase-read
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen suite-crasp-depth-1 0 runs/affine-phase 1500
"$bin" probes suite-crasp-depth-1 0 runs/affine-phase
```

## Bilinear phase memory

Two learned projections of the normalized input supply a left and a right
vector. Each position multiplies its right vector by the sum of all earlier
left vectors, coordinate by coordinate. The running sum of these products
adds an ordered interaction to the phase. A separate learned projection
of the current embedding can change the angle used in that position's
read. All positions share the same projections.

Combining two segments adds their individual ordered sums and one cross
term, the product of the first segment's left sum and the second segment's
right sum. Lean proves this identity, the exact second-order perturbation,
prefix causality and the one-step derivative. Initializing the right and
local read projections to zero recovers the input-phase model at any depth.
[Bilinear phase proofs](../../src/Transformer/Phasor/BilinearPhaseRead.lean)

At length 257 a separate FP64 sum over ordered pairs reproduced 49,344
GPU reads within `1.92e-5`. Differentiating every read independently
reproduced 98,688 projected-input derivatives within `7.43e-4`.
At widths 64 and 128 the full-network scalar reference matched logits
within `5.12e-7`. Zero-initialized contributions gave identical off-table
hidden values, logits and embedding gradients to the input-phase control.
All eighteen masked-loss parameter directions passed, including distinct
directions for left, right and transient projections.

For the combined embedding direction at width 64 and length nine, the
initial finite difference was 5.220870. Halving the step twice gave
5.138516 and 5.116463. The two extrapolations were 5.111065 and 5.109111
against the analytic 5.110137, using the convergence criterion of the
separate phase-gradient checks.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.BilinearPhaseRead
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,bilinear-phase-read
bin=./target/release/phasor-value-transport
"$bin" check
"$bin" nv11-train runs/bilinear-phase-nv11/seed0 0 3000
"$bin" nv11-score runs/bilinear-phase-nv11/seed0/nv11-train-3000.f32
```

## Rotation and reflection memory

Both states in a channel use one learned projection of the normalized
embedding. A rotation adds its angle to the previous angle. A reflection
subtracts the previous angle from the new angle. Both states start at zero.
One learned scalar per channel mixes the two sine coordinates and the two
centered cosines before the ordinary output maps. The gate starts at one
half. The model adds 32 scalars to the input-phase model at every width.

Any two adjacent copies of an arbitrary embedding cancel in the reflected
state, including after an arbitrary common suffix. The rotation state
continues to accumulate the same two inputs. Lean proves this insertion
identity, the signed derivative through a suffix, prefix causality, and
the derivatives for both angles and the shared gate. Zero gates recover
the input-phase model for all embeddings and all output weights. Two
even-length segments can also exchange places without changing either
endpoint phase. Each acts on the incoming reflected angle by a translation.
[Rotation and reflection proofs](../../src/Transformer/Phasor/ReflectionPhaseRead.lean)

At length 257 the Kernite scan matched an independent FP64 alternating
sum within `3.05e-6` on 49,344 read coordinates. The largest errors on
24,672 input derivatives and 32 shared gate derivatives were `1.18e-5`
and `3.37e-5`. At widths 64 and 128, inserting an off-table vector twice
changed the reflection-only endpoint by at most `2.99e-7`. The two
derivatives with respect to the inserted copies summed to zero within
`1.50e-8`. Rebuilding the input-phase control reproduced its 20-bit
seed-0 checkpoint byte for byte.

The 16-bit reflection run reached the source validation threshold at
500 updates on each seed. The 20-bit run reached it at 750 / 500 / 500.
Every selected checkpoint then scored 64/64 on both catalog lengths
and 256/256 at 256 bits. These scores count exact generated answers.

| Training bits | Source test out of 512, seeds 0 / 1 / 2 | 128-bit test out of 256 | 256-bit test out of 256 |
| ---: | --- | --- | --- |
| 16 | 512 / 509 / 512 | 253 / 256 / 256 | 256 / 256 / 256 |
| 20 | 512 / 506 / 511 | 256 / 256 / 256 | 256 / 256 / 256 |
| 24 | 512 / 512 / 497 | 256 / 256 / 256 | 256 / 256 / 256 |

The 24-bit checkpoints reached validation at 750 / 500 / 500 updates.
The 20-bit checkpoints also solved all 256 independently generated
512-bit prompts per seed. At 1,024 bits they scored 256 / 252 / 256.
The longer evaluation reuses the source test generator, recounts every
bit with a second oracle and checks prompt uniqueness within each length.
The checkpoint, validation, original test and all shorter probes replayed
identically when the batch cap was reduced to one.

All nine checkpoints trained on 16, 20 and 24 bits solved every 512-bit
prompt. Seven solved every 1,024-bit prompt. The other two scored 252/256.
The 24-bit checkpoints also solved all 256 prompts per seed at lengths
63, 65, 255, 257 and 1,025. Neighboring lengths expose errors hidden by
the perfect three-seed score at 1,024 bits.

| Test length | Correct out of 256, seeds 0 / 1 / 2 trained on 24 bits |
| ---: | --- |
| 1,023 | 256 / 246 / 256 |
| 2,047 | 256 / 221 / 256 |
| 2,048 | 256 / 221 / 253 |
| 2,049 | 253 / 255 / 256 |
| 2,303 | 255 / 253 / 256 |
| 2,304 | 245 / 215 / 254 |
| 2,305 | 256 / 217 / 255 |

The same 24-bit checkpoints emitted the correct end token on every
2,303–2,305-bit prompt. Teacher-forced and freely generated exact scores
agreed. A separate checkpoint trained at 16 bits (seed 1) scored 256/256
on the identical 2,304-bit prompts.

Across all nine mixed checkpoints, all 256 prompts passed at 65, 255
and 257 bits. At 63 bits, the 16-bit seed-0 model dropped to 132/256.
A single-length replay separated answer classes from the end token. It
produced the correct end token 256/256 times and the correct bit on
132/132 even and 0/124 odd prompts. At 65 bits the identical checkpoint
got both bit classes right, 128/128 each. For the 24-bit seed-1 model
at 1,023 bits, the end token was again correct on all 256. Its bit scores
were 131/135 even and 115/121 odd. Teacher-forced exact counts matched
free generation on all three replays.

The input-phase control trained on 24 bits scored 37 / 246 / 205 of
256 at 255 bits and 59 / 247 / 226 at 257 bits on the identical prompts.
The corresponding mixed-reflection checkpoints scored 256/256 throughout.
Every single-length replay retains the original source validation and
test checks and fingerprints the 256 independently recounted prompts.

```sh
# With the reflection-phase binary and its existing checkpoint directory.
PHASOR_EVAL_BATCH_CAP=1 "$bin" probes basis-parity16 0 runs/reflection-phase parity-length 63
```

On original short tasks, the same mixed reflection architecture crossed
99% for modular addition at 874 / 950 / 931 updates, scoring 4,667 /
4,674 / 4,690 of 4,705 held-out pairs. GPT key–value binding crossed at
512 / 576 / 576 updates with 507 / 510 / 508 out of 512. Its separate
512-example test scored 503 / 509 / 503. All six checkpoints were reloaded.

The same architecture scored 86 / 91 / 91 of 4,705 held-out NV11
pairs after 3,000 updates. Training accuracy was 4,703 / 4,704 / 4,695
of 4,704. NV11 remains open for this backbone.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.ReflectionPhaseRead
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,reflection-phase-read
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen basis-parity20 0 runs/reflection 3500
PHASOR_EVAL_BATCH_CAP=4 "$bin" probes basis-parity20 0 runs/reflection extended
PHASOR_EVAL_BATCH_CAP=1 "$bin" probes basis-parity20 0 runs/reflection parity-1024
PHASOR_EVAL_BATCH_CAP=1 "$bin" probes basis-parity24 0 runs/reflection parity-neighbors
```

## Single reflection memory

One learned projection per channel supplies the direction of a reflection
at each position. The state update subtracts the previous phase from this
direction. Its sine and centered cosine join the general contextual read.
The backbone uses one state per channel and learns both output projections.

Lean proves that this feature equals the mixed rotation and reflection
model with every gate set to one, for arbitrary input embeddings and
output maps. Over an even number of positions the state displacement
is the sum of the adjacent projected differences. A simultaneous change
in all directions propagates through the same signed recurrence.
[Single-state derivation](../../src/Transformer/Phasor/SingleReflection.lean)

At length 257 the single-state Kernite scan agreed with FP64 within
`4.85e-6` on 49,344 outputs and `1.50e-5` on 24,672 derivatives.
At widths 64 and 128, the complete single-state model matched the separate
unit-gate implementation within `3.58e-7` on arbitrary hidden vectors.
The input pullbacks were identical. The additive specialization reproduced
the original 20-bit seed-0 training observations and weights byte for byte.

The first three 20-bit crossings occurred at updates 500 / 500 / 250.
Their independent source test scores were 512 / 512 / 481 of 512 and
their 1,024-bit scores were 256 / 256 / 206 of 256. The fastest crossing
lost accuracy on both tests. The mixed reflection model retained the
stronger three-seed transfer in this comparison.

All three 16-bit runs and all three 24-bit runs crossed at 500 updates.
Every one of those six checkpoints scored 512/512 on its independent
source test, 64/64 at both 32 and 64 bits, and 256/256 at 128, 256, 512
and 1,024 bits. The 20-bit seed that crossed at 250 updates remains the
exception at 1,024 bits. Each score used the original frozen checkpoint
and the independently recounted test prompts.

```sh
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,single-reflection-read
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen basis-parity20 0 runs/single-reflection 3500
PHASOR_EVAL_BATCH_CAP=1 "$bin" probes basis-parity20 0 runs/single-reflection parity-1024
```

## Periodic phase activation

A channel with learned strength `g` and center `b` maps its projected
embedding coordinate `x` to `x - (g/2) sin(2(x-b))`. Both the rotation
and the reflection consume that increment. The same 32 activations are
shared across the entire sequence and all tasks. The strength starts at
one and the center at zero. With zero strength, all hidden features
agree with the mixed reflection backbone for every input.

At unit strength, an angle within `|e| ≤ 1/2` radians of any
`b + kπ` remains within `(2/3)|e|³ + (4/25)|e|⁵` of that angle's
center `b + kπ`. Lean proves this bound at every integer lift and sums
it through both scans for arbitrary finite prefixes. It also derives the
three trainable input, center and strength derivatives. The bound is
conditional on the learned projections reaching those neighborhoods.
[Periodic phase activation proofs](../../src/Transformer/Phasor/PeriodicPhaseActivation.lean)

At length 257 the Kernite activation agreed with FP64 within `1.34e-6`
on 24,672 coordinates. Its input and shared-parameter gradient errors
were at most `2.01e-6` and `1.24e-5`. All 771 sampled unit-strength
coordinates satisfied the cubic bound to FP32 tolerance. With active
output projections, setting the strengths to zero reproduced arbitrary
mixed-model hidden features, logits and input gradients at widths 64
and 128. A fresh mixed-model 20-bit run reproduced the saved training
observations and weights exactly.

The seed-0 16-bit checkpoint crossed at 750 updates and passed the
512-example source test. At 1,024 bits it scored 191/256. The original
mixed model crossed at 500 and scored 256/256 there. In the 3,000-update
NV11 comparison, the periodic model scored 4,702/4,704 in training
and 90/4,705 held out.

```sh
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,periodic-reflection-read
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen basis-parity16 0 runs/periodic 3500
PHASOR_EVAL_BATCH_CAP=1 "$bin" probes basis-parity16 0 runs/periodic parity-1024
```

## Contextual rotation

Each normalized input embedding supplies a learned angle for every pair
of hidden coordinates. Before adding its own increment, a position rotates
its full contextual vector by the sum of the preceding angles. The outer
classifier then scores that vector. The input projection starts at zero.
For arbitrary embeddings and projection weights, Lean proves that this
operation preserves the contextual norm and that negating the angles
recovers the original vector. The angle gradient uses the rotated vector
and the output gradient. Each increment receives angle gradients from the
strictly later positions in its sequence.
[Contextual rotation proofs](../../src/Transformer/Phasor/ContextRotation.lean)

On three sequences of length 257 and width 256, the Kernite v0.6.0 scan
agreed with a separate FP64 calculation within `6.22e-6` on 197,376
coordinates. The largest direct hidden-gradient error was `5.05e-6`
and the largest angle-increment gradient error was `1.12e-5`. At widths
64 and 128, a fixture with nonzero rotation weights preserved each contextual
vector's norm within `9.27e-7`, changed the hidden coordinates by up to
5.17 and preserved the causal prefix exactly. Finite differences checked
the tied embedding, backbone, stream read and rotation projection.
The rebuilt input-phase control reproduced its seed-0 20-bit observations
and checkpoint byte for byte.

The first short multiplication run crossed 99% at update 950 on seed 0
with 4,684/4,705 correct. On canonical NV11, the same architecture at
3,000 updates scored 91/4,705 held out and 4,703/4,704 in training.
The saved checkpoint passed 13,933/1,358,083 held-out shift rectangles.

```sh
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,context-rotation-read
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen arithmetic-mul 0 runs/context-rotation 3800
"$bin" nv11-train runs/context-nv11 0 3000
"$bin" nv11-score runs/context-nv11/nv11-train-3000.f32 shifts
```

## Shared bilinear phase

Eight latent coordinates summarize an arbitrary prefix of dense
embeddings. Each coordinate accumulates learned additive increments and
ordered products of earlier left and current right projections. A fourth
learned projection adds a local contribution at the read position. Each
of 32 learned frequency vectors mixes all eight coordinates before sine
and centered cosine are projected back into the contextual feature.

Lean proves the exact interaction between any two segments and the
gradients of both the shared state and the learned frequencies. A
displacement of the state rotates every observation by its frequency
applied to that displacement. Its squared chord across the observation
bank is independent of the starting state. The complete model is causal
on arbitrary dense vectors. Zero output projections recover the learned
stream model for every input.
[Shared bilinear phase proofs](../../src/Transformer/Phasor/SharedBilinearPhase.lean)

At length 257, the eight-coordinate Kernite scan agreed with separate
FP64 prefix sums within `1.81e-5` on 6,168 latent reads and `1.36e-5`
on 24,672 input gradients. Relabeling all eight coordinates in the input
and frequency matrices preserved arbitrary-vector logits within `1.79e-7`
and input gradients within `1.20e-7` at widths 64 and 128. Complete
network logits agreed with FP64 within `4.89e-7`, and perturbing a later
embedding left the checked prefixes identical. The full gradient check
also varied the frequency matrix independently of the four input maps.

Seed 0 crossed 99% on short addition at update 893, with 4,687/4,705
held-out pairs correct. On canonical NV11 at 3,000 updates it scored
4,701/4,704 on training and 88/4,705 held out.

```sh
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,shared-bilinear-phase
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen arithmetic-add 0 runs/shared-bilinear 3496
"$bin" nv11-train runs/shared-bilinear-nv11 0 3000
"$bin" nv11-score runs/shared-bilinear-nv11/nv11-train-3000.f32 shifts
```

## Crossed contextual rotations

Two learned projections of every normalized embedding supply two sets
of prefix angles. At each position the first stage rotates pairs across
the two halves of the contextual vector. The second rotates adjacent
coordinates. Each stage uses only earlier input angles. The classifier
reads the result in its original coordinate order.

Lean proves that the composition preserves norms, is exactly invertible,
and recovers the one-stage architecture when the second projection is
zero. The second pairing changes what can be represented. At width four,
it can move a unit vector from coordinate zero to coordinate one. Every
single half-split rotation leaves coordinate one zero on that input.
The proof also gives the contribution of four output gradients to a first-
stage angle and establishes prefix causality.
[Crossed rotation proofs](../../src/Transformer/Phasor/CrossedContextRotation.lean)

On the length-257, width-256 fixture, the adjacent stage agreed with a
separate FP64 calculation within `4.37e-6` across 197,376 outputs. The
largest hidden and increment gradient errors were `5.06e-6` and
`1.01e-5`. With an active first stage and a zero second stage, both test
widths reproduced the one-stage hidden vectors, logits and input
gradients exactly. With both stages active, the causal prefix was exact
and the largest norm difference on arbitrary vectors was `1.98e-6`.
The width-four GPU test reproduced both coordinate transfers in the
Lean example. At 3,000 updates on canonical NV11, seed 0 scored
4,702/4,704 on training and 80/4,705 held out.

```sh
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,crossed-context-rotation
bin=./target/release/phasor-value-transport
"$bin" check
"$bin" nv11-train runs/crossed-nv11 0 3000
"$bin" nv11-score runs/crossed-nv11/nv11-train-3000.f32 shifts
```

## Prefix-conditioned Toeplitz read

The strict prefix produces one linear operator on the learned channels
of the current embedding. Phasor projects its previous contextual vector
to a real Toeplitz stencil, a complex gain per channel and a complex bias.
The stencil mixes the current channels, the gain scales and rotates them,
and a learned output projection feeds the external classifier. At the
start of every sequence the read reduces to the value and output
projections. Internal channels have no assigned token or class meaning.

For a fixed prefix the action on the projected current embedding is
affine. The same stencil coefficient applies along each matrix diagonal.
Two such diagonals suffice to shift a supported interval of channels
cyclically, even when it occupies only part of the bank. All coefficients
can be learned, and the prefix can choose a different operator at every
position. [Prefix-conditioned operator proofs](../../src/Transformer/Phasor/PrefixToeplitz.lean)

The implementation uses one complex channel per embedding dimension. On
length-257 inputs, reads agreed with dense FP64 matrix multiplication to
within `6.28e-7`; directional derivatives across the four input and
control tensors differed by at most `6.79e-6`. The full width-128 logit
error was at most `8.76e-7`. The 11-channel cyclic shift inside a bank
of 16 also reproduced the Lean witness exactly.

The three 3,000-update NV11 checkpoints fit 4,689 / 4,694 / 4,693 of the
4,704 training pairs. Independent reloads classified 85 / 94 / 95 of
4,705 held-out pairs, including 34 / 43 / 47 of the 4,654 nonzero-row
pairs. The closest result to uniform guessing on those rows was seed 2
with 47 correct, against an expectation of 48.

```sh
cargo +stable build --release --locked --features fp32-affine-reference,prefix-toeplitz
bin=./target/release/phasor-value-transport
"$bin" check
"$bin" nv11-train runs/toeplitz-nv11/seed0 0 3000
"$bin" nv11-score runs/toeplitz-nv11/seed0/nv11-train-3000.f32 rows
```

## Two-sided frame read

One learned matrix maps current embeddings across every prefix. The
previous Phasor context selects two scalar coordinates. The first rotates
the embedding before this matrix and the second rotates its output.
Each scalar drives all channel pairs through a learned frequency per pair.
A prefix-dependent scalar gain and vector bias complete the read.

Before the gain and bias, every prefix operator attains the same set of
output norms on the unit sphere. Applying the difference of the input and
output frame coordinates transports one operator to another. Thus every
prefix shares the same learned transformation up to two rotations. The
model can also make its output constant through the scalar gain.
[Two-sided frame proofs](../../src/Transformer/Phasor/TwoSidedFrame.lean)

The native implementation reproduced independently evaluated FP64 logits
within `7.92e-7` at widths 64 and 128. Applying the measured frame
displacement between two off-table prefixes aligned their hidden vectors
within `3.44e-6`. Setting the gain to zero on an arbitrary prefix gave
the same learned bias for two different inputs within `7.83e-7`.

At 3,000 updates, the three NV11 seeds fit 4,205 / 4,380 / 4,182 of
4,704 training pairs and classified 90 / 85 / 92 of 4,705 held-out pairs.
All 51 held-out zero-row pairs were correct in each reloaded checkpoint.
The remaining 39 / 34 / 41 correct answers exceed the learned stream's
36 / 25 / 33 at this update. All six nonzero-row counts are below the
48 expected from uniform guesses.

At 100,000 updates seed 0 fit all 4,704 training pairs. Its independent
checkpoint replay recovered 92 of 4,705 held-out answers, including 41 of
4,654 with a nonzero first operand. Its highest held-out count across
the 250-update evaluation intervals was 104 at update 88,250.

```sh
cargo +stable build --release --locked --features fp32-affine-reference,two-sided-frame
bin=./target/release/phasor-value-transport
"$bin" check
"$bin" nv11-train runs/two-frame-nv11/seed0 0 3000
"$bin" nv11-score runs/two-frame-nv11/seed0/nv11-train-3000.f32 rows
```

## Learned Fourier transport

One real matrix encodes the current dense embedding into complex channels.
The preceding Phasor context chooses a softmax mixture of cyclic shifts of
these channels and a constant complex source of value one. A separate
learned angle per output channel rotates the mixture. The transpose of
the encoding matrix decodes the result. An empty prefix gives the identity
route and zero phase. The native read uses the smaller of the hidden width
and 96 channels. At NV11 width 128, one cycle has room for every nonzero
character of the field.

For an affine map of a residue, each output character factors into the
phase of the intercept and the input character at the scaled frequency.
Multiplication by a nonzero field element permutes the frequencies. In
generator order it shifts the channel index. Multiplication by zero sends
all frequencies to the constant character. The learned read receives dense
vectors and estimates both controls from its prefix.
In Lean, powers of five exhaust all 96 nonzero residues modulo 97, and a
single shift realizes each nonzero row for all frequencies and operands.
[Fourier transport proofs](../../src/Transformer/Phasor/FourierTransport.lean)

The exact nonzero reference uses a vertex of the routing simplex. A real
softmax approaches that vertex as the selected logit pulls away from the
others. The reference kernel check exercises all 903,264 complex outputs,
including the zero multiplier, in a relabelled channel chart. The largest
component error against FP64 was `1.20e-7` or less. Fractional softmax
routes, phases and their adjoints agreed with an independent FP64 calculation
within `2.73e-7`. The implementation also checks directional derivatives of
the masked loss, arbitrary dense input vectors and strict-prefix causality.

Training uses the original random embeddings and the original AdamW
schedule for 100,000 updates. The comparison checkpoint was update 250.
We declared a 10,000-update extension for all seeds that missed exactness,
with observations every 250 updates. There is one trainable spectrum
matrix, one affine projection for the routing logits and angles, and four
stream weights per hidden dimension. For width `D` and `M` channels the read
adds `4D + 2MD + (2M+1)(D+1)` parameters. At NV11 width 128 this is
**49,985** read parameters and **283,921** total. Triple transport at
`e67ad738` uses **2,114,564** read parameters and **2,348,500** total.

Every row below was rescored from its saved weights in a separate process.
All seeds use the same split of 4,704 training and 4,705 held-out pairs.
The two zero-row columns cover the 46 training and 51 held-out examples
with first operand zero. The margin is the lowest target logit minus its
strongest rival on the held-out split. Seconds cover the training command,
its scheduled evaluations and checkpoint writes on the GTX 1050, with
Kernite v0.6.0 and FP32 affine operations. Repeating the triple baseline
reproduced its three original checkpoint digests.

| Read | Update | Seed | Train correct | Held correct | Zero train | Zero held | Min held margin | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triple | 250 | 0 | 4,704 | 4,705 | 46 | 51 | 2.711 | 17.78 |
| Triple | 250 | 1 | 4,704 | 4,705 | 46 | 51 | 3.003 | 17.78 |
| Triple | 250 | 2 | 4,704 | 4,705 | 46 | 51 | 2.936 | 18.58 |
| Fourier | 250 | 0 | 353 | 48 | 7 | 2 | −0.895 | 2.56 |
| Fourier | 250 | 1 | 425 | 86 | 46 | 51 | −1.391 | 2.50 |
| Fourier | 250 | 2 | 390 | 39 | 4 | 1 | −0.956 | 2.50 |
| Fourier | 10,000 | 0 | 4,700 | 51 | 46 | 4 | −61.051 | 87.64 |
| Fourier | 10,000 | 1 | 4,699 | 85 | 46 | 51 | −54.822 | 88.06 |
| Fourier | 10,000 | 2 | 4,698 | 57 | 46 | 2 | −65.965 | 87.66 |

The largest held-out counts in the extension were 55 at update 750 for
seed 0, 100 at 7,250 for seed 1, and 57 at 10,000 for seed 2. Fresh runs
ending at those updates and independent checkpoint reloads reproduced
these counts. Their training counts were 2,040, 4,700 and 4,698.
At update 250 the Fourier read took about one seventh of the baseline
time. Further training fit nearly every training pair while the held-out
predictions remained poor.

The diagnostic measures each learned input channel on all 97 operands.
A one-to-one assignment maximizes the sum of squared complex correlations
with the 96 nonzero characters. It fits one complex gain per channel. This
accounts for a permutation of channel names and each channel's phase origin.
For each nonzero prefix and output channel, the routing test compares the
highest-weight choice with multiplication of the assigned frequency by
the prefix residue. The phase comparison compensates for the fitted origins
of both channels. The last two columns count held-out answers after forcing
all routes to the identity shift or forcing all phase angles to zero,
one intervention at a time.

| Update | Seed | Mean matched correlation | Correct top route / 9,216 | Zero-prefix collapse probability | Mean nonzero phase cosine | Identity route correct | Zero phase correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 0 | 0.218 | 106 | 0.00035 | 0.002 | 48 | 42 |
| 250 | 1 | 0.220 | 81 | 0.961 | 0.009 | 48 | 106 |
| 250 | 2 | 0.222 | 100 | 0.00052 | −0.0003 | 48 | 38 |
| 10,000 | 0 | 0.216 | 104 | 7.2e-9 | 0.020 | 42 | 49 |
| 10,000 | 1 | 0.214 | 108 | 0.99999 | 0.023 | 49 | 54 |
| 10,000 | 2 | 0.212 | 110 | 7.3e-9 | 0.018 | 48 | 44 |

The same matching procedure at initialization gave mean correlations of
0.220, 0.218 and 0.220. At update 10,000 the cosine between the complete
transported coordinates and the gain-adjusted ideal output, summed over
all pairs and channels, was 0.022–0.023 (ideal 1). Seed 1 learned the
constant route for the zero row. Across nonzero prefixes, the routing and
phase measurements show little alignment with the affine action. In all
six audits, consistently renaming the external vocabulary reproduced all
9,409 predictions and all 912,673 class logits bit for bit.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.FourierTransport
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,fourier-transport
bin=./target/release/phasor-value-transport
"$bin" check
for steps in 250 10000
do
    for seed in 0 1 2
    do
        trial="runs/fourier-$steps/seed$seed"
        /usr/bin/time -f '%e' "$bin" nv11-train "$trial" "$seed" "$steps"
        "$bin" nv11-score "$trial/nv11-train-$steps.f32" rows
        "$bin" nv11-fourier-audit "$trial/nv11-train-$steps.f32"
    done
done
```

For the matched baseline, build with
`--features fp32-affine-reference,triple-transport` and repeat the
250-update train and score commands in a fresh directory. The Fourier
geometry audit applies to Fourier checkpoints.

## Triple transport

A context can learn a partial map between a dictionary of hidden vectors.
Three distinct correspondences from one map relate two ordered triples.
Joining these relations across contexts gives a graph of triples. Given two
anchor correspondences, a third candidate can be read from the same connected
component. A missing connection leaves the read undecided.

Any quantity preserved by each individual match is constant on a connected
component. If that quantity separates candidates for the given anchors, every
connected candidate is correct. Renaming the hidden slots transports the
entire relation. These results specify when the read is sound. The network
must learn reliable correspondences for its own input distribution.
The confidence test uses a damped cosine. Cauchy–Schwarz gives a strict
lower bound on the energy of every trusted hidden value. An open neighborhood
of an unwritten value therefore supplies no edges.
[Triple transport proofs](../../src/Transformer/Phasor/TripleTransport.lean)

At each position, the memory addresses its current input by the nearest
prototype direction. It addresses its prefix by matching the previous
normalized input plus the previous Phasor context divided by `√D`.
The previous context also drives four learned affine coefficients that mix
the current contextual vector, a direct memory value, a constant-row
prototype and a transported prototype. Both structural coefficients start
at zero. The first position uses the contextual read. The same functions
operate at every sequence length.

The Kernite implementation searches prototype directions in parallel at
each position. It pairs the current input with the preceding position's
dense key. Equal positive maxima and queries with no positive score give
no address. Lean proves this pairing agrees with carrying the preceding
winner through the sequence. The device search matched the scalar scan
on 90 positions with arbitrary dense vectors, repeated directions, zero
queries and batch boundaries. [Prototype address equivalence](../../src/Transformer/Phasor/TripleTransport.lean)

The dictionary starts from the unordered values of the original random
lookup. Its capacity is `K = min(D, 128)`. Dense vector ordering selects up
to `K` initial directions, and independent random vectors fill vacant slots.
The learned memory contains a zero-initialized dense vector for each ordered
pair of slots. The original cross-entropy and AdamW update these vectors,
the prototypes, gates, backbone and external classifier. Only dense inputs,
hidden states and the read's own weights enter its forward pass. The class
projection follows the read.

The confidence test uses damping energy `D × f32::EPSILON`. For a
constant read, at least two other trusted inputs must agree on the same
prototype, with every trusted image in that row equal. Rows with distinct
trusted images supply triple edges. For a transport read the implementation
takes up to four other anchors in prototype-vector order. An anchor pair
can identify a candidate when a single available image shares its source
component. All pairs that identify a candidate must agree. The graph is
reconstructed from the weights when a checkpoint is loaded. At width 128,
the read adds 2,114,564 parameters, including the four-stream readout, and
its graph has at most 2,097,152 ordered triples.

Each seed used the same original split of `a³ + ab mod 97`, with independently
initialized random embeddings and a different training order. The first
scheduled observation after initialization was update 250. A fresh process
verified each checkpoint digest and scored every pair.

| Seed | Training correct | Held-out correct | Lowest held-out logit margin | Held-out correct with triple transport disabled | Held-out correct with both structural reads disabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 4,704 / 4,704 | 4,705 / 4,705 | 2.711 | 97 | 47 |
| 1 | 4,704 / 4,704 | 4,705 / 4,705 | 3.003 | 96 | 46 |
| 2 | 4,704 / 4,704 | 4,705 / 4,705 | 2.936 | 98 | 48 |

Every training pair addressed a written direct value. Every held-out pair
addressed an unwritten value. Keeping just the gated constant and transport
reads reproduced all 9,409 answers in each checkpoint. Independently shuffling
the external vocabulary or the internal slots preserved all 912,673 class
logits bit for bit. NV11's nonconstant rows act as affine permutations of
the second operand, which explains why triples can transfer between rows.
These trials establish performance on the canonical split. Each 250-update
training command, including its scheduled evaluations and checkpoint writes,
took 17.3–18.3 seconds on the GTX 1050 with the Rust host graph and Kernite
v0.6.0 FP32 kernels.

Replacing the host prototype search with the parallel search produced
byte-identical first-passing weights on all nine NV11, modular addition
and shifted-AND runs. On fresh NV11 runs through update 250, the original
search took 17.12 / 17.00 / 17.99 seconds and the parallel search took
15.31 / 15.77 / 15.88 seconds. On shifted AND, alternating the execution
order for each seed gave six faster runs out of six, with a geometric mean
of 0.813 times the original elapsed time. All twelve timed shifted-AND
checkpoints independently reproduced their held-out scores.

On long-tier 16-bit parity, the parallel address search also finished
sooner in all six alternating-order pairs at a fixed 250 updates.
Elapsed time fell from 2.89–3.08 seconds to 2.43–2.63 seconds, with a
geometric mean time ratio of 0.850. Every held-out observation matched
between the two implementations throughout each paired run.

The host now keeps the connected components between optimizer updates.
When every previous trusted correspondence survives, it adds the triples
that contain at least one new correspondence. If a correspondence disappears
or changes, or a previously injective row becomes ineligible, it rebuilds
the components. It recomputes the read with the current anchor order in
both cases. Lean proves that retaining the old connections and adding these
edges gives exactly the new graph's equivalence relation.
[Incremental graph proof](../../src/Transformer/Phasor/TripleTransport/Incremental.lean)

On modulo-193 division with 193 learned prototypes, the 292 updates to
the first perfect held-out score took 143.68 / 159.47 / 142.68 seconds
with full graph rebuilds and 30.53 / 31.93 / 30.78 seconds with incremental
updates on the same GTX 1050. The two implementations produced identical
checkpoint bytes for each seed. On NV11 with the default 128 prototypes,
seed 0 took 15.31 seconds with full rebuilds and 7.99 seconds with the
incremental graph. The independently reloaded weights recovered all 9,409
pairs. These times cover training and scheduled observations.
A second NV11 comparison alternated execution order for each seed. The
incremental graph reached update 250 in 7.52–7.95 seconds against
14.99–16.30 seconds for full rebuilds. It was faster in all six pairs,
with a geometric mean time ratio of 0.501. Each timed checkpoint was
independently reloaded.

The confidence classifier now returns one winning slot or an abstention for
each learned value. A GPU warp compares the cosine scores and counts all
maximizers above the strict threshold. This reduces the data copied back to
the host from `K³` scores to `K²` slot indices for a dictionary of size `K`.
The device check compares 632 cells with a scalar scan, including threshold
ties, tied maxima on the same lane and on different lanes, and masked cells.
[Unique maximum proof](../../src/Transformer/Phasor/TripleTransport/ConfidenceSelection.lean)

With the incremental graph in both implementations, NV11 training through
update 250 took 7.99 / 8.64 / 7.76 seconds with the host classifier and
7.33 / 7.25 / 7.62 seconds with the device classifier. The checkpoint
files were byte-identical on all three seeds. On modulo-193 division, seed
0 reached 18,528/18,528 at update 292 in 30.53 and 29.11 seconds, also
with identical weights. At 250 updates of long-tier 16-bit parity, the
host took 2.40–2.46 seconds and the device took 2.31–2.39 seconds. The
device was faster in all six alternating-order pairs, with a geometric
mean time ratio of 0.960. All measurements used the GTX 1050 and Kernite
v0.6.0. To reproduce the NV11 and fixed-work long comparisons, build
`control` from `3ff5cb2d` and `bin` from this revision with
`fp32-affine-reference,triple-transport`.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport.ConfidenceSelection
cd experiments/phasor_value_transport
export PHASOR_EVAL_BATCH_CAP=16
"$bin" check
./time_screens.sh --first-pass "$control" "$bin" runs/confidence-nv11 250 arithmetic-cubic-product
./time_screens.sh "$control" "$bin" runs/confidence-parity 250 basis-parity16
```

To reproduce the modulo-193 comparison, build the control from
`174cec4c` and the candidate from this revision, both with
`fp32-affine-reference,triple-lookup-prototypes`. Set `control` and
`bin` to those release executables and use fresh output directories.

```sh
export PHASOR_EVAL_BATCH_CAP=16
for seed in 0 1 2; do
    for arm in control candidate; do
        exe="$control"
        if [ "$arm" = candidate ]; then exe="$bin"; fi
        dest="runs/incremental-p193/$arm"
        "$exe" screen division-p193 "$seed" "$dest" 730
        "$exe" probes division-p193 "$seed" "$dest"
    done
    cmp "runs/incremental-p193/control/division-p193/seed$seed/weights.f32" \
        "runs/incremental-p193/candidate/division-p193/seed$seed/weights.f32"
done
```

To reproduce the paired address-search comparison, build `control` from
`63cbae95` and `bin` from `3912175d`, both with `fp32-affine-reference`.

```sh
PHASOR_EVAL_BATCH_CAP=16 ./time_screens.sh --first-pass "$control" "$bin" \
    runs/triple-address-timing/boolean 512 suite-boolean-and-shift
PHASOR_EVAL_BATCH_CAP=16 ./time_screens.sh "$control" "$bin" \
    runs/triple-address-timing/parity 250 basis-parity16
```

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,triple-transport
bin=./target/release/phasor-value-transport
"$bin" check
for seed in 0 1 2
do
    trial="runs/triple-transport-nv11/seed$seed"
    "$bin" nv11-train "$trial" "$seed" 250
    "$bin" nv11-score "$trial/nv11-train-250.f32" rows
    "$bin" nv11-slot-audit "$trial/nv11-train-250.f32"
done
```

The separately trained original GPT short-tier task `arithmetic-add` reached
4,704/4,704 training and 4,705/4,705 validation answers in every seed at
update 171, its first passing observation. The original mean-stream reference
first met the suite's 99% threshold at updates 893, 950 and 931. The source
addition case supplies no separate length-transfer split.

We screened eleven further original GPT cases through update 380 with seed 0.
All eight crossings occurred at update 171 and reproduced from the saved
weights. Subtraction, multiplication, half-sum, cube-plus, square-times
and affine-product each scored 4,705/4,705 on validation.
Division scored 4,656/4,656 and fifth-power division modulo 97 scored
4,622/4,656. Permuted addition scored 41/4,705 at the cutoff.
The two long-tier screens, absolute difference and sum cubed, scored
394/4,705 and 132/4,705. Those three counts are observations at update
380 within the original 100,000-update schedules.

```sh
for seed in 0 1 2
do
    PHASOR_EVAL_BATCH_CAP=16 "$bin" screen arithmetic-add "$seed" runs/triple-gpt 380
    PHASOR_EVAL_BATCH_CAP=16 "$bin" replay arithmetic-add "$seed" runs/triple-gpt
    PHASOR_EVAL_BATCH_CAP=16 "$bin" probes arithmetic-add "$seed" runs/triple-gpt
done

export PHASOR_EVAL_BATCH_CAP=16
for task in arithmetic-sub arithmetic-mul arithmetic-half-sum arithmetic-cube-plus \
    arithmetic-square-times arithmetic-affine-product arithmetic-div arithmetic-perm-add \
    arithmetic-abs-diff arithmetic-sum-cube power-division-p97-k5
do
    "$bin" screen "$task" 0 runs/triple-gpt 380
    if test -f "runs/triple-gpt/$task/seed0/result.json"; then
        "$bin" replay "$task" 0 runs/triple-gpt
    fi
done
```

On the original long-tier `basis-parity16` case, seed 0 reached 85/157 on
novel validation sequences at update 250. The historical closed-relation
control reached 157/157 at this checkpoint. A fresh process reproduced
the triple-transport scores on the source test (266/512), the 32-bit
transfer set (21/64) and the 64-bit transfer set (25/64). All 18 causal
prefix comparisons preserved their logits. This parity trial ended at the
matched checkpoint.

```sh
PHASOR_EVAL_BATCH_CAP=16 "$bin" memory-bench triple-transport basis-parity16 0 250 runs/triple-bench
```

### Triple transport on the canonical GPT tiers

The November 22 partial audit independently reloaded 55 checkpoints from
Kernite v0.6.0. Triple transport passed 46/75 short runs and 6/30 long
runs. All three NV11 checkpoints recovered 4,704 training and 4,705
held-out pairs. Fourteen short cases and two long cases crossed on all three
seeds. The other 29 short runs and 24 long runs reached recorded cutoffs
below their source budgets.

On associative division (`associative-divisor-n2-l1-w0`), seed 0 first
crossed at update 1,805 with 4,623/4,656 correct. Seed 1 crossed at
1,691 with 4,613/4,656 and seed 2 at 1,691 with 4,620/4,656.
Fresh processes reproduced all three counts on the
frozen validation split. This case supplies a single evaluation split.

All three easy key–value recall crossings reproduced on the novel validation
split. The original test split contains eight queries per sequence.
The delay probe inserts 64 filler tokens between the writes and reads
of those same examples and scores the full sequence.

| Recall seed | First passing update | Novel validation | Independent test | Test with 64 extra tokens |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4,200 | 507/512 | 497/512 | 360/512 |
| 1 | 4,700 | 507/512 | 500/512 | 499/512 |
| 2 | 5,050 | 507/512 | 496/512 | 178/512 |

Seed 2 reached 506/512 at update 3,650 and 503/512 at update 5,000.
The next scheduled observation passed. On the short GPT probes, every
seed passed key–value binding and linear transport. Easy depth reached
512/512 on its independent source test for every seed. At length 128,
the same depth checkpoints scored 512, 326 and 512 out of 512.

On hard depth, seeds 1 and 2 reached 512/512 at updates 750 and 1,000.
Both scored 512/512 on the independent test and at length 128. Seed 0 scored
319/512 on novel validation at update 2,250.

| Short case | First passing update, seeds 0 / 1 / 2 | Correct on primary | Correct on independent test |
| --- | --- | --- | --- |
| Key–value binding | 1,984 / 3,456 / 3,456 | 509 / 511 / 507 of 512 | 510 / 502 / 504 of 512 |
| Linear transport | 704 / 576 / 576 | 254 / 255 / 256 of 256 | 251 / 255 / 256 of 256 |
| Branch amplitudes | 576 / 544 / 576 | 256 / 256 / 256 of 256 | 256 / 256 / 256 of 256 |
| Easy depth | 500 / 500 / 750 | 512 / 512 / 512 of 512 | 512 / 512 / 512 of 512 |

On shifted Boolean AND all three seeds reached 7/7 at updates 84, 60 and 68. Their independent
test scores were 64, 27 and 27 out of 64. At length 64 they scored
29, 35 and 35 out of 64.

Matched gate removals show how the same four reads serve different tasks.
On all three recall seeds, removing the contextual read dropped both novel
validation and the independent test to 0/512. Keeping just that read
scored 507, 507 and 506 out of 512 on novel validation and 499, 500 and
499 out of 512 on the independent tests. On NV11 the two structural reads alone
recovered all 9,409 pairs on every seed. The ablation zeros each removed
gate's coefficient at every prefix, reloads the original weights for each
intervention and uses the original sequence scorer. The dense contributions
of any partition of the gates add to the original output at a fixed prefix.
[Read partition proof](../../src/Transformer/Phasor/TripleTransport.lean)

To reproduce a partial audit, use the default binary and run directory
from the opening commands. Each replay records the first passing update,
exact count, denominator, checkpoint digest and supplied test splits.

```sh
"$bin" report "$out" > "$out/report.json"
jq -r '.tasks[] as $t | $t.seeds[] | select(.status=="pass") | [$t.case,.seed] | @tsv' \
    "$out/report.json" | while read -r task seed; do
    "$bin" probes "$task" "$seed" "$out"
done > "$out/passing-tests.jsonl"
jq -es --argjson passing_only true --slurpfile report "$out/report.json" \
    --slurpfile nv11 "$out/nv11-scores.jsonl" -f verify_replays.jq "$out/passing-tests.jsonl"
for seed in 0 1 2; do
    "$bin" probes basis-recall-easy "$seed" "$out" recall-gap 64
    "$bin" slot-ablate basis-recall-easy "$seed" "$out"
    "$bin" slot-ablate arithmetic-cubic-product "$seed" "$out"
done
```

### Learning a dictionary from the whole lookup table

The `triple-lookup-prototypes` build sizes its learned dictionary from
the external embedding table, with a limit of 256 directions. It takes
the unordered values of the original random vectors and uses the same
trainable gates, four reads and tied classifier at every position. The
inner block receives dense vectors and accepts inputs outside the lookup
table. For a fixed bank, adding directions that participate in no trusted
match preserves and reflects connectivity between the original triples.
Training determines which matches each dictionary acquires.
[Dictionary capacity proof](../../src/Transformer/Phasor/TripleTransport/Capacity.lean)

| Case | Width | Default slots | Lookup-sized slots | Default read parameters | Lookup-sized read parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| NV11 | 128 | 128 | 97 | 2,114,564 | 1,217,796 |
| Modulo-193 division | 128 | 128 | 193 | 2,114,564 | 4,793,604 |

The lookup-sized variant recovered all 4,704 training and 4,705 held-out
NV11 pairs at update 250 on each seed. On the original short-tier
`division-p193` case, its first passing observation was update 292 on
all three seeds. Independent checkpoint reloads reproduced 18,528/18,528
held-out answers for each seed. The source supplies one validation split
for this task. At the same update, the default scored 110, 85 and 107 out
of 18,528. Seed 0 scored 90/18,528 at update 730.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport.Capacity
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,triple-lookup-prototypes
bin=./target/release/phasor-value-transport
out=runs/triple-lookup-prototypes
export PHASOR_EVAL_BATCH_CAP=16
"$bin" check
for seed in 0 1 2; do
    "$bin" screen arithmetic-cubic-product "$seed" "$out" 250
    "$bin" nv11-score "$out/arithmetic-cubic-product/seed$seed/weights.f32"
    "$bin" screen division-p193 "$seed" "$out" 730
    "$bin" probes division-p193 "$seed" "$out"
done
```

### Learned phase residual

The `triple-phase-read` model projects each normalized dense input into
32 learned increments and sums them over the causal prefix. A learned
matrix maps the sine and centered cosine of those angles back to the
hidden width. It adds this vector to the four gated triple-transport
reads before the tied classifier. One backbone and one cross-entropy
train the whole model from the original random embeddings. The projection
matrices add `96 × width` weights. Their output columns start at zero,
which recovers the original triple read for every dense prefix.
The `triple-lookup-phase` build uses the same read with the lookup-sized
dictionary. Each build has a distinct checkpoint identifier.
[Causality, zero projection and matched removals](../../src/Transformer/Phasor/TripleTransport/PhaseResidual.lean)

With the default dictionary size, all three NV11 checkpoints independently
recovered 4,704/4,704 training and 4,705/4,705 held-out pairs at update
250. The smallest held-out margins were 2.574, 3.004 and 2.842. Seed 0
also passed modular addition at update 171 with 4,705/4,705 held-out.
With the lookup-sized dictionary, three fresh NV11 runs also reached
4,704/4,704 training and 4,705/4,705 held-out at update 250. Independent
reloads checked each of those lookup-sized checkpoints. On modulo-193
division, seed 0 with that same architecture scored 18,528/18,528 at
update 292 and reproduced the score from its saved weights.
The following parity scores were independently reproduced from the
first passing weights with the default dictionary size. Each sequence
must match in full.

| Long case | Seed | First passing update | Novel validation | Source test | 32-bit transfer | 64-bit transfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16-bit parity | 0 | 1,250 | 157/157 | 512/512 | 64/64 | 64/64 |
| 16-bit parity | 1 | 750 | 157/157 | 512/512 | 64/64 | 64/64 |
| 16-bit parity | 2 | 750 | 157/157 | 512/512 | 64/64 | 52/64 |
| 20-bit parity | 0 | 1,000 | 284/284 | 512/512 | 64/64 | 64/64 |
| 24-bit parity | 0 | 750 | 306/306 | 512/512 | 64/64 | 64/64 |

At update 3,000 the default triple model scored 117, 95 and 97 out of
157 on 16-bit parity. To measure the trained phase residual, we zeroed
each output projection in turn and re-scored the same checkpoints with
the original scorer. All columns below give correct novel sequences.

| Case and seed | Full | Remove sine | Remove cosine | Remove both | Phase alone |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 bits, seed 0 | 157/157 | 144 | 157 | 83 | 0 |
| 16 bits, seed 1 | 157/157 | 82 | 82 | 72 | 54 |
| 20 bits, seed 0 | 284/284 | 223 | 225 | 151 | 0 |
| 24 bits, seed 0 | 306/306 | 305 | 279 | 153 | 0 |

The cosine removal preserved seed 0's full 16-bit score and cost seed 1
75 sequences. Removing both projections reduced the score on all four
parity checkpoints. On NV11 seed 0, removing both projections preserved
4,705/4,705 held-out answers. The phase projection alone scored 99/4,705.
The ablation command also checks the additive identity of
the two projections on arbitrary dense inputs and compares the
zero-projection hidden states and logits with a separately instantiated
triple trunk carrying the same trained weights.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport.PhaseResidual
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,triple-phase-read
bin=./target/release/phasor-value-transport
out=runs/triple-phase-read
export PHASOR_EVAL_BATCH_CAP=16
"$bin" check
for seed in 0 1 2; do
    "$bin" screen arithmetic-cubic-product "$seed" "$out" 250
    "$bin" nv11-score "$out/arithmetic-cubic-product/seed$seed/weights.f32"
    "$bin" screen basis-parity16 "$seed" "$out" 3000
    "$bin" probes basis-parity16 "$seed" "$out"
    "$bin" slot-phase-ablate basis-parity16 "$seed" "$out"
done
for task in basis-parity20 basis-parity24; do
    "$bin" screen "$task" 0 "$out" 3000
    "$bin" probes "$task" 0 "$out"
    "$bin" slot-phase-ablate "$task" 0 "$out"
done
```

To repeat the lookup-sized phase results, use the same build and NV11
commands with `triple-lookup-phase` and a fresh run directory. Then run
`"$bin" screen division-p193 0 "$out" 438` and
`"$bin" probes division-p193 0 "$out"`.

### Indexing the unowned images

For a fixed row and two ordered anchors, every query shares the same
set of unowned output slots. The triple search groups these slots by their
connected component on first use. A query can also admit its own stored
image when it belongs to that component and differs from both anchors.
The read returns the unique admissible image. For at most eight unowned
slots it scans that short list directly. Lean proves that restoring the
query's own image gives exactly the candidates of the original scan.
[Candidate index proof](../../src/Transformer/Phasor/TripleTransport/CandidateIndex.lean)

On the default NV11 run, wall time to update 250 fell on every seed.
The table includes process startup and training on the same GTX 1050.
All three pairs produced byte-identical checkpoints. Independent scoring
recovered 4,704/4,704 training and 4,705/4,705 held-out pairs for both
implementations on each seed.

| Seed | Full scan | Indexed search |
| --- | ---: | ---: |
| 0 | 7.21 s | 5.64 s |
| 1 | 7.25 s | 5.71 s |
| 2 | 7.49 s | 6.07 s |

Build the control from `fb7aa6db` and the indexed binary from this
revision with `fp32-affine-reference,triple-transport`. To check the
candidate scan and repeat the timings in both orders for each seed, run
the following from the experiment directory.

```sh
cargo +stable test --locked --release --features fp32-affine-reference,triple-transport --test slot_orbits
export PHASOR_EVAL_BATCH_CAP=16
./time_screens.sh --first-pass "$control" "$bin" runs/indexed-nv11 250 arithmetic-cubic-product
```

### Separate input and image roles

The `triple-bipartite` model uses two copies of the ordered prototype
triples. Each trusted map connects a triple of input slots to the
corresponding triple of image slots. A chain can switch contexts at a shared
input triple or a shared image triple. The model uses the same learned
prototypes, dense values, four gates and tied classifier on every task.
At width 128 the read has 2,114,564 learned parameters and the host graph
has space for twice as many triples as the shared graph.

Lean proves that the search and its unique candidate are equivariant under
an arbitrary permutation of the image names. It also gives a five-slot
example where identifying the two roles admits a path to an incorrect
third image. The native check measures derivatives of all four reads while
holding the discrete routing decisions fixed.
[Role separation and renaming proof](../../src/Transformer/Phasor/TripleTransport/SeparateRoles.lean)

Three fresh runs on permuted modular addition crossed the source's 99%
threshold. Each checkpoint was independently reloaded and scored on all
4,705 held-out pairs. The source supplies one validation split for this
case. With the default shared graph, seed 0 scored 45/4,705 at update
1,520. The role-separated model reached the following first crossings.

| Seed | Update | Held-out correct | Training time on GTX 1050 |
| --- | ---: | ---: | ---: |
| 0 | 190 | 4,705/4,705 | 6.87 s |
| 1 | 171 | 4,661/4,705 | 5.78 s |
| 2 | 228 | 4,705/4,705 | 8.39 s |

Repeating the three runs reproduced the checkpoint bytes. The same
architecture recovered all 4,704 training and all 4,705 held-out
NV11 pairs on every seed at update 250. Its minimum held-out margins were
2.663, 3.004 and 2.934. On ordinary modular addition its independently
verified first crossings were updates 171, 209 and 209 with 4,703,
4,705 and 4,704 correct out of 4,705.

To measure the role distinction, `slot-ablate` loads the same trained
weights into a fresh engine with shared roles. On permuted addition that
intervention scored 47/4,705 on each seed. Removing the transport read
also gave 47/4,705 on all three. Keeping only the two structural reads
scored 4,705, 4,660 and 4,705 out of 4,705.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport.SeparateRoles
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,triple-bipartite
bin=./target/release/phasor-value-transport
out=runs/separate-triple-roles
export PHASOR_EVAL_BATCH_CAP=16
"$bin" check
for seed in 0 1 2; do
    "$bin" screen arithmetic-cubic-product "$seed" "$out" 250
    "$bin" nv11-score "$out/arithmetic-cubic-product/seed$seed/weights.f32"
    for task in arithmetic-perm-add arithmetic-add; do
        "$bin" screen "$task" "$seed" "$out" 380
        "$bin" probes "$task" "$seed" "$out"
    done
    "$bin" slot-ablate arithmetic-perm-add "$seed" "$out"
done
```

### Aligned peer reads

The `triple-peer-phase` model uses shared hidden keys to relate the outputs
of two learned contexts. For example, if two contexts map one key to
the pair `(u, v)` and another to `(s, t)`, the read connects those two
ordered pairs. For each missing triple prediction, it looks for a unique
image consistent with one peer's output. It requires two other shared keys
per peer and agreement from at least two peers. A conflicting vote cancels
the prediction. The four gates and the 32 learned phase channels feed the
same tied classifier at every prefix.

The dictionary uses one slot per external embedding up to 256. For larger
lookups it uses the smaller of the hidden width and 128. The peer search
adds no weights. Lean proves when an invariant of the observed pairs
determines the hidden target. The proof permits repeated images and
independent renamings of contexts, keys and outputs. The bounded dictionary
can give every lookup vector its own slot whenever the ceiling permits it.
[Aligned peer proof](../../src/Transformer/Phasor/TripleTransport/AlignedPeers.lean) · [Dictionary capacity proof](../../src/Transformer/Phasor/TripleTransport/Capacity.lean)

A partial audit of this architecture independently verified **12/75 short,
1/30 long and 3/3 exact NV11**. Each seed started from fresh weights.
The other 63 short and 29 long runs were awaiting a completed run or
cutoff report in this snapshot. The table gives the first passing updates and their
reloaded primary counts. The arithmetic cases supply one evaluation split.

| Case | Seeds | First update | Correct on primary |
| --- | --- | --- | --- |
| Square-sum | 0 / 1 / 2 | 171 / 171 / 209 | 4,705 / 4,705 / 4,705 of 4,705 |
| Power division, modulus 97, exponent 2 | 0 / 1 / 2 | 171 / 171 / 171 | 4,656 / 4,656 / 4,645 of 4,656 |
| Power division, modulus 193, exponent 5 | 0 | 292 | 18,528/18,528 |
| Permuted addition | 0 | 190 | 4,705/4,705 |
| Shifted Boolean AND | 0 / 1 / 2 | 92 / 56 / 68 | 7 / 7 / 7 of 7 |
| Random Boolean AND | 0 | 104 | 8/8 |
| 16-bit parity | 0 | 1,000 | 157/157 |
| NV11 | 0 / 1 / 2 | 250 / 250 / 250 | 4,705 / 4,705 / 4,705 of 4,705 |

All three NV11 checkpoints also recovered 4,704/4,704 training pairs.
Shifted AND scored 63, 27 and 27 of 64 on the independent test and
29, 35 and 35 of 64 at length 64. Random AND seed 0 scored 64/64 on
the independent test and 28/64 at length 64. Parity seed 0 scored
504/512 on the independent test, 58/64 at 32 bits and 64/64 at 64 bits.

On square-sum seeds 0 and 1, disabling only the peer search reduced
validation from 4,705 to 45 correct with the same weights and dictionary.
On modulus-97 power division with exponent 2, the same intervention
reduced seed 0 from 4,656 to 100 correct.
On 16-bit parity seed 0, zeroing the sine and cosine projections together
reduced novel validation from 157 to 86 correct and the independent test
from 504 to 350. The zeroed model's dense output matched the phase-free
peer read at every tested position.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport.AlignedPeers
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,triple-peer-phase
bin=./target/release/phasor-value-transport
out=runs/aligned-peer-phase
export PHASOR_EVAL_BATCH_CAP=16
"$bin" check
for seed in 0 1 2; do
    for task in arithmetic-cubic-product arithmetic-square-sum power-division-p97-k2 suite-boolean-and-shift; do
        "$bin" cell "$task" "$seed" "$out"
        "$bin" probes "$task" "$seed" "$out"
    done
    "$bin" nv11-score "$out/arithmetic-cubic-product/seed$seed/weights.f32"
done
for task in power-division-p193-k5 arithmetic-perm-add suite-boolean-and-random basis-parity16; do
    "$bin" cell "$task" 0 "$out"
    "$bin" probes "$task" 0 "$out"
done
"$bin" slot-ablate arithmetic-square-sum 1 "$out"
"$bin" slot-phase-ablate basis-parity16 0 "$out"
```

### Shared witnesses across contexts

The `triple-witness-phase` model reads pairs of outputs observed at the
same hidden key in two contexts. It joins two context pairs when they
share six distinct ordered output pairs. Each connected group pools the
outputs it has observed for each source value. At a missing key, the
model intersects the nonempty candidate sets supplied by other contexts.
It uses a singleton supported by at least two contexts wherever the
constant and triple reads have abstained. The same search runs on every
task. It adds no weights and visits up to 16 reference rows ordered by
learned support, breaking ties by the learned prototype order.

Lean proves that pooling preserves relation membership when observations
are sound and six shared pairs distinguish different relations. If every
active candidate set contains the true output, their intersection singles
it out exactly when every competing output is excluded by some active
context. For squared shifts over a field of characteristic other than two,
three distinct shared pairs already identify the squared shift. The search
uses only learned output pairs. [Shared-witness proofs](../../src/Transformer/Phasor/TripleTransport/OverlappingEvidence.lean)

| Case | Seed | First passing update | Correct on reloaded primary split |
| --- | --- | --- | --- |
| Sum-square, short | 0 / 1 / 2 | 171 / 190 / 228 | 4,705 / 4,686 / 4,705 of 4,705 |
| Square-sum, short | 0 / 1 / 2 | 171 / 171 / 190 | 4,687 / 4,668 / 4,661 of 4,705 |
| Sum-cube, long | 0 / 1 / 2 | 171 / 171 / 228 | 4,705 / 4,705 / 4,686 of 4,705 |
| 16-bit parity, long | 0 | 1,000 | 157/157 |
| NV11 | 0 / 1 / 2 | 250 / 250 / 250 | 4,705 / 4,705 / 4,705 of 4,705 |

All three NV11 checkpoints also classified all 4,704 training pairs
correctly. The parity checkpoint scored 504/512 on the independent test,
58/64 at 32 bits and 64/64 at 64 bits. The arithmetic cases supply one
evaluation split. Disabling only the witness search on the seed 0
checkpoints reduced sum-square to 58/4,705 and sum-cube to 104/4,705.
Absolute difference scored 4,306/4,705 at the seed 0 screen cutoff of
950 updates.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport.OverlappingEvidence
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,triple-witness-phase
bin=./target/release/phasor-value-transport
out=runs/shared-witness-phase
export PHASOR_EVAL_BATCH_CAP=16
"$bin" check
for seed in 0 1 2; do
    for task in arithmetic-sum-square arithmetic-square-sum arithmetic-sum-cube; do
        "$bin" cell "$task" "$seed" "$out"
        "$bin" probes "$task" "$seed" "$out"
    done
    "$bin" cell arithmetic-cubic-product "$seed" "$out"
    "$bin" nv11-score "$out/arithmetic-cubic-product/seed$seed/weights.f32"
done
"$bin" cell basis-parity16 0 "$out"
"$bin" probes basis-parity16 0 "$out"
"$bin" slot-ablate arithmetic-sum-square 0 "$out"
"$bin" slot-ablate arithmetic-sum-cube 0 "$out"
"$bin" screen arithmetic-abs-diff 0 "$out" 950
```

### Repeated witness reads

The `triple-iterated-witness` model can use a prediction at one hidden key
as evidence at another. It runs the same search up to three times, using
at most 16, 32 and 32 reference rows. Each round preserves the learned
correspondences and all previous answers. It stops when no new evidence
appears or every structural read has an answer. All inferred evidence is
local to the current bank of learned weights. The next refresh starts
again from those weights. This change adds no parameters.

Lean proves that every round preserves a total interpretation if the
initial evidence agrees with it and each new inference is locally sound.
A finite example distinguishes one round from two. [Repeated-read proofs](../../src/Transformer/Phasor/TripleTransport/EvidenceRounds.lean)

The first partial audit verified **11/75 short, 5/30 long and 3/3 exact
NV11**. Two short and five long runs had reached a screen cutoff. The
remaining 62 short and 20 long runs were pending in that snapshot.

| Case | Seed | First passing update | Reloaded primary | Independent test |
| --- | --- | --- | --- | --- |
| Absolute difference, long | 0 / 1 / 2 | 209 / 171 / 247 | 4,664 / 4,669 / 4,662 of 4,705 | — |
| Sum-square, short | 0 | 171 | 4,705/4,705 | — |
| Sum-cube, long | 0 | 171 | 4,705/4,705 | — |
| Power division, modulus 97, exponent 2 | 0 | 171 | 4,656/4,656 | — |
| GPT branch amplitudes, short | 0 | 736 | 256/256 | 256/256 |
| GPT linear transport, short | 0 | 448 | 256/256 | 255/256 |
| 16-bit parity, long | 0 | 1,000 | 157/157 | 504/512 |
| NV11 | 0 / 1 / 2 | 250 / 250 / 250 | 4,705 / 4,705 / 4,705 of 4,705 | — |

On the same trained absolute-difference checkpoints, retaining one round
scored 4,246 / 4,211 / 4,255. Disabling all witness reads scored
399 / 289 / 407. The three NV11 checkpoints each scored 4,704/4,704 on
the training split. The parity checkpoint transferred to 58/64 at 32 bits
and 64/64 at 64 bits. Both GPT probes used the original free-generation
scorer. Blinded absolute difference scored 160 / 191 / 173 of 4,705
through the 950-update screen. GPT key–value binding seed 0 scored
423/512 at update 4,096. Both cases have larger original budgets.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.TripleTransport.EvidenceRounds
cd experiments/phasor_value_transport
cargo +stable build --locked --release --features fp32-affine-reference,triple-iterated-witness
bin=./target/release/phasor-value-transport
out=runs/iterated-witness
export PHASOR_EVAL_BATCH_CAP=16
"$bin" check
for seed in 0 1 2; do
    for task in arithmetic-cubic-product arithmetic-abs-diff; do
        "$bin" cell "$task" "$seed" "$out"
        "$bin" probes "$task" "$seed" "$out"
    done
    "$bin" nv11-score "$out/arithmetic-cubic-product/seed$seed/weights.f32"
    "$bin" slot-ablate arithmetic-abs-diff "$seed" "$out"
    "$bin" screen arithmetic-abs-diff-blind "$seed" "$out" 950
done
for task in arithmetic-sum-square arithmetic-sum-cube power-division-p97-k2 \
    basis-parity16 gpt-probe-branch-amplitudes gpt-probe-linear-transport; do
    "$bin" cell "$task" 0 "$out"
    "$bin" probes "$task" 0 "$out"
done
"$bin" screen gpt-probe-key-value-binding 0 "$out" 4096
```

## Single-coordinate frame orbit

One learned coordinate from the preceding context moves both frames.
Their frequencies remain independent. A second scalar controls the gain
around one learned center shared across every prefix. The current
embedding keeps all its dimensions. After undoing a nonzero gain, the
difference of two prefix coordinates transports the complete response
from one prefix to the other. This puts all normalized operators in one
one-parameter family. At zero gain every current embedding gives the
same center. [Frame orbit and adjoint proofs](../../src/Transformer/Phasor/SingleFrameOrbit.lean)

## Tied Toeplitz read

The same learned matrix projects the current embedding into complex
channels and projects the transformed channels back by its transpose.
The previous contextual vector still chooses the stencil, complex gains
and biases. This removes the independent output matrix and makes the
empty-prefix scores of normalized inputs a positive semidefinite Gram
matrix. For every prefix, each class vector pairs with the transformed
value through that same learned basis. Changing the shared matrix
contributes two terms to its gradient, one from each use.
[Tied read and gradient proofs](../../src/Transformer/Phasor/TiedToeplitz.lean)

Expanding the shared matrix into two independent matrices initialized
as transposes reproduced the same hidden states, logits and arbitrary
input gradients at widths 64 and 128. At width 128 the two matrix
gradients added to the actual shared gradient within `7.46e-9`.
The matched models used identical initial values for the embedding,
context backbone, value matrix, stencil, gain and bias.

After 3,000 updates, the tied model reproduced 4,680 / 4,685 / 4,675
of 4,704 training pairs and 88 / 89 / 79 of 4,705 held-out pairs.
Excluding the constant zero row gave 40 / 43 / 32 of 4,654, compared
with 34 / 43 / 47 for the untied run. At 20,000 updates the untied
seed-0 checkpoint still got only 45 of the nonzero-row pairs and 96
overall. Each reported final score was reproduced from saved weights.

```sh
cargo +stable build --release --locked --features fp32-affine-reference,tied-toeplitz
bin=./target/release/phasor-value-transport
"$bin" check
"$bin" nv11-train runs/tied-toeplitz-nv11/seed0 0 3000
"$bin" nv11-score runs/tied-toeplitz-nv11/seed0/nv11-train-3000.f32 rows
```

## Scalar phase bottleneck

The earlier shared bilinear model fit 4,701 of 4,704 training pairs on
NV11 seed 0 after 3,000 updates and generalized to 88 of 4,705 held-out
pairs. Its residual connection allows a full contextual vector to reach the
classifier. To test whether that path makes memorization easy, this model
combines one learned projection of the contextual vector with one ordered
bilinear state. Every hidden coordinate sent to the external classifier
is decoded from that single scalar using 32 learned frequencies.

Lean proves that any two prefixes with the same scalar have identical
logits under every linear classifier. At any scalar value, every
two-by-two minor of the score Jacobian vanishes. The frequencies and
output projections are arbitrary in both statements. These are
constraints on the representation that training can learn.
[Scalar phase bottleneck proofs](../../src/Transformer/Phasor/ScalarPhaseBottleneck.lean)

At widths 64 and 128, the native logits agreed with independent FP64
evaluation to within `1.14e-6`. Two unrelated classifier cotangents
at the same position produced input derivatives collinear to within
`2.0e-7` relative error. Prefixes with equal additive coordinates
gave identical final hidden vectors. Zeroing the coordinate gave zero
hidden vectors, logits and embedding derivatives.

At 3,000 updates the three NV11 seeds fit 467 / 386 / 445 of 4,704
training pairs and classified 58 / 62 / 50 of 4,705 held-out pairs.
Independent checkpoint replay reproduced all six counts. Constraining
the whole read to one scalar reduced training fit and held-out accuracy
on all three seeds compared with the eight-coordinate residual model.

```sh
cargo +stable build --release --locked --features fp32-affine-reference,scalar-phase-bottleneck
bin=./target/release/phasor-value-transport
"$bin" check
"$bin" nv11-train runs/scalar-nv11/seed0 0 3000
"$bin" nv11-score runs/scalar-nv11/seed0/nv11-train-3000.f32 rows
```

## Householder memory

Each embedding supplies three learned slopes per channel. With a leading
one these form a four-dimensional normal `n`. The recurrent update
reflects the state `s` across the plane perpendicular to `n` by subtracting
`2 * dot(n,s) * n / dot(n,n)`. The denominator is at least one. A learned
projection returns the difference between the current and initial states
to the contextual feature. The initial state is also learned.

Each prefix preserves its initial state norm. Two identical consecutive
embeddings cancel exactly in this memory at any insertion point and
after any suffix. Other pairs can depend on order. Two attainable planes
send the same starting vector to opposite endpoints when applied in
reverse order. Lean also gives the transpose on the state and the full
gradient with respect to the normal, including its denominator.
[Householder memory proofs](../../src/Transformer/Phasor/HouseholderMemory.lean)

The Kernite implementation uses 16 four-dimensional channels. At length
257 its 49,344 reflected coordinates agreed with independently formed
FP64 reflection matrices to within `2.23e-6`. The largest squared-norm
drift was `6.36e-6`. At widths 64 and 128 the full network agreed with
the FP64 logit reference to within `3.68e-7`. Off-table embedding
pullbacks and masked-loss derivatives also passed.

At update 3,000 the three canonical NV11 checkpoints reproduced training
scores of 4,703 / 4,703 / 4,702 out of 4,704 and held-out scores of
91 / 82 / 77 out of 4,705. This improved two seeds over the learned
stream's 86 / 74 / 84 and lost seven predictions on seed 2. It gives
no repeatable gain across these seeds.

```sh
cargo +stable build --release --locked --features fp32-affine-reference,householder-memory
bin=./target/release/phasor-value-transport
"$bin" check
"$bin" nv11-train runs/householder-nv11/seed0 0 3000
"$bin" nv11-score runs/householder-nv11/seed0/nv11-train-3000.f32 rows
```

## Associative write memory

Every pair of consecutive embeddings contributes to a learned matrix.
The previous embedding supplies the key, the current embedding supplies
the value, and the current query reads the updated matrix. The three
input projections use the same RMS-normalized vectors as the phase read.
A fourth learned projection adds the retrieved values to the mixed
reflection output. The projections are shared across all positions.

For a row of state `s`, the write adds `(v - dot(s,k)) * k`. Keys
and queries use `x / sqrt(1 + dot(x,x))`. At the updated key the error
shrinks by exactly `1 / (1 + dot(x,x))`. Lean derives the directional
derivative and its transpose when the key, value and state all move.
It proves prefix causality for arbitrary embeddings and exact recovery
of the mixed reflection backbone when the output matrix is zero.
[Associative write proofs](../../src/Transformer/Phasor/AssociativeWrite.lean)

The Kernite model uses eight key coordinates and four value rows.
At length 257 the largest FP64 discrepancy was `3.42e-7` across 24,672
matrix entries and `3.22e-7` across 3,084 reads. Seven independent full
directions of the raw key, query and value projections agreed with central
differences within `8.57e-6`. At lengths 1 and 7 the check also differentiated
every projected coordinate individually. A zero output map reproduced the
mixed backbone's arbitrary-vector features, logits and input gradients
exactly at widths 64 and 128. With nonzero maps, the largest full-network
feature discrepancy was `5.07e-7`, with unchanged causal prefixes.
The original mixed model reproduced its 20-bit seed-0 weights byte for byte.

On seed 0, GPT key–value binding crossed at update 576 with 507/512
validation and 506/512 independent test examples correct. The same model
trained on canonical NV11 for 3,000 updates reached 4,701/4,704 in
training and 78/4,705 on the held-out half.

On the original Basis easy recall task, seeds 0 / 1 / 2 crossed 99%
at updates 1,900 / 1,550 / 900. They solved 511 / 509 / 505 of the 512
independent test sequences, with 4,095 / 4,093 / 4,089 correct answers
among 4,096 queries. Adding 32 distractor values after the writes
reduced the sequence scores to 477 / 505 / 500 and the query scores to
4,060 / 4,089 / 4,084. All scores use the same saved checkpoints.

The paired delay test retains every source write, query, original filler
and target in its original order. Longer gaps extend the same random
filler stream for each test example. The evaluator checks the extended
targets with an independent last-write oracle, fingerprints the split and
replays the original validation and test sets. A zero gap reproduces the
published easy test fingerprint exactly. The same construction supports
hard recall, which includes overwrites.
[Recall gap proof](../../src/Transformer/Basis/RecallParsed.lean)

```sh
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,associative-reflection-read
bin=./target/release/phasor-value-transport
"$bin" check
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen gpt-probe-key-value-binding 0 runs/associative 2944
PHASOR_EVAL_BATCH_CAP=16 "$bin" probes gpt-probe-key-value-binding 0 runs/associative
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen basis-recall-easy 0 runs/associative 3500
PHASOR_EVAL_BATCH_CAP=4 "$bin" probes basis-recall-easy 0 runs/associative recall-gap 32
cargo +stable test --release --locked --test recall_gap
"$bin" nv11-train runs/associative-nv11 0 3000
"$bin" nv11-score runs/associative-nv11/nv11-train-3000.f32 shifts
```

## NV11 prediction shifts

CM121 describes row and column translations of class labels. To see
whether a checkpoint learns this structure, score its original top-one
predictions and subtract the true classes modulo 97 in the evaluator.
Two held-out cells in the same row agree up to a row shift when their
residuals coincide. For each pair of rows, group their common held-out
columns by residual difference. Two columns in the same group give a
rectangle whose four errors cancel. Every corner must belong to the
split being evaluated.

Lean proves that a row-plus-column translation passes all rectangles and,
on a complete table, that passing all rectangles characterizes exactly
these translations. Multiplication and NV11 differ by such a row shift.
[Prediction-shift identities](../../src/Transformer/Phasor/PredictionShifts.lean)

At 100,000 updates the learned stream's seed-0 checkpoint fit all
4,704 training pairs. Every one of the 1,355,860 training rectangles
passed. On the disjoint held-out half, 13,866 of 1,358,083 rectangles
passed (1.021%). Independent uniform residuals would pass in expectation
with probability 1/97 (1.031%). Rectangles share corners. In the same
held-out table, 2,453 of 112,950 within-row pairs and 1,115 of 113,021
within-column pairs had equal residuals. Thus the 90/4,705 held-out
predictions were also inconsistent with a global row-plus-column shift.

At 3,000 updates, among the same held-out rectangles, the learned stream
passed 13,932 / 13,878 / 13,946 and the input-phase model passed
13,888 / 14,159 / 14,103 across the three seeds. Enabling these counts
reproduced each original checkpoint's exact scores and minimum margins.
The native fixture compares the counter with direct four-corner
enumeration on incomplete modular tables.

Scoring by first operand separates the constant zero row from the other
96 rows, each of which contains every output class once. The held-out
split has 51 pairs in the zero row and 4,654 in the others. After 100,000
updates both seed-0 models fit all 4,704 training pairs and all 51 held-out
zero-row pairs. The learned stream classified 39 of the remaining 4,654
pairs and the shared bilinear model classified 46. Uniform predictions
would average about 48. Across all 401 observations from initialization
through update 100,000, the shared model first reached its highest total
of 105/4,705 at update 63,500 and finished at 97/4,705.

At 3,000 updates the learned stream classified 36 / 25 / 33 of the 4,654
nonzero-row pairs on seeds 0 / 1 / 2. The shared bilinear model classified
37 / 31 / 36. The evaluator recovers these counts from frozen weights and
masked inputs. Row membership and correctness are computed after inference.

```sh
# From experiments/phasor_value_transport, with the matching architecture build.
cargo +stable test --release --locked --test shift_diagnostics
target/release/phasor-value-transport nv11-score runs/nv11/seed0/nv11-train-100000.f32 shifts
target/release/phasor-value-transport nv11-score runs/nv11/seed0/nv11-train-100000.f32 rows
```

## Matched CM121 tasks

The fixed-budget catalog runner uses the frozen GPT plan for every task.
For the six CM121 arithmetic tables, this gives 4,704 training pairs in
the same order, 4,705 held-out pairs, a batch of 256 and an observation
every 250 updates on the 100,000-update WSD schedule. Each run records
its initial weight digest and separate digests of the ordered training
and held-out inputs. Each task trains its own instance of the selected
architecture with the original AdamW. The checkpoint replay scores every
held-out pair. The 250-update cubic-product checkpoint reproduced the
canonical contextual-rotation NV11 weights byte for byte.
[E6 task definitions and CM121](../convex_stress_schatten_mixed/README.md)

Both architectures solved multiplication and its affine shift on all
three seeds at 3,000 updates. The learned stream first crossed 99% on
multiplication at 1,000 updates for every seed. Contextual rotation
crossed at 1,000 / 1,000 / 1,250. On the affine shift, the respective
crossings were 1,250 / 1,500 / 1,500 and 1,500 / 1,500 / 1,500.

All 36 checkpoints were reloaded and scored. They used identical ordered
input splits. Within each architecture and seed, all six tasks used
identical initial weights. Each cubic-product checkpoint matched its
separately trained canonical NV11 counterpart byte for byte. The table
gives correct held-out pairs out of 4,705 in seed order 0 / 1 / 2.

| Target modulo 97 | Learned stream | Contextual rotation |
| --- | --- | --- |
| `ab` | 4,705 / 4,705 / 4,705 | 4,705 / 4,705 / 4,705 |
| `a + ab` | 4,705 / 4,705 / 4,705 | 4,705 / 4,705 / 4,705 |
| `a³ + ab` | 86 / 74 / 84 | 91 / 81 / 78 |
| `a² + ab` | 74 / 69 / 84 | 69 / 75 / 80 |
| `a² + ab + b²` | 61 / 75 / 87 | 60 / 62 / 60 |
| `a² + ab + b² + a` | 34 / 36 / 37 | 47 / 28 / 43 |

`catalog-shifts` replays both checkpoints, verifies their common inputs
and initialization, then sends the same input pairs through each model.
In the evaluator, the difference of the source targets tells it how far
to rotate the candidate's class probabilities. It checks all 9,409
target shifts for row-plus-column separability and compares ordinary
softmax probabilities on every class, separately for each split.

On seed 0, both multiplication self-comparisons agreed exactly. For contextual
rotation on the affine shift, every held-out argmax agreed after alignment
and the largest probability difference was `8.71e-7`. On the cubic
shift, the models' top classes differed after alignment on 1/4,704 training
pairs and 4,614/4,705 held-out pairs. The held-out mean total variation
distance was 0.98155. The affine endpoint was highly confident under
both trainings, and its earlier 99% crossings already differed.

```sh
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,context-rotation-read
bin=./target/release/phasor-value-transport
for name in $(jq -r '.tasks[]' ../convex_stress_schatten_mixed/protocol.json)
do
    PHASOR_EVAL_BATCH_CAP=16 "$bin" catalog-bench "$name" 0 3000 runs/cm121
done
# Repeat the same command to authenticate the stored result and score its weights again.
PHASOR_EVAL_BATCH_CAP=16 "$bin" catalog-shifts arithmetic-mul arithmetic-cubic-product 0 3000 runs/cm121
cargo +stable test --release --locked --test class_translation
```

## GPT order discrimination

The catalog adapter uses the native GPT mechanism-probe data port to load
the original seed-1 partition. It verifies both tensor fingerprints. The
496 training and 496 held-out examples cover all ordered pairs of distinct
symbols from a set of 32. The model receives their learned embeddings and
the outer classifier predicts the comparison label at the second position.
The source provides two splits. On seed 0 at 5,000 updates the learned
stream model scored 461/496 and the affine phase model scored 459/496.
The short and long extension registry requires a three-seed 99% crossing.
[Native GPT probe data](../convex_stress_probe_data/README.md)

At the end of the original 30,000-update schedule, both the input-phase
and mixed reflection models classified all 496 training pairs correctly
on seed 0. Their held-out results were 463/496 and 472/496. Across all
121 observations, the input-phase model peaked at 475/496 at update
10,000 and mixed reflection at 473/496 at update 15,250. The 99%
criterion requires 492 correct.

```sh
# From the repository root, set PINNED_TORCH_ROOT to the native ATen installation.
bash experiments/convex_stress_probe_data/native/build.sh "$PINNED_TORCH_ROOT" .lake/reproduction/gpt-probe-reference
cd experiments/phasor_value_transport
cargo +stable build --release --locked --features fp32-affine-reference,affine-phase-read
bin=./target/release/phasor-value-transport
PHASOR_EVAL_BATCH_CAP=16 "$bin" screen gpt-probe-order-discrimination 0 runs/affine-phase 5000
"$bin" catalog-report runs/affine-phase gpt-probe-order-discrimination
```

An existing native helper at a different path can be selected with
`PHASOR_GPT_PROBE_HELPER`.

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
cargo +stable build --release --locked --features fp32-affine-reference,learned-stream-read
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
version alongside the exhaustive scores. For a named training checkpoint
beside its observation log, the scorer checks the saved digest and
recorded architecture, recomputes both splits, and checks their exact
counts. The `first-exact` filename selects the earliest recorded 9,409/9,409
score. A standalone weights file can also be scored with the executable
for its architecture. The scorer masks the targets and reports the minimum
correct-class margin on each split.

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
