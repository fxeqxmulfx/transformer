# One Phasor architecture for NV11 and the GPT stress suite

One Phasor architecture passed all 135 GPT task/seed runs across 34 short
and eleven long tasks. It classified all 9,409 NV11 pairs correctly at update
750 in every seed. Each task trained its own weights with the original AdamW. The model
uses a causal successor read, full tied embedding displacement and a learned
residual gain initialized to 0.75.

The GPT suite also includes the official eight-needle text evaluation at up
to 128K and at 1M context. The result is a mean match ratio on each published
split. [MRCR V2 stress test](../mrcr_v2/README.md)

## Selected architecture on Kernite v0.6.0

Every position uses the same causal prefix scan and the same 17 learned
tables. Let `b` be the current token and `r` its most recently observed
successor in the completed prefix, with `r = b` for an unwritten key.
The residual adds `E[r] − E[b]` before RMS normalization, scales the
normalized vector by the learned gain and adds it to the backbone feature.
The half-gain predecessor and this model have the same parameter count.
[Tied read and score identities](../../src/Transformer/Phasor/SuccessorDisplacement.lean)

| Matched tier | Passing task/seed runs | Epoch ratio to half-gain predecessor | Faster / tied / slower |
| --- | ---: | ---: | ---: |
| Short, including seven catalog additions | 96/96 | 0.6223 | 67 / 23 / 6 |
| Short, including all nine short catalog additions | 102/102 | 0.6144 | 70 / 26 / 6 |
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

For a paired comparison, use a fresh directory for each model. The plain
FP32 build selects the 0.75-gain model. The baseline build uses
`fp32-affine-reference,prefix-successor-half`.

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.Phasor.SuccessorDisplacement
cd experiments/phasor_value_transport
export PHASOR_EVAL_BATCH_CAP=16
for arm in half selected; do
    features=fp32-affine-reference
    if [ "$arm" = half ]; then features=$features,prefix-successor-half; fi
    cargo +stable build --release --locked --features "$features"
    bin=./target/release/phasor-value-transport
    "$bin" shared-check
    ./run_stress.sh cell "runs/v060-$arm" extended
    ./run_stress.sh replay "runs/v060-$arm" extended
    "$bin" extended-report "runs/v060-$arm" > "runs/v060-$arm/report.json"
done
git show 849aa3be:experiments/phasor_value_transport/extensions.tsv > runs/seven-extensions.tsv
jq -s --rawfile extensions runs/seven-extensions.tsv -f compare.jq \
    runs/v060-half/report.json runs/v060-selected/report.json
jq -s --rawfile extensions extensions.tsv -f compare.jq \
    runs/v060-half/report.json runs/v060-selected/report.json
```

## Kernite v0.6.0

The Phasor runner uses the published v0.6.0 CUDA and macro crates with
Rust 1.100.0. The original DiffGPT trainer and AdamW are pinned to
`5a37c515`. Fresh results record the Kernite version from Cargo.lock.
Checkpoint replay reports both the training and scoring versions.

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

The selected binary lists all ten additions and reports each seed.

```sh
"$bin" extensions
"$bin" catalog-report runs/v060-selected suite-blocks suite-dyck arithmetic-sum-pow6 language-l1 arithmetic-sum-pow8 arithmetic-sum-interval arithmetic-sum-pow12 suite-parity-none arithmetic-sum-pow16 suite-crasp-depth-1
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
cargo +stable build --locked --release --features fp32-affine-reference
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
