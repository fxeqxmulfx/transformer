# Multi-query retrieval and two-hop lookup

Can the learned CM111 model retrieve two values in a longer memory or follow
two links? The single-query experiment (producer `339ba587`) fitted two
key/value writes at fixed positions. Here a fresh instance of the same model
trains on each original case. It sees an immutable causal prefix of object IDs
and emits one object from the complete original vocabulary. Training uses
next-object cross-entropy.

| Original case | Memory and supervision | Vocabulary |
| --- | --- | ---: |
| `suite-mqar` | Four distinct key/value writes, distractors and two supervised key queries | 164 |
| `suite-lookup` | Four entries in a cyclic lookup table and two queries whose answers require following two links | 100 |

The task320 native data sources construct 128 training, 32 validation and 64
test examples at length 24. Each sequence has two supervised positions.
Validation and test also include lengths 48 and 96. All 11 original, clean
and novel splits per case enter scoring. The protocol pins the original-source
and tensor fingerprints recorded by task320. The individual historical GPT
checkpoints and their split hashes are unavailable.

The architecture, parameter initialization, minibatch sampler, optimizer and
fixed 1,024-update endpoint come from the binding protocol. Each case uses
its complete vocabulary, a relative-position table sized for the published
maximum length 96, batches of 32 and seeds 0, 1 and 2 on one original dataset
seed. The model receives no task or role labels. The scored lengths 48 and 96
require position generalization beyond the length-24 training examples.

Mini differentiates the complete CE. Forward-only Mini makes two feasible
forward CE queries per scheduled coordinate. Each fit uses 32,768 original
training examples and 65,536 supervised targets across 256 epochs. The
runner seals all 12 fitted endpoints in each of two independent builds before
scoring either case. It preserves each full vocabulary logit vector, the
original 95% exact success on novel validation, and separate 99% raw and exact
flags on every split. Initialization, training and scoring repeat independently.

Run from the workspace root with the producing commit and a fresh output path.

```sh
bash experiments/learned_retrieval/run.sh "$PWD" /run/media/misha/storage PRODUCER \
  /run/media/misha/storage/transformer-runs/learned-retrieval/sequence-transfer-v1 transfer
```

## Results

Both independent runs trained from producer `3a69f86b`. Each entry below
gives the number of sequences in
which both query answers were correct across model seeds 0, 1 and 2. The
three seeds share the same data. Each test split contains 64 sequences and
128 supervised objects.

| Case and mode | Train /128 | Novel validation /32 | Test at 24 /64 | Test at 48 /64 | Test at 96 /64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multi-query recall, Mini | 128 | 31 | 63 | 58 | 62 |
| Multi-query recall, forward-only Mini | 128 | 32 | 64 | 64 | 64 |
| Two-hop lookup, Mini | 107–111 | 3–6 | 10–12 | 6–10 | 0 |
| Two-hop lookup, forward-only Mini | 30–36 | 6–7 | 14–15 | 8–11 | 3–4 |

On multi-query recall, every forward-only endpoint was correct on every
supervised object and sequence in all 11 splits. Mini met the original 95%
criterion in all three runs, with 31/32 novel validation sequences. Its
length-24/48/96 test raw counts were 127/128, 122/128 and 126/128 in every
seed. Its length-48 and length-96 novel validation counts were each 30/32
exact and 62/64 raw. Forward-only Mini trained on length-24 examples and
transferred to all tested lengths. The stricter joint goal remains open for Mini.

The training answers covered 63 of the 64 possible values. Object 132 appeared
33 times in training inputs and zero times as a supervised answer. Every Mini
error at every measured length and seed was a query for this object. Mini
missed every such query and answered every query for the other values
correctly. Forward-only Mini also answered the queries for object 132
correctly. This breakdown identifies the observed failure set on data seed 1.

Two-hop lookup missed its original criterion in all six endpoints. At length
24 Mini learned 234–237 of 256 training answers but only 49–52 of 128 test
answers. Forward-only Mini learned 131–136 training answers and 51–61 test
answers. At length 96 the raw counts fell to 10–16 and 25–36 of 128 test
answers respectively. One stage of learned matching and relative retrieval
has yet to demonstrate composition of two table lookups.

Each forward-only endpoint issued 454,656 complete minibatch CE queries.
Both independently built arms reproduced all 12 trained parameter files,
training traces, original score reports, and 264 raw files for the initial
and trained models. An independent decoder checked each raw vector against
the original labels, whole-sequence outcomes and full-class CE. The held-out
scores of the original one-query binding probe also replayed after sharing
the runner. Across the three measured catalog cases, one passes the full
joint goal in both modes at all three seeds.

Recompute the per-object exposure breakdown and verify the original counts
and CE directly from the retained logits. Substitute `suite-lookup` in both
places for the second case.

```sh
cargo run --release --locked --offline --manifest-path experiments/learned_retrieval_transfer/Cargo.toml -- \
  suite-mqar audit-raw "$PWD" /run/media/misha/storage \
  /run/media/misha/storage/transformer-runs/learned-retrieval/sequence-transfer-v1/original/run/suite-mqar
```
