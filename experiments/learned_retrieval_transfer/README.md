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
