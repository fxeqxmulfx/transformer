# Learned object magnitudes in the product channel

Can ordinary training exploit a separate magnitude for each opaque object in
the rotated product? CM117 gives each object a scalar in the closed unit
interval. The retrieved values are scaled before the normalized first read
and the query scales its rotation. The existing representation, two unrotated
reads and output offsets still participate in the complete score. Setting
every radius to one reproduces the CM116 logits.
[Physical operator and identifiability](../../../review/convex/moments.md)

## Training-only intervention on all 48 original arithmetic cases

The diagnostic takes frozen CM116 models and their original masked training
pairs. It recomputes every training logit and checks it bit-for-bit against
the archived full-vocabulary array. For a model with errors, it selects the
input object appearing in the most mistaken rows and the target class most
often missed. Ties use the smaller original ID. A repeated operand counts
once in the selection rule and twice in the actual product. The intervention
sets the selected input radius to 1, 0.75, 0.5, 0.25 or 0, keeping every
other radius at one. For each value it also checks whether an adjustment
of the selected output offset could make every training margin strictly
positive inside its existing absolute box.

Sixty of the 288 fitted models already classified every training pair
correctly. Among the remaining 228, at least one tested radius increased
the number correct for 106 and lowered ordinary CE for 74. A radius alone
removed every error for eight models. Allowing the bounded change in one
output offset raised this to ten. The other 218 had no such strict repair
on the tested grid. With every radius at one, adjusting the selected offset
alone yielded zero strict repairs.

| Original training task | Targets | Correct at radius 1, seeds 0 / 1 / 2 | Correct at radius 0, seeds 0 / 1 / 2 |
| --- | ---: | --- | --- |
| Multiplication modulo 97 | 4,704 | 4,656 / 4,678 / 4,649 | 4,704 / 4,704 / 4,694 |
| Division modulo 97 | 4,656 | 4,652 / 4,630 / 4,641 | 4,656 / 4,654 / 4,656 |
| Affine product, standard minibatch | 4,704 | 4,684 / 4,699 / 4,698 | 4,647 / 4,656 / 4,650 |

Both minibatch modes gave the displayed multiplication and division counts.
In both tasks training errors selected input ID 0 and output ID 0 at every
seed. Multiplication training CE at radius zero fell from 0.0857 to 0.0511,
0.0522 to 0.0279, and 0.1207 to 0.0825. On division at seed 1, an additional
output-offset displacement of −1 lies within the feasible strict interval
and corrects both remaining errors. On affine product the
selection included input ID 96 at seed 1. Its training CE rose from 0.0235
to 0.0965 at radius zero. These manual training interventions contribute no
additional held-out benchmark passes.

The independently trained original and repeat arms returned identical
reports. The command reads the endpoint receipts, frozen model and training
logits only. The Rust data adapter restores the original ordered pairs.

```sh
base=/run/media/misha/storage/transformer-runs/composed-retrieval
for arm in original repeat; do
  cargo run --release --locked --offline --manifest-path experiments/composed_retrieval/Cargo.toml \
    --features output-offsets --bin product-radii -- "$PWD" /run/media/misha/storage \
    "$base/output-offsets51-v1" "$arm" > "/tmp/product-radii-$arm.json"
done
cmp /tmp/product-radii-original.json /tmp/product-radii-repeat.json
```

## Joint training on 64 original cases

Each independent build fits all 64 distinct cases from the arithmetic,
aligned-prefix and Rewrite studies. The 48 available arithmetic cases,
query-specific binding, nine aligned synthetic cases and six greedy Rewrite
tasks use one architecture. Six further arithmetic tasks have unavailable
original inputs. The native inventory checks the original plan and all
source fingerprints before training. The upper bound on raw logit storage
for both builds is 18,528,751,296 bytes. [Frozen protocol](protocol.json)

Every radius starts at one. Each update trains it together with the original
phases, routing, gates and class offsets by ordinary full-vocabulary CE.
The radius uses projected Adam with the existing gate learning rate of
0.01. In forward-only minibatches the two loss probes use feasible
displacements and the actual distance between them. A repeated operand
affects both the retrieved value and the query. At either boundary the
projection still admits an inward loss probe.

Both complete 384-fit builds seal before any held-out score is read. The
matched comparison checks every initial raw array, the training minibatch
order, the extra loss queries, each original criterion and the 99% raw
and exact targets. It also resets all fitted radii to one on the original
training rows to measure their direct effect with the other weights fixed.

```sh
base=/run/media/misha/storage/transformer-runs
out="$base/composed-retrieval/learned-radii64-v1"
producer=$(git log --diff-filter=A --format=%H -- experiments/composed_retrieval/product_radii/protocol.json)
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  "$producer" "$out" radii-cohort "$base/arithmetic-data/inputs-v1"
"$out/original/target/release/compare-learned-radii" "$PWD" /run/media/misha/storage \
  "$base/composed-retrieval/output-offsets51-v1" "$base/composed-retrieval/output-prefix9-v1" \
  "$base/composed-retrieval/output-rewrite7-v1" "$out"
```

## Held-out results

Learning one radius per object changed which arithmetic case passed.
Multiplication modulo 97 crossed 99% at all three seeds in both
optimizers. Division modulo 193 fell to 98.70% at seed 0 in both.
The complete 64-case cohort has **10 original passes and 8 strict
passes** under each architecture. A case must pass at all three seeds
in both modes. The strict count requires at least 99% raw objects and
99% exact sequences on every held-out split.

| Original validation set | Examples per seed | CM116 exact, seeds 0 / 1 / 2 | CM117 exact, seeds 0 / 1 / 2 |
| --- | ---: | --- | --- |
| Addition modulo 97 | 4,705 | 4,705 / 4,705 / 4,705 | 4,705 / 4,705 / 4,705 |
| Subtraction modulo 97 | 4,705 | 4,705 / 4,686 / 4,704 | 4,705 / 4,705 / 4,705 |
| Multiplication modulo 97 | 4,705 | 4,596 / 4,612 / 4,577 | 4,701 / 4,700 / 4,681 |
| Affine product `a + ab` modulo 97 | 4,705 | 4,618 / 4,652 / 4,629 | 4,643 / 4,660 / 4,675 |
| Division modulo 97 | 4,656 | 4,624 / 4,591 / 4,617 | 4,574 / 4,654 / 4,651 |
| Division modulo 193 | 18,528 | 18,429 / 18,358 / 18,436 | 18,287 / 18,473 / 18,468 |

Both optimizers gave the same counts in the table. Across the 384
case, seed and optimizer combinations, original passes rose from 66
to 74 and strict passes rose from 54 to 62. Four of the six affine
product fits crossed 99%. Binding remained
exact on all 512 novel validation examples and all 512 test examples.
Two-hop lookup generated 64/64 correct sequences at each of lengths
24, 48 and 96. Distinct-symbol copy generated 32/32 novel validation
sequences and 64/64 ordinary test sequences, then 0/64 at length 32
and 0/64 at length 64. Multi-query recall met its original 95%
criterion with 32, 32 and 31 exact novel validation sequences out of
32. Three constant-answer controls are included in both pass counts.

With the fitted weights held fixed, resetting the radii to one reduced
the number of correct training objects in 242 of 384 fits, increased it
in 12 and preserved it in 130. Keeping the learned radii gave lower
training cross-entropy in 274 fits, higher in 12 and equal in 98.
This intervention uses the original training records.

Two independent builds of `e5bb48c7` sealed 768 fits before scoring.
Their 6,360 raw files matched and replayed. The parent comparison matched
all 1,590 initial raw files and every training minibatch fingerprint.
The full 165-case catalog
has 64 measured cases, six with unavailable original arithmetic data,
and 95 outside this comparison.
