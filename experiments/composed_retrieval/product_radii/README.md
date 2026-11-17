# Learned object magnitudes in the product channel

Can ordinary training exploit a separate magnitude for each opaque object in
the rotated product? CM117 gives each object a scalar in the closed unit
interval. The retrieved values are scaled before the normalized first read
and the query scales its rotation. The existing representation, two unrotated
reads and output offsets still participate in the complete score. Setting
every radius to one reproduces the CM116 logits.
[Physical operator and identifiability](../../../review/flat_spectrum.md)

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
