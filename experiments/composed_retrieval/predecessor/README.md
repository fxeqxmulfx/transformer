# Learning which occurrence of a repeated object to read

Can the object preceding a query help select between two occurrences of
the same object in memory? In CM117 the relative content weights of those
occurrences are independent of the query. CM120 adds one learned scalar
times the mean cosine similarity of the preceding objects' learned keys.
The comparison is zero when either position has no predecessor. The
outer query and every inner query share this scalar and the entire key
table. The model's input consists of the original objects, and the
training loss supervises only the next original object.

| Setting | Learned contextual comparison | Quantity measured |
| --- | --- | --- |
| Archived CM117 endpoint | Zero | Every original teacher-forced and freely generated training logit |
| Local perturbation | Two opposite displacements of magnitude $2^{-16}$ | Full training CE slope, also computed by differentiating both physical reads |
| Fixed training grid | −4, −2, −1, 0, 1, 2, 4 | Full training CE, teacher-forced objects, greedy objects and exact sequences |

Two observed content rows identify the coefficient at fixed keys when
the four predecessor similarities have a nonzero contrast. Observing
every final class score on prefixes of length at most two still leaves
opposite winning predictions possible on a length-three prefix. The
full logit bound holds for every prefix length, and CE is convex in the
class offsets when the other parameters are fixed. [Flat-spectrum CM120](../../../review/flat_spectrum.md)

## Training results

Both archived builds reproduced all 42 endpoints' original training
vectors at zero. The largest discrepancy between the analytic slope
and the two complete loss queries was below $1.12\times10^{-10}$. Both
optimizers gave the same exact-sequence counts at each seed.

| Training case | Exact sequences at zero, seeds 0 / 1 / 2 | Best exact count on the grid, seeds 0 / 1 / 2 |
| --- | --- | --- |
| Copy, repeated symbols | 2 / 0 / 3 | 2 / 0 / 4 |
| Reverse, repeated symbols | 0 / 0 / 0 | 1 / 2 / 0 |
| Sort, repeated symbols | 0 / 0 / 0 | 3 / 1 / 2 |
| Copy, distinct symbols | 128 / 128 / 128 | 128 / 128 / 128 |
| Reverse, distinct symbols | 94 / 91 / 95 | 94 / 91 / 95 |
| Sort, distinct symbols | 0 / 0 / 0 | 0 / 0 / 0 |
| Two-hop lookup | 128 / 128 / 128 | 128 / 128 / 128 |

Each denominator is 128. Every one of the 1,690 repeated-copy training
prefixes contained two occurrences of a symbol with distinct preceding
objects. All six repeated-copy fits had a positive CE derivative at zero,
between 0.00507 and 0.01676. Zero nevertheless had the lowest CE among
the seven grid settings in those fits. In 12 of 42 fits a grid setting
lowered CE, all on distinct-symbol copy or lookup, where all 128 training
answers were already correct. The table measures a change to one
coefficient at fixed trained weights. Held-out generalization requires a
separate fit of the complete model.

```sh
base=/run/media/misha/storage/transformer-runs
parent="$base/composed-retrieval/learned-radii64-v1"
manifest=experiments/composed_retrieval/Cargo.toml
for arm in original repeat; do
  cargo run --release --offline --locked --manifest-path "$manifest" \
    --features radii-cohort --bin predecessor-signal -- "$PWD" /run/media/misha/storage \
    "$parent" "$arm" > "$base/predecessor-signal-$arm.json"
done
cmp "$base/predecessor-signal-original.json" "$base/predecessor-signal-repeat.json"
```
