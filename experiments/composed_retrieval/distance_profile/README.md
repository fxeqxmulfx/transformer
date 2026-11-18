# Learning a relative-position rule that extends to longer sequences

Does a signed geometric position profile improve original free generation
while preserving unseen key/value binding, lookup, multi-query recall and
modular addition? The earlier model learned an independent potential for
every signed distance. Its short copy data accessed 65 of 257 entries
and gave perfect training accuracy across all seven tested decay settings.
[Training measurements](../distance_decay/README.md)

| Component | Parent CM117 | Signed profile CM119 |
| --- | --- | --- |
| Relative logits of each positional row | One learned value per signed offset | Center zero, separate learned adjacent levels and nonnegative decays for the two directions |
| Relative parameters at context 130 | 257 | 4 |
| Relative parameters at context 2 | 1 | 0 |
| Initial positional weights | Uniform on visible memory positions | Uniform on visible memory positions |
| Representation, optimizer and target | All other coordinates fit from ordinary masked CE | Same initialization, minibatch sequence, learning rates and 1,024 updates |

At positive distance $h$ from a source position the relative logit is
$b_+-a_+(h-1)$. Negative offsets use $b_-,a_-$. With nonnegative decay,
each additional step in that direction multiplies the positional weight
by a constant in $(0,1]$. Both normalized reads use the same parameters.
The analytic gradient sums each position's contribution to its direction's
level and decay. The forward-only optimizer makes two feasible loss queries
for every independent coordinate on every minibatch. The test computes both
updates against a separate implementation of the full class logits and
includes slopes at zero. [Flat-spectrum CM119](../../../review/flat_spectrum.md)

The ten original cases are repeated- and distinct-symbol copy, reverse and
sort, novel key/value binding, two-hop lookup, multi-query recall and
addition modulo 97. Each model sees only original object prefixes and
predicts one object from the original vocabulary. All three seeds and both
training methods use the original training rows, preserving target masks
and multiplicities. The preflight matches every included source entry
and all six missing-source entries against the verified parent inventory.
The upper bound for raw full-vocabulary arrays is 4,361,893,056 bytes.

Both independent builds seal all 120 fits before any original held-out
scoring. The comparison checks the initial logits byte-for-byte for every
split, all sampled-training receipts and the complete raw predictions of
the singleton-memory arithmetic control after fitting. The 165-case
report keeps the signed profile's measured cases and the old model's
matched scores under their respective model identities.

```sh
base=/run/media/misha/storage/transformer-runs
out="$base/composed-retrieval/signed-profile10-v1"
producer=$(git log --diff-filter=A --format=%H -- experiments/composed_retrieval/distance_profile/protocol.json)
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  "$producer" "$out" profile-cohort "$base/arithmetic-data/inputs-v1"
"$out/original/target/release/compare-signed-profile" "$PWD" /run/media/misha/storage \
  "$base/composed-retrieval/learned-radii64-v1" "$out"
```
