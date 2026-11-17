# Learned next-object retrieval

Can the CM111 head learn to return the value attached to the queried key on
held-out original prefixes? In the earlier content-growth experiment, the
old-geometry models chose correctly between two distinct stored values on
49.90% of novel examples. CM111 gives a sufficient margin in terms of learned
key, relative-position and decoder gaps. This experiment tests whether ordinary
training produces the required predictions. [Earlier diagnosis](../convex_binding_diagnostic/README.md)
and [CM111](../../review/flat_spectrum.md#cm111-shared-learned-matching-with-a-learned-relative-position-link).

For a prefix ending in object q, use all preceding objects x. Learn two complex
eight-dimensional vectors per original object. K serves both queries and keys.
V serves both input values and all output-class readouts. The first softmax
weights each j by a learned real scale times the real inner product of K[q]
and K[x_j]. A second softmax over k uses a learned real potential at k−j.
These path weights pool V[x_k]. The logit for each c is 16 times the real
inner product of V[c] with the pooled state. Both softmaxes include every
past position. Empty memory produces zero logits. Object IDs address rows.
The predictor receives the prefix alone.

This is a single generator of the CM111 functional convex hull. Optimizing
its embedding coordinates is a factor-training problem. The complete unit-ball
projection and the inner products use every real and imaginary coordinate.
The cap of 12 objects gives 21 relative offsets. Including the key scale,
the model has 3,222 trainable reals for 100 objects. Each seed initializes
both tables independently at radius 0.75, the key scale at 2, and every offset
at zero. The architectural ties favor matching identical objects. Training
must discover which relative positions to use. The original target occurs
only in the ordinary full-vocabulary loss during fitting.

| Training mode | Update on each original minibatch |
| --- | --- |
| Mini | Differentiate full CE through both softmaxes, both uses of each tied table and every readout. Apply Adam to every coordinate and project the embedding rows. |
| Forward-only Mini | For each queried coordinate, evaluate the full minibatch loss at both projected signed perturbations of size 0.0001. Update every relative potential and the key scale, plus 16 coordinates of each embedding table on a shuffled cycle. Each coordinate has its own Adam clock. |

Both modes use the original data seed 1, training seeds 0, 1 and 2, batches
of 32 and 1,024 updates (16 epochs). They share initial parameters and
minibatch order at each seed. The historical transformer's first recorded
seed-1 crossing was at 500 updates. The fixed endpoint here compares the two
new training modes. At a projection boundary the signed feasible CE difference
estimates a direction on the constrained model. The complete analytic-gradient
comparison uses interior weights.

The runner builds from two independent extractions of the pinned commit.
All six endpoints in each extraction seal before either evaluation. It compares
parameters, training traces, every original metric and full raw logit bytes,
then reruns scoring in separate processes. Each seeded initializer is also
scored. The original metric is sequence accuracy on novel validation, with a
99% threshold. The report retains all 2,048 training, 512 validation and 512
test examples, the supplied novel-validation alias and all 100 classes.
Two-value choice, outside-value errors and counts of previously unseen queried
pairs are calculated only after sealing. A pair may be absent from supervised
training answers yet present in the input memory of a training example. Both
definitions are reported. The CM111 sufficient margin is evaluated numerically
on the actual path and compared with the measured complete-class margin.

Run from the workspace root with the producing commit and a fresh output directory.

```sh
bash experiments/learned_retrieval/run.sh "$PWD" /run/media/misha/storage PRODUCER \
  /run/media/misha/storage/transformer-runs/learned-retrieval/binding-v1
```

## Results on the original binding probe

Producer `339ba587` reached **512/512 exact on novel validation and 512/512
on test** for each of the six trained endpoints. All also scored 2,048/2,048
on train. Each original example has one supervised token, so the raw and
exact counts coincide. The original success and both 99% criteria hold on
every held-out split. The initializers scored 67–75/512 on novel validation
and 62–92/512 on test across the three seeds.

| Endpoint across seeds 0, 1 and 2 | Novel validation CE | Test CE | Mean mass on the correct two-stage path on novel validation |
| --- | ---: | ---: | ---: |
| Mini | 0.0000825–0.0000868 | 0.0000831–0.0000872 | 0.9449–0.9463 |
| Forward-only Mini | 0.002147–0.002637 | 0.002133–0.002777 | 0.9932–0.9944 |

Every endpoint selected the correct value on all 498 distinct-value novel
examples and all 489 distinct-value test examples. Across the full splits it
also answered all 71 novel and 65 test queries whose key/value pair had never been supervised during
training. Under the stricter definition requiring that pair to be absent from
every training memory, all nine novel and all five test queries were correct.
The 512 novel and 512 test examples are fixed across seeds.

All four parameter groups moved in both modes. Each forward-only run made
110,592 complete minibatch loss queries and visited every embedding coordinate
at least ten times. The learned scale rose from 2 to 5.34–5.54 in Mini and
10.47–10.52 in forward-only Mini. The two modes learned different offset
tables. Mini ranked +1 highest over all offsets. Forward-only Mini ranked
−4 or −5 highest globally. Those negative offsets are invisible from the two
original key positions, 1 and 3. Its mean mass on the link from the matched
key to the correct value nevertheless exceeded 0.998 on novel validation.
Transfer to other key positions and overwritten keys requires separate data.

The numerical CM111 sufficient bound was positive on 453–480 of 512 novel
examples in Mini and 502–510 in forward-only Mini. All observed complete-class
margins satisfied the checked inequality within the declared floating-point
tolerance. The independent rebuilds had different executable hashes. All six
trained parameter files, training traces, 48 initial and trained raw-logit files
and the full score report were byte-identical across runs. Separate-process
replays matched. The score report SHA256 is
`bba23ff02967adba93061f6f8816155d200b1ad03634a886539a717705a09fab`.
This contributes one original case to the shared model's own 165-case tally.
