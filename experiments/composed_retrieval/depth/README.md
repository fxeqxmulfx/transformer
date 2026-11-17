# Training signal for two-hop lookup

All six original sparse fits ended with zero weight on the second read.
At initialization, every relative-position score is zero. CM114 proves
that both reads then yield the same value average and that the ordinary
cross-entropy derivative in depth and content-routing directions is zero.
Once relative scores differ, the label and the full predicted distribution
determine the sign of the depth derivative. [Flat-spectrum derivation](../../../review/flat_spectrum.md#cm114-what-the-two-hop-gate-can-learn-from-ordinary-cross-entropy)

A read-only observer replays the 1,024 original minibatches per fit with
both training modes and seeds 0, 1 and 2. At each update it records the
executed estimate, analytic depth slope, feasible secant, mean class
direction range, and root mean square of the visited key and relative-position
estimates. With the other learned weights fixed it evaluates complete
predictions at depths zero, one half and one. The reported margins are
the true target logit minus the highest competing logit.

The replay checks each observed depth estimate against the appropriate
full-vocabulary quantity, each original optimizer receipt and the final
model bytes. The complete training set supplies a second measurement at
the frozen endpoint. Two independent builds compare every observed
minibatch and endpoint. The observer never supplies its measurements to
the optimizer or predictor.

```sh
bash experiments/composed_retrieval/depth/run.sh "$PWD" /run/media/misha/storage \
  <diagnostic-commit> <original-four-sparse-output> <fresh-diagnostic-output>
bash experiments/composed_retrieval/depth/summarize.sh <fresh-diagnostic-output>
```

## Results on the six original fits

On the first minibatch, the maximum mean class-direction range over the
six fits was 1.4 × 10⁻¹⁵. All six already had nonzero relative-position
estimates. By minibatch two every fit had a row with class directions
differing by more than 10⁻¹⁰. Depth reached zero during the first 51 updates
and stayed there. Of 5,853 subsequent or boundary minibatches measured at
zero, 5,852 had a positive depth slope and one had a negative slope.

| Mode and seed | First minibatch starting at zero | Full training slope at zero | Training CE at depth 0 / ½ / 1 | Correct objects at depth 0 / ½ / 1, of 256 |
| --- | ---: | ---: | --- | --- |
| Standard 0 | 52 | +0.330 | 0.587 / 0.958 / 1.872 | 217 / 166 / 81 |
| Standard 1 | 50 | +0.294 | 0.328 / 0.750 / 2.204 | 242 / 196 / 68 |
| Standard 2 | 52 | +0.360 | 0.562 / 0.996 / 2.069 | 223 / 155 / 75 |
| Forward-only 0 | 49 | +0.283 | 1.800 / 2.074 / 2.638 | 119 / 112 / 67 |
| Forward-only 1 | 47 | +0.337 | 1.820 / 2.088 / 2.587 | 131 / 120 / 70 |
| Forward-only 2 | 47 | +0.370 | 1.828 / 2.146 / 2.760 | 129 / 102 / 52 |

The final product gate was zero in all six fits. Holding every other
weight fixed, the learned second read scored the target worse on average.
Both the analytic derivative and a direct feasible loss difference favored
depth zero on all 256 training targets taken together in each fit. CM114's
tangent bound makes a nonnegative full training slope at zero sufficient
for optimality along the entire depth interval. Jointly changing the
learned routing and depth is a different optimization problem.

The two builds of `d3dcdc09` reproduced every minibatch observation and
every optimizer receipt and final weight from original producer `0f0aadbd`.
The raw observations are in
`/run/media/misha/storage/transformer-runs/composed-retrieval/depth-signals-v1`.
