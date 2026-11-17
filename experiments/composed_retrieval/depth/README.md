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
```
