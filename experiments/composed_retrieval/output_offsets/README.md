# Learning one output offset per original object

Can the two-read successor learn tasks where its angular readout consistently
underpredicts one object? On the original training records for modulus 97,
every mistake on division, multiplication and affine product had target
zero at every seed of the physical-depth reference.

| Training case | Rows with target zero | Errors at seeds 0 / 1 / 2 |
| --- | ---: | --- |
| Modular division | 52 | 31 / 39 / 37 |
| Modular multiplication | 98 | 51 / 64 / 67 |
| Affine product | 81 | 47 / 54 / 53 |

The new model learns a scalar for each original output object. It adds the
scalar to the scaled angular score on every prefix. All offsets start at
zero, use the existing scalar learning rate of 0.06 and stay between −16
and 16. The keys, tied values and readout, rotations, sequence interactions
and gates continue to train jointly from ordinary full-vocabulary
cross-entropy. With all offsets zero, every starting prediction agrees with
the log-odds reference. [Frozen protocol](protocol.json)

CM116 bounds every logit by 32 and gives the exact strict-margin conditions
for a one-class change at fixed base scores. The derivative is predicted
class probability minus the observed indicator. Each forward-only query
computes the full partition after moving a feasible coordinate. A common
shift is invisible to observed class contrasts. The coverage and loss
claims for the fitted joint model are empirical. [Flat-spectrum notes](../../../review/flat_spectrum.md) · [Lean statements](../../../src/Transformer/Grokking/FlatSpectrum/ConvexChannel/ComposedRetrieval/OutputOffsets.lean)

One cohort contains the three retrieval controls and the 48 available
original arithmetic tasks, including the shared addition control once.
Six unavailable original arithmetic sources retain their entries. Each
build fits the same 51 cases, two modes and three seeds for 1,024 updates.
Both builds must seal all 306 endpoints before either scores a held-out
record. The independent decoder checks the original pass criterion and
99% object and exact-sequence accuracy on every applicable held-out split.

```sh
base=/run/media/misha/storage/transformer-runs
git show <producer-commit>:experiments/composed_retrieval/run.sh | bash -s -- \
  "$PWD" /run/media/misha/storage <producer-commit> "$base/composed-retrieval/<fresh-output>" \
  output-offsets "$base/arithmetic-data/inputs-v1"
"$base/composed-retrieval/<fresh-output>/original/target/release/audit-raw" \
  "$PWD" /run/media/misha/storage "$base/composed-retrieval/<fresh-output>"
```
