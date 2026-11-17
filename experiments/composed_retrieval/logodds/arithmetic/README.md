# Arithmetic after changing the depth coordinate

Do the arithmetic results reproduce with the training setup that learned
two-hop lookup? On a two-object prefix the memory has a single destination.
CM113's `score_pair` theorem shows that the real-valued logits depend on
the learned value and rotation tables and the product gate. Both reads
retrieve the same value at any depth. Floating-point evaluation and
optimization can still give different endpoints, so this study refits the
entire family. [Pair score](../../../../src/Transformer/Grokking/FlatSpectrum/ConvexChannel/ComposedRetrieval/Geometry.lean)

Both standard minibatch training and forward-only minibatch training use
the original 1,024 updates, seeds 0, 1 and 2, and initial log odds of
log 9. The 54-case source inventory supplies 48 reproducible original
datasets and six named missing tables. Both builds verify the whole source
inputs, train all available cases, seal their endpoints, and score all
original rows. The overlapping addition runs must reproduce the same
four-case study before combining coverage. [Protocol](protocol.json)

```sh
base=/run/media/misha/storage/transformer-runs
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  <producer-commit> "$base/composed-retrieval/<fresh-output>" logodds-arithmetic \
  "$base/arithmetic-data/inputs-v1"
cargo run --release --locked --offline --features arithmetic-cohort,logodds-depth \
  --manifest-path experiments/composed_retrieval/Cargo.toml --bin reconcile-arithmetic -- \
  "$PWD" /run/media/misha/storage "$base/composed-retrieval/logodds-depth-v1" \
  "$base/composed-retrieval/<fresh-output>"
```
