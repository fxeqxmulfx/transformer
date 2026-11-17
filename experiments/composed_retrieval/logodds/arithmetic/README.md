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
  "$base/composed-retrieval/<fresh-output>" "$base/composed-retrieval/arithmetic54-v1"
```

## Results

Six of 51 measured cases met both their original criterion and the 99%
raw-object and exact-sequence target on every required held-out split,
at all three seeds in both modes. The six were key/value binding, two-hop
lookup, modular addition, subtraction, division and the modulus-97
power-zero constant control. The other 45 measured cases missed at least
one requirement. Six arithmetic sources were unavailable and 108 other
cases have yet to enter this model's cohort.

| Full 51-case comparison | Original passes | Strict passes |
| --- | ---: | ---: |
| Physical depth, initialized at 0.5 | 6 | 5 |
| Log-odds depth, initialized at log 9 | 6 | 6 |

Two-hop lookup gained both passes. Multi-query recall lost its original
pass at seed 2, with 30/32 exact on novel validation. All 48 arithmetic
cases retained their original and strict outcomes. Their training and
validation counts also agreed at every matched seed and mode. Comparing
the final raw arithmetic predictions across depth parameterizations gave
zero changed decisions on 1,630,992 training targets and one on 1,631,190
validation targets. On that modulus-17 power-division row both forward-only
seed-1 models chose a wrong class. The largest absolute change in a final
arithmetic logit was 0.0387 or less. Counting each arm at its own training
protocol, standard and forward-only training differed on two of 1,631,091
arithmetic decisions in the log-odds study.

The two builds of `280f1eb4` sealed 576 endpoints and replayed 2,304 raw
files. The common addition control matched its six trained models,
training receipts and all 48 initial and final raw files in the separate
four-case study. The primary driver stopped after the final fit when its
working copy changed. The archived driver resumed at sealing, before any
scores had been written, and rechecked the frozen source and executables.
An uninterrupted end-to-end rerun reproduced all 288 models in each build,
all optimizer receipts and every decoded count and raw score file.
