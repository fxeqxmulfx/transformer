# Greedy copy, reverse and sort from original object prompts

Can the CM116 successor produce an entire answer after seeing only the
original prompt? This cohort includes all six Rewrite cases and repeats
two-hop lookup from the previous 58 cases. Each fit learns its object
representation, sequence interactions and full-vocabulary readout with the
same initialization law, optimizer and 1,024 updates as the previous cohort.
The model receives a sequence of original object IDs and emits one next ID.

| Original operation | Repeated-symbol training targets | Unique-symbol training targets |
| --- | ---: | ---: |
| Copy | 1,690 | 1,732 |
| Reverse | 1,606 | 1,714 |
| Sort | 1,742 | 1,678 |

The reviewed Rust source reconstructs the 128 training examples, the clean
training copy, 32 validation and 64 test examples, both longer word lengths
(32 and 64) and the original novel-prompt validation subsets. The six
complete metadata hashes include the original adapter's tensor hashes,
generation bounds and complete-record hashes for every split. The compact
November 15 archive preserves this metadata at SHA-256
`267d5dee6c378a7faf4133fbbfa967a1a75545fa588b5f3a0bcb1fbd6daf2f65`.
Historical per-example tensors from the original model runs are unavailable.
[Original generation adapter](../../convex_stress_generative_data/README.md)

For a word of length $n$, the prompt has $n+2$ objects and the original
limit is $n+1$. Training masks the first $n+1$ positions. The final prompt
position predicts the first answer object and the last target is EOS.
At evaluation, the generator sees
only the prompt and its own earlier predictions. It stops at EOS or the
input-derived limit. Each missing output after an early EOS counts as
incorrect. The longest permitted input to a single prediction is 130 objects.
Both the original 95% exact criterion on novel validation and the separate
99% raw-object and exact criteria use free generation. Ordinary CE and
teacher-forced accuracy appear separately in the reports.

The raw file preserves every full-vocabulary vector in original example
order, first the teacher-forced calls and then the free-generation calls for
that example. The independent decoder reconstructs the input to each call
from the original prompt and the archived winning objects, and checks its
digest against the live predictor's input trace. It checks counts, EOS,
transitions, per-class accuracy and teacher-forced CE. The lookup control
retains its original file format for bytewise comparison.

The bounded-score theorem applies on every prefix the model writes. With
capacity 16 and offsets bounded by 16, each logit has absolute value at
most 32. This alone leaves the winning object undetermined. Because the
predictor is deterministic and the reference terminates at the allowed
limit, a row is exactly correct in free generation if and only if every
teacher-forced winner matches. Both paths share the prefix through their
first error. Later per-object accuracy can differ as soon as the model
appends an incorrect object. The audit checks the exact counts agree on
each split. [Flat-spectrum CM116](../../../review/flat_spectrum.md)

Each independent build fits 42 endpoints at seeds 0, 1 and 2, using
analytic and forward-only minibatches of 32 original examples. All 84
endpoints seal before either build scores a held-out split. The preflight
uses 4,077,542,400 bytes as the upper bound for archived full-vocabulary
vectors. Early EOS can reduce the actual size. [Frozen protocol](protocol.json)

```sh
base=/run/media/misha/storage/transformer-runs
manifest=experiments/composed_retrieval/Cargo.toml
producer=$(git log --diff-filter=A --format=%H -- experiments/composed_retrieval/rewrite/protocol.json)
out="$base/composed-retrieval/output-rewrite7-v1"
cargo run --release --locked --offline --manifest-path "$manifest" --features rewrite-cohort \
  --bin prefix-inventory -- "$base/arithmetic-data/inputs-v1"
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  "$producer" "$out" rewrite-cohort \
  "$base/arithmetic-data/inputs-v1"
"$out/original/target/release/reconcile-rewrite" \
  "$PWD" /run/media/misha/storage "$base/composed-retrieval/output-offsets51-v1" \
  "$base/composed-retrieval/output-prefix9-v1" "$out"
```
