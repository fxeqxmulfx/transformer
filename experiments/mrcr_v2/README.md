# MRCR V2 eight-needle GPT stress test

The `mrcr-v2-8needle` case evaluates ordinal recall in a long conversation.
Eight user requests share a writing format, topic and style. The final query
specifies which response to reproduce and supplies a twelve-character
certificate. The input is the complete published prompt, including its
few-shot examples. Each length regime receives its own mean match ratio.
[Google DeepMind benchmark](https://github.com/google-deepmind/eval_hub/tree/d5637d5cfb420a040ad550b2125c6aa23f71f9e4/eval_hub/mrcr_v2)

| Official eight-needle split | Rows | Published context lengths in the checked CSV |
| --- | ---: | ---: |
| `upto-128k` (cumulative) | 484 | 4,986–130,875 |
| `at-1m` (pointwise) | 310 | 1,048,380–1,048,388 |

The existing Lean formalization gives the occurrence-count meaning of an
ordinal query and a three-layer retrieval construction for tokenized
format, topic and style keys. The stress test measures generated text from
the model that produced the completions. [Ordinal semantics](../../src/Transformer/MRCR/Ordinal.lean) · [Style-key lookup](../../src/Transformer/MRCR/ReRopeStyleRecall.lean)

## Run

Build the Lean formalization and Rust scorer from the repository root.
The downloader pins the public object generation, byte size and SHA-256
of each CSV. The Rust commands verify the file again before reading any
examples. The two files occupy 1.64 GB together. [Source manifest](sources.tsv)

```sh
LAKE_ARTIFACT_CACHE=false lake build Transformer.MRCR
cd experiments/mrcr_v2
cargo +1.99.1 test --locked --release
cargo +1.99.1 build --locked --release
bin=./target/release/mrcr-v2
for split in upto-128k at-1m; do
    ./fetch.sh "$split" runs
    "$bin" verify "$split" "runs/$split.csv"
    "$bin" prompts "$split" "runs/$split.csv" > "runs/$split.prompts.jsonl"
done
```

Give each exported `prompt` to the model being evaluated as a single text
input. Preserve the full prompt and the supplied row ID, such as
`upto-128k/0`. Save one JSON object per line with fields `id` and `response`.
An empty completion is the empty string. The scorer requires every ID exactly
once, accepts responses in any order, and records the supplied model identifier
and tool-access setting. Repeat the command for each split.

```sh
split=upto-128k
"$bin" score "$split" "runs/$split.csv" "runs/$split.responses.jsonl" MODEL_ID no-tools
```

The released evaluator finds the last occurrence of the certificate in the
stripped completion. An absent certificate scores zero. It compares the text
after that occurrence to the reference body with the character-level
SequenceMatcher ratio, with the reference as the first argument and default
popular-character filtering on the second. Whitespace stripping, matching
ties and the 200-character filtering boundary are covered by native tests.
[Release evaluator](https://github.com/google-deepmind/eval_hub/blob/d5637d5cfb420a040ad550b2125c6aa23f71f9e4/eval_hub/mrcr_v2/run_evaluation.py)

The reference-answer and empty-answer controls scored 1.0 and 0.0 respectively
on both published splits. Repeat the full-data check after downloading them
with `MRCR_DATA_DIR=runs cargo +1.99.1 test --release --locked -- --ignored`.
The benchmark supplies evaluation conversations.
A Phasor score requires text completions from a named checkpoint on the full
split. The short and long training-suite totals count their own held-out
sequence-accuracy runs.
