# Remaining synthetic catalog data

Task340 ports the last13 synthetic cases to Rust before new model work. It
constructs no model, supplies no features to a predictor and adds no accuracy
claim. The current measured scope remains58/165 cases, with107 unmeasured.
All23 reviewed additional cases from339 and the separate original35 comparison
remain complete inherited references.

| Family | Cases | Source lengths | Additional probes | Scoring |
| --- | ---: | --- | --- | --- |
| Count | 1 | 8–16 | 32,64 | Full generated interval and EOS |
| Addition | 8 | 2–4 digits | 8,16; forced carry chains4,8,16 | Full generated padded sum, hints where declared, and EOS |
| C-RASP | 3 | 8–16 input tokens | 32,64 | Every supervised input prefix |
| Random-LM control | 1 | 8 payload symbols | 16,32 | The independently sampled payload and EOS |

Addition crosses forward/reverse answers, plain/hinted formats and
standard/balanced carry sampling. Operands always remain most-significant digit
first, padded with one leading zero. The port consumes the original128 format
bits even without hints, seeds the independent hint RNG as a Python integer,
preserves each pair's length and final reversal, and retains all hard-carry
probes. The target oracle uses bounded exact integer arithmetic; a separate
digit-wise check verifies every generated addition split and prescribed chain.
Generation limits derive from the prompt format, independently of answers.

The C-RASP formula sampler preserves the seed0 witness bank, random draw order,
64-candidate nonconstant-selection limit and first accepting/rejecting witnesses.
Depth denotes syntactic count nesting, not a proved minimal language depth.
Recursive vector evaluation uses inclusive prefix sums. A separate postorder
machine recounts all past truth values, with an explicit work ceiling; tests
compare both on exhaustive short words, long words and overflow cases. Formula
metadata is marked as a data oracle and is unavailable through model inputs.

Random-LM retains8 training,64 validation and128 test examples and draws rows
individually. Its labels are independent random symbols, with no deterministic
prompt-to-answer oracle. Source novelty uses the full sampled teacher-forced
sequence, whereas ordinary generation uses prompts and C-RASP uses inputs.
No sampled label is replaced, inferred from the prompt or declared a success.

Seven native tests pass. The development history check matches all63 historical
test splits' example, target and transition denominators across all13 cases.
Historical raw tensor fingerprints are unavailable, so these are source-derived
datasets with reproducible native fingerprints, not a claim that historical
tensor artifacts were recovered.

Frozen producer `28dc5c946d360c9dfd8fc9593413a15280908c1d` preserves436 source
inputs. Its original and fresh independently rebuilt runs both exit0; their
complete inventories match SHA256
`a89e22f8896c693306c7eb19cdfa7ce9b960488285478e3cced4fb76d461d9b6`.
The original35 copies are also byte-identical. Runs take24.87s/24.68s and peak
at7,972,012/7,971,908KiB RSS, about7.60GiB. These inventories cover215 splits
across12 deterministic cases and the explicit random-label control. Data
preparation adds zero model evaluations;58/165 remains the measured scope.

The separate native `audit-retained` command reads each inventory back,
rehashes the entire canonical nested339 report and the original35 copy, checks
producer/binary/protocol bindings, and regenerates all13 case metadata records.
It also compares every complete copied historical record to the pinned source.
Both readbacks pass and match. Adding this read-only audit leaves the original
production binary byte-identical. Evidence is retained on storage under
`remaining-synthetic-data/{original-v1,review-v1}` and
`freezes/remaining-synthetic-data-v1`; generated inventories are not committed.

Source references are the task318 frozen catalog/plan and synthetic generator
files, authenticated by `reference-sources.sha256`. The catalog commit
`5a37c515639f23037d191b0659a5a0853984328d` belongs to `python_projects/diff-gpt`.
The task318 snapshot omitted `python/src/lab/domain/generative.py`; that file
matches Lean repository commit `0953b6a3bb62e55a03835e822ca21296cf104ec2` and the
reviewed337 data freeze byte-for-byte. It is included in this reference manifest.
Python files are reference text only and are never executed.

The full339 report is pinned by SHA256
`4501ba0f11e8e90da1f548b3f71fc619823ce2062e3f2039b0d57ce3be7d5e0d`.
The complete original35 summary is copied separately, without parsing it into
another resident tree. Before reading references, the inventory forecasts
64 times the baseline/history bytes, plus4GiB and the original35 file size;
it refuses above56GiB. Processes run under64GiB address-space limits. Buffered
output preserves complete canonical JSON without repeated small file writes.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/remaining-synthetic-data
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
manifest=experiments/convex_stress_remaining_synthetic_data/Cargo.toml
cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path "$manifest" -- --test-threads=2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path "$manifest" --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path "$manifest"
store=/run/media/misha/storage/transformer-runs
history="$store/stress-catalog/inputs-v1/diff-gpt/research/current_grokking/results/progress_snapshot.json"
binary="$CARGO_TARGET_DIR/release/convex-stress-remaining-synthetic-data"
prlimit --as=68719476736:68719476736 -- "$binary" check-history "$history" FRESH_CHECK_JSON
```

After committing and completing the direct Lean/global-axiom/INDEX/forbidden
checks, freeze the producer and run twice in fresh directories. Rebuild
independently and check binary equality before the second run. Keep every
failure and generated inventory on storage; commit only source/configuration,
required references and authored findings.

```sh
freeze="$store/freezes/remaining-synthetic-data-v1"
experiments/convex_stress_remaining_synthetic_data/freeze.sh "$freeze" "$binary"
mkdir -p "$store/remaining-synthetic-data"
for run in original-v1 review-v1; do
  if [ "$run" = review-v1 ]; then
    # Change only a timestamp to force an independent compilation.
    touch experiments/convex_stress_remaining_synthetic_data/src/lib.rs
    prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path "$manifest"
    cmp "$binary" "$freeze/convex-stress-remaining-synthetic-data"
  fi
  /usr/bin/time -v -o "$store/remaining-synthetic-data/$run.resources" \
    prlimit --as=68719476736:68719476736 -- \
    "$freeze/convex-stress-remaining-synthetic-data" prepare "$history" \
    "$store/frequency-programs/original-v1" "$store/remaining-synthetic-data/$run" \
    "$(cat "$freeze/producer.commit")"
done
cmp "$store/remaining-synthetic-data/original-v1/inventory.json" "$store/remaining-synthetic-data/review-v1/inventory.json"
cmp "$store/remaining-synthetic-data/original-v1/original35.json" "$store/remaining-synthetic-data/review-v1/original35.json"
for run in original-v1 review-v1; do
  prlimit --as=68719476736:68719476736 -- "$CARGO_TARGET_DIR/release/audit-retained" \
    "$store/remaining-synthetic-data/$run" "$freeze/convex-stress-remaining-synthetic-data" \
    "$(cat "$freeze/producer.commit")" "$history"
done
```
