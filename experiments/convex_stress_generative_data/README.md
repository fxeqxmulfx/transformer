# Original generative sequence data

Can the convex complex model reach99% raw-token and exact-sequence accuracy on
the original copy, reverse and sort tasks? This Rust data-only adapter prepares
six further cases from the165-case catalog. It trains and scores no new learned
model. Task328 will audit the causal Lean contract and evaluate a model after
this data adapter is reviewed. The separate original35 comparison and the six
reviewed retrieval/prefix cases remain unchanged.

| Case | Input sampling | Historical GPT test correct / targets | Historical exact /64 |
| --- | --- | --- | --- |
| Copy repeat | Two-symbol pool, with repetition | 409/772 | 2/64 |
| Copy unique | Without replacement from64 symbols | 606/842 | 29/64 |
| Reverse repeat | Two-symbol pool, with repetition | 318/842 | 0/64 |
| Reverse unique | Without replacement from64 symbols | 58/842 | 0/64 |
| Sort repeat | Pool of2–8 symbols, with repetition | 204/828 | 0/64 |
| Sort unique | Without replacement from64 symbols | 412/860 | 12/64 |

All six historical endpoints retain `budget_exhausted`, their original flags,
ordinary CE, teacher-forced metrics and free-generation results. Their model
identity is `01f18f86787511279fe52fe41883495fce0e3105`; the catalog source is
`5a37c515639f23037d191b0659a5a0853984328d`. They are historical references,
not a fresh matched-budget GPT experiment. The complete reviewed326 report
with all six earlier passes and all earlier failed baselines is also retained.
Its SHA256 is `9fcfd6aee0dbcba71c4a9dd51bf78ac631ef64f7225a291d9b9620ed529996f9`.
Existing model files remain preserved in the reviewed326 run directory; this
data inventory copies the complete report and its recorded state identities.

The native port follows the frozen `synthetic/{sequences,base,records,splits}.py`
sources read as reference only. It reuses the reviewed Python-compatible native
RNG and exact JSON/tensor serialization from `convex_stress_sequences`. It keeps
the unused128-bit format-seed draw, both independently sampled problems at one
random length, the final random pair reversal, and the original sampling branch
for unique symbols. `reference-sources.sha256` pins all ten direct reference
files; the freeze also collects reference manifests from transitive packages.
No Python is executed.

Every case uses data seed1,128 training,32 validation and64 test examples,
word lengths8–16, and original longer word lengths32 and64. The vocabulary
has100 raw classes. Each prompt contains BOS, the word and SEP. The answer
contains the rewritten word and EOS. Teacher-forced inputs contain the prompt
and all answer tokens except the last; each answer token is supervised at the
preceding input position. Thus longer contexts reach66 and130 tokens, and
generation limits are33 and65. Both longer test sets have64 examples and
2112/4160 supervised answer tokens, including EOS.

The original clean training copy and all novel-validation subsets are retained.
Novelty is determined by the raw prompt, independent of the answer. Selection
uses novel validation when present. Every case has11 splits. Metadata includes
original-style source-derived split hashes, padded tensor hashes and complete
generated-example hashes, so prompts, answers and EOS limits remain bound.
The inventory checks all historical test example/target counts against the port.
Historical per-example dataset hashes and checkpoints are unavailable; matching
aggregate counts and source-derived hashes do not establish past byte identity.

The scoring helper ports `synthetic/generation.py` directly as a scalar causal
loop: each next prediction receives its own growing prefix; EOS stops it, and
the prompt-derived limit bounds it. Only the separate scorer sees the answer.
Raw correct tokens use the full expected-answer denominator, including EOS;
exact success requires the entire answer at the correct length. Ending rate,
generated tokens, extra tokens and transition accuracy counts remain available.
Teacher-forced ordinary CE must remain a separate metric in the later model
runner. The original95% exact threshold and new99% raw/exact goals are separate.

Seven native checks cover historical denominator agreement, exact causal masks,
independent positional/multiset answer oracles, repeated/unique symbols, symbol
renaming, deterministic stream prefixes, split bounds and immutable source pins.
Generation fixtures verify that an incorrect token becomes the next visible
prefix, that teacher forcing and generation differ, and that early/missing/late
EOS cannot receive an exact pass. These are source-conformance tests, not new
model accuracy measurements. Copy/reverse commute with arbitrary symbol renaming;
sort retains its explicitly ordinal task semantics.

Generation refuses unsupported layouts before allocation:2–1024 symbols,
nonempty words of at most4096 tokens, at most20000 examples and at most2M
teacher-forced token positions per split. Unique words cannot exceed the symbol
inventory. All production commands are capped at64GiB address space. Fixed
production shapes are much smaller. Source tests may construct data fixtures;
the production inventory is generated only after committing and freezing the
source closure, executable and protocol.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/generative-data
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_generative_data/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_generative_data/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_generative_data/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_generative_data/Cargo.toml
experiments/convex_stress_generative_data/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-generative-data"
prlimit --as=68719476736:68719476736 -- FRESH_FREEZE/convex-stress-generative-data prepare FROZEN_HISTORY REVIEWED326_REPORT FRESH_INVENTORY.json
```

The history input is the pinned `progress_snapshot.json` under
`/run/media/misha/storage/transformer-runs/stress-catalog/inputs-v1/diff-gpt/research/current_grokking/results/`.
The baseline is `.../nested-programs/original-v1/report.json`. Repeat the final
command at another fresh output path and compare the complete bytes. Required
Lean validation uses `LAKE_ARTIFACT_CACHE=false lake build`, direct global axiom
audit, native INDEX check and forbidden-source checks. No Lean statement changes
are needed for this data/scoring port. Generated inventories, copied baselines,
logs and binaries stay on storage outside the temporary worktree.
