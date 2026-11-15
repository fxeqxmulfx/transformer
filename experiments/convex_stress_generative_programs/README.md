# Original sequence generation with shared complex programs

Can the convex complex model reach 99% raw-token and exact-sequence accuracy
on the original copy/reverse/sort cases, including longer sequences? This
extension preserves all six reviewed retrieval/prefix models and adds a fixed
input-only bank of positional and order-statistic programs. No original model
result has been measured yet. Synthetic fixtures are implementation checks.

| Cases | Status before this experiment | Change |
| --- | --- | --- |
| MQAR, lookup, Dyck, blocks, Dyck-2, Dyck-3 | Reviewed326: 100% raw/exact in every mode and split | Reconstruct the original complete states and require exact state/data/metric identity |
| Copy repeat/unique | Original GPT endpoints retain their recorded failures | Train a decoder over positional states and measure actual generated answers |
| Reverse repeat/unique | Original GPT endpoints retain their recorded failures | Same bank and training; no task-specific routing |
| Sort repeat/unique | Original GPT endpoints retain their recorded failures | Same bank, with ordinal rank as an explicit input bias |

Data comes from reviewed327 at `552b9738`, inventory SHA256
`2778983591a49bcde79dbcd9952b4999fa0afb914b9e37f4826834116a146e6f`.
It retains the complete326 report and historical GPT endpoints. Each new case
has seed1, 128/32/64 examples, lengths8–16 and32/64 probes, 100 raw classes,
and all11 clean/novel/short/long splits. Historical per-example hashes and
checkpoints are unavailable; these GPT results are historical references.
The original35 comparison, including its remaining Fibonacci raw99 miss, is
separate. This experiment does not claim a combined47-case model.

CM54 was proved and landed at `4c95deeb85821010f283ce31495181c700f71922`
before native implementation. It transfers CM53's actual complex argmax
through own-prefix generation and EOS, provided the unknown target is realized
by the bank, observations are truthful and every visited target prefix is
accepted. Five new theorems and inhabited examples compile without warnings;
the direct axiom audit uses only standard axioms. Full Lean/global checks pass
with36 existing `sorry`, zero dependencies on them and zero extra axioms.
Neither the proof nor source-code flatness establishes native target coverage.

The binder first runs the entire unchanged326 implementation. A usable parent
initialization has priority. Only a training-only refusal enables this bank:
every observed input symbol as a candidate delimiter, its first/last occurrence,
skip0–3 initial symbols, and original/reverse index or ascending/descending rank
with multiplicity. The number of tokens after the delimiter supplies the rank.
No candidate receives a task name, generator, answer function or held-out score.
An independent rank-counting interpreter checks the sorting/position evaluator.

Raw symbols use states0–125; exhausted and missing inputs use states126/127.
Each surviving program learns its first observed raw label per latent state;
conflicting original records reject that program. EOS's decoder entry is
learned, not assigned a raw value by the program. Unknown dictionary entries
remain `None`. Their numerical completion uses an explicit identity-copy prior
for raw-symbol states within the vocabulary, and raw0 otherwise. CM53's
`ofDecoders_mem` permits any completion consistent with observations. It does
not certify those unknown outputs. Reports retain unknown and ambiguous queries,
including cases where the copy prior happens to predict correctly.

The complete head stores both coordinates of every128-state complex Fourier
coefficient and every raw class. Margin24 and radius8192 are fixed. Decoder
templates freeze after discovery; only their simplex masses change during
ordinary full-vocabulary CE training. This affine slice is convex and introduces
no multiplication of independently learned factors. It does not optimize the
entire CM53 joint domain. All compatible observations already have the same
finite-margin logits, so the expected gain is representation/identification;
CE updates need not change predictions. Full uses16 steps, mini uses4 epochs
with batch32 and real short tails, and forward has zero gradient/update steps.

All12 training states are sealed before scoring. The six new tasks use actual
greedy generation until EOS or the original limit. Independent full complex
products and the separate327 generation scorer audit the outputs; teacher-forced
ordinary CE remains separate. Every mode retains raw99, exact99 and original95
criteria on every short/long split. Replays reject changed states, inputs,
coefficients, metrics or baselines. Each old control must reproduce its entire
old state hash and result, not merely its pass flag.

Before allocation or loops, component bounds limit the new vocabulary to126,
prefixes to4096, retained programs to128, search/training work to64B scalar
operations, and each all-split generation audit to128B. The larger bounds charge
the128-state head and repeated generation products; old families retain their
original bounds. A conservative factor16 accounts for nested construction and
replay kernels, and factor2 for scoring/replay. These are kernel-work counts;
hashing, JSON, data construction and all earlier-family work also remain visible
in whole-process time, RSS and I/O measurements. Every process has a64GiB cap.

Nine native checks cover independent rank interpretation, nonidentity learned
decoders, longer own-prefix/EOS generation, unknown-state diagnostics, full
complex gradients, weighted minibatches, real tails, zero forward updates,
immutable training boundaries and overflow-safe nested work accounting.
They use synthetic fixtures, not the six original model evaluations.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/generative-programs
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
cargo fmt --manifest-path experiments/convex_stress_generative_programs/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_generative_programs/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_generative_programs/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_generative_programs/Cargo.toml
prlimit --as=68719476736:68719476736 -- lake build
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/LatentGenerationAxioms.lean
experiments/convex_stress_generative_programs/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-generative-programs"
experiments/convex_stress_generative_programs/run.sh FRESH_FREEZE/convex-stress-generative-programs ORIGINAL_ROOT original PRODUCER_COMMIT REVIEWED327_INVENTORY
experiments/convex_stress_generative_programs/run.sh FRESH_FREEZE/convex-stress-generative-programs REPEAT_ROOT repeat ORIGINAL_ROOT
```

Also run the direct global axiom audit, native INDEX check and forbidden-source
checks before committing. `prepare-run` requires the clean producer checkout.
A repeat reuses only the validated frozen inventory/baseline, regenerates all
dataset identities and reconstructs every state. It copies no model or score.
The driver attempts all12 stages, then all12 scores and replays, retaining every
exit and resource log. Compare every JSON artifact byte-for-byte after a fresh
repeat. Generated states, reports and logs stay on storage; commit only source,
configuration and authored findings. No new99% attainment is inferred before
that evaluation and independent review finish.
