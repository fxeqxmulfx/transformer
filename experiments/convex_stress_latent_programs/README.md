# Shared latent prefix programs

Can training-learned raw decoders extend shared copying to noncopy sequence
labels? This Rust experiment implements CM53 in `review/convex/moments.md` and
`Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms`, reviewed at
`8c7b8933aaf53d85a0688a1bee848878a6285617` (task323). The goal is at least99%
raw-token and99% exact-sequence accuracy **per task**, with ordinary full-vocabulary
cross-entropy and original95% success reported separately.

The fixed family policy first runs the unchanged CM52 binder from
`convex_stress_shared_programs` at `c44632bb`. If its initialization succeeds,
its model and all three training outcomes are used verbatim. Otherwise the
new latent family runs. There is no task-name, held-out or per-query choice of
family. Copying retains its explicit identity-decoder prior, including unseen
raw classes. The learned dictionary family has no such unseen-label prior.

Every ordered pair of distinct vocabulary symbols is enumerated. One summary
records the sign of their signed count and whether a prefix count was negative.
The other records the first selected role and the number of runs after ignoring
other symbols, saturated at each fixed cap1–8 plus an overflow state. These are
generic input summaries. No designated delimiter, opening symbol, class label,
generator answer function or task identity enters the program interpreter.
Their states fit a fixed32-state alphabet, including unused states.

Each program learns a partial state-to-raw-label dictionary from the original
supervised training prefixes. Conflicting labels eliminate it and retain the
two original record IDs. Every compatible candidate survives, unless a stated
cap refuses the construction. Unknown states retain `None`; raw0 is an explicit
numerical completion, whose predictions are still scored and marked unknown.
Consensus requires every retained dictionary to be known and unanimous.
CM53 proves correctness only under an explicit unknown-target realizability
premise. The experiment does not infer that premise from fit or flatness.

The full normalized complex DFT is a basis on latent states. Every block uses
the complete conjugate head of its learned raw table at margin24, including
every real/imaginary coordinate and every raw class. The default rows are
included in the head. Native training uses an **affine slice** of CM53's joint
domain: dictionary templates are fixed after training-only fitting, and their
simplex masses are optimized. Thus gamma[j,s,c] is linear in the remaining
parameters. This is not full optimization over all unknown joint rows and does
not multiply independently learned factors. The entire realified head stores
2K*32*V coordinates. Its column budget64*32 conservatively covers CM53's48*32.
The empirical prefix matrix may repeat states; source unitarity does not imply
that matrix is flat. Numerical SVD/inversion checks are not exact certificates.

| Mode | Actual CE updates | Initialization |
| --- | --- | --- |
| Full |16 complete-data gradients, learning rate1 | Uniform retained masses |
| Mini |4 epochs in original prefix order, batch32 with real short tails, rate0.1 | Same |
| Forward | Zero gradients, projections and derivative-cache products | Same |

Every derivative uses actual complete complex logits and all raw classes.
The derivative cache also comes from full products. Training CE is constant
on the exactly compatible slice; report attempted steps and effective mass
changes separately. Representation fitting can improve generalization even
when CE steps make no effective change. The forward mode still computes raw
training/evaluation loss. An independent scalar interpreter, class-major complex
products and bisection projection verify discovery, logits and trajectories.
Nontrivial derivative tests use competing heads and finite differences, separate
from the perfectly fitted training face. Synthetic tests do not score the four
original cohort datasets before the producer freeze.

The dataset adapter, every short/long split and both old baselines remain
unchanged. The required task322 report has SHA256
`550e9d86ce9175a1c1a9fb3ecea03ad281ae1fb4e627e5565f0cd125901eb791`;
it embeds task320 and historical GPT results verbatim. Individual historical
dataset hashes/checkpoints are unavailable, so these source-derived datasets
do not assert historical artifact identity. All12 mode outcomes, including
bounded misses, remain in the denominator. The original35 suite, its transformer
comparisons, pending317/319 and the remaining126 additional cases are separate.

Freeze committed sources, the binary and protocol before the four training
stages. Seal all states before any held-out scoring. Save every supervised
prediction, CE, missing-state/consensus indicator and robust-argmax check.
Independent logits must agree within1e-8; margins at most1e-7 are explicitly
marked roundoff-sensitive. Replay reconstructs complete training states and
all metrics, including refusals; a fresh complete run must reproduce artifacts.

The latent family caps its catalogue at16384, retained programs at128,
individual search/training/all-split scoring forecasts at16B scalar-work units,
buffers at56GiB and each process at64GiB virtual memory. The higher work cap
is frozen for this new family; the CM52 component keeps its old2B limits.
Forecasts charge all blocks, raw classes and the quadratic independent prefix
interpreter. They are conservative work units, not measured FLOPs. Repeated
validation/replay is extra work recorded in process timing. Native execution
is serial, with generated evidence on storage outside the temporary worktree.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/latent-programs
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_latent_programs/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_latent_programs/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_latent_programs/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_latent_programs/Cargo.toml
experiments/convex_stress_latent_programs/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-latent-programs"
experiments/convex_stress_latent_programs/run.sh FRESH_FREEZE/convex-stress-latent-programs TASK322_REPORT FRESH_RUN
```

The runner, gradient/checker scaffolding and freeze machinery are ported from
the task322 producer `c44632bb`. All deviations are described above: latent
state dimensions, learned decoder blocks, the affine joint slice, family priority,
dictionary coverage and charged work. No new cohort accuracy is claimed until
the committed producer comparison and exact review complete.

## Frozen four-case findings

Producer `3badaa300f6e8ec0e7e960fe910fa9de2f78f4b5` was committed and frozen
before all four training stages and any original scoring. Original and fresh
review runs reproduce all23 JSON artifacts byte-for-byte, including discovery,
complete trained states, every supervised prediction, replays and the report.
The report SHA256 is
`1607addda8c8eafa98c7999b5b8e7e16c3a1dd1396b8cb8fd6c0b889907c8a97`.

| Case | Previous shared-copy model | New full / mini / forward primary test | Longer tests |
| --- | --- | --- | --- |
| MQAR |100% raw/exact | Each128/128 tokens,64/64 sequences; CE6.15347e-9 |48/96:100% raw/exact |
| Two-hop lookup |100% raw/exact | Each128/128 tokens,64/64 sequences; CE3.73738e-9 |48/96:100% raw/exact |
| Dyck prefixes | Three no-compatible-copy refusals | Each834/834 tokens,64/64 sequences; CE1.32130e-9 |32/64:100% raw/exact |
| Three alternating blocks | Three no-compatible-copy refusals | Each716/716 tokens,64/64 sequences; CE1.32130e-9 |32/64:100% raw/exact |

All training, clean, validation, novel and longer splits also reach100% in every
mode. Each mode has4/4 original95%, primary both99%, test both99% and all-held-out
both99% successes. All12 outcomes are usable, with no process failure or OOM.
No scored prefix has an unknown dictionary state, disagreement or roundoff-
sensitive argmax. Independent full logits differ by at most3.91e-14.
Retrieval inputs, complete copy models and all result fields exactly equal
task322; this is an extension that preserves those measurements.

The latent search checks11340 candidates per prefix case. Dyck retains one
signed summary, with roles7/8 selected by the observed labels and five covered
states. Blocks retains12 run summaries: both orientations of roles9/10 and all
caps3–8. Their dictionaries cover9–19 states each. All roles and caps came from
the fixed enumeration; none was supplied as task metadata to the model.
The actual realified heads store2304 and27648 coordinates. Numerical column
budgets are376.112 and447.440, below the fixed2048 bound.

Dyck full performs16 batches/25984 gradient examples; mini performs204
batches/6496 examples, with24-record tails. Blocks full performs16/22720 and
mini180/5680, with12-record tails. All derivatives and full products run.
Dyck masses never change. Blocks records one full and180 mini bitwise updates,
but final deviations from initialization are at most4.31e-16 and1.95e-16;
training CE is unchanged. Forward performs zero gradients/projections/cache
products. The improvement comes from the shared summary and learned decoder,
not a measured optimization gain on the compatible face.

The largest training forecast is6,778,027,008 work units (blocks full), and
largest all-split score forecast is6,262,785,024 (blocks per mode). The largest
binding buffer forecast across both families is2,032,189,824 bytes, about1.89GiB.
Original peak RSS is171,276KiB, about167.3MiB, including complete prediction
replay. The slowest training process takes2.76s; the slowest replay takes9.14s.
Fresh repeats and validation are separately timed extra work under the64GiB cap.

Evidence lives at
`/run/media/misha/storage/transformer-runs/latent-programs/{original-v1,fresh-review-v1}`.
The416-input source freeze, binary, mandatory task322 baseline and checks are
under `.../freezes/latent-prefix-v1`. Independent final review reruns native and
Lean checks plus every original replay at the submitted findings commit.
This remains a seed1 four-case component result. It does not re-evaluate the
original35, fix Fibonacci's approximately98%, establish a combined39-task pass,
or score the remaining126 additional cases. Those comparisons remain separate.
