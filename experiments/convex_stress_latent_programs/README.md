# Shared latent prefix programs

Can training-learned raw decoders extend shared copying to noncopy sequence
labels? This Rust experiment implements CM53 in `review/flat_spectrum.md` and
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
