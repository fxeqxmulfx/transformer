# Shared nested-state programs

Can the convex complex decoder extend to typed nesting while preserving the
four reviewed sequence passes? The goal is at least 99% raw-token and 99%
exact-sequence accuracy per task. Ordinary full-vocabulary cross-entropy and
the original 95% success flags remain separate measurements. This Rust
experiment adds the two original typed-bracket cases prepared in task325,
plus the four unchanged cases from task324 as controls.

CM53 in `review/flat_spectrum.md`, formalized at `8c7b8933`, already quantifies
over arbitrary fixed input programs. Its joint-domain convexity and complete
complex-head results therefore cover this new finite program bank without a
new Lean statement. The contract and all 55 declaration axioms were rechecked
before implementation. The native domain uses prefixes of at most 4096 tokens
and a 32-state output alphabet. Correctness still requires target realizability
and correct observations. The Rust interpreter and its refinement to Lean are
not formally proved; numerical audits do not certify those premises.

The program bank enumerates every partial directed matching of the observed
input alphabet with one, two or three disjoint pairs, both orientations, and
all choices of unused symbols. The alphabet is collected before inspecting
labels. A program treats one role as a push, its paired role as the expected
pop, and all unused symbols as neutral. It reports empty, nonempty, or a
persistent mismatch/underflow state. This is a strong, explicitly supplied
structural prior for nesting. No designated token pair, class decoder, task
name, generator or held-out label enters the model. Matching roles and the raw
state decoder are identified from observed training prefixes and labels.

Every compatible partial decoder is retained. Conflicts preserve the original
two row IDs; uncovered states remain unknown. The numerical completion uses
raw label zero, and its predictions remain in every score. The runner records
unknown states, disagreements, raw labels absent from training, input symbols
absent from training, and peak stack depth. An independent interpreter repeatedly
cancels adjacent matched pairs; an independent reflected count checks peak
storage. It does not call the producer stack interpreter.

The entire task324 binder has frozen priority. Only failure to construct a
usable initialization permits the new bank. This decision uses training data
alone, applies to every mode and query, and preserves all earlier errors in the
saved state. The old four datasets, complete models and metrics must reproduce
the reviewed baseline exactly. This policy does not use the best held-out result.

The complete conjugate Fourier heads, ordinary CE, all raw classes, margin 24,
column budget 2048 and 32 latent coordinates are unchanged from `3badaa30`.
Dictionary templates are fixed after training-only fitting; optimization
updates their simplex masses, an affine slice of CM53's joint tensor domain.
The full joint domain is not optimized, and two independently learned factors
are not multiplied. Source unitarity bounds this construction; the empirical
query matrix need not be flat. Observed CE is constant on the exactly compatible
slice, so effective mass changes and representation gains are reported separately.

| Mode | Actual gradient work | Initial masses |
| --- | --- | --- |
| Full | 16 full-data steps, learning rate 1 | Uniform compatible programs |
| Mini | 4 epochs, batches of 32 including real short tails, rate 0.1 | Same |
| Forward | Zero gradients, projections and derivative-cache products | Same |

`latent.rs` and the independent training checker are included directly from
task324. `training.rs` copies that producer verbatim except for the checker's
module path. Discovery, model, evaluator, runner, tests and freeze machinery
are adapted from the same source. Deviations are the nested bank, input alphabet,
depth/symbol diagnostics, quadratic cancellation forecast, parent priority and
six-case baseline accounting. The actual head implementation is reused directly.

Synthetic tests cover the complete bank counts and unique roles; exhaustive
stack/cancellation agreement; deeper, type-sensitive inputs; nontrivial finite
difference gradients and descent; weighted minibatch derivatives and short tails;
forward-zero work; immutable training boundaries; unknown states; refusals;
complete serialized replay and coefficient/role/alphabet/decoder/work mutations.
They do not score original cohort data before the producer freeze.

The required task325 inventory has SHA256
`897ff08174af605ca2fb4c5b10bd347314d1b7a82d96b54919173c870f5b60cc`.
It preserves both historical GPT rows and the entire task324 baseline, including
earlier failures. Those historical endpoints are not a new equal-budget GPT run.
Individual historical dataset hashes/checkpoints are unavailable; the reviewed
source-derived dataset fingerprints do not establish past artifact identity.
Both new cases use seed 1, 128 training sequences of lengths 12–16, 32 validation
and 64 test sequences, and the original longer probes at 32 and 64. Every clean,
novel and long split remains, with all supervised prefixes scored.

Commit and freeze sources, binary, protocol and baselines before training.
Seal all six states before any held-out score. Keep all 18 mode outcomes and
all per-prefix predictions. Independent complex logits must agree within 1e-8;
argmax disagreement is rejected when margin exceeds 1e-7. Replay reconstructs
every complete state and score; a fresh full run must reproduce all JSON bytes.
No accuracy result follows from the Lean theorem or synthetic checks alone.

The bank allows 2–9 observed input symbols, at most 16384 candidates and 128
retained programs. Search, training and all-split evaluation each have a 16B
work forecast; buffers are capped at 56 GiB and every process at 64 GiB address
space. Quadratic cancellation comparisons and moves are included. A larger
bank or longer input refuses explicitly before dependent allocations. Existing
parent limits stay unchanged. Forecasts are conservative work units, not FLOPs;
replay and repeated checking are additional, separately timed work.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/nested-programs
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_nested_programs/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked --manifest-path experiments/convex_stress_nested_programs/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_nested_programs/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_nested_programs/Cargo.toml
experiments/convex_stress_nested_programs/freeze.sh FRESH_FREEZE "$CARGO_TARGET_DIR/release/convex-stress-nested-programs"
experiments/convex_stress_nested_programs/run.sh FRESH_FREEZE/convex-stress-nested-programs TASK325_INVENTORY FRESH_RUN
```

Keep generated models, metrics, logs and source freezes on storage outside the
temporary worktree. This six-case component comparison leaves the original35
suite separate: Fibonacci remains near 98%, with tasks317/319 pending. It does
not establish a combined 41-task pass or evaluate the other 124 new cases.
