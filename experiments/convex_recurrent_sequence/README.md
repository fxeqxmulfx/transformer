# Learned complex recurrence on original object sequences

Can a learned recurrence improve the short-to-long transfer that the previous
bilinear reader failed? The input is a sequence of immutable objects and the
output is one next original object. Object IDs only address parameters. The
model receives no task name, arithmetic operation, bit role or answer rule.

CM95 in `review/flat_spectrum.md` and
`Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent` prove the
length-independent capacity bound, convex function domain, finite executed
mixtures, ordinary CE convexity and feasible signed class queries. CM96, committed
as `680dffb0`, proves that scalar parameters represent the complete complex
one-dimensional operator family and that absent input objects have exactly zero
transition response. It also proves this family's limitation: every mixture is
permutation-invariant. This is a Boolean/parity prototype, not a general
order-sensitive MQAR architecture. The proof's parity witness is never supplied
to the learner.

Each component has a complex initial state, one complex transition for every
original object and one complex readout for every original class. Execution
multiplies the state by the learned transition for every input occurrence, then
applies each readout. Every scalar has magnitude at most one. Nonnegative
component weights sum to the fixed capacity16; a stored zero component carries
unused mass. There is no fixed prefix table or maximum sequence length.

The joint domain is the convex hull of **all** such learned components. Component
discovery is an approximate numerical search; it is not a globally optimal
convex oracle. The physical transition parameters are not jointly convex.
Previously added components remain stored and their mixture weights update.
Every new component's numerical paths are learned from the current minibatch.

| Frozen run | Update count | Difference |
|---|---:|---|
| Original AND shift, seed1, both modes |128| Original batches, short and long probes |
| Original AND random, seed1, both modes |128| Original batches, short and long probes |
| Original parity16, seed1, both modes |512| Same learner; every answer/end target retained |

Mini uses full-class CE residuals. Forward Mini uses two finite CE evaluations
per original class and record. Both use the same capacity-reserving shrink and
query step0.0001. A component starts from trainable identity parameters, random
complex unit parameters or the preceding learned proposal. Each numerical
initial/transition coordinate gets four real/imaginary disk-projected proposals
at radii2,1,1/2,1/4,1/8,1/16. All original readouts are fitted to the complete
observed scalar signals, and proposals are compared by actual full-class CE at
a feasible mixture point. A64-iteration loss-only line search chooses the final
convex mixture step. There is no recurrent backward pass in either mode.

Only transition proposals for objects absent from **every** batch prefix may
be skipped, by CM96; their parameters remain stored. No magnitude, class,
record, length or semantic filter is used. Repeated objects, duplicate examples,
original answer/end tokens, target masks and short final batches are preserved.
Floating point capacity tolerances are2e-12 per operator and2e-10 times the
budget for total mass; these are numerical checks, not formal real certificates.

Eight native controls cover complex execution against independent real-matrix
composition, actual feasible signed query models, long-input bounds, opaque
object renaming, the order limitation, every original competitor, exact absence
versus dense search, tiny retained parameters, duplicates/tails, learning without
supplied witness weights, checkpoint replay and cohort-seal rejection. Full
Lean checks passed with eight new theorems, no new sorry and no extra axioms.
The original benchmark results below were measured after all six endpoints
were sealed.

Reproduce with native tools from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/recurrent-sequence
cargo test --offline --manifest-path experiments/convex_recurrent_sequence/Cargo.toml
cargo fmt --manifest-path experiments/convex_recurrent_sequence/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/convex_recurrent_sequence/Cargo.toml --all-targets -- -D warnings
cargo build --offline --release --manifest-path experiments/convex_recurrent_sequence/Cargo.toml --bin stress
"$CARGO_TARGET_DIR/release/stress" plan
```

Before running, freeze the executable, `protocol.json`, source commit/hash,
binary SHA256 and source archive outside the worktree. Include this experiment,
`convex_bilinear_sequence`, `convex_stress_minibatch` and `convex_stress` in the
archive. With a fresh output directory:

```sh
bash experiments/convex_recurrent_sequence/run.sh WORKSPACE STORAGE FREEZE COHORT
```

The runner trains both modes, seals all six endpoints, then evaluates every
original split and independently reloads/replays every raw logit and metric.
Generated evidence stays ignored under external storage. Model selection uses
fixed update counts, not held-out results. Preserve the original success flag
and report raw/exact99 separately, including every failed transfer probe.
The transformer baseline and earlier frozen producers are unchanged. Guards
enforce110GB aggregate RAM, at least2GiB free on the workspace disk and16GiB on
the storage disk. A low-space refusal stops work without further cleanup.

## Original short and long results, 2026-11-17

Producer `00d9936e`, freeze `recurrent-sequence-v1`, source fingerprint
`0375067e738010261e46facf75a05e3512f30459a54e9378eda00935a24b30be`, binary SHA256
`45f3b355dde50fdf45f9deae06f2d9ba379de94d98de6336a3178f09b872a5d4`.
Generated evidence remains in
`/run/media/misha/storage/transformer-runs/recurrent-sequence/original-v1`.
All18 train/score/replay commands exit0, and every complete saved raw logit and
metric replays exactly in both modes.

| Original task | Main held-out raw / exact counts, each mode | Mini main CE | Forward Mini main CE |
|---|---|---:|---:|
| AND shift |14/14 tokens;7/7 sequences|1.58677e-8|1.58548e-8|
| AND random |16/16 tokens;8/8 sequences|1.71247e-8|1.74367e-8|
| Parity16 |314/314 tokens;157/157 sequences|3.44944e-8|3.43698e-8|

All three tasks pass the original success criterion and raw/exact99 in both
modes. **Every original split has100% raw and exact accuracy**, including the
Boolean length32/64 and position-shift probes. Those long Boolean probes had
failed in the earlier bilinear/shared-factor experiments. Across all original
splits, the largest CE is7.59429e-7. The parity16 source has four splits at its
original task length; this run does not add a new longer-parity dataset.

Each mode correctly predicts all40000 training parity targets in20000 examples,
and all256 training targets in128 examples for each Boolean task. These full
training evaluations include records that were not visited during the fixed
512 parity updates:8192 sampled examples,16384 records,0.4096 epochs. Each
Boolean run uses4096 examples,8192 records,32 epochs in128 updates. Training
takes8.13/9.52s for AND shift,8.13/9.69s for AND random and18.47/19.27s for
parity16 (Mini/forward) with other producers active, not matched transformer
timings. Forward queries total1638400 per Boolean run and2228224 for parity.
Every component/line-search loss evaluation is also counted in the receipts.

This establishes empirical learning and length transfer for the tested original
Boolean tasks, and held-out parity16 learning, at seed1. It does not establish
three-seed recurrence reliability, general order-sensitive recall, all35/165
stress tasks or a theorem that the approximate learner identifies arbitrary
unseen answers. The scalar order obstruction still applies; no success is
assigned to an untested task. The generalization goal remains active.

## Prospective five-task, three-seed extension

`three_seed_protocol.json` keeps the model, capacity, queries, component search,
sampler and update budgets unchanged. It repeats AND shift/random128 updates
and parity16512 updates at training seeds1/2/3, and adds the original parity20
and parity24 tasks at512 updates, also in both modes and all three seeds.
The original fixed data seed1 and all original task fingerprints are retained.
The parity tasks use their original distributions up to the named bit length;
training a separate parity24 model is not an extrapolation test of parity16.

All30 endpoints must seal before any scoring. The repeated three seed1 tasks
provide a direct complete-logit control against the first producer. Preserve
all per-cell successes, failures, original split metrics and replay hashes.
No transition, feature, target, example or learned coordinate is changed for
the additional lengths.

Build with the commands above, use `stress plan-three-seed`, and freeze a fresh
executable/source archive with `three_seed_protocol.json` saved as the freeze's
`protocol.json`. Then run:

```sh
bash experiments/convex_recurrent_sequence/run.sh WORKSPACE STORAGE FREEZE COHORT -three-seed
```

Individual commands are `train-three-seed`, `score-three-seed` and
`verify-three-seed`, with the same positional arguments as the first cohort.

The expanded cohort is complete at producer `1786526f`, freeze
`recurrent-three-seed-v1`, source fingerprint
`07e81190c695ee67e9f95c280ec0bcc2351b4374050c0f7472b1a93ed3e5857d`, binary SHA256
`38a9ae459ae832c92a8eaf07f50230c085e1876bf79a3111722679aa0c49e8e9`.
Evidence stays in
`/run/media/misha/storage/transformer-runs/recurrent-sequence/three-seed-v1`.
All90 train/score/replay commands exit0, and all30 complete raw-logit replays
match. All six repeated seed1 split reports, including every raw-logit hash,
are exactly equal to the first recurrence producer.

| Original task | Successful cells | Main held-out correct tokens / sequences per cell | Worst CE over every split and cell |
|---|---:|---|---:|
| AND shift |6/6|14/14;7/7|2.10756e-6|
| AND random |6/6|16/16;8/8|4.89207e-5|
| Parity16 |6/6|314/314;157/157|3.67811e-8|
| Parity20 |6/6|568/568;284/284|3.63936e-8|
| Parity24 |6/6|612/612;306/306|3.18130e-8|

**Every original split of all30 models reaches100% raw and exact accuracy.**
Both original success and raw/exact99 are true in every cell, including all
Boolean length32/64 and position-shift probes. Thus this family passes these
five original tasks in both modes at all three training seeds on the unchanged
data seed1 splits. The result covers five of the selected35 tasks, not all35 or
the full165-case catalog. Addition's separately reviewed shared-factor result
is a different architecture and is not added to this model's pass count.
General order-sensitive recall remains unresolved by the scalar family.
