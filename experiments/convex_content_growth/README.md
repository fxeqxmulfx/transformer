# More learned content at fixed capacity

Does adding learned blocks improve original unseen key-value recall beyond
giving one existing learned block the same number of additional updates?
CM107's `finite_feasible` at `94b47aa8` already covers arbitrary finite convex
mixtures of complete old and Hilbert components. CM108 at `886e7d05` proves
the full complex Hilbert matrix normalization. CM109 at `1bb685d2` proves
exact full-vocabulary cross-entropy reuse for finite output-only queries.
The new code applies those results without changing the task or loss.

| Arm | Stage initialization | Trained content |
| --- | --- | --- |
| continue | Retain the complete input mixture | The existing plain learned block |
| grow | Scale every existing contribution by7/8; add one complete block at mass2 | The newly added block |

All eight `c620019a` endpoints are inputs, because every one improved training
CE over its own parent. No held-out ranking filters them. Both arms cover both
geometries, cautious/refined parents and Mini/forward-only Mini:16 runs total.
Each has eight stages of512 original minibatches32. Both arms use the same
stage-specific sampler and reset the active block's occurrence cache at each
stage. The prescribed checkpoints are0,1,2,4,8, with no early stopping,
line search, checkpoint selection or held-out scoring before all128 stages
seal. The original case is `gpt-probe-key-value-binding`, seed1, vocab100,
2048 training examples and512 examples per original held-out split.

The complete129-component learned rotary parent always remains stored.
Each new plain block contains two complex initial coordinates, all four
complex entries for every object's transition, and two complex coordinates
for every original class's readout:1,204 trainable real coordinates.
There are no new positional rotations, supplied object roles or task programs.
Growth keeps total capacity16. It enlarges the finite representation within
that capacity, rather than increasing the bound. Earlier blocks are held
within a stage, after their previous learning. Independent factor training is
not jointly convex; this is an approximate search inside a convex function
domain, with no proved global oracle or stochastic convergence guarantee.

Mini uses ordinary CE residuals with complete numerical logit differences.
Forward-only Mini uses only finite CE differences. Output-row queries affect
one original logit, so CM109 allows the complete partition to be reused using
`log1p(share * expm1(delta)) - target_indicator * delta`. Both actual feasible
queries are used, including asymmetry after projection. All classes remain
in the partition. Every output coordinate on each batch's first record and
one record-indexed output coordinate on every remaining record are checked
against full CE, at tolerance2e-12. Full and cached CE evaluations, audits and
finite logit queries have separate counters. This real identity is not a
floating-point certification. Native controls also compare every coordinate
against the original complete estimator, including both geometries and modes.

Each original supervised occurrence retains its last complete estimate and
actual pre-update parameter version. Duplicate occurrences, masks and tails
remain. Refresh is `(new-old)/N`, followed by one simultaneous complete update
and actual geometry normalization. All513 active-block versions per stage are
retained. Review reconstructs every stage from its previous complete endpoint,
verifies the entire held mixture and replays every cached occurrence bit for
bit. Final-stage current-estimate diagnostics stream training records without
updates. Neither cache consistency nor additional blocks supplies unseen labels.

Every original split reports raw/exact accuracy, CE, original success and a
separate99% flag at every declared checkpoint. Checkpoint0 must exactly match
the input's saved raw report; all80 checkpoint reports must replay. The
transformer and original tasks remain unchanged. This is one task and seed,
not a claim of success on the whole stress suite.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/content-growth
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1
cargo test --offline --locked --manifest-path experiments/convex_content_growth/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_content_growth/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_content_growth/Cargo.toml --check
cargo build --release --offline --locked --manifest-path experiments/convex_content_growth/Cargo.toml
```

Before training, archive the producing commit, then freeze the executable,
protocol, inputs, plan and wrappers with a SHA256 manifest. Reproduce with:

```sh
bash experiments/convex_content_growth/run.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/content-growth-v1 \
 /run/media/misha/storage/transformer-runs/content-growth/original-v1 \
 /run/media/misha/storage/transformer-runs/diagnostics/content-growth-v1
bash experiments/convex_content_growth/review.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/content-growth-v1 \
 /run/media/misha/storage/transformer-runs/content-growth/original-v1 \
 /run/media/misha/storage/transformer-runs/diagnostics/content-growth-v1 \
 /run/media/misha/storage/transformer-runs/reviews/task400-v1
```

One native worker runs under a32GiB address-space cap,110GB aggregate RAM,
2GiB workspace and16GiB storage free-space thresholds. Low disk space stops
the run; no evidence is deleted. Generated evidence stays on storage outside
the worktree. Review checks equal samplers, fresh raw and diagnostic replays,
and seven deliberate corruption refusals on copies of sealed artifacts.
