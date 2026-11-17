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

## Findings from the complete fixed cohort

Producer `f5d317a6` completed all16 runs and128 stages before any new held-out
scoring. Growth improves final novel and test raw accuracy, and CE on every
original split, over equal-update continuation in all eight matched cells.
Novel gains are13.67–21.09 percentage points; test gains are9.18–21.48 points.
Every growth endpoint also improves held-out raw accuracy and all split CEs
over its own immutable c620019a input. Neither arm reaches the original
success threshold or99% at any of the80 declared checkpoints.

The following table reports stage8, fixed before training. Percentages are
original raw accuracy; exact accuracy is identical because this case has one
supervised answer per example. `Fwd` means forward-only Mini. The CE triplet is
train / novel / test. No row or checkpoint is selected by its measured rank.

| Arm | Geometry | Start / mode | Train % | Novel % | Test % | CE train / novel / test |
| --- | --- | --- | ---: | ---: | ---: | --- |
| continue | old | cautious / Mini | 75.00 | 23.44 | 20.31 | 1.651 / 2.711 / 2.773 |
| continue | old | cautious / Fwd | 79.10 | 20.90 | 21.48 | 1.554 / 2.750 / 2.717 |
| continue | old | refined / Mini | 77.44 | 21.68 | 21.88 | 1.595 / 2.759 / 2.715 |
| continue | old | refined / Fwd | 77.93 | 26.76 | 23.24 | 1.657 / 2.728 / 2.733 |
| continue | hilbert | cautious / Mini | 81.64 | 21.09 | 24.80 | 1.651 / 2.784 / 2.766 |
| continue | hilbert | cautious / Fwd | 82.76 | 23.44 | 23.05 | 1.625 / 2.762 / 2.739 |
| continue | hilbert | refined / Mini | 81.35 | 20.70 | 22.85 | 1.643 / 2.770 / 2.776 |
| continue | hilbert | refined / Fwd | 80.37 | 20.31 | 22.27 | 1.680 / 2.775 / 2.770 |
| grow | old | cautious / Mini | 77.39 | 41.80 | 40.23 | 1.070 / 1.784 / 1.819 |
| grow | old | cautious / Fwd | 78.47 | 40.04 | 42.97 | 1.047 / 1.799 / 1.762 |
| grow | old | refined / Mini | 79.30 | 42.77 | 41.99 | 1.044 / 1.788 / 1.762 |
| grow | old | refined / Fwd | 78.27 | 40.43 | 44.53 | 1.063 / 1.810 / 1.768 |
| grow | hilbert | cautious / Mini | 79.10 | 37.70 | 33.98 | 1.447 / 2.184 / 2.258 |
| grow | hilbert | cautious / Fwd | 76.86 | 39.06 | 37.11 | 1.443 / 2.137 / 2.199 |
| grow | hilbert | refined / Mini | 78.86 | 39.26 | 38.09 | 1.425 / 2.135 / 2.183 |
| grow | hilbert | refined / Fwd | 77.69 | 38.67 | 35.35 | 1.465 / 2.163 / 2.211 |

The best final novel and test percentages belong to different cells. The
largest novel value among all logged checkpoints is43.75% at old/cautious/Mini
stage4; it is retained as a trajectory observation, not selected as an endpoint.
All80 reports, including every checkpoint0, remain in the generated evidence.

The trajectory below averages the four starting-variant/mode cells in each
group. All use seed1; these are descriptive means, not four independent seeds.

| Stage | Continue old novel / test % | Grow old novel / test % | Continue Hilbert novel / test % | Grow Hilbert novel / test % |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 23.73 / 23.63 | 23.73 / 23.63 | 23.58 / 23.73 | 23.58 / 23.73 |
| 1 | 22.90 / 22.71 | 29.00 / 30.91 | 23.44 / 23.39 | 27.93 / 27.98 |
| 2 | 23.78 / 22.22 | 34.91 / 35.11 | 23.10 / 22.75 | 33.40 / 34.33 |
| 4 | 22.75 / 22.56 | 39.99 / 40.48 | 22.95 / 22.80 | 35.84 / 35.25 |
| 8 | 23.19 / 21.73 | 41.26 / 42.43 | 21.39 / 23.24 | 38.67 / 36.13 |

Growth's old-geometry runs outperform the corresponding Hilbert growth runs
on both held-out raw metrics and all split CEs here. This is evidence for this
learning schedule, not a general dominance theorem about the norm families.
The old-geometry growth mean keeps training raw nearly unchanged (78.47% to
78.36%), while its training CE falls from1.606 to1.056 and held-out accuracy
increases substantially. Hilbert growth lowers mean training raw from81.30%
to78.125% while also improving CE and unseen accuracy. The useful gain is not
an increase in training raw accuracy alone.

Continuation improves training CE in six cells and worsens it in two, with
little held-out benefit. The final current-versus-retained estimate error norm
is0.1962–0.2281 for old-geometry continuation, compared with0.0316–0.0534 for
old-geometry growth. Hilbert continuation and growth have ranges0.0323–0.0421
and0.0176–0.0325. These are finite numerical parameter estimates, not analytic
gradient or convergence certificates; exact cache bookkeeping does not make
a stale estimate current. Growth stores138
components versus130 for continuation, with exactly the same allocated mass16
and all original129 parent components present in both arms. This comparison
does not isolate additional content from the accompanying redistribution of
mass; a separately optimized calibration or different growth schedule is not
excluded by these results.

The independent CM110 proof at `d19597f7` also limits what additional plain
two-state blocks can accomplish. Their actual scores satisfy one common
eight-word identity, retained by any finite mixture; a bounded three-state
witness violates it. A held rotary parent contributes its own scaled relation.
This separates state width from mixture size, but proves no MQAR accuracy
bound and does not explain these particular errors. Wider learned states and
query-specific binding remain questions for the next comparison. The current
results do not resolve unseen-answer identification or99% generalization.

## Evidence and reconstruction

The frozen source fingerprint is
`24ac015c45ea111d4edb070699bdb170f443eaa02e5b7b959e171926f991d727`,
and the executable SHA256 is
`c23dd1d88034d5cf7ba5240d58e1a8232effe8a32b7c8ca7e6decf78c8fcffb5`.
The immutable archive, protocol, inputs and wrappers are under
`transformer-runs/freezes/content-growth-v1` on storage. All primary runs,
diagnostics and raw replays complete successfully. Five native controls and
strict compiler checks also pass from a fresh extraction of the producing
source archive. The independent review in `reviews/task400-v1` reproduces all80
checkpoint reports, all128 historical-cache diagnostics and all16 final current
diagnostics. All stage samplers agree across the16 cells. Seven changed-producer,
endpoint, cache, history and held-base cases are refused; corrected copies match
the untouched review cohort again. Native and complete review exit codes are0
under `.kanban/artifacts/400` in the root. Original inputs and the transformer
remain unchanged.

The cohort makes65,536 minibatch updates and2,097,152 supervised visits.
Its128 caches cover every original occurrence and all262,144 retained estimates
replay bit for bit at their actual historical versions. The65,664 stored
versions include every stage's initial block. There are196,493,312 full CE
evaluations,867,106,816 compact CE evaluations and1,949,335,552 finite logit
queries. All56,492,032 sampled compact/full query comparisons agree within
1.777e-15. Actual complete logits equal the held-base reconstruction exactly
on every checked training batch. Maximum norm excess is2.221e-16 and retained
mean recomputation error1.458e-15. Batch CE rises on16,253 updates; no acceptance
filter removes them. Total measured training time is1,997.97 seconds, and peak
aggregate process RSS is34,729,590,784 bytes, below110GB. Both disks retain ample
free space. These floating measurements do not replace the exact-real proofs.

Generate the complete table and trajectory data without Python:

```sh
jq -s -f experiments/convex_content_growth/summarize.jq \
 /run/media/misha/storage/transformer-runs/content-growth/original-v1/*/*/*/*/scores.json \
 > /run/media/misha/storage/transformer-runs/content-growth/summary-v1.json
```

The generator refuses missing or duplicate cells, changed checkpoints and
missing original metrics. Generated summaries remain local and ignored.
