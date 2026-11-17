# Learned Hilbert content without positional rotations

Does the enlarged CM107 recurrence family improve the unresolved original
key-value recall failure? CM108 at `886e7d05` proves the exact full complex
two-by-two Hilbert norm and a feasible radial normalization. CM107 at
`94b47aa8` proves both complete old-family inclusion and a mixing witness that
the old maximum-norm constraint contracts. This comparison keeps the learned
old parent and tests a complete new block with each geometry.

| Arm | Initial state | Object transition | Class readout |
| --- | --- | --- | --- |
| old | Maximum norm at most one | Complete CM102 row projection | Complete row l1 norm at most one |
| hilbert | Euclidean norm at most one | Complete CM108 matrix normalization | Euclidean row norm at most one |

All four `48571783` control endpoints are immutable inputs. Their selection
uses improved training CE and actual negative training readout directions,
not held-out ranking. Training never opens a score file. All 129 original
components remain, at seven eighths of their weights. One new block receives
mass two, keeping total capacity 16. The block learns two complex initial
coordinates, every entry of each of the 100 complex object matrices, and two
complex readout coordinates for each of the 100 original output objects.
All 1,204 real coordinates remain present. No new positional rotation, object
role, answer formula or selected class subset is supplied. Noncommuting
object transitions can represent order. The old parent's learned paths are
held during this one new-block learning phase; this does not establish that
joint content-factor optimization is convex.

Each paired arm starts from the same seeded raw initial vector and generic
identity-plus-complex-jitter transitions, normalized by its own geometry.
Initial readouts are zero, so paired initial logits agree exactly. Both arms
receive 512 original minibatches of 32, with shuffle seed 30001, finite query
step 1e-4 and learning rate one. All settings are fixed in `protocol.json`
before training. There is no endpoint selection or early stopping.

Every minibatch queries both signs of every responding real coordinate.
Absent-object matrix coordinates have an exact zero response and skip their
forward evaluation; every original output class participates. Mini combines
the analytic ordinary CE residual with the full numerical logit difference.
Forward-only Mini uses only actual full-class CE differences. Every queried
block is feasible in the same mixture. These are finite constrained-coordinate
estimates, not a proved analytic gradient or a global factor oracle.

A zero-initialized cache stores the last complete estimate for each original
supervised occurrence. A visit updates its mean by `(new-old)/N`, preserving
duplicate identities, target masks and short tails. The update subtracts the
complete retained mean simultaneously, then normalizes the complete block.
It does not greedily select individual coordinates or require descent on the
visited batch. Stale estimates can still be inaccurate. CM106's earlier
readout bounds do not prove convergence of these parameter estimates.

The initial block and all post-update blocks are recorded with hashes. After
all eight endpoints seal, a training-only diagnostic replays every cached
estimate at its recorded parameter version, bit for bit. It separately streams
current estimates in batches, measures their disagreement with the retained
mean and their dot product, and reports actual state norms. This dot product
is not an analytic descent certificate. No diagnostic changes the model.

Every original split reports raw and exact accuracy, ordinary CE, the original
success predicate and a separate 99% flag. Complete raw logits must replay.
The transformer baseline, original task and all earlier failures stay intact.
Only one original task and seed are covered by this pilot. Neither the Lean
norm proof nor a positive estimate dot product supplies unseen answers.

Native controls compare the complete complex norm against independent LAPACK
SVD, including zero, diagonal, rank-one and complex off-diagonal matrices.
They check the rational mixing/contraction witness, old/new noninclusion,
noncommuting order, every real/imaginary query coordinate, both query modes,
paired identical predictions, absent-object zeros, retained duplicate/tail
refreshes and wrong history rejection. Floating calculations are not certified
rounding proofs.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/hilbert-content
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1
cargo test --offline --locked --manifest-path experiments/convex_hilbert_content/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_hilbert_content/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_hilbert_content/Cargo.toml --check
cargo build --release --offline --locked --manifest-path experiments/convex_hilbert_content/Cargo.toml
```

Archive the producing commit with `git archive` before any new training.
Freeze the binary, protocol, inputs, wrapper, plan and SHA256 manifest. Run one
native worker with a 32 GiB address-space cap:

```sh
bash experiments/convex_hilbert_content/run.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/hilbert-content-v1 \
 /run/media/misha/storage/transformer-runs/hilbert-content/original-v1 \
 /run/media/misha/storage/transformer-runs/diagnostics/hilbert-content-v1
```

The wrapper and each minibatch enforce the 110 GB aggregate RAM limit and
both disk thresholds. Low disk space stops work without deleting evidence.
Generated outputs stay on storage outside the worktree. Review repeats all
raw replays and diagnostics in fresh locations and rejects changed provenance,
endpoints, caches and histories. Generalization remains unresolved until the
unchanged original held-out results demonstrate otherwise.

```sh
bash experiments/convex_hilbert_content/review.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/hilbert-content-v1 \
 /run/media/misha/storage/transformer-runs/hilbert-content/original-v1 \
 /run/media/misha/storage/transformer-runs/diagnostics/hilbert-content-v1 \
 /run/media/misha/storage/transformer-runs/reviews/task397-v1
```

Producer `c620019a` completes all eight declared runs. Both geometries improve
novel and test accuracy and reduce CE on every split relative to their own
immutable starting parents. Hilbert geometry raises training raw accuracy in
all four paired comparisons with the old geometry, but its novel and test
accuracy each improve in only two pairs and decline in the other two. Its CE
is higher than the matched old arm on every split in all four pairs. These
results do not identify a consistent held-out advantage for the Hilbert norm.

| Arm | Source/mode | Train raw | Train CE | Novel raw | Test raw |
| --- | --- | ---: | ---: | ---: | ---: |
| old | cautious Mini | 79.0039% | 1.602124 | 23.8281% | 22.8516% |
| old | cautious forward | 78.8574% | 1.586483 | 21.6797% | 25.3906% |
| old | refined Mini | 77.8809% | 1.620586 | 24.0234% | 24.0234% |
| old | refined forward | 78.1250% | 1.616080 | 25.3906% | 22.2656% |
| hilbert | cautious Mini | 82.0313% | 1.672622 | 24.4141% | 24.6094% |
| hilbert | cautious forward | 82.3242% | 1.660829 | 22.8516% | 24.6094% |
| hilbert | refined Mini | 80.2246% | 1.682239 | 23.2422% | 22.8516% |
| hilbert | refined forward | 80.6152% | 1.700185 | 23.8281% | 22.8516% |

Raw and exact accuracy agree on these single-target examples. Every original
success and every 99% flag fails. The best novel and test accuracies, both
25.3906%, come from different old-geometry forward runs. The highest training
accuracy, 82.3242%, belongs to the cautious Hilbert forward run; it is not an
unseen result. Relative to the starting parents, training accuracy improves
in seven runs and falls in the old cautious forward run, despite its lower CE.
Old-arm novel CE is 2.674866–2.708314 and test CE 2.659878–2.717092; Hilbert-arm
novel CE is 2.762450–2.773239 and test CE 2.734005–2.776055. This is still one
original task and seed, not a full stress-suite or three-seed success.

All 4,096 updates finish, with 131,072 original occurrence visits and 690
visited-batch CE increases. Every run covers all 2,048 original records and
makes 16,384 cache refreshes. Every retained historical estimate replays bit
for bit across all 16,384 endpoint records. Independent means agree with the
incremental cache within 3.34e-16; maximum observed norm excess is 2.23e-16.
All complete raw logits replay. Initial and transition coordinates change
after the first update learns nonzero readouts; real and imaginary coordinates
remain represented, including unobserved-object matrices with zero estimates.

The current-versus-retained estimate dot product is positive in every cell,
0.01515–0.02785. Their Euclidean difference is 0.06832–0.09615 for the old arm
and 0.05465–0.05920 for Hilbert. These finite estimates are not certified true
derivatives, so their agreement is not a convergence or generalization proof.
Actual Hilbert state norms average 0.3714–0.4160, with minima 0.0100–0.0317,
despite initial norms near one. The feasible Hilbert family still permits
contractions: preserving a supplied isometry under normalization does not
force learned matrices to be isometries or preserve all input distinctions.
Old-arm maximum-state norms average 0.6417–0.6900; these use a different norm
and are not a direct comparison of identical trajectories.

Training uses 61,178,880 ordinary CE evaluations and 121,833,472 finite logit
queries; the latter includes both Mini and forward queries, while Mini uses
analytic CE residuals. Read-only current/history diagnostics add 15,261,952 CE
evaluations. Each training cell takes 16.82–31.52 seconds. Maximum recorded
aggregate RSS is 15.090 GB, well below the 110 GB cap. Neither disk guard fires.
The frozen source fingerprint is
`9ee1afcef03c26a3a1e572b02852378f915a918890c15b1a09a821636e1ca8ca`;
the evidence locations are the commands above. All original parents, earlier
failures and the transformer remain intact.

This pilot changes the active-block optimizer, representation and allocation
relative to the earlier content revisit; it does not isolate the optimizer as
the cause of the gains. Before expanding this recipe, check whether the new
block contributes useful input-dependent structure beyond a learned class
bias. The norm extension is valid, but the requested 99% original held-out
accuracy and generalization remain unresolved.
