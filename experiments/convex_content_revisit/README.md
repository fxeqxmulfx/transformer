# Revisit learned content with historical states

Does learning shared recurrent content improve the remaining key-value recall
failure after repairing the readout estimator? The reviewed CM106 extension at
`3c4705fb` proves how simultaneous state and signal changes affect retained
coefficients, following CM105's estimator-error correction. This experiment
compares continued readout training against revisiting the learned initial
states, object transitions and positional frequencies.

| Arm | Learned content during this phase | Readout update |
| --- | --- | --- |
| control | Held at its learned starting values | Complete retained coefficient, maximum step 1/1024 |
| revisit | One component receives a complete coordinate sweep per minibatch | Identical estimator, queries and step ceiling |

Both arms start from each of the four immutable `5380ac97` retained-small
endpoints: cautious/refined content and Mini/forward-only Mini. These inputs
were selected for their paired lower training CE and actual negative training
readout directions. All earlier failures remain recorded. `inputs.json` pins
the endpoint, metadata, receipt and cached signals of every input. Training
does not read a score file. Every run uses 512 further original minibatches of
32, sampler seed 20001, query reserve 1e-4 and twelve readout line iterations.
All eight endpoints must seal before new held-out scoring.

The input remains a sequence of original constant objects and the output an
original next-object label. All 100 classes, 129 stored components and both
complex coordinates remain. The structural zero component represents unused
capacity. Revisits cycle through components 1 through 128, including components
whose current weight might be zero. Each sweep tests the two complex initial
coordinates, every entry of each object matrix used by the current batch, and
both learned frequencies. Only matrices of absent original objects skip, since
they cannot change any batch score. Real and imaginary trials use radius 1/64;
frequency trials use 1/256. Initial coordinates project to the complex unit disk;
complete matrix rows use CM102's Euclidean projection to their unit l1 ball.
Four signed trials per complex coordinate and two per frequency compete with
the unchanged current point using ordinary full-class CE. The actual complete
model checks its reconstructed logits and CE after the sweep.

This is approximate coordinate search in the complete learned family. It is
not a global convex oracle in the independent content factors. The content
search uses finite actual CE evaluations in both modes. Forward readout signals
also use finite signed feasible CE queries; analytic CE residuals never enter
forward training. The readout step ports the reviewed `5380ac97` retained
update, replacing its fixed-content cache with CM106's historical-state cache.

Each original supervised occurrence stores its last signal, its actual state
under that observation's content, and that content version. Warming the cache
reuses every prior signal and computes the starting states from inputs alone,
without additional CE queries. A visit updates the complete mean by
`(new_signal * new_state - old_signal * old_state) / N`. In particular, changing
a state with an unchanged signal still changes its contribution. Duplicates
have separate indices; masks and short tails keep the original record order.
Every content version records its hash and its changed component, if any.
Review reconstructs every cached state from the immutable starting model and
this history, comparing both real and imaginary coordinates bit for bit.

After all endpoints seal, diagnostics stream the complete original training
set in batches, compute its actual coefficient A, and compare it with cached H.
They report the actual CM105 coefficient error, gap and direction as well as
CM106's separate signal error and current-versus-historical state error. The
state discrepancy is measured directly here; it is not a free online error
certificate. Independently summing the stored signal/state products checks
numerical accumulation. These read-only analytic diagnostics never update the
model or select an endpoint. No stochastic convergence, certified floating
rounding, factor optimum or unseen-answer theorem follows from this check.

Native controls cover changed states with unchanged signals, cancellation of
stale coefficient errors and its removal by partial refresh, duplicate indices,
short tails, wrong occurrence/version identities, independently recomputed
products, historical origin replay, all complex content trial directions,
feasible projection and both readout query modes. Full-class logits and CE are
checked against actual execution. Each original split reports raw/exact
accuracy, CE and a separate 99% flag; the original success predicate stays
unchanged. All raw-logit hashes and metrics must replay. The transformer
baseline remains unchanged.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/content-revisit
export CARGO_BUILD_JOBS=2
cargo test --offline --locked --manifest-path experiments/convex_content_revisit/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_content_revisit/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_content_revisit/Cargo.toml --check
cargo build --release --offline --locked --manifest-path experiments/convex_content_revisit/Cargo.toml
```

Before training, archive the producing commit with `git archive` and freeze its
binary, protocol, input pins, plan, wrapper and SHA256 manifest. Generated data
and review evidence stay on storage, outside the worktree. Execute one native
worker with a 32 GiB address-space cap:

```sh
bash experiments/convex_content_revisit/run.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/content-revisit-v1 \
 /run/media/misha/storage/transformer-runs/content-revisit/original-v1 \
 /run/media/misha/storage/transformer-runs/diagnostics/content-revisit-v1
```

Admission checks enforce the 110 GB aggregate RAM limit and both disk-space
thresholds before each minibatch. Low disk space stops work without deleting
evidence. Review repeats diagnostics in a fresh directory and replays saved
scores in fresh copies, then rejects changed producer, endpoint, cache and
history evidence. Generalization remains unresolved.

Producer `48571783` completes the eight declared runs. Content revisits modestly
raise novel accuracy relative to the matched continuation in all four pairs,
while lowering training accuracy and raising training CE in all four. Test
accuracy rises in two pairs, ties in one and falls in one. These seed-1 results
do not establish a reliable generalization improvement or an optimizer repair.

| Arm | Source/mode | Train raw | Train CE | Novel raw | Test raw |
| --- | --- | ---: | ---: | ---: | ---: |
| control | cautious Mini | 78.2715% | 1.799295 | 16.9922% | 17.7734% |
| control | cautious forward | 79.8340% | 1.779229 | 18.7500% | 18.7500% |
| control | refined Mini | 77.6367% | 1.805723 | 17.7734% | 19.1406% |
| control | refined forward | 76.2695% | 1.828541 | 18.7500% | 18.5547% |
| revisit | cautious Mini | 75.9766% | 1.925073 | 17.9688% | 18.3594% |
| revisit | cautious forward | 76.7578% | 1.916712 | 20.3125% | 18.7500% |
| revisit | refined Mini | 76.6602% | 1.933239 | 17.9688% | 19.7266% |
| revisit | refined forward | 75.1465% | 1.965040 | 19.1406% | 18.3594% |

Raw and exact accuracy agree on these single-target examples. Every original
success flag and every split's 99% flag is false. Relative to their four
starting retained-small endpoints, both revisit Mini runs have lower novel
accuracy and both forward runs have higher novel accuracy. All four revisit
training CEs are higher than their starting values. The continuation improves
training fit in all four controls, but does not close the large unseen gap.

All 4,096 scheduled updates finish. Every revisit step changes content, for
2,048 changed versions; each of the 128 learned components receives four full
sweeps. Across those sweeps, 4,085 initial-coordinate, 464,723 matrix-coordinate
and 3,974 frequency-coordinate proposals are accepted. Every content sweep
lowers or preserves its actual minibatch CE, and 2,411 readout updates take a
positive step. Thus current-batch descent still does not imply lower complete
training CE when content changes. No endpoint or trajectory is selected using
held-out outcomes.

Every endpoint retains all 129 components and covers all 2,048 indexed records,
with 16,384 new cache refreshes per run. All 16,384 cached record histories
reconstruct exactly from their recorded content versions. Independent complete
product sums agree with the incremental cache within 3.47e-16; the cache bug
targeted by CM106 is not present. Original full raw-logit replays agree in all
eight cells.

At the endpoints, actual joint readout gaps are 0.279–0.297 in controls and
0.349–0.371 in revisits. All eight estimated readout directions have negative
actual full-training first-order change, from -0.3653 to -0.1403. Nevertheless,
the sufficient CM105 corrected-gap condition is inconclusive in every cell:
its actual error corrections are 0.347–0.402 in controls and 0.490–0.758 in
revisits, exceeding their respective estimated gaps. This does not contradict
the directly measured negative directions; the bound is sufficient, not
necessary.

The CM106 split exposes genuine historical-state movement. Its mean state term
is zero in controls and 0.2290–0.2403 in revisits; the corresponding signal term
is 0.0298–0.0337 and 0.0434–0.0486. The resulting revisit corrections 8.718–9.235
are much looser than the actual CM105 coefficient-error corrections because
they sum absolute per-record discrepancies before possible cancellations.
Both sets of measured inequalities have zero positive excess at the declared
tolerances. These are floating evaluations, not finite-rounding certificates.

The eight runs consume 80,381,952 CE evaluations: 64,784,384 for content search
and 15,597,568 for readout updates, including 13,107,200 finite signed-bias
queries. Read-only diagnostics add 32,768 CE evaluations. Maximum recorded
aggregate RSS is 10.436 GB, complete-logit reconstruction error 1.51e-14,
operator-norm excess 2.23e-16 and shared-capacity excess zero. Low disk-space
guards never trigger. The transformer baseline and prior endpoints are intact.

Evidence is under `transformer-runs/freezes/content-revisit-v1`,
`content-revisit/original-v1` and `diagnostics/content-revisit-v1` on storage.
The frozen source fingerprint is
`377b1055938f8f7b8c687409c730d2617eedd52420663d0c38dbfd6db67cdbea`.
The history-aware estimator permits real content learning, but this finite
minibatch coordinate-search recipe is insufficient. The next change must
address shared-content learning rather than infer success from a norm bound,
negative readout direction or decrease on the visited minibatch.
