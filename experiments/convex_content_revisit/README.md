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
history evidence. The declared comparison is pending; generalization remains
unresolved.
