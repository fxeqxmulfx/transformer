# Retained coefficients and step size

Can a complete per-record coefficient estimate repair the failure of greedy
minibatch capacity allocation? CM105 at `b87fa714` proves explicit coefficient
error corrections and a finite feasible step that lowers minibatch CE but raises
the complete training mean. This experiment compares estimator and step size
separately on the original key-value recall failure.

| Arm | Direction coefficient | Maximum step | Runs |
| --- | --- | ---: | ---: |
| instantaneous-large | Current minibatch | 1/16 | Four immutable, already scored f5ae34ef endpoints |
| instantaneous-small | Current minibatch, unchanged f5ae34ef update | 1/1024 | Four new endpoints |
| retained-large | Last observed signal of every indexed training occurrence | 1/16 | Four new endpoints |
| retained-small | Same retained estimate | 1/1024 | Four new endpoints |

Each arm uses both cautious/refined learned-content inputs and Mini/forward-only
Mini. Every new run has512 updates, original minibatches32, sampler seed10001,
query reserve1e-4 and twelve line-search iterations. All twelve endpoints seal
before any new held-out scoring. The prior four endpoints are pinned by their
metadata, receipt, endpoint, score and replay hashes; no baseline is retrained
or selected from held-out results. The transformer baseline stays intact.

For weighted rows U, retain each training occurrence's last full-class CE signal
s_i and maintain H = sum_i state_i * s_i / N. Unvisited signals start at zero.
Refreshing a minibatch subtracts its old signals and adds its new ones, retaining
all100 original classes,129 components and two complex coordinates. Identical
records have separate indices. Masks and short tails preserve original record
order. A content hash checked at every refresh prevents reuse after changing
initial states, transitions or frequencies. This is a conditional optimizer
phase after learned-content training, not a prescribed-feature architecture.

H can be stale and biased. Neither unbiasedness nor stochastic convergence is
claimed. Both arms still line-search actual CE on the current minibatch, which
can reject a direction useful for the complete training set. Both reserve query
capacity identically. Forward training obtains every signal from finite signed
original-class CE queries; no analytic CE derivative enters its updates. The
cache does not issue extra signal queries. Receipts count centre validation,
signed queries, line evaluations and reconstructed-CE checks separately.

After sealing, read-only diagnostics stream the original training set in batches
to compute the actual complete coefficient A. For retained endpoints they report
G_H, G_A, D(A-H), the CM105 correction2B D(A-H), the corresponding direction
bound, and the independently recomputed cached-signal table. These analytic
diagnostics of forward-trained models are separate from forward-only training
and never select an endpoint or update. Small floating residuals are measured,
not certified rounding bounds. No coefficient gap proves unseen identification.

Controls cover partial cache coverage, staleness, duplicates, short tails, wrong
IDs and changed content, both query modes, actual reconstructed logits, and the
nonzero complex CM105 finite-step counterexample and nonzero-error witness.
Every original split reports raw/exact accuracy, CE and a separate99% flag.
The original primary success predicate remains unchanged. Replays compare all
raw-logit hashes and metrics. Generated outputs remain outside the worktree.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/retained-readouts
export CARGO_BUILD_JOBS=2
cargo test --offline --locked --manifest-path experiments/convex_retained_readouts/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_retained_readouts/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_retained_readouts/Cargo.toml --check
cargo build --release --offline --locked --manifest-path experiments/convex_retained_readouts/Cargo.toml
```

Archive the producing commit with `git archive`, freeze its binary, protocol,
both input-pin files, plan output and SHA256 identities before any training.
Use one native worker and a32GiB address-space cap:

```sh
bash experiments/convex_retained_readouts/run.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/rotary-recurrence \
 /run/media/misha/storage/transformer-runs/freezes/retained-readouts-v1 \
 /run/media/misha/storage/transformer-runs/retained-readouts/original-v1 \
 /run/media/misha/storage/transformer-runs/diagnostics/retained-readouts-v1
```

Native admission checks enforce110GB aggregate RAM and both disk thresholds
before each minibatch. Low disk space stops work without deleting evidence.
Review repeats every diagnostic byte-for-byte and replays all twelve saved
endpoints in fresh copies. This is a prospective protocol; results follow only
after execution and verification. Generalization remains unresolved.
