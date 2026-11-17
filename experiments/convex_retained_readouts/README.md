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
endpoints in fresh copies. Generalization remains unresolved.

Producer `5380ac97` completes the declared twelve runs. The combination of
retained coefficients and smaller steps improves every corresponding starting
model's training, novel and test accuracy. Neither change alone improves training
accuracy. These are paired comparisons on seed 1, not a multi-seed claim.

| Arm | Source/mode | Train raw | Train CE | Novel raw | Test raw |
| --- | --- | ---: | ---: | ---: | ---: |
| instantaneous-small | cautious Mini | 58.2520% | 2.461293 | 12.3047% | 11.3281% |
| instantaneous-small | cautious forward | 60.1074% | 2.454595 | 11.7188% | 10.9375% |
| instantaneous-small | refined Mini | 58.0078% | 2.475931 | 10.7422% | 9.5703% |
| instantaneous-small | refined forward | 57.2754% | 2.490214 | 10.9375% | 12.6953% |
| retained-large | cautious Mini | 41.9434% | 2.104067 | 14.0625% | 12.5000% |
| retained-large | cautious forward | 37.7441% | 2.192248 | 10.9375% | 11.5234% |
| retained-large | refined Mini | 42.4316% | 2.104738 | 12.8906% | 13.4766% |
| retained-large | refined forward | 34.4727% | 2.339390 | 11.1328% | 10.7422% |
| retained-small | cautious Mini | 76.1719% | 1.882480 | 18.9453% | 14.4531% |
| retained-small | cautious forward | 77.7832% | 1.858476 | 15.2344% | 16.6016% |
| retained-small | refined Mini | 76.4648% | 1.881731 | 18.1641% | 16.0156% |
| retained-small | refined forward | 74.5605% | 1.908065 | 16.7969% | 16.4063% |

The four starting projected-v1 models have training accuracy 64.6484–68.7500%,
training CE 2.032235–2.085518 and novel accuracy 11.5234–14.8438%. The pinned
instantaneous-large controls collapse to train 3.7109–4.0527% and novel
1.9531–3.5156%; their reports match the reviewed f5ae34ef evidence. Raw and exact
accuracy agree on these single-target records. Every new original success flag
and every split's 99% flag is false. All twelve complete raw replays are identical.

The coefficient diagnostics distinguish numerical accumulation error from
staleness. Independent recomputation of all stored signals agrees with the
incremental mean within 2.95e-16. Nevertheless, retained-large has estimated
gaps 0.643–0.887, CM105 corrections 6.152–9.345, and actual joint gaps
3.026–4.082. All four saved estimated directions have **positive** actual
full-training directional change, 0.0210–0.4049. A small estimated gap here is
misleading unless its coefficient error is included.

Retained-small reduces actual joint gaps to 0.4537–0.4704 and corrections to
0.3646–0.4719. All four estimated directions have negative actual change,
from -0.4487 to -0.2727. The sufficient corrected-gap condition holds in both
cautious cells, and is inconclusive in both refined cells. The measured gap and
direction inequalities have zero positive excess at the declared tolerance.
These are floating evaluations of the Lean inequalities, not rounding proofs.
The corresponding actual conditional CE lower bounds are about 1.391–1.454;
they concern fixed learned content and do not establish an accuracy ceiling.

All 6,144 scheduled updates finish; 5,230 have a positive accepted step.
Retained-large accepts 441–467 of 512 steps, and retained-small 318–356.
Every retained cache covers all 2,048 indexed training records with 16,384
refreshes. All models retain 129 components and unchanged learned content.
The twelve runs use 23,396,352 CE evaluations, including 19,660,800 finite
signed-bias queries; the read-only diagnostics use another 98,304. Maximum
observed aggregate RSS is 10.401 GB, logit reconstruction error 2.31e-14,
operator-norm excess 2.23e-16 and shared-capacity excess zero.

This repairs part of the optimizer failure, but leaves a large training/novel
gap. The next work must address learning shared content as well as controlling
coefficient staleness. No current endpoint meets the requested 99% target.
Evidence is under `transformer-runs/freezes/retained-readouts-v1`,
`retained-readouts/original-v1` and `diagnostics/retained-readouts-v1` on storage.
The frozen source fingerprint is
`738b1d49942d9daebb6665bde271683c073f13d3c0ede37485b5b879b4676905`.
