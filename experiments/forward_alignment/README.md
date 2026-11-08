# Local Forward–Forward update alignment

Why did a numerically correct local-learning rule fail to train a useful
classifier? **The hidden block A is the strongest next diagnostic target.**
At all eight failed local endpoints its actual update lowers the hidden
local objective, yet the A-only update raises validation CE in seven.
This exposes a failure of local loss descent to translate into task
improvement at these states; it does not identify the entire trajectory's
cause or establish a repaired training rule.

The [protocol](PROTOCOL.md) was frozen at `378ea4df` before outcomes. We
examined all eight original initializations and all 32 endpoints from the
[complete pilot](../forward_training/README.md), preserving each state's
actual parameters and AdamW moments. Every counterfactual starts from the
same saved state. No training trajectory was extended.

## What was measured

Each of 40 states has seven one-step probes: CE BP, pair BP, auxiliary BP,
local pair, local without the vocabulary contribution to E, local without
the readout contribution to E, and zero current gradient with stored moments
and decay retained. Three further evaluations isolate E, A and B portions
of the local displacement. Zero current gradient still permits motion.
Embedding removals are distinct nonlinear AdamW counterfactuals, not pieces
whose optimizer displacements can be added.

For the update minibatch, complete training split and complete validation
split, measure terminal CE/pair gradients, their alignment with every
displacement, and actual loss/accuracy changes after each probe. Gradient
cosines use the negative-gradient update convention; displacement cosines
are `-gradient dot displacement / (norm(gradient)*norm(displacement))`.
Positive means first-order descent. The frozen classification dead zone
is ±0.001, and zero-norm cases have no cosine.

All norms use **143360 independent real coordinates**. Complex gradients
pull back through the signed matrix expansion; displacement coordinates
come from their owners. Expanded copies are not treated as independent
parameters when computing angles. Full-split gradients weight the 96/97-row
tail batches correctly. Validation derivatives are diagnostic work only.

## Failed local endpoints: actual updates differ from current gradients

The hidden block's current local gradient has positive cosine with the
terminal validation CE gradient in all eight cases, from 0.0296 to 0.0582.
Its **actual AdamW displacement** points uphill to first order in five.
Finite steps make the distinction sharper: two of the remaining three
first-order descent predictions also become increases in actual validation
CE. All eight A-only probes still lower their own frozen-boundary local loss.

| Arm | Seed | A raw-gradient cosine | A displacement descent cosine | A local loss change | A-only validation CE change | Complete local validation CE change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | 0.02964 | 0.02566 | -0.00246 | -0.00735 | -0.01048 |
| Complex diagonal | 7 | 0.03883 | 0.01548 | -0.00240 | +0.00185 | -0.00418 |
| Complex product radial | 7 | 0.05818 | -0.01236 | -0.01488 | +0.02310 | +0.01238 |
| Complex product plain | 7 | 0.04238 | -0.00410 | -0.02303 | +0.01443 | +0.00129 |
| Real SwiGLU | 11 | 0.04231 | -0.00230 | -0.00221 | +0.02140 | +0.02082 |
| Complex diagonal | 11 | 0.04866 | 0.00139 | -0.00301 | +0.02398 | +0.01956 |
| Complex product radial | 11 | 0.04984 | -0.01193 | -0.02501 | +0.02472 | +0.02316 |
| Complex product plain | 11 | 0.05144 | -0.00233 | -0.02267 | +0.01504 | +0.00668 |

The A-only finite-change minus first-order residual is positive in all
eight cases, 0.00845–0.02510. It is an observed nonlinear remainder, not a
Hessian bound or proof that any particular smaller step would be sufficient.
The A-only terminal **pair** loss also increases in seven cases.

| Local displacement portion | Own frozen local loss decreases | Full-training CE increases | Validation CE increases |
| --- | ---: | ---: | ---: |
| E | 8/8 | 7/8 | 5/8 |
| A | 8/8 | 8/8 | 7/8 |
| B | 8/8 | 0/8 | 0/8 |

The complete local update decreases update-minibatch CE in all eight cases,
but increases full-training CE in six and validation CE in six. B-only
updates decrease validation CE in all eight; validation pair loss decreases
in five of eight. Thus even a favorable
minibatch direction and lower local losses do not explain useful whole-task
learning. Full histories are needed before assigning a trajectory-level cause.

## Tied embedding and optimizer-state controls

Removing the vocabulary contribution makes the complete step's validation
CE **worse in all eight** local endpoints, by 4.36e-5–4.84e-4 relative to the
original local step. Removing readout E improves four and worsens four.
Neither observation establishes whether removing a head from initialization
would improve eventual generalization; the stored moments here were learned
with both contributions present.

The zero-current-gradient control raises validation CE in seven local
endpoints. Adding the actual local current gradient improves the finite
validation outcome relative to that control in all eight. For A, the
displacement difference from the zero-gradient control is a first-order
validation descent direction in all eight. These observations motivate
separating moment carryover, adaptive scaling and step size. They do not
isolate those effects individually: each counterfactual also changes the
AdamW denominator, and all include decay.

## All states, including the other controls

These rows concern the same **local** update evaluated at different saved
states; they are not rankings of the training methods. Each row has eight
cases and retains its own method's saved moments. Unlisted classifications
in the cosine columns are near orthogonal.

| State origin | A raw-gradient descent cases | A displacement descent / ascent | A-only finite validation CE increases | Complete local finite validation CE increases |
| --- | ---: | ---: | ---: | ---: |
| Initial | 6 | 5 / 2 | 3 | 0 |
| CE BP endpoint | 2 | 2 / 4 | 8 | 8 |
| Pair BP endpoint | 2 | 1 / 6 | 8 | 8 |
| Auxiliary BP endpoint | 4 | 1 / 7 | 7 | 7 |
| Local pair endpoint | 8 | 3 / 5 | 7 | 6 |

The full report also retains all seven rules at each state, both terminal
losses, all three scopes, each parameter group, zero-gradient contrasts,
cross-group interaction residuals and every finite-difference error. No
arm, seed or unsuccessful counterfactual is removed from the evidence.

## Next decision and limits

The completed [A direction/scale follow-up](../forward_step/README.md)
crosses two directions with five fixed scales at these same 40 states.
At local endpoints, norm-matched current-gradient motion lowers training
and validation CE in all eight cases at every scale. Shrinking AdamW recovers
two finite-step overshoots but leaves five validation ascent cases. This
motivates a separately frozen from-scratch training comparison. Its decisions
must use permitted local information; these validation gradients cannot be
a free line search or a training safeguard.

The fixed hidden target also remains an open design issue: the initial
minibatch has weak, mixed A alignment, and trained BP states do not generally
align with it. An objective linked to the final classifier would require a
separate, explicit definition and costed comparison. This study justifies
neither replacing complex multiplication nor declaring the optimizer the
sole cause of failure. It provides no accuracy repair or efficiency result.

## Checks, cost and reproduction

All **40 states pass**. All **1120** independent FP64 central-difference
checks pass without changing tolerances; maximum absolute error is 8.68e-7.
All **6720** expanded/independent coordinate pairings pass, maximum error
7.51e-14. Input-E reconstruction differs by at most 1.08e-8. All 40 exact
local-step comparisons match bitwise; all source/pilot split metrics replay.
Every counterfactual has valid parameter/moment ties. All **58 input files**
are unchanged. Thirteen Rust tests, formatting and warning-free Clippy pass.
No Lean statement changed.

The measured diagnostic cost is 9400 source forward batches, 1880 terminal
backward batches, 80 output-local Jacobians, 120 hidden-local Jacobians,
160 vocabulary Jacobians and 80 additional embedding backward passes.
There are 280 counterfactual optimizer steps plus 40 identity checks, all
restored; CPU verification evaluates 1160 complete backbones on 780880 rows
and 240 frozen local objectives. The 1160 backbone calls include 1120
finite-difference forwards and 40 caches for local-objective checks.
The cohort finishes in 33.35 seconds under its fixed 600-second cap.
GPU snapshots show this process and the desktop. This is attribution cost,
not a latency benchmark or computation available to a forward-only trainer.

The build reuses the source, local kernels and optimizer from task #190,
verifying its complete compiled-source digest and all PTX pins. Generated
source copies remain in Cargo output. No Python is used.

```sh
cargo test --offline --locked --manifest-path experiments/forward_alignment/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/forward_alignment/Cargo.toml -- -D warnings
cargo fmt --check --manifest-path experiments/forward_alignment/Cargo.toml
rustfmt --edition 2024 --check experiments/forward_alignment/src/instrumentation.rs
cargo build --offline --locked --release --manifest-path experiments/forward_alignment/Cargo.toml
mkdir -p experiments/forward_alignment/runs
experiments/forward_alignment/target/release/rotate90-complex-kernite \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/forward_training/runs/pilot-01 \
  experiments/forward_alignment/runs/diagnostic
jq -f experiments/forward_alignment/summary.jq \
  experiments/forward_alignment/runs/diagnostic/report.json \
  > experiments/forward_alignment/runs/diagnostic/summary.json
```

Use a fresh output directory; the executable refuses to overwrite evidence
or write inside either input directory. Executed evidence is retained at
`experiments/forward_alignment/runs/diagnostic-01/` in the primary checkout,
with check logs and the executable in `.kanban/artifacts/193/`. Its compiled
source digest is
`b2d670389e3dbfb36f1b95b42edfbe9fc90d3a94dfa8a95d06f148f8c034d3ee`.
Reports, binaries and logs are ignored and are not committed.
