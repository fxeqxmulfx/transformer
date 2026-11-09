# Query-weighted CE comparison

Freeze this protocol and the complete constructor/verifier implementation
before evaluating the new coefficient on any recreated checkpoint. The
completed first two audits motivated this continuation; their sources,
forecasts and outcomes remain immutable. The new mathematical layer is
`fc7a3a3a`, and the independently tested native evaluator is `70176b61`.
No new training or optimizer modification belongs to this comparison.

## Inventory and access boundary

Retain the same three cohorts (`lr001`, `lr01`, `settle`), eight seeds
`7,11,1,3,5,13,17,19`, and anchors `0,1000,4000,6000,8000,9500`.
Keep every horizon `0,1,2,4,8,16,32,64,128,250,500,1000,4000` whose endpoint
is at most 10000: 144 anchors and 1800 endpoints. Keep both original and
pairwise-refined probability/feature forecasts, both checkpoint-fixed metrics,
and all three coefficient methods below. Report initialization separately
from the 120 moving anchors; do not discard initialization measurements.

The constructor may read each source identity, its named anchor checkpoint,
both old-only construction reports/seals and the matching frozen forecast
payloads. It must verify seals, original source identities and forecast
hashes. The expected original/refined source fingerprints are
`b237a39dddb8fc1f905849c390c8cbc7f4f5cb3cdd1d962a2c174a6e08fa1181`
and `e1a3a4adab7f179aa5e45b552f8b4fefc9f00cc8c3cef05ab81ead0b702d9d68`.
It may execute only the old model on its 4704 training inputs. It cannot open
training reports, later checkpoints, traces, gradients or previous future-state
verification reports. Seal each new case payload and then the complete
construction report before running the verifier. Keep an explicit input ledger.

Use the exact recovered trainer, Kernite and optimizer pins from the original
reproduction. The actual source Plan uses batches of 512 and a final batch of
96 in each 4704-example epoch. Recreate its order and boundaries, validating
the order hash; do not replace it by a fixed-size approximation. For horizon H
take batches at states `anchor..=min(anchor+H,9999)`. Thus H=0 covers the
current batch and H=1 includes the next state's batch. This inclusive convention
preserves the previous future-state meaning. State 10000 is retained as a
terminal endpoint without inventing a subsequent batch or gradient.

## Coefficients fixed before future evidence

For each source family and horizon, use all its sealed per-query probabilities
q_i and ideal feature-motion radii S_i. Set

```
q_i^+ = probabilityCeiling(q_i, 2*1e-3),
S_i^exec = 0 at H=0; otherwise S_i + 2*1e-3,
M = 128*(1+5e-6), R = max_k r_k.
```

The first conversion explicitly covers real-CE dot products through the
existing per-logit error allowance. The second covers both endpoint feature
execution errors. Retain s and r exactly from the sealed anchor; the isotropic
metric has r=1. No future clipping, gradient, feature alignment, state or
singular vector may tighten construction. Future common clipping is bounded
by one. All maxima below include every known batch in the declared window.

Compare these coefficients L and their stability numbers `19*s*L`:

1. Uniform: `R*M*max_all_inputs(q_i^+)`, with the same explicit conversion.
2. Rowwise trace: maximum over batches of `R*M*mean_batch(q_i^+)`.
3. Transported Gram: maximum over batches of the smaller of its rowwise trace
   and `(sqrt(L_old)+sqrt(R*mean_batch(q_i^+*(S_i^exec)^2)))^2`.

For each batch, form `C_kl=sum_i q_i^+*T_ik*T_il` from its old executed
features. `L_old` is the smaller of the outward-expanded weighted trace and
`sqrt(sum_kl r_k*r_l*C_kl^2)/batch_size`. Every off-diagonal entry remains.
Reuse the same covariance for both metrics. Use the explicit FP64 dot-error
allowance and outward positive reductions in `gram/covariance.rs` and
`gram/lib.rs`; retain their checked input ranges and fail on invalid values.
The logarithmic/exponential ceiling conversion remains a numerical operation,
not a proved floating-point enclosure. Report this limitation explicitly.

A deterministic optimization may omit a batch's Gram product if the transported
Gram expression evaluated with old coefficient zero is already at least its
rowwise trace. Nonnegativity and monotonicity then guarantee that the selected
minimum is the trace for any admissible old coefficient. Zero must never be
used as a claimed old-operator upper bound. Record evaluated and trace-dominated
batch counts, and mark unevaluated Gram quantities as such. All batches still
enter the maximum; this is not a subset chosen from outcomes.

Record each method's maximum coefficient, the attaining batch, all useful
horizons (including zero), and the largest useful positive horizon or zero.
A prospective raw CE condition passes exactly when its stability number is
at most two; numerical tolerances do not change this classification. Do not
interpret a smaller coefficient, or a positive reference horizon, as control
of full AdamW or of geometry discovery.

## Verification and retained full-model outcomes

The verifier checks all new seals, source fingerprints and input hashes and
reconstructs each payload through the same old-only interface. Only after
construction is sealed may it read future checkpoints and the two completed
source verifications. Verify those source verifications' input ledgers and
their correspondence to the unchanged source forecast seals. Retain all
their full-parameter, ideal-feature, contrast and GPU-probability findings.

At every declared endpoint recompute executed features/logits for all training
inputs. Check real-CE per-query probabilities against q_i^+ and executed-feature
motion against S_i^exec, and retain all excesses before tolerance. Check the
same feature-energy and per-logit execution allowances. At nonterminal states
execute the actual source minibatch, require its feature values to equal the
corresponding all-input snapshot values, and independently compute the real
partial CE gradient. Check it against the source GPU partial gradient with
the existing scaled tolerance 1e-5. Record the complete raw and clipped
unused-gradient radial/weighted-energy values and the prospective inequalities
for every method. Retain the reference-potential changes with the actual
clipping scalar, arbitrary saved momentum, and the fixed anchor metric/step.
These future quantities are diagnostics only.

Recompute each endpoint's full saved parameter/moment potential in both fixed
anchor metrics and require agreement with both earlier audits. Preserve every
increase, including increases despite a reference-condition success. Retain
the earlier anchor-to-next-update complete signed decomposition, unchanged.
Terminal endpoints keep their probability, feature and full-potential checks;
their next-minibatch diagnostics are explicitly unavailable.

Use the existing probability absolute tolerance 1e-8, scaled inequality and
potential tolerance 1e-8, execution-feature norm allowance 1e-3, readout-dot
allowance 1e-3 and energy cap M. Record maximum excesses before tolerances and
every failed check. A failed coefficient condition is an outcome, not a
numerical error. A numerical error, missing input, cap or failed reconstruction
keeps its slot and error in the report. Finite endpoint checks do not establish
the execution allowances at every intermediate update. The proved whole-window
reference guarantee remains conditional on those allowances throughout it.

## Budget and evidence

Construction, all verification attempts (including old-only reconstruction),
input-ledger checks and the final comparison share a cumulative 7200-second
budget. Compilation, the already completed source studies and synthetic tests
are excluded. The pre-study synthetic 512-by-128, two-metric benchmark takes
about 1.81 ms per batch on this machine. The longer independent budget accounts
for both source families, all known batches and complete reconstruction.
Record all attempts and charge them cumulatively; never overwrite a failed
attempt or hide capped cases. Use single-threaded BLAS for this comparison.

Preserve generated forecasts, seals, reports and summaries under the root's
ignored `experiments/probability_tube/runs/`; preserve source fingerprints,
executable hashes and logs. Commit authored findings, source, configuration
and generators only. Strict clippy, native independent tests and format checks
must pass before evaluating any recreated checkpoint with the new coefficient.
