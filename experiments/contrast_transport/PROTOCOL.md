# Training-state transport on frozen cubic checkpoints

Fixed before measurement, 2026-11-08, task #180. No model training. Keep all
24 original states: four arms, seeds 7/11, steps 0/5000/10000. Reuse the verified
FP64 source, split and scalar oracle from flat_approx at 9328bc83. Verify all
state/history/split identities, sampled scalar logits and stored correctness
counts. Preserve both earlier diagnostics. Use four OpenBLAS threads and a
five-minute analysis cap, excluding compilation.

Construct bounds without access to query labels, query logits, or query target
errors. The constructor receives only operand pairs, raw residual states, tied
embedding weights, and training-index/label pairs. It returns candidate labels
and lower margin bounds. Hash that construction before a separate audit reads
query labels and logits. This is an input-dependent deterministic condition,
not an optimizer theorem or a prediction made without evaluating query states.

Declare two ways to group inputs, both preserving the cubic answer:

- Equal sums modulo 97.
- Sums in the same orbit under multiplication by 1, 35 and 61. The key is the
  smallest of those three residues, calculated without cubing the query sum.

Each nonempty training group supplies a label from a training anchor. Check
an explicit multiplier route from that anchor to every group member, and
check that all observed training labels in the group agree. A query group
without a training anchor abstains. Never infer a missing label from the task
formula during construction. The route's label preservation is proved in Lean.

For each group, let mu be the mean training state, H_i = h_i-mu, and
d_v(i) = (E_target-E_v) dot H_i. Compute each center contrast g_v and its
training variation sum_i d_v(i)^2. All 97 classes compete.

For each query displacement e=h_query-mu and each declared ridge scale,
solve (H^T H + lambda I) z=e, set a=H z and r=e-H^T a, and form

```
G = sum_i a_i^2 + ||r||_2^2/lambda
B_v = sum_i d_v(i)^2 + lambda*||E_target-E_v||_2^2
query contrast >= g_v - sqrt(G*B_v).
```

Retain the reconstruction residual explicitly even when the training states
are rank deficient. Use lambda = factor * max(trace(H^T H)/width, 1e-12),
with the fixed factors [1e-4, 1e-2, 1, 100]. Record every factor. Also compute
the Euclidean-center bound g_v-||e||_2*||E_target-E_v||_2 as a baseline.
For each group method, report individual coverage, the union over complete
query certificates, and the bound taking the best valid choice separately
for each competitor. Choosing among valid upper bounds uses no target errors.

A query passes only if every competitor lower bound is strictly positive.
Numerical execution error is initially zero. Record the smallest bound,
transport energies, residual energies, training counts, regularization and
route checks. Preserve all abstentions and all failed conditions.

After freezing construction, audit candidate labels against the task, and
check every positive bound against actual source margins and correctness.
Report train and held-out coverage, actual accuracy, certified-but-incorrect
counts, worst numerical bound violations and the source final denominator.
All computations are FP64 diagnostics, not a certified floating-point proof;
positive results still need verified arithmetic/backend error bounds.
The original validation split has already been examined. No independent
statistical coverage or optimizer-discovery claim is made.
