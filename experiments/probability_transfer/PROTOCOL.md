# Training-to-query probability transfer on frozen cubic checkpoints

Fixed 2026-11-08, task #182, before measurement. No training or optimizer
intervention. Retain all 24 original states: four arms, seeds 7/11, and steps
0/5000/10000. Reuse flat_approx's FP64 source, frozen scalar oracle and split
reconstruction. Verify state/history/split hashes, sampled scalar logits and
all saved correctness counts. Four OpenBLAS threads; five-minute analysis cap
excluding compilation. Preserve all earlier reports and checkpoint files.

Construct using raw operand pairs, the actual final-normalized states, tied
readout weights, and training-index/label pairs. No query-label, query-logit,
query-probability or target-error argument enters the constructor. Normalize
each residual state by its actual final RMS denominator, with the original
epsilon. This requires query evaluation; it is not a forecast without running
the model. Model/oracle verification computes logits separately.

Group by exact sums modulo 97, or by their orbits under 1, 35 and 61. The
constructor checks an explicit allowed multiplier/translation route from a
training anchor to every member and verifies observed training-label agreement.
Do not cube query sums to create labels. No training anchor means abstention.
For training queries, exclude the query itself from both anchors and centers;
with no remaining training reference, abstain.

For target t and competitor c, reference state h_i and query state h, use

```
U_i,c = log(p_i,c/p_i,t) + ||h-h_i||_2 * ||E_c-E_t||_2
R = sum_(c != t) exp(U_c)
query p_t >= 1/(1+R),     query CE <= log(1+R).
```

Training log odds are computed stably as training logit differences. The
training center uses mean normalized states and mean training log odds; it is
a synthetic reference, not the arithmetic mean of training probabilities.
No query class score is evaluated inside the bound constructor.

Fix five methods: training center, nearest training anchor by Euclidean state
distance (ties by index), best whole training anchor by total bounded odds,
the per-competitor envelope over all training anchors, and the per-competitor
envelope including the training center. Retain every method. Each competitor
may select its own valid bound; training-label consistency fixes the candidate
answer. This is a finite choice of proved bounds, not fitting a scale to query
targets. Compute state distances explicitly from coordinate differences.

Hash both complete constructions, including candidate labels and per-query
bounds, before the diagnostic audit accesses query labels and logits. At zero
assumed execution error, record the probability condition R<1 and the weaker
argmax condition max_c U_c<0. Report both coverages and incorrect positive
counts. The available per-logit execution allowance for the probability
condition is -log(R)/2; it is a budget, not an established backend error bound.

Audit the probability floors, CE ceilings and minimum-logit-gap bounds against
the actual source, with 1e-8 absolute tolerance. Retain every failed condition,
abstention, group count and all 24 cases. Report full split mean CE and its
constructed upper bound, including failed conditions. FP64 computations are
not certified arithmetic; the previously examined validation split is not an
independent statistical test. No learned equivariance, optimizer convergence
or universal coverage is assumed.
