# Fixed signed directional transport diagnostic

Fixed 2026-11-08, task #183, before measurement. Retain all 24 original cubic
states: four arms, seeds 7/11, steps 0/5000/10000. No training or intervention.
Reuse flat_approx's source, scalar oracle and split reconstruction; verify all
state/history/split identities and saved correctness counts. Four OpenBLAS
threads; five-minute analysis cap excluding compilation. Preserve earlier runs.

The constructor receives operand pairs, actual final-normalized states,
readout weights and training-index/label pairs. It never receives query labels,
query logits, query probabilities or target errors. Only training states are
read out. Known sum or cubic-orbit routes transfer observed training labels;
query sums are not cubed to create labels. Missing references cause abstention.

Validation queries use all training references in their group. Training
queries use the opposite half of the group's training references, split by
alternating sorted training indices. This fixed two-fold construction prevents
self reconstruction. Retain both training and validation results; their
reference sets differ. Check label agreement and explicit allowed routes.

For each eligible training set, center normalized states at their mean and
compute a thin real SVD. Keep singular directions with s > 1e-10 * s_max.
Fix requested ranks 0,1,2,4,8,16,32,64,128; clamp each to the available rank.
The basis uses training states only, without readout weights or query states.
Do not select a rank using held-out results. Record achieved rank, singular
spectrum and retained training-state energy. Rank 128 is the ambient-dimension
limit, which can re-express the original readout; it is not automatically
evidence of a compressed representation or optimizer discovery.

Let T_i be centered training states, d the query displacement, B_j the retained
right singular vectors and A_i,j = U_i,j/s_j. Compute alpha_j = <B_j,d>,
a_i = sum_j alpha_j*A_i,j, and the actual residual r = d - sum_i a_i*T_i.
Keep the signed training contribution

```
prediction_c = mean_training_log_odds_c
             + sum_i a_i*(training_log_odds_i,c - mean_training_log_odds_c).
```

For w_c = E_c-E_t, b_c,j = <B_j,w_c> and
u_c = w_c - sum_j b_c,j*B_j, compare two fixed bounds at every rank:

```
signed_c = prediction_c + ||r||_2*||w_c||_2
dual_c   = prediction_c + ||r||_2*||u_c||_2
         + sum_j |b_c,j|*|<B_j,r>|.
```

The second bound retains the orthogonality defect explicitly; no exact
orthogonality of numerical SVD or reconstruction is assumed. Readout residuals
depend only on weights and the training basis; query residuals and their basis
scores are geometric. Do not evaluate <w_c,query> inside the constructor.
Rank zero reduces both formulas to the normalized training-center baseline.

Hash both complete constructions before the separate audit accesses query
labels and logits. The source/oracle evaluator computes logits separately.
For each method record R=sum_(c!=t) exp(bound_c), probability floor 1/(1+R),
CE ceiling log(1+R), minimum-gap floor -max bound_c, the probability condition
R<1, the weaker argmax condition max bound_c<0, and incorrect positive counts.
Report full-split mean CE ceilings including failed conditions; use null for
a full-split ceiling if any reference is missing. Record residual norms,
coefficient norms, remaining readout norms and orthogonality defects.

Audit probability, CE and gap bounds at 1e-8 absolute tolerance and retain every
case, rank, failure and abstention in aggregate results. Execution eta starts
at zero; the available per-logit allowance -log(R)/2 is a budget, not a proven
backend error. These FP64 audits are not certified arithmetic or an independent
statistical test. Query evaluation and known task symmetry are required.
