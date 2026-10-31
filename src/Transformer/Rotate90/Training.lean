module

public import Transformer.Rotate90.Training.Failure
public import Transformer.Rotate90.Training.Sampling

/-!
# Actual Rotate90 training can preserve an incorrect quotient code

Source: `review/flat_spectrum.md`, *Alias symmetry and training*. At every finite
width and depth, the actual full-table embedding gradient preserves invariance
under a multiplicative subgroup S when gcd(K, |S|)=1. All three uses of the tied
embedding, every RMSNorm, the residual stack and the full softmax are retained.

`Training.trainingRun_aliased` propagates this property through the source's
Schedule-Free AdamW buffers. `trajectory_aliased` also covers external schedules.
`trainingRun_not_solves` proves that a nontrivial subgroup prevents a unique
correct full-table classifier forever. `trainingRun_loss_floor` gives log |S|
on nonzero targets. `aliased_limit` and `limit_not_solves` preserve the obstruction
at a pointwise limit, if one exists. Nonzero initialization examples accompany
the theorems. The branch trajectory is arbitrary, allowing joint branch training.

These results complement the actual-model capacity construction in `Power`.
They disprove an all-initializations convergence assertion, not convergence
from a specified random initialization distribution. Partial-table experiments
have nonsymmetric gradients; their observed plateaus are not proved permanent.

`Sampling` proves an actual escape mechanism: at collapsed nonzero embeddings,
sampled label imbalance creates row-separating gradients. A valid singleton
breaks aliases in one source Schedule-Free update, even where the full division
table has zero embedding gradient. Its explicit finite-depth witness includes
both query and evaluation parameters; it does not imply eventual generalization.
-/
