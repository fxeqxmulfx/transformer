module

public import Transformer.Rotate90.Division
public import Transformer.Rotate90.Power
public import Transformer.Rotate90.Training
public import Transformer.Rotate90.Learning
public import Transformer.Rotate90.Memory

/-!
# Powered-division expressivity and a training obstruction for Rotate90 SwiGLU

Source: `grokking_abmod/docs/rotate90-swiglu-model.md` and the actual
`scripts/grok-simple.py` forward pass, inspected 2026-10-31.

`Rotate90.exists_division_weights` proves that for every prime p and every
positive RMSNorm epsilon, explicit one-block weights solve a/b on all a and
all nonzero b. Eight active residual coordinates and fourteen hidden units
suffice. Extra channel pairs and hidden units may be padded with zeros.
`exists_division_width16` covers the documented 3616-parameter p=97 model.

`Power.exists_weights` extends actual-model expressivity to every natural exponent,
including zero. Fourteen active residual coordinates and twenty hidden units suffice.
`Power.exists_weights_width16_any_depth` covers width 16, hidden size 43, and every
positive finite depth, by inserting zero-weight residual blocks after the active block.
All finite extra channel pairs and hidden units are allowed by the general theorems.

The construction retains all four normalizations, the fixed quarter-turn,
residual, genuine SiLU gates, and one shared embedding for input and output.
It uses a faithful multiplicative character plus a zero coordinate. Matching
the branch and input norms makes cancellation exact despite Peri-RMSNorm.
These are expressivity results in real arithmetic, independent of the
flat-spectrum interpolation surrogate and of optimizer convergence.

`Training.gradient_sharedLoss_aliased` supplies an actual full-table training
invariant at arbitrary finite width/depth. Under its subgroup/coprimality
conditions, `trainingRun_not_solves` proves that Schedule-Free AdamW cannot
separate exact aliases, even though successful weights exist. Partial-table
escape experiments and the theorem's scope are recorded in the source note.

`Learning` connects the entire finite network and actual sampled optimizer to
evaluation-loss movement. It proves conditional finite-time bounds using the
complete update direction and curvature over each step segment, and converts
full-softmax loss into a bound on classification failures. It does not infer
random-start convergence from static character rank or representational capacity.

`Memory` derives the exact zero-rate continuation with retained Schedule-Free
history. Evaluation converges to the saved base and inherits its strict logit
comparisons. Perfect antipodal powered-division solutions with a tied midpoint
prove that correctness cannot be transferred through arbitrary parameter averages.
-/
