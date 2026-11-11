module

public import Transformer.Grokking.Protocol.Decision
public import Transformer.Grokking.Protocol.Sampler
public import Transformer.Grokking.Protocol.Adam
public import Transformer.Grokking.Protocol.AdamSource
public import Transformer.Grokking.Protocol.AdamBounds
public import Transformer.Grokking.Protocol.AdamGrowth
public import Transformer.Grokking.Protocol.AdamBox
public import Transformer.Grokking.Protocol.Target
public import Transformer.Grokking.Protocol.Witness
public import Transformer.Grokking.Protocol.Model
public import Transformer.Grokking.Protocol.Causal
public import Transformer.Grokking.Protocol.Gradient
public import Transformer.Grokking.Protocol.Division
public import Transformer.Grokking.Protocol.DivisionTarget

/-! The protocol behind the division curves of Power et al., *Grokking: generalization beyond
overfitting on small algorithmic datasets*, and what "held-out accuracy tends to 1" means for
it: the prediction with its tie rule, the competitor margin and the accuracy; the split and
the batches as functions of a run's shuffles; the author's Adam and AdamW update with its
schedule, statement by statement as in the code, the bounds of its moments, and how far it
can move a parameter whatever the gradients; the law of a run and the three modes of
convergence; a model that generalizes beside one that does not; the decoder in exact reals
with every parameter group, its loss, and the gradient of the source's backward pass with its
rule at the kinks of the activation; division modulo 97 with its tokens, supervision and
initialization law; and the target for that protocol as an equivalence. The target is a
predicate of the model, its labels and its gradient oracle: no convergence of any transformer
is proved here. The map of what is proved elsewhere and what is missing is
`review/grok_division_convergence.md`. -/

@[expose] public section
