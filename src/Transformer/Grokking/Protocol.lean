module

public import Transformer.Grokking.Protocol.Decision
public import Transformer.Grokking.Protocol.Sampler
public import Transformer.Grokking.Protocol.Adam
public import Transformer.Grokking.Protocol.AdamBounds
public import Transformer.Grokking.Protocol.Target
public import Transformer.Grokking.Protocol.Witness

/-! The protocol behind the division curves of Power et al., *Grokking: generalization beyond
overfitting on small algorithmic datasets*, and what "held-out accuracy tends to 1" means for
it: the prediction with its tie rule, the competitor margin and the accuracy; the split and
the batches as functions of a run's shuffles; the author's Adam and AdamW update with its
schedule and the bounds of its moments; the law of a run and the three modes of convergence;
and a model that generalizes beside one that does not. The target is a predicate of the model,
its labels and its gradient oracle: no convergence of any transformer is proved here. The map
of what is proved elsewhere and what is missing is `review/grok_division_convergence.md`. -/

@[expose] public section
