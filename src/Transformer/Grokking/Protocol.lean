module

public import Transformer.Grokking.Protocol.Decision
public import Transformer.Grokking.Protocol.Sampler
public import Transformer.Grokking.Protocol.Adam
public import Transformer.Grokking.Protocol.AdamSource
public import Transformer.Grokking.Protocol.AdamBounds
public import Transformer.Grokking.Protocol.AdamGrowth
public import Transformer.Grokking.Protocol.AdamBox
public import Transformer.Grokking.Protocol.AdamLimit
public import Transformer.Grokking.Protocol.AdamLimitWitness
public import Transformer.Grokking.Protocol.Target
public import Transformer.Grokking.Protocol.Witness
public import Transformer.Grokking.Protocol.Model
public import Transformer.Grokking.Protocol.Causal
public import Transformer.Grokking.Protocol.Gradient
public import Transformer.Grokking.Protocol.Division
public import Transformer.Grokking.Protocol.DivisionTarget
public import Transformer.Grokking.Protocol.Symmetry
public import Transformer.Grokking.Protocol.AdamSymmetry
public import Transformer.Grokking.Protocol.Uncovered
public import Transformer.Grokking.Protocol.Exchange
public import Transformer.Grokking.Protocol.Refutation
public import Transformer.Grokking.Protocol.RefutationModes
public import Transformer.Grokking.Protocol.MarginTransfer
public import Transformer.Grokking.Protocol.QuotientTransfer
public import Transformer.Grokking.Protocol.QuotientWitness
public import Transformer.Grokking.Protocol.DiscreteLog
public import Transformer.Grokking.Protocol.NormScale
public import Transformer.Grokking.Protocol.SolverParams
public import Transformer.Grokking.Protocol.SolverLookup
public import Transformer.Grokking.Protocol.SolverAttention
public import Transformer.Grokking.Protocol.SolverGap
public import Transformer.Grokking.Protocol.SolverForward
public import Transformer.Grokking.Protocol.SolverLogits
public import Transformer.Grokking.Protocol.SolverScale
public import Transformer.Grokking.Protocol.OrbitAnchors
public import Transformer.Grokking.Protocol.HeldOutCertificate
public import Transformer.Grokking.Protocol.Enclosure

/-! The protocol behind the division curves of Power et al., *Grokking: generalization beyond
overfitting on small algorithmic datasets*, and what "held-out accuracy tends to 1" means for
it: the prediction with its tie rule, the competitor margin and the accuracy; the split and
the batches as functions of a run's shuffles; the author's Adam and AdamW update with its
schedule, statement by statement as in the code, the bounds of its moments, how far it can
move a parameter whatever the gradients, and the equation `wd · θ = −g / (|g| + ε)` of a limit
of a run on one gradient field, which the minimizers of a weight-decayed loss fail in two
witnesses; the law of a run and the three modes of
convergence; a model that generalizes beside one that does not; the decoder in exact reals
with every parameter group, its loss, and the gradient of the source's backward pass with its
rule at the kinks of the activation; division modulo 97 with its tokens, supervision and
initialization law; the target for that protocol as an equivalence; the proof that this
target is false without a condition on the split: a uniformly random half of the equations
misses two residues with positive probability, exchanging their tokens is then a symmetry of
the whole run, and the run and its exchange cannot both answer the held-out equations; and a
sufficient condition for every answer: a row of logits within `ε` of a relabelled row keeps
its margin less `2 ε`, so two equations answered with margins above `2 ε`, one with a nonzero
quotient and one with the quotient zero, give all 9312 answers when the logits move with the
scalings of numerator and denominator up to `ε`, with a witness of positive error and the
decoder with every parameter zero, which has the symmetry and no answer; parameters of the
decoder, built and not trained, that answer all 9312 equations and satisfy that condition
with a positive error: two attention heads subtract the logarithms to the base 5 of the
numerator and the denominator, and the hidden units look the quotient up; and the same
condition as a certificate of the held-out accuracy from training answers, with an anchor for
each held-out equation or one for each orbit, the splits that give an anchor to each orbit,
and parameters that fit every training answer of a split and miss a held-out one; and what two
rows that enclose the logits, as a program that rounds outward computes them, decide about the
real ones: the answers, the accuracy and the premise of that certificate. The target is a
predicate of the model, its labels and its gradient oracle: no convergence of any transformer
is proved here, and no trained decoder is proved to have that structure, for an enclosure of
its logits is a hypothesis here. The map of what is proved elsewhere and what is missing is
`review/grok_division_convergence.md`. -/

@[expose] public section
