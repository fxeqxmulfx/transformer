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
public import Transformer.Grokking.Protocol.ReadoutBound
public import Transformer.Grokking.Protocol.TargetMean
public import Transformer.Grokking.Protocol.ReadoutGradient
public import Transformer.Grokking.Protocol.ReadoutPairing
public import Transformer.Grokking.Protocol.AdamStationary
public import Transformer.Grokking.Protocol.ReadoutStationary
public import Transformer.Grokking.Protocol.ReadoutWitness
public import Transformer.Grokking.Protocol.DecayedLogits
public import Transformer.Grokking.Protocol.ReadoutSymmetry
public import Transformer.Grokking.Protocol.Unlabelled
public import Transformer.Grokking.Protocol.LabelCoverage
public import Transformer.Grokking.Protocol.LabelWitness
public import Transformer.Grokking.Protocol.LabelProbability
public import Transformer.Grokking.Protocol.Continuity
public import Transformer.Grokking.Protocol.LimitNotFitted
public import Transformer.Grokking.Protocol.Differentiability
public import Transformer.Grokking.Protocol.OffKinks
public import Transformer.Grokking.Protocol.GradientMeasurable
public import Transformer.Grokking.Protocol.RunMeasurable
public import Transformer.Grokking.Protocol.AccuracyMeasurable
public import Transformer.Grokking.Protocol.DivisionMeasurable
public import Transformer.Grokking.Protocol.MeanAccuracy
public import Transformer.Grokking.Protocol.RenameSymmetry
public import Transformer.Grokking.Protocol.CarriedSplit
public import Transformer.Grokking.Protocol.TwistedSplits
public import Transformer.Grokking.Protocol.TwinSplit
public import Transformer.Grokking.Protocol.TwinWitness
public import Transformer.Grokking.Protocol.LabelNotSufficient

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
rule at the kinks of the activation, which is the derivative of the loss with frozen gates at
every parameter and the gradient of the loss where no hidden unit has the input zero; division
modulo 97 with its tokens, supervision and initialization law; the target for that protocol
as an equivalence; the proof that this
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
real ones: the answers, the accuracy and the premise of that certificate; and two ways of
proving a convergence that the runs themselves exclude: by the gradient of the readout in
closed form and its pairing with the readout, a run of Adam without weight decay whose
parameters converge does not keep its batches fitted with a margin, and its limit, the decoder
being continuous in its parameters, does not fit the training set with a margin; and by the
box of AdamW, for almost every run with weight decay 1 the logits stay bounded, the loss of a
batch stays above a positive floor and no margin grows without bound; and a condition on the
split that every true form of the target needs: a residue that is the answer of no training
equation has a token that is no training target, exchanging its readout row with that of a
token that division never shows is then a symmetry of the run, and so the target is false
under every condition that admits a split with such a residue, the condition that the split
has an anchor for each orbit and shows every residue among them, almost surely and in
probability: given such a split a held-out answer is wrong with a conditional probability of
at least one half at every update; and the measurability that the two weaker modes need: the
gradient of the backward pass is measurable in the parameters, so every parameter of a run,
the held-out accuracy and the event of a wrong held-out answer are measurable in the sample,
the almost sure target of a model with measurable logits and gradients gives the other two,
and for division the mean held-out accuracy stays below `1` by a positive constant at every
update, so the target fails in mean as well; and that the condition on the split is not
enough: a renaming of the tokens that carries the training examples of a split to those of
another, with the list of equations reordered, is a symmetry of the run, and a split that
labels every residue while its training equations show the residues `2` and `95` in
`2 / 1 = 2` and `95 / 1 = 95` only is carried so by the exchange of the two tokens, which
turns the held-out `2 / 3` into `95 / 3`, so the target is false on the splits that label
every residue, in all three modes. The target is a predicate of the model, its labels and its
gradient oracle: no convergence of any transformer is proved here, and no trained decoder is
proved to have that structure, for an enclosure of its logits is a hypothesis here. The map
of what is proved elsewhere and what is missing is `review/grok_division_convergence.md`. -/

@[expose] public section
