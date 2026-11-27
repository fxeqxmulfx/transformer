module

public import Transformer.Grokking.FlatSpectrum
public import Transformer.Grokking.FiniteRecovery.ArgandSamples
public import Transformer.Grokking.FiniteRecovery.Capacity
public import Transformer.Grokking.FiniteRecovery.SeriesAttention
public import Transformer.Grokking.FiniteRecovery.SeriesGrid
public import Transformer.Grokking.FiniteRecovery.SeriesStability
public import Transformer.Grokking.FiniteRecovery.PeriodicForecast
public import Transformer.Grokking.FiniteRecovery.TransitionTraining
public import Transformer.Grokking.FiniteRecovery.CouplingConvergence
public import Transformer.Grokking.FiniteRecovery.SeriesLearning
public import Transformer.Grokking.FiniteRecovery.TransitionNoDecay
public import Transformer.Grokking.FiniteRecovery.CommonBatchGradient
public import Transformer.Grokking.FiniteRecovery.ArgandMomentumCopy
public import Transformer.Grokking.FiniteRecovery.ArgandFullGradientConvergence
public import Transformer.Grokking.FiniteRecovery.ArgandRecoveryObstruction
public import Transformer.Grokking.FiniteRecovery.ArgandStrictRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandCapacityExact
public import Transformer.Grokking.FiniteRecovery.ArgandDecayObstruction
public import Transformer.Grokking.FiniteRecovery.ArgandObservationProbability
public import Transformer.Grokking.FiniteRecovery.ArgandGaussianRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandFlatLimit
public import Transformer.Grokking.FiniteRecovery.ArgandParityFeatures
public import Transformer.Grokking.FiniteRecovery.ArgandLearnedNorm
public import Transformer.Grokking.FiniteRecovery.TransversalTraining
public import Transformer.Grokking.FiniteRecovery.TransversalExamples
public import Transformer.Grokking.FiniteRecovery.ArgandModularSum
public import Transformer.Grokking.FiniteRecovery.ArgandPAGESchedule
public import Transformer.Grokking.FiniteRecovery.ArgandPAGERestarts
public import Transformer.Grokking.FiniteRecovery.CheckpointFactors
public import Transformer.Grokking.FiniteRecovery.AdamWFloor
public import Transformer.Grokking.FiniteRecovery.FlatAnchoredRecovery

/-! Finite recovery from observed labeled words. The full single-stream
Argand forward pass and Flat Spectrum share the same targets and observation
masks. The examples compare exact observation thresholds, embedding and FFN
widths, and repeated product layers. Series windows can use independently
compiled difference codecs, or share a grid with axial LeRoPE and a time mask.
The training examples cover modular sums, bilinear couplings, encoded and
additive periodic series, and compiled finite-state laws. Argand's native
parameter layout supports both the bias-corrected AMSGradW control and
default clipped momentum SGD from the initialized draws. For SGD on the
positive WSD floor, the criterion is convergence of accumulated clipped
gradients. Recurring singleton observations of the binary copying task
preclude a finite limit with a strict correct readout. Full gradient
accumulation uses one fixed cross-entropy objective with a quadratic
penalty. With a sufficiently small constant tail, every weight converges.
A bounded analytic lift of the exact RMS supplies the local energy
inequality. Approximate nuclear completion certifies strict logits at
finite weights. Zero sampled embeddings yield tied logits even at a
limit with uniquely identifying observations. With independent Gaussian
draws, small positive coupled decay gives strict limit predictions on
two entirely omitted binary words for the copy and constant tasks.
Large decay can send the full weights to zero with probability arbitrarily
close to one. Observation counts bound the worst-case success probability
for the unknown bilinear coupling at any model width. Prefix energy
bounds the nuclear norm of the trained centered phase table. For binary
modular sums the signed readout gaps, or the normalized hidden states,
give a bound at the rank-one target scale. For the binary copy and
constant examples the observed loss controls nuclear distance directly.

For any finite class, supports of competitors of equal or lower nuclear
cost form a hypergraph. Its transversal number is the minimum number
of observations for unique completion. The factor continuation eventually
decodes every word of every member. A full Argand search pairs native
momentum proposals with rational displacements and selects feasible weights
by the nuclear norm of their decoded phase tables. For a representable
class it stabilizes from each native draw at a positive logit margin.
An observation fiber of size m has optimal worst-case hidden success 1/m.
For an identifiable class, PAGE with coupled decay about an observation-only
strict reference generalizes above any prescribed confidence below one from
native draws, for every representable member and every sampling path.
Ordinary AdamW with decoupled decay has a dimension-independent bound on
each update. A positive summable continuation of the WSD prefix converges
in every weight for any minibatch stream. A sufficiently small continuation
preserves the strict decoded function of an identified checkpoint. -/
