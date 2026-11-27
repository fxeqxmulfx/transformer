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
limit with uniquely identifying observations. -/
