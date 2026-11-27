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

/-! Finite recovery from observed labeled words. The full single-stream
Argand forward pass and Flat Spectrum share the same targets and observation
masks. The examples compare exact observation thresholds, embedding and FFN
widths, and repeated product layers. Series windows can use independently
compiled difference codecs, or share a grid with axial LeRoPE and a time mask.
The training examples cover modular sums, bilinear couplings, encoded and
additive periodic series, and compiled finite-state laws. The native Argand
trajectory starts from the actual initialization and uses the full clipped,
bias-corrected AMSGradW equations with a scheduled minibatch stream.
At a positive rate floor convergence is equivalent to a common gradient
limit. Shared interpolation alone does not imply that limit. -/
