module

public import Transformer.Grokking.FlatSpectrum
public import Transformer.Grokking.FiniteRecovery.ArgandSamples
public import Transformer.Grokking.FiniteRecovery.Capacity
public import Transformer.Grokking.FiniteRecovery.SeriesAttention
public import Transformer.Grokking.FiniteRecovery.SeriesGrid
public import Transformer.Grokking.FiniteRecovery.SeriesStability
public import Transformer.Grokking.FiniteRecovery.PeriodicForecast
public import Transformer.Grokking.FiniteRecovery.TransitionTraining

/-! Finite recovery from observed labeled words. The full single-stream
Argand forward pass and Flat Spectrum share the same targets and observation
masks. The examples compare exact observation thresholds, embedding and FFN
widths, and repeated product layers. Series windows can use independently
compiled difference codecs, or share a grid with axial LeRoPE and a time mask.
The forecast examples cover additive periodic series and compiled finite-state
laws with a convergent scalar training run in ordinary Argand. -/
