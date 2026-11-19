module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Table
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Coefficient
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Gap
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Witness

/-!
# Complete training-only conditional readout optimality

Source: `review/convex/moments.md`, CM103. Exact complex linear minimization,
actual full-class indexed CE coefficients and a complete conditional gap.
The theorem does not certify floating-point execution, factor optimization,
unseen-answer identification or raw-label accuracy.
-/
