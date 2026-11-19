module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial.Gradient
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Radial.Failure

/-!
# Radial feasibility does not imply ordinary-CE optimization

Source: `review/convex/moments.md`, CM101. An actual nonconstant bounded recurrent
example has fixed radial gradient proposals and strictly better feasible CE.
-/
