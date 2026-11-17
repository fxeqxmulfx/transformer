module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Normalization
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Execution
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Support
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Loss
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Gap
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Witness
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Counterexample
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Examples
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate

/-!
# Complete weighted readouts with jointly redistributable component capacity

Source: `review/flat_spectrum.md`, CM104. A convex shared budget embeds exactly
in the original learned family. Complete actual CE coefficients give an explicit
support minimizer and enlarged conditional gap. A nonzero complex counterexample
shows why a zero fixed-weight gap does not cover capacity redistribution.
-/
