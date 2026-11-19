module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Gap
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Witness
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Counterexample
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Examples

/-!
# Actual coefficient-error correction and minibatch descent failure

Source: `review/convex/moments.md`, CM105. Complete shared support bounds the
effect of coefficient error on CE gaps and actual first-order directions.
Nonzero complex examples inhabit strict descent despite inexact coefficients.
A finite feasible update lowers minibatch CE but raises the full training mean.
-/
