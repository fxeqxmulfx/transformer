module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Execution
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Table
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Error
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Current
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Guarantees
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Witness
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Examples

/-!
# Complete retained coefficients with changing learned content

Source: `review/convex/moments.md`, CM106. Exact indexed refresh and transport
identities retain every state and signal contribution. Actual changed initial
states and transported operators bound state drift; both signal and state error
enter current-content CE and direction bounds. No unseen-answer claim follows.
-/
