module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.Mixing
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.Projection
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.Domain
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.Queries
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.OutputQueries

/-!
# Add norm-preserving learned mixing while retaining the old family

Source: `review/convex/moments.md`, CM107–109. Complete complex Hilbert coordinates,
an exact isometric mixing witness and contraction under the old row projection,
plus an enlarged bounded functional convex domain containing both complete
learned branches. An explicit attained matrix norm gives a feasible numerical
normalization preserving every unit point. Exact finite single-class CE queries
can reuse their complete partition. This proves neither factor discovery nor
unseen answers.
-/
