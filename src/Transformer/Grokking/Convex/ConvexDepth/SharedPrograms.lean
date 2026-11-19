module

public import Transformer.Grokking.Convex.ConvexDepth.SharedPrograms.Basic
public import Transformer.Grokking.Convex.ConvexDepth.SharedPrograms.Learner
public import Transformer.Grokking.Convex.ConvexDepth.SharedPrograms.Transfer

/-!
# Shared input programs and identified complex copy predictions

Source: `review/convex/moments.md`, CM52. A fixed canonical copy head and a
finite input-program bank have a convex ordinary-CE mixture domain. An
executable training-only filter and input-only consensus identify accepted
raw answers under explicit bank realizability, including symbols absent from
training labels. The actual Fourier products and CE obey this identification;
flatness supplies only bounds, and ambiguous queries remain uncertified.
-/
