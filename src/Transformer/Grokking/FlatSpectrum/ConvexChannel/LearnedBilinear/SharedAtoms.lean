module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Domain
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Model
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Witness
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Queries
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Scaling
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.SharedAtoms.Support

/-!
# Bounded complex components with a shared learned output factor

Source: `review/convex/moments.md`, CM92. The complete joint domain is convex,
has finite executable learned-factor representations, and supports feasible
forward-only bias queries. Original-object inputs, outputs and ordinary CE
remain unchanged. Component discovery and generalization are separate open
algorithmic problems; convexity in independent factor coordinates is not claimed.
-/
