module

public import Transformer.Grokking.Convex.ConvexDepth.PackedRecords.Basic
public import Transformer.Grokking.Convex.ConvexDepth.PackedRecords.Observations
public import Transformer.Grokking.Convex.ConvexDepth.PackedRecords.Complex
public import Transformer.Grokking.Convex.ConvexDepth.PackedRecords.Witness

/-!
# Lossless packed observations for the complete convex complex decoder

Source: `review/convex/moments.md`, CM68. Bounded UInt16 words preserve every
program state and original raw label. Precomputed observations retain decoder
lookups, the complete candidate filter, full joint domains, complex products
and ordinary indexed CE, including duplicates and actual minibatch tails.
-/
