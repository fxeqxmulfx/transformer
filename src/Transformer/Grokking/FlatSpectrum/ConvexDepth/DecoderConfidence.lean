module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.DecoderConfidence.Batch

/-!
# Confidence for declared categorical predictions through actual complex heads

Source: `review/convex/moments.md`, CM44. A fixed state map and raw decoder
admit an actual conjugate head with an explicit coefficient budget. Ordinary
CE is convex in the whole fixed-feature head. Its initializer's batch CE is
the correct-margin loss plus margin times the actual declared-label error rate.
-/
