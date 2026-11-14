module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextScale.Batch

/-!
# Fixed confidence scaling for context-interpolated complex heads

Source: `review/flat_spectrum.md`, CM43. Fixed shrinking preserves feature
bounds and positive shrinking preserves raw rankings. Ordinary CE remains
convex in the full head tensor; observed scale one preserves original batch
objectives and derivatives. Scalar CE convexity for a fixed logit vector is
a separate statement, not joint convexity in the scale and head.
-/
