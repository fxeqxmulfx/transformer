module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.WeightedObservations.Head

/-!
# Exact grouping of repeated observations

Source: `review/flat_spectrum.md`, CM60. Original record multiplicities preserve
the actual complete-head ordinary cross-entropy and its derivatives. Each
minibatch uses its own counts and denominator. Equal observed constraints can
also be grouped, but noisy labels remain observations and do not become clean
identifying anchors. The joint CM57 domain and its convexity remain explicit.
-/
