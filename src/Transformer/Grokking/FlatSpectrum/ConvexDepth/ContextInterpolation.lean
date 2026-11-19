module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation.Witness

/-!
# Fixed context interpolation for ordinary raw-label CE

Source: `review/convex/moments.md`, CM42. A frozen stochastic context map acts
on actual complex features with the original coefficient budget. The raw CE
is convex, obeys Jensen's inequality and retains original observed-batch
derivatives. Local raw-label information is an explicit additional premise;
flatness and observed fit do not remove the alias counterexample.
-/
