module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Transport
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Obstruction
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PathBounds
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Interaction

/-!
# Nonlinear residual depth with convex joint path training

Source: `review/convex_depth.md`, CD1–CD7. Joint weights over complete nonlinear
paths retain interactions through ordinary residual or phase-Fourier blocks, while
ordinary CE is convex in those weights. A parity witness separates correlated and
independent routing with identical marginals. The finite block library and decoder
are fixed; dense coordinates grow exponentially. A concrete Fourier skip has unit
singular values and supplies additive bounds for bounded nonlinear branches.
The additive-injection baseline and a strict residual/mHC nonconvexity obstruction
record why this lifting is needed. No convexity of raw transformer parameters is claimed.
-/
