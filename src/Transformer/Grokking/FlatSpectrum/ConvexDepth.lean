module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Transport
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Obstruction
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PathBounds
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Interaction
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.JointHead
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PrefixFeatures

/-!
# Nonlinear residual depth with convex joint path training

Source: `review/convex_depth.md`, CD1–CD9. Joint weights over complete nonlinear
paths retain interactions through ordinary residual or phase-Fourier blocks, while
ordinary CE is convex in those weights. A parity witness separates correlated and
independent routing with identical marginals. The finite block library is fixed;
CD5 also fixes the decoder, and dense coordinates grow exponentially. A concrete Fourier skip has unit
singular values and supplies additive bounds for bounded nonlinear branches.
An extended learned output tensor uses those full path features under convex
absolute-sum budgets, without a product of separately learned routers and decoders.
Retaining features from every shorter path embeds each shallower head at the
same budget, while making the geometric storage cost explicit.
The additive-injection baseline and a strict residual/mHC nonconvexity obstruction
record why this lifting is needed. No convexity of raw transformer parameters is claimed.
-/
