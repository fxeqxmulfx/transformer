module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Transport
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Obstruction
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PathBounds
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Interaction
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.JointHead
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PrefixFeatures
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.FourierCode
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientWitness
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RefinedBox
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RefinedCE
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback

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
CFt5 in `review/flat_spectrum.md` binds actual complex categorical features to
this convex head, proves the conjugate initializer's exact logits and budget,
and supplies flat Fourier query blocks for arbitrary nonzero state counts.
CM31 refines observed inputs to fit raw aliases of a learned quotient, proves
the actual complex-head loss and budget, and gives convex affine constraints
that retain declared defaults at unseen inputs. Its four-class witness keeps
the distinction between decoded correctness and unidentified raw aliases.
CM32 realizes the same states with latent-fiber Fourier blocks and proves
feasibility of clipped table updates, including exact retention of defaults.
The arbitrary-table coefficient bound covers every resulting iterate.
CM33 derives ordinary CE through the actual complex head and proves that
projected nonnegative full/minibatch steps preserve observed margins and
cannot increase observed row CE while retaining the same budget and defaults.
CM38 freezes a parent head outside a candidate's fixed support, deriving actual
complex logits, charged budgets, convex ordinary CE and equality of covered
batch derivatives. A supported unseen alias can still regress despite flat
component codes; support is not an unknown-target correctness certificate.
The additive-injection baseline and a strict residual/mHC nonconvexity obstruction
record why this lifting is needed. No convexity of raw transformer parameters is claimed.
-/
