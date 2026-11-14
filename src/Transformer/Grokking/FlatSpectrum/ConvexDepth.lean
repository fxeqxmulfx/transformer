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
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.CoverageExtension
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PartialFallback
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextInterpolation
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextScale
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.DecoderConfidence
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.GatedContext
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.RawRegions
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.SignedContext

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
CM39 conservatively extends coverage: a supplemental complex head acts only
where the prior refuses and the supplement accepts. All other prior logits
and original covered-batch CE derivatives are preserved exactly.
CM41 treats a saved predecessor with a restricted input domain: zero-extended
parent features are used only where the parent is available and the candidate
refuses. The candidate remains elsewhere, with the same convex CE and charged
budgets and exact supported-batch objective and derivative identities.
CM42 uses fixed stochastic context weights to interpolate actual complex
source features. It preserves observed batches, convex raw-label CE and the
coefficient bound, while local raw-label information remains an explicit
premise distinct from flatness and decoded correctness.
CM43 permits fixed confidence shrinking on context features. It preserves
the feature bound and raw rankings, and leaves original batch objectives and
derivatives intact when observed scales equal one. Convexity in a fixed
head's scalar confidence is separate from convexity in a scaled fixed-feature head.
CM44 realizes a fixed input-state map and many-to-one declared raw decoder
with a complete bounded complex tensor. Its exact batch CE separates the
correct-margin loss from margin times empirical raw error; no unknown target
is inferred by increasing confidence.
CM45 learns simplex mixtures of fixed context maps with the complete head
frozen. Actual products retain the feature and coefficient bounds, with convex
ordinary diagnostic CE and explicit first-order and quadratic controls. When
all maps preserve observed rows, the original CE cannot select a mixture.
CM46 uses fixed input gates and an anchor to learn local simplex mixtures.
Their actual products keep the source bound and convex ordinary CE in all
local coefficients, with explicit slopes and original-batch identities.
A nonzero Fourier example separates local from shared-mixture capacity;
the fixed gates still do not certify any unseen raw target.
CM47 adds fixed raw-margin halfspaces to the actual CM46 coefficient domain.
It preserves convex ordinary CE, admits a maximizing reference and retains
the first maximizing class under positive margin retention. Confidence may
change, but a wrong reference label remains wrong; zero retention can flip
a deterministic decision through a tie.
CM48 adds an explicit convex raw-class region hypothesis and observable
same-label input-hull certificates. They identify covered missing raw labels
and transfer actual complex-head margins and ordinary CE. The original feature
and coefficient bounds remain; uncovered queries can still be ambiguous.
CM49 excludes candidate raw labels using conflicts between augmented and
observed class hulls. With a separate candidate-pool or correct decoder-fiber
premise, this identifies some exterior queries. Positive source mixtures
retain complete-head margins, ordinary CE and bounds; counterexamples retain
the limits of pool closure and ambiguous gaps.
CM51 permits fixed signed contexts, with absolute-mass normalization preserving
the source bound and complete-head convex CE. Query error and strict margins
require separate reference-score residual bounds; an explicit alternative
target agrees on observations and decoded answers while disagreeing raw.
The additive-injection baseline and a strict residual/mHC nonconvexity obstruction
record why this lifting is needed. No convexity of raw transformer parameters is claimed.
-/
