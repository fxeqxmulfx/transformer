module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Blend
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Loss
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Consistency
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.PredictionHull
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Relabeling
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.RelabelingObstruction
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFibers
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ModularCoding
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery

/-!
# Shared class distributions and complex spectral bounds

Source: `review/flat_spectrum.md`, CM1–CM28. Multiple fixed complex code readings share the
same learned class probabilities. The domain and ordinary CE are convex, Fourier coordinates
preserve the full distribution, and a feasible exact observed fit certifies a CE minimum.
No universal identification of unseen labels follows; SB3 remains a counterexample.
CM17 (`CodingDiscovery`): for a table with an unknown bijective phase coding, the coding is
found from the training rectangles and the completion is the table.
CM18 (`DecodedRecovery`): behind a fixed decoder that merges opposite latent states, a
training-only certificate identifies unseen decoded labels; the spectral relaxation alone
does not.
CM19 (`DecodedPropagation`): finite support pruning handles arbitrary fixed decoder fibers
and finite decoder banks. Singleton unions certify labels and give convex spectral
restrictions without losing the realizable CE optimum; latent states may remain ambiguous.
CM20 (`FourierExtraction`): constant amplitude selects a single mode in an exact
Fourier subspace with separated differences; amplitude error bounds cross-mode products.
This is a conditional extraction result, not discovery of the subspace from partial data.
CM21 (`DensityExtraction`): a convex PSD matrix lift admits all pure-mode mixtures;
two observable linear objectives select a single mode in an exact rotated subspace.
The exact-subspace assumption remains explicit and is not a missing-label recovery claim.
CM22 (`DensityBounds`): a convex interval relaxation tolerates diagonal and feature errors;
an explicit objective gap controls the remaining mixed mass. Subspace discovery and
missing-label identification are still separate requirements.
CM23 (`ObservedRefinement`): verified assignment duals and decoder counts certify
local observed-CE descent while preserving the decoded spectral hull. Discovery
and unseen-label correctness still require separate evidence.
CM24 (`BalancedFeatures`): measured row normalization admits an explicit uniform
density; completing unobserved phase axes preserves observed CE and spectral
budgets. Feasibility does not establish phase recovery or unseen labels.
CM25 (`TaggedPhases`): a larger latent domain permits constant categorical input
strips alongside nonconstant interiors and preserves prior phase vertices.
Observable agreement certifies CE optimality, not discovery of unseen labels.
CM26 (`BalancedFibers`): injective phase coordinates and complete same-label
groups give exact additive fiber-sum equations. Completeness and common fiber
mass remain explicit hypotheses; observed multiplicities do not certify them.
CM27 (`ModularCoding`): exact residual equations characterize modular lifts;
checked Bezout recombination and unit gauges support composite latent domains.
Counterexamples retain failed lifts and noncoprime product obstructions.
CM28 (`CompositeDiscovery`): a cyclic joint observed module identifies a table
with unknown bijective class coding over composite domains. Repeated phases
are allowed; genuinely missing axes cannot be silently pinned for this check.
-/

@[expose] public section
