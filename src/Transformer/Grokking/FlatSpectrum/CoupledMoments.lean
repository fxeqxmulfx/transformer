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

/-!
# Shared class distributions and complex spectral bounds

Source: `review/flat_spectrum.md`, CM1–CM19. Multiple fixed complex code readings share the
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
-/

@[expose] public section
