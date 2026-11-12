module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Blend
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Loss
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Consistency
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.PredictionHull
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Relabeling
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.RelabelingObstruction

/-!
# Shared class distributions and complex spectral bounds

Source: `review/flat_spectrum.md`, CM1–CM13. Multiple fixed complex code readings share the
same learned class probabilities. The domain and ordinary CE are convex, Fourier coordinates
preserve the full distribution, and a feasible exact observed fit certifies a CE minimum.
No universal identification of unseen labels follows; SB3 remains a counterexample.
-/

@[expose] public section
