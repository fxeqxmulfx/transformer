module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction.Witness

/-!
# Constant-amplitude extraction inside a separated Fourier subspace

Source: `review/flat_spectrum.md`, CM20. Exact amplitude identifies one mode,
centering excludes the constant mode, and amplitude error bounds cross-mode
products. The subspace must be exact and have separated frequency differences;
the two-state counterexample shows why amplitude alone is insufficient.
The unit-amplitude coefficient set is nonconvex, so it cannot be imposed
directly as a convex constraint on those coefficients.
-/

@[expose] public section
