module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.NormComparison
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Recovery
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Optimality
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Hull

/-!
# Shared complex moments with singular-value budgets

Source: `review/flat_spectrum.md`, CM5–CM7. Convex bounds on the largest singular value
and their sum accommodate different spectral shapes. A concrete rank-one recovery proof
starts with ordinary empirical CE and does not assume correct missing labels or a trained
factorization. Arbitrary table recovery and numerical solver convergence are not asserted.
-/

@[expose] public section
