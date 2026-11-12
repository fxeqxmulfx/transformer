module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.NormComparison
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Recovery
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Optimality
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Hull
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Intermediate
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Completion
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Quotient
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Sharpen

/-!
# Shared complex moments with singular-value budgets

Source: `review/flat_spectrum.md`, CM5–CM7 and CM9–CM11. Convex bounds on the largest singular value
and their sum accommodate different spectral shapes. A concrete rank-one recovery proof
starts with ordinary empirical CE and does not assume correct missing labels or a trained
factorization. The two endpoint budgets miss an explicit intermediate-rank moment.
An uncertain completion can still minimize CE after that capacity gap is repaired;
fixed correct sharing recovers the same witness's missing columns.
Decoded endpoints can be sharpened and checked against both spectral budgets,
with an ordinary-CE excess bound from their common uniform mixing coefficient.
Arbitrary table recovery and numerical solver convergence are not asserted.
-/

@[expose] public section
