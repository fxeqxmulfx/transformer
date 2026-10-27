module

public import Transformer.Grokking.FlatSpectrum.Defs
public import Transformer.Grokking.FlatSpectrum.Basic
public import Transformer.Grokking.FlatSpectrum.Factorization
public import Transformer.Grokking.FlatSpectrum.Roots
public import Transformer.Grokking.FlatSpectrum.Kernels
public import Transformer.Grokking.FlatSpectrum.Unimodular

/-! Why `x³ + x y mod p` resists a minimum-norm learner, `review/flat_spectrum.md`: the
nuclear norm, its contraction duality and its balanced-factorization formula
(Lemma 1), the variational form of the weight-decay penalty on a bilinear layer; the
spectra of the task kernels (Proposition 2): `a + b` has rank one and `‖K‖_S1 = p`,
`a³ + a b` has `K K* = p I` and `‖K‖_S1 = p^{3/2}`, the extremes for unimodular entries,
and relabeling keeps the spectrum. -/

@[expose] public section
