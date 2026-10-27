module

public import Transformer.Grokking.FlatSpectrum.Defs
public import Transformer.Grokking.FlatSpectrum.Basic
public import Transformer.Grokking.FlatSpectrum.Factorization
public import Transformer.Grokking.FlatSpectrum.Roots
public import Transformer.Grokking.FlatSpectrum.Kernels
public import Transformer.Grokking.FlatSpectrum.Unimodular
public import Transformer.Grokking.FlatSpectrum.Interpolation
public import Transformer.Grokking.FlatSpectrum.CubicInterpolant

/-! The flat-spectrum note on `x³ + x y mod p`, `review/flat_spectrum.md`, with the complex
readout `ψ x = exp (2πi x / p)`. Lemma 1: the nuclear norm is the least balanced
factorization cost `½ (‖U‖² + ‖V‖²)`, the weight-decay penalty of a bilinear layer.
Proposition 2: `ψ (a + b)` has rank one and `‖K‖_S1 = p`; `ψ (a³ + a b)` has `K K* = p I`
and `‖K‖_S1 = p^{3/2}`; these are the extremes for unimodular entries, and relabeling keeps
the spectrum. Theorem A: an invertible `K` is the least-`‖·‖_S1` matrix agreeing with it on
`Ω` iff its polar factor vanishes off `Ω`; the polar factor of `ψ (a³ + a b)` is `K / √p`,
with no zero entry, so no proper `Ω` recovers that kernel. -/

@[expose] public section
