module

public import Transformer.Grokking.FlatSpectrum.Defs
public import Transformer.Grokking.FlatSpectrum.Basic
public import Transformer.Grokking.FlatSpectrum.Factorization
public import Transformer.Grokking.FlatSpectrum.Roots
public import Transformer.Grokking.FlatSpectrum.Kernels
public import Transformer.Grokking.FlatSpectrum.Unimodular
public import Transformer.Grokking.FlatSpectrum.Interpolation
public import Transformer.Grokking.FlatSpectrum.CubicInterpolant
public import Transformer.Grokking.FlatSpectrum.Blocks
public import Transformer.Grokking.FlatSpectrum.OneMissing
public import Transformer.Grokking.FlatSpectrum.FlatFill
public import Transformer.Grokking.FlatSpectrum.RankOneFill
public import Transformer.Grokking.FlatSpectrum.Decoding
public import Transformer.Grokking.FlatSpectrum.CubicFill

/-! The flat-spectrum note on `x³ + x y mod p`, `review/flat_spectrum.md`, with the complex
readout `ψ x = exp (2πi x / p)`. Lemma 1: the nuclear norm is the least balanced
factorization cost `½ (‖U‖² + ‖V‖²)`, the weight-decay penalty of a bilinear layer.
Proposition 2: `ψ (a + b)` has rank one and `‖K‖_S1 = p`; `ψ (a³ + a b)` has `K K* = p I`
and `‖K‖_S1 = p^{3/2}`; these are the extremes for unimodular entries, and relabeling keeps
the spectrum. Theorem A: an invertible `K` is the least-`‖·‖_S1` matrix agreeing with it on
`Ω` iff its polar factor vanishes off `Ω`; the polar factor of `ψ (a³ + a b)` is `K / √p`,
with no zero entry, so no proper `Ω` recovers that kernel. Corollary: with one cell missing,
the least-`‖·‖_S1` fill of a flat kernel (`K* K = N I`) is the opposite phase `−K_ij`, so for
`ψ (a³ + a b)` it decodes to `f + (p ± 1)/2` for odd `p` and never to `f`; the fill of the
rank-one `ψ (a + b)` is the truth. -/

@[expose] public section
