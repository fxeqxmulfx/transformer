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
public import Transformer.Grokking.FlatSpectrum.CubicSums
public import Transformer.Grokking.FlatSpectrum.CubicBound
public import Transformer.Grokking.FlatSpectrum.Conditional
public import Transformer.Grokking.FlatSpectrum.Intermediate
public import Transformer.Grokking.FlatSpectrum.OneHotFill

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
rank-one `ψ (a + b)` is the truth. Theorem B: for prime `p` and `r ≠ 0`, the cubic sum
`S_r = ∑_x ψ(r x³)` has `|S_r| ≤ 2 √p`, and `S_r = 0` when `p ≡ 2 mod 3`; over uniform
`(a, b)`, `‖E[ψ_r(a b) | a³ + a b]‖²` is the source's exact expression in `S_r`, at most
`9/(2p) + 8/p^{3/2}`, so a readout of the label correlates with `ψ_r(a b)` by at most its
norm times `O(p^{-1/2})`. Torch check: every minimum-nuclear-norm completion is zero on a row
whose seen entries are zero, so a hidden 1 of the one-hot kernel `[a³ + a b = c]` in a row
`a ≠ 0` is never recovered, from any training set. -/

@[expose] public section
