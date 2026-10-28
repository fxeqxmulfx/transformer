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
public import Transformer.Grokking.FlatSpectrum.RealKernels
public import Transformer.Grokking.FlatSpectrum.PartialIsometry
public import Transformer.Grokking.FlatSpectrum.RealSpectrum
public import Transformer.Grokking.FlatSpectrum.GaussSums
public import Transformer.Grokking.FlatSpectrum.CharacterBases
public import Transformer.Grokking.FlatSpectrum.ShiftKernel
public import Transformer.Grokking.FlatSpectrum.JacobiSums
public import Transformer.Grokking.FlatSpectrum.MulIntermediate
public import Transformer.Grokking.FlatSpectrum.QuotientRoute
public import Transformer.Grokking.FlatSpectrum.ThreeEigenvalues
public import Transformer.Grokking.FlatSpectrum.MulCharKernel
public import Transformer.Grokking.FlatSpectrum.MulCharSpectrum
public import Transformer.Grokking.FlatSpectrum.NumericalChecks
public import Transformer.Grokking.FlatSpectrum.PolynomialFibers
public import Transformer.Grokking.FlatSpectrum.PolynomialMulChar
public import Transformer.Grokking.FlatSpectrum.PolynomialLegendre

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
norm times `O(p^{-1/2})`. Each step of `v = a b`, `f = a³ + v` is rank 1 in its own basis,
`χ(a b) = χ(a) χ(b)` and `ψ_r(a³ + v) = ψ_r(a³) ψ_r(v)`, and flat in the other; the inner
products of the two bases are Gauss sums of modulus `√p`, and for `p ≡ 1 mod 3`,
`S_r = g(χ, ψ_r) + g(χ², ψ_r)` with `χ` of order 3. For a nontrivial multiplicative character
`χ` and `c ≠ 0`, Jacobi sums give Weil's bound `|∑_a χ(c − a³)| ≤ 2 √p`, so
`‖E[χ(a b) | a³ + a b]‖²` has the same bound `9/(2p) + 8/p^{3/2}`. Over `a ≠ 0`,
`1 + b/a² = f/a³`, and `E[χ(1 + b/a²) | f]` is `χ(f)` if `χ³ = 1` and `0` otherwise. For
nontrivial `χ` and odd `p`, the kernel `[χ(a³ + a b)]` has rank `(p − 1)/2` and
`r_eff = (√(p + 1) + (p − 3)/2 · √(2p))²/(p − 1)²`, between `47.9` and `48` at `p = 97`. For any
function `P` of `a`, the fibers of `P(a) + a b` give `‖E[ψ_r(a b) | f]‖²` exactly in terms of
`T = ∑_{a ≠ 0} ψ(−r P(a))`, and a bound on `∑_x ψ(r P(x))` or on `∑_a χ(c − P(a))` bounds the
additive or the multiplicative readout. For `P` of odd degree `m`, `p ≥ 64` and `8 m ≤ p`,
Stepanov's bound (`Transformer.Weil`) gives `|∑_a χ₂(c − P(a))| ≤ 7 m √p`, so the Legendre readout
`‖E[χ₂(a b) | P(a) + a b]‖²` is at most `(7 m √p + 1)²/p² + (p − 1)²/(p² (2p − 1))`. Of the
numerical checks, the bound at `p = 97` is below `0.053`, `‖E[ψ_r(a + b) | (a + b)²]‖² = ½`, and the
counting rule `r (2p − r) ≤ ρ p²` iff `r ≤ (1 − √(1 − ρ)) p` gives `28.4` at `ρ = ½`, `p = 97`.
Torch check: every minimum-nuclear-norm completion is zero on a row whose seen entries are zero, so
a hidden 1 of the one-hot kernel `[a³ + a b = c]` in a row `a ≠ 0` is never recovered, from any
training set. The real readouts `cos(2πr f/p)` and `sin(2πr f/p)` of `f = a³ + a b` have rows `a`
and `−a` equal and opposite, so both are singular and Theorem A does not apply; for odd prime `p`
and `r ≠ 0` they have rank `(p + 1)/2` and `(p − 1)/2`, and every nonzero singular value is `√p`. -/

@[expose] public section
