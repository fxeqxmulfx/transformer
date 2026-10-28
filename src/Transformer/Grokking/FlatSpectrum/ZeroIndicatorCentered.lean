module

public import Transformer.Grokking.FlatSpectrum.GramBlock
public import Transformer.Grokking.FlatSpectrum.MulCharSpectrum
public import Transformer.Grokking.FlatSpectrum.ZeroIndicatorShiftGram

/-!
# The centered indicator of answer `0`

Source: `review/flat_spectrum.md`, Torch check: "31.5 at p=97, 60.0 at p=199 | These are the
centered values", and "r_eff = 0.359p, 0.345p, 0.325p, 0.302p at p = 47, 61, 97, 199".
Centering subtracts the mean `μ = (2p − 1)/p²` of `[a³ + a b = 0]` over the `p²` pairs
(`sum_centeredZeroIndicatorKernel`).

For every shift `μ`, `K_μᴴ K_μ = 2 E + U M Uᴴ` (`conjTranspose_mul_shiftedZeroIndicatorKernel`)
with `Uᴴ U = diag d`, `d = ((p − 1)/2, (p + 1)/2)`, `tr E = (p − 3)/2`,
`tr (M diag d) = p + 2 − 2(2p − 1)μ + p²μ²` and `|det M| d₀ d₁ = (1 − μ)²(p + 1)`. So
`nuclearNorm_of_gram_eq_proj_add_diagonal` gives
`‖K_μ‖_S1 = (p − 3)/√2 + √(p + 2 − 2(2p − 1)μ + p²μ² + 2|1 − μ|√(p + 1))`, and
`‖K_μ‖²_S2 = 2p − 1 − 2(2p − 1)μ + p²μ²`. At `μ = 0` the square root is `√(p + 1) + 1`, and this
is the uncentered `nuclearNorm_zeroIndicatorKernel`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {p : ℕ} [Fact p.Prime]

/-- `tr (M diag d) = p + 2 − 2(2p − 1)μ + p²μ²` for `d = ((p − 1)/2, (p + 1)/2)`. -/
lemma trace_shiftedGramBlock_mul_diagonal (μ : ℝ) :
    (shiftedGramBlock p μ *
        diagonal fun i => ((![((p : ℝ) - 1) / 2, ((p : ℝ) + 1) / 2] i : ℝ) : ℂ)).trace =
      ((p + 2 - 2 * (2 * p - 1) * μ + p ^ 2 * μ ^ 2 : ℝ) : ℂ) := by
  have hp1 : (p : ℝ) - 1 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast (Fact.out : p.Prime).ne_one)
  rw [trace_fin_two, mul_diagonal, mul_diagonal]
  simp only [shiftedGramBlock, of_apply, cons_val', cons_val_zero, cons_val_one, empty_val',
    cons_val_fin_one, Fin.isValue]
  rw [← Complex.ofReal_mul, ← Complex.ofReal_mul, ← Complex.ofReal_add]
  congr 1
  field_simp
  ring

/-- `det M = 4 (1 − μ)²/(p − 1)`. -/
lemma det_shiftedGramBlock (μ : ℝ) :
    (shiftedGramBlock p μ).det = ((4 * (1 - μ) ^ 2 / (p - 1) : ℝ) : ℂ) := by
  have hp1 : (p : ℝ) - 1 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast (Fact.out : p.Prime).ne_one)
  rw [shiftedGramBlock, det_fin_two_of, ← Complex.ofReal_mul, ← Complex.ofReal_mul,
    ← Complex.ofReal_sub]
  congr 1
  field_simp
  ring

/-- `K_μᴴ K_μ = (√2)² E + U M Uᴴ`, the form of `nuclearNorm_of_gram_eq_proj_add_diagonal`. -/
lemma conjTranspose_mul_shiftedZeroIndicatorKernel_eq_sq (hp : p ≠ 2) (μ : ℝ) :
    (shiftedZeroIndicatorKernel p μ)ᴴ * shiftedZeroIndicatorKernel p μ =
      ((√2 ^ 2 : ℝ) : ℂ) • negSqProjection p +
        negSqFrame p * shiftedGramBlock p μ * (negSqFrame p)ᴴ := by
  rw [Real.sq_sqrt zero_le_two, Complex.ofReal_ofNat,
    conjTranspose_mul_shiftedZeroIndicatorKernel hp]

example := @conjTranspose_mul_shiftedZeroIndicatorKernel_eq_sq 5 ⟨Nat.prime_five⟩ (by decide) 0

/-- **The shifted indicator of answer `0`**, `review/flat_spectrum.md`, Torch check: for every
odd prime `p` and every shift `μ`,
`‖K_μ‖_S1 = (p − 3)/√2 + √(p + 2 − 2(2p − 1)μ + p²μ² + 2|1 − μ|√(p + 1))`. -/
theorem nuclearNorm_shiftedZeroIndicatorKernel (hp : p ≠ 2) (μ : ℝ) :
    nuclearNorm (shiftedZeroIndicatorKernel p μ) = (p - 3) / √2 +
      √(p + 2 - 2 * (2 * p - 1) * μ + p ^ 2 * μ ^ 2 + 2 * |1 - μ| * √(p + 1)) := by
  have hp3 := three_le_of_ne_two hp
  have hd (i : Fin 2) : 0 < ![((p : ℝ) - 1) / 2, ((p : ℝ) + 1) / 2] i := by
    fin_cases i
    · simp only [Fin.zero_eta, Fin.isValue, cons_val_zero]
      linarith
    · simp only [Fin.mk_one, Fin.isValue, cons_val_one, cons_val_fin_one]
      linarith
  have h2 : √2 * (((p : ℝ) - 3) / 2) = (p - 3) / √2 := by
    rw [eq_div_iff (by positivity)]
    linear_combination ((p : ℝ) - 3) / 2 * Real.mul_self_sqrt (zero_le_two (α := ℝ))
  have hdet : 4 * (1 - μ) ^ 2 / ((p : ℝ) - 1) * (((p : ℝ) - 1) / 2 * (((p : ℝ) + 1) / 2)) =
      (1 - μ) ^ 2 * (p + 1) := by
    have : (p : ℝ) - 1 ≠ 0 := (show (0 : ℝ) < p - 1 by linarith).ne'
    field_simp
    ring
  rw [nuclearNorm_of_gram_eq_proj_add_diagonal (Real.sqrt_nonneg 2) hd
      negSqProjection_conjTranspose (negSqProjection_mul_self hp)
      (negSqFrame_conjTranspose_mul hp) (negSqProjection_mul_negSqFrame hp)
      (conjTranspose_mul_shiftedZeroIndicatorKernel_eq_sq hp μ),
    trace_negSqProjection hp, trace_shiftedGramBlock_mul_diagonal, det_shiftedGramBlock,
    Complex.ofReal_re, Complex.ofReal_re,
    Complex.norm_of_nonneg (div_nonneg (by positivity) (by linarith))]
  simp only [cons_val_zero, cons_val_one, cons_val_fin_one]
  rw [h2, hdet, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq_eq_abs]
  congr 2
  ring

example := @nuclearNorm_shiftedZeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide) 0

/-- `‖K_μ‖²_S2 = 2p − 1 − 2(2p − 1)μ + p²μ²`, `review/flat_spectrum.md`, Torch check. -/
theorem frobeniusSq_shiftedZeroIndicatorKernel (hp : p ≠ 2) (μ : ℝ) :
    frobeniusSq (shiftedZeroIndicatorKernel p μ) =
      2 * p - 1 - 2 * (2 * p - 1) * μ + p ^ 2 * μ ^ 2 := by
  rw [frobeniusSq_of_gram_eq_add (conjTranspose_mul_shiftedZeroIndicatorKernel_eq_sq hp μ),
    negSqFrame_conjTranspose_mul hp, trace_negSqProjection hp,
    trace_shiftedGramBlock_mul_diagonal, Complex.ofReal_re, Complex.ofReal_re,
    Real.sq_sqrt zero_le_two]
  ring

example := @frobeniusSq_shiftedZeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide) 0

/-- `r_eff (K_μ) = ‖K_μ‖²_S1 / ‖K_μ‖²_S2`, `review/flat_spectrum.md`, Torch check. -/
theorem effectiveRank_shiftedZeroIndicatorKernel (hp : p ≠ 2) (μ : ℝ) :
    effectiveRank (shiftedZeroIndicatorKernel p μ) = ((p - 3) / √2 +
      √(p + 2 - 2 * (2 * p - 1) * μ + p ^ 2 * μ ^ 2 + 2 * |1 - μ| * √(p + 1))) ^ 2 /
        (2 * p - 1 - 2 * (2 * p - 1) * μ + p ^ 2 * μ ^ 2) := by
  rw [effectiveRank, div_pow, Real.sq_sqrt (frobeniusSq_nonneg _),
    nuclearNorm_shiftedZeroIndicatorKernel hp, frobeniusSq_shiftedZeroIndicatorKernel hp]

example := @effectiveRank_shiftedZeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide) 0

/-- **The centered indicator of answer `0`**, `review/flat_spectrum.md`, Torch check: "31.5 at
p=97, 60.0 at p=199 | These are the centered values". At `μ = (2p − 1)/p²`, for every odd prime
`p`, `r_eff = (p(p − 3)/√2 + √(p³ − 2p² + 4p − 1 + 2(p − 1)²√(p + 1)))² / ((2p − 1)(p − 1)²)`. -/
theorem effectiveRank_centeredZeroIndicatorKernel (hp : p ≠ 2) :
    effectiveRank (centeredZeroIndicatorKernel p) =
      (p * (p - 3) / √2 + √(p ^ 3 - 2 * p ^ 2 + 4 * p - 1 + 2 * (p - 1) ^ 2 * √(p + 1))) ^ 2 /
        ((2 * p - 1) * (p - 1) ^ 2) := by
  have hp3 := three_le_of_ne_two hp
  have hp0 : (p : ℝ) ≠ 0 := (show (0 : ℝ) < p by linarith).ne'
  have hp1 : (p : ℝ) - 1 ≠ 0 := (show (0 : ℝ) < p - 1 by linarith).ne'
  have hp2 : 2 * (p : ℝ) - 1 ≠ 0 := (show (0 : ℝ) < 2 * p - 1 by linarith).ne'
  have hs2 : √2 ≠ 0 := by positivity
  have h1 : 1 - (2 * (p : ℝ) - 1) / (p : ℝ) ^ 2 = ((p : ℝ) - 1) ^ 2 / (p : ℝ) ^ 2 := by
    field_simp
    ring
  have habs : |1 - (2 * (p : ℝ) - 1) / (p : ℝ) ^ 2| = ((p : ℝ) - 1) ^ 2 / (p : ℝ) ^ 2 := by
    rw [h1]
    exact abs_of_nonneg (by positivity)
  have hN : (p : ℝ) + 2 - 2 * (2 * p - 1) * ((2 * p - 1) / p ^ 2) +
      p ^ 2 * ((2 * p - 1) / p ^ 2) ^ 2 + 2 * ((p - 1) ^ 2 / p ^ 2) * √(p + 1) =
      (p ^ 3 - 2 * p ^ 2 + 4 * p - 1 + 2 * (p - 1) ^ 2 * √(p + 1)) / p ^ 2 := by
    field_simp
    ring
  have hF : 2 * (p : ℝ) - 1 - 2 * (2 * p - 1) * ((2 * p - 1) / p ^ 2) +
      p ^ 2 * ((2 * p - 1) / p ^ 2) ^ 2 = (2 * p - 1) * (p - 1) ^ 2 / p ^ 2 := by
    field_simp
    ring
  rw [centeredZeroIndicatorKernel, effectiveRank_shiftedZeroIndicatorKernel hp, habs, hN, hF,
    Real.sqrt_div' _ (by positivity), Real.sqrt_sq (by positivity)]
  field_simp

example := @effectiveRank_centeredZeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide)

end Transformer.Grokking.FlatSpectrum
