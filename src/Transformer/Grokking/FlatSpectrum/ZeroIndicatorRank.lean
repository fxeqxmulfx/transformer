module

public import Transformer.Grokking.FlatSpectrum.ZeroIndicatorSpectrum

/-!
# The indicator of answer `0`: rank, norms and effective rank

Source: `review/flat_spectrum.md`, Torch check: "These are the centered values. Uncentered, the
exact value is (√(p+1)+1+(p−3)/√2)²/(2p−1) = 31.01 and 59.53; the spectrum matches to 1e-14.
p/4+√(p/2) gives 31.21 and 59.72". By the multiplicities of `ZeroIndicatorSpectrum`, the
singular values of `K = [a³ + a b = 0]` are `√(p + 1)` and `1` once each, `√2` `(p − 3)/2` times
and `0` otherwise. So `rank K = (p + 1)/2`, `‖K‖_S1 = √(p + 1) + 1 + (p − 3)/2 · √2`,
`‖K‖²_S2 = 2p − 1`, and `r_eff K` is the torch check's exact value, for every odd prime `p`.
At `p = 97` it lies in `(31.014, 31.015)` and at `p = 199` in `(59.532, 59.533)`, so it rounds to
`31.01` and `59.53`; the note's `p/4 + √(p/2)` lies in `(31.214, 31.215)` and `(59.724, 59.725)`,
rounding to `31.21` and `59.72`. The centered values `31.5` and `60.0` are not formalized.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {p : ℕ} [Fact p.Prime]

/-- Every singular value of `K = [a³ + a b = 0]` is `0`, `1`, `√2` or `√(p + 1)`. -/
theorem singularValues_zeroIndicatorKernel_mem (hp : p ≠ 2) (i : ZMod p) :
    singularValues (zeroIndicatorKernel p) i = 0 ∨ singularValues (zeroIndicatorKernel p) i = 1 ∨
      singularValues (zeroIndicatorKernel p) i = √2 ∨
      singularValues (zeroIndicatorKernel p) i = √(p + 1) := by
  rw [singularValues_eq_sqrt_eigenvalues_mul_conjTranspose]
  rcases eigenvalues_zeroIndicatorKernel_mem hp i with h | h | h | h <;> rw [h]
  · exact Or.inl Real.sqrt_zero
  · exact Or.inr (Or.inl Real.sqrt_one)
  · exact Or.inr (Or.inr (Or.inl rfl))
  · exact Or.inr (Or.inr (Or.inr rfl))

example := @singularValues_zeroIndicatorKernel_mem 5 ⟨Nat.prime_five⟩ (by decide) 0

/-- `rank K = (p + 1)/2`. -/
theorem rank_zeroIndicatorKernel (hp : p ≠ 2) :
    ((zeroIndicatorKernel p).rank : ℝ) = (p + 1) / 2 := by
  have hp3 := three_le_of_ne_two hp
  obtain ⟨h₁, h₂, h₃⟩ := card_eigenvalues_zeroIndicatorKernel hp
  rw [← rank_self_mul_conjTranspose, rank_eq_of_mem_four _ one_ne_zero two_ne_zero
    (by positivity) (by norm_num) (fun h => by linarith) (fun h => by linarith)
    (eigenvalues_zeroIndicatorKernel_mem hp), h₁, h₂, h₃]
  ring

example := @rank_zeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide)

/-- `‖K‖_S1 = √(p + 1) + 1 + (p − 3)/2 · √2`. -/
theorem nuclearNorm_zeroIndicatorKernel (hp : p ≠ 2) :
    nuclearNorm (zeroIndicatorKernel p) = √(p + 1) + 1 + (p - 3) / 2 * √2 := by
  have hp3 := three_le_of_ne_two hp
  obtain ⟨h₁, h₂, h₃⟩ := card_eigenvalues_zeroIndicatorKernel hp
  simp only [nuclearNorm, singularValues_eq_sqrt_eigenvalues_mul_conjTranspose]
  rw [sum_comp_eigenvalues_of_mem_four _ (by norm_num) (fun h => by linarith)
    (fun h => by linarith) (eigenvalues_zeroIndicatorKernel_mem hp) Real.sqrt Real.sqrt_zero,
    h₁, h₂, h₃, Real.sqrt_one]
  ring

example := @nuclearNorm_zeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide)

/-- `‖K‖²_S2 = 2p − 1`. -/
theorem frobeniusSq_zeroIndicatorKernel (hp : p ≠ 2) :
    frobeniusSq (zeroIndicatorKernel p) = 2 * p - 1 := by
  rw [frobeniusSq_eq_re_trace, trace_mul_comm, trace_zeroIndicatorKernel_mul_conjTranspose hp,
    show 2 * (p : ℂ) - 1 = ((2 * (p : ℝ) - 1 : ℝ) : ℂ) by push_cast; ring, Complex.ofReal_re]

example := @frobeniusSq_zeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide)

/-- **The uncentered indicator of answer `0`**, `review/flat_spectrum.md`, Torch check: "the
exact value is (√(p+1)+1+(p−3)/√2)²/(2p−1)", for every odd prime `p`. -/
theorem effectiveRank_zeroIndicatorKernel (hp : p ≠ 2) :
    effectiveRank (zeroIndicatorKernel p) =
      (√(p + 1) + 1 + (p - 3) / √2) ^ 2 / (2 * p - 1) := by
  have hp3 := three_le_of_ne_two hp
  have h2 : ((p : ℝ) - 3) / 2 * √2 = (p - 3) / √2 := by
    rw [eq_div_iff (by positivity), mul_assoc, Real.mul_self_sqrt (by norm_num)]
    ring
  rw [effectiveRank, nuclearNorm_zeroIndicatorKernel hp, frobeniusSq_zeroIndicatorKernel hp,
    div_pow, Real.sq_sqrt (by linarith), h2]

example := @effectiveRank_zeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide)

/-- `review/flat_spectrum.md`, Torch check: the uncentered value is `31.01` at `p = 97`, as
`31.014 < r_eff < 31.015`. -/
theorem effectiveRank_zeroIndicatorKernel_97 [Fact (Nat.Prime 97)] :
    31.014 < effectiveRank (zeroIndicatorKernel 97) ∧
      effectiveRank (zeroIndicatorKernel 97) < 31.015 := by
  rw [effectiveRank, nuclearNorm_zeroIndicatorKernel (by norm_num),
    frobeniusSq_zeroIndicatorKernel (by norm_num), div_pow, Real.sq_sqrt (by norm_num),
    show ((97 : ℕ) : ℝ) = 97 by norm_num, show (97 : ℝ) + 1 = 98 by norm_num,
    show ((97 : ℝ) - 3) / 2 = 47 by norm_num, show (2 : ℝ) * 97 - 1 = 193 by norm_num]
  have hs1 : (9.899494 : ℝ) < √98 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : √98 < (9.899495 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have ht1 : (1.4142135 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have ht2 : √2 < (1.4142136 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hlo := mul_self_lt_mul_self (by norm_num : (0 : ℝ) ≤ 77.3675)
    (show (77.3675 : ℝ) < √98 + 1 + 47 * √2 by linarith)
  have hhi := mul_self_lt_mul_self (by positivity : (0 : ℝ) ≤ √98 + 1 + 47 * √2)
    (show √98 + 1 + 47 * √2 < (77.3676 : ℝ) by linarith)
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  · rw [div_lt_iff₀ (by norm_num)]
    nlinarith

example := @effectiveRank_zeroIndicatorKernel_97 ⟨by norm_num⟩

/-- `review/flat_spectrum.md`, Torch check: the uncentered value is `59.53` at `p = 199`, as
`59.532 < r_eff < 59.533`. -/
theorem effectiveRank_zeroIndicatorKernel_199 [Fact (Nat.Prime 199)] :
    59.532 < effectiveRank (zeroIndicatorKernel 199) ∧
      effectiveRank (zeroIndicatorKernel 199) < 59.533 := by
  rw [effectiveRank, nuclearNorm_zeroIndicatorKernel (by norm_num),
    frobeniusSq_zeroIndicatorKernel (by norm_num), div_pow, Real.sq_sqrt (by norm_num),
    show ((199 : ℕ) : ℝ) = 199 by norm_num, show (199 : ℝ) + 1 = 200 by norm_num,
    show ((199 : ℝ) - 3) / 2 = 98 by norm_num, show (2 : ℝ) * 199 - 1 = 397 by norm_num]
  have hs1 : (14.142135 : ℝ) < √200 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : √200 < (14.142136 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have ht1 : (1.4142135 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have ht2 : √2 < (1.4142136 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hlo := mul_self_lt_mul_self (by norm_num : (0 : ℝ) ≤ 153.735)
    (show (153.735 : ℝ) < √200 + 1 + 98 * √2 by linarith)
  have hhi := mul_self_lt_mul_self (by positivity : (0 : ℝ) ≤ √200 + 1 + 98 * √2)
    (show √200 + 1 + 98 * √2 < (153.7351 : ℝ) by linarith)
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  · rw [div_lt_iff₀ (by norm_num)]
    nlinarith

example := @effectiveRank_zeroIndicatorKernel_199 ⟨by norm_num⟩

/-- `review/flat_spectrum.md`, Torch check: "p/4+√(p/2) gives 31.21" at `p = 97`, as
`31.214 < 97/4 + √(97/2) < 31.215`. -/
theorem quarter_add_sqrt_half_97 :
    31.214 < (97 : ℝ) / 4 + √(97 / 2) ∧ (97 : ℝ) / 4 + √(97 / 2) < 31.215 := by
  have h1 : (6.964 : ℝ) < √(97 / 2) := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have h2 : √(97 / 2) < (6.965 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  constructor <;> linarith

/-- `review/flat_spectrum.md`, Torch check: "p/4+√(p/2) gives … 59.72" at `p = 199`, as
`59.724 < 199/4 + √(199/2) < 59.725`. -/
theorem quarter_add_sqrt_half_199 :
    59.724 < (199 : ℝ) / 4 + √(199 / 2) ∧ (199 : ℝ) / 4 + √(199 / 2) < 59.725 := by
  have h1 : (9.974 : ℝ) < √(199 / 2) := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have h2 : √(199 / 2) < (9.975 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  constructor <;> linarith

end Transformer.Grokking.FlatSpectrum
