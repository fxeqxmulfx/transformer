module

public import Transformer.Grokking.FlatSpectrum.CopiesSampleSize

/-!
# The best readout of `k` copies: the intermediate's energy scales by `p^{−2(k−1)}`

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV6, *Sample size*. The
energy of the centered conditional mean of `h` given the label `f` is
`∑_t |Cov(1{f = t}, h)|² / P(f = t)` (`condEnergy`). For one pair and `h = ψ_r(x y)` it is
Theorem B's `‖E[ψ_r(ab) | f]‖²` less `|E h|² = 1/p²`; the root compares its maximum over `r`,
`0.044` at `p = 97`, with the sampling noise `2/p` of a half split.

Take `k = 1 + m` pairs with the label `f = ∑ᵢ (P(xᵢ) + xᵢ yᵢ)`, `P(0) = 0`, and any `h` of the
first pair. Each covariance with an indicator scales by exactly `p^{−m}`
(`cov_ite_copiesLabel`). The label law of one pair is `(1 − 1/p) U + δ₀/p`
(`meanSq_ite_pair`) and that of `k` pairs is `(1 − p^{−k}) U + p^{−k} δ₀`
(`meanSq_ite_copiesLabel`); at every value the first is at most twice the second and at least
`1 − 1/p` times it (`law_bounds`). So the energy of `k` pairs is `p^{−2m}` times the energy of
one pair, up to a factor in `[1 − 1/p, 2]` (`condEnergy_copies`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ}

/-- The energy of the centered conditional mean of `h` given the label `f`,
`∑_t P(f = t) |E[h − E h | f = t]|² = ∑_t |Cov(1{f = t}, h)|² / P(f = t)`, where `P(f = t)` is
the mean square of the indicator of `f = t`. -/
noncomputable def condEnergy [NeZero p] {α : Type*} [Fintype α] (f : α → ZMod p) (h : α → ℂ) :
    ℝ :=
  ∑ t, ‖cov (fun x => if f x = t then (1 : ℂ) else 0) h‖ ^ 2 /
    meanSq fun x => if f x = t then (1 : ℂ) else 0

lemma ofReal_meanSq_ite {α : Type*} [Fintype α] (f : α → ZMod p) (t : ZMod p) :
    (meanSq (fun x => if f x = t then (1 : ℂ) else 0) : ℂ) =
      mean fun x => if f x = t then (1 : ℂ) else 0 := by
  rw [ofReal_meanSq_eq_mean_mul_star]
  congr 1
  funext x
  split_ifs <;> simp

/-- The two label laws at one value, with `i = 1/p`, `e = p^{−k} ≤ i ≤ 1/2` and
`b = [t = 0] ∈ [0, 1]`: the law `(1 − e) i + e b` of `k` pairs is positive, and the law
`(1 − i) i + i b` of one pair is at most twice it and at least `1 − i` times it. -/
lemma law_bounds {i e b : ℝ} (hi0 : 0 < i) (hi : i ≤ 1 / 2) (he0 : 0 ≤ e) (he : e ≤ i)
    (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    0 < (1 - e) * i + e * b ∧ (1 - i) * i + i * b ≤ 2 * ((1 - e) * i + e * b) ∧
      (1 - i) * ((1 - e) * i + e * b) ≤ (1 - i) * i + i * b := by
  have h1 : 0 ≤ i - 2 * e * i + i ^ 2 := by
    nlinarith [mul_nonneg hi0.le (by linarith : (0 : ℝ) ≤ 1 - 2 * e)]
  have h2 : 0 ≤ 2 * e * (1 - i) + i ^ 2 := by
    nlinarith [mul_nonneg he0 (by linarith : (0 : ℝ) ≤ 1 - i), sq_nonneg i]
  refine ⟨by nlinarith [mul_pos (by linarith : (0 : ℝ) < 1 - e) hi0, mul_nonneg he0 hb0],
    by nlinarith [mul_nonneg (sub_nonneg.2 hb1) h1, mul_nonneg hb0 h2], ?_⟩
  nlinarith [mul_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ 1 - i) hi0.le) he0,
    mul_nonneg hb0 (by nlinarith [mul_nonneg hi0.le he0] : (0 : ℝ) ≤ i - (1 - i) * e)]

example := law_bounds (i := 1 / 2) (e := 1 / 4) (b := 1) (by norm_num) le_rfl (by norm_num)
  (by norm_num) zero_le_one le_rfl

variable [Fact p.Prime] {τ : Type*} [Fintype τ] [DecidableEq τ]

/-- `k = 1 + m` pairs have `E ψ_r(f) = p^{−k}` for every `r ≠ 0`. -/
lemma mean_stdAddChar_copiesLabel (P : ZMod p → ZMod p) (hP : P 0 = 0) {r : ZMod p}
    (hr : r ≠ 0) :
    mean (fun z => ZMod.stdAddChar (r * copiesLabel (τ := τ) P z)) =
      (p : ℂ)⁻¹ ^ (Fintype.card τ + 1) := by
  have := mean_stdAddChar_add (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2)
    (fun z : τ → ZMod p × ZMod p => ∑ j, (P (z j).1 + (z j).1 * (z j).2)) r
  rw [mean_stdAddChar_poly_add_mul P hP hr, mean_stdAddChar_copies P hP hr, ← pow_succ'] at this
  exact this

example := @mean_stdAddChar_copiesLabel 7 ⟨Nat.prime_seven⟩ (Fin 2) _ _ (fun x => x ^ 3)
  (by simp) 1 (by decide)

/-- The label law of one pair: `P(q = t) = (1 − 1/p)/p + [t = 0]/p`. -/
lemma meanSq_ite_pair (P : ZMod p → ZMod p) (hP : P 0 = 0) (t : ZMod p) :
    meanSq (fun ab : ZMod p × ZMod p => if P ab.1 + ab.1 * ab.2 = t then (1 : ℂ) else 0) =
      (1 - (p : ℝ)⁻¹) * (p : ℝ)⁻¹ + (p : ℝ)⁻¹ * if t = 0 then 1 else 0 := by
  apply Complex.ofReal_injective
  rw [ofReal_meanSq_ite,
    mean_ite_eq_of_mean_stdAddChar (fun r hr => mean_stdAddChar_poly_add_mul P hP hr) t]
  split_ifs <;> push_cast <;> ring

example := @meanSq_ite_pair 7 ⟨Nat.prime_seven⟩ (fun x => x ^ 3) (by simp) 0

/-- The label law of `k = 1 + m` pairs: `P(f = t) = (1 − p^{−k})/p + p^{−k} [t = 0]`. -/
lemma meanSq_ite_copiesLabel (P : ZMod p → ZMod p) (hP : P 0 = 0) (t : ZMod p) :
    meanSq (fun z => if copiesLabel (τ := τ) P z = t then (1 : ℂ) else 0) =
      (1 - (p : ℝ)⁻¹ ^ (Fintype.card τ + 1)) * (p : ℝ)⁻¹ +
        (p : ℝ)⁻¹ ^ (Fintype.card τ + 1) * if t = 0 then 1 else 0 := by
  apply Complex.ofReal_injective
  rw [ofReal_meanSq_ite, mean_ite_eq_of_mean_stdAddChar
    (fun r hr => mean_stdAddChar_copiesLabel (τ := τ) P hP hr) t]
  split_ifs <;> push_cast <;> ring

example := @meanSq_ite_copiesLabel 7 ⟨Nat.prime_seven⟩ (Fin 2) _ _ (fun x => x ^ 3) (by simp) 0

/-- The covariance of the indicator of `f = t` with a function `h` of the first pair scales by
exactly `p^{−m}`. -/
lemma cov_ite_copiesLabel (P : ZMod p → ZMod p) (hP : P 0 = 0) (h : ZMod p × ZMod p → ℂ)
    (t : ZMod p) :
    cov (fun z => if copiesLabel (τ := τ) P z = t then (1 : ℂ) else 0) (fun z => h z.1) =
      (p : ℂ)⁻¹ ^ Fintype.card τ *
        cov (fun ab : ZMod p × ZMod p => if P ab.1 + ab.1 * ab.2 = t then (1 : ℂ) else 0) h :=
  cov_comp_add_copies P hP (fun s => if s = t then 1 else 0)
    (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h

example := @cov_ite_copiesLabel 7 ⟨Nat.prime_seven⟩ (Fin 2) _ _ (fun x => x ^ 3) (by simp)
  (fun _ => 1) 0

/-- **The best readout of `k = 1 + m` pairs**: for `P(0) = 0` and every function `h` of the first
pair, the energy of the conditional mean of `h` given the label of `k` pairs is `p^{−2m}` times
its energy given the first pair's own label, up to a factor in `[1 − 1/p, 2]`. -/
theorem condEnergy_copies (P : ZMod p → ZMod p) (hP : P 0 = 0) (h : ZMod p × ZMod p → ℂ) :
    (1 - (p : ℝ)⁻¹) * ((p : ℝ)⁻¹ ^ Fintype.card τ) ^ 2 *
          condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h ≤
        condEnergy (copiesLabel (τ := τ) P) (fun z => h z.1) ∧
      condEnergy (copiesLabel (τ := τ) P) (fun z => h z.1) ≤
        2 * ((p : ℝ)⁻¹ ^ Fintype.card τ) ^ 2 *
          condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h := by
  have hp : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
  have hi0 : 0 < (p : ℝ)⁻¹ := by positivity
  have hi : (p : ℝ)⁻¹ ≤ 1 / 2 := by rw [one_div]; exact inv_anti₀ two_pos hp
  have he : (p : ℝ)⁻¹ ^ (Fintype.card τ + 1) ≤ (p : ℝ)⁻¹ :=
    pow_le_of_le_one hi0.le (by linarith) (by omega)
  have hb (t : ZMod p) :
      (0 : ℝ) ≤ (if t = 0 then 1 else 0) ∧ (if t = 0 then (1 : ℝ) else 0) ≤ 1 := by
    split_ifs <;> norm_num
  have key (t : ZMod p) := law_bounds hi0 hi (by positivity) he (hb t).1 (hb t).2
  have hP1 (t : ZMod p) := (law_bounds hi0 hi hi0.le le_rfl (hb t).1 (hb t).2).1
  have up {q c a b : ℝ} (hq : 0 ≤ q) (hc : 0 ≤ c) (ha : 0 < a) (hb : 0 < b)
      (h : b ≤ 2 * a) : q * c / a ≤ 2 * q * (c / b) := by
    rw [mul_div_assoc', div_le_div_iff₀ ha hb]
    nlinarith [mul_le_mul_of_nonneg_left h (mul_nonneg hq hc)]
  have low {q c a b K : ℝ} (hq : 0 ≤ q) (hc : 0 ≤ c) (ha : 0 < a) (hb : 0 < b)
      (h : K * b ≤ a) : K * q * (c / a) ≤ q * c / b := by
    rw [mul_div_assoc', div_le_div_iff₀ ha hb]
    nlinarith [mul_le_mul_of_nonneg_left h (mul_nonneg hq hc)]
  simp only [condEnergy, cov_ite_copiesLabel P hP h, meanSq_ite_pair P hP,
    meanSq_ite_copiesLabel (τ := τ) P hP, norm_mul, norm_pow, norm_inv, Complex.norm_natCast,
    mul_pow, mul_sum]
  constructor
  · refine sum_le_sum fun t _ => ?_
    exact low (sq_nonneg _) (sq_nonneg _) (hP1 t) (key t).1 (key t).2.2
  · refine sum_le_sum fun t _ => ?_
    exact up (sq_nonneg _) (sq_nonneg _) (key t).1 (hP1 t) (key t).2.1

example := @condEnergy_copies 7 ⟨Nat.prime_seven⟩ (Fin 2) _ _ (fun x => x ^ 3) (by simp)
  fun ab => ZMod.stdAddChar (ab.1 * ab.2)

end Transformer.Grokking.FlatSpectrum
