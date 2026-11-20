module

public import Transformer.Grokking.FlatSpectrum.CopiesTheoremB

/-!
# The sampling noise of an empirical covariance

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV6, *Sample size*. The
covariance of a fixed readout `A` with `h` is estimated from `N` independent uniform inputs
`s : Fin N → Ω` by the empirical mean of the centered product
`X = (A − E A) conj(h − E h)`, whose mean is `Cov(A, h)` (`mean_centeredProduct`). The empirical
mean is unbiased (`mean_empiricalMean`), and its mean square error is exactly `Var X / N`
(`meanSq_empiricalMean_sub`), because `N` independent copies of a centered variable have
`E |∑_n Y(s_n)|² = N E |Y|²` (`meanSq_sum_pi_of_mean_eq_zero`).

If `|h| = 1` and `|E h| ≤ ε ≤ 1`, then `(1 − ε)² Var A ≤ E |X|² ≤ (1 + ε)² Var A`
(`meanSq_centeredProduct_ge`, `meanSq_centeredProduct_le`): the noise of a readout is of the
order of its own spread over `√N`, however weak its covariance with `h`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {α β Ω : Type*} [Fintype α] [Fintype β] [Fintype Ω]

lemma mean_comp_equiv {Ω' : Type*} [Fintype Ω'] (e : Ω' ≃ Ω) (F : Ω → ℂ) :
    mean (fun x => F (e x)) = mean F := by
  rw [mean, mean, Equiv.sum_comp e F, Fintype.card_congr e]

lemma meanSq_comp_equiv {Ω' : Type*} [Fintype Ω'] (e : Ω' ≃ Ω) (F : Ω → ℂ) :
    meanSq (fun x => F (e x)) = meanSq F := by
  rw [meanSq, meanSq, Equiv.sum_comp e fun y => ‖F y‖ ^ 2, Fintype.card_congr e]

lemma meanSq_const_mul (c : ℂ) (A : α → ℂ) :
    meanSq (fun x => c * A x) = ‖c‖ ^ 2 * meanSq A := by
  simp only [meanSq, norm_mul, mul_pow, ← mul_sum, mul_div_assoc]

lemma ofReal_meanSq_eq_mean_mul_star (A : α → ℂ) :
    (meanSq A : ℂ) = mean fun x => A x * star (A x) := by
  rw [ofReal_meanSq]
  simp only [Complex.star_def, Complex.mul_conj', Complex.ofReal_pow]

/-- Independent parts, one of them centered: `E |Y(a) + S(b)|² = E |Y|² + E |S|²`. -/
lemma meanSq_add_prod [Nonempty α] [Nonempty β] {Y : α → ℂ} (hY : mean Y = 0) (S : β → ℂ) :
    meanSq (fun z : α × β => Y z.1 + S z.2) = meanSq Y + meanSq S := by
  have (a : α) : (mean fun b => (Y a + S b) * star (Y a + S b)) =
      Y a * star (Y a) + Y a * star (mean S) + mean S * star (Y a) +
        mean fun b => S b * star (S b) := by
    simp only [star_add, add_mul, mul_add, mean_add, mean_const, mean_const_mul, mean_mul_const,
      mean_star]
    ring
  apply Complex.ofReal_injective
  rw [ofReal_meanSq_eq_mean_mul_star, mean_prod]
  simp only [this, mean_add, mean_const, mean_mul_const, mean_const_mul, mean_star, hY,
    star_zero, mul_zero, zero_mul, add_zero, Complex.ofReal_add, ofReal_meanSq_eq_mean_mul_star]

example (S : Unit → ℂ) :=
  meanSq_add_prod (Y := fun b : Bool => if b then (1 : ℂ) else -1) (by simp [mean]) S

/-- `N` independent copies: `E ∑_n Y(s_n) = N E Y`. -/
lemma mean_sum_pi [Nonempty Ω] (Y : Ω → ℂ) (N : ℕ) :
    mean (fun s : Fin N → Ω => ∑ n, Y (s n)) = N * mean Y := by
  induction N with
  | zero => simp [mean]
  | succ N ih =>
    rw [← mean_comp_equiv (Fin.consEquiv fun _ => Ω)]
    simp only [Fin.consEquiv_apply, Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
    rw [mean_add, mean_prod fun z : Ω × (Fin N → Ω) => Y z.1,
      mean_prod fun z : Ω × (Fin N → Ω) => ∑ n, Y (z.2 n)]
    simp only [mean_const, ih]
    push_cast
    ring

/-- **`N` independent copies of a centered variable**: `E |∑_n Y(s_n)|² = N E |Y|²`. -/
theorem meanSq_sum_pi_of_mean_eq_zero [Nonempty Ω] {Y : Ω → ℂ} (hY : mean Y = 0) (N : ℕ) :
    meanSq (fun s : Fin N → Ω => ∑ n, Y (s n)) = N * meanSq Y := by
  induction N with
  | zero => simp [meanSq]
  | succ N ih =>
    rw [← meanSq_comp_equiv (Fin.consEquiv fun _ => Ω)]
    simp only [Fin.consEquiv_apply, Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
    rw [meanSq_add_prod hY fun b : Fin N → Ω => ∑ n, Y (b n), ih]
    push_cast
    ring

/-- A fair sign: `E |∑_n ±1|² = N E |±1|²`. -/
example (N : ℕ) := meanSq_sum_pi_of_mean_eq_zero (Ω := Bool)
  (Y := fun b => if b then (1 : ℂ) else -1) (by simp [mean]) N

/-- The empirical mean of `X` over the samples `s : Fin N → Ω`. -/
noncomputable def empiricalMean (N : ℕ) (X : Ω → ℂ) (s : Fin N → Ω) : ℂ := (∑ n, X (s n)) / N

/-- The empirical mean of `N ≠ 0` independent uniform samples is unbiased. -/
theorem mean_empiricalMean [Nonempty Ω] (X : Ω → ℂ) {N : ℕ} (hN : N ≠ 0) :
    mean (empiricalMean N X) = mean X := by
  have : empiricalMean N X = fun s => (N : ℂ)⁻¹ * ∑ n, X (s n) := by
    funext s
    rw [empiricalMean, div_eq_inv_mul]
  rw [this, mean_const_mul, mean_sum_pi, ← mul_assoc, inv_mul_cancel₀ (by exact_mod_cast hN),
    one_mul]

example := mean_empiricalMean (Ω := ZMod 3) (fun _ => 1) one_ne_zero

/-- **The sampling noise**: the mean square error of the empirical mean of `N ≠ 0` independent
uniform samples is exactly `Var X / N`. -/
theorem meanSq_empiricalMean_sub [Nonempty Ω] (X : Ω → ℂ) {N : ℕ} (hN : N ≠ 0) :
    meanSq (fun s : Fin N → Ω => empiricalMean N X s - mean X) = var X / N := by
  have hN' : (N : ℂ) ≠ 0 := by exact_mod_cast hN
  have hY : mean (fun x => X x - mean X) = 0 := by rw [mean_sub, mean_const, sub_self]
  have : (fun s : Fin N → Ω => empiricalMean N X s - mean X) =
      fun s => (N : ℂ)⁻¹ * ∑ n, (X (s n) - mean X) := by
    funext s
    rw [empiricalMean, sum_sub_distrib, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  rw [this, meanSq_const_mul, meanSq_sum_pi_of_mean_eq_zero hY, var, norm_inv,
    Complex.norm_natCast]
  have : (N : ℝ) ≠ 0 := by exact_mod_cast hN
  field_simp

example := meanSq_empiricalMean_sub (Ω := ZMod 3) (fun _ => 1) one_ne_zero

/-- The centered product `X = (A − E A) conj(h − E h)`, whose mean is `Cov(A, h)`. -/
noncomputable def centeredProduct (A h : α → ℂ) (x : α) : ℂ :=
  (A x - mean A) * star (h x - mean h)

lemma mean_centeredProduct (A h : α → ℂ) : mean (centeredProduct A h) = cov A h := by
  cases isEmpty_or_nonempty α
  · simp [cov, mean]
  rw [cov_eq_mean_sub_mul, show centeredProduct A h =
    fun x => (A x - mean A) * star (h x - mean h) from rfl]
  simp only [star_sub, mul_sub, mean_sub, mean_mul_const, mean_const, sub_self, zero_mul,
    sub_zero]

/-- With `|h| = 1` and `|E h| ≤ ε ≤ 1`: `E |X|² ≥ (1 − ε)² Var A`. -/
lemma meanSq_centeredProduct_ge {A h : α → ℂ} (hh : ∀ x, ‖h x‖ = 1) {ε : ℝ}
    (hε : ‖mean h‖ ≤ ε) (hε1 : ε ≤ 1) : (1 - ε) ^ 2 * var A ≤ meanSq (centeredProduct A h) := by
  have key (x : α) :
      (1 - ε) ^ 2 * ‖A x - mean A‖ ^ 2 ≤ ‖centeredProduct A h x‖ ^ 2 := by
    have h1 : 1 - ε ≤ ‖h x - mean h‖ := by
      have := norm_sub_norm_le (h x) (mean h)
      rw [hh x] at this
      linarith
    have h2 : (1 - ε) ^ 2 ≤ ‖h x - mean h‖ ^ 2 := pow_le_pow_left₀ (by linarith) h1 2
    rw [centeredProduct, norm_mul, norm_star, mul_pow, mul_comm (‖A x - mean A‖ ^ 2)]
    exact mul_le_mul_of_nonneg_right h2 (by positivity)
  rw [var, meanSq, meanSq, ← mul_div_assoc, mul_sum]
  gcongr with x
  exact key x

/-- With `|h| = 1` and `|E h| ≤ ε`: `E |X|² ≤ (1 + ε)² Var A`. -/
lemma meanSq_centeredProduct_le {A h : α → ℂ} (hh : ∀ x, ‖h x‖ = 1) {ε : ℝ}
    (hε : ‖mean h‖ ≤ ε) : meanSq (centeredProduct A h) ≤ (1 + ε) ^ 2 * var A := by
  have key (x : α) :
      ‖centeredProduct A h x‖ ^ 2 ≤ (1 + ε) ^ 2 * ‖A x - mean A‖ ^ 2 := by
    have h1 : ‖h x - mean h‖ ≤ 1 + ε := by
      have := norm_sub_le (h x) (mean h)
      rw [hh x] at this
      linarith
    have h2 : ‖h x - mean h‖ ^ 2 ≤ (1 + ε) ^ 2 := pow_le_pow_left₀ (norm_nonneg _) h1 2
    rw [centeredProduct, norm_mul, norm_star, mul_pow, mul_comm (‖A x - mean A‖ ^ 2)]
    exact mul_le_mul_of_nonneg_right h2 (by positivity)
  rw [var, meanSq, meanSq, ← mul_div_assoc, mul_sum]
  gcongr with x
  exact key x

/-- A fair sign `h = ±1` has `|h| = 1` and `E h = 0`. -/
example (A : Bool → ℂ) :=
  meanSq_centeredProduct_ge (A := A) (h := fun b => if b then (1 : ℂ) else -1)
    (fun b => by cases b <;> simp) (ε := 0) (by simp [mean]) zero_le_one

example (A : Bool → ℂ) :=
  meanSq_centeredProduct_le (A := A) (h := fun b => if b then (1 : ℂ) else -1)
    (fun b => by cases b <;> simp) (ε := 0) (by simp [mean])

end Transformer.Grokking.FlatSpectrum
