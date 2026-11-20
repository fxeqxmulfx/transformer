module

public import Transformer.Grokking.FlatSpectrum.SampleNoise

/-!
# One pair among `k` copies needs `N ≳ p^{2k−1}` samples to be seen by a fixed readout

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV6, *Sample size*. If
`|h| = 1`, `|E h| ≤ ε ≤ 1` and the signal is weak, `|Cov(A, h)|² ≤ c Var A`, then the empirical
covariance on `N` independent uniform inputs has a mean square error of at least
`((1 − ε)² − c) Var A / N` (`le_mul_meanSq_empiricalMean_sub`). Against the signal
(`sq_norm_cov_mul_le`),

    |Cov(A, h)|² ((1 − ε)² − c) ≤ c N · MSE,

so the error is at most the squared signal only if `(1 − ε)² ≤ (N + 1) c`
(`sq_le_of_meanSq_empiricalMean_le`).

For the label `f` of `k = 1 + m` copies of `x³ + x y` (`copiesLabel`) and the first pair's
intermediate `h = ψ_r(x₁ y₁)` (`firstIntermediate`), `|E h| = 1/p` and NV6 gives
`c = p^{−2m} (9/p + 16/p^{3/2})` (`sq_norm_cov_mul_le_copies`): the error stays above the
squared signal until `N + 1 ≥ (1 − 1/p)² p^{2k−1} / (9 + 16/√p)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

section Floor

variable {α : Type*} [Fintype α] [Nonempty α] {A h : α → ℂ} {ε c : ℝ} {N : ℕ}

/-- **The noise floor of a weak covariance**: if `|h| = 1`, `|E h| ≤ ε ≤ 1` and
`|Cov(A, h)|² ≤ c Var A`, the empirical covariance of `N ≠ 0` samples has a mean square error
of at least `((1 − ε)² − c) Var A / N`. -/
theorem le_mul_meanSq_empiricalMean_sub (hh : ∀ x, ‖h x‖ = 1) (hε : ‖mean h‖ ≤ ε)
    (hε1 : ε ≤ 1) (hc : ‖cov A h‖ ^ 2 ≤ c * var A) (hN : N ≠ 0) :
    ((1 - ε) ^ 2 - c) * var A ≤
      N * meanSq (fun s : Fin N → α => empiricalMean N (centeredProduct A h) s - cov A h) := by
  have hN' : (N : ℝ) ≠ 0 := by exact_mod_cast hN
  rw [← mean_centeredProduct, meanSq_empiricalMean_sub _ hN, mul_div_cancel₀ _ hN']
  have hX := meanSq_sub_eq_var_add (centeredProduct A h) 0
  simp only [sub_zero, mean_centeredProduct] at hX
  have hge := meanSq_centeredProduct_ge (A := A) hh hε hε1
  linarith

/-- **The signal against the noise floor**: `|Cov(A, h)|² ((1 − ε)² − c) ≤ c N · MSE`. -/
theorem sq_norm_cov_mul_le (hh : ∀ x, ‖h x‖ = 1) (hε : ‖mean h‖ ≤ ε) (hε1 : ε ≤ 1)
    (hc0 : 0 ≤ c) (hc : ‖cov A h‖ ^ 2 ≤ c * var A) (hN : N ≠ 0) :
    ‖cov A h‖ ^ 2 * ((1 - ε) ^ 2 - c) ≤
      c * (N * meanSq fun s : Fin N → α => empiricalMean N (centeredProduct A h) s - cov A h) := by
  have hfloor := le_mul_meanSq_empiricalMean_sub hh hε hε1 hc hN
  have hM := mul_nonneg (Nat.cast_nonneg (α := ℝ) N)
    (meanSq_nonneg fun s : Fin N → α => empiricalMean N (centeredProduct A h) s - cov A h)
  rcases le_or_gt 0 ((1 - ε) ^ 2 - c) with hD | hD
  · nlinarith [mul_le_mul_of_nonneg_right hc hD, mul_le_mul_of_nonneg_left hfloor hc0]
  · nlinarith [mul_nonneg hc0 hM, sq_nonneg ‖cov A h‖]

/-- **Reading a weak covariance off a sample**: if moreover `Var A > 0` and the mean square
error is at most `|Cov(A, h)|²`, then `(1 − ε)² ≤ (N + 1) c`. -/
theorem sq_le_of_meanSq_empiricalMean_le (hh : ∀ x, ‖h x‖ = 1) (hε : ‖mean h‖ ≤ ε)
    (hε1 : ε ≤ 1) (hc : ‖cov A h‖ ^ 2 ≤ c * var A) (hA : 0 < var A) (hN : N ≠ 0)
    (hnoise : meanSq (fun s : Fin N → α => empiricalMean N (centeredProduct A h) s - cov A h) ≤
      ‖cov A h‖ ^ 2) :
    (1 - ε) ^ 2 ≤ (N + 1) * c := by
  have hfloor := le_mul_meanSq_empiricalMean_sub hh hε hε1 hc hN
  have h1 := mul_le_mul_of_nonneg_left (hnoise.trans hc) (Nat.cast_nonneg (α := ℝ) N)
  have hlt : (1 - ε) ^ 2 * var A ≤ ((N : ℝ) + 1) * c * var A := by nlinarith
  exact le_of_mul_le_mul_right hlt hA

end Floor

/-- A sign read off itself: `A = h = ±1` has `Cov(A, h) = Var A = 1`. -/
example := le_mul_meanSq_empiricalMean_sub (A := fun b : Bool => if b then (1 : ℂ) else -1)
  (h := fun b => if b then (1 : ℂ) else -1) (fun b => by cases b <;> simp) (ε := 0)
  (by simp [mean]) zero_le_one (c := 1) (by norm_num [cov, var, meanSq, mean]) (N := 1)
  one_ne_zero

example := sq_norm_cov_mul_le (A := fun b : Bool => if b then (1 : ℂ) else -1)
  (h := fun b => if b then (1 : ℂ) else -1) (fun b => by cases b <;> simp) (ε := 0)
  (by simp [mean]) zero_le_one zero_le_one (by norm_num [cov, var, meanSq, mean]) (N := 1)
  one_ne_zero

/-- On one sample the empirical covariance of a sign with itself is exact. -/
example : (1 - (0 : ℝ)) ^ 2 ≤ (((1 : ℕ) : ℝ) + 1) * 1 := by
  have hA : var (fun b : Bool => if b then (1 : ℂ) else -1) = 1 := by
    norm_num [var, meanSq, mean]
  have hcov : cov (fun b : Bool => if b then (1 : ℂ) else -1)
      (fun b => if b then (1 : ℂ) else -1) = 1 := by
    norm_num [cov, mean]
  have hX : centeredProduct (fun b : Bool => if b then (1 : ℂ) else -1)
      (fun b => if b then (1 : ℂ) else -1) = fun _ => 1 := by
    funext b
    cases b <;> norm_num [centeredProduct, mean]
  exact sq_le_of_meanSq_empiricalMean_le (A := fun b : Bool => if b then (1 : ℂ) else -1)
    (h := fun b => if b then (1 : ℂ) else -1) (fun b => by cases b <;> simp) (by simp [mean])
    zero_le_one (by rw [hcov, hA]; norm_num) (by rw [hA]; norm_num) one_ne_zero
    (by rw [hX, hcov]; norm_num [empiricalMean, meanSq])

variable {p : ℕ} {τ : Type*} [Fintype τ]

/-- The label `f = ∑ᵢ (P(xᵢ) + xᵢ yᵢ)` of the first pair `z.1` and the other pairs `z.2`. -/
def copiesLabel (P : ZMod p → ZMod p) (z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p)) :
    ZMod p :=
  P z.1.1 + z.1.1 * z.1.2 + ∑ j, (P (z.2 j).1 + (z.2 j).1 * (z.2 j).2)

/-- The first pair's intermediate `ψ_r(x₁ y₁)`, as a function of all the pairs. -/
noncomputable def firstIntermediate [NeZero p] (r : ZMod p)
    (z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p)) : ℂ :=
  ZMod.stdAddChar (r * (z.1.1 * z.1.2))

variable [Fact p.Prime] [DecidableEq τ]

/-- The first pair's intermediate has the mean `1/p`. -/
lemma mean_firstIntermediate {r : ZMod p} (hr : r ≠ 0) :
    mean (firstIntermediate (τ := τ) r) = (p : ℂ)⁻¹ := by
  rw [show firstIntermediate (τ := τ) r = fun z => ZMod.stdAddChar (r * (z.1.1 * z.1.2)) from rfl,
    mean_prod]
  simp only [mean_const]
  simpa using mean_stdAddChar_poly_add_mul (p := p) (fun _ => 0) rfl hr

example := @mean_firstIntermediate 7 (Fin 1) _ ⟨Nat.prime_seven⟩ _ 1 (by decide)

/-- **A fixed readout of `k = 1 + m` copies of `x³ + x y`**: with
`c = p^{−2m} (9/p + 16/p^{3/2})`, the empirical covariance of `g(f)` with `ψ_r(x₁ y₁)` on
`N ≠ 0` independent uniform inputs has `|Cov|² ((1 − 1/p)² − c) ≤ c N · MSE`, so its mean square
error exceeds the squared covariance unless `N + 1 ≥ (1 − 1/p)²/c`. -/
theorem sq_norm_cov_mul_le_copies {r : ZMod p} (hr : r ≠ 0) (g : ZMod p → ℂ) {N : ℕ}
    (hN : N ≠ 0) :
    ‖cov (fun z => g (copiesLabel (τ := τ) (· ^ 3) z)) (firstIntermediate r)‖ ^ 2 *
        ((1 - (p : ℝ)⁻¹) ^ 2 - ((p : ℝ)⁻¹ ^ Fintype.card τ) ^ 2 * (9 / p + 16 / (p * √p))) ≤
      ((p : ℝ)⁻¹ ^ Fintype.card τ) ^ 2 * (9 / p + 16 / (p * √p)) *
        (N * meanSq fun s => empiricalMean N (centeredProduct
          (fun z => g (copiesLabel (τ := τ) (· ^ 3) z)) (firstIntermediate r)) s -
            cov (fun z => g (copiesLabel (τ := τ) (· ^ 3) z)) (firstIntermediate r)) := by
  have hp : (1 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).one_lt.le
  refine sq_norm_cov_mul_le (fun z => by simp [firstIntermediate]) ?_
    (inv_le_one_of_one_le₀ hp) (by positivity) ?_ hN
  · rw [mean_firstIntermediate hr, norm_inv, Complex.norm_natCast]
  · have hv := var_nonneg fun z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
      g (z.1.1 ^ 3 + z.1.1 * z.1.2 + ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2))
    have hsq := pow_le_pow_left₀ (norm_nonneg _)
      (norm_cov_copies_cube_add_mul_le (τ := τ) hr g) 2
    rw [mul_pow, mul_pow, Real.sq_sqrt hv, Real.sq_sqrt (by positivity)] at hsq
    calc _ ≤ _ := hsq
      _ = _ := by simp only [copiesLabel]; ring

example := @sq_norm_cov_mul_le_copies 7 (Fin 1) _ ⟨Nat.prime_seven⟩ _ 1 (by decide) (fun _ => 0)
  1 one_ne_zero

end Transformer.Grokking.FlatSpectrum
