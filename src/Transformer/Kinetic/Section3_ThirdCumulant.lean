module

public import Transformer.Kinetic.Section3_ThirdOrder

/-!
# The third cumulant of a product law

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: the
third cumulant `𝔼[F̄ Ḡ H̄]` is the covariance of `F̄Ḡ` with `H`. Summing
`abs_swap_term_le` over the swap identity gives

  `|κ| ≤ ¼ Σ_l Σ_m [s_m(F)(s²_{l,m}(G) s_l(H) + s_l(G) s²_{l,m}(H))
          + s_m(G)(s²_{l,m}(F) s_l(H) + s_l(F) s²_{l,m}(H))] + ½ Σ_l s_l(F) s_l(G) s_l(H)`.
-/

@[expose] public section

open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {N : ℕ} (μs : Idx N → Measure ℝ) [∀ i, IsProbabilityMeasure (μs i)]

/-- **The third-order inequality.** -/
theorem abs_thirdCumulant_le {F G H : (Idx N → ℝ) → ℝ} (hF : Measurable F) (hG : Measurable G)
    (hH : Measurable H) {CF CG CH : ℝ} (hFb : ∀ x, |F x| ≤ CF) (hGb : ∀ x, |G x| ≤ CG)
    (hHb : ∀ x, |H x| ≤ CH) {sF sG sH : Idx N → ℝ} {s2F s2G s2H : Idx N → Idx N → ℝ}
    (hsF : ∀ x l z, |F x - F (Function.update x l z)| ≤ sF l)
    (hsG : ∀ x l z, |G x - G (Function.update x l z)| ≤ sG l)
    (hsH : ∀ x l z, |H x - H (Function.update x l z)| ≤ sH l)
    (hs2F : ∀ x l m z z', m ≠ l → |F x - F (Function.update x l z) - F (Function.update x m z') +
      F (Function.update (Function.update x l z) m z')| ≤ s2F l m)
    (hs2G : ∀ x l m z z', m ≠ l → |G x - G (Function.update x l z) - G (Function.update x m z') +
      G (Function.update (Function.update x l z) m z')| ≤ s2G l m)
    (hs2H : ∀ x l m z z', m ≠ l → |H x - H (Function.update x l z) - H (Function.update x m z') +
      H (Function.update (Function.update x l z) m z')| ≤ s2H l m)
    (hdF : ∀ l, sF l ≤ s2F l l) (hdG : ∀ l, sG l ≤ s2G l l) (hdH : ∀ l, sH l ≤ s2H l l) :
    |∫ x, (F x - ∫ y, F y ∂(Measure.pi μs)) * (G x - ∫ y, G y ∂(Measure.pi μs)) *
        (H x - ∫ y, H y ∂(Measure.pi μs)) ∂(Measure.pi μs)| ≤
      1 / 4 * ∑ l, ∑ m, (sF m * (s2G l m * sH l + sG l * s2H l m) +
        sG m * (s2F l m * sH l + sF l * s2H l m)) +
      1 / 2 * ∑ l, sF l * sG l * sH l := by
  set μ := Measure.pi μs
  set cF := ∫ y, F y ∂μ
  set cG := ∫ y, G y ∂μ
  set cH := ∫ y, H y ∂μ
  set U : (Idx N → ℝ) → ℝ := fun x => (F x - cF) * (G x - cG)
  have hcF : |cF| ≤ CF := by
    simpa using norm_integral_le_of_norm_le_const (μ := μ) (f := F)
      (ae_of_all _ fun x => by simpa using hFb x)
  have hcG : |cG| ≤ CG := by
    simpa using norm_integral_le_of_norm_le_const (μ := μ) (f := G)
      (ae_of_all _ fun x => by simpa using hGb x)
  have hU : Measurable U := (hF.sub measurable_const).mul (hG.sub measurable_const)
  have hUb : ∀ x, |U x| ≤ (CF + CF) * (CG + CG) := fun x => by
    simp only [U, abs_mul]
    exact mul_le_mul ((abs_sub _ _).trans (add_le_add (hFb x) hcF))
      ((abs_sub _ _).trans (add_le_add (hGb x) hcG)) (abs_nonneg _)
      (by linarith [abs_nonneg cF, hcF])
  have hHint : Integrable H μ := integrable_of_abs_le hH hHb
  have hUHint : Integrable (fun x => U x * (H x - cH)) μ :=
    integrable_of_abs_le (hU.mul (hH.sub measurable_const)) fun x => by
      rw [abs_mul]
      have hcH : |cH| ≤ CH := by
        simpa using norm_integral_le_of_norm_le_const (μ := μ) (f := H)
          (ae_of_all _ fun x => by simpa using hHb x)
      exact mul_le_mul (hUb x) ((abs_sub _ _).trans (add_le_add (hHb x) hcH)) (abs_nonneg _)
        ((abs_nonneg _).trans (hUb 0))
  have hcov : ∫ x, (F x - cF) * (G x - cG) * (H x - cH) ∂μ = cov[U, H; μ] := by
    have hH0 : ∫ x, (H x - cH) ∂μ = 0 := by
      rw [integral_sub hHint (integrable_const cH), integral_const]
      simp [cH]
    have hHc : Integrable (fun x => (∫ y, U y ∂μ) * (H x - cH)) μ :=
      (hHint.sub (integrable_const cH)).const_mul _
    have h1 : cov[U, H; μ] = ∫ x, (U x * (H x - cH) - (∫ y, U y ∂μ) * (H x - cH)) ∂μ := by
      unfold covariance
      exact integral_congr_ae (ae_of_all _ fun x => by simp only [cH]; ring)
    rw [h1, integral_sub hUHint hHc, integral_const_mul, hH0, mul_zero, sub_zero]
  rw [hcov, covariance_eq_sum_swap μs hU hH hUb hHb]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun l _ => ?_
  have h := abs_swap_term_le μs hF hG hH hFb hGb hHb hsF hsG hsH hs2F hs2G hs2H hdF hdG hdH l
  rw [abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 1 / 2), Finset.sum_add_distrib]
  linarith

/-- The hypotheses of `abs_thirdCumulant_le` hold for constant observables under Dirac
masses, with zero bounds. -/
example := abs_thirdCumulant_le (fun _ : Idx 2 => Measure.dirac (0 : ℝ)) (F := fun _ => 0)
  (G := fun _ => 0) (H := fun _ => 0) measurable_const measurable_const measurable_const
  (CF := 0) (CG := 0) (CH := 0) (fun _ => by simp) (fun _ => by simp) (fun _ => by simp)
  (sF := fun _ => 0) (sG := fun _ => 0) (sH := fun _ => 0) (s2F := fun _ _ => 0)
  (s2G := fun _ _ => 0) (s2H := fun _ _ => 0) (fun _ _ _ => by simp) (fun _ _ _ => by simp)
  (fun _ _ _ => by simp) (fun _ _ _ _ _ _ => by simp) (fun _ _ _ _ _ _ => by simp)
  (fun _ _ _ _ _ _ => by simp) (fun _ => le_rfl) (fun _ => le_rfl) (fun _ => le_rfl)

end Kinetic
end Transformer
