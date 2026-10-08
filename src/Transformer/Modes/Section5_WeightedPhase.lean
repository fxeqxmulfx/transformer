import Transformer.Modes.Section5_FirstDerivativeTest

/-!
# Oscillatory integral bounds with a smooth amplitude

The proof of `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, multiplies the
oscillatory phase by a Gaussian density and a cutoff. This module transfers
a bound on every partial integral to the integral with that amplitude.
Integration by parts gives the factor `|A b| + ∫ |A'|`; both the endpoint
term and the amplitude derivative are retained.

Applied to the first derivative test, the resulting bound is
`4/(|ρ| δ) * (|A b| + ∫ |A'|)`. If `|A| ≤ M` and `|A'| ≤ L`, the factor
is at most `M + L * (b - a)`. These estimates apply on a bounded interval;
they do not assume that the source's fixed Gaussian tail decays with `ρ`.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- Transfer a uniform bound on a primitive to a weighted integral,
using the actual derivative of the amplitude.
Source: arXiv:2412.09080v3, §5.5, the weighted oscillatory integrals. -/
theorem weighted_integral_le_of_primitive_bound {a b C : ℝ} {f : ℝ → ℂ} {A B : ℝ → ℝ}
    (hab : a ≤ b) (hf : Continuous f)
    (hA : ∀ t ∈ Set.uIcc a b, HasDerivAt A (B t) t)
    (hB : ContinuousOn B (Set.uIcc a b))
    (hprim : ∀ t ∈ Set.uIcc a b, ‖∫ x in a..t, f x‖ ≤ C) :
    ‖∫ t in a..b, f t * (A t : ℂ)‖ ≤ C * (|A b| + ∫ t in a..b, |B t|) := by
  let H : ℝ → ℂ := fun t => ∫ x in a..t, f x
  have hH (t : ℝ) : HasDerivAt H (f t) t :=
    intervalIntegral.integral_hasDerivAt_right (hf.intervalIntegrable a t)
      (hf.stronglyMeasurableAtFilter volume (nhds t)) hf.continuousAt
  have hi := intervalIntegral.integral_mul_deriv_eq_deriv_mul
    (a := a) (b := b) (fun t _ => hH t) (fun t ht => (hA t ht).ofReal_comp)
    (hf.intervalIntegrable a b)
    ((Complex.continuous_ofReal.comp_continuousOn hB).intervalIntegrable)
  have h0 : H a = 0 := intervalIntegral.integral_same
  rw [h0, zero_mul, sub_zero] at hi
  have heq : (∫ t in a..b, f t * (A t : ℂ)) =
      H b * (A b : ℂ) - ∫ t in a..b, H t * (B t : ℂ) := by
    linear_combination hi
  have hJ : ‖∫ t in a..b, H t * (B t : ℂ)‖ ≤ C * ∫ t in a..b, |B t| := by
    rw [← intervalIntegral.integral_const_mul]
    apply intervalIntegral.norm_integral_le_of_norm_le hab
    · refine ae_of_all _ fun t ht => ?_
      have htu : t ∈ Set.uIcc a b := by
        rw [Set.uIcc_of_le hab]
        exact Set.Ioc_subset_Icc_self ht
      rw [norm_mul, Complex.norm_real, Real.norm_eq_abs]
      exact mul_le_mul_of_nonneg_right (hprim t htu) (abs_nonneg _)
    · exact (continuousOn_const.mul hB.abs).intervalIntegrable
  rw [heq]
  calc _ ≤ ‖H b * (A b : ℂ)‖ + ‖∫ t in a..b, H t * (B t : ℂ)‖ := norm_sub_le _ _
    _ ≤ C * |A b| + C * ∫ t in a..b, |B t| := by
      refine add_le_add ?_ hJ
      rw [norm_mul, Complex.norm_real, Real.norm_eq_abs]
      exact mul_le_mul_of_nonneg_right (hprim b Set.right_mem_uIcc) (abs_nonneg _)
    _ = _ := by ring

example : (0 : ℝ) ≤ 1 ∧ Continuous (fun _ : ℝ => (1 : ℂ)) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, ‖∫ _ in (0 : ℝ)..t, (1 : ℂ)‖ ≤ 1) := by
  refine ⟨zero_le_one, continuous_const, fun t _ => hasDerivAt_id t,
    continuousOn_const, ?_⟩
  intro t ht
  rw [Set.uIcc_of_le zero_le_one] at ht
  calc _ ≤ 1 * |t - 0| := intervalIntegral.norm_integral_le_of_norm_le_const
          (fun _ _ => by simp)
    _ = t := by rw [sub_zero, one_mul, abs_of_nonneg ht.1]
    _ ≤ 1 := ht.2

/-- The first derivative test with a smooth amplitude, controlling
the amplitude by its endpoint and total variation.
Source: arXiv:2412.09080v3, §5.5, nonstationary integration by parts. -/
theorem weighted_first_derivative_test {a b ρ δ : ℝ} {φ p q A B : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hδ : 0 < δ) (hcφ : Continuous φ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : (∀ t ∈ Set.uIcc a b, 0 ≤ q t) ∨ (∀ t ∈ Set.uIcc a b, q t ≤ 0))
    (hmin : ∀ t ∈ Set.uIcc a b, δ ≤ |p t|)
    (hA : ∀ t ∈ Set.uIcc a b, HasDerivAt A (B t) t)
    (hB : ContinuousOn B (Set.uIcc a b)) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t * (A t : ℂ)‖ ≤
      (4 / (|ρ| * δ)) * (|A b| + ∫ t in a..b, |B t|) := by
  have hK : Continuous (oscillatoryKernel ρ φ) := by unfold oscillatoryKernel; fun_prop
  apply weighted_integral_le_of_primitive_bound hab hK hA hB
  intro t ht
  have htab : a ≤ t ∧ t ≤ b := by simpa only [Set.uIcc_of_le hab, Set.mem_Icc] using ht
  have hsub : Set.uIcc a t ⊆ Set.uIcc a b := by
    intro r hr
    rw [Set.uIcc_of_le htab.1] at hr
    rw [Set.uIcc_of_le hab]
    exact ⟨hr.1, hr.2.trans htab.2⟩
  exact first_derivative_test htab.1 hρ hδ
    (fun r hr => hφ r (hsub hr)) (fun r hr => hp r (hsub hr)) (hcq.mono hsub)
    (hq.elim (fun h => Or.inl (fun r hr => h r (hsub hr)))
      (fun h => Or.inr (fun r hr => h r (hsub hr)))) (fun r hr => hmin r (hsub hr))

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ Continuous (fun t : ℝ => t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    ((∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0) ∨
      (∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0)) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ |1|) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Set.uIcc (0 : ℝ) 1) := by
  exact ⟨zero_le_one, one_ne_zero, one_pos, continuous_id,
    fun t _ => hasDerivAt_id t, fun t _ => hasDerivAt_const t 1, continuousOn_const,
    Or.inl (fun _ _ => le_rfl), by simp, fun t _ => hasDerivAt_id t, continuousOn_const⟩

/-- An amplitude and derivative bound give an explicit first derivative
estimate with the interval length included.
Source: arXiv:2412.09080v3, §5.5, the amplitude norm estimates. -/
theorem weighted_first_derivative_test_bounded {a b ρ δ M L : ℝ} {φ p q A B : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0) (hδ : 0 < δ) (hcφ : Continuous φ)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hcq : ContinuousOn q (Set.uIcc a b))
    (hq : (∀ t ∈ Set.uIcc a b, 0 ≤ q t) ∨ (∀ t ∈ Set.uIcc a b, q t ≤ 0))
    (hmin : ∀ t ∈ Set.uIcc a b, δ ≤ |p t|)
    (hA : ∀ t ∈ Set.uIcc a b, HasDerivAt A (B t) t)
    (hB : ContinuousOn B (Set.uIcc a b))
    (hM : ∀ t ∈ Set.uIcc a b, |A t| ≤ M) (hL : ∀ t ∈ Set.uIcc a b, |B t| ≤ L) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t * (A t : ℂ)‖ ≤
      (4 / (|ρ| * δ)) * (M + L * (b - a)) := by
  have hI := intervalIntegral.norm_integral_le_of_norm_le_const
    (a := a) (b := b) (f := fun t => |B t|) (C := L) (fun t ht => by
      rw [Real.norm_eq_abs, abs_abs]
      exact hL t (Set.uIoc_subset_uIcc ht))
  rw [Real.norm_eq_abs, abs_of_nonneg (intervalIntegral.integral_nonneg hab
    (fun t _ => abs_nonneg (B t))), abs_of_nonneg (sub_nonneg.mpr hab)] at hI
  refine (weighted_first_derivative_test hab hρ hδ hcφ hφ hp hcq hq hmin hA hB).trans ?_
  exact mul_le_mul_of_nonneg_left (add_le_add (hM b Set.right_mem_uIcc) hI) (by positivity)

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ Continuous (fun t : ℝ => t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    ((∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0) ∨
      (∀ t ∈ Set.uIcc (0 : ℝ) 1, (0 : ℝ) ≤ 0)) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ |1|) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, |t| ≤ 1) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≤ 1) := by
  refine ⟨zero_le_one, one_ne_zero, one_pos, continuous_id,
    fun t _ => hasDerivAt_id t, fun t _ => hasDerivAt_const t 1, continuousOn_const,
    Or.inl (fun _ _ => le_rfl), by simp, fun t _ => hasDerivAt_id t,
    continuousOn_const, ?_, fun _ _ => le_rfl⟩
  intro t ht
  rw [Set.uIcc_of_le zero_le_one] at ht
  rw [abs_of_nonneg ht.1]
  exact ht.2

end Transformer.Modes
