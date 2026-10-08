import Mathlib.MeasureTheory.Integral.IntervalIntegral.IntegrationByParts
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# Integration by parts on the nonstationary intervals

The proof of `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, partitions the
phase into stationary and nonstationary regions. The nonstationary estimate
is an integration by parts against `exp(-i ρ φ)`, with amplitude
`χ_R(x) exp(-x²/2)`. This module proves the identity and its norm bound for
an arbitrary differentiable real amplitude, on each interval where the
phase derivative does not vanish.

The source's displayed bound on `W(θ)` omits boundary terms. Here the two
endpoint terms are retained: the cutoff can be nonzero at the endpoints of
the removed stationary intervals. They cancel only when an application
proves that cancellation. The denominator and its derivative in the
interior term are exactly the ordinary quotient rule from the source.

These are local estimates, so no assumption on the phase outside the
integration interval is needed. They supply the nonstationary part of the
corrected fixed-`t`, `0 < β < 2` Fourier decay problem.
Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay` and its proof.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- The oscillatory factor of the Fourier integral in arXiv:2412.09080v3,
§5.5, with an arbitrary real phase. -/
noncomputable def oscillatoryKernel (ρ : ℝ) (φ : ℝ → ℝ) (t : ℝ) : ℂ :=
  Complex.exp (-Complex.I * ((ρ * φ t : ℝ) : ℂ))

/-- A real phase gives an oscillatory factor of modulus one.
Source: arXiv:2412.09080v3, §5.5, integral preceding `eq:uniform-decay`. -/
theorem norm_oscillatoryKernel (ρ : ℝ) (φ : ℝ → ℝ) (t : ℝ) :
    ‖oscillatoryKernel ρ φ t‖ = 1 := by
  simp [oscillatoryKernel, Complex.norm_exp]

/-- The actual derivative of the oscillatory factor.
Source: arXiv:2412.09080v3, §5.5, nonstationary integration by parts. -/
theorem hasDerivAt_oscillatoryKernel {ρ : ℝ} {φ : ℝ → ℝ} {p t : ℝ}
    (hφ : HasDerivAt φ p t) :
    HasDerivAt (oscillatoryKernel ρ φ)
      (oscillatoryKernel ρ φ t * (-Complex.I * (ρ : ℂ)) * (p : ℂ)) t := by
  have h := ((hφ.const_mul ρ).ofReal_comp.const_mul (-Complex.I)).cexp
  convert h using 1
  · rfl
  · simp only [oscillatoryKernel, Complex.ofReal_mul]
    ring

example : HasDerivAt (fun t : ℝ => t) 1 0 := hasDerivAt_id 0

/-- Exact nonstationary integration by parts, including both endpoint
terms. The source's formula on `W(θ)` omits these terms; no vanishing of the
amplitude at its internal boundaries is assumed here.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`. -/
theorem oscillatory_integral_parts {a b ρ : ℝ} {φ p q A B : ℝ → ℝ}
    (hρ : ρ ≠ 0)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hA : ∀ t ∈ Set.uIcc a b, HasDerivAt A (B t) t)
    (hq : ContinuousOn q (Set.uIcc a b)) (hB : ContinuousOn B (Set.uIcc a b))
    (hne : ∀ t ∈ Set.uIcc a b, p t ≠ 0) :
    (∫ t in a..b, oscillatoryKernel ρ φ t * (A t : ℂ)) =
      (-Complex.I * (ρ : ℂ))⁻¹ *
        (oscillatoryKernel ρ φ b * ((A b / p b : ℝ) : ℂ)
          - oscillatoryKernel ρ φ a * ((A a / p a : ℝ) : ℂ)
          - ∫ t in a..b, oscillatoryKernel ρ φ t *
            (((B t * p t - A t * q t) / (p t) ^ 2 : ℝ) : ℂ)) := by
  have hcp : ContinuousOn p (Set.uIcc a b) :=
    continuousOn_of_forall_continuousAt fun t ht => (hp t ht).continuousAt
  have hcA : ContinuousOn A (Set.uIcc a b) :=
    continuousOn_of_forall_continuousAt fun t ht => (hA t ht).continuousAt
  have hcK : ContinuousOn (oscillatoryKernel ρ φ) (Set.uIcc a b) :=
    continuousOn_of_forall_continuousAt fun t ht =>
      (hasDerivAt_oscillatoryKernel (ρ := ρ) (hφ t ht)).continuousAt
  have hcR : ContinuousOn (fun t => (B t * p t - A t * q t) / (p t) ^ 2)
      (Set.uIcc a b) := (hB.mul hcp |>.sub (hcA.mul hq)).div (hcp.pow 2)
        (fun t ht => pow_ne_zero 2 (hne t ht))
  have hv : ∀ t ∈ Set.uIcc a b, HasDerivAt (fun s => ((A s / p s : ℝ) : ℂ))
      (((B t * p t - A t * q t) / (p t) ^ 2 : ℝ) : ℂ) t := fun t ht =>
    ((hA t ht).fun_div (hp t ht) (hne t ht)).ofReal_comp
  have hi := intervalIntegral.integral_mul_deriv_eq_deriv_mul hv
    (fun t ht => hasDerivAt_oscillatoryKernel (ρ := ρ) (hφ t ht))
    ((Complex.continuous_ofReal.comp_continuousOn hcR).intervalIntegrable)
    (((hcK.mul continuousOn_const).mul
      (Complex.continuous_ofReal.comp_continuousOn hcp)).intervalIntegrable)
  have heq : (∫ t in a..b, ((A t / p t : ℝ) : ℂ) *
      (oscillatoryKernel ρ φ t * (-Complex.I * (ρ : ℂ)) * (p t : ℂ))) =
      (-Complex.I * (ρ : ℂ)) * ∫ t in a..b, oscillatoryKernel ρ φ t * (A t : ℂ) := by
    rw [← intervalIntegral.integral_const_mul]
    apply intervalIntegral.integral_congr
    intro t ht
    have hpt : (p t : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr (hne t ht)
    push_cast
    field_simp
  rw [heq] at hi
  have hnρ : (-Complex.I * (ρ : ℂ)) ≠ 0 :=
    mul_ne_zero (neg_ne_zero.mpr Complex.I_ne_zero) (Complex.ofReal_ne_zero.mpr hρ)
  have hcomm : (∫ t in a..b,
      (((B t * p t - A t * q t) / (p t) ^ 2 : ℝ) : ℂ) * oscillatoryKernel ρ φ t) =
      ∫ t in a..b, oscillatoryKernel ρ φ t *
        (((B t * p t - A t * q t) / (p t) ^ 2 : ℝ) : ℂ) :=
    intervalIntegral.integral_congr fun t _ => mul_comm _ _
  rw [hcomm, mul_comm ((A b / p b : ℝ) : ℂ), mul_comm ((A a / p a : ℝ) : ℂ)] at hi
  calc _ = (-Complex.I * (ρ : ℂ))⁻¹ *
      ((-Complex.I * (ρ : ℂ)) * ∫ t in a..b, oscillatoryKernel ρ φ t * (A t : ℂ)) := by
        rw [← mul_assoc, inv_mul_cancel₀ hnρ, one_mul]
    _ = _ := congrArg (fun z : ℂ => (-Complex.I * (ρ : ℂ))⁻¹ * z) hi

/-- The norm bound from the exact integration by parts formula, retaining
the two boundary contributions missing from the source's displayed bound.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`. -/
theorem norm_oscillatory_integral_le {a b ρ : ℝ} {φ p q A B : ℝ → ℝ}
    (hab : a ≤ b) (hρ : ρ ≠ 0)
    (hφ : ∀ t ∈ Set.uIcc a b, HasDerivAt φ (p t) t)
    (hp : ∀ t ∈ Set.uIcc a b, HasDerivAt p (q t) t)
    (hA : ∀ t ∈ Set.uIcc a b, HasDerivAt A (B t) t)
    (hq : ContinuousOn q (Set.uIcc a b)) (hB : ContinuousOn B (Set.uIcc a b))
    (hne : ∀ t ∈ Set.uIcc a b, p t ≠ 0) :
    ‖∫ t in a..b, oscillatoryKernel ρ φ t * (A t : ℂ)‖ ≤
      (|A b / p b| + |A a / p a| +
        ∫ t in a..b, |(B t * p t - A t * q t) / (p t) ^ 2|) / |ρ| := by
  have hJ : ‖∫ t in a..b, oscillatoryKernel ρ φ t *
      (((B t * p t - A t * q t) / (p t) ^ 2 : ℝ) : ℂ)‖ ≤
      ∫ t in a..b, |(B t * p t - A t * q t) / (p t) ^ 2| := by
    simpa only [norm_mul, norm_oscillatoryKernel, Complex.norm_real, Real.norm_eq_abs,
      one_mul] using intervalIntegral.norm_integral_le_integral_norm (f := fun t =>
        oscillatoryKernel ρ φ t *
          (((B t * p t - A t * q t) / (p t) ^ 2 : ℝ) : ℂ)) hab
  have he : ‖oscillatoryKernel ρ φ b * ((A b / p b : ℝ) : ℂ)
      - oscillatoryKernel ρ φ a * ((A a / p a : ℝ) : ℂ)
      - ∫ t in a..b, oscillatoryKernel ρ φ t *
          (((B t * p t - A t * q t) / (p t) ^ 2 : ℝ) : ℂ)‖ ≤
      |A b / p b| + |A a / p a| +
        ∫ t in a..b, |(B t * p t - A t * q t) / (p t) ^ 2| := by
    refine (norm_sub_le _ _).trans ?_
    have h := add_le_add (norm_sub_le
      (oscillatoryKernel ρ φ b * ((A b / p b : ℝ) : ℂ))
      (oscillatoryKernel ρ φ a * ((A a / p a : ℝ) : ℂ))) hJ
    simpa only [norm_mul, norm_oscillatoryKernel, Complex.norm_real, Real.norm_eq_abs,
      one_mul] using h
  rw [oscillatory_integral_parts hρ hφ hp hA hq hB hne,
    norm_mul, norm_inv, norm_mul, norm_neg, Complex.norm_I, Complex.norm_real,
    Real.norm_eq_abs, one_mul]
  exact (mul_le_mul_of_nonneg_left he (inv_nonneg.mpr (abs_nonneg ρ))).trans_eq (by ring)

/-- All hypotheses above hold on a nondegenerate interval with a nonzero
amplitude and oscillation: `φ(t) = A(t) = t`, `ρ = 1`, `0 ≤ t ≤ 1`. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun _ : ℝ => (1 : ℝ)) 0 t) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => s) 1 t) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    ContinuousOn (fun _ : ℝ => (1 : ℝ)) (Set.uIcc (0 : ℝ) 1) ∧
    (∀ t ∈ Set.uIcc (0 : ℝ) 1, (1 : ℝ) ≠ 0) := by
  exact ⟨zero_le_one, one_ne_zero, fun t _ => hasDerivAt_id t,
    fun t _ => hasDerivAt_const t 1, fun t _ => hasDerivAt_id t,
    continuousOn_const, continuousOn_const, fun _ _ => one_ne_zero⟩

end Transformer.Modes
