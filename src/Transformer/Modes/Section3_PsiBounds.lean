/-
# The number of modes of a Gaussian KDE — a Gaussian majorant for `ψ`

In arXiv:2412.09080v3, §3.1, `eq:psi` is a Gaussian times four cubic Hermite
polynomials.  Their valid global bound includes a linear term, as proved in
`Section3_Hermite`.  Both the linear and cubic terms are absorbed into half
the Gaussian exponent.  This gives a majorant usable in `lem:error-3`
without the source's false purely cubic estimate near the origin.
-/

import Transformer.Modes.Section3_CumulantBounds
import Transformer.Modes.Section2_WidthTerms
import Transformer.Modes.Section2_GaussianShiftBounds

open Real MeasureTheory

namespace Transformer.Modes

/-- The moments used in `lem:eta` are nonnegative, including when an
integral is undefined and Lean assigns it zero.
Source: arXiv:2412.09080v3, §3.1, `lem:eta`. -/
theorem etaMoment_nonneg (β t : ℝ) (s : ℕ) : 0 ≤ etaMoment β t s :=
  integral_nonneg fun z => by unfold eucl; positivity

/-- Absorb the linear and cubic Hermite majorant into a Gaussian.
Source: arXiv:2412.09080v3, §3.1, proof of `lem:error-3`; the linear term
corrects the bound before `eq:H3-bound`. -/
theorem linear_cube_mul_exp_le {r : ℝ} (hr : 0 ≤ r) :
    (r + r ^ 3) * Real.exp (-(r ^ 2) / 4) ≤ 73 := by
  have hp : r + r ^ 3 ≤ 1 + 2 * r ^ 2 + r ^ 4 := by
    have hs := sq_nonneg (r - 1)
    have hs' := mul_nonneg (sq_nonneg r) hs
    have hn : 0 ≤ r + r ^ 3 := by positivity
    nlinarith
  have h2 : r ^ 2 * Real.exp (-(r ^ 2) / 4) ≤ 4 := by
    have h := mul_exp_neg_le (x := r ^ 2) (by norm_num : (0 : ℝ) < 1 / 4)
    norm_num only [show -(1 / 4 : ℝ) * r ^ 2 = -(r ^ 2) / 4 by ring] at h
    exact h
  have h8 : r ^ 2 * Real.exp (-(r ^ 2) / 8) ≤ 8 := by
    have h := mul_exp_neg_le (x := r ^ 2) (by norm_num : (0 : ℝ) < 1 / 8)
    norm_num only [show -(1 / 8 : ℝ) * r ^ 2 = -(r ^ 2) / 8 by ring] at h
    exact h
  have h4 : r ^ 4 * Real.exp (-(r ^ 2) / 4) ≤ 64 := by
    have hs := pow_le_pow_left₀ (by positivity) h8 2
    have he : Real.exp (-(r ^ 2) / 8) ^ 2 = Real.exp (-(r ^ 2) / 4) := by
      rw [pow_two, ← Real.exp_add]
      congr 1
      ring
    simpa only [mul_pow, ← pow_mul, he, Nat.reduceMul, show (8 : ℝ) ^ 2 = 64 by norm_num] using hs
  have he : Real.exp (-(r ^ 2) / 4) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
  calc (r + r ^ 3) * Real.exp (-(r ^ 2) / 4)
      ≤ (1 + 2 * r ^ 2 + r ^ 4) * Real.exp (-(r ^ 2) / 4) := by gcongr
    _ ≤ 73 := by nlinarith

example : (0 : ℝ) ≤ 1 := zero_le_one

/-- A third Hermite polynomial times the standard Gaussian has a wider
Gaussian majorant.  Source: arXiv:2412.09080v3, §3.1, `eq:psi`,
`eq:H3-bound` corrected to include the linear term. -/
theorem phi2_mul_abs_hermite_le (z : ℝ × ℝ) (k : ℕ) (hk : k ≤ 3) :
    phi2 z * |hermite3 k z| ≤
      219 * (2 * π)⁻¹ * Real.exp (-(eucl z ^ 2) / 4) := by
  have hr : 0 ≤ eucl z := Real.sqrt_nonneg _
  have hs : eucl z ^ 2 = z.1 ^ 2 + z.2 ^ 2 := Real.sq_sqrt (by positivity)
  have he : Real.exp (-(eucl z ^ 2) / 2) =
      Real.exp (-(eucl z ^ 2) / 4) * Real.exp (-(eucl z ^ 2) / 4) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc phi2 z * |hermite3 k z|
      ≤ phi2 z * (3 * (eucl z + eucl z ^ 3)) := by
        exact mul_le_mul_of_nonneg_left (abs_hermite_le k hk z)
          (by unfold phi2; positivity)
    _ = (3 * (2 * π)⁻¹ * Real.exp (-(eucl z ^ 2) / 4)) *
        ((eucl z + eucl z ^ 3) * Real.exp (-(eucl z ^ 2) / 4)) := by
          rw [phi2, ← hs, he]
          ring
    _ ≤ (3 * (2 * π)⁻¹ * Real.exp (-(eucl z ^ 2) / 4)) * 73 := by
        gcongr
        exact linear_cube_mul_exp_le hr
    _ = _ := by ring

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The third-order Edgeworth correction is bounded by `η₃` times a
Gaussian with doubled covariance.  Source: arXiv:2412.09080v3, §3.1,
`eq:psi`, `lem:eta`, proof of `lem:error-3`.  This uses the corrected global
Hermite bound and controls the absolute value of the signed correction. -/
theorem abs_psiOf_le_gaussian {μ : Measure (ℝ × ℝ)} [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hexp : HasExpMoments μ) (z : ℝ × ℝ) :
    |psiOf μ z| ≤ 876 * (2 * π)⁻¹ * (∫ x, eucl x ^ 3 ∂μ) *
      Real.exp (-(eucl z ^ 2) / 4) := by
  let η : ℝ := ∫ x, eucl x ^ 3 ∂μ
  have hη : 0 ≤ η := integral_nonneg fun x => by unfold eucl; positivity
  have hφ : 0 ≤ phi2 z := by unfold phi2; positivity
  have hterm (k : ℕ) (hk : k ∈ Finset.range 4) :
      phi2 z * |cumulantOf μ k (3 - k) /
        ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * hermite3 k z| ≤
          219 * (2 * π)⁻¹ * η * Real.exp (-(eucl z ^ 2) / 4) := by
    have hk3 : k ≤ 3 := by simp only [Finset.mem_range] at hk; omega
    have hc := abs_cumulant_three_le hμ hexp k hk3
    have hden : (1 : ℝ) ≤ (k.factorial : ℝ) * ((3 - k).factorial : ℝ) := by
      have h : 1 ≤ k.factorial * (3 - k).factorial :=
        Nat.one_le_iff_ne_zero.mpr (Nat.mul_ne_zero
          (Nat.factorial_pos k).ne' (Nat.factorial_pos (3 - k)).ne')
      exact_mod_cast h
    have hcoef : |cumulantOf μ k (3 - k) /
        ((k.factorial : ℝ) * ((3 - k).factorial : ℝ))| ≤ η := by
      rw [abs_div, abs_of_pos (lt_of_lt_of_le zero_lt_one hden)]
      exact (div_le_self (abs_nonneg _) hden).trans hc
    rw [abs_mul]
    calc phi2 z * (|cumulantOf μ k (3 - k) /
          ((k.factorial : ℝ) * ((3 - k).factorial : ℝ))| * |hermite3 k z|)
        = |cumulantOf μ k (3 - k) /
          ((k.factorial : ℝ) * ((3 - k).factorial : ℝ))| *
            (phi2 z * |hermite3 k z|) := by ring
      _ ≤ η * (phi2 z * |hermite3 k z|) :=
        mul_le_mul_of_nonneg_right hcoef (mul_nonneg hφ (abs_nonneg _))
      _ ≤ η * (219 * (2 * π)⁻¹ * Real.exp (-(eucl z ^ 2) / 4)) :=
        mul_le_mul_of_nonneg_left (phi2_mul_abs_hermite_le z k hk3) hη
      _ = _ := by ring
  rw [psiOf, abs_mul, abs_of_nonneg hφ]
  calc phi2 z * |∑ k ∈ Finset.range 4, cumulantOf μ k (3 - k) /
        ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * hermite3 k z|
      ≤ phi2 z * ∑ k ∈ Finset.range 4, |cumulantOf μ k (3 - k) /
          ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * hermite3 k z| :=
        mul_le_mul_of_nonneg_left (Finset.abs_sum_le_sum_abs _ _) hφ
    _ = ∑ k ∈ Finset.range 4, phi2 z * |cumulantOf μ k (3 - k) /
        ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * hermite3 k z| := by
      rw [Finset.mul_sum]
    _ ≤ ∑ k ∈ Finset.range 4,
        219 * (2 * π)⁻¹ * η * Real.exp (-(eucl z ^ 2) / 4) := Finset.sum_le_sum hterm
    _ = _ := by simp [η]; ring

example : IsStandardized stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨isStandardized_stdGauss2, hasExpMoments_stdGauss2⟩

/-- The Gaussian majorant for the KDE's Edgeworth correction.
Source: arXiv:2412.09080v3, §3.1, `lem:error-3`, `eq:psi`. -/
theorem abs_psi_lawY_le_gaussian {β : ℝ} (hβ : 0 < β) (t : ℝ) (z : ℝ × ℝ) :
    |psiOf (lawY β t) z| ≤ 876 * (2 * π)⁻¹ * etaMoment β t 3 *
      Real.exp (-(eucl z ^ 2) / 4) :=
  abs_psiOf_le_gaussian (isStandardized_lawY hβ t) (hasExpMoments_lawY hβ t) z

example : (0 : ℝ) < 1 := one_pos

/-- The squared norm on the Kac–Rice line, with the exact Gaussian shift.
Source: arXiv:2412.09080v3, §2.3, `lem:phi-t`, used in §3.1 `lem:error-3`. -/
theorem eucl_whiten_kr_sq {β : ℝ} (hβ : 0 < β) (n : ℕ) (t y : ℝ) :
    eucl (whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
      (-muFst n β t, y - muSnd n β t)) ^ 2 =
        phiA n β t + phiAlpha β t * (y - phiDelta n β t) ^ 2 := by
  have hf := sigmaFst_pos hβ t
  have hD := sigmaDet_pos hβ t
  rw [eucl_whiten_sq hf hD]
  have h := krQuad_zero_eq (n := n) hf.ne' hD.ne' y
  simpa only [krQuad, zero_sub] using h

example : (0 : ℝ) < 1 := one_pos

/-- The wider Gaussian used for `ψ` has a first moment bounded linearly in
the shift.  Source: arXiv:2412.09080v3, §3.1, proof of `lem:error-3`. -/
theorem lintegral_mul_exp_quarter_shift_sq_le {a δ : ℝ} (ha : 0 < a) :
    (∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal
      (y * Real.exp (-(a / 4) * (y - δ) ^ 2))) ≤
        ENNReal.ofReal (2 * a⁻¹ * (1 + 3 * (|δ| * Real.sqrt a))) := by
  have hh : 0 < a / 2 := by positivity
  have he : a / 2 / 2 = a / 4 := by ring
  have hi := integrableOn_mul_exp_neg_shift_sq (δ := δ) hh
  rw [he] at hi
  rw [← ofReal_integral_eq_lintegral_ofReal hi (by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
    exact mul_nonneg (le_of_lt hy) (Real.exp_pos _).le)]
  apply ENNReal.ofReal_le_ofReal
  have h := integral_mul_exp_neg_shift_sq_le (δ := δ) hh
  rw [he, show (a / 2)⁻¹ = 2 * a⁻¹ by field_simp] at h
  refine h.trans ?_
  gcongr
  linarith

example : (0 : ℝ) < 1 := one_pos

end Transformer.Modes
