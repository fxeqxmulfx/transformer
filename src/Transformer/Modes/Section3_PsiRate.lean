/-
# The number of modes of a Gaussian KDE — the inner Edgeworth integral

The inner integral of `lem:error-3`, arXiv:2412.09080v3, §3.1.  The Gaussian
majorant for `ψ` integrates over the positive half-line using the exact
completed square and the shifted first-moment estimate.  The shift `δ_t`
is retained: replacing `y = δ_t + u/√α_t` by `u/√α_t`, as in the source's
proof, would omit a contribution that need not be uniformly bounded.
The covariance and shift estimates then give a uniform Gaussian rate on
`T`, with the third-moment prefactor still explicit.
-/

import Transformer.Modes.Section3_PsiBounds
import Transformer.Modes.Section3_ErrorThird
import Transformer.Modes.Section2_ProxyKRShift

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer.Modes

/-- The inner Edgeworth integral is controlled by its moment, covariance
scale, rate exponent and linear shift.  The cutoff can be enlarged to the
positive half-line; for `y ≤ 0` the nonnegative integrand is zero.
Source: arXiv:2412.09080v3, §3.1, proof of `lem:error-3`.  The Gaussian
majorant includes the linear Hermite term and the first moment retains `δ_t`. -/
theorem lintegral_psi_line_le {n : ℕ} (hn : 0 < n) {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    (∫⁻ y in Set.Ioi (deltaCut n β t), ENNReal.ofReal
      (y * ((n : ℝ) * sigmaDet β t) ^ (-(1 : ℝ) / 2) *
        |psiOf (lawY β t) (whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
          (-muFst n β t, y - muSnd n β t))|)) ≤
      ENNReal.ofReal (1752 * (2 * π)⁻¹ * ((n : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment β t 3) *
        (sigmaDet β t ^ (-(1 : ℝ) / 2) * (phiAlpha β t)⁻¹) *
        Real.exp (-phiA n β t / 4) *
        (1 + 3 * (|phiDelta n β t| * Real.sqrt (phiAlpha β t)))) := by
  have hD := sigmaDet_pos hβ t
  have hα : 0 < phiAlpha β t := div_pos (sigmaFst_pos hβ t) hD
  have hη := etaMoment_nonneg β t 3
  have hf : 0 ≤ ((n : ℝ) * sigmaDet β t) ^ (-(1 : ℝ) / 2) := by positivity
  let f : ℝ → ℝ≥0∞ := fun y => ENNReal.ofReal
    (y * ((n : ℝ) * sigmaDet β t) ^ (-(1 : ℝ) / 2) *
      |psiOf (lawY β t) (whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
        (-muFst n β t, y - muSnd n β t))|)
  have hsupp : Function.support f ⊆ Set.Ioi (0 : ℝ) := by
    intro y hy
    by_contra h
    have hy0 : y ≤ 0 := le_of_not_gt h
    exact hy (ENNReal.ofReal_eq_zero.mpr
      (mul_nonpos_of_nonpos_of_nonneg (mul_nonpos_of_nonpos_of_nonneg hy0 hf) (abs_nonneg _)))
  let H : ℝ := 876 * (2 * π)⁻¹ * ((n : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment β t 3) *
    sigmaDet β t ^ (-(1 : ℝ) / 2) * Real.exp (-phiA n β t / 4)
  have hH : 0 ≤ H := by dsimp [H]; positivity
  have hpt (y : ℝ) (hy : y ∈ Set.Ioi (0 : ℝ)) :
      f y ≤ ENNReal.ofReal (H * (y *
        Real.exp (-(phiAlpha β t / 4) * (y - phiDelta n β t) ^ 2))) := by
    apply ENNReal.ofReal_le_ofReal
    have hψ := abs_psi_lawY_le_gaussian hβ t
      (whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
        (-muFst n β t, y - muSnd n β t))
    rw [eucl_whiten_kr_sq hβ n t y] at hψ
    have he : -(phiA n β t + phiAlpha β t * (y - phiDelta n β t) ^ 2) / 4 =
        -phiA n β t / 4 + -(phiAlpha β t / 4) * (y - phiDelta n β t) ^ 2 := by ring
    rw [he, Real.exp_add] at hψ
    have hfactor : ((n : ℝ) * sigmaDet β t) ^ (-(1 : ℝ) / 2) =
        (n : ℝ) ^ (-(1 : ℝ) / 2) * sigmaDet β t ^ (-(1 : ℝ) / 2) :=
      Real.mul_rpow (by positivity) hD.le
    calc y * ((n : ℝ) * sigmaDet β t) ^ (-(1 : ℝ) / 2) *
          |psiOf (lawY β t) (whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
            (-muFst n β t, y - muSnd n β t))|
        ≤ y * ((n : ℝ) * sigmaDet β t) ^ (-(1 : ℝ) / 2) *
          (876 * (2 * π)⁻¹ * etaMoment β t 3 *
            (Real.exp (-phiA n β t / 4) *
              Real.exp (-(phiAlpha β t / 4) * (y - phiDelta n β t) ^ 2))) :=
          mul_le_mul_of_nonneg_left hψ (mul_nonneg (le_of_lt hy) hf)
      _ = _ := by rw [hfactor]; dsimp [H]; ring
  calc ∫⁻ y in Set.Ioi (deltaCut n β t), f y
      ≤ ∫⁻ y, f y := setLIntegral_le_lintegral _ _
    _ = ∫⁻ y in Set.Ioi (0 : ℝ), f y := (setLIntegral_eq_of_support_subset hsupp).symm
    _ ≤ ∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal
        (H * (y * Real.exp (-(phiAlpha β t / 4) * (y - phiDelta n β t) ^ 2))) :=
      setLIntegral_mono' measurableSet_Ioi hpt
    _ = ENNReal.ofReal H * ∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal
        (y * Real.exp (-(phiAlpha β t / 4) * (y - phiDelta n β t) ^ 2)) := by
      simp_rw [ENNReal.ofReal_mul hH]
      exact lintegral_const_mul' _ _ ENNReal.ofReal_ne_top
    _ ≤ ENNReal.ofReal H * ENNReal.ofReal
        (2 * (phiAlpha β t)⁻¹ * (1 + 3 * (|phiDelta n β t| * Real.sqrt (phiAlpha β t)))) := by
      gcongr
      exact lintegral_mul_exp_quarter_shift_sq_le hα
    _ = _ := by rw [← ENNReal.ofReal_mul hH]; congr 1; dsimp [H]; ring

example : 0 < (1 : ℕ) ∧ (0 : ℝ) < 1 := ⟨Nat.zero_lt_one, one_pos⟩

/-- The inner integral for `ψ` is bounded by the covariance scale and a
Gaussian rate with a linear shift factor.  The factor `n^{-1/2}η₃` is kept
explicit so the same estimate can be integrated on `T` and `T'`.
Source: arXiv:2412.09080v3, §3.1, proof of `lem:error-3`. -/
theorem eventually_psi_inner_le_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C κ K : ℝ, 0 < C ∧ 0 < κ ∧ 0 ≤ K ∧
      ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
        (∫⁻ y in Set.Ioi (deltaCut (N k) (B k) t), ENNReal.ofReal
          (y * ((N k : ℝ) * sigmaDet (B k) t) ^ (-(1 : ℝ) / 2) *
            |psiOf (lawY (B k) t)
              (whiten (sigmaFst (B k) t) (sigmaCov (B k) t) (sigmaSnd (B k) t)
                (-muFst (N k) (B k) t, y - muSnd (N k) (B k) t))|)) ≤
          ENNReal.ofReal (C * ((N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3) *
            Real.sqrt (B k)) * ENNReal.ofReal
              (Real.exp (-κ * phiRate (N k) (B k) t) *
                (1 + K * Real.sqrt (rateScale (N k) (B k) t))) := by
  obtain ⟨A₁, A₂, hA₁, -, hA⟩ := phiA_isTheta hreg hω
  obtain ⟨X₁, X₂, -, hX₂, hX⟩ := eventually_det_inv_sqrt_mul_alpha_inv_bounds hreg hω
  obtain ⟨Cδ, hCδ, hδ⟩ := eventually_shift_sq_le_rateScale hreg hω
  refine ⟨1752 * (2 * π)⁻¹ * X₂, A₁ / 4, 3 * Real.sqrt Cδ,
    by positivity, by positivity, by positivity, ?_⟩
  filter_upwards [hA, hX, hδ, hreg.tendsto_N.eventually_ge_atTop 1]
    with k hAk hXk hδk hN t ht
  have hβ := hreg.B_pos k
  have hn : 0 < N k := by exact_mod_cast (by linarith : (0 : ℝ) < N k)
  have hD0 := sigmaDet_pos hβ t
  have hα0 : 0 < phiAlpha (B k) t := div_pos (sigmaFst_pos hβ t) hD0
  have hη := etaMoment_nonneg (B k) t 3
  have hR0 : 0 ≤ rateScale (N k) (B k) t := by unfold rateScale; positivity
  have hs : |phiDelta (N k) (B k) t| * Real.sqrt (phiAlpha (B k) t) ≤
      Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t) := by
    refine (sq_le_sq₀ (by positivity) (by positivity)).1 ?_
    rw [mul_pow, sq_abs, Real.sq_sqrt hα0.le, mul_pow,
      Real.sq_sqrt hCδ.le, Real.sq_sqrt hR0]
    nlinarith [hδk t ht]
  have hAle : Real.exp (-phiA (N k) (B k) t / 4) ≤
      Real.exp (-(A₁ / 4) * phiRate (N k) (B k) t) := by
    apply Real.exp_le_exp.mpr
    have h := (hAk t ht).1
    change A₁ * phiRate (N k) (B k) t ≤ phiA (N k) (B k) t at h
    linarith
  refine (lintegral_psi_line_le hn hβ t).trans ?_
  rw [← ENNReal.ofReal_mul (by positivity)]
  apply ENNReal.ofReal_le_ofReal
  calc 1752 * (2 * π)⁻¹ * ((N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3) *
        (sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (phiAlpha (B k) t)⁻¹) *
        Real.exp (-phiA (N k) (B k) t / 4) *
        (1 + 3 * (|phiDelta (N k) (B k) t| * Real.sqrt (phiAlpha (B k) t)))
      ≤ 1752 * (2 * π)⁻¹ * ((N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3) *
        (X₂ * Real.sqrt (B k)) * Real.exp (-(A₁ / 4) * phiRate (N k) (B k) t) *
        (1 + 3 * (Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t))) := by
          have hx := mul_le_mul_of_nonneg_left (hXk t ht).2
            (by positivity : 0 ≤ 1752 * (2 * π)⁻¹ *
              ((N k : ℝ) ^ (-(1 : ℝ) / 2) * etaMoment (B k) t 3))
          have he := mul_le_mul hx hAle (Real.exp_pos _).le (by positivity)
          exact mul_le_mul he (by linarith [hs]) (by positivity) (by positivity)
    _ = _ := by ring

example := eventually_psi_inner_le_T isRegime_succ isSlowGrowth_sqrt_log_log

end Transformer.Modes
