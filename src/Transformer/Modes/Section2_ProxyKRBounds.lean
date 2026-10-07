/-
# The number of modes of a Gaussian KDE — full-window proxy bounds

Upper and lower estimates for `lem:main-int-phi` of arXiv:2412.09080v3,
§2.3, retaining the Gaussian shift.  The upper bound integrates the shift
majorant; away from `[-1,1]`, the lower bound absorbs `α_t δ_t²` in the
rate exponent.  Neither estimate requires `n ≲ β^{5/2}`.
-/

import Transformer.Modes.Section2_ProxyKRShift

open Real Filter Asymptotics MeasureTheory
open scoped Topology ENNReal

namespace Transformer.Modes

/-- The full-window proxy integral is controlled by its linear shift.
Source: arXiv:2412.09080v3, §2.3, `lem:phi-t`, `lem:main-int-phi`.
Unlike `eq:int-phi-final`, this bound needs no `n ≲ β^{5/2}`. -/
theorem eventually_inner_le_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C κ K : ℝ, 0 < C ∧ 0 < κ ∧ 0 ≤ K ∧
      ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)),
        (∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal
          (y * sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * krPhi (N k) (B k) t 0 y)) ≤
          ENNReal.ofReal (C * Real.sqrt (B k)) * ENNReal.ofReal
            (Real.exp (-κ * phiRate (N k) (B k) t) *
              (1 + K * Real.sqrt (rateScale (N k) (B k) t))) := by
  obtain ⟨A₁, A₂, hA₁, -, hA⟩ := phiA_isTheta hreg hω
  obtain ⟨L, U, hL, -, hα⟩ := phiAlpha_isTheta hreg hω
  obtain ⟨X₁, X₂, -, hX₂, hX⟩ := eventually_det_inv_sqrt_mul_alpha_inv_bounds hreg hω
  obtain ⟨Cδ, hCδ, hδ⟩ := eventually_shift_sq_le_rateScale hreg hω
  refine ⟨X₂, A₁ / 2, 3 * Real.sqrt Cδ, hX₂, by positivity, by positivity, ?_⟩
  filter_upwards [hA, hα, hX, hδ, eventually_sigmaDet_pos hreg hω]
    with k hAk hαk hXk hδk hDk t ht
  have hβ := hreg.B_pos k
  have hb : 0 < B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by positivity
  have hα0 : 0 < phiAlpha (B k) t := lt_of_lt_of_le (mul_pos hL hb) (hαk t ht).1
  have hD0 : 0 < sigmaDet (B k) t := hDk t ht
  have hf : 0 < sigmaDet (B k) t ^ (-(1 : ℝ) / 2) := Real.rpow_pos_of_pos hD0 _
  have hR0 : 0 ≤ rateScale (N k) (B k) t := by unfold rateScale; positivity
  have hs : |phiDelta (N k) (B k) t| * Real.sqrt (phiAlpha (B k) t) ≤
      Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t) := by
    refine (sq_le_sq₀ (by positivity) (by positivity)).1 ?_
    rw [mul_pow, sq_abs, Real.sq_sqrt hα0.le, mul_pow,
      Real.sq_sqrt hCδ.le, Real.sq_sqrt hR0]
    nlinarith [hδk t ht]
  have hAle : Real.exp (-phiA (N k) (B k) t / 2) ≤
      Real.exp (-(A₁ / 2) * phiRate (N k) (B k) t) := by
    apply Real.exp_le_exp.mpr
    have h := (hAk t ht).1
    change A₁ * phiRate (N k) (B k) t ≤ phiA (N k) (B k) t at h
    linarith
  rw [lintegral_det_krPhi_eq hα0 hD0, ← ENNReal.ofReal_mul hf.le,
    ← ENNReal.ofReal_mul (by positivity)]
  apply ENNReal.ofReal_le_ofReal
  have hJ := integral_mul_krPhi_le (n := N k) hα0
  have hx := (hXk t ht).2
  calc sigmaDet (B k) t ^ (-(1 : ℝ) / 2) *
        (∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y)
      ≤ sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (Real.exp (-phiA (N k) (B k) t / 2) *
          ((phiAlpha (B k) t)⁻¹ *
            (1 + 3 * (|phiDelta (N k) (B k) t| * Real.sqrt (phiAlpha (B k) t))))) :=
        mul_le_mul_of_nonneg_left hJ hf.le
    _ = (sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (phiAlpha (B k) t)⁻¹) *
        Real.exp (-phiA (N k) (B k) t / 2) *
          (1 + 3 * (|phiDelta (N k) (B k) t| * Real.sqrt (phiAlpha (B k) t))) := by ring
    _ ≤ X₂ * Real.sqrt (B k) * Real.exp (-(A₁ / 2) * phiRate (N k) (B k) t) *
        (1 + 3 * (Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t))) := by
      refine mul_le_mul (mul_le_mul hx hAle (Real.exp_pos _).le (by positivity)) ?_
        (by positivity) (by positivity)
      linarith
    _ = _ := by ring

example := eventually_inner_le_T isRegime_succ isSlowGrowth_sqrt_log_log
/-- Integrating the full-window bound gives `√β` times the window length
plus a constant.  In particular the nonnegative integral is finite.
Source: arXiv:2412.09080v3, §2.3, `lem:main-int-phi`. -/
theorem eventually_proxyKR_T_le {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C D : ℝ, 0 < C ∧ 0 ≤ D ∧ ∀ᶠ k in atTop,
      proxyKR (N k) (B k) (intervalT (N k) (B k) (ω (B k))) ≤
        ENNReal.ofReal (C * Real.sqrt (B k) *
          (Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k)) + D)) := by
  obtain ⟨C, κ, K, hC, hκ, hK, hin⟩ := eventually_inner_le_T hreg hω
  let A := 1 + K * (1 + κ⁻¹)
  let D := 10 * K * Real.sqrt (π / (κ / 2))
  have hA : 0 < A := by dsimp [A]; positivity
  have hD : 0 ≤ D := by dsimp [D]; positivity
  refine ⟨C * (2 * A + 1), D, by positivity, hD, ?_⟩
  filter_upwards [hin, hreg.tendsto_N.eventually_ge_atTop 1] with k hk hN
  let a := Real.sqrt (2 * Real.log (N k) - Real.log (B k) - ω (B k))
  let ρ := B k ^ (-(3 : ℝ) / 2) * (N k : ℝ)
  have hβ := hreg.B_pos k
  have hρ : 0 < ρ := by dsimp [ρ]; positivity
  have ha : 0 ≤ a := Real.sqrt_nonneg _
  have hwidth := lintegral_shift_width_le hκ hK hρ ha
  have heq (t : ℝ) : phiRate (N k) (B k) t =
      ρ * Real.exp (-(t ^ 2) / 2) * t ^ 2 := by dsimp [ρ, phiRate]; ring
  calc
    proxyKR (N k) (B k) (intervalT (N k) (B k) (ω (B k)))
        ≤ ∫⁻ t in intervalT (N k) (B k) (ω (B k)), ENNReal.ofReal (C * Real.sqrt (B k)) *
            ENNReal.ofReal (Real.exp (-κ * phiRate (N k) (B k) t) *
              (1 + K * Real.sqrt (rateScale (N k) (B k) t))) :=
          setLIntegral_mono' measurableSet_Icc hk
    _ = ENNReal.ofReal (C * Real.sqrt (B k)) *
          ∫⁻ t in Set.Icc (-a) a, ENNReal.ofReal
            (Real.exp (-κ * (ρ * Real.exp (-(t ^ 2) / 2) * t ^ 2)) *
              (1 + K * Real.sqrt (ρ * Real.exp (-(t ^ 2) / 2)))) := by
        simp_rw [heq, show rateScale (N k) (B k) =
          fun t => ρ * Real.exp (-(t ^ 2) / 2) from rfl]
        exact lintegral_const_mul' _ _ ENNReal.ofReal_ne_top
    _ ≤ ENNReal.ofReal (C * Real.sqrt (B k)) * ENNReal.ofReal (A * (2 * a) + D) := by
        gcongr
    _ ≤ _ := by
      rw [← ENNReal.ofReal_mul (by positivity)]
      apply ENNReal.ofReal_le_ofReal
      change C * Real.sqrt (B k) * (A * (2 * a) + D) ≤
        C * (2 * A + 1) * Real.sqrt (B k) * (a + D)
      have hle : A * (2 * a) + D ≤ (2 * A + 1) * (a + D) := by
        nlinarith [mul_nonneg hA.le hD]
      convert mul_le_mul_of_nonneg_left hle
        (by positivity : 0 ≤ C * Real.sqrt (B k)) using 1
      ring

example := eventually_proxyKR_T_le isRegime_succ isSlowGrowth_sqrt_log_log
/-- Away from the unit interval, the shifted proxy dominates a positive
multiple of `√β exp(-γ R t²)`.
Source: arXiv:2412.09080v3, §2.3, `lem:phi-t`, `lem:main-int-phi`.
The shift is included in `γ`, with no bound on `n/β^{5/2}`. -/
theorem eventually_inner_ge_T {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) {ω : ℝ → ℝ} (hω : IsSlowGrowth ω) :
    ∃ C γ : ℝ, 0 < C ∧ 0 < γ ∧
      ∀ᶠ k in atTop, ∀ t ∈ intervalT (N k) (B k) (ω (B k)), 1 ≤ t ^ 2 →
        ENNReal.ofReal (C * Real.sqrt (B k)) *
          ENNReal.ofReal (Real.exp (-γ * phiRate (N k) (B k) t)) ≤
            ∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal
              (y * sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * krPhi (N k) (B k) t 0 y) := by
  obtain ⟨A₁, A₂, -, hA₂, hA⟩ := phiA_isTheta hreg hω
  obtain ⟨L, U, hL, -, hα⟩ := phiAlpha_isTheta hreg hω
  obtain ⟨X₁, X₂, hX₁, -, hX⟩ := eventually_det_inv_sqrt_mul_alpha_inv_bounds hreg hω
  obtain ⟨Cδ, hCδ, hδ⟩ := eventually_shift_sq_le_rateScale hreg hω
  refine ⟨X₁ / (4 * π), A₂ / 2 + Cδ, by positivity, by positivity, ?_⟩
  filter_upwards [hA, hα, hX, hδ, eventually_sigmaDet_pos hreg hω]
    with k hAk hαk hXk hδk hDk t ht ht1
  have hβ := hreg.B_pos k
  have hb : 0 < B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by positivity
  have ha : 0 < phiAlpha (B k) t := lt_of_lt_of_le (mul_pos hL hb) (hαk t ht).1
  have hd : 0 < sigmaDet (B k) t := hDk t ht
  have hf : 0 < sigmaDet (B k) t ^ (-(1 : ℝ) / 2) := Real.rpow_pos_of_pos hd _
  have hR0 : 0 ≤ rateScale (N k) (B k) t := by unfold rateScale; positivity
  have hr : rateScale (N k) (B k) t ≤ phiRate (N k) (B k) t := by
    have := mul_le_mul_of_nonneg_left ht1 hR0
    simpa only [rateScale, phiRate, mul_assoc, mul_comm, mul_left_comm, one_mul] using this
  have hδ' : phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2 ≤
      Cδ * phiRate (N k) (B k) t :=
    (hδk t ht).trans (mul_le_mul_of_nonneg_left hr hCδ.le)
  have hmoment := (integral_mul_exp_neg_shift_sq_bounds ha hδ').1
  have hJ : (∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y) =
      ((2 * π)⁻¹ * Real.exp (-phiA (N k) (B k) t / 2)) *
        ∫ y in Set.Ioi (0 : ℝ), y *
          Real.exp (-(phiAlpha (B k) t / 2) * (y - phiDelta (N k) (B k) t) ^ 2) := by
    simp_rw [mul_krPhi_eq ha]
    exact integral_const_mul _ _
  have hAupper : phiA (N k) (B k) t ≤ A₂ * phiRate (N k) (B k) t := (hAk t ht).2
  have hE : Real.exp (-(A₂ / 2 + Cδ) * phiRate (N k) (B k) t) ≤
      Real.exp (-phiA (N k) (B k) t / 2 - Cδ * phiRate (N k) (B k) t) :=
    Real.exp_le_exp.mpr (by nlinarith)
  rw [lintegral_det_krPhi_eq ha hd, ← ENNReal.ofReal_mul hf.le,
    ← ENNReal.ofReal_mul (by positivity)]
  apply ENNReal.ofReal_le_ofReal
  calc
    (X₁ / (4 * π) * Real.sqrt (B k)) *
        Real.exp (-(A₂ / 2 + Cδ) * phiRate (N k) (B k) t)
        ≤ (X₁ * Real.sqrt (B k)) / (4 * π) *
          Real.exp (-phiA (N k) (B k) t / 2 - Cδ * phiRate (N k) (B k) t) := by
            convert mul_le_mul_of_nonneg_left hE
              (by positivity : 0 ≤ X₁ * Real.sqrt (B k) / (4 * π)) using 1
            ring
    _ ≤ (sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (phiAlpha (B k) t)⁻¹) / (4 * π) *
          Real.exp (-phiA (N k) (B k) t / 2 - Cδ * phiRate (N k) (B k) t) := by
            exact mul_le_mul_of_nonneg_right
              (div_le_div_of_nonneg_right (hXk t ht).1 (by positivity)) (Real.exp_pos _).le
    _ = sigmaDet (B k) t ^ (-(1 : ℝ) / 2) *
        (((2 * π)⁻¹ * Real.exp (-phiA (N k) (B k) t / 2)) *
          (Real.exp (-(Cδ * phiRate (N k) (B k) t)) * (2 * phiAlpha (B k) t)⁻¹)) := by
            rw [show -phiA (N k) (B k) t / 2 - Cδ * phiRate (N k) (B k) t =
              -phiA (N k) (B k) t / 2 + -(Cδ * phiRate (N k) (B k) t) by ring,
              Real.exp_add]
            field_simp
            norm_num
    _ ≤ sigmaDet (B k) t ^ (-(1 : ℝ) / 2) *
        (((2 * π)⁻¹ * Real.exp (-phiA (N k) (B k) t / 2)) *
          ∫ y in Set.Ioi (0 : ℝ), y *
            Real.exp (-(phiAlpha (B k) t / 2) * (y - phiDelta (N k) (B k) t) ^ 2)) := by
              gcongr
    _ = _ := by rw [hJ]

example := eventually_inner_ge_T isRegime_succ isSlowGrowth_sqrt_log_log
end Transformer.Modes
