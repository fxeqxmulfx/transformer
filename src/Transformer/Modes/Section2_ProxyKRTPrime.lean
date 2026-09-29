/-
# The number of modes of a Gaussian KDE — the inner Kac–Rice integral on `T'`

`lem:main-int-phi` on `T'`, the inner integral.  On `T'` the source proof of
`eq:int-phi-final` is not available: it rests on `lem:phi-t`, whose second
display needs `α_t δ_t² = O(1)`, i.e. `n ≲ β^{5/2}`, which the regime does not
give (`Section2_PhiTDelta.lean`).  Here the shift is handled directly.  For
`t ∈ T'` (which lies in `T` for large `k`, `Section2_IntervalTPrime.lean`), with
`R = β^{-3/2} n e^{-t²/2}`:

* `(det Σ_t)^{-1/2} α_t⁻¹ ≲ √β`, `A_t ≳ R t²`, from the estimates on `T`;
* `α_t δ_t² ≲ R (4 + 6t²)²` (`phiAlpha_mul_phiDelta_sq_le_rate`), without a bound
  on `n`;
* `∫_0^∞ y φ dy ≤ e^{-A_t/2} α_t⁻¹ (1 + 3|δ_t|√α_t)`
  (`integral_mul_krPhi_le`), linear in the shift.

Together: the inner integral is at most `C √β e^{-κ R t²}(1 + K √R (4 + 6t²))`.
The remaining `t`-integral is `Section2_WidthIntegral.lean`.

Source: arXiv:2412.09080v3, `lem:main-int-phi`, `eq:int-phi-final`, `lem:phi-t`.
-/

import Transformer.Modes.Section2_ProxyKRInner
import Transformer.Modes.Section2_PhiTDeltaRate
import Transformer.Modes.Section2_IntervalTPrime

open Real MeasureTheory Filter
open scoped ENNReal Topology

namespace Transformer
namespace Modes

/-- **The inner Kac–Rice integral on `T'`.**  Uniformly for `t ∈ T'`, for all large
`k`, `∫_0^∞ y (det Σ_t)^{-1/2} φ(…) dy ≤ C √β · e^{-κ R t²}(1 + K √R (4 + 6t²))`
with `R = β^{-3/2} n e^{-t²/2}`; no bound on `n` is needed.

Source: arXiv:2412.09080v3, `lem:main-int-phi`, `eq:int-phi-final`. -/
theorem eventually_inner_le_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ} (hreg : IsRegime c N B) :
    ∃ C κ K₁ : ℝ, 0 < C ∧ 0 < κ ∧ 0 ≤ K₁ ∧ ∀ᶠ k in atTop, ∀ t ∈ intervalT' (N k) (B k),
      (∫⁻ y in Set.Ioi (0 : ℝ), ENNReal.ofReal
          (y * sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * krPhi (N k) (B k) t 0 y)) ≤
        ENNReal.ofReal (C * Real.sqrt (B k)) *
          ENNReal.ofReal (Real.exp (-κ * (rateScale (N k) (B k) t * t ^ 2)) *
            (1 + K₁ * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2))) := by
  have hω := isSlowGrowth_sqrt_log_log
  obtain ⟨A₁, A₂, hA₁, -, hA⟩ := phiA_isTheta hreg hω
  obtain ⟨L, U, hL, -, hα⟩ := phiAlpha_isTheta hreg hω
  obtain ⟨X₁, X₂, -, hX₂, hX⟩ := eventually_det_inv_sqrt_mul_alpha_inv_bounds hreg hω
  obtain ⟨Cδ, hCδ, hδ⟩ := phiAlpha_mul_phiDelta_sq_le_rate hreg hω
  refine ⟨X₂, A₁ / 2, 3 * Real.sqrt Cδ, hX₂, by positivity, by positivity, ?_⟩
  filter_upwards [hA, hα, hX, hδ, eventually_sigmaDet_pos hreg hω,
    eventually_intervalT'_subset hreg, hreg.tendsto_B.eventually_ge_atTop 1]
    with k hAk hαk hXk hδk hDk hsub hB1 t ht'
  have ht := hsub ht'
  have hβ := hreg.B_pos k
  have hb : 0 < B k ^ ((1 : ℝ) / 2) * Real.exp (t ^ 2 / 2) := by positivity
  have hα0 : 0 < phiAlpha (B k) t := lt_of_lt_of_le (mul_pos hL hb) (hαk t ht).1
  have hD0 : 0 < sigmaDet (B k) t := hDk t ht
  have hf : 0 < sigmaDet (B k) t ^ (-(1 : ℝ) / 2) := Real.rpow_pos_of_pos hD0 _
  have hR0 : 0 ≤ rateScale (N k) (B k) t := by unfold rateScale; positivity
  -- `n β^{-5/2} e^{-t²/2} ≤ R`, since `β ≥ 1`
  have hRle : (N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2) ≤
      rateScale (N k) (B k) t := by
    unfold rateScale
    have h1 : B k ^ (-(5 : ℝ) / 2) ≤ B k ^ (-(3 : ℝ) / 2) :=
      Real.rpow_le_rpow_of_exponent_le hB1 (by norm_num)
    calc (N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2)
        ≤ (N k : ℝ) * B k ^ (-(3 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2) := by gcongr
      _ = B k ^ (-(3 : ℝ) / 2) * N k * Real.exp (-(t ^ 2) / 2) := by ring
  -- the size of the shift: `|δ_t| √α_t ≤ √C_δ √R (4 + 6t²)`
  have hs : |phiDelta (N k) (B k) t| * Real.sqrt (phiAlpha (B k) t) ≤
      Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2) := by
    have hs2 : (|phiDelta (N k) (B k) t| * Real.sqrt (phiAlpha (B k) t)) ^ 2 =
        phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2 := by
      rw [mul_pow, sq_abs, Real.sq_sqrt hα0.le]; ring
    have hz : 0 ≤ Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2) := by
      positivity
    have hz2 : (Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2)) ^ 2 =
        Cδ * rateScale (N k) (B k) t * (4 + 6 * t ^ 2) ^ 2 := by
      rw [mul_pow, mul_pow, Real.sq_sqrt hCδ.le, Real.sq_sqrt hR0]
    refine (sq_le_sq₀ (by positivity) hz).1 ?_
    rw [hs2, hz2]
    calc phiAlpha (B k) t * phiDelta (N k) (B k) t ^ 2
        ≤ Cδ * ((N k : ℝ) * B k ^ (-(5 : ℝ) / 2) * Real.exp (-(t ^ 2) / 2))
          * (4 + 6 * t ^ 2) ^ 2 := hδk t ht
      _ ≤ Cδ * rateScale (N k) (B k) t * (4 + 6 * t ^ 2) ^ 2 := by gcongr
  -- `e^{-A_t/2} ≤ e^{-(A₁/2) R t²}`
  have hAle : Real.exp (-phiA (N k) (B k) t / 2) ≤
      Real.exp (-(A₁ / 2) * (rateScale (N k) (B k) t * t ^ 2)) := by
    refine Real.exp_le_exp.mpr ?_
    have h := (hAk t ht).1
    have hrs : rateScale (N k) (B k) t * t ^ 2 =
        B k ^ (-(3 : ℝ) / 2) * N k * t ^ 2 * Real.exp (-(t ^ 2) / 2) := by
      unfold rateScale; ring
    rw [hrs]
    linarith
  -- assemble
  rw [lintegral_det_krPhi_eq hα0 hD0, ← ENNReal.ofReal_mul hf.le,
    ← ENNReal.ofReal_mul (by positivity)]
  refine ENNReal.ofReal_le_ofReal ?_
  have hJ := integral_mul_krPhi_le (n := N k) hα0
  have hx : sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (phiAlpha (B k) t)⁻¹ ≤ X₂ * Real.sqrt (B k) :=
    (hXk t ht).2
  set s := |phiDelta (N k) (B k) t| * Real.sqrt (phiAlpha (B k) t) with hs_def
  have hs0 : 0 ≤ s := by positivity
  calc sigmaDet (B k) t ^ (-(1 : ℝ) / 2) *
        (∫ y in Set.Ioi (0 : ℝ), y * krPhi (N k) (B k) t 0 y)
      ≤ sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (Real.exp (-phiA (N k) (B k) t / 2) *
          ((phiAlpha (B k) t)⁻¹ * (1 + 3 * s))) := mul_le_mul_of_nonneg_left hJ hf.le
    _ = (sigmaDet (B k) t ^ (-(1 : ℝ) / 2) * (phiAlpha (B k) t)⁻¹) *
          Real.exp (-phiA (N k) (B k) t / 2) * (1 + 3 * s) := by ring
    _ ≤ (X₂ * Real.sqrt (B k)) * Real.exp (-(A₁ / 2) * (rateScale (N k) (B k) t * t ^ 2)) *
          (1 + 3 * (Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2))) := by
        refine mul_le_mul (mul_le_mul hx hAle (Real.exp_pos _).le (by positivity)) ?_
          (by positivity) (by positivity)
        linarith
    _ = X₂ * Real.sqrt (B k) * (Real.exp (-(A₁ / 2) * (rateScale (N k) (B k) t * t ^ 2)) *
          (1 + 3 * Real.sqrt Cδ * Real.sqrt (rateScale (N k) (B k) t) * (4 + 6 * t ^ 2))) := by
        ring

/-- The regime hypothesis of `eventually_inner_le_T'` is satisfiable. -/
example := eventually_inner_le_T' isRegime_succ

end Modes
end Transformer
