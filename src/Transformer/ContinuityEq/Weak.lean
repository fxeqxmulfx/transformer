/-
# The weak continuity equation against `C¹` test functions

A curve `t ↦ μ_t` of probability measures on a finite-dimensional space solves
the linear continuity equation `∂_t μ + div(w μ) = 0` weakly on `[0, T]` when

  `∫ g dμ_t = ∫ g dμ_0 + ∫_0^t ∫ Dg(x) w(σ, x) dμ_σ(x) dσ`

for every smooth `g` with compact support and every `t ∈ [0, T]`.  If the
measures live near each time in a fixed ball and are weakly continuous, the
identity extends to `C¹` test functions with compact support
(`weak_of_contDiff_one`), and between any two times of `[0, T]`
(`weak_sub_of_contDiff_one`): approximating `h` in `C¹` by smooth `g`
(`SmoothApprox.exists_contDiff_near_fderiv`) moves each term by `O(ε)`.

The duality proof of uniqueness for `c:wellposedtransformers` of
arXiv:2305.05465v6, §6, tests the equation against `h ∘ Φ`, with `Φ` the
flow of `w`, which is only `C¹`.
-/

import Transformer.GlobalFlow.Variational
import Transformer.SmoothApprox
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

open Set Metric Filter Topology MeasureTheory
open scoped NNReal

namespace Transformer
namespace ContinuityEq

open SmoothApprox

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]

/-- A continuous function is integrable against a finite measure carried by a
ball, on which it is bounded. -/
theorem integrable_of_carried {F : Type*} [NormedAddCommGroup F] {μ : Measure E}
    [IsFiniteMeasure μ] {R : ℝ} (hR : μ (closedBall 0 R)ᶜ = 0) {f : E → F} (hf : Continuous f) :
    Integrable f μ := by
  obtain ⟨C, hC⟩ := (isCompact_closedBall (0 : E) R).exists_bound_of_continuousOn hf.continuousOn
  refine Integrable.of_bound hf.aestronglyMeasurable C ?_
  filter_upwards [(ae_iff.2 hR : ∀ᵐ x ∂μ, x ∈ closedBall (0 : E) R)] with x hx
  exact hC x hx

/-- A continuous function is integrable against a finite measure on `E × E`
whose two coordinates lie in `B̄(0, R)` almost everywhere. -/
theorem integrable_of_ae_norm_le {F : Type*} [NormedAddCommGroup F] {γ : Measure (E × E)}
    [IsFiniteMeasure γ] {R : ℝ} (hγ : ∀ᵐ p ∂γ, ‖p.1‖ ≤ R ∧ ‖p.2‖ ≤ R) {f : E × E → F}
    (hf : Continuous f) : Integrable f γ :=
  integrable_of_carried (R := R)
    (ae_iff.1 (hγ.mono fun _ hp => mem_closedBall_zero_iff.2 (max_le hp.1 hp.2))) hf

variable {w : ℝ → E → E} {μ : ℝ → Measure E} [∀ t, IsProbabilityMeasure (μ t)]

/-- The `μ_σ`-average of `Dh · w(σ, ·)` is continuous in `σ`. -/
theorem continuous_integral_fderiv
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    (hw : Continuous (Function.uncurry w)) {h : E → ℝ} (hh : ContDiff ℝ 1 h) :
    Continuous fun σ => ∫ x, fderiv ℝ h x (w σ x) ∂μ σ :=
  continuous_integral_param hR hμ continuous_id (f := fun σ x => fderiv ℝ h x (w σ x))
    (((hh.continuous_fderiv one_ne_zero).comp continuous_snd).clm_apply hw)

/-- **The weak identity holds for `C¹` test functions with compact support.**
For `g ∈ C_c^∞` within `ε` of `h` in `C¹`, the three terms of the identity
move by at most `ε`, `ε` and `t M ε`. -/
theorem weak_of_contDiff_one
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    {L : ℝ≥0} {M : ℝ} (hw : GlobalFlow.IsFlowField w L M) {T : ℝ}
    (hweak : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g → ∀ t ∈ Icc 0 T,
      ∫ x, g x ∂μ t = ∫ x, g x ∂μ 0 + ∫ σ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x (w σ x) ∂μ σ)
    {h : E → ℝ} (hh : ContDiff ℝ 1 h) (hc : HasCompactSupport h) {t : ℝ} (ht : t ∈ Icc 0 T) :
    ∫ x, h x ∂μ t = ∫ x, h x ∂μ 0 + ∫ σ in (0 : ℝ)..t, ∫ x, fderiv ℝ h x (w σ x) ∂μ σ := by
  refine eq_of_forall_dist_le fun ε hε => ?_
  have hM := hw.nonneg
  have hD : 0 < 2 + t * M + 1 := by nlinarith [ht.1]
  set ε' := ε / (2 + t * M + 1)
  have hε' : 0 < ε' := div_pos hε hD
  obtain ⟨g, hg, hgc, hgh⟩ := exists_contDiff_near_fderiv hh hc hε'
  have hg1 : ContDiff ℝ 1 g := hg.of_le (by exact_mod_cast le_top)
  have hI : ∀ σ {f : E → ℝ}, Continuous f → Integrable f (μ σ) := fun σ f hf => by
    obtain ⟨R, hRσ⟩ := hR σ
    exact integrable_of_carried hRσ.self_of_nhds hf
  have hDc : ∀ {f : E → ℝ}, ContDiff ℝ 1 f → ∀ σ, Continuous fun x => fderiv ℝ f x (w σ x) :=
    fun hf σ => (hf.continuous_fderiv one_ne_zero).clm_apply (hw.lipschitz σ).continuous
  have hval : ∀ σ, ‖∫ x, h x ∂μ σ - ∫ x, g x ∂μ σ‖ ≤ ε' := fun σ => by
    rw [norm_sub_rev]
    exact norm_integral_sub_le_of_forall_le (hI σ hh.continuous) (hI σ hg.continuous)
      fun x => (hgh x).1
  have hder : ∀ σ, ‖∫ x, fderiv ℝ h x (w σ x) ∂μ σ - ∫ x, fderiv ℝ g x (w σ x) ∂μ σ‖
      ≤ ε' * M := fun σ => by
    rw [norm_sub_rev]
    refine norm_integral_sub_le_of_forall_le (hI σ (hDc hh σ)) (hI σ (hDc hg1 σ)) fun x => ?_
    show ‖fderiv ℝ g x (w σ x) - fderiv ℝ h x (w σ x)‖ ≤ ε' * M
    rw [← sub_apply]
    exact ((fderiv ℝ g x - fderiv ℝ h x).le_opNorm _).trans
      (mul_le_mul (hgh x).2 (hw.bound σ x) (norm_nonneg _) hε'.le)
  have hGh := continuous_integral_fderiv hR hμ hw.continuous hh
  have hGg := continuous_integral_fderiv hR hμ hw.continuous hg1
  have hint : ‖(∫ σ in (0 : ℝ)..t, ∫ x, fderiv ℝ h x (w σ x) ∂μ σ) -
      ∫ σ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x (w σ x) ∂μ σ‖ ≤ ε' * M * t := by
    rw [← intervalIntegral.integral_sub (hGh.intervalIntegrable _ _) (hGg.intervalIntegrable _ _)]
    refine (intervalIntegral.norm_integral_le_of_norm_le_const fun σ _ => hder σ).trans ?_
    rw [sub_zero, abs_of_nonneg ht.1]
  have e := hweak g hg hgc t ht
  have h1 := hval t
  have h2 := hval 0
  have h3 := hint
  rw [Real.norm_eq_abs, abs_le] at h1 h2 h3
  obtain ⟨h1l, h1r⟩ := h1
  obtain ⟨h2l, h2r⟩ := h2
  obtain ⟨h3l, h3r⟩ := h3
  have hCε : 2 * ε' + ε' * M * t ≤ ε := by
    have := div_mul_cancel₀ ε hD.ne'
    linarith
  rw [Real.dist_eq, abs_le]
  constructor <;> linarith

/-- **The weak identity between two times**, for `C¹` test functions with
compact support. -/
theorem weak_sub_of_contDiff_one
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    {L : ℝ≥0} {M : ℝ} (hw : GlobalFlow.IsFlowField w L M) {T : ℝ}
    (hweak : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g → ∀ t ∈ Icc 0 T,
      ∫ x, g x ∂μ t = ∫ x, g x ∂μ 0 + ∫ σ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x (w σ x) ∂μ σ)
    {h : E → ℝ} (hh : ContDiff ℝ 1 h) (hc : HasCompactSupport h) {s t : ℝ} (hs : s ∈ Icc 0 T)
    (ht : t ∈ Icc 0 T) :
    ∫ x, h x ∂μ t - ∫ x, h x ∂μ s = ∫ σ in s..t, ∫ x, fderiv ℝ h x (w σ x) ∂μ σ := by
  have hG := continuous_integral_fderiv hR hμ hw.continuous hh
  rw [weak_of_contDiff_one hR hμ hw hweak hh hc ht, weak_of_contDiff_one hR hμ hw hweak hh hc hs,
    add_sub_add_left_eq_sub,
    intervalIntegral.integral_interval_sub_left (hG.intervalIntegrable _ _) (hG.intervalIntegrable _ _)]

/-- The hypotheses of the results above are satisfiable: the constant curve
`δ_0`, carried by `B̄(0, 0)`, solves the equation for the zero field on `[0, 1]`,
`0` is a `C¹` test function with compact support, and `δ_{(0, 0)}` is carried
by `B̄(0, 0)²`. -/
example : GlobalFlow.IsFlowField (fun (_ : ℝ) (_ : E) => (0 : E)) 0 0 ∧
    Continuous (Function.uncurry fun (_ : ℝ) (_ : E) => (0 : E)) ∧
    Measure.dirac (0 : E) (closedBall 0 0)ᶜ = 0 ∧
    (∀ t₀ : ℝ, ∃ R : ℝ, ∀ᶠ _ in 𝓝 t₀, Measure.dirac (0 : E) (closedBall 0 R)ᶜ = 0) ∧
    (∀ᵐ p ∂Measure.dirac ((0 : E), (0 : E)), ‖p.1‖ ≤ 0 ∧ ‖p.2‖ ≤ 0) ∧
    (∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous (fun _ : ℝ => ∫ x, g x ∂(Measure.dirac (0 : E))) ∧ ∀ t ∈ Icc (0 : ℝ) 1,
        ∫ x, g x ∂(Measure.dirac (0 : E)) = ∫ x, g x ∂(Measure.dirac (0 : E)) +
          ∫ _ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x 0 ∂(Measure.dirac (0 : E))) ∧
    ContDiff ℝ 1 (0 : E → ℝ) ∧ HasCompactSupport (0 : E → ℝ) ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 :=
  ⟨⟨continuous_const, fun _ => LipschitzWith.const 0, fun _ _ => by simp⟩, continuous_const,
    by simp, fun _ => ⟨0, Eventually.of_forall fun _ => by simp⟩, by rw [ae_dirac_eq]; simp,
    fun g _ _ => ⟨continuous_const, fun _ _ => by simp⟩, contDiff_const, HasCompactSupport.zero,
    ⟨le_rfl, zero_le_one⟩⟩

end ContinuityEq
end Transformer
