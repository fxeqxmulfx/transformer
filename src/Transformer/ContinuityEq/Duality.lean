/-
# Uniqueness for the linear continuity equation by duality

Let `w` be an `IsFlowField` whose spatial derivative is jointly continuous,
and let `t ↦ μ_t` solve `∂_t μ + div(w μ) = 0` weakly on `[0, T]`
(`ContinuityEq/Weak.lean`).  Then `μ_t` is the image of `μ_0` under the flow,
`μ_t = (Φ_{0,t})_# μ_0` (`eq_map_flow`).

The proof is by duality.  For `g ∈ C_c^∞` and `t ∈ [0, T]`, the function
`F(s) = ∫ g(Φ_{s,t} x) dμ_s(x)` is continuous.  Near `s < t` it splits as
`∫ (h(Φ_{s',s} x) - h(x)) dμ_{s'} + ∫ h dμ_{s'}` with `h = g ∘ Φ_{s,t} ∈ C¹_c`.
The first term has derivative `-∫ Dh · w(s) dμ_s`, by the expansion of the
flow in the initial time (`hasDerivAt_integral_comp_flow`); the second has
right derivative `∫ Dh · w(s) dμ_s`, by the weak identity for `h`.  So `F` is
constant, and `∫ g dμ_t = F(t) = F(0) = ∫ g ∘ Φ_{0,t} dμ_0`.

This is the uniqueness half of the theory of the linear continuity equation
with a `C¹` field, which the proof of `c:wellposedtransformers` of
arXiv:2305.05465v6, §6, uses.
-/

import Transformer.ContinuityEq.Weak

open Set Metric Filter Topology MeasureTheory Asymptotics
open scoped NNReal

namespace Transformer
namespace ContinuityEq

open SmoothApprox GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]
  {w : ℝ → E → E} {L : ℝ≥0} {M : ℝ} {μ : ℝ → Measure E} [∀ t, IsProbabilityMeasure (μ t)]

/-- **Transporting a test function by the flow.**  For `h ∈ C¹_c`,
`s' ↦ ∫ (h(Φ_{s',s} x) - h(x)) dμ_{s'}` has derivative `-∫ Dh(x) w(s, x) dμ_s`
at `s`: `h(Φ_{s',s} x) - h(x) = (s - s') Dh(x) w(s, x) + o(s' - s)`, uniformly
on the ball carrying the measures near `s`. -/
theorem hasDerivAt_integral_comp_flow (hw : IsFlowField w L M)
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    {h : E → ℝ} (hh : ContDiff ℝ 1 h) (hc : HasCompactSupport h) (s : ℝ) :
    HasDerivAt (fun s' => ∫ x, (h (flow w s' s x) - h x) ∂μ s')
      (-∫ x, fderiv ℝ h x (w s x) ∂μ s) s := by
  have hM := hw.nonneg
  obtain ⟨R, hRs⟩ := hR s
  have hDh : Continuous (fderiv ℝ h) := hh.continuous_fderiv one_ne_zero
  obtain ⟨B, hB⟩ := hDh.bounded_above_of_compact_support (hc.fderiv (𝕜 := ℝ))
  have hB0 : 0 ≤ B := (norm_nonneg _).trans (hB 0)
  have hDw : Continuous fun x => fderiv ℝ h x (w s x) :=
    hDh.clm_apply (hw.lipschitz s).continuous
  have hcB : Continuous fun s' => ∫ x, fderiv ℝ h x (w s x) ∂μ s' :=
    continuous_integral_param hR hμ continuous_id (f := fun _ x => fderiv ℝ h x (w s x))
      (hDw.comp continuous_snd)
  rw [hasDerivAt_iff_isLittleO, isLittleO_iff]
  intro c hc
  obtain ⟨δ, hδ, hhδ⟩ := norm_sub_sub_le_of_uniform (f := fun _ : ℝ => h)
    (f' := fun _ => fderiv ℝ h) (hDh.comp continuous_snd)
    (fun _ x => (hh.differentiable one_ne_zero x).hasFDerivAt)
    (isCompact_singleton (x := (0 : ℝ))) (isCompact_closedBall (0 : E) (R + M))
    (convex_closedBall 0 (R + M)) (ε := c / (4 * (M + 1))) (by positivity)
  filter_upwards [hRs, hw.flow_expansion s R (ε := c / (4 * (B + 1))) (by positivity),
    Metric.continuousAt_iff'.1 hcB.continuousAt (c / 2) (by positivity),
    ball_mem_nhds s (lt_min one_pos (div_pos hδ (by linarith : (0 : ℝ) < M + 1)))]
    with s' hR' hexp hcont hs'
  rw [mem_ball, Real.dist_eq, lt_min_iff] at hs'
  have hpt : ∀ x ∈ closedBall (0 : E) R,
      ‖h (flow w s' s x) - h x + (s' - s) * fderiv ℝ h x (w s x)‖ ≤ c / 2 * |s' - s| := by
    intro x hx
    have hu : ‖flow w s' s x - x‖ ≤ M * |s' - s| := by
      have := hw.norm_flow_sub_le s' s x
      rwa [abs_sub_comm] at this
    rw [mem_closedBall_zero_iff] at hx
    have hxM : x ∈ closedBall (0 : E) (R + M) := mem_closedBall_zero_iff.2 (by linarith)
    have hΦM : flow w s' s x ∈ closedBall (0 : E) (R + M) := by
      rw [mem_closedBall_zero_iff]
      have : M * |s' - s| ≤ M := mul_le_of_le_one_right hM hs'.1.le
      linarith [norm_le_norm_add_norm_sub' (flow w s' s x) x]
    have hlt : ‖flow w s' s x - x‖ < δ := by
      have := (lt_div_iff₀ (by linarith : (0 : ℝ) < M + 1)).1 hs'.2
      nlinarith [abs_nonneg (s' - s)]
    have e1 := hhδ 0 (mem_singleton 0) x hxM _ hΦM hlt
    beta_reduce at e1
    have e3 : ‖fderiv ℝ h x (flow w s' s x - x - (s - s') • w s x)‖
        ≤ B * (c / (4 * (B + 1)) * |s' - s|) := by
      refine ((fderiv ℝ h x).le_opNorm _).trans (mul_le_mul (hB x) ?_ (norm_nonneg _) hB0)
      rw [abs_sub_comm]
      exact hexp x (mem_closedBall_zero_iff.2 hx)
    have e4 : h (flow w s' s x) - h x + (s' - s) * fderiv ℝ h x (w s x) =
        (h (flow w s' s x) - h x - fderiv ℝ h x (flow w s' s x - x)) +
          fderiv ℝ h x (flow w s' s x - x - (s - s') • w s x) := by
      simp only [map_sub, map_smul, smul_eq_mul]
      ring
    have hc' := mul_nonneg hc.le (abs_nonneg (s' - s))
    have k1 : c / (4 * (M + 1)) * ‖flow w s' s x - x‖ ≤ c / 4 * |s' - s| := by
      have hM1 : (0 : ℝ) < 4 * (M + 1) := by linarith
      calc c / (4 * (M + 1)) * ‖flow w s' s x - x‖ ≤ c / (4 * (M + 1)) * (M * |s' - s|) :=
            mul_le_mul_of_nonneg_left hu (div_nonneg hc.le hM1.le)
        _ = (c * M * |s' - s|) / (4 * (M + 1)) := by ring
        _ ≤ c / 4 * |s' - s| := by
            rw [div_le_iff₀ hM1]
            nlinarith
    have k2 : B * (c / (4 * (B + 1)) * |s' - s|) ≤ c / 4 * |s' - s| := by
      have hB1 : (0 : ℝ) < 4 * (B + 1) := by linarith
      rw [show B * (c / (4 * (B + 1)) * |s' - s|) = (B * c * |s' - s|) / (4 * (B + 1)) by ring,
        div_le_iff₀ hB1]
      nlinarith
    rw [e4]
    refine (norm_add_le _ _).trans ?_
    linarith
  have hint1 : Integrable (fun x => h (flow w s' s x) - h x) (μ s') :=
    integrable_of_carried hR' ((hh.continuous.comp (hw.continuous_flow s' s)).sub hh.continuous)
  have hint2 : Integrable (fun x => fderiv ℝ h x (w s x)) (μ s') := integrable_of_carried hR' hDw
  have hI : ‖∫ x, (h (flow w s' s x) - h x + (s' - s) * fderiv ℝ h x (w s x)) ∂μ s'‖
      ≤ c / 2 * |s' - s| := by
    have := norm_integral_le_of_norm_le_const (μ := μ s') (C := c / 2 * |s' - s|)
      (f := fun x => h (flow w s' s x) - h x + (s' - s) * fderiv ℝ h x (w s x)) (by
        filter_upwards [(ae_iff.2 hR' : ∀ᵐ x ∂μ s', x ∈ closedBall (0 : E) R)] with x hx
        exact hpt x hx)
    rwa [probReal_univ, mul_one] at this
  rw [integral_add hint1 (hint2.const_mul _), integral_const_mul] at hI
  have h0 : ∫ x, (h (flow w s s x) - h x) ∂μ s = 0 := by simp [hw.flow_self]
  rw [h0, sub_zero, smul_eq_mul, Real.norm_eq_abs, Real.norm_eq_abs]
  rw [Real.norm_eq_abs] at hI
  rw [Real.dist_eq] at hcont
  set I₁ := ∫ x, (h (flow w s' s x) - h x) ∂μ s'
  set B' := ∫ x, fderiv ℝ h x (w s x) ∂μ s'
  set B₀ := ∫ x, fderiv ℝ h x (w s x) ∂μ s
  calc |I₁ - (s' - s) * -B₀| = |(I₁ + (s' - s) * B') - (s' - s) * (B' - B₀)| := by
        congr 1; ring
    _ ≤ |I₁ + (s' - s) * B'| + |(s' - s) * (B' - B₀)| := abs_sub _ _
    _ ≤ c / 2 * |s' - s| + |s' - s| * (c / 2) := by
        rw [abs_mul]
        gcongr
    _ = c * |s' - s| := by ring

/-- **The dual problem is stationary from the right.**  For `g ∈ C_c^∞` and
`0 ≤ s < t ≤ T`, `s' ↦ ∫ g(Φ_{s',t} x) dμ_{s'}` has right derivative `0` at
`s`: with `h = g ∘ Φ_{s,t}`, it is `∫ (h(Φ_{s',s} x) - h(x)) dμ_{s'}` plus
`∫ h dμ_{s'}`, whose derivatives cancel. -/
theorem hasDerivWithinAt_integral_comp_flow_zero (hw : IsFlowField w L M)
    {A : ℝ → E → E →L[ℝ] E} (hA : Continuous (Function.uncurry A))
    (hwA : ∀ t x, HasFDerivAt (w t) (A t x) x)
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    {T : ℝ} (hweak : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g → ∀ t ∈ Icc 0 T,
      ∫ x, g x ∂μ t = ∫ x, g x ∂μ 0 + ∫ σ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x (w σ x) ∂μ σ)
    {g : E → ℝ} (hg : ContDiff ℝ (⊤ : ℕ∞) g) (hgc : HasCompactSupport g) {s t : ℝ}
    (hs : 0 ≤ s) (hst : s < t) (htT : t ≤ T) :
    HasDerivWithinAt (fun s' => ∫ x, g (flow w s' t x) ∂μ s') 0 (Ici s) s := by
  set h := g ∘ flow w s t with hdef
  have hh : ContDiff ℝ 1 h :=
    (hg.of_le (by exact_mod_cast le_top)).comp (hw.contDiff_flow hA hwA hst.le)
  have hhc : HasCompactSupport h := hw.hasCompactSupport_comp_flow hgc s t
  have hG := continuous_integral_fderiv hR hμ hw.continuous hh
  have h1 := (hasDerivAt_integral_comp_flow hw hR hμ hh hhc s).hasDerivWithinAt (s := Ici s)
  have h2 : HasDerivWithinAt (fun s' => ∫ x, h x ∂μ s') (∫ x, fderiv ℝ h x (w s x) ∂μ s)
      (Ici s) s := by
    refine (((hG.integral_hasStrictDerivAt s s).hasDerivAt.const_add
      (∫ x, h x ∂μ s)).hasDerivWithinAt).congr_of_eventuallyEq ?_ (by simp)
    filter_upwards [Ico_mem_nhdsGE (hst.trans_le htT)] with s' hs'
    rw [← weak_sub_of_contDiff_one hR hμ hw hweak hh hhc ⟨hs, hst.le.trans htT⟩
      ⟨hs.trans hs'.1, hs'.2.le⟩]
    ring
  have hsplit : ∀ s', ∫ x, g (flow w s' t x) ∂μ s' =
      ∫ x, (h (flow w s' s x) - h x) ∂μ s' + ∫ x, h x ∂μ s' := fun s' => by
    have hi : Integrable (fun x => h (flow w s' s x)) (μ s') :=
      (hh.continuous.comp (hw.continuous_flow s' s)).integrable_of_hasCompactSupport
        (hw.hasCompactSupport_comp_flow hhc s' s)
    rw [integral_sub hi (hh.continuous.integrable_of_hasCompactSupport hhc), sub_add_cancel]
    simp only [hdef, Function.comp, hw.flow_flow]
  have h3 := h1.add h2
  rw [neg_add_cancel] at h3
  exact h3.congr (fun s' _ => hsplit s') (hsplit s)

/-- The hypotheses of the two derivative theorems are satisfiable: the zero
field, with derivative `0`, the constant curve `δ_0`, which solves its
continuity equation on `[0, 1]`, the test function `0`, and `0 ≤ 0 < 1 ≤ 1`. -/
example : IsFlowField (fun (_ : ℝ) (_ : E) => (0 : E)) 0 0 ∧
    Continuous (Function.uncurry fun (_ : ℝ) (_ : E) => (0 : E →L[ℝ] E)) ∧
    (∀ t x, HasFDerivAt ((fun (_ : ℝ) (_ : E) => (0 : E)) t) (0 : E →L[ℝ] E) x) ∧
    (∀ t₀ : ℝ, ∃ R : ℝ, ∀ᶠ _ in 𝓝 t₀, Measure.dirac (0 : E) (closedBall 0 R)ᶜ = 0) ∧
    (∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous (fun (_ : ℝ) => ∫ x, g x ∂(Measure.dirac (0 : E))) ∧ ∀ t ∈ Icc (0 : ℝ) 1,
        ∫ x, g x ∂(Measure.dirac (0 : E)) = ∫ x, g x ∂(Measure.dirac (0 : E)) +
          ∫ _ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x 0 ∂(Measure.dirac (0 : E))) ∧
    ContDiff ℝ (⊤ : ℕ∞) (0 : E → ℝ) ∧ HasCompactSupport (0 : E → ℝ) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 :=
  ⟨⟨continuous_const, fun _ => LipschitzWith.const 0, fun _ _ => by simp⟩, continuous_const,
    fun _ _ => hasFDerivAt_const 0 _, fun _ => ⟨0, Eventually.of_forall fun _ => by simp⟩,
    fun _ _ _ => ⟨continuous_const, fun _ _ => by simp⟩, contDiff_const, HasCompactSupport.zero,
    le_rfl, one_pos, le_rfl⟩

end ContinuityEq
end Transformer
