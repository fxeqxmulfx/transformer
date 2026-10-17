/-
# Stability of transported measures in `W_2`

Let `w` and `u` be `IsFlowField`s on a finite-dimensional inner product
space, and push probability measures `μ₀`, `ν₀` carried by `B̄(0, R)` along
their flows: `μ_τ = (Φ^w_{0,τ})_# μ₀` and `ν_τ = (Φ^u_{0,τ})_# ν₀`.  If on
`[0, T]` the fields satisfy, where the trajectories can be,

  `‖w(τ, a) - u(τ, b)‖ ≤ K ‖a - b‖ + C W_2(μ_τ, ν_τ)`,

then `W_2(μ_t, ν_t) ≤ e^{(K + C) t} W_2(μ₀, ν₀)` (`W2_map_flow_le`).

For a coupling `γ` of `μ₀` and `ν₀`, the cost `P(τ) = ∫ ‖Φ^w_τ x - Φ^u_τ y‖² dγ`
of transporting it along the two flows bounds `W_2(μ_τ, ν_τ)²`
(`W2_map_le_sqrt`) and has derivative `2 ∫ ⟨Φ^w_τ x - Φ^u_τ y, w - u⟩ dγ`
(`hasDerivAt_cost`).  That is at most `2 (K + C) P(τ)`, because
`2 C ‖D‖ W ≤ C (‖D‖² + W²)` and `W² ≤ P`; Grönwall bounds `P(t)` by
`e^{2 (K + C) t} P(0)`, and the infimum over couplings gives the estimate.

This is the proof of `e:w2estimate` in arXiv:2305.05465v6, §6, once the
solutions are known to be transported by their flows.
-/

import Transformer.ContinuityEq.Weak
import Transformer.Wasserstein.Basic
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.ODE.Gronwall

open Set Metric Filter Topology MeasureTheory
open scoped NNReal

namespace Transformer
namespace ContinuityEq

open GlobalFlow Wasserstein

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  {w u : ℝ → E → E} {Lw Lu : ℝ≥0} {Mw Mu : ℝ}

/-- The integrand `‖Φ^w_τ x - Φ^u_τ y‖²` of the transported cost is
continuous. -/
theorem continuous_cost (hw : IsFlowField w Lw Mw) (hu : IsFlowField u Lu Mu) (τ : ℝ) :
    Continuous fun p : E × E => ‖flow w 0 τ p.1 - flow u 0 τ p.2‖ ^ 2 :=
  ((((hw.continuous_flow 0 τ).comp continuous_fst).sub
    ((hu.continuous_flow 0 τ).comp continuous_snd)).norm).pow 2

/-- The integrand of the derivative of the transported cost is continuous. -/
theorem continuous_cost_deriv (hw : IsFlowField w Lw Mw) (hu : IsFlowField u Lu Mu) (τ : ℝ) :
    Continuous fun p : E × E => 2 * inner (𝕜 := ℝ) (flow w 0 τ p.1 - flow u 0 τ p.2)
      (w τ (flow w 0 τ p.1) - u τ (flow u 0 τ p.2)) := by
  have h1 := (hw.continuous_flow 0 τ).comp (continuous_fst (X := E) (Y := E))
  have h2 := (hu.continuous_flow 0 τ).comp (continuous_snd (X := E) (Y := E))
  exact continuous_const.mul ((h1.sub h2).inner
    (((hw.lipschitz τ).continuous.comp h1).sub ((hu.lipschitz τ).continuous.comp h2)))

variable [MeasurableSpace E] [BorelSpace E]

/-- **Two maps push a coupling to a coupling**:
`W_2(f_# μ, g_# ν) ≤ (∫ ‖f x - g y‖² dγ)^{1/2}` for every coupling `γ` of `μ`
and `ν`. -/
theorem W2_map_le_sqrt {μ ν : Measure E} {γ : Measure (E × E)} (hγ : IsCoupling μ ν γ)
    {f g : E → E} (hf : Continuous f) (hg : Continuous g) :
    W2 (μ.map f) (ν.map g) ≤ Real.sqrt (∫ p, ‖f p.1 - g p.2‖ ^ 2 ∂γ) := by
  have hm : Measurable fun p : E × E => (f p.1, g p.2) :=
    ((hf.comp continuous_fst).prodMk (hg.comp continuous_snd)).measurable
  have hc : IsCoupling (μ.map f) (ν.map g) (γ.map fun p => (f p.1, g p.2)) := by
    constructor
    · rw [Measure.map_map measurable_fst hm, ← hγ.1, Measure.map_map hf.measurable measurable_fst]
      rfl
    · rw [Measure.map_map measurable_snd hm, ← hγ.2, Measure.map_map hg.measurable measurable_snd]
      rfl
  refine (W2_le_of_coupling _ hc).trans (le_of_eq ?_)
  rw [integral_map hm.aemeasurable (by fun_prop)]
  simp only [dist_eq_norm]

/-- **The transported cost is differentiable**, with derivative
`2 ∫ ⟨Φ^w_τ x - Φ^u_τ y, w(τ, Φ^w_τ x) - u(τ, Φ^u_τ y)⟩ dγ`: differentiate
under the integral, the integrand being bounded near `τ` on the ball carrying
`γ`. -/
theorem hasDerivAt_cost (hw : IsFlowField w Lw Mw) (hu : IsFlowField u Lu Mu)
    {γ : Measure (E × E)} [IsProbabilityMeasure γ] {R : ℝ}
    (hγ : ∀ᵐ p ∂γ, ‖p.1‖ ≤ R ∧ ‖p.2‖ ≤ R) (τ : ℝ) :
    HasDerivAt (fun τ => ∫ p, ‖flow w 0 τ p.1 - flow u 0 τ p.2‖ ^ 2 ∂γ)
      (∫ p, 2 * inner (𝕜 := ℝ) (flow w 0 τ p.1 - flow u 0 τ p.2)
        (w τ (flow w 0 τ p.1) - u τ (flow u 0 τ p.2)) ∂γ) τ := by
  have hMw := hw.nonneg
  have hMu := hu.nonneg
  set D := 2 * R + (Mw + Mu) * (|τ| + 1)
  have hD : ∀ p : E × E, ‖p.1‖ ≤ R ∧ ‖p.2‖ ≤ R → ∀ σ ∈ ball τ 1,
      ‖flow w 0 σ p.1 - flow u 0 σ p.2‖ ≤ D := by
    intro p hp σ hσ
    have hσ' : |σ| ≤ |τ| + 1 := by
      rw [mem_ball, Real.dist_eq] at hσ
      linarith [abs_sub_abs_le_abs_sub σ τ]
    have e1 := hw.norm_flow_le hp.1 0 σ
    have e2 := hu.norm_flow_le hp.2 0 σ
    rw [sub_zero] at e1 e2
    nlinarith [norm_sub_le (flow w 0 σ p.1) (flow u 0 σ p.2)]
  refine (hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := γ) (x₀ := τ)
    (F := fun σ p => ‖flow w 0 σ p.1 - flow u 0 σ p.2‖ ^ 2)
    (F' := fun σ p => 2 * inner (𝕜 := ℝ) (flow w 0 σ p.1 - flow u 0 σ p.2)
      (w σ (flow w 0 σ p.1) - u σ (flow u 0 σ p.2)))
    (bound := fun _ => 2 * (D * (Mw + Mu))) (ball_mem_nhds τ one_pos)
    (Eventually.of_forall fun σ => (continuous_cost hw hu σ).aestronglyMeasurable)
    (integrable_of_ae_norm_le hγ (continuous_cost hw hu τ))
    (continuous_cost_deriv hw hu τ).aestronglyMeasurable ?_ (integrable_const _)
    (Eventually.of_forall fun p σ _ =>
      ((hw.hasDerivAt_flow 0 σ p.1).fun_sub (hu.hasDerivAt_flow 0 σ p.2)).norm_sq)).2
  filter_upwards [hγ] with p hp σ hσ
  rw [Real.norm_eq_abs, abs_mul, abs_two]
  refine mul_le_mul_of_nonneg_left ((abs_real_inner_le_norm _ _).trans ?_) zero_le_two
  exact mul_le_mul (hD p hp σ hσ)
    ((norm_sub_le _ _).trans (add_le_add (hw.bound σ _) (hu.bound σ _))) (norm_nonneg _)
    ((norm_nonneg _).trans (hD p hp σ hσ))

/-- **`W_2` stability of transported measures**, the estimate `e:w2estimate`
of arXiv:2305.05465v6, §6, for measures pushed along flows: if the two fields
differ by at most `K ‖a - b‖ + C W_2(μ_τ, ν_τ)` wherever the trajectories can
be at time `τ`, the transported measures separate at rate at most
`e^{(K + C) t}`. -/
theorem W2_map_flow_le (hw : IsFlowField w Lw Mw) (hu : IsFlowField u Lu Mu)
    {μ₀ ν₀ : Measure E} [IsProbabilityMeasure μ₀] [IsProbabilityMeasure ν₀] {R : ℝ}
    (hμ₀ : μ₀ (closedBall 0 R)ᶜ = 0) (hν₀ : ν₀ (closedBall 0 R)ᶜ = 0)
    {T K C : ℝ} (hC : 0 ≤ C)
    (hwu : ∀ τ ∈ Icc 0 T, ∀ a ∈ closedBall (0 : E) (R + Mw * τ),
      ∀ b ∈ closedBall (0 : E) (R + Mu * τ),
        ‖w τ a - u τ b‖ ≤ K * ‖a - b‖ + C * W2 (μ₀.map (flow w 0 τ)) (ν₀.map (flow u 0 τ)))
    {t : ℝ} (ht : t ∈ Icc 0 T) :
    W2 (μ₀.map (flow w 0 t)) (ν₀.map (flow u 0 t)) ≤ Real.exp ((K + C) * t) * W2 μ₀ ν₀ := by
  set W := W2 (μ₀.map (flow w 0 t)) (ν₀.map (flow u 0 t))
  set e := Real.exp ((K + C) * t)
  have hc : ∀ c ∈ transportCosts μ₀ ν₀, W ^ 2 ≤ c * e ^ 2 := by
    rintro c ⟨γ, hγ, rfl⟩
    have : IsProbabilityMeasure γ := ⟨by
      have := Measure.map_apply measurable_fst MeasurableSet.univ (μ := γ)
      rw [hγ.1] at this; simpa using this.symm⟩
    have hae : ∀ᵐ p ∂γ, ‖p.1‖ ≤ R ∧ ‖p.2‖ ≤ R := by
      filter_upwards [ae_fst_mem hγ measurableSet_closedBall hμ₀,
        ae_snd_mem hγ measurableSet_closedBall hν₀] with p h1 h2
      exact ⟨mem_closedBall_zero_iff.1 h1, mem_closedBall_zero_iff.1 h2⟩
    set P := fun τ => ∫ p, ‖flow w 0 τ p.1 - flow u 0 τ p.2‖ ^ 2 ∂γ
    set P' := fun τ => ∫ p, 2 * inner (𝕜 := ℝ) (flow w 0 τ p.1 - flow u 0 τ p.2)
      (w τ (flow w 0 τ p.1) - u τ (flow u 0 τ p.2)) ∂γ
    have hP : ∀ τ, HasDerivAt P (P' τ) τ := hasDerivAt_cost hw hu hae
    have hWP : ∀ τ, W2 (μ₀.map (flow w 0 τ)) (ν₀.map (flow u 0 τ)) ^ 2 ≤ P τ := fun τ =>
      (Real.le_sqrt (W2_nonneg _ _) (integral_nonneg fun _ => by positivity)).1
        (W2_map_le_sqrt hγ (hw.continuous_flow 0 τ) (hu.continuous_flow 0 τ))
    have hbound : ∀ τ ∈ Ico 0 t, P' τ ≤ 2 * (K + C) * P τ + 0 := by
      intro τ hτ
      set Wτ := W2 (μ₀.map (flow w 0 τ)) (ν₀.map (flow u 0 τ))
      have hi := integrable_of_ae_norm_le hae (continuous_cost hw hu τ)
      have hle : P' τ ≤ ∫ p, ((2 * K + C) * ‖flow w 0 τ p.1 - flow u 0 τ p.2‖ ^ 2 + C * Wτ ^ 2) ∂γ := by
        refine integral_mono_ae (integrable_of_ae_norm_le hae (continuous_cost_deriv hw hu τ))
          ((hi.const_mul _).add (integrable_const _)) ?_
        filter_upwards [hae] with p hp
        have h := hwu τ ⟨hτ.1, hτ.2.le.trans ht.2⟩ _
          (mem_closedBall_zero_iff.2 (by simpa [abs_of_nonneg hτ.1] using hw.norm_flow_le hp.1 0 τ)) _
          (mem_closedBall_zero_iff.2 (by simpa [abs_of_nonneg hτ.1] using hu.norm_flow_le hp.2 0 τ))
        have hn := norm_nonneg (flow w 0 τ p.1 - flow u 0 τ p.2)
        nlinarith [real_inner_le_norm (flow w 0 τ p.1 - flow u 0 τ p.2)
          (w τ (flow w 0 τ p.1) - u τ (flow u 0 τ p.2)), mul_le_mul_of_nonneg_left h hn,
          mul_nonneg hC (sq_nonneg (‖flow w 0 τ p.1 - flow u 0 τ p.2‖ - Wτ))]
      rw [integral_add (hi.const_mul _) (integrable_const _), integral_const_mul, integral_const,
        probReal_univ, one_smul] at hle
      nlinarith [hWP τ]
    have hG := le_gronwallBound_of_liminf_deriv_right_le (f := P) (f' := P')
      (fun τ _ => (hP τ).continuousAt.continuousWithinAt)
      (fun τ _ r hr => (hP τ).hasDerivWithinAt.liminf_right_slope_le hr)
      le_rfl hbound t ⟨ht.1, le_rfl⟩
    rw [gronwallBound_ε0, sub_zero] at hG
    have h0 : P 0 = ∫ p, dist p.1 p.2 ^ 2 ∂γ := by
      simp only [P, hw.flow_self, hu.flow_self, dist_eq_norm]
    calc W ^ 2 ≤ P t := hWP t
      _ ≤ P 0 * Real.exp (2 * (K + C) * t) := hG
      _ = (∫ p, dist p.1 p.2 ^ 2 ∂γ) * e ^ 2 := by
        rw [h0, ← Real.exp_nat_mul]; push_cast; ring_nf
  obtain ⟨c0, hc0⟩ := transportCosts_nonempty μ₀ ν₀
  have he : 0 < e := Real.exp_pos _
  have hsq : W ^ 2 / e ^ 2 ≤ sInf (transportCosts μ₀ ν₀) :=
    le_csInf ⟨c0, hc0⟩ fun c hcm => by rw [div_le_iff₀ (by positivity)]; exact hc c hcm
  rw [W2, ← Real.sqrt_sq he.le, ← Real.sqrt_mul (by positivity)]
  refine Real.le_sqrt_of_sq_le ?_
  rw [div_le_iff₀ (by positivity)] at hsq
  linarith

/-- The hypotheses of the results above are satisfiable: the zero field twice, whose
flows are the identity, the Dirac mass at the origin, carried by `B̄(0, 0)` and
coupled with itself by `δ_{(0, 0)}`, `C = 0` and `t = 0 ∈ [0, 1]`. -/
example : IsFlowField (fun (_ : ℝ) (_ : ℝ) => (0 : ℝ)) 0 0 ∧
    Measure.dirac (0 : ℝ) (closedBall 0 0)ᶜ = 0 ∧
    IsCoupling (Measure.dirac (0 : ℝ)) (Measure.dirac 0) (Measure.dirac (0, 0)) ∧
    (∀ᵐ p ∂Measure.dirac ((0 : ℝ), (0 : ℝ)), ‖p.1‖ ≤ 0 ∧ ‖p.2‖ ≤ 0) ∧
    Continuous (fun x : ℝ => x) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 ∧
    ∀ τ ∈ Icc (0 : ℝ) 1, ∀ a ∈ closedBall (0 : ℝ) (0 + 0 * τ), ∀ b ∈ closedBall (0 : ℝ) (0 + 0 * τ),
      ‖(0 : ℝ) - 0‖ ≤ 0 * ‖a - b‖ + 0 * W2 ((Measure.dirac (0 : ℝ)).map
        (flow (fun _ _ => 0) 0 τ)) ((Measure.dirac (0 : ℝ)).map (flow (fun _ _ => 0) 0 τ)) :=
  ⟨⟨continuous_const, fun _ => LipschitzWith.const 0, fun _ _ => by simp⟩, by simp,
    ⟨by simp, by simp⟩, by rw [ae_dirac_eq]; simp, continuous_id, le_rfl, ⟨le_rfl, zero_le_one⟩,
    fun _ _ _ _ _ _ => by simp⟩

end ContinuityEq
end Transformer
