/-
# Finite-net comparison for the original attention weak law

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §2, `prop: targets.atoms`.
Finite nets of initial sphere points evolve along the actual attention
characteristics. Smooth-cone comparison bounds every Lipschitz test
with explicit net and smoothing errors. Choosing the scales in order
removes those errors for the original arbitrary weak probability law.
-/

import Transformer.Interpolation.Section2_AttentionFlow
import Transformer.Interpolation.Section1_AutonomousTransport

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The finite comparison gives an explicit original weak transport error
for every Lipschitz test and arbitrary weak law.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem integral_attention_lipschitz_test_le_transport_net {ι : Type*} [Fintype ι] [Nonempty ι]
    (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (M L : ℝ≥0) (δ ε β T : ℝ) (hε : 0 < ε) (hβ : 0 < β) (hT : 0 ≤ T)
    (hΦT : LipschitzWith M (fun x => attentionFlow B μ hμ x T))
    (z : ι → SSphere d) (hnet : ∀ x : SSphere d, ∃ i, dist x (z i) ≤ δ)
    (f : SSphere d → ℝ) (hf : LipschitzWith L f) :
    (∫ x : SSphere d, f x ∂(μ T : Measure (SSphere d))) ≤
      (∫ x : SSphere d, f (attentionFlow B μ hμ x T) ∂(μ 0 : Measure (SSphere d))) +
        (L : ℝ) * (M : ℝ) * δ + (L : ℝ) * exp ((2 * exp (2 * ‖B‖) * ‖B‖ + 2) * T) * (δ + ε) + log (Fintype.card ι) / β := by
  let K : ℝ := 2 * exp (2 * ‖B‖) * ‖B‖ + 2
  let Φ := attentionFlow B μ hμ
  let Y (i : ι) := Φ (z i)
  let E := (L : ℝ) * (M : ℝ) * δ + (L : ℝ) * exp ((2 * exp (2 * ‖B‖) * ‖B‖ + 2) * T) * (δ + ε) + log (Fintype.card ι) / β
  have hpoint (x : SSphere d) : finiteTransportTest K L ε β T Y f (0, (x : EucSpace d)) ≤ f (attentionFlow B μ hμ x T) + E := by
    obtain ⟨i, hi⟩ := hnet x
    have hdist : dist (z i) x ≤ δ := by simpa only [dist_comm] using hi
    have hfi := (hf.comp hΦT).dist_le_mul (z i) x
    rw [Real.dist_eq, NNReal.coe_mul] at hfi
    have hfi' : f (Y i T) ≤ f (attentionFlow B μ hμ x T) + (L : ℝ) * (M : ℝ) * δ := by
      have hh := (le_abs_self (f (Φ (z i) T) - f (attentionFlow B μ hμ x T))).trans
        (hfi.trans (mul_le_mul_of_nonneg_left hdist (mul_nonneg L.coe_nonneg M.coe_nonneg)))
      dsimp only [Y, Φ]
      linarith
    have hx : ‖(x : EucSpace d) - (Y i 0 : EucSpace d)‖ ≤ δ := by
      dsimp only [Y, Φ]
      rw [attentionFlow_zero]
      simpa only [Subtype.dist_eq, dist_eq_norm] using hi
    have he := finiteTransportTest_le_entry K L ε β T L.coe_nonneg hε.le hβ Y f (0, (x : EucSpace d)) i
    simp only [sub_zero] at he
    have hb := mul_le_mul_of_nonneg_left (add_le_add hx (le_refl ε))
      (mul_nonneg L.coe_nonneg (exp_pos ((K : ℝ) * T)).le)
    dsimp only [E, K] at he hb ⊢
    linarith
  have hc := (contDiff_finiteTransportTest K L ε β T hε Y
    (fun i => contDiff_attentionFlow B μ hμ (z i)) f).continuous
  have hib := integrable_spherical_continuous (μ 0)
    (fun x : SSphere d => finiteTransportTest K L ε β T Y f (0, (x : EucSpace d)))
    (hc.comp (continuous_const.prodMk continuous_subtype_val))
  have hif := integrable_spherical_continuous (μ 0) (fun x => f (attentionFlow B μ hμ x T)) (hf.continuous.comp hΦT.continuous)
  have hb := integral_mono_ae hib (hif.add (integrable_const E)) (ae_of_all _ hpoint)
  simp only [Pi.add_apply] at hb
  rw [integral_add hif (integrable_const E), integral_const, probReal_univ, one_smul] at hb
  have hi := integral_attention_test_le_finiteTransport_initial B μ hμ L ε β T hε hβ hT z f hf
  have hh := hi.trans hb
  dsimp only [E] at hh
  linarith


/-- Removing all finite-net and smoothing errors gives the exact
characteristic inequality for every original weak solution.
Auxiliary for arXiv:2411.04551v3, §1, the well-posedness argument. -/
theorem integral_attention_lipschitz_test_le_transport (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (M L : ℝ≥0) (T : ℝ) (hT : 0 ≤ T)
    (hΦT : LipschitzWith M (fun x => attentionFlow B μ hμ x T))
    (f : SSphere d → ℝ) (hf : LipschitzWith L f) :
    (∫ x : SSphere d, f x ∂(μ T : Measure (SSphere d))) ≤
      ∫ x : SSphere d, f (attentionFlow B μ hμ x T) ∂(μ 0 : Measure (SSphere d)) := by
  classical
  apply le_of_forall_pos_le_add
  intro η hη
  let C := (L : ℝ) * (M : ℝ) + 2 * (L : ℝ) * exp ((2 * exp (2 * ‖B‖) * ‖B‖ + 2) * T)
  have hC : 0 ≤ C := add_nonneg (mul_nonneg L.coe_nonneg M.coe_nonneg)
    (mul_nonneg (mul_nonneg (by norm_num) L.coe_nonneg) (exp_pos _).le)
  let δ := η / (C + 1)
  have hδ : 0 < δ := div_pos hη (by linarith)
  obtain ⟨x₀, _⟩ := Measure.nonempty_support (IsProbabilityMeasure.ne_zero (μ 0 : Measure (SSphere d)))
  obtain ⟨S, hS, hnet⟩ := exists_spherical_finite_net x₀ δ hδ
  let I := {x : SSphere d // x ∈ S}
  let : Nonempty I := by obtain ⟨x, hx⟩ := hS; exact ⟨⟨x, hx⟩⟩
  let β := (Fintype.card I : ℝ) / δ
  have hn : (0 : ℝ) < Fintype.card I := Nat.cast_pos.mpr Fintype.card_pos
  have hβ : 0 < β := div_pos hn hδ
  have hnet' (x : SSphere d) : ∃ i : I, dist x (i : SSphere d) ≤ δ := by
    obtain ⟨y, hy, hxy⟩ := hnet x
    exact ⟨⟨y, hy⟩, hxy.le⟩
  have he := integral_attention_lipschitz_test_le_transport_net B μ hμ M L δ δ β T hδ hβ hT hΦT (fun i : I => (i : SSphere d)) hnet' f hf
  have hlog : log (Fintype.card I) / β ≤ δ := by
    apply (div_le_iff₀ hβ).mpr
    have hb : δ * β = (Fintype.card I : ℝ) := by dsimp only [β]; field_simp
    rw [hb]
    exact (log_le_sub_one_of_pos hn).trans (by linarith)
  have hmul : (C + 1) * δ = η := by dsimp only [δ]; field_simp
  dsimp only [C] at hmul
  nlinarith


/-- Applying the comparison to a test and its negative gives equality
for every Lipschitz test along the actual attention law.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem integral_attention_lipschitz_test_eq_transport (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (L : ℝ≥0) (T : ℝ) (hT : 0 ≤ T) (f : SSphere d → ℝ) (hf : LipschitzWith L f) :
    (∫ x : SSphere d, f x ∂(μ T : Measure (SSphere d))) =
      ∫ x : SSphere d, f (attentionFlow B μ hμ x T) ∂(μ 0 : Measure (SSphere d)) := by
  have hM := lipschitzWith_attentionFlow_initial B μ hμ T
  apply le_antisymm (integral_attention_lipschitz_test_le_transport B μ hμ _ L T hT hM f hf)
  have hn := integral_attention_lipschitz_test_le_transport B μ hμ _ L T hT hM (-f) hf.neg
  simpa only [Pi.neg_apply, integral_neg, neg_le_neg_iff] using hn

/-- Every original attention weak probability law is exactly the
pushforward of its initial law along its actual characteristics, for
nonnegative time. No atomicity or representation assumption is used.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, used in §2. -/
theorem cauchyPB_attention_transport_nonneg (B : ParamMatrix d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (T : ℝ) (hT : 0 ≤ T) : (μ T : Measure (SSphere d)) =
      Measure.map (fun x => attentionFlow B μ hμ x T) (μ 0 : Measure (SSphere d)) := by
  have hc := (lipschitzWith_attentionFlow_initial B μ hμ T).continuous
  apply measure_eq_of_sphere_relu_ridges
  intro u b
  obtain ⟨L, hL⟩ := exists_lipschitz_spherical_ridge u b
  rw [integral_map hc.measurable.aemeasurable hL.continuous.measurable.aestronglyMeasurable]
  exact integral_attention_lipschitz_test_eq_transport B μ hμ L T hT _ hL

/-- A genuine attention weak law, its proved spatial flow modulus,
a finite spherical net and a nonconstant Lipschitz test jointly
realize every finite-comparison and exact-transport hypothesis. -/
example : ∃ (B : ParamMatrix 2) (μ : ℝ → Perspective.ProbSphere 2)
    (hμ : cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) μ)
    (M : ℝ≥0) (S : Finset (SSphere 2)) (f : SSphere 2 → ℝ),
    LipschitzWith M (fun x => attentionFlow B μ hμ x 1) ∧ LipschitzWith 1 f ∧
    S.Nonempty ∧ (∀ x : SSphere 2, ∃ y ∈ S, dist x y ≤ 1) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  let B := ContinuousLinearMap.id ℝ (EucSpace 2)
  let p : SSphere 2 := basePoint 1
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  have hθ : pureAttentionParams 2 1 = (fun _ => ⟨B, B, 0, 0, 0⟩) := by
    funext t
    simp only [pureAttentionParams, B, one_smul]
  have hμ : cauchyPB 2 (fun _ => ⟨B, B, 0, 0, 0⟩) μ := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  obtain ⟨S, hS, hnet⟩ := exists_spherical_finite_net p 1 zero_lt_one
  refine ⟨B, μ, hμ, _, S, (fun x => dist x p), lipschitzWith_attentionFlow_initial B μ hμ 1,
    LipschitzWith.dist_left _, hS, ?_, zero_lt_one, zero_le_one⟩
  intro x
  obtain ⟨y, hy, hxy⟩ := hnet x
  exact ⟨y, hy, hxy.le⟩

end Transformer.Interpolation
