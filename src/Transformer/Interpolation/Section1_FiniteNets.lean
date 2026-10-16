/-
# Removing the finite-net error in original weak transport

Auxiliary to arXiv:2411.04551v3, §1, the Lipschitz well-posedness
argument for `eq: cauchy.pb`, and the characteristic tests in §3.1.

Compactness gives finite nets of actual spherical initial points.
Transporting those centers and applying the smooth comparison gives
an explicit estimate for every original weak law and Lipschitz test.
The error contains the net radius, the regularization scale, and the
logarithmic approximation error of the finite minimum.

Choose equal net and regularization scales, then choose the inverse
smoothing scale after the finite net. Every error can be made arbitrarily
small. This proves the original characteristic inequality for arbitrary
initial probabilities, with no spatial derivative of the flow assumed.
-/

import Transformer.Interpolation.Section1_FiniteTransport
import Mathlib.MeasureTheory.Measure.Support

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The compact original sphere has nonempty finite nets of every positive
radius. Auxiliary for arXiv:2411.04551v3, §1, the weak transport tests. -/
theorem exists_spherical_finite_net (x₀ : SSphere d) (δ : ℝ) (hδ : 0 < δ) :
    ∃ s : Finset (SSphere d), s.Nonempty ∧ ∀ x : SSphere d, ∃ y ∈ s, dist x y < δ := by
  classical
  obtain ⟨s, hs⟩ := (isCompact_univ : IsCompact (univ : Set (SSphere d))).elim_finite_subcover
    (fun x : SSphere d => Metric.ball x δ) (fun _ => Metric.isOpen_ball) (by
      intro x _
      exact mem_iUnion.mpr ⟨x, Metric.mem_ball_self hδ⟩)
  have hc (x : SSphere d) : ∃ y ∈ s, dist x y < δ := by
    obtain ⟨y, hy⟩ := mem_iUnion.mp (hs (mem_univ x))
    obtain ⟨hys, hxy⟩ := mem_iUnion.mp hy
    exact ⟨y, hys, hxy⟩
  obtain ⟨y, hy, _⟩ := hc x₀
  exact ⟨s, ⟨y, hy⟩, hc⟩


/-- The finite comparison gives an explicit original weak transport error
for every Lipschitz test and arbitrary weak law.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem integral_lipschitz_test_le_transport_net {ι : Type*} [Fintype ι] [Nonempty ι]
    (θ : TimeParams d) (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (M L : ℝ≥0) (δ ε β T : ℝ)
    (hε : 0 < ε) (hβ : 0 < β) (hT : 0 ≤ T)
    (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x s, HasDerivAt (fun r => (Φ x r : EucSpace d)) (F (Φ x s)) s)
    (hΦ₀ : ∀ x, Φ x 0 = x) (hΦT : LipschitzWith M (fun x => Φ x T))
    (z : ι → SSphere d) (hnet : ∀ x : SSphere d, ∃ i, dist x (z i) ≤ δ)
    (f : SSphere d → ℝ) (hf : LipschitzWith L f)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    (∫ x : SSphere d, f x ∂(μ T : Measure (SSphere d))) ≤
      (∫ x : SSphere d, f (Φ x T) ∂(μ 0 : Measure (SSphere d))) +
        (L : ℝ) * (M : ℝ) * δ + (L : ℝ) * exp ((K : ℝ) * T) * (δ + ε) + log (Fintype.card ι) / β := by
  let Y (i : ι) := Φ (z i)
  let E := (L : ℝ) * (M : ℝ) * δ + (L : ℝ) * exp ((K : ℝ) * T) * (δ + ε) + log (Fintype.card ι) / β
  have hpoint (x : SSphere d) : finiteTransportTest K L ε β T Y f (0, (x : EucSpace d)) ≤ f (Φ x T) + E := by
    obtain ⟨i, hi⟩ := hnet x
    have hdist : dist (z i) x ≤ δ := by simpa only [dist_comm] using hi
    have hfi := (hf.comp hΦT).dist_le_mul (z i) x
    rw [Real.dist_eq, NNReal.coe_mul] at hfi
    have hfi' : f (Y i T) ≤ f (Φ x T) + (L : ℝ) * (M : ℝ) * δ := by
      have hh := (le_abs_self (f (Φ (z i) T) - f (Φ x T))).trans
        (hfi.trans (mul_le_mul_of_nonneg_left hdist (mul_nonneg L.coe_nonneg M.coe_nonneg)))
      dsimp only [Y]
      linarith
    have hx : ‖(x : EucSpace d) - (Y i 0 : EucSpace d)‖ ≤ δ := by
      dsimp only [Y]
      rw [hΦ₀]
      simpa only [Subtype.dist_eq, dist_eq_norm] using hi
    have he := finiteTransportTest_le_entry K L ε β T L.coe_nonneg hε.le hβ Y f (0, (x : EucSpace d)) i
    simp only [sub_zero] at he
    have hb := mul_le_mul_of_nonneg_left (add_le_add hx (le_refl ε))
      (mul_nonneg L.coe_nonneg (exp_pos ((K : ℝ) * T)).le)
    dsimp only [E]
    linarith
  have hc := (contDiff_finiteTransportTest K L ε β T hε Y
    (fun i => contDiff_spherical_particle F hF _ (hΦ (z i))) f).continuous
  have hib := integrable_spherical_continuous (μ 0)
    (fun x : SSphere d => finiteTransportTest K L ε β T Y f (0, (x : EucSpace d)))
    (hc.comp (continuous_const.prodMk continuous_subtype_val))
  have hif := integrable_spherical_continuous (μ 0) (fun x => f (Φ x T)) (hf.continuous.comp hΦT.continuous)
  have hb := integral_mono_ae hib (hif.add (integrable_const E)) (ae_of_all _ hpoint)
  simp only [Pi.add_apply] at hb
  rw [integral_add hif (integrable_const E), integral_const, probReal_univ, one_smul] at hb
  have hi := integral_test_le_finiteTransport_initial θ F hF hfield K hK L ε β T hε hβ hT Y
    (fun i => hΦ (z i)) f hf μ hμ
  have hh := hi.trans hb
  dsimp only [E] at hh
  linarith


/-- Removing all finite-net and smoothing errors gives the exact
characteristic inequality for every original weak solution.
Auxiliary for arXiv:2411.04551v3, §1, the well-posedness argument. -/
theorem integral_lipschitz_test_le_transport (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (M L : ℝ≥0) (T : ℝ) (hT : 0 ≤ T)
    (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x s, HasDerivAt (fun r => (Φ x r : EucSpace d)) (F (Φ x s)) s)
    (hΦ₀ : ∀ x, Φ x 0 = x) (hΦT : LipschitzWith M (fun x => Φ x T))
    (f : SSphere d → ℝ) (hf : LipschitzWith L f)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    (∫ x : SSphere d, f x ∂(μ T : Measure (SSphere d))) ≤
      ∫ x : SSphere d, f (Φ x T) ∂(μ 0 : Measure (SSphere d)) := by
  classical
  apply le_of_forall_pos_le_add
  intro η hη
  let C := (L : ℝ) * (M : ℝ) + 2 * (L : ℝ) * exp ((K : ℝ) * T)
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
  have he := integral_lipschitz_test_le_transport_net θ F hF hfield K hK M L δ δ β T hδ hβ hT
    Φ hΦ hΦ₀ hΦT (fun i : I => (i : SSphere d)) hnet' f hf μ hμ
  have hlog : log (Fintype.card I) / β ≤ δ := by
    apply (div_le_iff₀ hβ).mpr
    have hb : δ * β = (Fintype.card I : ℝ) := by dsimp only [β]; field_simp
    rw [hb]
    exact (log_le_sub_one_of_pos hn).trans (by linarith)
  have hmul : (C + 1) * δ = η := by dsimp only [δ]; field_simp
  dsimp only [C] at hmul
  nlinarith

/-- An actual projection neuron, its complete original flow and weak
law, a nonconstant Lipschitz test and a finite spherical net jointly
realize all hypotheses, with positive scales and nonnegative time. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2) (K M : ℝ≥0)
    (Φ : SSphere 2 → ℝ → SSphere 2) (μ : ℝ → Perspective.ProbSphere 2)
    (f : SSphere 2 → ℝ) (S : Finset (SSphere 2)),
    Continuous F ∧ LipschitzOnWith K F (Sphere 2) ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    (∀ x s, HasDerivAt (fun r => (Φ x r : EucSpace 2)) (F (Φ x s)) s) ∧
    (∀ x, Φ x 0 = x) ∧ LipschitzWith M (fun x => Φ x 1) ∧
    LipschitzWith 1 f ∧ cauchyPB 2 θ μ ∧ S.Nonempty ∧
    (∀ x : SSphere 2, ∃ y ∈ S, dist x y ≤ 1) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  let a : EucSpace 2 := basePoint 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) a 0 1
  let F (x : EucSpace 2) := proj 2 x a
  let Φ := ridgeFlow a 0 1
  let μ := ridgeLaw a 0 1 (Perspective.diracProb 2 (basePoint 1))
  have hF : ContDiff ℝ 1 F := contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  obtain ⟨K, hK⟩ := GlobalFlow.exists_lipschitzOnWith_closedBall hF.locallyLipschitz 1
  obtain ⟨M, hM⟩ := exists_lipschitzWith_ridgeFlow a 0 1 1
  obtain ⟨S, hS, hnet⟩ := exists_spherical_finite_net (basePoint 1) 1 zero_lt_one
  have hsphere : Sphere 2 ⊆ Metric.closedBall 0 1 := fun x hx =>
    mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp hx).le
  refine ⟨θ, F, K, M, Φ, μ, fun x => dist x (basePoint 1), S,
    hF.continuous, hK.mono hsphere, ?_, ?_, ridgeFlow_zero _ _ _, hM,
    LipschitzWith.dist_left _, cauchyPB_ridgeLaw _ _ _ _ _, hS, ?_, zero_lt_one, zero_le_one⟩
  · intro ρ s x
    rw [fullVF_reluRidgeParams]
    simp [F]
  · intro x t
    simpa [F, Φ] using hasDerivAt_ridgeFlow a 0 1 x t
  · intro x
    obtain ⟨y, hy, hxy⟩ := hnet x
    exact ⟨y, hy, hxy.le⟩

end Transformer.Interpolation
