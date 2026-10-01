/-
# Conditional concentration of an attention numerator

Conditioning on the query token leaves the other tokens independent.
Source: arXiv:2510.22026v2, Appendix B, proof of Theorem 4.2.
-/

import Transformer.Normalization.InitialBounds
import Transformer.Normalization.InitialVectorTail
import Transformer.Normalization.InitialProduct

open scoped BigOperators
open MeasureTheory

namespace Transformer.Normalization

/-- Each numerator exceeds the common threshold with probability at most
`(n+1)^{-2}`. The self term is bounded separately; the remaining `n` tokens
are independent after conditioning on the query. Source:
arXiv:2510.22026v2, Appendix B, proof of Theorem 4.2. -/
theorem measure_pi_initialNumerator_gt_le {d n : ℕ} (hd : 0 < d) (hn : 0 < n)
    (μ : Measure (SSphere d)) [IsProbabilityMeasure μ]
    (hμ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
      μ.map (Perspective.sphereMap d U) = μ) (Q K : ParamMatrix d)
    (hQK : ∀ x y : EucSpace d, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖)
    (j : Fin (n + 1)) :
    Measure.pi (fun _ : Fin (n + 1) => μ)
      {Θ | initialThreshold d (n + 1) < ‖initialNumerator Q K Θ j‖} ≤
      ENNReal.ofReal ((((n + 1 : ℕ) : ℝ) ^ 2)⁻¹) := by
  let : Nonempty (SSphere d) :=
    ⟨⟨EuclideanSpace.single (⟨0, hd⟩ : Fin d) 1, by simp⟩⟩
  have hs : MeasurableSet {Θ : SphereTuple d (n + 1) |
      initialThreshold d (n + 1) < ‖initialNumerator Q K Θ j‖} :=
    measurableSet_lt measurable_const (continuous_initialNumerator Q K j).norm.measurable
  apply measure_pi_insertNth_le μ n j _ hs _
  intro a
  let z := contractedQuery Q K a
  have hz : ‖z‖ ≤ 1 := norm_contractedQuery_le Q K hQK a
  let f := tiltedSphere z
  have hf : Continuous f := continuous_tiltedSphere z
  have hb : ∀ x, ‖f x‖ ≤ 3 := norm_tiltedSphere_le z hz
  have hmean : ‖∫ x, f x ∂μ‖ ≤ 2 / (d : ℝ) := norm_integral_tiltedSphere_le hd μ hμ z hz
  let t := 12 * Real.sqrt (((n + 1 : ℕ) : ℝ) * Real.log ((n + 1 : ℕ) : ℝ))
  have ht : 0 ≤ t := by dsimp [t]; positivity
  have hNn : (n : ℝ) ≤ (n + 1 : ℕ) := by exact_mod_cast Nat.le_succ n
  have hsq : Real.sqrt n ≤ Real.sqrt (n + 1 : ℕ) := Real.sqrt_le_sqrt hNn
  have hmb : (n : ℝ) * ‖∫ x, f x ∂μ‖ ≤ 2 * ((n + 1 : ℕ) : ℝ) / d := by
    calc _ ≤ (n : ℝ) * (2 / (d : ℝ)) :=
        mul_le_mul_of_nonneg_left hmean (Nat.cast_nonneg n)
      _ = 2 * (n : ℝ) / d := by ring
      _ ≤ _ := div_le_div_of_nonneg_right (by linarith) (Nat.cast_nonneg d)
  have hsub : {x : Fin n → SSphere d |
      initialThreshold d (n + 1) <
        ‖initialNumerator Q K (Fin.insertNth j a x) j‖} ⊆
      {x | (n : ℝ) * ‖∫ a, f a ∂μ‖ + 6 * Real.sqrt n + t < ‖∑ i, f (x i)‖} := by
    intro x hx
    change initialThreshold d (n + 1) < ‖initialNumerator Q K (Fin.insertNth j a x) j‖ at hx
    change (n : ℝ) * ‖∫ a, f a ∂μ‖ + 6 * Real.sqrt n + t < ‖∑ i, f (x i)‖
    have heq : initialNumerator Q K (Fin.insertNth j a x) j = f a + ∑ i, f (x i) := by
      unfold initialNumerator
      rw [Fin.sum_univ_succAbove _ j]
      simp only [Fin.insertNth_apply_same, Fin.insertNth_apply_succAbove]
      rfl
    rw [heq] at hx
    have hnorm := norm_add_le (f a) (∑ i, f (x i))
    have ha := hb a
    dsimp [initialThreshold, t] at hx ⊢
    change _ < _
    linarith
  apply (ENNReal.le_ofReal_iff_toReal_le
    (measure_ne_top (Measure.pi (fun _ : Fin n => μ)) _) (by positivity)).mpr
  change (Measure.pi (fun _ : Fin n => μ)).real _ ≤ _
  apply (measureReal_mono (μ := Measure.pi (fun _ : Fin n => μ)) hsub).trans
  exact (measureReal_pi_norm_sum_gt_le μ f hf hb n t ht).trans (initial_tail_exp_le hn)

/-- The two-point uniform law, one remaining token, and zero query/key maps
realize all hypotheses. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) ∧ IsProbabilityMeasure oneDimUniform ∧
    (∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
      oneDimUniform.map (Perspective.sphereMap 1 U) = oneDimUniform) ∧
    (∀ x y : EucSpace 1,
      |inner (𝕜 := ℝ) ((0 : ParamMatrix 1) x) ((0 : ParamMatrix 1) y)| ≤ ‖x‖ * ‖y‖) := by
  refine ⟨by decide, by decide, inferInstance, oneDimUniform_invariant, ?_⟩
  intro x y
  simp
  positivity

/-- With probability at least `1-(n+1)^{-1}`, every numerator is below the
common threshold. Source: arXiv:2510.22026v2, Appendix B, final union bound
in the proof of Theorem 4.2. -/
theorem measure_pi_all_initialNumerator_le {d n : ℕ} (hd : 0 < d) (hn : 0 < n)
    (μ : Measure (SSphere d)) [IsProbabilityMeasure μ]
    (hμ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
      μ.map (Perspective.sphereMap d U) = μ) (Q K : ParamMatrix d)
    (hQK : ∀ x y : EucSpace d, |inner (𝕜 := ℝ) (Q x) (K y)| ≤ ‖x‖ * ‖y‖) :
    1 - ENNReal.ofReal (((n + 1 : ℕ) : ℝ)⁻¹) ≤
      Measure.pi (fun _ : Fin (n + 1) => μ)
        {Θ | ∀ j, ‖initialNumerator Q K Θ j‖ ≤ initialThreshold d (n + 1)} := by
  let σ := Measure.pi (fun _ : Fin (n + 1) => μ)
  let bad : Fin (n + 1) → Set (SphereTuple d (n + 1)) :=
    fun j => {Θ | initialThreshold d (n + 1) < ‖initialNumerator Q K Θ j‖}
  have hbad : ∀ j, MeasurableSet (bad j) := fun j =>
    measurableSet_lt measurable_const (continuous_initialNumerator Q K j).norm.measurable
  have hN : (0 : ℝ) < (n + 1 : ℕ) := by positivity
  have hj : ∀ j, σ.real (bad j) ≤ (((n + 1 : ℕ) : ℝ) ^ 2)⁻¹ := by
    intro j
    exact (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top σ _) (by positivity)).mp
      (measure_pi_initialNumerator_gt_le hd hn μ hμ Q K hQK j)
  have hsum : σ.real (⋃ j, bad j) ≤ ((n + 1 : ℕ) : ℝ)⁻¹ := by
    calc _ ≤ ∑ j, σ.real (bad j) := measureReal_iUnion_fintype_le bad
      _ ≤ ∑ _ : Fin (n + 1), (((n + 1 : ℕ) : ℝ) ^ 2)⁻¹ :=
        Finset.sum_le_sum fun j _ => hj j
      _ = ((n + 1 : ℕ) : ℝ)⁻¹ := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        field_simp
  have hprob : σ (⋃ j, bad j) ≤ ENNReal.ofReal (((n + 1 : ℕ) : ℝ)⁻¹) :=
    (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top σ _) (by positivity)).mpr hsum
  have hgood : {Θ : SphereTuple d (n + 1) |
      ∀ j, ‖initialNumerator Q K Θ j‖ ≤ initialThreshold d (n + 1)} = (⋃ j, bad j)ᶜ := by
    ext Θ
    simp only [Set.mem_ofPred_eq, Set.mem_compl_iff, Set.mem_iUnion, not_exists, not_lt, bad]
  rw [hgood]
  change _ ≤ σ (⋃ j, bad j)ᶜ
  rw [measure_compl (MeasurableSet.iUnion hbad) (measure_ne_top σ _), measure_univ]
  exact tsub_le_tsub_left hprob 1

/-- Positive dimension, one remaining token, the two-point uniform law,
and zero query/key maps realize the hypotheses. -/
example : 0 < (1 : ℕ) ∧ 0 < (1 : ℕ) ∧ IsProbabilityMeasure oneDimUniform ∧
    (∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
      oneDimUniform.map (Perspective.sphereMap 1 U) = oneDimUniform) ∧
    (∀ x y : EucSpace 1,
      |inner (𝕜 := ℝ) ((0 : ParamMatrix 1) x) ((0 : ParamMatrix 1) y)| ≤ ‖x‖ * ‖y‖) := by
  refine ⟨by decide, by decide, inferInstance, oneDimUniform_invariant, ?_⟩
  intro x y
  simp
  positivity

end Transformer.Normalization
