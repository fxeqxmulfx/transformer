module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Transfer

/-! The handoff to the recovery theorem. A bound r̄ on the limiting
residual over the updates B ≤ t ≤ H, together with an allowance for the
updates after H, yields an excess that is measurable, integrable and
nonnegative, bounds the finite residual at every update t ≥ B and every
fraction, and has expectation at most max(r̄, 0) + widthError + E[tail].
These are the hypotheses on the excess in `permanent_of_observed_drift`. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Finset

variable {P K J : Type*}

theorem continuous_l1_meanField_sub {α : Type*} [TopologicalSpace α] [Fintype J] {N : ℕ}
    {Ξ : α → J → ℝ} (hΞ : ∀ j, Continuous fun x => Ξ x j) (c : J → ℝ) :
    Continuous fun z : Fin N → α => l1 (meanField Ξ z - c) := by
  unfold l1
  refine continuous_finsetSum _ fun j _ => ?_
  change Continuous fun z : Fin N → α => |(∑ i, Ξ (z i) j) / N - c j|
  exact (((continuous_finsetSum _ fun i _ => (hΞ j).comp (continuous_apply i)).div_const _).sub
    continuous_const).abs

section Measurable

variable [Fintype P] [Fintype K] [Fintype J] (A : Training P K) (μ : Measure (P → ℝ))
  [IsProbabilityMeasure μ] {Γ : (J → ℝ) → ℝ} {χ : ℝ → (P → ℝ) → (P → ℝ) → J → ℝ}
  {η LG Lφ Cφ LΓ Lχ Mχ Cχ : ℝ}

omit [IsProbabilityMeasure μ] in
theorem continuous_fluctuation (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (N s : ℕ) :
    Continuous fun z : Fin N → P → ℝ => fluctuation A μ z s := by
  have hφ := continuous_of_l1 h.Lφ_nonneg h.field
  have hw := (continuous_flow A μ h.eps_pos h.continuous_grad s).1
  exact continuous_l1_meanField_sub (Ξ := fun x => A.field (flow A μ s x).weights)
    (fun k => (continuous_apply k).comp (hφ.comp hw)) _

theorem continuous_segmentError (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (N t : ℕ) (u : ℝ) :
    Continuous fun z : Fin N → P → ℝ => segmentError A μ χ z t u :=
  continuous_l1_meanField_sub (fun j => (integrable_features A μ h t u j).1) _

theorem measurable_majorant (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) (N B H n : ℕ) :
    Measurable fun z : Fin N → P → ℝ => majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n := by
  refine Continuous.measurable ?_
  unfold majorant
  have h1 := continuous_finsetSum (range (H+1)) fun s _ => continuous_fluctuation A μ h N s
  have h2 := continuous_finsetSum (Icc B H) fun t _ =>
    continuous_finsetSum (range (n+1)) fun k _ => continuous_segmentError A μ h N t ((k:ℝ)/n)
  exact continuous_const.mul (((continuous_const.mul (continuous_const.mul h1)).add h2).add
    continuous_const)

/-- The excess handed to the recovery theorem: the limiting bound, the
width majorant over B ≤ t ≤ H, and the allowance for the tail. -/
theorem drift_excess (h : Regular A Γ χ η LG Lφ Cφ LΓ Lχ Mχ Cχ) {N : ℕ} (hN : 0 < N)
    {B H n : ℕ} (hn : 0 < n) {r : ℝ}
    (hlim : ∀ t ∈ Icc B H, ∀ u ∈ Set.Ioo (0:ℝ) 1, limitResidual A μ Γ χ t u ≤ r)
    (tail : (Fin N → P → ℝ) → ℝ) (htm : Measurable tail)
    (hti : Integrable tail (Measure.pi fun _ : Fin N => μ)) (htn : ∀ z, 0 ≤ tail z)
    (htail : ∀ z t, H < t → ∀ u ∈ Set.Ioo (0:ℝ) 1, residual A Γ χ z t u ≤ tail z) :
    Measurable (fun z => max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z) ∧
      Integrable (fun z => max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z)
        (Measure.pi fun _ : Fin N => μ) ∧
      (∀ z, 0 ≤ max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z) ∧
      ∫ z, (max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z)
          ∂(Measure.pi fun _ : Fin N => μ) ≤
        max r 0 + widthError A.eps η LG Lφ Cφ LΓ Lχ Mχ Cχ (Fintype.card K) (Fintype.card J) N B H n +
          ∫ z, tail z ∂(Measure.pi fun _ : Fin N => μ) ∧
      ∀ z t, B ≤ t → ∀ u ∈ Set.Ioo (0:ℝ) 1,
        residual A Γ χ z t u ≤ max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n + tail z := by
  obtain ⟨hi, hle⟩ := integral_majorant_le A μ h hN B H n
  have hm := measurable_majorant A μ h N B H n
  have hc : Integrable (fun z => max r 0 + majorant A μ χ η LG Lφ LΓ Lχ Mχ z B H n)
      (Measure.pi fun _ : Fin N => μ) := (integrable_const _).add hi
  refine ⟨(measurable_const.add hm).add htm, hc.add hti, fun z => ?_, ?_, fun z t ht u hu => ?_⟩
  · have := majorant_nonneg A μ h z B H n
    have := htn z
    have := le_max_right r 0
    linarith
  · rw [integral_add hc hti, integral_add (integrable_const _) hi, integral_const]
    simp only [probReal_univ, smul_eq_mul, one_mul]
    linarith
  · have h0 := majorant_nonneg A μ h z B H n
    have hr := le_max_left r 0
    have hr0 := le_max_right r 0
    rcases le_or_gt t H with htH | htH
    · have hmem : t ∈ Icc B H := mem_Icc.2 ⟨ht, htH⟩
      have := residual_sub_le_majorant A μ h hN z hn hmem hu
      have := hlim t hmem u hu
      have := htn z
      linarith
    · have := htail z t htH u hu
      linarith

end Measurable

section Examples

example := continuous_l1_meanField_sub (N := 2) (Ξ := fun (x : ℝ) (_ : Fin 1) => x)
  (fun _ => continuous_id) 0
example := continuous_fluctuation identityTraining (Measure.dirac 0) identityRegular 2 3
example := continuous_segmentError identityTraining (Measure.dirac 0) identityRegular 2 3 (1/2)
example := measurable_majorant identityTraining (Measure.dirac 0) identityRegular 2 1 3 4
example := drift_excess identityTraining (Measure.dirac 0) identityRegular (N := 2) two_pos
  (B := 1) (H := 3) (n := 4) (by norm_num) (r := 0) (fun _ _ _ _ => by simp [limitResidual])
  (fun _ => 0) measurable_const (integrable_const _) (fun _ => le_rfl)
  (fun _ _ _ _ _ => by simp [residual])

end Examples

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
