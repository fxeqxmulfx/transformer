module

public import Transformer.FrankWolfe.Section5_HistoryKernel
public import Transformer.FrankWolfe.Section5_Transition

/-!
# Existence of the actual softmax attention process

ArXiv:2508.09628v1, §5, `eq: softmax.process`. The original
process predicate constrains its law on infinite token paths.
This module constructs such a law from the actual choice histories.
Every deterministic input, real inverse temperature and real step size
has a genuine process; the intended convex regime is included.

A past event factors through a finite history prefix. The proved
Ionescu–Tulcea joint law gives its next-step probability, and the
actual history-kernel marginal is the original softmax row.
The bounded nonnegative row expectation converts this integral identity
to the original real-valued transition identity. The constructed law is
a probability measure and has the required original initial value.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Set
open scoped BigOperators ENNReal
namespace Transformer.FrankWolfe
/-- Each actual history path is mapped to its original token configurations.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def saHistoryPath {d n : ℕ} (γ : ℝ) (X₀ : Idx n → EucSpace d)
    (h : ℕ → SAHistory n) : ℕ → Idx n → EucSpace d :=
  fun t => saHistoryConfig γ X₀ (h t)

/-- The actual token-path map is measurable.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem measurable_saHistoryPath {d n : ℕ} (γ : ℝ) (X₀ : Idx n → EucSpace d) :
    Measurable (saHistoryPath γ X₀) :=
  Measurable.of_eval fun t => (measurable_of_countable (saHistoryConfig γ X₀)).comp
    (measurable_pi_apply t)

/-- An actual finite history prefix is extended by empty histories after its endpoint.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
def saExtendHistoryPrefix {n : ℕ} (t : ℕ) (h : (Finset.Iic t) → SAHistory n)
    (s : ℕ) : SAHistory n :=
  if hs : s ≤ t then h ⟨s, Finset.mem_Iic.mpr hs⟩ else SAHistory.empty n

/-- Every original measurable past event factors through the actual finite history prefix.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem pastDet_history_factor {d n : ℕ} (γ : ℝ) (X₀ : Idx n → EucSpace d)
    (t : ℕ) (S : Set (ℕ → Idx n → EucSpace d)) (hS : PastDet t S) :
    ∃ B : Set ((Finset.Iic t) → SAHistory n), MeasurableSet B ∧
      ∀ h : ℕ → SAHistory n, saHistoryPath γ X₀ h ∈ S ↔ Preorder.frestrictLe t h ∈ B := by
  let B : Set ((Finset.Iic t) → SAHistory n) :=
    {h | saHistoryPath γ X₀ (saExtendHistoryPrefix t h) ∈ S}
  refine ⟨B, (Set.to_countable _).measurableSet, fun h => ?_⟩
  apply hS.2
  intro s hs
  simp only [saHistoryPath, saExtendHistoryPrefix, hs, ↓reduceDIte, Preorder.frestrictLe_apply]

/-- The actual history law satisfies the original conditional particle transition identity.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem saHistoryLaw_transition {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d)
    (t : ℕ) (i : Idx n) (A : Set (EucSpace d))
    (S : Set (ℕ → Idx n → EucSpace d)) (hS : PastDet t S) :
    saHistoryLaw β γ X₀ ((saHistoryPath γ X₀) ⁻¹' S ∩
      {h | saHistoryConfig γ X₀ (h (t + 1)) i ∈ A}) =
      ∫⁻ h in (saHistoryPath γ X₀) ⁻¹' S,
        ∑ j : Idx n, ENNReal.ofReal (attWeight β (saHistoryPath γ X₀ h t) i j) *
          A.indicator (fun _ => (1 : ℝ≥0∞))
            ((1 - γ) • saHistoryPath γ X₀ h t i + γ • saHistoryPath γ X₀ h t j)
            ∂saHistoryLaw β γ X₀ := by
  obtain ⟨B, hB, hfactor⟩ := pastDet_history_factor γ X₀ t S hS
  let D : Set (SAHistory n) := {h | saHistoryConfig γ X₀ h i ∈ A}
  have hD : MeasurableSet D := (Set.to_countable _).measurableSet
  have hJ := congrArg (fun μ => μ (B ×ˢ D)) (saHistoryLaw_joint β γ X₀ t)
  rw [Measure.compProd_apply_prod hB hD,
    Measure.map_apply ((Preorder.measurable_frestrictLe t).prodMk (measurable_pi_apply _))
      (hB.prod hD)] at hJ
  have hpre : (fun h : ℕ → SAHistory n => (Preorder.frestrictLe t h, h (t + 1))) ⁻¹' (B ×ˢ D) =
      ((saHistoryPath γ X₀) ⁻¹' S ∩ {h | saHistoryConfig γ X₀ (h (t + 1)) i ∈ A}) := by
    ext h
    exact and_congr (hfactor h).symm Iff.rfl
  rw [hpre] at hJ
  rw [← hJ, setLIntegral_map hB (measurable_of_countable _) (Preorder.measurable_frestrictLe t)]
  have hpreB : (Preorder.frestrictLe t) ⁻¹' B = (saHistoryPath γ X₀) ⁻¹' S := by
    ext h
    exact (hfactor h).symm
  rw [hpreB]
  apply lintegral_congr
  intro h
  change saHistoryKernel β γ X₀ (h t) D = _
  exact saHistoryKernel_marginal β γ X₀ (h t) i A

/-- The actual history law starts at the genuine empty history almost surely.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem saHistoryLaw_initial {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) :
    ∀ᵐ h ∂saHistoryLaw β γ X₀, h 0 = SAHistory.empty n := by
  have hp : ∀ᵐ p ∂(saHistoryLaw β γ X₀).map (Preorder.frestrictLe 0),
      p = (fun _ => SAHistory.empty n) := by
    rw [saHistoryLaw_prefix, Kernel.partialTraj_self, Kernel.id_apply]
    exact (ae_dirac_iff ((Set.to_countable _).measurableSet)).mpr rfl
  have he := (ae_map_iff (Preorder.measurable_frestrictLe 0).aemeasurable
    ((Set.to_countable _).measurableSet)).mp hp
  filter_upwards [he] with h hh
  exact congrFun hh ⟨0, by simp⟩

/-- The constructed law on the original infinite token-path space.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
noncomputable def saProcessLaw {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) :
    Measure (ℕ → Idx n → EucSpace d) :=
  (saHistoryLaw β γ X₀).map (saHistoryPath γ X₀)

instance isProbabilityMeasure_saProcessLaw {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) :
    IsProbabilityMeasure (saProcessLaw β γ X₀) := by
  unfold saProcessLaw
  exact (Measure.isProbabilityMeasure_map_iff (measurable_saHistoryPath γ X₀).aemeasurable).mpr
    inferInstance

/-- The constructed probability law satisfies every original process condition.
Source: arXiv:2508.09628v1, §5, `eq: softmax.process`. -/
theorem isSAProcess_saProcessLaw {d n : ℕ} (β γ : ℝ) (X₀ : Idx n → EucSpace d) :
    IsSAProcess β γ X₀ (saProcessLaw β γ X₀) := by
  classical
  refine ⟨inferInstance, ?_, fun t i A S hA hS hpast => ?_⟩
  · apply (ae_map_iff (measurable_saHistoryPath γ X₀).aemeasurable
      (measurableSet_eq_fun (measurable_pi_apply 0) measurable_const)).mpr
    filter_upwards [saHistoryLaw_initial β γ X₀] with h hh
    change saHistoryConfig γ X₀ (h 0) = X₀
    rw [hh]
    rfl
  · let f : (ℕ → Idx n → EucSpace d) → ℝ := fun x =>
      ∑ j : Idx n, attWeight β (x t) i j *
        A.indicator (fun _ => (1 : ℝ)) ((1 - γ) • x t i + γ • x t j)
    have hm : Measurable f := Finset.measurable_sum _ fun j _ =>
      ((continuous_attWeight β i j).measurable.comp (measurable_pi_apply t)).mul
        ((measurable_const.indicator hA).comp (by fun_prop))
    have hnn (x : ℕ → Idx n → EucSpace d) : 0 ≤ f x :=
      Finset.sum_nonneg fun j _ => mul_nonneg (attWeight_pos β (x t) i j).le
        (Set.indicator_nonneg (fun _ _ => zero_le_one) _)
    have hint : Integrable f ((saProcessLaw β γ X₀).restrict S) := by
      apply Integrable.of_bound hm.aestronglyMeasurable 1
      filter_upwards with x
      rw [Real.norm_eq_abs, abs_of_nonneg (hnn x)]
      calc f x ≤ ∑ j : Idx n, attWeight β (x t) i j * 1 :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left
            (Set.indicator_le_self' (fun _ _ => zero_le_one) _) (attWeight_pos β (x t) i j).le
        _ = 1 := by simp [sum_attWeight]
    have hof (x : ℕ → Idx n → EucSpace d) : ENNReal.ofReal (f x) =
        ∑ j : Idx n, ENNReal.ofReal (attWeight β (x t) i j) *
          A.indicator (fun _ => (1 : ℝ≥0∞)) ((1 - γ) • x t i + γ • x t j) := by
      rw [ENNReal.ofReal_sum_of_nonneg (fun j _ => mul_nonneg (attWeight_pos β (x t) i j).le
        (Set.indicator_nonneg (fun _ _ => zero_le_one) _))]
      apply Finset.sum_congr rfl
      intro j _
      rw [ENNReal.ofReal_mul (attWeight_pos β (x t) i j).le]
      congr 1
      by_cases hx : (1 - γ) • x t i + γ • x t j ∈ A <;> simp [Set.indicator, hx]
    have hnext : MeasurableSet {x : ℕ → Idx n → EucSpace d | x (t + 1) i ∈ A} :=
      hA.preimage ((measurable_pi_apply i).comp (measurable_pi_apply (t + 1)))
    have htrans : saProcessLaw β γ X₀ (S ∩ {x | x (t + 1) i ∈ A}) =
        ∫⁻ x in S, ENNReal.ofReal (f x) ∂saProcessLaw β γ X₀ := by
      rw [saProcessLaw, Measure.map_apply (measurable_saHistoryPath γ X₀) (hS.inter hnext),
        setLIntegral_map (f := fun x => ENNReal.ofReal (f x)) hS
          (ENNReal.measurable_ofReal.comp hm) (measurable_saHistoryPath γ X₀)]
      simp only [hof]
      exact saHistoryLaw_transition β γ X₀ t i A S ⟨hS, hpast⟩
    rw [← ofReal_integral_eq_lintegral_ofReal hint (Filter.Eventually.of_forall hnn)] at htrans
    rw [htrans, ENNReal.toReal_ofReal (integral_nonneg hnn)]

/-- The past-event hypotheses hold jointly for the whole actual path space;
the same two distinct original tokens admit the complete genuine process. -/
example : PastDet 2 (univ : Set (ℕ → Idx 2 → EucSpace 1)) ∧
    IsSAProcess 1 (1 / 2) (![(0 : EucSpace 1), EuclideanSpace.single (0 : Fin 1) (1 : ℝ)])
      (saProcessLaw 1 (1 / 2) (![(0 : EucSpace 1), EuclideanSpace.single (0 : Fin 1) (1 : ℝ)])) :=
  ⟨pastDet_univ 2, isSAProcess_saProcessLaw _ _ _⟩

end Transformer.FrankWolfe
