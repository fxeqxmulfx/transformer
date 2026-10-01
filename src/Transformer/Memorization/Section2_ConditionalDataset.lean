import Transformer.Memorization.Section2_EntropyVectors

/-!
# Conditional dataset entropy

arXiv:2505.24832v3, Section 2.1 and Appendix A.6. The appendix clarifies
that samples are independent conditional on the ground-truth model; they
need not be independent before that model is sampled.
-/

namespace Transformer.Memorization

open MeasureTheory ProbabilityTheory
open scoped BigOperators ProbabilityTheory

variable {Ω A B : Type*} [MeasurableSpace Ω] [MeasurableSpace A]
  [MeasurableSpace B]

/-- Section 2.1, Proposition 1, Appendix A.6 footnote: joint independence
on every positive-probability reference-model fiber. Identical conditional
marginals are unnecessary for the super-additivity argument. -/
def ConditionallyIndependent {n : ℕ} (μ : Measure Ω)
    (X : Fin n → Ω → A) (reference : Ω → B) : Prop :=
  ∀ b, μ (reference ⁻¹' {b}) ≠ 0 → iIndepFun X (μ[|reference ⁻¹' {b}])

variable [Fintype A] [Fintype B] [MeasurableSingletonClass A]
  [MeasurableSingletonClass B]

/-- Section 2.1, proof of Proposition 1: conditional subadditivity for the
whole dataset vector. -/
theorem condEntropy_vector_le_sum (μ : Measure Ω) [IsProbabilityMeasure μ]
    {n : ℕ} (X : Fin n → Ω → A) (reference : Ω → B)
    (hX : ∀ i, Measurable (X i)) (href : Measurable reference) :
    condEntropy (fun ω i => X i ω) reference μ ≤
      ∑ i, condEntropy (X i) reference μ := by
  rw [condEntropy_eq_sum_fintype _ _ _ href]
  simp_rw [condEntropy_eq_sum_fintype _ _ _ href]
  rw [Finset.sum_comm]
  simp_rw [← Finset.mul_sum]
  apply Finset.sum_le_sum
  intro b hb
  by_cases hzero : μ (reference ⁻¹' {b}) = 0
  · simp [Measure.real, hzero]
  · let := cond_isProbabilityMeasure hzero
    exact mul_le_mul_of_nonneg_left (entropy_vector_le_sum _ X hX) ENNReal.toReal_nonneg

/-- Section 2.1: two constant dataset coordinates and a constant reference
on a one-point probability space satisfy the side conditions. -/
example : IsProbabilityMeasure (Measure.dirac ()) ∧
    (∀ i : Fin 2, Measurable (fun _ : Unit => i)) ∧
    Measurable (fun _ : Unit => ()) :=
  ⟨inferInstance, fun _ => measurable_const, measurable_const⟩

/-- Section 2.1, proof of Proposition 1: the conditional independence
assumption implies additive entropy before observing the trained model. -/
theorem condEntropy_vector_eq_sum (μ : Measure Ω) [IsProbabilityMeasure μ]
    {n : ℕ} (X : Fin n → Ω → A) (reference : Ω → B)
    (hX : ∀ i, Measurable (X i)) (href : Measurable reference)
    (hind : ConditionallyIndependent μ X reference) :
    condEntropy (fun ω i => X i ω) reference μ =
      ∑ i, condEntropy (X i) reference μ := by
  rw [condEntropy_eq_sum_fintype _ _ _ href]
  simp_rw [condEntropy_eq_sum_fintype _ _ _ href]
  rw [Finset.sum_comm]
  simp_rw [← Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro b hb
  by_cases hzero : μ (reference ⁻¹' {b}) = 0
  · simp [Measure.real, hzero]
  · let := cond_isProbabilityMeasure hzero
    rw [entropy_vector_eq_sum _ X hX (hind b hzero)]

/-- Section 2.1: the conditional-independence premise is satisfiable. -/
example : IsProbabilityMeasure (Measure.dirac ()) ∧
    (∀ i : Fin 1, Measurable (fun _ : Unit => i)) ∧
    Measurable (fun _ : Unit => ()) ∧
    ConditionallyIndependent (Measure.dirac ())
      (fun i : Fin 1 => fun _ : Unit => i) (fun _ => ()) := by
  refine ⟨inferInstance, fun _ => measurable_const, measurable_const, ?_⟩
  intro b hb
  let := cond_isProbabilityMeasure hb
  exact iIndepFun.of_subsingleton

end Transformer.Memorization
