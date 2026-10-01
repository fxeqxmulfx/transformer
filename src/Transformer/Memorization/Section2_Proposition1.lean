import Transformer.Memorization.Section2_ConditionalDataset

/-!
# Super-additivity of unintended memorization

arXiv:2505.24832v3, Section 2.1, Proposition 1; Appendix A.6.
-/

namespace Transformer.Memorization

open MeasureTheory ProbabilityTheory
open scoped BigOperators ProbabilityTheory

variable {Ω A B C : Type*} [MeasurableSpace Ω] [MeasurableSpace A]
  [MeasurableSpace B] [MeasurableSpace C] [Fintype A] [Fintype B] [Fintype C]
  [MeasurableSingletonClass A] [MeasurableSingletonClass B] [MeasurableSingletonClass C]

/-- Section 2.1, Proposition 1 (finite-alphabet version). The paper says
"i.i.d."; its Appendix A.6 specifies independence conditional on Θ.
This essential qualification is explicit here. Measurability and finite
alphabets ensure that every information quantity is a genuine finite
entropy. Both inequalities are measured in bits. -/
theorem proposition1_superadditivity (μ : Measure Ω) [IsProbabilityMeasure μ]
    {n : ℕ} (X : Fin n → Ω → A) (trained : Ω → B) (reference : Ω → C)
    (hX : ∀ i, Measurable (X i)) (htrained : Measurable trained)
    (href : Measurable reference) (hind : ConditionallyIndependent μ X reference) :
    (∑ i, shannonUnintended μ (X i) trained reference) ≤
      shannonUnintended μ (fun ω i => X i ω) trained reference ∧
    shannonUnintended μ (fun ω i => X i ω) trained reference ≤
      entropy trained μ / Real.log 2 := by
  have hvec : Measurable (fun ω i => X i ω) := by fun_prop
  constructor
  · unfold shannonUnintended
    rw [← Finset.sum_div]
    apply div_le_div_of_nonneg_right _ (Real.log_pos (by norm_num)).le
    rw [condMutualInfo_eq' hvec htrained href,
      condEntropy_vector_eq_sum μ X reference hX href hind]
    simp_rw [condMutualInfo_eq' (hX _) htrained href]
    rw [Finset.sum_sub_distrib]
    exact sub_le_sub_left (condEntropy_vector_le_sum μ X
      (fun ω => (trained ω, reference ω)) hX (htrained.prodMk href)) _
  · exact shannonUnintended_le_model_entropy μ _ _ _ hvec htrained href

/-- Section 2.1, Proposition 1: all hypotheses hold for a constant
one-sample dataset, trained model, and reference model on a Dirac space. -/
example : IsProbabilityMeasure (Measure.dirac ()) ∧
    (∀ i : Fin 1, Measurable (fun _ : Unit => i)) ∧
    Measurable (fun _ : Unit => ()) ∧ Measurable (fun _ : Unit => ()) ∧
    ConditionallyIndependent (Measure.dirac ())
      (fun i : Fin 1 => fun _ : Unit => i) (fun _ => ()) := by
  refine ⟨inferInstance, fun _ => measurable_const, measurable_const, measurable_const, ?_⟩
  intro b hb
  let := cond_isProbabilityMeasure hb
  exact iIndepFun.of_subsingleton

end Transformer.Memorization
