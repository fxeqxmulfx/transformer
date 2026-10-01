import Transformer.Memorization.Section2_Shannon
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Capacity of a learning algorithm

arXiv:2505.24832v3, Section 3.1, Definition 5. The paper uses a maximum
over distributions. A supremum in the extended nonnegative reals is used
here, and agrees with every attained maximum. This represents empty or
unbounded distribution classes without claiming a maximizer exists.
-/

namespace Transformer.Memorization

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {A B : Type*} [MeasurableSpace A] [MeasurableSpace B]

/-- Section 3.1, Definition 5: the supremum of genuine Shannon information
retained by the fixed deterministic learner over input distributions. -/
noncomputable def learningCapacity (train : A → B) : ℝ≥0∞ :=
  ⨆ μ : ProbabilityMeasure A, ENNReal.ofReal (shannonMem (μ : Measure A) id train)

/-- Section 3.1: each measured distribution supplies a lower bound on
theoretical capacity, as emphasized in Section 3.2. -/
theorem measured_information_le_capacity (train : A → B) (μ : ProbabilityMeasure A) :
    ENNReal.ofReal (shannonMem (μ : Measure A) id train) ≤ learningCapacity train :=
  le_iSup (fun ν : ProbabilityMeasure A =>
    ENNReal.ofReal (shannonMem (ν : Measure A) id train)) μ

/-- Section 3.1, Definition 5: when an actual maximizing distribution
exists, the corrected supremum is the paper's maximum. -/
theorem learningCapacity_eq_maximum (train : A → B) (μ : ProbabilityMeasure A)
    (hmax : ∀ ν : ProbabilityMeasure A,
      shannonMem (ν : Measure A) id train ≤ shannonMem (μ : Measure A) id train) :
    learningCapacity train = ENNReal.ofReal (shannonMem (μ : Measure A) id train) := by
  apply le_antisymm
  · exact iSup_le (fun ν => ENNReal.ofReal_le_ofReal (hmax ν))
  · exact measured_information_le_capacity train μ

/-- Section 3.1: a constant learner attains its maximum on a Dirac space. -/
example : ∀ ν : ProbabilityMeasure Unit,
    shannonMem (ν : Measure Unit) id (fun _ => ()) ≤
      shannonMem (Measure.dirac ()) id (fun _ => ()) := by
  intro ν
  have hid : (id : Unit → Unit) = fun _ => () := by funext x; cases x; rfl
  rw [hid]
  simp [shannonMem, mutualInfo, entropy_const]

variable [Fintype A] [Fintype B] [MeasurableSingletonClass A]
  [MeasurableSingletonClass B]

/-- Sections 2.1 and 3.1: for deterministic learning, all information
in the trained model comes from its input. -/
theorem deterministic_mem_eq_model_entropy (μ : Measure A) [IsProbabilityMeasure μ]
    (train : A → B) : shannonMem μ id train = entropy train μ / Real.log 2 := by
  have hm : Measurable train := measurable_of_countable train
  have hp := entropy_prod_comp (X := id) measurable_id μ train
  simp only [Function.comp_id] at hp
  have hc : condEntropy train id μ = 0 := by
    rw [chain_rule'' μ hm measurable_id, entropy_comm hm measurable_id, hp, sub_self]
  rw [shannonMem, mutualInfo_eq_entropy_sub_condEntropy' measurable_id hm, hc, sub_zero]

/-- Sections 2.1 and 3.1: a Dirac distribution satisfies the probability
assumption for deterministic learning. -/
example : IsProbabilityMeasure (Measure.dirac ()) := inferInstance

/-- Section 2.1: intended memorization is nonnegative for a deterministic
learner; this uses the learning relationship, not arbitrary three variables. -/
theorem deterministic_generalization_nonneg (μ : Measure A) [IsProbabilityMeasure μ]
    (train : A → B) {C : Type*} [MeasurableSpace C] [Fintype C]
    [MeasurableSingletonClass C] (reference : A → C) :
    0 ≤ shannonGeneralization μ id train reference := by
  unfold shannonGeneralization
  rw [deterministic_mem_eq_model_entropy]
  exact sub_nonneg.mpr (shannonUnintended_le_model_entropy μ id train reference
    measurable_id (measurable_of_countable _) (measurable_of_countable _))

/-- Section 2.1: a constant learner/reference on a Dirac space is admissible. -/
example : IsProbabilityMeasure (Measure.dirac ()) := inferInstance

/-- Section 3.1, Definition 5 and Proposition 1's storage interpretation:
capacity cannot exceed the logarithm of the number of model states. -/
theorem learningCapacity_le_log_card (train : A → B) :
    learningCapacity train ≤ ENNReal.ofReal (Real.log (Fintype.card B) / Real.log 2) := by
  apply iSup_le
  intro μ
  apply ENNReal.ofReal_le_ofReal
  rw [deterministic_mem_eq_model_entropy]
  exact div_le_div_of_nonneg_right (entropy_le_log_card _ _) (Real.log_pos (by norm_num)).le

/-- Section 3.2, "How does precision affect capacity?": a model stored
in `parameters * bits` binary positions has at most that many Shannon bits
of capacity. This storage upper bound does not identify an empirical bpp. -/
theorem learningCapacity_le_storage_bits (parameters bits : ℕ)
    (train : A → (Fin (parameters * bits) → Bool)) :
    learningCapacity train ≤ (parameters * bits : ℕ) := by
  have h := learningCapacity_le_log_card train
  have hlog : Real.log 2 ≠ 0 := (Real.log_pos (by norm_num)).ne'
  simpa [Fintype.card_fun, Real.log_pow, hlog] using h

end Transformer.Memorization
