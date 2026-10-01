import PFR.ForMathlib.Entropy.Basic

/-!
# Shannon memorization in bits

arXiv:2505.24832v3, Section 2.1. The source's `I([X | Θ], Θ̂)` is
conditional mutual information, not mutual information with a new random
variable called `X | Θ`. All entropies below use actual distributions.
PFR uses natural logarithms; division by `log 2` converts to bits.
-/

namespace Transformer.Memorization

open MeasureTheory ProbabilityTheory

variable {Ω A B C : Type*} [MeasurableSpace Ω] [MeasurableSpace A]
  [MeasurableSpace B] [MeasurableSpace C]

/-- Section 2.1: the information the trained model retains about the data. -/
noncomputable def shannonMem (μ : Measure Ω) (X : Ω → A) (trained : Ω → B) : ℝ :=
  mutualInfo X trained μ / Real.log 2

/-- Section 2.1: dataset information remaining after conditioning on the
ground-truth model. -/
noncomputable def shannonUnintended (μ : Measure Ω) (X : Ω → A)
    (trained : Ω → B) (reference : Ω → C) : ℝ :=
  condMutualInfo X trained reference μ / Real.log 2

/-- Section 2.1: intended memorization is the difference of total and
unintended memorization, as specified by the paper. -/
noncomputable def shannonGeneralization (μ : Measure Ω) (X : Ω → A)
    (trained : Ω → B) (reference : Ω → C) : ℝ :=
  shannonMem μ X trained - shannonUnintended μ X trained reference

/-- Section 2.1: the intended/unintended decomposition is exact. -/
theorem shannon_decomposition (μ : Measure Ω) (X : Ω → A)
    (trained : Ω → B) (reference : Ω → C) :
    shannonGeneralization μ X trained reference +
      shannonUnintended μ X trained reference = shannonMem μ X trained := by
  unfold shannonGeneralization
  ring

variable [Fintype A] [Fintype B] [Fintype C]
  [MeasurableSingletonClass A] [MeasurableSingletonClass B] [MeasurableSingletonClass C]

/-- Section 2.1: mutual-information memorization is nonnegative. -/
theorem shannonMem_nonneg (μ : Measure Ω) (X : Ω → A) (trained : Ω → B)
    (hX : Measurable X) (htrained : Measurable trained) :
    0 ≤ shannonMem μ X trained := by
  exact div_nonneg (mutualInfo_nonneg hX htrained μ) (Real.log_pos (by norm_num)).le

/-- Section 2.1: the measurability hypotheses hold for constant variables. -/
example : Measurable (fun _ : Unit => ()) ∧ Measurable (fun _ : Unit => ()) :=
  ⟨measurable_const, measurable_const⟩

omit [Fintype C] [MeasurableSingletonClass C] in
/-- Section 2.1: conditional-information memorization is nonnegative. -/
theorem shannonUnintended_nonneg (μ : Measure Ω) (X : Ω → A)
    (trained : Ω → B) (reference : Ω → C)
    (hX : Measurable X) (htrained : Measurable trained) :
    0 ≤ shannonUnintended μ X trained reference := by
  exact div_nonneg (condMutualInfo_nonneg hX htrained) (Real.log_pos (by norm_num)).le

/-- Section 2.1: this theorem also admits genuine constant random variables. -/
example : Measurable (fun _ : Unit => ()) ∧ Measurable (fun _ : Unit => ()) :=
  ⟨measurable_const, measurable_const⟩

/-- Section 2.1: the displayed entropy-difference formula is conditional
mutual information. Finite alphabets make all entropies finite. -/
theorem shannonUnintended_eq (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → A) (trained : Ω → B) (reference : Ω → C)
    (hX : Measurable X) (htrained : Measurable trained) (href : Measurable reference) :
    shannonUnintended μ X trained reference =
      (condEntropy X reference μ -
        condEntropy X (fun ω => (trained ω, reference ω)) μ) / Real.log 2 := by
  rw [shannonUnintended, condMutualInfo_eq' hX htrained href]

/-- Section 2.1: a Dirac probability measure witnesses all side conditions. -/
example : IsProbabilityMeasure (Measure.dirac ()) ∧
    Measurable (fun _ : Unit => ()) ∧ Measurable (fun _ : Unit => ()) ∧
    Measurable (fun _ : Unit => ()) :=
  ⟨inferInstance, measurable_const, measurable_const, measurable_const⟩

/-- Section 2.1, Proposition 1: unintended memorization is at most the
entropy of the trained model, without any independence requirement. -/
theorem shannonUnintended_le_model_entropy (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → A) (trained : Ω → B) (reference : Ω → C)
    (hX : Measurable X) (htrained : Measurable trained) (href : Measurable reference) :
    shannonUnintended μ X trained reference ≤ entropy trained μ / Real.log 2 := by
  apply div_le_div_of_nonneg_right _ (Real.log_pos (by norm_num)).le
  rw [condMutualInfo_comm hX htrained, condMutualInfo_eq' htrained hX href]
  exact (sub_le_self _ (condEntropy_nonneg _ _ _)).trans
    (condEntropy_le_entropy μ htrained href)

/-- Section 2.1, Proposition 1: the upper-bound hypotheses are satisfiable. -/
example : IsProbabilityMeasure (Measure.dirac ()) ∧
    Measurable (fun _ : Unit => ()) ∧ Measurable (fun _ : Unit => ()) ∧
    Measurable (fun _ : Unit => ()) :=
  ⟨inferInstance, measurable_const, measurable_const, measurable_const⟩

end Transformer.Memorization
