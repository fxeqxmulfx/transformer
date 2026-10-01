/-
# The sampled normalized SGD transition

Formalization of arXiv:2602.15322v1, Section 5's update and Appendix A.4.
Gradient samples and masks have an explicit independent product law.
The auxiliary state may contain momentum, EMA scales, and a clock. It is
updated densely before masking the parameter displacement.
-/

import Transformer.Magma.Section5_Paths
import Transformer.Magma.Section2_BlockExpansion

open scoped BigOperators

noncomputable section

namespace Transformer.Magma

variable {Z ι E Aux : Type*} [Fintype Z] [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

omit [NormedAddCommGroup E] [NormedSpace ℝ E] in
/-- Independent minibatch and Bernoulli-mask weights.
Source: arXiv:2602.15322v1, Section 5, conditional sampling. -/
def jointMass (w : Z → ℝ) (p : ℝ) (sample : Z × (ι → Bool)) : ℝ :=
  w sample.1 * maskMass p sample.2

omit [NormedAddCommGroup E] [NormedSpace ℝ E] in
/-- Product-law expectation is the iterated expectation used in the
descent proof. Source: arXiv:2602.15322v1, Appendix A.2 and A.4. -/
theorem jointExpectation (w : Z → ℝ) (p : ℝ) (f : Z → (ι → Bool) → ℝ) :
    finiteExpectation (jointMass w p) (fun sample => f sample.1 sample.2) =
      finiteExpectation w (fun z => maskExpectation p (f z)) := by
  simp [finiteExpectation, jointMass, maskExpectation, Fintype.sum_prod_type,
    Finset.mul_sum, mul_assoc]

omit [NormedAddCommGroup E] [NormedSpace ℝ E] in
/-- The product law has total mass one whenever the minibatch law does.
Source: arXiv:2602.15322v1, Appendix A.4. -/
theorem jointMass_sum (w : Z → ℝ) (hw : ∑ z, w z = 1) (p : ℝ) :
    ∑ sample, jointMass (ι := ι) w p sample = 1 := by
  simp [jointMass, Fintype.sum_prod_type, ← Finset.mul_sum, maskMass_sum, hw]

/-- Normalization holds for one gradient sample and one masked block.
Source: arXiv:2602.15322v1, Section 5. -/
example : (∑ _ : Unit, (1 : ℝ)) = 1 := by simp

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [Fintype Z] [DecidableEq ι] in
/-- Admissible probabilities give nonnegative product weights.
Source: arXiv:2602.15322v1, Section 5. -/
theorem jointMass_nonneg (w : Z → ℝ) (hw0 : ∀ z, 0 ≤ w z)
    (p : ℝ) (hp : 0 ≤ p) (hp' : p ≤ 1) (sample : Z × (ι → Bool)) :
    0 ≤ jointMass w p sample :=
  mul_nonneg (hw0 sample.1) (maskMass_nonneg p hp hp' sample.2)

/-- Product probability hypotheses are jointly satisfiable.
Source: arXiv:2602.15322v1, Algorithm 1, survival p=1/2. -/
example : (∀ _ : Unit, (0 : ℝ) ≤ 1) ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by
  norm_num

/-- The normalized SGD recurrence analyzed in Section 5. candidate
returns the new dense auxiliary state and the damped block directions.
The survival bit affects only the parameter part. Source:
arXiv:2602.15322v1, Section 5, theta_{t+1}, and Algorithm 1's dense state. -/
def normalizedTransition (rate p : ℝ)
    (candidate : E × Aux → Z → Aux × (ι → E))
    (state : E × Aux) (sample : Z × (ι → Bool)) : E × Aux :=
  let pending := candidate state sample.1
  (state.1 - rate • maskedSum p pending.2 sample.2, pending.1)

/-- Expected loss of the actual transition agrees with the nested mask
and gradient expectation. Source: arXiv:2602.15322v1, Appendix A.4. -/
theorem normalizedTransition_expectation (loss : E → ℝ) (rate p : ℝ)
    (w : Z → ℝ) (candidate : E × Aux → Z → Aux × (ι → E)) (state : E × Aux) :
    finiteExpectation (jointMass w p)
      (fun sample => loss (normalizedTransition rate p candidate state sample).1) =
      finiteExpectation w (fun z => maskExpectation p
        (fun mask => loss (state.1 - rate • maskedSum p (candidate state z).2 mask))) := by
  exact jointExpectation (ι := ι) w p
    (fun z mask => loss (state.1 - rate • maskedSum p (candidate state z).2 mask))

end Transformer.Magma
