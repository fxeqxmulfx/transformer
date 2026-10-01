/-
# Mask moments and increased variance

Formalization of arXiv:2602.15322v1, Section 2, eq:rsu_def, Appendix A.1.
The normalized mask is unbiased; it increases second and third moments.
-/

import Transformer.Magma.Section2_MaskLaw

noncomputable section

namespace Transformer.Magma

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Normalized Bernoulli coefficient. Source: arXiv:2602.15322v1,
Section 2, eq:rsu_def, s=1/p. -/
def maskCoefficient (p : ℝ) (bit : Bool) : ℝ := if bit then 1 / p else 0

/-- Unbiased normalized block updates. Source: arXiv:2602.15322v1,
Section 2, eq:rsu_def, and Appendix A.1. -/
theorem maskCoefficient_mean (p : ℝ) (hp : p ≠ 0) (b : ι) :
    maskExpectation p (fun mask => maskCoefficient p (mask b)) = 1 := by
  rw [maskExpectation_coordinate]
  simp [maskCoefficient, hp]

/-- Survival probabilities are nonzero. Source: arXiv:2602.15322v1,
Section 2. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

/-- The diagonal second moment is amplified by 1/p. Source:
arXiv:2602.15322v1, Appendix A.1, diagonal Hessian term. -/
theorem maskCoefficient_second (p : ℝ) (hp : p ≠ 0) (b : ι) :
    maskExpectation p (fun mask => maskCoefficient p (mask b) ^ 2) = 1 / p := by
  rw [maskExpectation_coordinate p b (fun bit => maskCoefficient p bit ^ 2)]
  simp only [maskCoefficient, ite_true, Bool.false_eq_true, ite_false]
  field_simp
  ring

/-- The second-moment domain is nonempty. Source: arXiv:2602.15322v1,
Section 2, p=1/2 in Algorithm 1. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

/-- Off-diagonal normalized mask moments equal one. Source:
arXiv:2602.15322v1, Appendix A.1, off-diagonal Hessian term. -/
theorem maskCoefficient_cross (p : ℝ) (hp : p ≠ 0) (b c : ι) (hbc : b ≠ c) :
    maskExpectation p (fun mask => maskCoefficient p (mask b) *
      maskCoefficient p (mask c)) = 1 := by
  rw [maskExpectation_pair p b c hbc]
  simp [maskCoefficient, hp]

/-- All cross-moment hypotheses are satisfiable. Source:
arXiv:2602.15322v1, Section 2. -/
example : (1 / 2 : ℝ) ≠ 0 ∧ (0 : Fin 2) ≠ 1 := by norm_num

/-- The cubic moment's explicit p dependence; it is not uniform as p tends
to zero. Source: arXiv:2602.15322v1, Appendix A.1, final paragraph. -/
theorem maskCoefficient_third (p : ℝ) (hp : p ≠ 0) (b : ι) :
    maskExpectation p (fun mask => maskCoefficient p (mask b) ^ 3) = 1 / p ^ 2 := by
  rw [maskExpectation_coordinate p b (fun bit => maskCoefficient p bit ^ 3)]
  simp only [maskCoefficient, ite_true, Bool.false_eq_true, ite_false]
  field_simp
  ring

/-- The cubic-moment hypotheses are satisfiable. Source:
arXiv:2602.15322v1, Section 2. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

/-- Unbiased masking adds variance (1-p)/p; this is increased noise, not a
general variance reduction claim. Source: arXiv:2602.15322v1, Section 2. -/
theorem maskCoefficient_variance (p : ℝ) (hp : p ≠ 0) (b : ι) :
    maskExpectation p (fun mask => (maskCoefficient p (mask b) - 1) ^ 2) =
      (1 - p) / p := by
  rw [maskExpectation_coordinate p b (fun bit => (maskCoefficient p bit - 1) ^ 2)]
  simp only [maskCoefficient, ite_true, Bool.false_eq_true, ite_false]
  field_simp
  ring

/-- The variance hypotheses are satisfiable. Source: arXiv:2602.15322v1,
Algorithm 1, p=1/2. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

end Transformer.Magma
