/-
# Mean and variance under minibatch averaging

arXiv:2506.12543v1, Section 4.3, definition of the stochastic gradient
and equation (2). Independence is needed for variance scaling, while
unbiasedness alone determines the mean.
-/

import Transformer.BatchSize.Section4_Coefficients

open MeasureTheory ProbabilityTheory
open scoped BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The actual average of B stochastic gradient coordinates, Section 4.3. -/
def batchMean {Ω : Type*} {B : ℕ} (X : Fin B → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ i, X i ω) / B

/-- Averaging preserves the true gradient, Section 4.3, equation (2).
This does not require Gaussianity or independence. -/
theorem batchMean_expectation {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] {B : ℕ} (hB : 0 < B)
    (X : Fin B → Ω → ℝ) (g : ℝ) (hX : ∀ i, Integrable (X i) P)
    (hmean : ∀ i, (∫ ω, X i ω ∂P) = g) :
    (∫ ω, batchMean X ω ∂P) = g := by
  change (∫ ω, (∑ i, X i ω) / B ∂P) = g
  rw [integral_div, integral_finsetSum _ (fun i _ => hX i)]
  simp only [hmean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hb : (B : ℝ) ≠ 0 := by exact_mod_cast hB.ne'
  exact mul_div_cancel_left₀ _ hb

/-- Nonvacuity of unbiased minibatch gradients, Section 4.3. -/
example : 0 < (2 : ℕ) ∧
    Integrable (fun _ : Unit => (1 : ℝ)) (Measure.dirac ()) ∧
    (∫ _ : Unit, (1 : ℝ) ∂Measure.dirac ()) = 1 := by
  refine ⟨by norm_num, integrable_const _, ?_⟩
  simp

/-- Independent equal-variance samples give sigma^2/B noise covariance,
Section 4.3, equation (2). Pairwise independence suffices. -/
theorem batchMean_variance {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] {B : ℕ} (hB : 0 < B)
    (X : Fin B → Ω → ℝ) (σ : ℝ) (hX : ∀ i, MemLp (X i) 2 P)
    (hind : ∀ i j, i ≠ j → IndepFun (X i) (X j) P)
    (hvar : ∀ i, variance (X i) P = σ ^ 2) :
    variance (batchMean X) P = σ ^ 2 / B := by
  have hsum := IndepFun.variance_sum (s := Finset.univ) (fun i _ => hX i)
    (fun i _ j _ hij => hind i j hij)
  have heq : batchMean X = fun ω => (B : ℝ)⁻¹ * (∑ i, X i ω) := by
    funext ω
    simp [batchMean, div_eq_mul_inv, mul_comm]
  rw [heq, variance_const_mul]
  have hsum' : variance (fun ω => ∑ i, X i ω) P = ∑ i, variance (X i) P := by
    rw [show (∑ i ∈ Finset.univ, X i) = (fun ω => ∑ i, X i ω) from
      by funext ω; simp] at hsum
    simpa using hsum
  rw [hsum']
  simp only [hvar, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hb : (B : ℝ) ≠ 0 := by exact_mod_cast hB.ne'
  field_simp

/-- Nonvacuity of the independence and common-variance hypotheses;
Section 4.3. A constant random variable is independent of all others. -/
example : 0 < (2 : ℕ) ∧
    MemLp (fun _ : Unit => (0 : ℝ)) 2 (Measure.dirac ()) ∧
    IndepFun (fun _ : Unit => (0 : ℝ)) (fun _ : Unit => (0 : ℝ)) (Measure.dirac ()) ∧
    variance (fun _ : Unit => (0 : ℝ)) (Measure.dirac ()) = 0 ^ 2 := by
  refine ⟨by norm_num, memLp_const _, ?_, ?_⟩
  · exact indepFun_const_left _ _
  · simp

end Transformer.BatchSize
