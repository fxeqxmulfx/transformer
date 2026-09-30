/-
# AdaFisher: distributed Kronecker-factor aggregation

arXiv:2405.16397v3, Appendix A.4, `eq:distributedkf`.
Linearity preserves unbiasedness of each factor. Products and min-max
normalization are nonlinear and do not inherit that claim automatically.
-/

import Transformer.AdaFisher.Section3_Algorithm
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open scoped BigOperators
open MeasureTheory

noncomputable section

namespace Transformer.AdaFisher

variable {k d : ℕ}

/-- Equal-weight GPU aggregation, Appendix A.4, `eq:distributedkf`.
The summation index of the source's S factor is corrected from `n` to `k`. -/
def distributedAverage (x : Fin k → Fin d → ℝ) : Fin d → ℝ :=
  fun i => (∑ gpu, x gpu i) / k

/-- Averaging unbiased local KF estimates remains unbiased, Appendix A.4.
Integrability and a positive number of GPUs make the probabilistic claim
well-defined. Independence between the GPU estimates is unnecessary. -/
theorem distributedAverage_unbiased {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (X : Fin k → Ω → Fin d → ℝ) (target : Fin d → ℝ)
    (hk : 0 < k) (hX : ∀ gpu i, Integrable (fun ω => X gpu ω i) μ)
    (hmean : ∀ gpu i, (∫ ω, X gpu ω i ∂μ) = target i) (i : Fin d) :
    (∫ ω, distributedAverage (fun gpu => X gpu ω) i ∂μ) = target i := by
  simp only [distributedAverage, div_eq_mul_inv]
  rw [integral_mul_const, integral_finsetSum _ (fun gpu _ => hX gpu i)]
  simp only [hmean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hk0 : (k : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hk
  field_simp

example : (0 : ℕ) < 2 ∧
    (∀ (gpu : Fin 2) (i : Fin 1), Integrable
      (fun _ : ℝ => (fun (_ : Fin 2) (_ : Fin 1) => (2 : ℝ)) gpu i) (Measure.dirac 0)) ∧
    (∀ (gpu : Fin 2) (i : Fin 1), (∫ _ : ℝ,
      (fun (_ : Fin 2) (_ : Fin 1) => (2 : ℝ)) gpu i ∂Measure.dirac 0) = 2) := by
  refine ⟨by decide, ?_, ?_⟩
  · intro gpu i
    exact integrable_const 2
  · intro gpu i
    simp

/-- Factor aggregation commutes with EMA in exact arithmetic,
Appendix A.4 and §3.2, `eq:expkronfactors`. -/
theorem distributedAverage_factorEMA (γ : ℝ) (old fresh : Fin k → Fin d → ℝ) :
    distributedAverage (fun gpu => factorEMA γ (old gpu) (fresh gpu)) =
      factorEMA γ (distributedAverage old) (distributedAverage fresh) := by
  funext i
  simp only [distributedAverage, factorEMA, Finset.sum_add_distrib, ← Finset.mul_sum]
  ring

/-- The Kronecker product of averaged factors differs from averaging
their products in general, Appendix A.4. The source's unbiasedness claim
is about factors; it does not prove unbiasedness of the normalized EFIM. -/
theorem distributed_product_counterexample :
    let h : Fin 2 → Fin 1 → ℝ := fun gpu _ => 2 * (gpu : ℝ)
    let s : Fin 2 → Fin 1 → ℝ := fun gpu _ => 2 * (gpu : ℝ)
    distributedAverage h 0 * distributedAverage s 0 = 1 ∧
    distributedAverage (fun gpu i => h gpu i * s gpu i) 0 = 2 := by
  norm_num [distributedAverage, Fin.sum_univ_two]

end Transformer.AdaFisher
