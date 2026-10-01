/-
# Real analytic preparation: SequenceConvolution

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.Basic
import Mathlib.Analysis.Normed.Lp.lpSpace
import Mathlib.Analysis.Normed.Operator.NormedSpace
import Mathlib.Analysis.Normed.Ring.InfiniteSum
import Mathlib.Data.Finsupp.Antidiagonal
import Mathlib.Data.Finsupp.Weight

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Real `ℓ¹` coefficients indexed by `I`. -/
abbrev L1Coeff (I : Type*) := lp (fun _ : I ↦ ℝ) 1

namespace L1Coeff

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma summable_norm {I : Type*} (f : L1Coeff I) : Summable (fun i ↦ ‖f i‖) := by
  simpa using (lp.memℓp f).summable (p := (1 : ℝ≥0∞)) (by norm_num)

end L1Coeff

namespace L1Coeff

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma norm_eq_tsum_norm {I : Type*} (f : L1Coeff I) : ‖f‖ = ∑' i, ‖f i‖ := by
  simpa using lp.norm_eq_tsum_rpow (p := (1 : ℝ≥0∞)) (by norm_num) f

end L1Coeff

/-- Antidiagonal Cauchy product of two `ℓ¹` coefficient families. -/
def convolutionFun (f g : L1Coeff A) (n : A) : ℝ :=
  ∑ kl ∈ Finset.antidiagonal n, f kl.1 * g kl.2

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma summable_antidiagonal_norm_product (f g : L1Coeff A) :
    Summable (fun n : A ↦ ∑ kl ∈ Finset.antidiagonal n, ‖f kl.1‖ * ‖g kl.2‖) := by
  apply summable_sum_mul_antidiagonal_of_summable_mul
    (A := A) (α := ℝ) (f := fun a ↦ ‖f a‖) (g := fun a ↦ ‖g a‖)
  exact (L1Coeff.summable_norm f).mul_of_nonneg (L1Coeff.summable_norm g)
      (fun _ ↦ norm_nonneg _) (fun _ ↦ norm_nonneg _)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma summable_norm_convolutionFun (f g : L1Coeff A) :
    Summable (fun n ↦ ‖convolutionFun f g n‖) := by
  refine (summable_antidiagonal_norm_product f g).of_nonneg_of_le
    (fun _ ↦ norm_nonneg _) (fun n ↦ ?_)
  calc
    ‖convolutionFun f g n‖ ≤
        ∑ kl ∈ Finset.antidiagonal n, ‖f kl.1 * g kl.2‖ := norm_sum_le _ _
    _ ≤ ∑ kl ∈ Finset.antidiagonal n, ‖f kl.1‖ * ‖g kl.2‖ := by
      gcongr with kl hkl
      exact norm_mul_le _ _

/-- Antidiagonal convolution as an `ℓ¹` coefficient family. -/
def convolution (f g : L1Coeff A) : L1Coeff A :=
  ⟨convolutionFun f g, by
    apply memℓp_gen
    simpa using summable_norm_convolutionFun f g⟩

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
lemma convolution_apply (f g : L1Coeff A) (n : A) :
    convolution f g n = ∑ kl ∈ Finset.antidiagonal n, f kl.1 * g kl.2 := rfl

/-- The ordinary `ℓ¹` Cauchy-product estimate.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_convolution_le (f g : L1Coeff A) : ‖convolution f g‖ ≤ ‖f‖ * ‖g‖ := by
  rw [L1Coeff.norm_eq_tsum_norm, L1Coeff.norm_eq_tsum_norm, L1Coeff.norm_eq_tsum_norm]
  have hf := L1Coeff.summable_norm f
  have hg := L1Coeff.summable_norm g
  have hprod : Summable (fun x : A × A ↦ ‖f x.1‖ * ‖g x.2‖) :=
    hf.mul_of_nonneg hg (fun _ ↦ norm_nonneg _) (fun _ ↦ norm_nonneg _)
  have hant := summable_antidiagonal_norm_product f g
  have hconv := summable_norm_convolutionFun f g
  calc
    (∑' n, ‖convolution f g n‖) ≤
        ∑' n, ∑ kl ∈ Finset.antidiagonal n, ‖f kl.1‖ * ‖g kl.2‖ := by
      apply hconv.tsum_le_tsum
      · intro n
        exact calc
          ‖convolution f g n‖ ≤
              ∑ kl ∈ Finset.antidiagonal n, ‖f kl.1 * g kl.2‖ := norm_sum_le _ _
          _ ≤ ∑ kl ∈ Finset.antidiagonal n, ‖f kl.1‖ * ‖g kl.2‖ := by
            gcongr with kl hkl
            exact norm_mul_le _ _
      · exact hant
    _ = (∑' a, ‖f a‖) * ∑' a, ‖g a‖ := by
      symm
      exact hf.tsum_mul_tsum_eq_tsum_sum_antidiagonal hg hprod

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma convolution_add_left (f₁ f₂ g : L1Coeff A) :
    convolution (f₁ + f₂) g = convolution f₁ g + convolution f₂ g := by
  apply lp.ext
  funext n
  simp only [convolution_apply, lp.coeFn_add, Pi.add_apply, add_mul, Finset.sum_add_distrib]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma convolution_smul_left (c : ℝ) (f g : L1Coeff A) :
    convolution (c • f) g = c • convolution f g := by
  apply lp.ext
  funext n
  simp only [convolution_apply, lp.coeFn_smul, Pi.smul_apply, smul_eq_mul, mul_assoc,
    Finset.mul_sum]

/-- Right convolution as a linear map. -/
def convolutionRightLinear (g : L1Coeff A) : L1Coeff A →ₗ[ℝ] L1Coeff A where
  toFun f := convolution f g
  map_add' f₁ f₂ := convolution_add_left f₁ f₂ g
  map_smul' c f := convolution_smul_left c f g

/-- Right convolution as a continuous linear map. -/
def convolutionRight (g : L1Coeff A) : L1Coeff A →L[ℝ] L1Coeff A :=
  (convolutionRightLinear g).mkContinuous ‖g‖ (fun f ↦ by
    change ‖convolution f g‖ ≤ ‖g‖ * ‖f‖
    simpa [mul_comm] using norm_convolution_le f g)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
lemma convolutionRight_apply (g f : L1Coeff A) : convolutionRight g f = convolution f g := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_convolutionRight_le (g : L1Coeff A) : ‖convolutionRight g‖ ≤ ‖g‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _)
  intro f
  change ‖convolution f g‖ ≤ ‖g‖ * ‖f‖
  simpa [mul_comm] using norm_convolution_le f g

end Transformer.AnalyticPreparation
