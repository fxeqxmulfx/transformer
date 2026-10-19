module

public import Transformer.Kinetic.Section6_InteriorMinimum
public import Transformer.Kinetic.Section6_PositiveCoefficients

/-!
# Kinetic theory for Transformers — the original Fourier aggregate

The profile `ℋ` from arXiv:2605.09213v1, §6, is built from the actual
Gaussian weights `exp(-π²(n+1)²/(2M²))` and actual eigenvalue parameters
`c_{n+1}=t a_{n+1}`. Their positivity, summability, boundedness and decay
establish the coefficient hypotheses of the preceding entire-series
proofs. In particular, the non-strict source slope bound still gives a
strictly negative aggregate slope because some actual eigenvalue is
strictly below the positive threshold.
-/

@[expose] public section

open scoped BigOperators Topology
open Set Filter Real

namespace Transformer
namespace Kinetic

/-- The original Gaussian coefficient of the positive Fourier mode `n+1`.
Source: arXiv:2605.09213v1, `eq:Sb-def`. -/
noncomputable def gaussianModeWeight (M : ℝ) (n : ℕ) : ℝ :=
  Real.exp (-(π ^ 2 / (2 * M ^ 2)) * ((n : ℝ) + 1) ^ 2)

/-- The actual parameter `c_{n+1}=t a_{n+1}` of the original output mode.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def outputModeParameter (β t : ℝ) (n : ℕ) : ℝ := t * aCoeff β (n + 1)

/-- The source's aggregate `ℋ`, with the original Fourier coefficients.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def softProfile (β lam M t : ℝ) : ℝ → ℝ :=
  seriesProfile lam (weightedPsiCoeffs (gaussianModeWeight M) (outputModeParameter β t))

/-- Every original positive Fourier-mode weight is strictly positive.
Source: arXiv:2605.09213v1, `eq:Sb-def`. -/
theorem gaussianModeWeight_pos (M : ℝ) (n : ℕ) : 0 < gaussianModeWeight M n := Real.exp_pos _

/-- The actual original Gaussian weights are summable for `M > 0`.
Source: arXiv:2605.09213v1, `eq:Sb-def`, and §6, the aggregate `ℋ`. -/
theorem summable_gaussianModeWeight {M : ℝ} (hM : 0 < M) : Summable (gaussianModeWeight M) := by
  unfold gaussianModeWeight
  apply Real.summable_exp_nat_mul_of_ge (neg_lt_zero.mpr (by positivity : 0 < π ^ 2 / (2 * M ^ 2)))
  intro n
  have hn : (0 : ℝ) ≤ n := by positivity
  nlinarith

example := summable_gaussianModeWeight one_pos

/-- The original modes are positive under the source standing assumptions.
Source: arXiv:2605.09213v1, §6, `c_n=a_n t`. -/
theorem outputModeParameter_pos {β t : ℝ} (hβ : 0 < β) (ht : 0 < t) (n : ℕ) :
    0 < outputModeParameter β t n := mul_pos ht (aCoeff_pos hβ (by omega))

example := outputModeParameter_pos one_pos one_pos 0

/-- Actual output parameters tend to zero, at every `β` and finite time.
Source: arXiv:2605.09213v1, §6, the Fourier-decay input to `thm:U-shape`. -/
theorem outputModeParameter_tendsto_zero (β t : ℝ) :
    Tendsto (outputModeParameter β t) atTop (𝓝 0) := by
  change Tendsto (fun n => t * aCoeff β (n + 1)) atTop (𝓝 0)
  simpa only [Function.comp_def, mul_zero] using
    ((aCoeff_tendsto_zero β).comp (tendsto_add_atTop_nat 1)).const_mul t

/-- Some actual mode lies strictly below the positive slope threshold,
including when other modes attain the source's non-strict bound.
Source: arXiv:2605.09213v1, §6, correcting the slope step in `thm:U-shape`. -/
theorem exists_strict_outputModeParameter (β t : ℝ) {lam : ℝ} (hlam : 0 < lam) :
    ∃ n : ℕ, outputModeParameter β t n < 2 * (1 - Real.exp (-lam)) := by
  have hd : 0 < 2 * (1 - Real.exp (-lam)) := by
    linarith [Real.exp_lt_one_iff.mpr (by linarith : -lam < 0)]
  exact ((outputModeParameter_tendsto_zero β t).eventually_lt_const hd).exists

example := exists_strict_outputModeParameter 1 1 one_pos

/-- The original coefficient aggregate is entire under any uniform bound
on its positive mode parameters. Source: arXiv:2605.09213v1, §6, the aggregate `ℋ`. -/
theorem isEntire_softProfileCoeffs {β M t C : ℝ} (hβ : 0 < β) (hM : 0 < M) (ht : 0 < t)
    (hC : ∀ n, outputModeParameter β t n ≤ C) :
    IsEntire (weightedPsiCoeffs (gaussianModeWeight M) (outputModeParameter β t)) :=
  isEntire_weightedPsiCoeffs (summable_gaussianModeWeight hM)
    (fun n => (gaussianModeWeight_pos M n).le) (fun n => (outputModeParameter_pos hβ ht n).le) hC

/-- The actual original soft correction is `ℋ(Y(σ₀))` throughout `(0,1)`.
Source: arXiv:2605.09213v1, §6, the output-layer identity before `thm:U-shape`. -/
theorem softCorrection_eq_softProfile {β lam M t C σ₀ : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C)
    (hσ₀ : 0 < σ₀) (hσ₁ : σ₀ < 1) :
    softCorrection β lam M t σ₀ = softProfile β lam M t (Yfun lam 1 σ₀) := by
  rw [softProfile, seriesProfile_weighted_eq (summable_gaussianModeWeight hM)
    (fun n => (gaussianModeWeight_pos M n).le) (fun n => (outputModeParameter_pos hβ ht n).le) hC]
  unfold softCorrection
  congr 1
  ext n
  rw [hardyProfile_one_eq hlam hσ₀ hσ₁
    (mul_nonneg (aCoeff_pos hβ (by omega)).le ht.le)]
  simp only [gaussianModeWeight, outputModeParameter, mul_comm t]

/-- Under the original non-strict slope bound the actual aggregate has a
strictly negative endpoint slope. Source: arXiv:2605.09213v1, §6, `eq:affine-smallness`. -/
theorem softProfile_initialSlope_neg {β lam M t C : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C)
    (hsmall : ∀ n, outputModeParameter β t n ≤ 2 * (1 - Real.exp (-lam))) :
    seriesDeriv lam (weightedPsiCoeffs (gaussianModeWeight M) (outputModeParameter β t)) 0 < 0 := by
  obtain ⟨i, hi⟩ := exists_strict_outputModeParameter β t hlam
  exact seriesDeriv_weighted_zero_neg (summable_gaussianModeWeight hM)
    (fun n => (gaussianModeWeight_pos M n).le) (fun n => (outputModeParameter_pos hβ ht n).le)
    hC hlam hsmall i (gaussianModeWeight_pos M i) (outputModeParameter_pos hβ ht i) hi

/-- The actual original aggregate has a unique positive global minimum.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem softProfile_unique_minimum {β lam M t : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t)
    (hcurv : ∀ n, outputModeParameter β t n ≤ 3 - Real.sqrt 3)
    (hslope : ∀ n, outputModeParameter β t n ≤ 2 * (1 - Real.exp (-lam))) :
    ∃! y : ℝ, 0 < y ∧ ∀ z : ℝ, 0 < z → softProfile β lam M t y ≤ softProfile β lam M t z := by
  obtain ⟨i, hi⟩ := exists_strict_outputModeParameter β t hlam
  exact exists_unique_minimum_weightedProfile (summable_gaussianModeWeight hM)
    (fun n => (gaussianModeWeight_pos M n).le) (fun n => (outputModeParameter_pos hβ ht n).le)
    hcurv hlam hslope i (gaussianModeWeight_pos M i) (outputModeParameter_pos hβ ht i) hi

/-- The actual original aggregate profile is differentiable at every real
argument with its proved differentiated coefficient formula.
Source: arXiv:2605.09213v1, §6, the differentiation of `ℋ`. -/
theorem hasDerivAt_softProfile {β M t C : ℝ} (hβ : 0 < β) (hM : 0 < M) (ht : 0 < t)
    (hC : ∀ n, outputModeParameter β t n ≤ C) (lam y : ℝ) :
    HasDerivAt (softProfile β lam M t)
      (seriesDeriv lam (weightedPsiCoeffs (gaussianModeWeight M) (outputModeParameter β t)) y) y :=
  hasDerivAt_seriesProfile (isEntire_softProfileCoeffs hβ hM ht hC) lam y

/-- Continuity of the original aggregate, including its series endpoint extension.
Source: arXiv:2605.09213v1, §6, the continuous profile `ℋ`. -/
theorem continuous_softProfile {β M t C : ℝ} (hβ : 0 < β) (hM : 0 < M) (ht : 0 < t)
    (hC : ∀ n, outputModeParameter β t n ≤ C) (lam : ℝ) :
    Continuous (softProfile β lam M t) :=
  continuous_seriesProfile (isEntire_softProfileCoeffs hβ hM ht hC) lam

/-- The original Fourier aggregate tends to infinity at infinity.
Source: arXiv:2605.09213v1, §6, the growth step in `thm:U-shape`. -/
theorem softProfile_tendsto_atTop {β lam M t C : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t) (hC : ∀ n, outputModeParameter β t n ≤ C) :
    Tendsto (softProfile β lam M t) atTop atTop :=
  weightedProfile_tendsto_atTop (summable_gaussianModeWeight hM)
    (fun n => (gaussianModeWeight_pos M n).le) (fun n => (outputModeParameter_pos hβ ht n).le)
    hC hlam 0 (gaussianModeWeight_pos M 0) (outputModeParameter_pos hβ ht 0)

/-- The original `β = λ = M = 1` modes, with a sufficiently small positive
time, witness every source parameter bound used above.
Source: arXiv:2605.09213v1, `eq:affine-smallness`. -/
example : ∃ t : ℝ, 0 < t ∧ (∀ n, outputModeParameter 1 t n ≤ 3 - Real.sqrt 3) ∧
    (∀ n, outputModeParameter 1 t n ≤ 2 * (1 - Real.exp (-1))) := by
  obtain ⟨t, ht, hb⟩ := exists_affine_smallness 1 1 one_pos
  exact ⟨t, ht, fun n => (hb (n + 1) (by omega)).trans (min_le_left _ _),
    fun n => (hb (n + 1) (by omega)).trans (min_le_right _ _)⟩

end Kinetic
end Transformer
