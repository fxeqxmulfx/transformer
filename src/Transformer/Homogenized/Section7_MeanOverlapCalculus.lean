/-
# Actual differential and sphere Hessian of the original mean overlap

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
`eq:m_decomp_clean` and `eq:F_def_clean`. The empirical mean
observable is the original normalized sum of all original overlaps.
Its differential and second differential are computed from that sum.

The differential is bundled linearly in the original configuration.
Its application is exactly the average of the genuine pair-overlap
noise integrands. Its diagonal value is twice the original mean overlap.
The genuine ambient second derivative is independent of the base point.

The Riemannian sphere Hessian is the same normalized average of the
original overlap sphere Hessians, including their actual sphere
correction terms. This identity justifies averaging the original
Gaussian generator; no drift or martingale decomposition is included
as a hypothesis of these calculus statements.
-/

import Transformer.Homogenized.Section7_MeanOverlapDrift

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The genuine mean-overlap differential, bundled linearly in its
original base configuration. Source: arXiv:2604.01978v1,
§7, the averaged noise integrand in `eq:m_decomp_clean`. -/
noncomputable def meanOverlapForm {d n : ℕ} :
    (Idx n → EucSpace d) →L[ℝ] (Idx n → EucSpace d) →L[ℝ] ℝ :=
  ((n : ℝ) ^ 2)⁻¹ • ∑ i : Idx n, ∑ j : Idx n, overlapForm i j

/-- The actual mean differential is the original average of the
pair differentials. Source: arXiv:2604.01978v1, §7,
the averaged martingale integrand after `eq:m_decomp_clean`. -/
theorem meanOverlapForm_apply {d n : ℕ} (X V : Idx n → EucSpace d) :
    meanOverlapForm X V = ((n : ℝ) ^ 2)⁻¹ *
      ∑ i : Idx n, ∑ j : Idx n, overlapForm i j X V := by
  simp only [meanOverlapForm, smul_apply, sum_apply, smul_eq_mul]

/-- The original mean observable has the stated genuine Fréchet derivative.
Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean`,
the differential used in its Itô decomposition. -/
theorem hasFDerivAt_meanOverlap {d n : ℕ} (X : Idx n → EucSpace d) :
    HasFDerivAt (meanOverlap (d := d) (n := n)) (meanOverlapForm X) X := by
  have h := (HasFDerivAt.fun_sum (fun i (_ : i ∈ (Finset.univ : Finset (Idx n))) =>
    HasFDerivAt.fun_sum (fun j (_ : j ∈ (Finset.univ : Finset (Idx n))) =>
      hasFDerivAt_overlap i j X))).const_mul (((n : ℝ) ^ 2)⁻¹)
  change HasFDerivAt (fun Y : Idx n → EucSpace d =>
    ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, inner ℝ (Y i) (Y j)) _ X
  convert h using 1
  simp only [meanOverlapForm, smul_apply, sum_apply]

/-- Evaluation of the actual original mean derivative.
Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean`. -/
theorem fderiv_meanOverlap {d n : ℕ} (X : Idx n → EucSpace d) :
    fderiv ℝ (meanOverlap (d := d) (n := n)) X = meanOverlapForm X :=
  (hasFDerivAt_meanOverlap X).fderiv

/-- The original mean overlap is smooth as an actual ambient observable.
Source: arXiv:2604.01978v1, §7, the smooth test in `eq:m_decomp_clean`. -/
theorem contDiff_meanOverlap {d n : ℕ} :
    ContDiff ℝ (⊤ : ℕ∞) (meanOverlap (d := d) (n := n)) := by
  have hs (i j : Idx n) : ContDiff ℝ (⊤ : ℕ∞)
      (fun X : Idx n → EucSpace d => inner ℝ (X i) (X j)) :=
    (contDiff_apply ℝ (EucSpace d) i).inner ℝ (contDiff_apply ℝ (EucSpace d) j)
  unfold meanOverlap
  fun_prop

/-- The original mean's second derivative is the original bundled
linear differential, independent of its base point.
Source: arXiv:2604.01978v1, §7, the quadratic mean observable. -/
theorem fderiv_fderiv_meanOverlap {d n : ℕ} (X : Idx n → EucSpace d) :
    fderiv ℝ (fderiv ℝ (meanOverlap (d := d) (n := n))) X = meanOverlapForm := by
  have he : fderiv ℝ (meanOverlap (d := d) (n := n)) =
      ⇑(meanOverlapForm (d := d) (n := n)) := funext fderiv_meanOverlap
  rw [he, (meanOverlapForm (d := d) (n := n)).fderiv]

/-- The diagonal second differential is twice the actual original
mean overlap of its direction. Source: arXiv:2604.01978v1,
§7, the quadratic variation of the mean observable. -/
theorem meanOverlapForm_self {d n : ℕ} (V : Idx n → EucSpace d) :
    meanOverlapForm V V = 2 * meanOverlap V := by
  simp only [meanOverlapForm_apply, overlapForm_apply, ← two_mul, meanOverlap, ← Finset.mul_sum]
  ring

/-- The actual second differential is the average of the actual
pair second differentials. Source: arXiv:2604.01978v1,
§7, the Itô correction in `eq:m_decomp_clean`. -/
theorem iteratedFDeriv_two_meanOverlap {d n : ℕ} (X V : Idx n → EucSpace d) :
    iteratedFDeriv ℝ 2 (meanOverlap (d := d) (n := n)) X ![V, V] =
      ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        iteratedFDeriv ℝ 2 (fun Y : Idx n → EucSpace d => inner ℝ (Y i) (Y j)) X ![V, V] := by
  rw [iteratedFDeriv_two_apply, fderiv_fderiv_meanOverlap]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, meanOverlapForm_apply,
    overlapForm_apply, iteratedFDeriv_two_overlap, ← two_mul]

/-- **The actual original sphere Hessian commutes with the mean average.**
Both ambient second derivatives and all genuine sphere correction terms
are retained. Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean`,
obtained by averaging the source's original overlap Itô formula. -/
theorem sphHess_meanOverlap {d n : ℕ} (X V : Idx n → EucSpace d) :
    sphHess (meanOverlap (d := d) (n := n)) X V =
      ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        sphHess (fun Y : Idx n → EucSpace d => inner ℝ (Y i) (Y j)) X V := by
  have hfirst (k : Idx n) :
      fderiv ℝ (meanOverlap (d := d) (n := n)) X (Pi.single k (X k)) =
        ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
          fderiv ℝ (fun Y : Idx n → EucSpace d => inner ℝ (Y i) (Y j)) X (Pi.single k (X k)) := by
    simp only [fderiv_meanOverlap, meanOverlapForm_apply, (hasFDerivAt_overlap _ _ _).fderiv]
  rw [sphHess, iteratedFDeriv_two_meanOverlap]
  simp only [hfirst, sphHess, Finset.sum_sub_distrib, mul_sub, Finset.mul_sum, Finset.sum_mul, mul_assoc]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.sum_comm]

/-- The actual sphere Hessian is continuous jointly in the original
base and direction, as required by the original generator's path integral.
Source: arXiv:2604.01978v1, §7, `eq:m_decomp_clean`. -/
theorem continuous_sphHess_meanOverlap {d n : ℕ} :
    Continuous (fun P : (Idx n → EucSpace d) × (Idx n → EucSpace d) =>
      sphHess (meanOverlap (d := d) (n := n)) P.1 P.2) := by
  simp only [sphHess_meanOverlap, sphHess_overlap]
  fun_prop

/-- The mean's original directional quadratic form is nonnegative,
since its exact diagonal value is twice the actual squared centroid norm.
Source: arXiv:2604.01978v1, §7, the mean quadratic observable. -/
theorem meanOverlapForm_self_nonneg {d n : ℕ} (V : Idx n → EucSpace d) :
    0 ≤ meanOverlapForm V V := by
  rw [meanOverlapForm_self]
  exact mul_nonneg (by norm_num) (meanOverlap_nonneg V)

/-- Actual orthogonal tokens and an actual nonzero direction give a
nonzero diagonal differential and a finite-dimensional sphere correction. -/
example : ∃ X : Idx 2 → EucSpace 2,
    meanOverlapForm X X = 1 ∧
      iteratedFDeriv ℝ 2 (meanOverlap (d := 2) (n := 2)) X ![X, X] = 1 := by
  let X : Idx 2 → EucSpace 2 := fun i => EuclideanSpace.single i 1
  have hm : meanOverlap X = 1 / 2 := by
    norm_num [meanOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]
  refine ⟨X, ?_, ?_⟩
  · rw [meanOverlapForm_self, hm]
    norm_num
  · rw [iteratedFDeriv_two_apply, fderiv_fderiv_meanOverlap]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, meanOverlapForm_self, hm]
    norm_num

end Transformer.Homogenized
