/-
# Exponentiating the original overlap observable

ArXiv:2604.01978v1, §7, the martingale maximal inequality after
eq:Gronwall_E_selfcontained and the observable of lem:Ito_formula.
An exponential supermartingale must be obtained from the actual
martingale problem, rather than assumed from the desired tail.
Here its smooth test function exp(c R_ij) is differentiated twice.

The Riemannian Hessian retains the original sphere correction.
Its chain rule separates c Hess R from c²(DR)², so the subsequent
generator calculation uses exactly the original overlap drift and
noise second moment. Unit configurations also give genuine uniform
bounds on the observable and its first derivative.
-/

import Transformer.Homogenized.OverlapObservable
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual exponential test of the original pair overlap.
Source: arXiv:2604.01978v1, §7, the maximal-inequality step. -/
noncomputable def exponentialOverlap {d n : ℕ} (c : ℝ) (i j : Idx n)
    (X : Idx n → EucSpace d) : ℝ := Real.exp (c * inner (𝕜 := ℝ) (X i) (X j))

/-- This original observable is a valid smooth martingale-problem test.
Source: arXiv:2604.01978v1, §7, the maximal inequality and def:ito.formula. -/
theorem contDiff_exponentialOverlap {d n : ℕ} (c : ℝ) (i j : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (exponentialOverlap (d := d) c i j) :=
  (contDiff_const.mul ((contDiff_apply ℝ (EucSpace d) i).inner ℝ
    (contDiff_apply ℝ (EucSpace d) j))).exp

/-- Its actual first derivative, as a continuous linear map.
Source: arXiv:2604.01978v1, §7, lem:Ito_formula and the maximal inequality. -/
theorem hasFDerivAt_exponentialOverlap {d n : ℕ} (c : ℝ) (i j : Idx n) (X : Idx n → EucSpace d) :
    HasFDerivAt (exponentialOverlap c i j) ((c * exponentialOverlap c i j X) • overlapForm i j X) X := by
  have h := ((hasFDerivAt_overlap i j X).const_mul c).exp
  apply h.congr_fderiv
  ext U
  simp only [smul_apply, smul_eq_mul, exponentialOverlap]
  ring

/-- The first derivative used by the original generator.
Source: arXiv:2604.01978v1, §7, def:ito.formula. -/
theorem fderiv_exponentialOverlap {d n : ℕ} (c : ℝ) (i j : Idx n) (X : Idx n → EucSpace d) :
    fderiv ℝ (exponentialOverlap c i j) X = (c * exponentialOverlap c i j X) • overlapForm i j X :=
  (hasFDerivAt_exponentialOverlap c i j X).fderiv

/-- The actual second derivative contains the original overlap Hessian
and the square of its original first derivative.
Source: arXiv:2604.01978v1, §7, lem:Ito_formula and the maximal inequality. -/
theorem iteratedFDeriv_two_exponentialOverlap {d n : ℕ} (c : ℝ) (i j : Idx n)
    (X V : Idx n → EucSpace d) :
    iteratedFDeriv ℝ 2 (exponentialOverlap c i j) X ![V, V] =
      exponentialOverlap c i j X *
        (2 * c * inner (𝕜 := ℝ) (V i) (V j) + c ^ 2 * (overlapForm i j X V) ^ 2) := by
  have hfirst : fderiv ℝ (exponentialOverlap (d := d) c i j) =
      fun Y => (c * exponentialOverlap c i j Y) • overlapForm i j Y :=
    funext (fderiv_exponentialOverlap c i j)
  have hc := (hasFDerivAt_exponentialOverlap c i j X).const_mul c
  have hd := hc.smul (overlapForm (d := d) i j).hasFDerivAt
  change HasFDerivAt (fun Y => (c * exponentialOverlap c i j Y) • overlapForm i j Y) _ X at hd
  rw [iteratedFDeriv_two_apply, hfirst, hd.fderiv]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, add_apply, smul_apply,
    ContinuousLinearMap.smulRight_apply, smul_eq_mul, overlapForm_apply]
  ring

/-- The same chain rule with the actual Riemannian sphere correction.
Source: arXiv:2604.01978v1, §7, def:ito.formula and the maximal inequality. -/
theorem sphHess_exponentialOverlap {d n : ℕ} (c : ℝ) (i j : Idx n) (X V : Idx n → EucSpace d) :
    sphHess (exponentialOverlap c i j) X V = exponentialOverlap c i j X *
      (c * sphHess (fun Y : Idx n → EucSpace d => inner (𝕜 := ℝ) (Y i) (Y j)) X V +
        c ^ 2 * (overlapForm i j X V) ^ 2) := by
  have hsum : (∑ k, fderiv ℝ (exponentialOverlap c i j) X (Pi.single k (X k)) * ‖V k‖ ^ 2) =
      (c * exponentialOverlap c i j X) * ∑ k, overlapForm i j X (Pi.single k (X k)) * ‖V k‖ ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k hk
    rw [fderiv_exponentialOverlap]
    simp only [smul_apply, smul_eq_mul]
    ring
  rw [sphHess, iteratedFDeriv_two_exponentialOverlap, hsum, sphHess, iteratedFDeriv_two_overlap]
  simp only [(hasFDerivAt_overlap i j X).fderiv]
  ring

/-- The actual exponential observable is always positive.
Source: arXiv:2604.01978v1, §7, the exponential maximal-inequality argument. -/
theorem exponentialOverlap_pos {d n : ℕ} (c : ℝ) (i j : Idx n) (X : Idx n → EucSpace d) :
    0 < exponentialOverlap c i j X := Real.exp_pos _

/-- Unit tokens give the actual uniform bound exp(|c|).
Source: arXiv:2604.01978v1, §7, the bounded original overlap R_ij. -/
theorem exponentialOverlap_le_exp_abs {d n : ℕ} (c : ℝ) (i j : Idx n)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    exponentialOverlap c i j X ≤ Real.exp |c| := by
  have hR : |inner (𝕜 := ℝ) (X i) (X j)| ≤ 1 := by
    simpa [hX i, hX j] using abs_real_inner_le_norm (X i) (X j)
  apply Real.exp_le_exp.mpr
  calc
    _ ≤ |c * inner (𝕜 := ℝ) (X i) (X j)| := le_abs_self _
    _ = |c| * |inner (𝕜 := ℝ) (X i) (X j)| := abs_mul _ _
    _ ≤ |c| := mul_le_of_le_one_right (abs_nonneg _) hR

/-- The original overlap differential is uniformly bounded on unit tokens.
Source: arXiv:2604.01978v1, §7, eq:martingale_Mij_clean. -/
theorem abs_overlapForm_apply_le {d n : ℕ} (i j : Idx n) (X U : Idx n → EucSpace d)
    (hX : ∀ k, ‖X k‖ = 1) : |overlapForm i j X U| ≤ 2 * ‖U‖ := by
  rw [overlapForm_apply]
  calc
    _ ≤ |inner (𝕜 := ℝ) (U i) (X j)| + |inner (𝕜 := ℝ) (X i) (U j)| := abs_add_le _ _
    _ ≤ ‖U i‖ * ‖X j‖ + ‖X i‖ * ‖U j‖ :=
      add_le_add (abs_real_inner_le_norm _ _) (abs_real_inner_le_norm _ _)
    _ = ‖U i‖ + ‖U j‖ := by rw [hX i, hX j]; ring
    _ ≤ 2 * ‖U‖ := by linarith [norm_le_pi_norm U i, norm_le_pi_norm U j]

/-- The first derivative of the actual exponential test has a uniform
bound on the sphere product. Source: arXiv:2604.01978v1, §7,
the bounded test functions in the exponential martingale argument. -/
theorem abs_fderiv_exponentialOverlap_le {d n : ℕ} (c : ℝ) (i j : Idx n)
    (X U : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |fderiv ℝ (exponentialOverlap c i j) X U| ≤ 2 * |c| * Real.exp |c| * ‖U‖ := by
  rw [fderiv_exponentialOverlap, smul_apply, smul_eq_mul, abs_mul, abs_mul,
    abs_of_pos (exponentialOverlap_pos c i j X)]
  have hcoef := mul_le_mul_of_nonneg_left (exponentialOverlap_le_exp_abs c i j X hX) (abs_nonneg c)
  calc
    _ ≤ (|c| * Real.exp |c|) * (2 * ‖U‖) :=
      mul_le_mul hcoef (abs_overlapForm_apply_le i j X U hX) (abs_nonneg _)
        (mul_nonneg (abs_nonneg _) (Real.exp_nonneg _))
    _ = _ := by ring

/-- Unit basis tokens give an actual exponential observable with nonzero
first and second derivatives; none of the differentiated terms is vacuous. -/
example : ∃ X U : Idx 2 → EucSpace 2,
    (∀ k, ‖X k‖ = 1) ∧ exponentialOverlap 1 0 1 X = 1 ∧
    fderiv ℝ (exponentialOverlap 1 0 1) X U = 1 ∧
    iteratedFDeriv ℝ 2 (exponentialOverlap 1 0 1) X ![U, U] = 3 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  let U : Idx 2 → EucSpace 2 := fun _ => EuclideanSpace.single (0 : Idx 2) 1
  have hval : exponentialOverlap 1 0 1 X = 1 := by
    norm_num [exponentialOverlap, X, EuclideanSpace.inner_single_left, PiLp.single_apply]
  refine ⟨X, U, fun k => by simp [X], hval, ?_, ?_⟩
  · rw [fderiv_exponentialOverlap, hval]
    norm_num [overlapForm_apply, X, U, EuclideanSpace.inner_single_left, PiLp.single_apply]
  · rw [iteratedFDeriv_two_exponentialOverlap, hval]
    norm_num [overlapForm_apply, X, U, EuclideanSpace.inner_single_left, PiLp.single_apply]

end Transformer.Homogenized
