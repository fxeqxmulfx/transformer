module

public import Transformer.Homogenized.Section7_MeanLogCalculus
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Actual exponential tests of the original regularized log mean

ArXiv:2604.01978v1, §7, proof of `thm:clustering_small_beta`,
the exponential maximal estimate after the mean bracket calculation,
and Appendix, `def:ito.formula`. The genuine smooth test below is
`exp(c log(m(X)+a))` for the original empirical mean and positive
regularization. Its exponent `c` may have either sign.

Both derivatives are derived from that actual test. Its sphere
Hessian separates the actual logarithmic Hessian and the square of
the actual logarithmic differential. This supplies the original
exponential generator calculation without assuming an exponential
supermartingale or a probability estimate.

On original unit configurations the logarithmic test has a finite
explicit bound, giving a genuine deterministic bound for both signs
of the exponential parameter. The test remains globally smooth on
the whole ambient space, even at an original zero centroid.
-/

@[expose] public section

open Real
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual exponential of the regularized original mean logarithm.
Source: arXiv:2604.01978v1, §7, the exponential mean-noise maximal argument. -/
noncomputable def exponentialLogMeanOverlap {d n : ℕ} (c a : ℝ) (X : Idx n → EucSpace d) : ℝ :=
  Real.exp (c * logMeanOverlap a X)

/-- The genuine logarithmic exponential is a globally smooth test
for the original martingale problem.
Source: arXiv:2604.01978v1, §7 and Appendix, `def:ito.formula`. -/
theorem contDiff_exponentialLogMeanOverlap {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a) :
    ContDiff ℝ (⊤ : ℕ∞) (exponentialLogMeanOverlap (d := d) (n := n) c a) :=
  (contDiff_const.mul (contDiff_logMeanOverlap ha)).exp

/-- The genuine exponential differential is the actual logarithmic
differential with its original chain-rule coefficient.
Source: arXiv:2604.01978v1, §7, the exponential maximal estimate. -/
theorem hasFDerivAt_exponentialLogMeanOverlap {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (X : Idx n → EucSpace d) :
    HasFDerivAt (exponentialLogMeanOverlap c a)
      ((c * exponentialLogMeanOverlap c a X) • fderiv ℝ (logMeanOverlap a) X) X := by
  have h := (((hasFDerivAt_logMeanOverlap ha X).differentiableAt.hasFDerivAt).const_mul c).exp
  apply h.congr_fderiv
  ext V
  simp only [smul_apply, smul_eq_mul, exponentialLogMeanOverlap]
  ring

/-- Evaluation of the genuine logarithmic exponential differential.
Source: arXiv:2604.01978v1, §7, the selected exponential test. -/
theorem fderiv_exponentialLogMeanOverlap {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (X : Idx n → EucSpace d) :
    fderiv ℝ (exponentialLogMeanOverlap c a) X =
      (c * exponentialLogMeanOverlap c a X) • fderiv ℝ (logMeanOverlap a) X :=
  (hasFDerivAt_exponentialLogMeanOverlap c ha X).fderiv

/-- The genuine second exponential derivative retains the actual
logarithmic Hessian and actual differential square.
Source: arXiv:2604.01978v1, §7, the exponential mean-noise argument
and Appendix, `lem:Ito_formula`. -/
theorem iteratedFDeriv_two_exponentialLogMeanOverlap {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (X V : Idx n → EucSpace d) :
    iteratedFDeriv ℝ 2 (exponentialLogMeanOverlap c a) X ![V, V] =
      exponentialLogMeanOverlap c a X *
        (c * iteratedFDeriv ℝ 2 (logMeanOverlap a) X ![V, V] +
          c ^ 2 * (fderiv ℝ (logMeanOverlap a) X V) ^ 2) := by
  have hfirst : fderiv ℝ (exponentialLogMeanOverlap (d := d) (n := n) c a) =
      fun Y => (c * exponentialLogMeanOverlap c a Y) • fderiv ℝ (logMeanOverlap a) Y :=
    funext (fderiv_exponentialLogMeanOverlap c ha)
  have hD : ContDiff ℝ 1 (fderiv ℝ (logMeanOverlap (d := d) (n := n) a)) :=
    (contDiff_logMeanOverlap ha).fderiv_right (by simp)
  have hc := (hasFDerivAt_exponentialLogMeanOverlap c ha X).const_mul c
  have hd := hc.fun_smul ((hD.differentiable (by norm_num)) X).hasFDerivAt
  rw [iteratedFDeriv_two_apply, hfirst, hd.fderiv, iteratedFDeriv_two_apply]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, add_apply, smul_apply,
    ContinuousLinearMap.smulRight_apply, smul_eq_mul]
  ring

/-- **The genuine exponential sphere Hessian keeps the actual
logarithmic Itô correction and its true noise square.**
Source: arXiv:2604.01978v1, §7, the exponential mean-noise estimate
and Appendix, `def:ito.formula`, on the original sphere product. -/
theorem sphHess_exponentialLogMeanOverlap {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (X V : Idx n → EucSpace d) :
    sphHess (exponentialLogMeanOverlap c a) X V = exponentialLogMeanOverlap c a X *
      (c * sphHess (logMeanOverlap a) X V + c ^ 2 * (fderiv ℝ (logMeanOverlap a) X V) ^ 2) := by
  have hs : (∑ k : Idx n, fderiv ℝ (exponentialLogMeanOverlap c a) X (Pi.single k (X k)) * ‖V k‖ ^ 2) =
      (c * exponentialLogMeanOverlap c a X) *
        ∑ k : Idx n, fderiv ℝ (logMeanOverlap a) X (Pi.single k (X k)) * ‖V k‖ ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k hk
    rw [fderiv_exponentialLogMeanOverlap c ha]
    simp only [smul_apply, smul_eq_mul, mul_assoc]
  rw [sphHess, iteratedFDeriv_two_exponentialLogMeanOverlap c ha, hs, sphHess]
  ring

/-- Every genuine logarithmic exponential test is strictly positive,
including negative exponential parameters.
Source: arXiv:2604.01978v1, §7, the exponential maximal argument. -/
theorem exponentialLogMeanOverlap_pos {d n : ℕ} (c a : ℝ) (X : Idx n → EucSpace d) :
    0 < exponentialLogMeanOverlap c a X := Real.exp_pos _

/-- Original unit configurations give a finite genuine bound for
both signs of the logarithmic exponential parameter.
Source: arXiv:2604.01978v1, §7, the bounded tests in the exponential maximal estimate. -/
theorem exponentialLogMeanOverlap_le_exp_abs {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    exponentialLogMeanOverlap c a X ≤ Real.exp (|c| * (|Real.log a| + |Real.log (1 + a)|)) := by
  apply Real.exp_le_exp.mpr
  calc
    _ ≤ |c * logMeanOverlap a X| := le_abs_self _
    _ = |c| * |logMeanOverlap a X| := abs_mul _ _
    _ ≤ _ := mul_le_mul_of_nonneg_left (abs_logMeanOverlap_le ha X hX) (abs_nonneg c)

/-- The same genuine bound controls the absolute value required
in the original integrating-factor maximal theorem.
Source: arXiv:2604.01978v1, §7, the bounded exponential test argument. -/
theorem abs_exponentialLogMeanOverlap_le {d n : ℕ} (c : ℝ) {a : ℝ} (ha : 0 < a)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) :
    |exponentialLogMeanOverlap c a X| ≤ Real.exp (|c| * (|Real.log a| + |Real.log (1 + a)|)) := by
  rw [abs_of_pos (exponentialLogMeanOverlap_pos c a X)]
  exact exponentialLogMeanOverlap_le_exp_abs c ha X hX

/-- Positive regularization and genuine unit queries give an actual
exponential logarithm with nonzero first and second derivatives. -/
example : ∃ X : Idx 2 → EucSpace 2, (0 : ℝ) < 1 ∧ (∀ k, ‖X k‖ = 1) ∧
    exponentialLogMeanOverlap 1 1 X = 3 / 2 ∧
    fderiv ℝ (exponentialLogMeanOverlap 1 1) X X = 1 ∧
    iteratedFDeriv ℝ 2 (exponentialLogMeanOverlap 1 1) X ![X, X] = 1 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hm : meanOverlap X = 1 / 2 := by
    norm_num [meanOverlap, X, Fin.sum_univ_two, EuclideanSpace.inner_single_left, PiLp.single_apply]
  have he : exponentialLogMeanOverlap 1 1 X = 3 / 2 := by
    rw [exponentialLogMeanOverlap, one_mul, logMeanOverlap, Real.exp_log (meanOverlap_add_pos one_pos X), hm]
    norm_num
  refine ⟨X, one_pos, fun k => by simp [X], he, ?_, ?_⟩
  · rw [fderiv_exponentialLogMeanOverlap 1 one_pos, smul_apply, smul_eq_mul,
      fderiv_logMeanOverlap one_pos, smul_apply, smul_eq_mul, meanOverlapForm_self, hm, he]
    norm_num
  · rw [iteratedFDeriv_two_exponentialLogMeanOverlap 1 one_pos, he,
      iteratedFDeriv_two_logMeanOverlap one_pos, fderiv_logMeanOverlap one_pos,
      smul_apply, smul_eq_mul, meanOverlapForm_self, hm]
    norm_num

end Transformer.Homogenized
