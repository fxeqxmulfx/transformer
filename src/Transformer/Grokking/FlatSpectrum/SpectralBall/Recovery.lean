module

public import Transformer.Grokking.FlatSpectrum.SpectralBall.Basic
public import Transformer.Grokking.FlatSpectrum.Training.Scope.UnseenCell
public import Transformer.Grokking.FlatSpectrum.Training.SignUnique

/-!
# A convex repair of the three-observation counterexample

Source: `review/flat_spectrum.md`, SB2, correcting the unseen sign in XE15 by changing the
constraint, not the theorem about nuclear-norm training. The bound is `‖X‖op ≤ √2` with unit
entry disks. Three observed entries of `[[1,1],[-1,1]]` force its fourth entry. The ordinary
binary CE minimum attains those observations. No arbitrary-sample recovery is asserted.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.SpectralBall

open Matrix Finset Training Training.Scope
open scoped Matrix.Norms.L2Operator ComplexOrder

/-- The energy bound is a positive-semidefinite Gram slack (SB2). -/
lemma two_sub_gram_posSemidef {X : Matrix (Fin 2) (Fin 2) ℂ} (h : ‖X‖ ≤ √2) :
    (2 - Xᴴ * X).PosSemidef := by
  have htwo : (2 : Matrix (Fin 2) (Fin 2) ℂ) = 1 + 1 := by norm_num
  apply posSemidef_of_re_nonneg
  · rw [htwo]
    exact (isHermitian_one.add isHermitian_one).sub (isHermitian_conjTranspose_mul_self X)
  · intro v
    have hv := energy_le h v
    rw [Real.sq_sqrt zero_le_two] at hv
    rw [htwo, sub_mulVec, add_mulVec, one_mulVec, ← mulVec_mulVec,
      dotProduct_sub, dotProduct_add, dotProduct_mulVec (star v) Xᴴ, ← star_mulVec,
      star_dotProduct_self_eq_sum_norm_sq, star_dotProduct_self_eq_sum_norm_sq]
    simpa only [Complex.sub_re, Complex.add_re, Complex.ofReal_re, two_mul,
      sub_nonneg] using hv

/-- A saturated first column forces the unseen sign (SB2). -/
theorem missing_eq_one {X : Matrix (Fin 2) (Fin 2) ℂ} (h : ‖X‖ ≤ √2)
    (h00 : X 0 0 = 1) (h01 : X 0 1 = 1) (h10 : X 1 0 = -1) : X 1 1 = 1 := by
  have hp := two_sub_gram_posSemidef h
  have htwo : (2 : Matrix (Fin 2) (Fin 2) ℂ) = 1 + 1 := by norm_num
  have hz : star (![1, 0] : Fin 2 → ℂ) ⬝ᵥ (2 - Xᴴ * X) *ᵥ ![1, 0] = 0 := by
    simp [htwo, dotProduct, mulVec, Matrix.mul_apply, Fin.sum_univ_two, h00, h10]
  have he := congrFun (hp.dotProduct_mulVec_zero_iff.mp hz) 1
  simp [htwo, mulVec, dotProduct, Matrix.mul_apply, Fin.sum_univ_two, h00, h01, h10] at he
  have he' : X 1 1 + -1 = 0 := by simpa using congrArg star he
  linear_combination he'

/-- The flat target satisfies the bound and the disks (SB2). -/
theorem feasible_signKernel : Feasible (√2) signKernel := by
  constructor
  · intro i j
    fin_cases i <;> fin_cases j <;> norm_num [signKernel]
  · have hg : signKernelᴴ * signKernel = (2 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
      ext i j
      fin_cases i <;> fin_cases j <;> norm_num [signKernel, Matrix.mul_apply, Fin.sum_univ_two]
    have hn := l2_opNorm_conjTranspose_mul_self signKernel
    rw [hg, norm_smul, norm_one] at hn
    norm_num at hn
    nlinarith [norm_nonneg signKernel, Real.sqrt_nonneg 2, Real.sq_sqrt zero_le_two]

example := two_sub_gram_posSemidef feasible_signKernel.2

example := missing_eq_one feasible_signKernel.2 rfl rfl rfl

/-- A unit-disk entry has no larger signed target reading than the code (SB2). -/
lemma sign_reading_le_one {z : ℂ} (hz : ‖z‖ ≤ 1) (b : Bool) :
    (signCode b).re * z.re ≤ 1 := by
  have h := (Complex.abs_re_le_norm z).trans hz
  cases b <;> simp [signCode] <;> linarith [le_abs_self z.re, neg_le_abs z.re]

example := sign_reading_le_one (z := 1) (by norm_num) true

/-- The target minimizes each observed ordinary CE term over the disks (SB2). -/
theorem signKernel_loss_le {X : Matrix (Fin 2) (Fin 2) ℂ}
    (hX : ∀ i j, ‖X i j‖ ≤ 1) :
    ceLoss (codeProbe signCode threeCells) threeLabels signKernel ≤
      ceLoss (codeProbe signCode threeCells) threeLabels X := by
  rw [ceLoss_signCode_eq, ceLoss_signCode_eq]
  apply Finset.sum_le_sum
  intro s _
  have ht : (signCode (threeLabels s)).re *
      (signKernel (threeCells s).1 (threeCells s).2).re = 1 := by
    fin_cases s <;> norm_num [signCode, signKernel, threeCells, threeLabels, signLabel]
  rw [ht]
  apply Real.log_le_log (by positivity)
  apply add_le_add_right
  apply Real.exp_le_exp.mpr
  linarith [sign_reading_le_one (hX (threeCells s).1 (threeCells s).2) (threeLabels s)]

example := signKernel_loss_le feasible_signKernel.1

/-- Every constrained CE minimizer is the entire target, including its unseen cell (SB2).
The inequality says only that `X` is no worse than the feasible target. -/
theorem eq_signKernel_of_loss_le {X : Matrix (Fin 2) (Fin 2) ℂ}
    (hX : Feasible (√2) X)
    (hL : ceLoss (codeProbe signCode threeCells) threeLabels X ≤
      ceLoss (codeProbe signCode threeCells) threeLabels signKernel) : X = signKernel := by
  have hre (s : Fin 3) : (X (threeCells s).1 (threeCells s).2).re =
      (signKernel (threeCells s).1 (threeCells s).2).re := by
    by_contra h
    have hstrict := ceLoss_signCode_midpoint_lt (y := threeLabels) h
    have hmid : Feasible (√2) ((1 / 2 : ℝ) • (X + signKernel)) := by
      rw [smul_add]
      exact feasible_convex _ hX feasible_signKernel (by norm_num) (by norm_num) (by norm_num)
    have hmin := signKernel_loss_le hmid.1
    linarith
  have h00 : X 0 0 = 1 := by
    apply eq_ofReal_of_norm_le_of_le_re (r := 1) (hX.1 0 0)
    simpa [threeCells, signKernel] using (hre 0).ge
  have h01 : X 0 1 = 1 := by
    apply eq_ofReal_of_norm_le_of_le_re (r := 1) (hX.1 0 1)
    simpa [threeCells, signKernel] using (hre 1).ge
  have h10 : X 1 0 = -1 := by
    have he : -X 1 0 = (1 : ℂ) := by
      apply eq_ofReal_of_norm_le_of_le_re (r := 1) (by simpa using hX.1 1 0)
      have h := hre 2
      simp [threeCells, signKernel] at h
      simp [h]
    linear_combination -he
  have h11 := missing_eq_one hX.2 h00 h01 h10
  ext i j
  fin_cases i <;> fin_cases j <;> simpa [signKernel] using (by assumption)

example := eq_signKernel_of_loss_le feasible_signKernel le_rfl

end Transformer.Grokking.FlatSpectrum.SpectralBall
