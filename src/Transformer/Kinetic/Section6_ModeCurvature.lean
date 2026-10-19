module

public import Transformer.Kinetic.Section6_ModeProfile

/-!
# Kinetic theory for Transformers — curvature of the original output modes

The proof of arXiv:2605.09213v1, §6, `thm:U-shape`, expands
`ψ_c'' - 2ψ_c' + ψ_c` and shows every coefficient nonnegative when
`0 < c ≤ 3 - √3`. The positive constant coefficient of `ψ_c''` then
gives strictly positive curvature of the actual profile on `[0,∞)`.

The source also writes `h_c'(0) < 0` at the non-strict upper bound
`c ≤ 2(1-e^{-λ})`. At equality the slope is zero. We retain the bound
and prove the correct nonpositive slope, with strict negativity whenever
the bound is strict. The full Fourier family has such a mode because its
eigenvalues tend to zero.
-/

@[expose] public section

open Real

namespace Transformer
namespace Kinetic

/-- The actual coefficients of `ψ_c'` after cancelling factorials.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem derivCoeffs_psiCoeffs (c : ℝ) (k : ℕ) :
    derivCoeffs (psiCoeffs c) k = c ^ (k + 2) / (k.factorial * (k + 2).factorial) := by
  have hk : (k.factorial : ℝ) ≠ 0 := by positivity
  have hk1 : ((k + 1 : ℕ) : ℝ) ≠ 0 := by positivity
  have hk2 : ((k + 2).factorial : ℝ) ≠ 0 := by positivity
  simp only [derivCoeffs, psiCoeffs, Nat.add_assoc]
  rw [Nat.factorial_succ k, Nat.cast_mul, Nat.cast_add, Nat.cast_one]
  field_simp

/-- The actual coefficients of `ψ_c''` after cancelling factorials.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem derivCoeffs2_psiCoeffs (c : ℝ) (k : ℕ) :
    derivCoeffs (derivCoeffs (psiCoeffs c)) k =
      c ^ (k + 3) / (k.factorial * (k + 3).factorial) := by
  have hk : (k.factorial : ℝ) ≠ 0 := by positivity
  have hk1 : ((k + 1 : ℕ) : ℝ) ≠ 0 := by positivity
  have hk3 : ((k + 3).factorial : ℝ) ≠ 0 := by positivity
  rw [derivCoeffs, derivCoeffs_psiCoeffs]
  simp only [Nat.add_assoc]
  rw [Nat.factorial_succ k, Nat.cast_mul, Nat.cast_add, Nat.cast_one]
  field_simp

/-- The coefficients of the actual combination `ψ_c'' - 2ψ_c' + ψ_c`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def curvatureCoeffs (c : ℝ) (k : ℕ) : ℝ :=
  derivCoeffs (derivCoeffs (psiCoeffs c)) k - 2 * derivCoeffs (psiCoeffs c) k + psiCoeffs c k

/-- The displayed coefficient formula in the source's curvature argument.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem curvatureCoeffs_eq (c : ℝ) (k : ℕ) :
    curvatureCoeffs c k = c ^ (k + 1) / (k.factorial * (k + 3).factorial) *
      (c ^ 2 - 2 * (k + 3) * c + (k + 2) * (k + 3)) := by
  have hk : (k.factorial : ℝ) ≠ 0 := by positivity
  have h1 : ((k + 1).factorial : ℝ) ≠ 0 := by positivity
  have h2 : ((k + 2 : ℕ) : ℝ) ≠ 0 := by positivity
  have h3 : ((k + 3 : ℕ) : ℝ) ≠ 0 := by positivity
  rw [curvatureCoeffs, derivCoeffs2_psiCoeffs, derivCoeffs_psiCoeffs, psiCoeffs,
    show k + 3 = k + 2 + 1 by omega, Nat.factorial_succ,
    show k + 2 = k + 1 + 1 by omega, Nat.factorial_succ]
  push_cast
  simp only [pow_succ]
  field_simp
  ring

/-- Every actual curvature coefficient is nonnegative under the original bound.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem curvatureCoeffs_nonneg {c : ℝ} (hc : 0 ≤ c) (hsmall : c ≤ 3 - Real.sqrt 3)
    (k : ℕ) : 0 ≤ curvatureCoeffs c k := by
  have hs := Real.sqrt_nonneg 3
  have hs2 := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)
  have hbase : 0 ≤ c ^ 2 - 6 * c + 6 := by
    have h := mul_nonneg (by linarith : 0 ≤ 3 - c - Real.sqrt 3)
      (by linarith : 0 ≤ 3 - c + Real.sqrt 3)
    nlinarith
  have hk : (0 : ℝ) ≤ k := by positivity
  have hrest : 0 ≤ (k : ℝ) * (k + 5 - 2 * c) := by
    have hs1 : 1 ≤ Real.sqrt 3 := by nlinarith
    exact mul_nonneg hk (by nlinarith)
  rw [curvatureCoeffs_eq]
  exact mul_nonneg (by positivity) (by nlinarith)

example := curvatureCoeffs_nonneg (c := 1) zero_le_one (by
  nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_nonneg 3]) 0

/-- The combination of differentiated original coefficients is still entire.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem isEntire_curvatureCoeffs (c : ℝ) : IsEntire (curvatureCoeffs c) := by
  convert ((isEntire_psiCoeffs c).derivCoeffs.derivCoeffs.add
    ((isEntire_psiCoeffs c).derivCoeffs.const_mul (-2))).add (isEntire_psiCoeffs c) using 1
  funext k
  simp only [curvatureCoeffs]
  ring

/-- The actual series combination equals the series of curvature coefficients.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem psi_curvature_eq (c y : ℝ) :
    psiDeriv2 c y - 2 * psiDeriv c y + psi c y = powerSum (curvatureCoeffs c) y := by
  have h := isEntire_psiCoeffs c
  rw [show curvatureCoeffs c = fun k =>
      (derivCoeffs (derivCoeffs (psiCoeffs c)) k + (-2) * derivCoeffs (psiCoeffs c) k) +
        psiCoeffs c k by funext k; simp only [curvatureCoeffs]; ring,
    powerSum_add (h.derivCoeffs.derivCoeffs.add (h.derivCoeffs.const_mul (-2))) h,
    powerSum_add h.derivCoeffs.derivCoeffs (h.derivCoeffs.const_mul (-2)), powerSum_const_mul]
  unfold psiDeriv2 psiDeriv psi
  ring

/-- Strictly positive actual mode curvature, including `y = 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem modeDeriv2_pos {lam c y : ℝ} (hlam : 0 < lam) (hc : 0 < c)
    (hsmall : c ≤ 3 - Real.sqrt 3) (hy : 0 ≤ y) : 0 < modeDeriv2 lam c y := by
  have hψ : 0 < psiDeriv2 c y := by
    have hle := le_powerSum (isEntire_psiCoeffs c).derivCoeffs.derivCoeffs
      (fun k => by rw [derivCoeffs2_psiCoeffs]; positivity) hy 0
    rw [derivCoeffs2_psiCoeffs] at hle
    norm_num at hle
    exact (by positivity : (0 : ℝ) < c ^ 3 / 6).trans_le hle
  have hcurv := powerSum_nonneg (curvatureCoeffs_nonneg hc.le hsmall) hy
  rw [modeDeriv2, psi_curvature_eq]
  have hw : 0 < lam / (Real.exp lam - 1) :=
    div_pos hlam (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam))
  exact add_pos_of_pos_of_nonneg (mul_pos hw hψ) (mul_nonneg (by positivity) hcurv)

example := modeDeriv2_pos (lam := 1) (c := 1) (y := 0) one_pos one_pos (by
  nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_nonneg 3]) le_rfl

/-- The exact source endpoint slope, on the continuous series extension.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem modeDeriv_zero_eq {lam : ℝ} (hlam : 0 < lam) (c : ℝ) :
    modeDeriv lam c 0 = lam * c * (c / (2 * (1 - Real.exp (-lam))) - 1) := by
  have he : Real.exp lam - 1 ≠ 0 := (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)).ne'
  have hn : 1 - Real.exp (-lam) ≠ 0 :=
    (sub_pos.mpr (Real.exp_lt_one_iff.mpr (by linarith))).ne'
  simp only [modeDeriv, psi, psiDeriv, powerSum_zero, derivCoeffs_psiCoeffs,
    psiCoeffs, neg_zero, Real.exp_zero, mul_one]
  norm_num
  rw [Real.exp_neg] at hn ⊢
  field_simp
  ring

example := modeDeriv_zero_eq one_pos 1

/-- The non-strict source bound gives a nonpositive slope, with equality
allowed at its upper endpoint. Source: arXiv:2605.09213v1, §6, `eq:affine-smallness`. -/
theorem modeDeriv_zero_nonpos {lam c : ℝ} (hlam : 0 < lam) (hc : 0 ≤ c)
    (hsmall : c ≤ 2 * (1 - Real.exp (-lam))) : modeDeriv lam c 0 ≤ 0 := by
  have hd : 0 < 2 * (1 - Real.exp (-lam)) := by
    have h := Real.exp_lt_one_iff.mpr (by linarith : -lam < 0)
    linarith
  rw [modeDeriv_zero_eq hlam]
  exact mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hlam.le hc)
    (sub_nonpos.mpr ((div_le_one hd).mpr hsmall))

example := modeDeriv_zero_nonpos (c := 0) one_pos le_rfl (by
  linarith [Real.exp_lt_one_iff.mpr (by norm_num : -(1 : ℝ) < 0)])

/-- A strict source bound gives a strictly negative original mode slope.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`, correcting its equality case. -/
theorem modeDeriv_zero_neg {lam c : ℝ} (hlam : 0 < lam) (hc : 0 < c)
    (hsmall : c < 2 * (1 - Real.exp (-lam))) : modeDeriv lam c 0 < 0 := by
  have hd : 0 < 2 * (1 - Real.exp (-lam)) := by
    have h := Real.exp_lt_one_iff.mpr (by linarith : -lam < 0)
    linarith
  rw [modeDeriv_zero_eq hlam]
  exact mul_neg_of_pos_of_neg (mul_pos hlam hc) (sub_neg.mpr ((div_lt_one hd).mpr hsmall))

example := modeDeriv_zero_neg (lam := 1) (c := 1 - Real.exp (-1)) one_pos
  (by linarith [Real.exp_lt_one_iff.mpr (by norm_num : -(1 : ℝ) < 0)])
  (by linarith [Real.exp_lt_one_iff.mpr (by norm_num : -(1 : ℝ) < 0)])

end Kinetic
end Transformer
