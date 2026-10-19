module

public import Transformer.Kinetic.Section6_ModeCurvature
public import Mathlib.Analysis.Convex.Deriv
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Kinetic theory for Transformers — calculus of the aggregate profile

The aggregate profile in arXiv:2605.09213v1, §6, proof of `thm:U-shape`,
is the same scalar weight as each output mode times an entire series.
Here are its actual first and second derivatives and the coefficient
criterion for strictly positive curvature. This permits the source's
infinite weighted sum to use the same calculus as each original mode.
-/

@[expose] public section

open Real

namespace Transformer
namespace Kinetic

/-- The source's profile weight times an actual entire series.
Source: arXiv:2605.09213v1, §6, the aggregate profile `ℋ`. -/
noncomputable def seriesProfile (lam : ℝ) (a : ℕ → ℝ) (y : ℝ) : ℝ :=
  (lam / (Real.exp lam - 1) + lam * Real.exp (-y)) * powerSum a y

/-- The actual differentiated weighted profile.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def seriesDeriv (lam : ℝ) (a : ℕ → ℝ) (y : ℝ) : ℝ :=
  -(lam * Real.exp (-y)) * powerSum a y +
    (lam / (Real.exp lam - 1) + lam * Real.exp (-y)) * powerSum (derivCoeffs a) y

/-- The second derivative of the original weighted profile.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def seriesDeriv2 (lam : ℝ) (a : ℕ → ℝ) (y : ℝ) : ℝ :=
  lam / (Real.exp lam - 1) * powerSum (derivCoeffs (derivCoeffs a)) y +
    lam * Real.exp (-y) *
      (powerSum (derivCoeffs (derivCoeffs a)) y - 2 * powerSum (derivCoeffs a) y + powerSum a y)

/-- The original coefficients of `ψ'' - 2ψ' + ψ`, for the aggregate series.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
def seriesCurvatureCoeffs (a : ℕ → ℝ) (k : ℕ) : ℝ :=
  derivCoeffs (derivCoeffs a) k - 2 * derivCoeffs a k + a k

variable {a : ℕ → ℝ}

/-- The original curvature combination remains entire.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem isEntire_seriesCurvatureCoeffs (ha : IsEntire a) : IsEntire (seriesCurvatureCoeffs a) := by
  convert (ha.derivCoeffs.derivCoeffs.add (ha.derivCoeffs.const_mul (-2))).add ha using 1
  funext k
  unfold seriesCurvatureCoeffs
  ring

example := isEntire_seriesCurvatureCoeffs isEntire_inv_factorial

/-- The curvature series equals the actual differentiated series combination.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem powerSum_curvature_eq (ha : IsEntire a) (y : ℝ) :
    powerSum (seriesCurvatureCoeffs a) y =
      powerSum (derivCoeffs (derivCoeffs a)) y - 2 * powerSum (derivCoeffs a) y + powerSum a y := by
  rw [show seriesCurvatureCoeffs a = fun k =>
      (derivCoeffs (derivCoeffs a) k + (-2) * derivCoeffs a k) + a k by
        funext k; unfold seriesCurvatureCoeffs; ring,
    powerSum_add (ha.derivCoeffs.derivCoeffs.add (ha.derivCoeffs.const_mul (-2))) ha,
    powerSum_add ha.derivCoeffs.derivCoeffs (ha.derivCoeffs.const_mul (-2)), powerSum_const_mul]
  ring

example := powerSum_curvature_eq isEntire_inv_factorial 0

/-- The actual first derivative of the aggregate weighted series.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem hasDerivAt_seriesProfile (ha : IsEntire a) (lam y : ℝ) :
    HasDerivAt (seriesProfile lam a) (seriesDeriv lam a y) y := by
  have hw : HasDerivAt (fun y => lam / (Real.exp lam - 1) + lam * Real.exp (-y))
      (-(lam * Real.exp (-y))) y := by
    convert ((hasDerivAt_id' (x := y)).neg.exp.const_mul lam).const_add
      (lam / (Real.exp lam - 1)) using 1
    dsimp
    ring
  exact hw.mul (hasDerivAt_powerSum ha y)

example := hasDerivAt_seriesProfile isEntire_inv_factorial 1 0

/-- The actual second derivative of the aggregate weighted series.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem hasDerivAt_seriesDeriv (ha : IsEntire a) (lam y : ℝ) :
    HasDerivAt (seriesDeriv lam a) (seriesDeriv2 lam a y) y := by
  have hn : HasDerivAt (fun y => -(lam * Real.exp (-y))) (lam * Real.exp (-y)) y := by
    convert ((hasDerivAt_id' (x := y)).neg.exp.const_mul lam).neg using 1
    dsimp
    ring
  have hw : HasDerivAt (fun y => lam / (Real.exp lam - 1) + lam * Real.exp (-y))
      (-(lam * Real.exp (-y))) y := by
    convert ((hasDerivAt_id' (x := y)).neg.exp.const_mul lam).const_add
      (lam / (Real.exp lam - 1)) using 1
    dsimp
    ring
  convert (hn.mul (hasDerivAt_powerSum ha y)).add
    (hw.mul (hasDerivAt_powerSum ha.derivCoeffs y)) using 1
  · funext z
    rfl
  · unfold seriesDeriv2
    ring

example := hasDerivAt_seriesDeriv isEntire_inv_factorial 1 0

/-- The original aggregate profile and its first derivative are continuous.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem continuous_seriesProfile (ha : IsEntire a) (lam : ℝ) : Continuous (seriesProfile lam a) :=
  continuous_iff_continuousAt.mpr fun y => (hasDerivAt_seriesProfile ha lam y).continuousAt

example := continuous_seriesProfile isEntire_inv_factorial 1

/-- Continuity of the actual first derivative of the aggregate series.
Source: arXiv:2605.09213v1, §6, the recency limit in `thm:U-shape`. -/
theorem continuous_seriesDeriv (ha : IsEntire a) (lam : ℝ) : Continuous (seriesDeriv lam a) :=
  continuous_iff_continuousAt.mpr fun y => (hasDerivAt_seriesDeriv ha lam y).continuousAt

example := continuous_seriesDeriv isEntire_inv_factorial 1

/-- The actual iterated derivative is the displayed second-derivative formula.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem deriv2_seriesProfile (ha : IsEntire a) (lam y : ℝ) :
    deriv^[2] (seriesProfile lam a) y = seriesDeriv2 lam a y := by
  rw [Function.iterate_succ_apply, Function.iterate_one,
    show deriv (seriesProfile lam a) = seriesDeriv lam a from
      funext fun z => (hasDerivAt_seriesProfile ha lam z).deriv]
  exact (hasDerivAt_seriesDeriv ha lam y).deriv

example := deriv2_seriesProfile isEntire_inv_factorial 1 0

/-- Nonnegative curvature coefficients and a positive second-derivative
constant term give strictly positive actual curvature on the half-line.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem seriesDeriv2_pos (ha : IsEntire a) {lam y : ℝ} (hlam : 0 < lam)
    (hdd : ∀ k, 0 ≤ derivCoeffs (derivCoeffs a) k)
    (hdd0 : 0 < derivCoeffs (derivCoeffs a) 0)
    (hq : ∀ k, 0 ≤ seriesCurvatureCoeffs a k) (hy : 0 ≤ y) : 0 < seriesDeriv2 lam a y := by
  have hψ : 0 < powerSum (derivCoeffs (derivCoeffs a)) y := by
    have hle := le_powerSum ha.derivCoeffs.derivCoeffs hdd hy 0
    simp only [pow_zero, mul_one] at hle
    exact hdd0.trans_le hle
  have hcurv := powerSum_nonneg hq hy
  rw [seriesDeriv2, ← powerSum_curvature_eq ha]
  have hw : 0 < lam / (Real.exp lam - 1) :=
    div_pos hlam (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam))
  exact add_pos_of_pos_of_nonneg (mul_pos hw hψ) (mul_nonneg (by positivity) hcurv)

/-- The source's coefficient criterion proves strict convexity of the actual
weighted series on `[0,∞)`. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem strictConvexOn_seriesProfile (ha : IsEntire a) {lam : ℝ} (hlam : 0 < lam)
    (hdd : ∀ k, 0 ≤ derivCoeffs (derivCoeffs a) k)
    (hdd0 : 0 < derivCoeffs (derivCoeffs a) 0)
    (hq : ∀ k, 0 ≤ seriesCurvatureCoeffs a k) :
    StrictConvexOn ℝ (Set.Ici 0) (seriesProfile lam a) := by
  refine strictConvexOn_of_deriv2_pos' (convex_Ici 0)
    (continuous_seriesProfile ha lam).continuousOn fun y hy => ?_
  rw [deriv2_seriesProfile ha]
  exact seriesDeriv2_pos ha hlam hdd hdd0 hq hy

/-- The actual mode with `c = λ = 1` witnesses both curvature criteria.
Source: arXiv:2605.09213v1, §6, the one-mode case of `thm:U-shape`. -/
example : IsEntire (psiCoeffs 1) ∧ (0 : ℝ) < 1 ∧
    (∀ k, 0 ≤ derivCoeffs (derivCoeffs (psiCoeffs 1)) k) ∧
    0 < derivCoeffs (derivCoeffs (psiCoeffs 1)) 0 ∧
    (∀ k, 0 ≤ seriesCurvatureCoeffs (psiCoeffs 1) k) := by
  refine ⟨isEntire_psiCoeffs 1, one_pos, ?_, ?_, ?_⟩
  · intro k
    rw [derivCoeffs2_psiCoeffs]
    positivity
  · rw [derivCoeffs2_psiCoeffs]
    positivity
  · exact curvatureCoeffs_nonneg zero_le_one (by
      nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_nonneg 3])

end Kinetic
end Transformer
