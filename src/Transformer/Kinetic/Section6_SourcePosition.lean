/-
# Kinetic theory for Transformers — inverting the output change of variable

The proof of arXiv:2605.09213v1, §6, `thm:U-shape`, uses the decreasing
change of variable `Y(σ₀)=log((e^λ-1)/(e^{λσ₀}-1))`. Its actual inverse
is `σ₀(y)=log(1+(e^λ-1)e^{-y})/λ`. We prove both inverse identities
and the correspondence of the open intervals, then compute the actual
derivative of the original `Yfun` on its positive-position domain.
-/

import Transformer.Kinetic.Section6_ModeProfile
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

open Real Set Filter
open scoped Topology

namespace Transformer
namespace Kinetic

/-- The actual inverse of the output-layer variable `Y`, for `λ > 0`.
Source: arXiv:2605.09213v1, §6, the bijection used in `thm:U-shape`, solving `eq:Ylambda`. -/
noncomputable def sourcePosition (lam y : ℝ) : ℝ :=
  Real.log (1 + (Real.exp lam - 1) * Real.exp (-y)) / lam

/-- The inverse formula gives a positive actual source position at every
finite real `y`. Source: arXiv:2605.09213v1, §6, `eq:Ylambda`. -/
theorem sourcePosition_pos {lam : ℝ} (hlam : 0 < lam) (y : ℝ) : 0 < sourcePosition lam y := by
  have he : 0 < Real.exp lam - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)
  exact div_pos (Real.log_pos (by nlinarith [mul_pos he (Real.exp_pos (-y))])) hlam

example := sourcePosition_pos one_pos 1

/-- Positive `y` gives an actual interior source position.
Source: arXiv:2605.09213v1, §6, the bijection in the proof of `thm:U-shape`. -/
theorem sourcePosition_lt_one {lam y : ℝ} (hlam : 0 < lam) (hy : 0 < y) :
    sourcePosition lam y < 1 := by
  have he : 0 < Real.exp lam - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)
  have hexp : Real.exp (-y) < 1 := Real.exp_lt_one_iff.mpr (by linarith)
  have harg : 0 < 1 + (Real.exp lam - 1) * Real.exp (-y) := by positivity
  have hlt : 1 + (Real.exp lam - 1) * Real.exp (-y) < Real.exp lam := by
    nlinarith [mul_lt_mul_of_pos_left hexp he]
  have hl := Real.log_lt_log harg hlt
  rw [Real.log_exp] at hl
  exact (div_lt_one hlam).mpr hl

example := sourcePosition_lt_one one_pos one_pos

/-- The exponential at the actual inverse position is its defining positive argument.
Source: arXiv:2605.09213v1, §6, solving `eq:Ylambda`. -/
theorem exp_mul_sourcePosition {lam : ℝ} (hlam : 0 < lam) (y : ℝ) :
    Real.exp (lam * sourcePosition lam y) = 1 + (Real.exp lam - 1) * Real.exp (-y) := by
  have he : 0 < Real.exp lam - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)
  rw [sourcePosition, mul_div_cancel₀ _ hlam.ne']
  exact Real.exp_log (by positivity)

example := exp_mul_sourcePosition one_pos 1

/-- The original `Yfun` composed with its actual inverse is the original argument.
Source: arXiv:2605.09213v1, §6, the bijection used in `thm:U-shape`. -/
theorem Yfun_one_sourcePosition {lam : ℝ} (hlam : 0 < lam) (y : ℝ) :
    Yfun lam 1 (sourcePosition lam y) = y := by
  have he : Real.exp lam - 1 ≠ 0 := (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)).ne'
  have h := exp_neg_Yfun_one hlam (sourcePosition_pos hlam y)
  rw [exp_mul_sourcePosition hlam, add_sub_cancel_left, mul_div_cancel_left₀ _ he] at h
  exact neg_injective (Real.exp_injective h)

example := Yfun_one_sourcePosition one_pos 1

/-- The actual inverse composed with the original `Yfun` recovers the original position.
Source: arXiv:2605.09213v1, §6, the bijection used in `thm:U-shape`. -/
theorem sourcePosition_Yfun_one {lam σ₀ : ℝ} (hlam : 0 < lam) (hσ₀ : 0 < σ₀) :
    sourcePosition lam (Yfun lam 1 σ₀) = σ₀ := by
  have he : Real.exp lam - 1 ≠ 0 := (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)).ne'
  rw [sourcePosition, exp_neg_Yfun_one hlam hσ₀, mul_div_cancel₀ _ he,
    show 1 + (Real.exp (lam * σ₀) - 1) = Real.exp (lam * σ₀) by ring,
    Real.log_exp, mul_div_cancel_left₀ _ hlam.ne']

example := sourcePosition_Yfun_one one_pos one_half_pos

/-- The actual output-layer `Yfun` is injective on positive source positions.
Source: arXiv:2605.09213v1, §6, the bijection used in `thm:U-shape`. -/
theorem Yfun_one_injOn {lam : ℝ} (hlam : 0 < lam) : InjOn (Yfun lam 1) (Ioi 0) := by
  intro x hx z hz heq
  have h := congrArg (sourcePosition lam) heq
  rwa [sourcePosition_Yfun_one hlam hx, sourcePosition_Yfun_one hlam hz] at h

example := Yfun_one_injOn one_pos

/-- The original output variable equals a difference of actual logarithms
on its positive-position domain. Source: arXiv:2605.09213v1, `eq:Ylambda`. -/
theorem Yfun_one_eq_log_sub {lam σ₀ : ℝ} (hlam : 0 < lam) (hσ₀ : 0 < σ₀) :
    Yfun lam 1 σ₀ = Real.log (Real.exp lam - 1) - Real.log (Real.exp (lam * σ₀) - 1) := by
  have he : Real.exp lam - 1 ≠ 0 := (sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)).ne'
  have hs : Real.exp (lam * σ₀) - 1 ≠ 0 :=
    (sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hlam hσ₀))).ne'
  rw [Yfun, ite_eq_right hlam.ne', mul_one, Real.log_div he hs]

example := Yfun_one_eq_log_sub one_pos one_half_pos

/-- The actual derivative of the original change of variable on positive positions.
Source: arXiv:2605.09213v1, §6, the smooth decreasing bijection in `thm:U-shape`. -/
theorem hasDerivAt_Yfun_one {lam σ₀ : ℝ} (hlam : 0 < lam) (hσ₀ : 0 < σ₀) :
    HasDerivAt (Yfun lam 1) (-(lam * Real.exp (lam * σ₀) / (Real.exp (lam * σ₀) - 1))) σ₀ := by
  have hs : Real.exp (lam * σ₀) - 1 ≠ 0 :=
    (sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hlam hσ₀))).ne'
  have hd := (((hasDerivAt_id' (x := σ₀)).const_mul lam).exp.sub_const 1).log hs
  have h := hd.const_sub (Real.log (Real.exp lam - 1))
  have h' : HasDerivAt (fun s => Real.log (Real.exp lam - 1) - Real.log (Real.exp (lam * s) - 1))
      (-(lam * Real.exp (lam * σ₀) / (Real.exp (lam * σ₀) - 1))) σ₀ := by
    convert h using 1
    ring
  apply h'.congr_of_eventuallyEq
  filter_upwards [Ioi_mem_nhds hσ₀] with s hs
  exact Yfun_one_eq_log_sub hlam hs

example := hasDerivAt_Yfun_one one_pos one_half_pos

/-- The series endpoint `y = 0` corresponds to the actual output position.
Source: arXiv:2605.09213v1, §6, the closed-endpoint bijection in `thm:U-shape`. -/
theorem sourcePosition_zero {lam : ℝ} (hlam : 0 < lam) : sourcePosition lam 0 = 1 := by
  rw [sourcePosition]
  simp only [neg_zero, Real.exp_zero, mul_one]
  rw [show 1 + (Real.exp lam - 1) = Real.exp lam by ring, Real.log_exp, div_self hlam.ne']

example := sourcePosition_zero one_pos

/-- The original output variable is a bijection from interior positions
onto positive real arguments. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem bijOn_Yfun_one {lam : ℝ} (hlam : 0 < lam) :
    BijOn (Yfun lam 1) (Ioo (0 : ℝ) 1) (Ioi 0) := by
  refine ⟨?_, ?_, ?_⟩
  · intro s hs
    exact Yfun_one_pos hlam hs.1 hs.2
  · intro s hs z hz heq
    exact Yfun_one_injOn hlam hs.1 hz.1 heq
  · intro y hy
    exact ⟨sourcePosition lam y, ⟨sourcePosition_pos hlam y, sourcePosition_lt_one hlam hy⟩,
      Yfun_one_sourcePosition hlam y⟩

example := bijOn_Yfun_one one_pos

/-- The actual inverse is injective on the whole real line: applying the
original output variable recovers each argument.
Source: arXiv:2605.09213v1, §6, the inverse of `eq:Ylambda`. -/
theorem sourcePosition_injective {lam : ℝ} (hlam : 0 < lam) :
    Function.Injective (sourcePosition lam) := by
  intro y z h
  have heq := congrArg (Yfun lam 1) h
  rwa [Yfun_one_sourcePosition hlam, Yfun_one_sourcePosition hlam] at heq

example := sourcePosition_injective one_pos

end Kinetic
end Transformer
