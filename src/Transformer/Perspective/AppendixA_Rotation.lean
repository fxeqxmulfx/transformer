/-
# Appendix A — the block rotation `e^{tB}` exists

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`Perspective.PerturbationBy` describes the curve `e:helpcl` is read along by
the differential equation `ẋ_i = B x_i` its rotated tokens solve.  Nothing so
far says that such a curve exists, and `Perspective.hessian_at_critical` is
worth nothing until it does: a second derivative of a curve that is not there
is a statement about the empty set.

The curve is the matrix exponential, and two facts about it are needed.  It
solves the equation — `d/dt e^{tB} v = e^{tB} B v = B e^{tB} v`, the last step
being that `B` commutes with its own exponential.  And it stays on the sphere
when `B` is skew — `d/dt ⟨y, y⟩ = ⟨By, y⟩ + ⟨y, By⟩ = 0`, so `‖y(t)‖` never
moves from `‖y(0)‖ = 1`.  That second fact is the only place skewness is used
here, and it is what makes `e^{tB}` a rotation.
-/

import Transformer.Perspective.AppendixA_Hessian
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.Calculus.MeanValue

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- `t ↦ e^{tB} v` solves `ẏ = B y`.

The derivative of `t ↦ e^{tB}` is `e^{tB} B`; the vector field wants
`B e^{tB}`, and the two agree because `B` commutes with `e^{tB}`. -/
theorem hasDerivAt_expSkew (B : ParamMatrix d) (v : EucSpace d) (t : ℝ) :
    HasDerivAt (fun s : ℝ => NormedSpace.exp (s • B) v)
      (B (NormedSpace.exp (t • B) v)) t := by
  have hcomm : Commute (NormedSpace.exp (t • B)) B := by
    refine Commute.exp_left ?_
    show (t • B) * B = B * (t • B)
    ext x
    simp
  have h := (hasDerivAt_exp_smul_const B t).clm_apply (hasDerivAt_const t v)
  rw [hcomm.eq] at h
  simpa using h

/-- At `t = 0` the rotation is the identity. -/
theorem expSkew_zero (B : ParamMatrix d) (v : EucSpace d) :
    NormedSpace.exp ((0 : ℝ) • B) v = v := by
  have h : ((0 : ℝ) • B) = 0 := by ext x; simp
  rw [h, NormedSpace.exp_zero, one_apply_eq_self]

/-- **`e^{tB}` is a rotation when `B` is skew.**  `⟨y, y⟩` has zero derivative
along `ẏ = B y`, so the norm is the one it started with. -/
theorem norm_expSkew (B : ParamMatrix d) (hB : IsSkew d B) (t : ℝ) {v : EucSpace d}
    (hv : ‖v‖ = 1) : ‖NormedSpace.exp (t • B) v‖ = 1 := by
  set y : ℝ → EucSpace d := fun s => NormedSpace.exp (s • B) v with hy
  have hderiv : ∀ s : ℝ, HasDerivAt y (B (y s)) s := fun s => hasDerivAt_expSkew d B v s
  have hg : ∀ s : ℝ, HasDerivAt (fun u : ℝ => inner (𝕜 := ℝ) (y u) (y u)) 0 s := by
    intro s
    have h := HasDerivAt.inner ℝ (hderiv s) (hderiv s)
    have hz : inner (𝕜 := ℝ) (y s) (B (y s)) + inner (𝕜 := ℝ) (B (y s)) (y s) = 0 := by
      rw [hB (y s) (y s)]; ring
    rwa [hz] at h
  have hconst : inner (𝕜 := ℝ) (y t) (y t) = inner (𝕜 := ℝ) (y 0) (y 0) :=
    is_const_of_deriv_eq_zero (fun u => (hg u).differentiableAt) (fun u => (hg u).deriv) t 0
  have h0 : y 0 = v := expSkew_zero d B v
  have hvv : inner (𝕜 := ℝ) v v = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_mul_norm, hv]; ring
  rw [h0, hvv, real_inner_self_eq_norm_mul_norm] at hconst
  have hfac : (‖y t‖ - 1) * (‖y t‖ + 1) = 0 := by nlinarith [hconst]
  rcases mul_eq_zero.mp hfac with h | h
  · linarith
  · linarith [norm_nonneg (y t)]

/-- **The perturbation of `e:helpcl` exists.**  For every skew `B` and every
block `𝒮` the curve that rotates the tokens of `𝒮` by `e^{tB}` and leaves the
others alone is a `PerturbationBy`.

Source: arXiv:2312.10794v5, Appendix A, the curve `e:helpcl` is read along. -/
theorem exists_perturbationBy (B : ParamMatrix d) (hB : IsSkew d B)
    (𝒮 : Finset (Idx n)) (X : SphereTuple d n) :
    ∃ Y : ℝ → SphereTuple d n, PerturbationBy d n B 𝒮 X Y := by
  classical
  refine ⟨fun t i => if i ∈ 𝒮 then
      (⟨NormedSpace.exp (t • B) ((X i : EucSpace d)),
        mem_sphere_zero_iff_norm.mpr
          (norm_expSkew d B hB t (mem_sphere_zero_iff_norm.mp (X i).2))⟩ : SSphere d)
    else X i, ?_, ?_, ?_⟩
  · funext i
    by_cases hi : i ∈ 𝒮
    · simp only [ite_eq_left hi]
      exact Subtype.ext (expSkew_zero d B _)
    · simp only [ite_eq_right hi]
  · intro i hi t
    simp only [ite_eq_left hi]
    exact hasDerivAt_expSkew d B _ t
  · intro i hi t
    simp only [ite_eq_right hi]

/-- The hypothesis of `exists_perturbationBy` is satisfiable: `B = 0` is
skew. -/
example : IsSkew 2 0 := fun x y => by simp

end Perspective
end Transformer
