/-
# Attention's forward pass and Frank-Wolfe — the singular ODE

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4, `eq: hardmax.ode`.

The continuous-time hardmax dynamics has a discontinuous right-hand side, so
no classical existence theory applies.  Under the genericity conditions of
`thm: exp.fast.polytope` every particle nevertheless runs along a straight
line towards the vertex of its cell, `eq: solution.short`:

  `x_i(t) = e^{-t} x_i^0 + (1 - e^{-t}) v_{σ(i)}`.

This module holds the equation (`IsHardmaxODESolution`) and that curve
(`hardmaxSol`) with the facts about it that need no polytope.  The curve is a
solution in `Section4_ODESolution.lean` and the only one in
`Section4_ODEUnique.lean`; `thm: ode` is in `Section4_ODEWellPosed.lean`.
-/

import Transformer.FrankWolfe.Section4_Polytope
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real Set

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- `eq: hardmax.ode`: `ẋ_i(t) = argmax_{y ∈ {x_j(t)}} ⟨B x_i(t), y⟩ - x_i(t)`
for `0 < t < T`, by a curve continuous on `[0, T]`.

The `argmax` is over the *particles*, not over their convex hull, and is
carried as an existential, so a solution is a curve which at every time moves
towards some maximizing particle.  Under the genericity conditions of
`thm: ode` that maximizer is unique, so nothing is lost; without them the
existential is the honest reading of an `argmax` that need not be a singleton.

Solutions are classical, as in the source, which sets Filippov solutions
aside: `x` is differentiable at every `t ∈ (0, T)` and satisfies the equation
there.  The source imposes it for `t > 0` only, so at `t = 0` a solution is
merely continuous, and the datum `x(0) = x^0` is the Cauchy problem's business.
Nothing is asked at the end point `T` either: the uniqueness of `thm: ode`
then covers every classical solution on `[0, T]`, while the solution it has is
differentiable everywhere (`hasDerivAt_hardmaxSol`).

Source: arXiv:2508.09628v1, §4, `eq: hardmax.ode`. -/
def IsHardmaxODESolution (B : ParamMatrix d) (T : ℝ) (x : ℝ → Idx n → EucSpace d) : Prop :=
  ContinuousOn x (Set.Icc 0 T) ∧
  ∀ t ∈ Set.Ioo (0 : ℝ) T, ∀ i : Idx n, ∃ y ∈ Set.range (x t),
    (∀ z ∈ Set.range (x t),
      inner (𝕜 := ℝ) (B (x t i)) z ≤ inner (𝕜 := ℝ) (B (x t i)) y) ∧
    HasDerivAt (fun s => x s i) (y - x t i) t

/-- `eq: solution.short`: each particle leaves `x_i^0` along the straight line
towards its target `w_i`, `x_i(t) = e^{-t} x_i^0 + (1 - e^{-t}) w_i`.  With
`w_i = v_{σ(i)}`, the vertex of the cell of `x_i^0`, this is the solution of
`eq: hardmax.ode` (`isHardmaxODESolution_hardmaxSol`).

Source: arXiv:2508.09628v1, proof of `thm: ode` (`sec: proof.thm.ode`),
`eq: solution.short`. -/
noncomputable def hardmaxSol (X₀ w : Idx n → EucSpace d) (t : ℝ) : Idx n → EucSpace d :=
  fun i => Real.exp (-t) • X₀ i + (1 - Real.exp (-t)) • w i

/-- At `t = 0` the curve is at its initial datum. -/
@[simp] theorem hardmaxSol_zero (X₀ w : Idx n → EucSpace d) : hardmaxSol X₀ w 0 = X₀ := by
  ext1 i
  simp [hardmaxSol]

/-- A particle which starts at its target never moves. -/
theorem hardmaxSol_of_eq {X₀ w : Idx n → EucSpace d} {i : Idx n} (h : X₀ i = w i) (t : ℝ) :
    hardmaxSol X₀ w t i = w i := by
  rw [hardmaxSol, h, ← add_smul, add_sub_cancel, one_smul]

/-- The hypothesis of `hardmaxSol_of_eq` is satisfiable: one particle at its
target. -/
example : (fun _ : Idx 1 => (0 : EucSpace 1)) 0 = (fun _ : Idx 1 => (0 : EucSpace 1)) 0 := rfl

theorem continuous_hardmaxSol (X₀ w : Idx n → EucSpace d) : Continuous (hardmaxSol X₀ w) := by
  unfold hardmaxSol
  fun_prop

/-- The curve solves `eq: hardmax.reduction`, `ẋ_i = w_i - x_i`. -/
theorem hasDerivAt_hardmaxSol (X₀ w : Idx n → EucSpace d) (i : Idx n) (t : ℝ) :
    HasDerivAt (fun s => hardmaxSol X₀ w s i) (w i - hardmaxSol X₀ w t i) t := by
  have he : HasDerivAt (fun s => Real.exp (-s)) (-Real.exp (-t)) t := by
    simpa using (hasDerivAt_neg' (x := t)).exp
  show HasDerivAt (fun s => Real.exp (-s) • X₀ i + (1 - Real.exp (-s)) • w i) _ t
  convert (he.smul_const (X₀ i)).add ((he.const_sub 1).smul_const (w i)) using 1
  simp only [hardmaxSol]
  module

/-- The curve is linear in its data: two curves differ by the curve of the
differences. -/
theorem hardmaxSol_sub (X₀ Y₀ w w' : Idx n → EucSpace d) (t : ℝ) (i : Idx n) :
    hardmaxSol X₀ w t i - hardmaxSol Y₀ w' t i = hardmaxSol (X₀ - Y₀) (w - w') t i := by
  simp only [hardmaxSol, Pi.sub_apply]
  module

/-- For `t ≥ 0` the weight `e^{-t}` of the initial position lies in `(0, 1]`. -/
theorem exp_neg_mem_Ioc {t : ℝ} (ht : 0 ≤ t) : Real.exp (-t) ∈ Ioc 0 1 :=
  ⟨exp_pos _, exp_le_one_iff.2 (neg_nonpos.2 ht)⟩

/-- The hypothesis of `exp_neg_mem_Ioc` is satisfiable: `t = 0`. -/
example : (0 : ℝ) ≤ 0 := le_rfl

/-- For `t ≥ 0` a particle stays in every convex set containing both its
initial position and its target. -/
theorem hardmaxSol_mem {C : Set (EucSpace d)} (hC : Convex ℝ C) {X₀ w : Idx n → EucSpace d}
    {i : Idx n} (hX : X₀ i ∈ C) (hw : w i ∈ C) {t : ℝ} (ht : 0 ≤ t) :
    hardmaxSol X₀ w t i ∈ C :=
  hC hX hw (exp_neg_mem_Ioc ht).1.le (sub_nonneg.2 (exp_neg_mem_Ioc ht).2) (by ring)

/-- The hypotheses of `hardmaxSol_mem` are satisfiable: `C = univ`, `t = 0`. -/
example : Convex ℝ (univ : Set (EucSpace 1)) ∧ (0 : EucSpace 1) ∈ (univ : Set (EucSpace 1)) ∧
    (0 : ℝ) ≤ 0 :=
  ⟨convex_univ, mem_univ _, le_rfl⟩

/-- **A particle reaches an extreme point only if it started there.**  For
`t ≥ 0`, `x_i(t)` lies on the segment from `x_i^0` to `w_i`; at an extreme point
of a convex set containing both ends that forces `x_i^0 = x_i(t)`. -/
theorem eq_of_hardmaxSol_mem_extremePoints {C : Set (EucSpace d)} {X₀ w : Idx n → EucSpace d}
    {i : Idx n} (hX : X₀ i ∈ C) (hw : w i ∈ C) {t : ℝ} (ht : 0 ≤ t)
    (hz : hardmaxSol X₀ w t i ∈ C.extremePoints ℝ) : X₀ i = hardmaxSol X₀ w t i := by
  rcases (mem_extremePoints_iff_forall_segment.1 hz).2 _ hX _ hw
    ⟨_, _, (exp_neg_mem_Ioc ht).1.le, sub_nonneg.2 (exp_neg_mem_Ioc ht).2, by ring, rfl⟩
    with h | h
  · exact h
  · have hE : hardmaxSol X₀ w t i = Real.exp (-t) • X₀ i + (1 - Real.exp (-t)) • w i := rfl
    rw [h] at hE
    have h0 : Real.exp (-t) • (X₀ i - hardmaxSol X₀ w t i) = 0 := by
      calc Real.exp (-t) • (X₀ i - hardmaxSol X₀ w t i)
          = (Real.exp (-t) • X₀ i + (1 - Real.exp (-t)) • hardmaxSol X₀ w t i) -
              hardmaxSol X₀ w t i := by module
        _ = 0 := by rw [← hE, sub_self]
    exact sub_eq_zero.1 ((smul_eq_zero.1 h0).resolve_left (exp_pos _).ne')

/-- The hypotheses of `eq_of_hardmaxSol_mem_extremePoints` are satisfiable: the
one-point set `{0}`, one particle resting there, `t = 0`. -/
example : (0 : EucSpace 1) ∈ ({0} : Set (EucSpace 1)) ∧ (0 : ℝ) ≤ 0 ∧
    hardmaxSol (fun _ : Idx 1 => (0 : EucSpace 1)) (fun _ => 0) 0 0 ∈
      ({0} : Set (EucSpace 1)).extremePoints ℝ := by
  refine ⟨rfl, le_rfl, ?_⟩
  rw [extremePoints_singleton, hardmaxSol_zero]
  rfl

end FrankWolfe
end Transformer
