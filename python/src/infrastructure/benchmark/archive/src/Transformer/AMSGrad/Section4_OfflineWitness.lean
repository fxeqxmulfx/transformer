/-
# AMSGrad — a nonconstant fixed-loss witness for the original algorithm

arXiv:1904.03590v4, §2 and Corollary 4.5. The linear objective on the
source's bounded box has a genuine feasible minimizer. It witnesses
offline convergence assumptions without using a zero objective.
-/

import Transformer.AMSGrad.Section3_Example

noncomputable section

namespace Transformer.AMSGrad

/-- Fixed nonconstant linear cost with the source's box projection.
Source: arXiv:1904.03590v4, §2, Algorithm 1 and §4, offline specialization. -/
def offlineLinearSetup (α β₁ : ℕ → ℝ) (β₂ : ℝ) : Setup 1 :=
  ⟨fun _ => boxProj (-1) 1, fun _ x => 1 * x 0, α, β₁, β₂, fun _ => 1⟩

/-- The actual fixed nonconstant cost satisfies the online assumptions.
Source: arXiv:1904.03590v4, §2 and §4, offline specialization. -/
theorem offlineLinearSetup_online (α β₁ : ℕ → ℝ) (β₂ : ℝ) :
    IsOnlineConvex (offlineLinearSetup α β₁ β₂)
      (Set.Icc (fun _ => -1) (fun _ => 1)) 2 1 where
  proj := isWeightedProj_boxProj (by norm_num)
  convex := convex_Icc _ _
  x₁_mem := ⟨fun _ => by norm_num [offlineLinearSetup], fun _ => le_rfl⟩
  convexOn t := ⟨convex_univ, fun x _ y _ a b _ _ _ => le_of_eq (by
    simp only [offlineLinearSetup, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring)⟩
  differentiable t y := ((hasFDerivAt_apply (𝕜 := ℝ) 0 y).const_mul (1 : ℝ)).differentiableAt
  diam x hx y hy i := by
    have := hx.1 i
    have := hx.2 i
    have := hy.1 i
    have := hy.2 i
    rw [abs_le]
    constructor <;> linarith
  grad_le t x hx i := by
    change |grad (fun x : Vec 1 => 1 * x 0) x i| ≤ 1
    rw [grad_linear]
    norm_num

/-- This fixed cost has a feasible global minimizer on the box.
Source: arXiv:1904.03590v4, §2, offline specialization of Corollary 4.5. -/
theorem offlineLinearSetup_minimum :
    (fun _ : Fin 1 => (-1 : ℝ)) ∈ Set.Icc (fun _ => -1) (fun _ => 1) ∧
    (∀ x ∈ Set.Icc (fun _ : Fin 1 => (-1 : ℝ)) (fun _ => (1 : ℝ)),
      (fun y : Vec 1 => 1 * y 0) (fun _ => -1) ≤ (fun y : Vec 1 => 1 * y 0) x) := by
  refine ⟨⟨le_rfl, fun i => by norm_num⟩, ?_⟩
  intro x hx
  simpa only [one_mul] using hx.1 (0 : Fin 1)

end Transformer.AMSGrad
