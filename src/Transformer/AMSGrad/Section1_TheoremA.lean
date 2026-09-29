import Transformer.AMSGrad.Section1_AMSGrad
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Calculus.FDeriv.Const

/-
# AMSGrad — standing assumptions for Theorem A

§1 of arXiv:1904.03590v4: the assumptions every convergence theorem of the
paper is made under. Theorem A itself is proved in `Section4_TheoremAFinite`.

**What the source says and what is carried here.**

* The assumptions are bundled as `IsOnlineConvex S F D G`: the projection
  projects onto `F`, `F` is convex, contains `x₁` and has `ℓ^∞` diameter at
  most `D`, and each `f_t` is convex and differentiable with
  `‖∇f_t(x)‖_∞ ≤ G` on `F`.  `isOnlineConvex_zero` witnesses them.

* The theorem's printed constants are established by an alternative proof in
  `Section4_TheoremAFinite`. The invalid telescoping step identified by the
  paper remains invalid, even on an admissible run.

Source: arXiv:1904.03590v4, §1, Theorem A.
-/

open Finset

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- The standing assumptions of the convergence theorems of arXiv:1904.03590v4:
the online convex optimization setup of §2 on a feasible set `F` with
`ℓ^∞` diameter `D` and gradients bounded by `G` in `ℓ^∞`. -/
structure IsOnlineConvex (S : Setup d) (F : Set (Vec d)) (D G : ℝ) : Prop where
  /-- The projection of the run projects onto `F`. -/
  proj : IsWeightedProj F S.proj
  /-- `F` is convex. -/
  convex : Convex ℝ F
  /-- The starting point lies in `F`. -/
  x₁_mem : S.x₁ ∈ F
  /-- Each `f_t` is convex. -/
  convexOn : ∀ t, ConvexOn ℝ Set.univ (S.f t)
  /-- Each `f_t` is differentiable. -/
  differentiable : ∀ t, Differentiable ℝ (S.f t)
  /-- `‖x - y‖_∞ ≤ D` on `F`. -/
  diam : ∀ x ∈ F, ∀ y ∈ F, ∀ i, |x i - y i| ≤ D
  /-- `‖∇f_t(x)‖_∞ ≤ G` on `F`. -/
  grad_le : ∀ t, ∀ x ∈ F, ∀ i, |grad (S.f t) x i| ≤ G

/-- The zero cost on the box `[-1, 1]^d`, with the clamping projection. -/
def zeroSetup (α β₁ : ℕ → ℝ) (β₂ : ℝ) : Setup d :=
  ⟨fun _ => boxProj (-1) 1, fun _ _ => 0, α, β₁, β₂, 0⟩

/-- The standing assumptions are satisfiable: the zero cost on `[-1, 1]^d`. -/
theorem isOnlineConvex_zero (α β₁ : ℕ → ℝ) (β₂ : ℝ) :
    IsOnlineConvex (zeroSetup (d := d) α β₁ β₂) (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 where
  proj := isWeightedProj_boxProj (by norm_num)
  convex := convex_Icc _ _
  x₁_mem := ⟨fun _ => by norm_num [zeroSetup], fun _ => by norm_num [zeroSetup]⟩
  convexOn _ := convexOn_const 0 convex_univ
  differentiable _ := differentiable_const 0
  diam x hx y hy i := by
    have := hx.1 i; have := hx.2 i; have := hy.1 i; have := hy.2 i
    rw [abs_le]; constructor <;> linarith
  grad_le t x _ i := by simp [zeroSetup, grad]


end AMSGrad
end Transformer
