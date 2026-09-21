import Transformer.AMSGrad.Section1_AMSGrad
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Calculus.FDeriv.Const

/-
# AMSGrad — the standing assumptions, and Theorem A

§1 of arXiv:1904.03590v4: the assumptions every convergence theorem of the
paper is made under, and Theorem A, Reddi et al.'s bound on the regret of
AMSGrad (Theorem 4 of arXiv:1904.09237v1), whose proof the paper shows to be
flawed.

**What the source says and what is carried here.**

* The assumptions are bundled as `IsOnlineConvex S F D G`: the projection
  projects onto `F`, `F` is convex, contains `x₁` and has `ℓ^∞` diameter at
  most `D`, and each `f_t` is convex and differentiable with
  `‖∇f_t(x)‖_∞ ≤ G` on `F`.  `isOnlineConvex_zero` witnesses them.

* Theorem A is stated as Reddi et al. state it, with the corrections of the
  module docstring of `Section1_AMSGrad` (`γ < 1`, `0 < β₂ < 1`) and with the
  implicit `0 ≤ β_{1,t}` and `β₁ < 1`, without which `1/(1 - β₁)` is not a
  bound.  The paper does not show the statement false, only its proof: §3
  exhibits the step that fails, and Reddi et al.'s revision proves a bound
  "with a constant factor missing".  It is unproved here, and so is its
  negation.

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

/-- **Theorem A** (Reddi et al., Theorem 4; "problematic").  For AMSGrad with
`α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`, `0 < β₂ < 1` and
`γ = β₁/√β₂ < 1`, under the standing assumptions,

  `R(T) ≤ D²√T/(α(1-β₁)) Σᵢ √v̂_{T,i} + D²/(2(1-β₁)) Σᵢ Σ_{t=1}^T β_{1,t}√v̂_{t,i}/α_t`
  `      + α√(1 + ln T)/((1-β₁)²(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

Not proved here; see the module docstring.

Source: arXiv:1904.03590v4, §1, Theorem A. -/
theorem theorem_A {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    {T : ℕ} (hT : 1 ≤ T) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (2 * (1 - S.β₁ 1)) *
          ∑ i, ∑ t ∈ Icc 1 T, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  sorry

/-- The hypotheses of `theorem_A` are satisfiable: the zero cost on `[-1, 1]`,
`α = 1`, `β_{1,t} = 0`, `β₂ = 1/2`, `T = 1`, `x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ S.β₁ 1 / Real.sqrt S.β₂ < 1 ∧ 1 ≤ 1 ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    by simp [zeroSetup], le_rfl, fun _ => by norm_num, fun _ => by norm_num⟩

end AMSGrad
end Transformer
