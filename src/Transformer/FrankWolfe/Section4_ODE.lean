/-
# Attention's forward pass and Frank-Wolfe — the singular ODE

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4, `eq: hardmax.ode` and
`thm: ode`.

The continuous-time hardmax dynamics has a discontinuous right-hand side, so
no classical existence theory applies; under the genericity conditions of
`thm: exp.fast.polytope` it is nevertheless well posed.
-/

import Transformer.FrankWolfe.Section4_Polytope

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- `eq: hardmax.ode`: `ẋ_i(t) = argmax_{y ∈ {x_j(t)}} ⟨B x_i(t), y⟩ - x_i(t)`
on `[0, T]`.

The `argmax` is over the *particles*, not over their convex hull, and is
carried as an existential, so a solution is a curve which at every time moves
towards some maximizing particle.  Under the genericity conditions of
`thm: ode` that maximizer is unique, so nothing is lost; without them the
existential is the honest reading of an `argmax` that need not be a singleton.

Source: arXiv:2508.09628v1, §4, `eq: hardmax.ode`. -/
def IsHardmaxODESolution (B : ParamMatrix d) (T : ℝ) (x : ℝ → Idx n → EucSpace d) : Prop :=
  ContinuousOn x (Set.Icc 0 T) ∧
  ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ i : Idx n, ∃ y ∈ Set.range (x t),
    (∀ z ∈ Set.range (x t),
      inner (𝕜 := ℝ) (B (x t i)) z ≤ inner (𝕜 := ℝ) (B (x t i)) y) ∧
    HasDerivWithinAt (fun s => x s i) (y - x t i) (Set.Icc 0 T) t

/-- **Theorem (thm: ode) — well-posedness.**

Let `B ≻ 0` and let the initial configuration satisfy the genericity
conditions of `thm: exp.fast.polytope`.  Then for every `T > 0` the Cauchy
problem for `eq: hardmax.ode` with data `x_i(0) = x_i^0` has a solution in
`C⁰([0,T]; (ℝ^d)^n)`, that solution is unique on `[0,T]`, and it stays in
`𝒦 = conv{x_i^0}`.

**What the source says and what is changed here.**  The source's conditions
1 and 2 are `GenericPolytope`, with condition 2 recorded there as "`x_i^0`
lies in exactly one cell"; see its docstring.

Uniqueness is stated on `[0, T]`, where the equation is imposed — two
solutions may of course differ outside the interval, so `∃!` over all of `ℝ`
would be false for a reason having nothing to do with the theorem.

Continuous dependence on the initial data is the separate statement
`hardmax_ode_continuous_dependence`.

Not proved here.

Source: arXiv:2508.09628v1, §4, `thm: ode`. -/
theorem hardmax_ode_wellposed (B : ParamMatrix d) (hB : IsPosDef B)
    (X₀ : Idx n → EucSpace d) (v : Idx κ → EucSpace d) (hgen : GenericPolytope B v X₀)
    (T : ℝ) (hT : 0 < T) :
    (∃ x : ℝ → Idx n → EucSpace d, x 0 = X₀ ∧ IsHardmaxODESolution B T x ∧
        ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ i : Idx n, x t i ∈ configHull X₀) ∧
      ∀ x y : ℝ → Idx n → EucSpace d, x 0 = X₀ → y 0 = X₀ →
        IsHardmaxODESolution B T x → IsHardmaxODESolution B T y →
        ∀ t ∈ Set.Icc (0 : ℝ) T, x t = y t := by
  sorry

/-- **Theorem (thm: ode) — continuity with respect to the initial data.**

Under the hypotheses of `hardmax_ode_wellposed`, the solution map is
continuous at `x^0`: initial configurations that are themselves generic and
close to `x^0` produce solutions uniformly close on `[0, T]`.

**What the source says and what is changed here.**  The source says only
"which is continuous with respect to the initial data".  The nearby data are
restricted to those that are themselves generic — for which a solution is
known to exist and be unique at all — since that is the class the theorem
constructs solutions in.

Not proved here.

Source: arXiv:2508.09628v1, §4, `thm: ode`. -/
theorem hardmax_ode_continuous_dependence (B : ParamMatrix d) (hB : IsPosDef B)
    (X₀ : Idx n → EucSpace d) (v : Idx κ → EucSpace d) (hgen : GenericPolytope B v X₀)
    (T : ℝ) (hT : 0 < T) :
    ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ (Y₀ : Idx n → EucSpace d) (w : Idx κ → EucSpace d),
      GenericPolytope B w Y₀ → dist Y₀ X₀ < δ →
      ∀ x y : ℝ → Idx n → EucSpace d, x 0 = X₀ → y 0 = Y₀ →
        IsHardmaxODESolution B T x → IsHardmaxODESolution B T y →
        ∀ t ∈ Set.Icc (0 : ℝ) T, dist (x t) (y t) < ε := by
  sorry

/-- The hypotheses shared by `hardmax_ode_wellposed` and
`hardmax_ode_continuous_dependence` are satisfiable: `B = I₁`, one particle at
the origin, `T = 1`. -/
example :
    IsPosDef (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    GenericPolytope (ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ : Idx 1 => (0 : EucSpace 1)) (fun _ : Idx 1 => (0 : EucSpace 1)) ∧
    (0 : ℝ) < 1 := by
  have hhull : configHull (fun _ : Idx 1 => (0 : EucSpace 1)) = {(0 : EucSpace 1)} := by
    rw [configHull, Set.range_const, convexHull_singleton]
  have hcell : ∀ j : Idx 1, (0 : EucSpace 1) ∈
      cell (ContinuousLinearMap.id ℝ (EucSpace 1)) (configHull (fun _ : Idx 1 => (0 : EucSpace 1)))
        (fun _ : Idx 1 => (0 : EucSpace 1)) j := by
    intro j
    refine ⟨by rw [hhull]; rfl, fun y hy => ?_⟩
    simp
  refine ⟨⟨fun x y => rfl, fun x hx => real_inner_self_pos.mpr hx⟩,
    ⟨⟨fun a b _ => Subsingleton.elim a b, ?_⟩,
      fun j => ⟨hcell j, fun i hij => absurd (Subsingleton.elim i j) hij⟩,
      fun i => ⟨0, hcell 0, fun j _ => Subsingleton.elim j 0⟩⟩, one_pos⟩
  rw [Set.range_const, hhull, extremePoints_singleton]

end FrankWolfe
end Transformer
