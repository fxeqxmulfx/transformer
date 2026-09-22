/-
# Attention's forward pass and Frank-Wolfe — well-posedness of the singular ODE

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4, `thm: ode`.

Under the genericity conditions of `thm: exp.fast.polytope` the Cauchy problem
for `eq: hardmax.ode` has exactly one solution: the explicit curve
`eq: solution.short` solves it (`isHardmaxODESolution_hardmaxSol`, in
`Section4_ODESolution.lean`) and every solution is that curve
(`eqOn_hardmaxSol`, in `Section4_ODEUnique.lean`).
-/

import Transformer.FrankWolfe.Section4_ODEUnique

open scoped BigOperators
open Real Set

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- **Theorem (thm: ode) — well-posedness.**

Let the initial configuration satisfy the genericity conditions of
`thm: exp.fast.polytope`.  Then for every `T` the Cauchy problem for
`eq: hardmax.ode` with data `x_i(0) = x_i^0` has a solution in
`C⁰([0,T]; (ℝ^d)^n)`, that solution is unique on `[0,T]`, and it stays in
`𝒦 = conv{x_i^0}`.

**What the source says and what is changed here.**  The source's conditions
1 and 2 are `GenericPolytope`, with condition 2 recorded there as "`x_i^0`
lies in exactly one cell"; see its docstring.

The source assumes `B ≻ 0` and `T > 0`.  The proof uses neither, so both are
dropped and the theorem proved is stronger than the source's (for `T ≤ 0` it
is immediate).

Solutions are the classical ones of `IsHardmaxODESolution`: continuous on
`[0, T]`, solving the equation on `(0, T)`.

Uniqueness is stated on `[0, T]`, where the equation is imposed — two
solutions may of course differ outside the interval, so `∃!` over all of `ℝ`
would be false for a reason having nothing to do with the theorem.  For the
same reason the solution is said to stay in `𝒦` for `t ∈ [0, T]`, where it
lives; the source says `t ≥ 0`.

The source obtains the solution as a limit of the Euler scheme (Part 1 of its
proof); here the limit it finds, `eq: solution.short`, is checked to solve the
equation directly.

Continuous dependence on the initial data is the separate statement
`hardmax_ode_continuous_dependence`.

Source: arXiv:2508.09628v1, §4, `thm: ode`. -/
theorem hardmax_ode_wellposed (B : ParamMatrix d) (X₀ : Idx n → EucSpace d)
    (v : Idx κ → EucSpace d) (hgen : GenericPolytope B v X₀) (T : ℝ) :
    (∃ x : ℝ → Idx n → EucSpace d, x 0 = X₀ ∧ IsHardmaxODESolution B T x ∧
        ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ i : Idx n, x t i ∈ configHull X₀) ∧
      ∀ x y : ℝ → Idx n → EucSpace d, x 0 = X₀ → y 0 = X₀ →
        IsHardmaxODESolution B T x → IsHardmaxODESolution B T y →
        ∀ t ∈ Set.Icc (0 : ℝ) T, x t = y t := by
  choose σ hσ using fun i => (hgen.uniqueCell i).exists
  refine ⟨⟨hardmaxSol X₀ (v ∘ σ), hardmaxSol_zero _ _, isHardmaxODESolution_hardmaxSol hgen hσ T,
    fun t ht i => hardmaxSol_mem_configHull hgen ht.1 i⟩, fun x y hx0 hy0 hx hy t ht => ?_⟩
  rw [eqOn_hardmaxSol hgen hσ hx hx0 ht, eqOn_hardmaxSol hgen hσ hy hy0 ht]

/-- **Theorem (thm: ode) — continuity with respect to the initial data.**

Let `B ≻ 0`, let the initial configuration `x^0` satisfy the genericity
conditions of `thm: exp.fast.polytope`, and let `T > 0`.  The solution map is
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

/-- The hypotheses of `hardmax_ode_wellposed` and
`hardmax_ode_continuous_dependence` are satisfiable: `B = I₁`, one particle at
the origin (`genericPolytope_single`), `T = 1`. -/
example :
    IsPosDef (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    GenericPolytope (ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ : Idx 1 => (0 : EucSpace 1)) (fun _ : Idx 1 => (0 : EucSpace 1)) ∧
    (0 : ℝ) < 1 :=
  ⟨⟨fun _ _ => rfl, fun _ hx => real_inner_self_pos.mpr hx⟩, genericPolytope_single _ _, one_pos⟩

end FrankWolfe
end Transformer
