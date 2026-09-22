/-
# Attention's forward pass and Frank-Wolfe — well-posedness of the singular ODE

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4, `thm: ode`.

Under the genericity conditions of `thm: exp.fast.polytope` the Cauchy problem
for `eq: hardmax.ode` has exactly one solution: the explicit curve
`eq: solution.short` solves it (`isHardmaxODESolution_hardmaxSol`, in
`Section4_ODESolution.lean`) and every solution is that curve
(`eqOn_hardmaxSol`, in `Section4_ODEUnique.lean`).  The strict gaps which make
the curve the solution survive a small perturbation of the data
(`eventually_inner_lt`, the source's `cl: stab.cell`), so the solution depends
continuously on them (`hardmax_ode_continuous_dependence`).
-/

import Transformer.FrankWolfe.Section4_ODEUnique

open scoped BigOperators Topology
open Real Set Filter

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

/-- **Claim (cl: stab.cell) — the strict gaps survive a small perturbation.**
Let `p(k)` be a particle that starts at the vertex `v_k`.  For every
configuration `y` close enough to `x^0` and every `i`, the functional
`⟨B y_i, ·⟩` scores `y_{p(σ(i))}` strictly above `y_j` for every particle `j`
that does not start at `v_{σ(i)}`.

**What the source says and what is changed here.**  The source compares
`ṽ_{σ(i)}` with every other perturbed particle, and that is false as soon as a
vertex `v_k` carries a particle besides `p(k)`: the two score level at `x^0`,
and for `i = p(k)`, moving the other one alone by any `ε > 0` in the direction
of `B v_k` puts it ahead (level, if `B v_k = 0`).  The claim is stated here
for the particles that do not start at `v_{σ(i)}`, which is what Part 3 needs.

Source: arXiv:2508.09628v1, `sec: proof.thm.ode`, Part 3, `cl: stab.cell`. -/
theorem eventually_inner_lt {B : ParamMatrix d} {X₀ : Idx n → EucSpace d}
    {v : Idx κ → EucSpace d} {σ : Idx n → Idx κ} (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) {p : Idx κ → Idx n}
    (hp : ∀ k, X₀ (p k) = v k) :
    ∀ᶠ Y in 𝓝 X₀, ∀ i j, X₀ j ≠ v (σ i) →
      inner (𝕜 := ℝ) (B (Y i)) (Y j) < inner (𝕜 := ℝ) (B (Y i)) (Y (p (σ i))) := by
  have hc : ∀ i k, Continuous fun Y : Idx n → EucSpace d => inner (𝕜 := ℝ) (B (Y i)) (Y k) :=
    fun i k => (B.continuous.comp (continuous_apply i)).inner (continuous_apply k)
  refine eventually_all.2 fun i => eventually_all.2 fun j => ?_
  by_cases hj : X₀ j = v (σ i)
  · exact Eventually.of_forall fun _ h => absurd hj h
  · have h0 := inner_hardmaxSol_lt hgen hσ le_rfl i (subset_convexHull ℝ _ (mem_range_self j)) hj
    rw [hardmaxSol_zero, ← hp (σ i)] at h0
    exact (((hc i j).tendsto X₀).eventually_lt ((hc i (p (σ i))).tendsto X₀) h0).mono
      fun _ h _ => h

/-- **Theorem (thm: ode) — continuity with respect to the initial data.**

Let the initial configuration `x^0` satisfy the genericity conditions of
`thm: exp.fast.polytope`.  The solution map is continuous at `x^0`, uniformly
on `[0, T]`: for every `ε > 0` there is `δ > 0` such that the solution from any
generic initial configuration `y^0` within `δ` of `x^0` stays within `ε` of the
solution from `x^0` at every `t ∈ [0, T]`.

**What the source says and what is changed here.**  The source says only
"which is continuous with respect to the initial data".  The nearby data are
restricted to those that are themselves generic — for which a solution is
known to exist and be unique at all — since that is the class the theorem
constructs solutions in.  Their number `κ'` of vertices is left free.

The source assumes `B ≻ 0` and `T > 0`.  The proof uses neither, so both are
dropped, as in `hardmax_ode_wellposed`.

The proof is Part 3 of the source's.  By `cl: stab.cell`
(`eventually_inner_lt`) the vertex `w_{σ'(i)}` that `y_i` heads for is the
perturbation of a particle that starts at `v_{σ(i)}`.  Both solutions are
explicit (`eqOn_hardmaxSol`), and their difference
`e^{-t}(x_i^0 - y_i^0) + (1 - e^{-t})(v_{σ(i)} - w_{σ'(i)})` is a convex
combination of two vectors shorter than `δ`.

Source: arXiv:2508.09628v1, §4, `thm: ode`; `sec: proof.thm.ode`, Part 3. -/
theorem hardmax_ode_continuous_dependence (B : ParamMatrix d) (X₀ : Idx n → EucSpace d)
    (v : Idx κ → EucSpace d) (hgen : GenericPolytope B v X₀) (T : ℝ) :
    ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ (κ' : ℕ) (Y₀ : Idx n → EucSpace d)
      (w : Idx κ' → EucSpace d), GenericPolytope B w Y₀ → dist Y₀ X₀ < δ →
      ∀ x y : ℝ → Idx n → EucSpace d, x 0 = X₀ → y 0 = Y₀ →
        IsHardmaxODESolution B T x → IsHardmaxODESolution B T y →
        ∀ t ∈ Set.Icc (0 : ℝ) T, dist (x t) (y t) < ε := by
  choose σ hσ using fun i => (hgen.uniqueCell i).exists
  have hp : ∀ k, ∃ q, X₀ q = v k := fun k => (hgen.exists_particle hσ k).imp fun _ h => h.1
  choose p hp using hp
  obtain ⟨δ₀, hδ₀, hgap⟩ := Metric.eventually_nhds_iff.1 (eventually_inner_lt hgen hσ hp)
  intro ε hε
  refine ⟨min ε δ₀, lt_min hε hδ₀, fun κ' Y₀ w hgen' hY x y hx0 hy0 hx hy t ht => ?_⟩
  choose σ' hσ' using fun i => (hgen'.uniqueCell i).exists
  have hYX : ∀ j, dist (X₀ j) (Y₀ j) < min ε δ₀ := fun j =>
    (dist_le_pi_dist X₀ Y₀ j).trans_lt (by rwa [dist_comm])
  rw [eqOn_hardmaxSol hgen hσ hx hx0 ht, eqOn_hardmaxSol hgen' hσ' hy hy0 ht]
  refine ((dist_pi_lt_iff (lt_min hε hδ₀)).2 fun i => ?_).trans_le (min_le_left _ _)
  -- `y_i` heads for the perturbation of a particle that starts at `v_{σ(i)}`
  obtain ⟨q, hq, -⟩ := hgen'.exists_particle hσ' (σ' i)
  have hXq : X₀ q = v (σ i) := by
    by_contra hne
    have hle := (hσ' i).2 (Y₀ (p (σ i))) (subset_convexHull ℝ _ (mem_range_self _))
    rw [← hq] at hle
    exact lt_irrefl _ ((hgap (hY.trans_le (min_le_right _ _)) i q hne).trans_le hle)
  rw [dist_eq_norm, hardmaxSol_sub, ← mem_ball_zero_iff]
  refine hardmaxSol_mem (convex_ball 0 _) ?_ ?_ ht.1
  · rw [mem_ball_zero_iff, Pi.sub_apply, ← dist_eq_norm]
    exact hYX i
  · rw [mem_ball_zero_iff, Pi.sub_apply, Function.comp_apply, Function.comp_apply, ← hXq, ← hq,
      ← dist_eq_norm]
    exact hYX q

/-- The hypotheses of `hardmax_ode_wellposed`, `eventually_inner_lt` and
`hardmax_ode_continuous_dependence` are satisfiable: `B = I₁`, one particle at
the origin (`genericPolytope_single`), in the one cell, resting at the one
vertex. -/
example : GenericPolytope (ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ : Idx 1 => (0 : EucSpace 1)) (fun _ : Idx 1 => (0 : EucSpace 1)) ∧
    (∀ i : Idx 1, (fun _ : Idx 1 => (0 : EucSpace 1)) i ∈
      cell (ContinuousLinearMap.id ℝ (EucSpace 1)) (configHull fun _ : Idx 1 => (0 : EucSpace 1))
        (fun _ : Idx 1 => (0 : EucSpace 1)) ((fun _ : Idx 1 => (0 : Idx 1)) i)) ∧
    ∀ k : Idx 1, (fun _ : Idx 1 => (0 : EucSpace 1)) ((fun _ : Idx 1 => (0 : Idx 1)) k) =
      (fun _ : Idx 1 => (0 : EucSpace 1)) k :=
  have hgen := genericPolytope_single (ContinuousLinearMap.id ℝ (EucSpace 1)) 0
  ⟨hgen, fun _ => (hgen.ownCell 0).1, fun _ => rfl⟩

end FrankWolfe
end Transformer
