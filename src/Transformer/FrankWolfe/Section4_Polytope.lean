/-
# Attention's forward pass and Frank-Wolfe — positive-definite key-query

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4.

For `B ≻ 0` the Frank-Wolfe objective is concave.  The generic literature
bound on the duality gap is `prop: trash`; under genericity assumptions on the
initial polytope (`GenericPolytope`, here) the dynamics is instead solved
exactly (`thm: exp.fast.polytope`, in `Section4_ExpFast.lean`), and the same
geometry gives well-posedness of the singular ODE `eq: hardmax.ode`
(`thm: ode`, in `Section4_ODE.lean`).
-/

import Transformer.FrankWolfe.Section4_Cells

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- **Proposition (prop: trash).**  Suppose `B^t = β^t B` for `B ≽ 0`, with
`β^t/β^{t+1} = γ^t/γ^{t+1}` for all `t ≥ 0`.  Then particles evolving according
to `(SA_∞)` satisfy

  `min_{τ ∈ [1,t]} max_{y ∈ 𝒦^τ} ⟨∇J^τ(x_i^τ), x_i^τ - y⟩
      ≤ (1/t) (J^1(x_i^1)/γ^1 - inf_{y ∈ 𝒦^t} J^t(y)/γ^t)`,

where `J^τ(y) = -½ ⟨B^τ y, y⟩`, so `∇J^τ(x) = -B^τ x`.

**What the source says and what is changed here.**  The ratio condition is
written as the cross-product `β^t γ^{t+1} = β^{t+1} γ^t`, which says the same
and needs no division.

Not proved here; the source omits the proof as well, quoting
Yurtsever-Sra, `Lemma 2.1`.

Source: arXiv:2508.09628v1, §4, `prop: trash`. -/
theorem duality_gap (B : ParamMatrix d) (β γ : ℕ → ℝ) (x : ℕ → Idx n → EucSpace d)
    (hB : ContinuousLinearMap.IsPositive B)
    (hratio : ∀ t : ℕ, β t * γ (t + 1) = β (t + 1) * γ t)
    (hflow : ∀ t : ℕ, IsHardmaxStep (β t • B) (γ t) (x t) (x (t + 1)))
    (i : Idx n) (t : ℕ) (ht : 1 ≤ t) :
    (Finset.Icc 1 t).inf' (Finset.nonempty_Icc.mpr ht)
        (fun τ => sSup {r : ℝ | ∃ y ∈ configHull (x τ),
          r = inner (𝕜 := ℝ) (-(β τ • B) (x τ i)) (x τ i - y)})
      ≤ (1 / (t : ℝ)) * (-quadForm (β 1 • B) (x 1 i) / γ 1
          - sInf {r : ℝ | ∃ y ∈ configHull (x t), r = -quadForm (β t • B) y} / γ t) := by
  sorry

/-- The hypotheses of `duality_gap` are satisfiable: `B = 0`, one particle
resting at the origin, `β^t ≡ γ^t ≡ 1`, `t = 1`. -/
example :
    ContinuousLinearMap.IsPositive (0 : ParamMatrix 1) ∧
    (∀ _ : ℕ, (1 : ℝ) * 1 = 1 * 1) ∧
    (∀ t : ℕ, IsHardmaxStep ((1 : ℝ) • (0 : ParamMatrix 1)) 1
      ((fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) t)
      ((fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) (t + 1))) ∧
    (1 : ℕ) ≤ 1 := by
  refine ⟨ContinuousLinearMap.isPositive_zero, fun _ => rfl, fun t i => ⟨0, ⟨?_, ?_⟩, by simp⟩,
    le_rfl⟩
  · exact subset_convexHull ℝ _ ⟨i, rfl⟩
  · intro z _
    simp

/-- The genericity conditions `thm: exp.fast.polytope` puts on the initial
configuration: `v` lists the vertices of `𝒦 = conv{x_i^0}`, each vertex lies
in its own cell and in no other, and every particle lies in exactly one cell.

**What the source says and what is changed here.**  The source's second
condition reads "if `x_i^0` is not a vertex, then it doesn't lie on any face
of two adjacent cells".  It is recorded here as "`x_i^0` lies in exactly one
cell", which is what makes the assignment map `σ` well-defined and is the only
consequence the proof uses; for a vertex it follows from the first condition.

Source: arXiv:2508.09628v1, §4, `thm: exp.fast.polytope`, conditions 1 and 2. -/
structure GenericPolytope (B : ParamMatrix d) (v : Idx κ → EucSpace d)
    (X : Idx n → EucSpace d) : Prop where
  /-- `v` enumerates the vertices of `𝒦 = conv{x_i^0}`. -/
  isVertexList : IsVertexList (configHull X) v
  /-- `eq: vertices.own.cell`: `v_j ∈ 𝒞_j(v) \ ⋃_{i ≠ j} 𝒞_i(v)`. -/
  ownCell : ∀ j : Idx κ, v j ∈ cell B (configHull X) v j ∧
    ∀ i : Idx κ, i ≠ j → v j ∉ cell B (configHull X) v i
  /-- Every particle lies in exactly one cell. -/
  uniqueCell : ∀ i : Idx n, ∃! j : Idx κ, X i ∈ cell B (configHull X) v j

/-- The running products `∏_{s = a}^{b-1} (1 - γ^s)` of the solution formula of
`thm: exp.fast.polytope`. -/
noncomputable def gammaProd (γ : ℕ → ℝ) (a b : ℕ) : ℝ :=
  ∏ s ∈ Finset.Ico a b, (1 - γ s)

end FrankWolfe
end Transformer
