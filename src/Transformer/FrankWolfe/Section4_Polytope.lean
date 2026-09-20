/-
# Attention's forward pass and Frank-Wolfe — positive-definite key-query

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4.

For `B ≻ 0` the Frank-Wolfe objective is concave.  The generic literature
bound on the duality gap is `prop: trash`; under genericity assumptions on the
initial polytope the dynamics is instead solved exactly
(`thm: exp.fast.polytope`), and the same geometry gives well-posedness of the
singular ODE `eq: hardmax.ode` (`thm: ode`).
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

/-- **Theorem (thm: exp.fast.polytope) — the dynamics, solved.**

Let `B^t ≡ B ≻ 0` and `γ^t ∈ (0, 1)`.  For an initial configuration
satisfying `GenericPolytope` and an assignment `σ` with `x_i^0 ∈ 𝒞_{σ(i)}(v)`,
particles evolving according to `(SA_∞)` satisfy

  `x_i^t = (∏_{τ=0}^{t-1}(1 - γ^τ)) x_i^0
            + (Σ_{τ=0}^{t-1} γ^τ ∏_{s=τ+1}^{t-1}(1 - γ^s)) v_{σ(i)}`.

**What the source says and what is changed here.**  The source lets `σ` be
produced by the statement ("the map `σ` … is well-defined"); here it is taken
as data, with `x_i^0 ∈ 𝒞_{σ(i)}(v)` as a hypothesis, which `uniqueCell`
makes available and determines uniquely.

The `argmax` of `(SA_∞)` is a relation, not a function, so the theorem covers
every selection of maximizers.  That costs nothing: `ownCell` forces the
maximizing face of `⟨B x, ·⟩` at `x = x_i^t` to be the single vertex
`v_{σ(i)}`.

Not proved here.

Source: arXiv:2508.09628v1, §4, `thm: exp.fast.polytope`. -/
theorem exp_fast_polytope (B : ParamMatrix d) (hB : IsPosDef B) (γ : ℕ → ℝ)
    (x : ℕ → Idx n → EucSpace d) (v : Idx κ → EucSpace d) (σ : Idx n → Idx κ)
    (hγ : ∀ t : ℕ, γ t ∈ Set.Ioo (0 : ℝ) 1)
    (hgen : GenericPolytope B v (x 0))
    (hσ : ∀ i : Idx n, x 0 i ∈ cell B (configHull (x 0)) v (σ i))
    (hflow : HardmaxFlow (fun _ : ℕ => B) γ x) :
    ∀ (t : ℕ) (i : Idx n),
      x t i = gammaProd γ 0 t • x 0 i
        + (∑ τ ∈ Finset.range t, γ τ * gammaProd γ (τ + 1) t) • v (σ i) := by
  sorry

/-- The hypotheses of `exp_fast_polytope` are satisfiable: `B = I` on `ℝ^1`,
one particle sitting at the single vertex `0`, `γ^t ≡ 1/2`. -/
example :
    IsPosDef (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    (∀ _ : ℕ, (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1) ∧
    GenericPolytope (ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ : Idx 1 => (0 : EucSpace 1)) (fun _ : Idx 1 => (0 : EucSpace 1)) ∧
    HardmaxFlow (fun _ : ℕ => ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ : ℕ => (1 / 2 : ℝ)) (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) := by
  have hhull : configHull (fun _ : Idx 1 => (0 : EucSpace 1)) = {(0 : EucSpace 1)} := by
    rw [configHull, Set.range_const, convexHull_singleton]
  have hcell : ∀ j : Idx 1, (0 : EucSpace 1) ∈
      cell (ContinuousLinearMap.id ℝ (EucSpace 1)) (configHull (fun _ : Idx 1 => (0 : EucSpace 1)))
        (fun _ : Idx 1 => (0 : EucSpace 1)) j := by
    intro j
    refine ⟨by rw [hhull]; rfl, fun y hy => ?_⟩
    simp
  refine ⟨⟨fun x y => rfl, fun x hx => real_inner_self_pos.mpr hx⟩, fun _ => by norm_num,
    ⟨⟨?_, ?_⟩, fun j => ⟨hcell j, fun i hij => absurd (Subsingleton.elim i j) hij⟩,
      fun i => ⟨0, hcell 0, fun j _ => Subsingleton.elim j 0⟩⟩,
    fun t i => ⟨0, ⟨?_, ?_⟩, by simp⟩⟩
  · intro a b _
    exact Subsingleton.elim a b
  · rw [Set.range_const, hhull, extremePoints_singleton]
  · exact subset_convexHull ℝ _ ⟨i, rfl⟩
  · intro z _
    simp

/-- **Theorem (thm: exp.fast.polytope), the convergence statement.**  Under the
hypotheses of `exp_fast_polytope` and a uniform lower bound `c ≤ γ^t`,

  `x_i^t → v_{σ(i)}`   exponentially fast.

**What the source says and what is changed here.**  The source concludes "in
particular, `x_i^t` converges to `v_{σ(i)}` at least exponentially fast" from
`γ^t ∈ (0,1)` alone, and that is false: for `γ^t = 2^{-t-1}` the product
`∏_{τ<t}(1 - γ^τ)` decreases to a strictly positive limit, so the solution
formula leaves `x_i^t` at a nontrivial convex combination of `x_i^0` and
`v_{σ(i)}` forever.  A uniform lower bound on the step-size repairs it, and is
what "exponentially fast" needs in any case.

Not proved here.

Source: arXiv:2508.09628v1, §4, `thm: exp.fast.polytope`, final sentence. -/
theorem exp_fast_polytope_tendsto (B : ParamMatrix d) (hB : IsPosDef B) (γ : ℕ → ℝ)
    (x : ℕ → Idx n → EucSpace d) (v : Idx κ → EucSpace d) (σ : Idx n → Idx κ) (c : ℝ)
    (hc : 0 < c) (hγ : ∀ t : ℕ, γ t ∈ Set.Ico c 1)
    (hgen : GenericPolytope B v (x 0))
    (hσ : ∀ i : Idx n, x 0 i ∈ cell B (configHull (x 0)) v (σ i))
    (hflow : HardmaxFlow (fun _ : ℕ => B) γ x) (i : Idx n) :
    ∃ C > 0, ∀ t : ℕ, ‖x t i - v (σ i)‖ ≤ C * (1 - c) ^ t := by
  sorry

/-- The hypotheses of `exp_fast_polytope_tendsto` are satisfiable: the witness
of `exp_fast_polytope`, with `c = 1/2`. -/
example : (0 : ℝ) < 1 / 2 ∧ ∀ _ : ℕ, (1 / 2 : ℝ) ∈ Set.Ico (1 / 2 : ℝ) 1 :=
  ⟨by norm_num, fun _ => by norm_num⟩

end FrankWolfe
end Transformer
