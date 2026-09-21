/-
# Attention's forward pass and Frank-Wolfe — dynamic metastability

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §5, `lem: first.phase` and
`lem: metastab.1`.

The self-attention process `(SA_ℙ)` first clusters at the vertices of the
polytope in `O(1)` steps, and then stays there for a time exponential in `β`.
-/

import Transformer.FrankWolfe.Section5_Process
import Transformer.FrankWolfe.Section4_Cells
import Mathlib.Geometry.Euclidean.Angle.Unoriented.Basic

open scoped BigOperators Pointwise
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- The interior erosion `A ⊖ δB₁ = {x : x + δB₁ ⊆ A}`.

Source: arXiv:2508.09628v1, §5. -/
def erosion (A : Set (EucSpace d)) (δ : ℝ) : Set (EucSpace d) :=
  {x | Metric.closedBall x δ ⊆ A}

/-- `𝓘_i(η) = {x ∈ 𝒦 : ⟨v_i - v_j, x⟩ ≥ η for all j ≠ i}`, the cone in which
`v_i` dominates every other vertex by at least `η`.  For `η = 0` and `B = I_d`
this is the cell `𝒞_i(v)` of `Section4_Cells`.

Source: arXiv:2508.09628v1, §5. -/
def dominanceCone (K : Set (EucSpace d)) (v : Idx κ → EucSpace d) (i : Idx κ) (η : ℝ) :
    Set (EucSpace d) :=
  {x ∈ K | ∀ j : Idx κ, j ≠ i → η ≤ inner (𝕜 := ℝ) (v i - v j) x}

/-- The hypotheses `lem: first.phase` puts on the polytope `𝒦` and its
vertices `v_1, …, v_κ`.

**What the source says and what is changed here.**  `eq: hypothesis.polytope`
is written `sup_{x,y ∈ 𝒦} arccos⟨x - v_i, y - v_i⟩ < π/2`.  `arccos` of an
unnormalized inner product is not an angle, and the intended quantity is the
angle of the cone of `𝒦` at `v_i`, so it is recorded here as
`InnerProductGeometry.angle`.  The supremum being `< π/2` is recorded as a
uniform bound `θ < π/2`, which is what a supremum over a compact set gives and
what the proof uses.  The pair `x = y = v_i` is excluded: the directions are
then `0`, and Mathlib's `angle 0 0 = π/2` would make the hypothesis
unsatisfiable while saying nothing about the cone.

Source: arXiv:2508.09628v1, §5, `lem: first.phase`, hypotheses i)–iii). -/
structure ConePolytope (K : Set (EucSpace d)) (v : Idx κ → EucSpace d) (c₀ : ℝ) : Prop where
  /-- `𝒦` is the convex hull of `v`. -/
  hull : K = convexHull ℝ (Set.range v)
  /-- `v` enumerates the vertices of `𝒦`. -/
  isVertexList : IsVertexList K v
  /-- `eq: hypothesis.polytope`: the cone of `𝒦` at each vertex is acute. -/
  acute : ∀ i : Idx κ, ∃ θ < π / 2, ∀ x ∈ K, ∀ y ∈ K, x ≠ v i → y ≠ v i →
    InnerProductGeometry.angle (x - v i) (y - v i) ≤ θ
  /-- `eq: hypothesis.voronoi`: all vertices have the same norm. -/
  equinorm : ∀ i j : Idx κ, ‖v i‖ = ‖v j‖
  /-- `c₀` is positive. -/
  sep_pos : 0 < c₀
  /-- `eq: bound.max`: `⟨v_ι, v_ι⟩ ≥ ⟨v_ι, v_ℓ⟩ + c₀` for `ι ≠ ℓ`. -/
  sep : ∀ i j : Idx κ, i ≠ j →
    inner (𝕜 := ℝ) (v i) (v j) + c₀ ≤ inner (𝕜 := ℝ) (v i) (v i)

/-- **Theorem (lem: first.phase) — clustering.**

For a polytope satisfying `ConePolytope` and `n ≥ κ` particles, there is
`β_* > 0` such that for `β ≥ β_*` the process `(SA_ℙ)` started from a
configuration whose first `κ` particles are the vertices and whose remaining
particles sit in the eroded cones `𝓘_{σ(i)}(β^{-1/8}) ⊖ β^{-1/4}B₁` satisfies,
at the time `T₁`,

  `ℙ[all vertices within β^{-1/4}, all other particles within Cτ of their
     vertex] ≥ 1 - β^{-1/8}`

with `C > 1` universal.

**What the source says and what is changed here.**  Three bookkeeping changes,
no weakening.

`C` is quantified outermost, which is what "universal constant" means and is
stronger than letting it depend on the data.

`τ` and the quantity `m = min_{j > κ} ‖x_j^0 - v_{σ(j)}‖` it is built from are
carried as parameters pinned by the properties that characterize them
(`IsGreatest`, `IsLeast`), rather than written as iterated `min`s needing
nonemptiness side conditions.  `IsGreatest` of the three-fold constraint set
*is* the source's minimum.

The particles `i ∈ [1, κ]` and `i ∈ [κ+1, n]` are separated by `(i : ℕ) < κ`,
the vertices being `v ⟨i, h⟩`.

Not proved here.

Source: arXiv:2508.09628v1, §5, `lem: first.phase`. -/
theorem first_phase :
    ∃ C : ℝ, 1 < C ∧
      ∀ (d κ n : ℕ) (K : Set (EucSpace d)) (v : Idx κ → EucSpace d) (c₀ γ : ℝ),
        ConePolytope K v c₀ → γ ∈ Set.Ioo (0 : ℝ) 1 → κ ≤ n →
        ∃ βstar > (0 : ℝ), ∀ β ≥ βstar,
          ∀ (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ) (m τ : ℝ) (T₁ : ℕ)
            (P : Measure (ℕ → Idx n → EucSpace d)),
            (∀ i : Idx n, X₀ i ∈ K) →
            (∀ (i : Idx n) (h : (i : ℕ) < κ), X₀ i = v ⟨i, h⟩) →
            (∀ i : Idx n, X₀ i ∈
              cell (ContinuousLinearMap.id ℝ (EucSpace d)) K v (σ i)) →
            (∀ i : Idx n, κ ≤ (i : ℕ) →
              X₀ i ∈ erosion (dominanceCone K v (σ i) (β ^ (-(1 : ℝ) / 8)))
                (β ^ (-(1 : ℝ) / 4))) →
            IsLeast {r : ℝ | ∃ j : Idx n, κ ≤ (j : ℕ) ∧ r = ‖X₀ j - v (σ j)‖} m →
            IsGreatest {r : ℝ | (∀ i j : Idx κ, j ≠ i → r * (2 * ‖v i - v j‖) ≤ c₀) ∧
              r ≤ Real.sqrt (2 * c₀) / 2 ∧ r ≤ (1 - γ) * m} τ →
            T₁ = ⌊Real.log (τ / m) / Real.log (1 - γ)⌋₊ →
            IsSAProcess β γ X₀ P →
            1 - β ^ (-(1 : ℝ) / 8) ≤
              (P {x : ℕ → Idx n → EucSpace d |
                (∀ (i : Idx n) (h : (i : ℕ) < κ),
                    x T₁ i ∈ Metric.ball (v ⟨i, h⟩) (β ^ (-(1 : ℝ) / 4))) ∧
                (∀ i : Idx n, κ ≤ (i : ℕ) →
                    x T₁ i ∈ Metric.ball (v (σ i)) (C * τ))}).toReal := by
  sorry

/-- The hypotheses `first_phase` carries in its binders are satisfiable: one
vertex `e₀` in `ℝ^1`, `𝒦 = {e₀}`, `c₀ = 1`, `γ = 1/2`, `n = κ = 1`. -/
example :
    ConePolytope {(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))}
      (fun _ : Idx 1 => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))) 1 ∧
    (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧ (1 : ℕ) ≤ 1 := by
  refine ⟨⟨?_, ⟨?_, ?_⟩, fun i => ⟨0, by positivity, fun x hx y hy hxi hyi => ?_⟩,
    fun i j => rfl, one_pos, fun i j hij => absurd (Subsingleton.elim i j) hij⟩,
    ⟨by norm_num, by norm_num⟩, le_rfl⟩
  · rw [Set.range_const, convexHull_singleton]
  · intro a b _
    exact Subsingleton.elim a b
  · rw [Set.range_const, extremePoints_singleton]
  · exact absurd hx hxi

/-- The convex hull of the particles that `σ` assigns to the vertex `a`.

Source: arXiv:2508.09628v1, §5, `lem: metastab.1`. -/
def groupHull (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ) (a : Idx κ) : Set (EucSpace d) :=
  convexHull ℝ {y : EucSpace d | ∃ j : Idx n, σ j = a ∧ y = X₀ j}

/-- **Theorem (lem: metastab.1) — metastability.**

In the setup of `lem: first.phase`, there is `ε_* > 0` such that for
`ε ∈ (0, ε_*)` and `γ ∈ (0,1)` with `ε/γ ≥ 2 d(𝒦)`, a configuration in `𝒦ⁿ`
whose particles all sit within `Cτ` of their vertex has exit time

  `T₂ = inf{t : x_{ji}^t ∉ conv{x_{ji}^0}_j + B(0,ε) for some i, j}`

obeying, for `t > 1`,

  `ℙ[T₂ ≥ t] ≥ 1 - exp((1 + ε/γ) log(γt/ε) + (1 + ε/γ) log n - β c₀ ε / (2γ))`.

Since `γ/ε ≲ 1/d(𝒦)`, the right-hand side is close to `1` up to `t ∼ e^{βc₀/2}`.

**What the source says and what is changed here.**  "The setup of
`lem: first.phase`" is read out in full.  `C > 1` is the universal constant of
that lemma, quantified outermost as there; `β ≥ β_*` with `β_*` depending on
`n`, the polytope and `γ`, as there.  `τ` is the minimum of that lemma; its
third term, `(1-γ) min_j ‖x_j^0 - v_{σ(j)}‖`, refers to the configuration the
first phase started from, which is not in this statement, so `τ` is any
positive number below the first two terms: every value the source's `τ` can
take is such a number, and this is no weaker.  The radius must not be left
free: a particle far from its vertex but assigned to it is pulled out of its
group's hull at once.

`T₂` is not introduced as an `ℕ∞`-valued random variable.  The event
`{T₂ ≥ t}` is the event that no particle has left its group's `ε`-neighbourhood
at any step `s < t`, and that is what the conclusion is stated about — the same
set, without the extra definition.

Relabelling `x_ℓ^0` as `x_{ji}^0` is the fibre `σ⁻¹(i)`, so `conv{x_{ji}^0}_j`
is `groupHull X₀ σ (σ i)`.

Not proved here.

Source: arXiv:2508.09628v1, §5, `lem: metastab.1`, `eq: metastability.bound`. -/
theorem metastability :
    ∃ C : ℝ, 1 < C ∧
      ∀ (d κ n : ℕ) (K : Set (EucSpace d)) (v : Idx κ → EucSpace d) (c₀ : ℝ),
        ConePolytope K v c₀ → κ ≤ n →
        ∃ εstar > (0 : ℝ), ∀ γ ∈ Set.Ioo (0 : ℝ) 1, ∃ βstar > (0 : ℝ),
          ∀ ε ∈ Set.Ioo (0 : ℝ) εstar, 2 * Metric.diam K ≤ ε / γ →
          ∀ (β τ : ℝ) (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ)
            (P : Measure (ℕ → Idx n → EucSpace d)),
            βstar ≤ β → 0 < τ →
            (∀ i j : Idx κ, j ≠ i → τ * (2 * ‖v i - v j‖) ≤ c₀) →
            τ ≤ Real.sqrt (2 * c₀) / 2 →
            (∀ i : Idx n, X₀ i ∈ K) →
            (∀ i : Idx n, X₀ i ∈ Metric.ball (v (σ i)) (C * τ)) →
            IsSAProcess β γ X₀ P →
            ∀ t : ℝ, 1 < t →
              1 - Real.exp ((1 + ε / γ) * Real.log (γ / ε * t) + (1 + ε / γ) * Real.log n
                  - β * (c₀ / 2) * (ε / γ))
                ≤ (P {x : ℕ → Idx n → EucSpace d | ∀ s : ℕ, (s : ℝ) < t → ∀ i : Idx n,
                    x s i ∈ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε}).toReal := by
  sorry

/-- The hypotheses of `metastability` are satisfiable, the process included:
one vertex `e₀` in `ℝ^1`, `𝒦 = {e₀}`, `c₀ = 1`, `γ = 1/2`, `ε = 1/4`,
`τ = 1/2`, one particle sitting at the vertex, whose process is the point mass
at the constant path (`isSAProcess_single`), for whatever `C > 0`. -/
example (C : ℝ) (hC : 0 < C) :
    ConePolytope {(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))}
      (fun _ : Idx 1 => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))) 1 ∧ (1 : ℕ) ≤ 1 ∧
    (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧
    2 * Metric.diam {(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))} ≤ (1 / 4 : ℝ) / (1 / 2) ∧
    (0 : ℝ) < 1 / 2 ∧
    (∀ i j : Idx 1, j ≠ i → (1 / 2 : ℝ) * (2 * ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) -
      (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖) ≤ 1) ∧
    (1 / 2 : ℝ) ≤ Real.sqrt (2 * 1) / 2 ∧
    (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ∈ Metric.ball
      (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) (C * (1 / 2)) ∧
    IsSAProcess 1 (1 / 2) (fun _ : Idx 1 => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)))
      (Measure.dirac fun _ => fun _ => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))) := by
  refine ⟨⟨?_, ⟨?_, ?_⟩, fun i => ⟨0, by positivity, fun x hx y hy hxi hyi => ?_⟩,
    fun i j => rfl, one_pos, fun i j hij => absurd (Subsingleton.elim i j) hij⟩, le_rfl,
    ⟨by norm_num, by norm_num⟩, by simp, by norm_num, fun i j h => by simp,
    ?_, Metric.mem_ball_self (by positivity), isSAProcess_single _ _ _⟩
  · rw [Set.range_const, convexHull_singleton]
  · intro a b _
    exact Subsingleton.elim a b
  · rw [Set.range_const, extremePoints_singleton]
  · exact absurd hx hxi
  · rw [div_le_div_iff_of_pos_right two_pos, Real.le_sqrt zero_le_one (by norm_num)]
    norm_num

end FrankWolfe
end Transformer
