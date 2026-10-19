module

public import Transformer.FrankWolfe.Section5_Process
public import Transformer.FrankWolfe.Section4_Cells
public import Mathlib.Geometry.Euclidean.Angle.Unoriented.Basic

/-!
# Attention's forward pass and Frank-Wolfe — dynamic metastability

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §5, `lem: first.phase` and
`lem: metastab.1`.

The source claims clustering at the polytope vertices in `O(1)` steps,
followed by a time exponential in `β` near those vertices. Its printed
first-phase claim is refuted by `not_first_phase`.
This module retains the geometric data and proves the valid one-step
non-entry bound. The conditional metastability proof, starting from a
configuration already close to the vertices, is in `Section5_MetaBound`,
after the transition, drift, and stopped-event probability estimates.
-/

@[expose] public section

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
`v_i` dominates every other vertex by at least `η`. For `η = 0` and `B = I_d`
this is the cell `𝒞_i(v)` of `Section4_Cells`.

Source: arXiv:2508.09628v1, §5. -/
def dominanceCone (K : Set (EucSpace d)) (v : Idx κ → EucSpace d) (i : Idx κ) (η : ℝ) :
    Set (EucSpace d) :=
  {x ∈ K | ∀ j : Idx κ, j ≠ i → η ≤ inner (𝕜 := ℝ) (v i - v j) x}

/-- The hypotheses `lem: first.phase` puts on the polytope `𝒦` and its
vertices `v_1, …, v_κ`.

**What the source says and what is changed here.**  `eq: hypothesis.polytope`
is written `sup_{x,y ∈ 𝒦} arccos⟨x - v_i, y - v_i⟩ < π/2`. `arccos` of an
unnormalized inner product is not an angle, and the intended quantity is the
angle of the cone of `𝒦` at `v_i`, so it is recorded here as
`InnerProductGeometry.angle`. The supremum being `< π/2` is recorded as a
uniform bound `θ < π/2`, which is what a supremum over a compact set gives and
what the proof uses. The pair `x = y = v_i` is excluded: the directions are
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

/-! **The printed theorem (lem: first.phase) — clustering, refuted.**

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
nonemptiness side conditions. `IsGreatest` of the three-fold constraint set
*is* the source's minimum.

The particles `i ∈ [1, κ]` and `i ∈ [κ+1, n]` are separated by `(i : ℕ) < κ`,
the vertices being `v ⟨i, h⟩`.

The printed universal radius and minimum-distance time are incompatible.
`not_first_phase`, in `Section5_FirstPhaseRefutation`, proves a counterexample
with four actual scalar particles and a genuine process law. Every geometric,
erosion, cell, minimum, radius and time hypothesis above is satisfied there.
The source's valid non-entry estimate is proved separately below.

Source: arXiv:2508.09628v1, §5, `lem: first.phase`. -/
/-- The original geometric hypotheses of `lem: first.phase` are satisfiable: one
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

/-- The original acute-cone assumption gives the valid one-step non-entry
bound from Step 1 of the source's proof: a convex sampled step cannot
reduce the distance to a vertex faster than the factor `1-γ`.
Source: arXiv:2508.09628v1, §5, `lem: cone.condition`, `eq: aprior.bound.vertex`. -/
theorem ConePolytope.norm_step_sub_vertex_ge {K : Set (EucSpace d)}
    {v : Idx κ → EucSpace d} {c₀ : ℝ} (hK : ConePolytope K v c₀)
    (γ : ℝ) (hγ₀ : 0 ≤ γ) (hγ₁ : γ ≤ 1) (i : Idx κ) (x y : EucSpace d)
    (hx : x ∈ K) (hy : y ∈ K) :
    (1 - γ) * ‖x - v i‖ ≤ ‖(1 - γ) • x + γ • y - v i‖ := by
  have hab : 0 ≤ inner (𝕜 := ℝ) (x - v i) (y - v i) := by
    by_cases hxi : x = v i
    · simp [hxi]
    by_cases hyi : y = v i
    · simp [hyi]
    obtain ⟨θ, hθ, hangle⟩ := hK.acute i
    have hlt := lt_of_le_of_lt (hangle x hx y hy hxi hyi) hθ
    have hp : 0 < inner (𝕜 := ℝ) (x - v i) (y - v i) := by
      by_contra hn
      exact (not_le_of_gt hlt)
        (InnerProductGeometry.inner_nonpos_iff_pi_div_two_le_angle.mp (le_of_not_gt hn))
    exact hp.le
  have hγ : 0 ≤ 1 - γ := sub_nonneg.mpr hγ₁
  have heq : (1 - γ) • x + γ • y - v i = (1 - γ) • (x - v i) + γ • (y - v i) := by module
  rw [heq]
  apply (sq_le_sq₀ (mul_nonneg hγ (norm_nonneg _)) (norm_nonneg _)).mp
  rw [norm_add_sq_real, norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_nonneg hγ, abs_of_nonneg hγ₀, real_inner_smul_left, real_inner_smul_right]
  have hcross : 0 ≤ 2 * ((1 - γ) * (γ * inner (𝕜 := ℝ) (x - v i) (y - v i))) := by positivity
  nlinarith [sq_nonneg (γ * ‖y - v i‖)]

/-- The convex hull of the particles that `σ` assigns to the vertex `a`.

Source: arXiv:2508.09628v1, §5, `lem: metastab.1`. -/
def groupHull (X₀ : Idx n → EucSpace d) (σ : Idx n → Idx κ) (a : Idx κ) : Set (EucSpace d) :=
  convexHull ℝ {y : EucSpace d | ∃ j : Idx n, σ j = a ∧ y = X₀ j}


end FrankWolfe
end Transformer
