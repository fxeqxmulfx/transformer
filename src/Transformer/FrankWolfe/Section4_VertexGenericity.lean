/-
# Attention's forward pass and Frank-Wolfe — when a vertex owns its cell

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4, the two remarks on the
genericity of `eq: vertices.own.cell`.

A vertex of `𝒦` need not lie in its own cell.  `prop: d.to.infty` says that in
high dimension Gaussian vertices do, with probability tending to one;
`prop: polytope.condition` characterizes it, for a fixed polytope, by a
condition on the edges at that vertex.
-/

import Transformer.FrankWolfe.Section4_Polytope
import Transformer.FrankWolfe.Section4_Edges
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Geometry.Euclidean.Angle.Unoriented.Affine

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

variable {d κ : ℕ}

/-- **Remark (prop: d.to.infty) — Gaussian vertices in high dimension.**

Let `v_1, …, v_κ` be i.i.d. `𝒩(0, I_d)` and let `B ≻ 0` have condition number
bounded independently of `d`.  With `κ` fixed and `d → +∞`, with probability
tending to `1`,

  `⟨B v_i, v_i⟩ > ⟨B v_i, v_j⟩`   for all `j ≠ i`,

that is, every vertex lies in its own cell and no two vertices lie in the same
cell.

**What the source says and what is changed here.**  "Condition number bounded
independently of `d`" is spelled out as the two-sided bound the source's own
proof uses: constants `0 < λ_min ≤ λ_max` independent of `d` with
`λ_min ‖x‖² ≤ ⟨B x, x⟩ ≤ λ_max ‖x‖²`.  The family `B` is therefore indexed by
`d`, as it must be for `d → ∞` to mean anything.

Not proved here.

Source: arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
theorem gaussian_vertices_own_cell (κ : ℕ) (B : (d : ℕ) → ParamMatrix d) (lmin lmax : ℝ)
    (hlmin : 0 < lmin)
    (hlb : ∀ (d : ℕ) (x : EucSpace d), lmin * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) (B d x) x)
    (hub : ∀ (d : ℕ) (x : EucSpace d), inner (𝕜 := ℝ) (B d x) x ≤ lmax * ‖x‖ ^ 2) :
    Filter.Tendsto
      (fun d : ℕ => (Measure.pi fun _ : Idx κ => ProbabilityTheory.stdGaussian (EucSpace d))
        {v : Idx κ → EucSpace d | ∀ i j : Idx κ, j ≠ i →
          inner (𝕜 := ℝ) (B d (v i)) (v j) < inner (𝕜 := ℝ) (B d (v i)) (v i)})
      Filter.atTop (nhds 1) := by
  sorry

/-- The hypotheses of `gaussian_vertices_own_cell` are satisfiable: `B_d = I_d`,
`λ_min = λ_max = 1`. -/
example :
    (0 : ℝ) < 1 ∧
    (∀ (d : ℕ) (x : EucSpace d),
      (1 : ℝ) * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace d) x) x) ∧
    (∀ (d : ℕ) (x : EucSpace d),
      inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace d) x) x ≤ (1 : ℝ) * ‖x‖ ^ 2) := by
  refine ⟨one_pos, fun d x => ?_, fun d x => ?_⟩ <;> simp

/-- `eq: neigh`: the vertices adjacent to `v_i` in `𝒦 = conv{v_ℓ}` — those
`v ≠ v_i` for which the segment `[v_i, v]` is a face of `𝒦`.

A face is an extreme subset (`IsExtreme`), and a one-dimensional face of a
polytope is exactly a segment joining two vertices, so this is the source's
"there exists an edge connecting `v` and `v_i`".

Source: arXiv:2508.09628v1, §4, `eq: neigh`. -/
def neigh (v : Idx κ → EucSpace d) (i : Idx κ) : Set (EucSpace d) :=
  {w | w ∈ Set.range v ∧ w ≠ v i ∧ IsExtreme ℝ (configHull v) (segment ℝ (v i) w)}

/-- **Remark (prop: polytope.condition) — the tangent-hyperplane condition.**

Let `B ≻ 0` and let `𝒥(x) = ½⟨Bx, x⟩`.  The hyperplane tangent at `v` to the
level set `{𝒥 = 𝒥(v)}` is `{x | ⟨a_v, x⟩ + b_v = 0}` with `a_v = Bv`,
`b_v = -⟨Bv, v⟩`, so `⟨a_v, x⟩ + b_v = ⟨Bv, x - v⟩`.  For a vertex `v_i ≠ 0`,

  `v_i` owns its cell   ↔   `⟨B v_i, w - v_i⟩ < 0` for all `w ∈ neigh(v_i) ∪ {0}`.

**What the source says and what is changed here.**  Two deviations, both
forced.

First, the source's `eq: cond.hyperplane` is a disjunction: all of
`neigh(v_i) ∪ {0}` strictly on one side *or* strictly on the other.  The
positive branch is empty: at `w = 0` it reads `⟨B v_i, -v_i⟩ > 0`, i.e.
`⟨B v_i, v_i⟩ < 0`, which `B ≻ 0` forbids for `v_i ≠ 0`.  Only the negative
branch survives, and it is the one kept here.

Second, the source states the equivalence against `v_i ∈ 𝒞_i(v)`, whose
defining inequality is non-strict; a neighbour `w` with
`⟨B v_i, w⟩ = ⟨B v_i, v_i⟩` then satisfies the left side and not the right,
so the equivalence is false as written.  The strict condition characterizes
`eq: vertices.own.cell` — `v_i ∈ 𝒞_i(v)` *and* `v_i ∉ 𝒞_j(v)` for `j ≠ i` —
which is the hypothesis `thm: exp.fast.polytope` actually needs, and that is
what is stated here.

The hypothesis `v_i ≠ 0` is the source's, implicitly: `ℋ(v_i)` is a
hyperplane only when its normal `a_{v_i} = B v_i` is nonzero, which under
`B ≻ 0` is `v_i ≠ 0`.

The proof: the left side says that `⟨B v_i, ·⟩` is strictly larger at `v_i`
than at every other vertex; `w = 0` is automatic from `B ≻ 0`; and a vertex
beating its neighbours beats every vertex (`lt_of_forall_isExtreme_segment`).

Source: arXiv:2508.09628v1, §4, `prop: polytope.condition`,
`eq: cond.hyperplane`. -/
theorem tangent_hyperplane_condition (B : ParamMatrix d) (hB : IsPosDef B)
    (v : Idx κ → EucSpace d) (hv : IsVertexList (configHull v) v) (i : Idx κ)
    (hvi : v i ≠ 0) :
    (v i ∈ cell B (configHull v) v i ∧ ∀ j : Idx κ, j ≠ i → v i ∉ cell B (configHull v) v j)
      ↔ ∀ w ∈ insert (0 : EucSpace d) (neigh v i),
          inner (𝕜 := ℝ) (B (v i)) (w - v i) < 0 := by
  have hvK : ∀ j, v j ∈ configHull v := fun j => subset_convexHull ℝ _ (Set.mem_range_self j)
  constructor
  · rintro ⟨hi, hj⟩ w hw
    rcases hw with rfl | ⟨⟨j, rfl⟩, hne, -⟩
    · rw [zero_sub, inner_neg_right, neg_lt_zero]
      exact hB.2 _ hvi
    · have hji : j ≠ i := fun h => hne (congrArg v h)
      have hnot := hj j hji
      simp only [cell, Set.mem_sep_iff, not_and, not_forall, not_le] at hnot
      obtain ⟨y, hy, hlt⟩ := hnot (hvK i)
      rw [inner_sub_right, sub_neg]
      exact hlt.trans_le (hi.2 y hy)
  · intro h
    have hlt : ∀ x ∈ configHull v, x ≠ v i →
        inner (𝕜 := ℝ) (B (v i)) x < inner (𝕜 := ℝ) (B (v i)) (v i) :=
      lt_of_forall_isExtreme_segment (Set.finite_range v) hv.2.subset (Set.mem_range_self i)
        (innerSL ℝ (B (v i))) fun w hw hwi hseg => by
          have := h w (Set.mem_insert_of_mem _ ⟨hw, hwi, hseg⟩)
          rwa [inner_sub_right, sub_neg] at this
    refine ⟨⟨hvK i, fun y hy => ?_⟩, fun j hji hj => ?_⟩
    · rcases eq_or_ne y (v i) with rfl | hyi
      · exact le_rfl
      · exact (hlt y hy hyi).le
    · exact (hj.2 (v i) (hvK i)).not_gt (hlt _ (hvK j) (hv.1.ne hji))

/-- **Remark (prop: polytope.condition) — the angle condition.**

For `B = I_d` the tangent-hyperplane condition is the angle condition
`eq: cond.angle`: a vertex `v_i ≠ 0` owns its cell if and only if the angle at
`v_i` between `w` and the origin is acute for every adjacent vertex `w`.

**What the source says and what is changed here.**  The same two repairs as in
`tangent_hyperplane_condition`: "owns its cell" is `v_i ∈ 𝒞_i(v)` together
with `v_i ∉ 𝒞_j(v)` for `j ≠ i`, and `v_i ≠ 0` is required — at `v_i = 0` the
angle `∠_{v_i}(w, 0)` is not defined by a nonzero pair.

The proof is `tangent_hyperplane_condition` at `B = I_d`: the angle at `v_i` is
acute iff `⟨w - v_i, -v_i⟩ > 0`, and the condition at `w = 0` holds.

Source: arXiv:2508.09628v1, §4, `prop: polytope.condition`,
`eq: cond.angle`. -/
theorem angle_condition (v : Idx κ → EucSpace d) (hv : IsVertexList (configHull v) v)
    (i : Idx κ) (hvi : v i ≠ 0) :
    (v i ∈ cell (ContinuousLinearMap.id ℝ (EucSpace d)) (configHull v) v i ∧
        ∀ j : Idx κ, j ≠ i →
          v i ∉ cell (ContinuousLinearMap.id ℝ (EucSpace d)) (configHull v) v j)
      ↔ ∀ w ∈ neigh v i, EuclideanGeometry.angle w (v i) (0 : EucSpace d) < π / 2 := by
  have hid : IsPosDef (ContinuousLinearMap.id ℝ (EucSpace d)) :=
    ⟨fun x y => rfl, fun x hx => real_inner_self_pos.mpr hx⟩
  have hacute : ∀ x y : EucSpace d,
      InnerProductGeometry.angle x y < π / 2 ↔ 0 < inner (𝕜 := ℝ) x y := fun x y => by
    rw [← not_le, ← InnerProductGeometry.inner_nonpos_iff_pi_div_two_le_angle, not_le]
  rw [tangent_hyperplane_condition _ hid v hv i hvi, Set.forall_mem_insert]
  have h0 : inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace d) (v i)) (0 - v i) < 0 := by
    rw [zero_sub, inner_neg_right, neg_lt_zero]
    exact hid.2 _ hvi
  simp only [h0, true_and]
  refine forall₂_congr fun w _ => ?_
  rw [EuclideanGeometry.angle, hacute, vsub_eq_sub, vsub_eq_sub, zero_sub, inner_neg_right,
    neg_pos, ContinuousLinearMap.id_apply, real_inner_comm]

/-- The hypotheses shared by `tangent_hyperplane_condition` and
`angle_condition` are satisfiable: `d = 1`, a single vertex `v ≡ e₀ ≠ 0`,
`B = I₁`. -/
example :
    IsPosDef (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    IsVertexList (configHull (fun _ : Idx 1 => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))))
      (fun _ : Idx 1 => (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))) ∧
    (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ≠ 0 := by
  have he : (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ≠ 0 := by
    intro h
    have h0 : (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 0 = (0 : EucSpace 1) 0 := by rw [h]
    simp at h0
  refine ⟨⟨fun x y => rfl, fun x hx => real_inner_self_pos.mpr hx⟩, ⟨?_, ?_⟩, he⟩
  · intro a b _
    exact Subsingleton.elim a b
  · rw [configHull, Set.range_const, convexHull_singleton, extremePoints_singleton]

end FrankWolfe
end Transformer
