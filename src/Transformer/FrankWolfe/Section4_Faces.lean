/-
# Attention's forward pass and Frank-Wolfe — maximizing faces of a polytope

Two geometric facts behind the proof of `thm: exp.fast.polytope`.  The source
states "the maximizer of `⟨B x_i^0, y⟩` over `𝒦` is `v_{σ(i)}`" and "by
convexity of the cell … `x_i^1 ∈ 𝒞_{σ(i)}(v)`"; here:

* the maximizers of `⟨a, ·⟩` over a polytope form an exposed face, which by
  Krein–Milman is the convex hull of the vertices it contains, so a unique
  maximizing vertex is the unique maximizer (`eq_of_isMaximizerOn`);
* the segment from a point of the cell `𝒞_k` to `v_k` meets no other cell
  when neither end does (`mem_cell_segment_iff`).

Source: arXiv:2508.09628v1, §4, proof of `thm: exp.fast.polytope`
(`sec: proof.exp.fast`).
-/

import Transformer.FrankWolfe.Section4_Cells
import Mathlib.Analysis.Convex.KreinMilman
import Mathlib.Analysis.Convex.Exposed

open Set

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- A polytope is the convex hull of its vertices (Krein–Milman); used for
`𝒦 = conv{v_j}` in the proof of `thm: exp.fast.polytope`. -/
theorem configHull_eq_convexHull {X : Idx n → EucSpace d} {v : Idx κ → EucSpace d}
    (hv : IsVertexList (configHull X) v) : configHull X = convexHull ℝ (range v) := by
  have hK := closure_convexHull_extremePoints (s := configHull X) ((finite_range X).isCompact_convexHull (𝕜 := ℝ))
    (convex_convexHull ℝ _)
  rw [← hv.2, ((finite_range v).isCompact_convexHull (𝕜 := ℝ)).isClosed.closure_eq] at hK
  exact hK.symm

/-- **The only maximizer is the only maximizing vertex.**  If `v k` is the
only vertex of the polytope `K` maximizing `⟨a, ·⟩`, every maximizer is `v k`:
the maximizers form an exposed face of `K`, whose extreme points are vertices
of `K`, and which is the convex hull of them (Krein–Milman). -/
theorem eq_of_isMaximizerOn {K : Set (EucSpace d)} {v : Idx κ → EucSpace d}
    (hKc : IsCompact K) (hKv : Convex ℝ K) (hv : range v = K.extremePoints ℝ)
    {a y : EucSpace d} {k : Idx κ} (hy : IsMaximizerOn a K y)
    (hk : ∀ j, (∀ z ∈ K, inner (𝕜 := ℝ) a z ≤ inner (𝕜 := ℝ) a (v j)) → j = k) :
    y = v k := by
  set M := {x ∈ K | ∀ z ∈ K, inner (𝕜 := ℝ) a z ≤ inner (𝕜 := ℝ) a x}
  have hM : IsExposed ℝ K M := fun _ => ⟨innerSL ℝ a, by simp [M]⟩
  have hext : M.extremePoints ℝ ⊆ {v k} := by
    intro x hx
    rw [hM.isExtreme.extremePoints_eq] at hx
    obtain ⟨j, rfl⟩ : x ∈ range v := hv ▸ hx.2
    rw [hk j hx.1.2]
    rfl
  have hsub : M ⊆ {v k} := by
    rw [← closure_convexHull_extremePoints (hM.isCompact hKc) (hM.convex hKv)]
    exact closure_minimal (convexHull_min hext (convex_singleton _)) isClosed_singleton
  exact hsub hy

/-- **The segment to the vertex stays in its cell alone.**  If `x` and `v_k`
lie in the cell `𝒞_k` and in no other, so does every point of the segment
between them. -/
theorem mem_cell_segment_iff (B : ParamMatrix d) {K : Set (EucSpace d)} (hK : Convex ℝ K)
    {v : Idx κ → EucSpace d} (hvK : ∀ j, v j ∈ K) {x : EucSpace d} {k : Idx κ}
    (hx : ∀ j, x ∈ cell B K v j ↔ j = k) (hv : ∀ j, v k ∈ cell B K v j ↔ j = k)
    {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (j : Idx κ) :
    a • x + (1 - a) • v k ∈ cell B K v j ↔ j = k := by
  have hxk := (hx k).2 rfl
  have hvk := (hv k).2 rfl
  have hmem : a • x + (1 - a) • v k ∈ cell B K v k :=
    convex_cell B hK v k hxk hvk ha0 (by linarith) (by ring)
  -- Both ends prefer `v k` strictly to every other vertex.
  have hstrict : ∀ z ∈ cell B K v k, (∀ j, z ∈ cell B K v j → j = k) → ∀ j, j ≠ k →
      inner (𝕜 := ℝ) (B z) (v j) < inner (𝕜 := ℝ) (B z) (v k) := by
    intro z hz hzj j hj
    by_contra hle
    exact hj (hzj j ⟨hz.1, fun y hy => (hz.2 y hy).trans (not_lt.1 hle)⟩)
  refine ⟨fun hj => ?_, fun h => h ▸ hmem⟩
  by_contra hjk
  have h1 := hstrict x hxk (fun j h => (hx j).1 h) j hjk
  have h2 := hstrict (v k) hvk (fun j h => (hv j).1 h) j hjk
  have h3 := hj.2 (v k) (hvK k)
  simp only [map_add, map_smul, inner_add_left, inner_smul_left, RCLike.conj_to_real] at h3
  rcases ha0.eq_or_lt with rfl | hapos
  · linarith
  · nlinarith [mul_pos hapos (sub_pos.2 h1), mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 h2.le)]

/-- The hypotheses of `eq_of_isMaximizerOn` and `mem_cell_segment_iff` are
satisfiable: the one-point polytope `{0}` of `ℝ^1`, its single vertex, `B = I`. -/
example : IsCompact ({0} : Set (EucSpace 1)) ∧ Convex ℝ ({0} : Set (EucSpace 1)) ∧
    range (fun _ : Idx 1 => (0 : EucSpace 1)) = ({0} : Set (EucSpace 1)).extremePoints ℝ ∧
    IsMaximizerOn 0 ({0} : Set (EucSpace 1)) 0 ∧
    ∀ j : Idx 1, (0 : EucSpace 1) ∈
      cell (ContinuousLinearMap.id ℝ (EucSpace 1)) {0} (fun _ => 0) j ↔ j = 0 := by
  refine ⟨isCompact_singleton, convex_singleton _, by simp [range_const],
    ⟨rfl, fun z hz => by simp_all⟩, fun j => ?_⟩
  simp [cell, Subsingleton.elim j 0]

end FrankWolfe
end Transformer
