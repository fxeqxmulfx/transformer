/-
# Attention's forward pass and Frank-Wolfe — the cells

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4.1.

The cells `𝒞_i(v)` of `eq: cells` carve the polytope `𝒦` into convex pieces,
one per vertex (`lem: cells`), and when all vertices sit on one level set of
`J` these pieces are exactly the `B`-norm Voronoi cells (`prop: voronoi`).
-/

import Transformer.FrankWolfe.Section1_Models

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d κ : ℕ}

/-- Over a polytope `𝒦 = conv{v_1, …, v_κ}` the defining condition of a cell
need only be tested against the vertices: a linear functional attains its
maximum over a hull at a generator. -/
theorem mem_cell_iff (B : ParamMatrix d) {K : Set (EucSpace d)}
    {v : Idx κ → EucSpace d} (hK : K = convexHull ℝ (Set.range v)) (i : Idx κ)
    (x : EucSpace d) :
    x ∈ cell B K v i ↔
      x ∈ K ∧ ∀ j : Idx κ, inner (𝕜 := ℝ) (B x) (v j) ≤ inner (𝕜 := ℝ) (B x) (v i) := by
  constructor
  · rintro ⟨hxK, hx⟩
    exact ⟨hxK, fun j => hx (v j) (hK ▸ subset_convexHull ℝ _ ⟨j, rfl⟩)⟩
  · rintro ⟨hxK, hx⟩
    refine ⟨hxK, fun y hy => ?_⟩
    have hlin : IsLinearMap ℝ (fun y : EucSpace d => inner (𝕜 := ℝ) (B x) y) :=
      ⟨fun y z => inner_add_right _ _ _, fun c y => real_inner_smul_right _ _ _⟩
    have hsub : Set.range v ⊆ {y : EucSpace d | inner (𝕜 := ℝ) (B x) y
        ≤ inner (𝕜 := ℝ) (B x) (v i)} := by
      rintro _ ⟨j, rfl⟩
      exact hx j
    exact convexHull_min hsub (convex_halfSpace_le hlin _) (hK ▸ hy)

/-- The hypothesis of `mem_cell_iff` is satisfiable: one vertex at the
origin. -/
example : (convexHull ℝ (Set.range (fun _ : Idx 1 => (0 : EucSpace 1))) : Set (EucSpace 1))
    = convexHull ℝ (Set.range (fun _ : Idx 1 => (0 : EucSpace 1))) := rfl

/-- **Lemma (lem: cells), part 1.**  Each cell `𝒞_i(v)` is convex.

Source: arXiv:2508.09628v1, §4.1, `lem: cells`. -/
theorem convex_cell (B : ParamMatrix d) {K : Set (EucSpace d)} (hK : Convex ℝ K)
    (v : Idx κ → EucSpace d) (i : Idx κ) : Convex ℝ (cell B K v i) := by
  rintro x ⟨hxK, hx⟩ y ⟨hyK, hy⟩ a b ha hb hab
  refine ⟨hK hxK hyK ha hb hab, fun z hz => ?_⟩
  have hBl : B (a • x + b • y) = a • B x + b • B y := by
    rw [map_add, map_smul, map_smul]
  have hx' := hx z hz
  have hy' := hy z hz
  rw [hBl]
  simp only [inner_add_left, real_inner_smul_left]
  nlinarith

/-- The hypothesis of `convex_cell` is satisfiable: the whole space is
convex. -/
example : Convex ℝ (Set.univ : Set (EucSpace 1)) := convex_univ

/-- **Lemma (lem: cells), part 2.**  Distinct cells have disjoint interiors:
`int(𝒞_i(v) ∩ 𝒞_j(v)) = ∅` for `i ≠ j`.

Source: arXiv:2508.09628v1, §4.1, `lem: cells`. -/
theorem interior_cell_inter (B : ParamMatrix d) (hB : IsPosDef B)
    {K : Set (EucSpace d)} {v : Idx κ → EucSpace d} (hv : Function.Injective v)
    (hvK : ∀ i, v i ∈ K) {i j : Idx κ} (hij : i ≠ j) :
    interior (cell B K v i ∩ cell B K v j) = ∅ := by
  by_contra hne
  obtain ⟨x, hx⟩ := Set.nonempty_iff_ne_empty.mpr hne
  obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.mp isOpen_interior x hx
  set w : EucSpace d := v i - v j with hw
  have hw0 : w ≠ 0 := sub_ne_zero_of_ne (fun h => hij (hv h))
  have hwpos : 0 < inner (𝕜 := ℝ) (B w) w := hB.2 w hw0
  have hzero : ∀ z ∈ Metric.ball x ε, inner (𝕜 := ℝ) (B z) w = 0 := by
    intro z hz
    obtain ⟨⟨-, hzi⟩, ⟨-, hzj⟩⟩ := interior_subset (hball hz)
    have h1 := hzi (v j) (hvK j)
    have h2 := hzj (v i) (hvK i)
    have : inner (𝕜 := ℝ) (B z) (v i) = inner (𝕜 := ℝ) (B z) (v j) := le_antisymm h2 h1
    rw [hw, inner_sub_right, this, sub_self]
  set δ : ℝ := ε / (2 * ‖w‖) with hδ
  have hnw : 0 < ‖w‖ := norm_pos_iff.mpr hw0
  have hδpos : 0 < δ := by positivity
  have hmem : x + δ • w ∈ Metric.ball x ε := by
    rw [Metric.mem_ball, dist_eq_norm]
    have : ‖x + δ • w - x‖ = δ * ‖w‖ := by
      rw [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_pos hδpos]
    rw [this, hδ]
    field_simp
    linarith
  have hval : inner (𝕜 := ℝ) (B (x + δ • w)) w
      = inner (𝕜 := ℝ) (B x) w + δ * inner (𝕜 := ℝ) (B w) w := by
    rw [map_add, map_smul, inner_add_left, real_inner_smul_left]
  have h0x : inner (𝕜 := ℝ) (B x) w = 0 :=
    hzero x (Metric.mem_ball_self hε)
  have := hzero _ hmem
  rw [hval, h0x, zero_add] at this
  nlinarith

/-- The hypotheses of `interior_cell_inter` are satisfiable: two distinct
vertices of a segment in `ℝ^1`, with `B = I`. -/
example :
    IsPosDef (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    Function.Injective
      (![(0 : EucSpace 1), EuclideanSpace.single (0 : Fin 1) (1 : ℝ)]) ∧
    (∀ i : Idx 2, (![(0 : EucSpace 1), EuclideanSpace.single (0 : Fin 1) (1 : ℝ)]) i ∈
      convexHull ℝ (Set.range
        (![(0 : EucSpace 1), EuclideanSpace.single (0 : Fin 1) (1 : ℝ)]))) ∧
    (0 : Idx 2) ≠ 1 := by
  have hne : (EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ≠ 0 := by
    intro h
    have : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 0 := by rw [h, norm_zero]
    rw [PiLp.norm_single] at this
    norm_num at this
  refine ⟨⟨fun x y => rfl, fun x hx => real_inner_self_pos.mpr hx⟩, ?_,
    fun i => subset_convexHull ℝ _ ⟨i, rfl⟩, by decide⟩
  intro a b hab
  fin_cases a <;> fin_cases b <;> simp_all [eq_comm]

/-- **Lemma (lem: cells), part 3.**  The cells cover the polytope:
`⋃_i 𝒞_i(v) = 𝒦`.

Source: arXiv:2508.09628v1, §4.1, `lem: cells`. -/
theorem iUnion_cell (B : ParamMatrix d) {K : Set (EucSpace d)} {v : Idx κ → EucSpace d}
    (hK : K = convexHull ℝ (Set.range v)) :
    ⋃ i : Idx κ, cell B K v i = K := by
  apply Set.Subset.antisymm
  · exact Set.iUnion_subset fun i x hx => hx.1
  · intro x hxK
    rcases isEmpty_or_nonempty (Idx κ) with hκ | hκ
    · rw [hK, Set.range_eq_empty (f := v), convexHull_empty] at hxK
      exact absurd hxK (Set.notMem_empty x)
    · obtain ⟨i, -, hi⟩ := Finset.exists_max_image Finset.univ
        (fun j : Idx κ => inner (𝕜 := ℝ) (B x) (v j)) Finset.univ_nonempty
      exact Set.mem_iUnion.mpr ⟨i, (mem_cell_iff B hK i x).mpr
        ⟨hxK, fun j => hi j (Finset.mem_univ j)⟩⟩

/-- The hypothesis of `iUnion_cell` is satisfiable: a single vertex at the
origin. -/
example : (convexHull ℝ (Set.range (fun _ : Idx 1 => (0 : EucSpace 1))) : Set (EucSpace 1))
    = convexHull ℝ (Set.range (fun _ : Idx 1 => (0 : EucSpace 1))) := rfl

/-- The `B`-norm Voronoi cell of the vertex `v_i`:

  `Vor_B(v_i) = {x : ‖x - v_i‖_B ≤ ‖x - v_j‖_B for all j}`,

written with the squared `B`-norms so that no square root is needed.

Source: arXiv:2508.09628v1, §4.1, `prop: voronoi`. -/
def vorCell (B : ParamMatrix d) (v : Idx κ → EucSpace d) (i : Idx κ) : Set (EucSpace d) :=
  {x | ∀ j : Idx κ, inner (𝕜 := ℝ) (B (x - v i)) (x - v i)
        ≤ inner (𝕜 := ℝ) (B (x - v j)) (x - v j)}

/-- **Proposition (prop: voronoi).**  If all vertices lie on one level set of
`J(x) = ½ ⟨B x, x⟩`, then

  `𝒞_i(v) = Vor_B(v_i) ∩ 𝒦`.

**What the source says and what is changed here.**  Two hypotheses of the
source are dropped, because the proof uses neither: `B ≻ 0` is weakened to
`B` symmetric — the polarization identity is all that the argument needs — and
`c > 0` is dropped outright.  The resulting statement is stronger.

The source's `Vor_B(v_i)` is compared against every `y ∈ 𝒦`; here, as in the
source's own proof, the comparison is against the vertices.  Against every
`y ∈ 𝒦` the claim would be false: `y = x` gives `‖x - v_i‖_B ≤ 0`, so only the
vertices themselves could lie in their cells.

Source: arXiv:2508.09628v1, §4.1, `prop: voronoi`. -/
theorem cell_eq_vorCell_inter (B : ParamMatrix d) (hsym : B.IsSymmetric)
    {K : Set (EucSpace d)} {v : Idx κ → EucSpace d} {c : ℝ}
    (hK : K = convexHull ℝ (Set.range v)) (hc : ∀ i, quadForm B (v i) = c) (i : Idx κ) :
    cell B K v i = vorCell B v i ∩ K := by
  have key : ∀ (x : EucSpace d) (j : Idx κ),
      inner (𝕜 := ℝ) (B (x - v j)) (x - v j)
        = inner (𝕜 := ℝ) (B x) x - 2 * inner (𝕜 := ℝ) (B x) (v j) + 2 * c := by
    intro x j
    have hq : inner (𝕜 := ℝ) (B (v j)) (v j) = 2 * c := by
      have := hc j
      rw [quadForm] at this
      linarith
    have hswap : inner (𝕜 := ℝ) (B (v j)) x = inner (𝕜 := ℝ) (B x) (v j) := by
      have h := hsym (v j) x
      simp only [ContinuousLinearMap.coe_coe] at h
      rw [h, real_inner_comm]
    rw [map_sub, inner_sub_left, inner_sub_right, inner_sub_right, hq, hswap]
    ring
  ext x
  rw [mem_cell_iff B hK i x, Set.mem_inter_iff, and_comm (a := x ∈ vorCell B v i)]
  refine and_congr_right fun _ => ?_
  constructor
  · intro h j
    have := h j
    rw [key x i, key x j]
    linarith
  · intro h j
    have := h j
    rw [key x i, key x j] at this
    linarith

/-- The hypotheses of `cell_eq_vorCell_inter` are satisfiable: `B = I` and a
single vertex at the origin, on the level set `c = 0`. -/
example :
    (ContinuousLinearMap.id ℝ (EucSpace 1)).IsSymmetric ∧
    (convexHull ℝ (Set.range (fun _ : Idx 1 => (0 : EucSpace 1))) : Set (EucSpace 1))
      = convexHull ℝ (Set.range (fun _ : Idx 1 => (0 : EucSpace 1))) ∧
    (∀ i : Idx 1, quadForm (ContinuousLinearMap.id ℝ (EucSpace 1))
      ((fun _ : Idx 1 => (0 : EucSpace 1)) i) = 0) :=
  ⟨fun x y => rfl, rfl, fun i => by simp [quadForm]⟩

end FrankWolfe
end Transformer
