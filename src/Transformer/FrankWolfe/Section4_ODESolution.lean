/-
# Attention's forward pass and Frank-Wolfe — the solution of the singular ODE

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, proof of `thm: ode`, Part 1.

Under `GenericPolytope`, with `σ(i)` the cell of `x_i^0`:

* every vertex is a particle, assigned to itself (`exists_particle`), so the
  vertices never move;
* `x_i(t) = e^{-t} x_i^0 + (1 - e^{-t}) v_{σ(i)}` lies in the cell of
  `v_{σ(i)}` and in no other (`hardmaxSol_mem_cell_iff`), so `v_{σ(i)}` is the
  only maximizer of `⟨B x_i(t), ·⟩` over `𝒦`, strictly ahead of every other
  point of `𝒦` (`inner_hardmaxSol_lt`);
* hence the curve solves `eq: hardmax.ode` (`isHardmaxODESolution_hardmaxSol`).

The source obtains the solution as the limit of the Euler scheme of
`thm: exp.fast.polytope`; here it is checked directly.

Source: arXiv:2508.09628v1, `sec: proof.thm.ode`, Part 1.
-/

import Transformer.FrankWolfe.Section4_ODE
import Transformer.FrankWolfe.Section4_Faces

open scoped BigOperators
open Real Set

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ} {B : ParamMatrix d} {X₀ : Idx n → EucSpace d} {v : Idx κ → EucSpace d}
  {σ : Idx n → Idx κ}

namespace GenericPolytope

/-- The vertices lie in the polytope. -/
theorem vertex_mem (hgen : GenericPolytope B v X₀) (j : Idx κ) : v j ∈ configHull X₀ :=
  (hgen.isVertexList.2 ▸ mem_range_self j : v j ∈ (configHull X₀).extremePoints ℝ).1

/-- A vertex lies in its own cell and in no other. -/
theorem mem_cell_vertex_iff (hgen : GenericPolytope B v X₀) (k j : Idx κ) :
    v k ∈ cell B (configHull X₀) v j ↔ j = k :=
  ⟨fun h => by_contra fun hjk => (hgen.ownCell k).2 j hjk h, fun h => h ▸ (hgen.ownCell k).1⟩

/-- A particle lies in the cell `σ` assigns it and in no other. -/
theorem mem_cell_iff (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) (i : Idx n) (j : Idx κ) :
    X₀ i ∈ cell B (configHull X₀) v j ↔ j = σ i := by
  obtain ⟨k, -, hk⟩ := hgen.uniqueCell i
  exact ⟨fun h => (hk j h).trans (hk _ (hσ i)).symm, fun h => h ▸ hσ i⟩

/-- Every vertex is a particle, which `σ` assigns to that vertex. -/
theorem exists_particle (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) (k : Idx κ) :
    ∃ p, X₀ p = v k ∧ σ p = k := by
  obtain ⟨p, hp⟩ := extremePoints_convexHull_subset (hgen.isVertexList.2 ▸ mem_range_self k :
    v k ∈ (configHull X₀).extremePoints ℝ)
  refine ⟨p, hp, (hgen.mem_cell_vertex_iff k (σ p)).1 ?_⟩
  rw [← hp]
  exact hσ p

end GenericPolytope

/-- A particle resting at the vertex it is assigned to never moves. -/
theorem hardmaxSol_of_vertex {p : Idx n} {k : Idx κ} (hp : X₀ p = v k) (hpσ : σ p = k) (t : ℝ) :
    hardmaxSol X₀ (v ∘ σ) t p = v k := by
  rw [hardmaxSol_of_eq (w := v ∘ σ) (by rw [Function.comp_apply, hpσ, hp]) t,
    Function.comp_apply, hpσ]

/-- The hypotheses of `hardmaxSol_of_vertex` are satisfiable: one particle at
the one vertex. -/
example : (fun _ : Idx 1 => (0 : EucSpace 1)) 0 = (fun _ : Idx 1 => (0 : EucSpace 1)) 0 ∧
    (fun _ : Idx 1 => (0 : Idx 1)) 0 = 0 :=
  ⟨rfl, rfl⟩

/-- For `t ≥ 0` the particles stay in `𝒦`. -/
theorem hardmaxSol_mem_configHull (hgen : GenericPolytope B v X₀) {t : ℝ} (ht : 0 ≤ t)
    (i : Idx n) : hardmaxSol X₀ (v ∘ σ) t i ∈ configHull X₀ :=
  hardmaxSol_mem (convex_convexHull ℝ _) (subset_convexHull ℝ _ (mem_range_self i))
    (hgen.vertex_mem _) ht

/-- For `t ≥ 0` the particle `i` lies in the cell of `v_{σ(i)}` and in no
other. -/
theorem hardmaxSol_mem_cell_iff (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) {t : ℝ} (ht : 0 ≤ t) (i : Idx n)
    (j : Idx κ) : hardmaxSol X₀ (v ∘ σ) t i ∈ cell B (configHull X₀) v j ↔ j = σ i :=
  mem_cell_segment_iff B (convex_convexHull ℝ _) hgen.vertex_mem (hgen.mem_cell_iff hσ i)
    (hgen.mem_cell_vertex_iff (σ i)) (exp_neg_mem_Ioc ht).1.le (exp_neg_mem_Ioc ht).2 j

/-- **The strict gap.**  For `t ≥ 0`, every point of `𝒦` other than `v_{σ(i)}`
scores strictly less than `v_{σ(i)}` against `B x_i(t)`: `v_{σ(i)}` is the only
maximizer (`eq_of_isMaximizerOn`). -/
theorem inner_hardmaxSol_lt (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) {t : ℝ} (ht : 0 ≤ t) (i : Idx n)
    {z : EucSpace d} (hz : z ∈ configHull X₀) (hne : z ≠ v (σ i)) :
    inner (𝕜 := ℝ) (B (hardmaxSol X₀ (v ∘ σ) t i)) z <
      inner (𝕜 := ℝ) (B (hardmaxSol X₀ (v ∘ σ) t i)) (v (σ i)) := by
  have hcell := hardmaxSol_mem_cell_iff hgen hσ ht i
  have hle := ((hcell (σ i)).2 rfl).2
  refine lt_of_le_of_ne (hle z hz) fun heq => hne ?_
  exact eq_of_isMaximizerOn ((finite_range X₀).isCompact_convexHull (𝕜 := ℝ))
    (convex_convexHull ℝ _) hgen.isVertexList.2 ⟨hz, fun w hw => (hle w hw).trans heq.ge⟩
    fun j hj => (hcell j).1 ⟨hardmaxSol_mem_configHull hgen ht i, hj⟩

/-- **A particle at a vertex rests there forever.**  If at some `t ≥ 0` the
particle `j` sits at the vertex `v_k`, it started there, is assigned to it, and
so never moves. -/
theorem hardmaxSol_eq_vertex (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) {t : ℝ} (ht : 0 ≤ t) {j : Idx n}
    {k : Idx κ} (h : hardmaxSol X₀ (v ∘ σ) t j = v k) (s : ℝ) :
    hardmaxSol X₀ (v ∘ σ) s j = v k := by
  have hext : hardmaxSol X₀ (v ∘ σ) t j ∈ (configHull X₀).extremePoints ℝ := by
    rw [h]
    exact hgen.isVertexList.2 ▸ mem_range_self k
  have hX : X₀ j = v k := (eq_of_hardmaxSol_mem_extremePoints
    (subset_convexHull ℝ _ (mem_range_self j)) (hgen.vertex_mem _) ht hext).trans h
  refine hardmaxSol_of_vertex hX ((hgen.mem_cell_vertex_iff k (σ j)).1 ?_) s
  rw [← hX]
  exact hσ j

/-- **`thm: ode`, Part 1 — the curve solves `eq: hardmax.ode`.**  On every
interval `[0, T]`, `x_i(t) = e^{-t} x_i^0 + (1 - e^{-t}) v_{σ(i)}` moves towards
the particle resting at `v_{σ(i)}`, which maximizes `⟨B x_i(t), ·⟩`.

Source: arXiv:2508.09628v1, `sec: proof.thm.ode`, Part 1. -/
theorem isHardmaxODESolution_hardmaxSol (hgen : GenericPolytope B v X₀)
    (hσ : ∀ i, X₀ i ∈ cell B (configHull X₀) v (σ i)) (T : ℝ) :
    IsHardmaxODESolution B T (hardmaxSol X₀ (v ∘ σ)) := by
  refine ⟨(continuous_hardmaxSol _ _).continuousOn, fun t ht i => ⟨v (σ i), ?_, ?_,
    hasDerivAt_hardmaxSol X₀ (v ∘ σ) i t⟩⟩
  · obtain ⟨p, hp, hpσ⟩ := hgen.exists_particle hσ (σ i)
    exact ⟨p, hardmaxSol_of_vertex hp hpσ t⟩
  · rintro _ ⟨j, rfl⟩
    exact ((hardmaxSol_mem_cell_iff hgen hσ ht.1.le i (σ i)).2 rfl).2 _
      (hardmaxSol_mem_configHull hgen ht.1.le j)

/-- The hypotheses of the theorems of this module are satisfiable together:
`B = I` on `ℝ^1`, two particles at the vertices `±e` of `𝒦 = [-e, e]`, each in
its own cell (`σ = id`), `t = 0`, the point `z = e` against the particle at
`-e`, and that particle resting at its vertex. -/
example : ∃ (X₀ : Idx 2 → EucSpace 1) (σ : Idx 2 → Idx 2) (z : EucSpace 1),
    GenericPolytope (ContinuousLinearMap.id ℝ (EucSpace 1)) X₀ X₀ ∧
    (∀ i, X₀ i ∈ cell (ContinuousLinearMap.id ℝ (EucSpace 1)) (configHull X₀) X₀ (σ i)) ∧
    (0 : ℝ) ≤ 0 ∧ z ∈ configHull X₀ ∧ z ≠ X₀ (σ 0) ∧ hardmaxSol X₀ (X₀ ∘ σ) 0 0 = X₀ 0 := by
  obtain ⟨e, he⟩ : ∃ e : EucSpace 1, ‖e‖ = 1 := ⟨EuclideanSpace.single 0 1, by simp⟩
  set v : Idx 2 → EucSpace 1 := ![-e, e] with hv
  have hvv : ∀ a b, inner (𝕜 := ℝ) (v a) (v b) = if a = b then 1 else -1 := by
    intro a b
    fin_cases a <;> fin_cases b <;> simp [v, he]
  have hinj : Function.Injective v := fun a b hab => by_contra fun h => by
    have := hvv a b
    rw [ite_eq_right h, hab, hvv b b, ite_eq_left rfl] at this
    norm_num at this
  have hK : configHull v = segment ℝ (v 0) (v 1) := by
    rw [configHull, hv, Matrix.range_cons_cons_empty, convexHull_pair]
    rfl
  -- each vertex is the only maximizer of its own score, hence exposed
  have hext : range v = (configHull v).extremePoints ℝ := by
    refine Subset.antisymm ?_ extremePoints_convexHull_subset
    rintro _ ⟨j, rfl⟩
    refine exposedPoints_subset_extremePoints ⟨subset_convexHull ℝ _ (mem_range_self j),
      innerSL ℝ (v j), ?_⟩
    rw [hK]
    rintro _ ⟨a, b, ha, hb, hab, rfl⟩
    simp only [innerSL_apply_apply, inner_add_right, inner_smul_right, hvv]
    fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, ↓reduceIte, mul_one, zero_ne_one, mul_neg,
        add_neg_le_iff_le_add, le_add_neg_iff_add_le]
      refine ⟨by linarith, fun h => ?_⟩
      obtain rfl : b = 0 := by linarith
      obtain rfl : a = 1 := by linarith
      simp
    · simp only [Fin.mk_one, Fin.isValue, one_ne_zero, ↓reduceIte, mul_neg, mul_one,
        neg_add_le_iff_le_add, le_neg_add_iff_add_le]
      refine ⟨by linarith, fun h => ?_⟩
      obtain rfl : a = 0 := by linarith
      obtain rfl : b = 1 := by linarith
      simp
  have hcell : ∀ a b, v a ∈ cell (ContinuousLinearMap.id ℝ (EucSpace 1)) (configHull v) v b ↔
      b = a := by
    intro a b
    rw [mem_cell_iff (K := configHull v) _ rfl]
    simp only [ContinuousLinearMap.id_apply, hvv]
    constructor
    · rintro ⟨-, h⟩
      by_contra hba
      have := h a
      rw [ite_eq_left rfl, ite_eq_right (Ne.symm hba)] at this
      norm_num at this
    · rintro rfl
      exact ⟨subset_convexHull ℝ _ (mem_range_self _),
        fun j => by rw [ite_eq_left rfl]; split_ifs <;> norm_num⟩
  have hgen : GenericPolytope (ContinuousLinearMap.id ℝ (EucSpace 1)) v v :=
    ⟨⟨hinj, hext⟩, fun j => ⟨(hcell j j).2 rfl, fun i hij h => hij ((hcell j i).1 h)⟩,
      fun i => ⟨i, (hcell i i).2 rfl, fun j hj => (hcell i j).1 hj⟩⟩
  exact ⟨v, id, v 1, hgen, fun i => (hcell i i).2 rfl, le_rfl,
    subset_convexHull ℝ _ (mem_range_self 1), fun h => absurd (hinj h) (by decide),
    hardmaxSol_zero _ _ ▸ rfl⟩

end FrankWolfe
end Transformer
