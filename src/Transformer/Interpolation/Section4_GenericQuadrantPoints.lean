module

public import Transformer.Interpolation.Section4_QuadrantTransfers
public import Mathlib.Topology.Algebra.Module.FiniteDimension
public import Mathlib.Topology.Baire.Lemmas
public import Mathlib.LinearAlgebra.Dimension.Constructions

/-!
# Positive unit points avoiding finitely many original planes

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
the finite isolation construction. A proper real linear subspace has
empty interior. Its complement is dense and, in the original finite
Euclidean dimension, open. A finite intersection of these complements
therefore meets a small ball around the all-ones vector.

Every vector in that ball has strictly positive coordinates. Unit
normalization preserves those coordinates and exclusion from each
linear subspace. In dimension at least three every two-vector span
is proper, so finitely many prescribed planes can all be avoided by
one actual positive-quadrant unit point. This gives intermediate point
choices for the original protected atomic transfer construction.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ} {ι : Type*}

/-- The complement of a proper original real linear subspace is dense.
Source auxiliary: arXiv:2411.04551v3, §1 and §3.2, generic intermediate
point choices in the protected interpolation construction. -/
theorem dense_compl_proper_subspace (P : Submodule ℝ (EucSpace d)) (hP : P ≠ ⊤) :
    Dense (P : Set (EucSpace d))ᶜ := by
  apply interior_eq_empty_iff_dense_compl.mp
  by_contra hne
  exact hP (P.eq_top_of_nonempty_interior' (nonempty_iff_ne_empty.mpr hne))

/-- Every finite family of proper original subspaces can be avoided
by a unit point strictly inside the original positive quadrant.
Source auxiliary: arXiv:2411.04551v3, §1 and §3.2, finite point matching. -/
theorem exists_quadrant_unit_avoiding_subspaces [Fintype ι] (hd : 1 ≤ d)
    (P : ι → Submodule ℝ (EucSpace d)) (hP : ∀ i, P i ≠ ⊤) :
    ∃ w : SSphere d, w ∈ positiveQuadrant d ∧ ∀ i, (w : EucSpace d) ∉ P i := by
  have hD : Dense (⋂ i, (P i : Set (EucSpace d))ᶜ) :=
    dense_iInter_of_isOpen (fun i => (P i).closed_of_finiteDimensional.isOpen_compl)
      (fun i => dense_compl_proper_subspace (P i) (hP i))
  obtain ⟨v, hvball, hvoutside⟩ := hD.inter_open_nonempty
    (Metric.ball (onesBias d) (1 / 2)) Metric.isOpen_ball (Metric.nonempty_ball.mpr (by norm_num))
  have hvnorm : ‖v - onesBias d‖ < (1 / 2 : ℝ) := by
    simpa only [Metric.mem_ball, dist_eq_norm] using hvball
  have hcoord (i : Fin d) : 0 < v i := by
    have hi := (PiLp.norm_apply_le (v - onesBias d) i).trans_lt hvnorm
    change ‖v i - 1‖ < (1 / 2 : ℝ) at hi
    rw [Real.norm_eq_abs] at hi
    linarith [neg_abs_le (v i - 1)]
  have hv : v ≠ 0 := by
    intro he
    have hi := hcoord ⟨0, by omega⟩
    rw [he] at hi
    exact (lt_irrefl 0) hi
  let w : SSphere d := ⟨‖v‖⁻¹ • v, mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm hv)⟩
  refine ⟨w, ?_, ?_⟩
  · intro i
    change 0 < ‖v‖⁻¹ * v i
    exact mul_pos (inv_pos.mpr (norm_pos_iff.mpr hv)) (hcoord i)
  · intro i hwi
    have hvi : v ∉ P i := mem_iInter.mp hvoutside i
    apply hvi
    have hh := (P i).smul_mem ‖v‖ hwi
    change ‖v‖ • (‖v‖⁻¹ • v) ∈ P i at hh
    rw [smul_smul, mul_inv_cancel₀ (norm_ne_zero_iff.mpr hv), one_smul] at hh
    exact hh

/-- Every original two-vector plane is proper in ambient dimension
at least three. Source auxiliary: arXiv:2411.04551v3, §1,
the dimension used by the protected point matching construction. -/
theorem span_pair_ne_top_of_three_le (hd : 3 ≤ d) (p q : EucSpace d) :
    Submodule.span ℝ ({p, q} : Set (EucSpace d)) ≠ ⊤ := by
  classical
  have hcard : ({p, q} : Set (EucSpace d)).toFinset.card ≤ 2 := by
    rw [Set.toFinset_insert, Set.toFinset_singleton]
    simpa only [Finset.card_singleton] using Finset.card_insert_le p ({q} : Finset (EucSpace d))
  have hdim : Module.finrank ℝ (Submodule.span ℝ ({p, q} : Set (EucSpace d))) ≤ 2 :=
    (finrank_span_le_card (R := ℝ) ({p, q} : Set (EucSpace d))).trans hcard
  intro htop
  rw [htop, finrank_top, finrank_euclideanSpace_fin] at hdim
  omega

/-- One genuine positive unit point avoids every prescribed original
plane, with no nonempty-family restriction. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, intermediate point choices. -/
theorem exists_quadrant_point_avoiding_planes [Fintype ι] (hd : 3 ≤ d)
    (p q : ι → EucSpace d) :
    ∃ w : SSphere d, w ∈ positiveQuadrant d ∧
      ∀ i, ∀ α β : ℝ, (w : EucSpace d) ≠ α • p i + β • q i := by
  obtain ⟨w, hw, havoid⟩ := exists_quadrant_unit_avoiding_subspaces (by omega : 1 ≤ d)
    (fun i => Submodule.span ℝ ({p i, q i} : Set (EucSpace d)))
    (fun i => span_pair_ne_top_of_three_le hd (p i) (q i))
  refine ⟨w, hw, fun i α β he => havoid i ?_⟩
  exact Submodule.mem_span_pair.mpr ⟨α, β, he.symm⟩

/-- Excluding the prospective target from the active-other plane
ensures that the other point is outside the active-target plane.
Source auxiliary: arXiv:2411.04551v3, §1 and §3.2,
choosing a protected transfer endpoint. -/
theorem plane_exclusion_reversed (p q m : EucSpace d)
    (hm : ∀ α : ℝ, m ≠ α • p) (hq : ∀ α β : ℝ, q ≠ α • p + β • m) :
    ∀ α β : ℝ, m ≠ α • p + β • q := by
  intro α β he
  by_cases hβ : β = 0
  · rw [hβ, zero_smul, add_zero] at he
    exact hm α he
  have hh : β⁻¹ • m = (β⁻¹ * α) • p + q := by
    rw [he, smul_add, smul_smul, smul_smul, inv_mul_cancel₀ hβ, one_smul]
  apply hq (-(β⁻¹ * α)) β⁻¹
  rw [hh, neg_smul]
  abel

/-- Distinct positive original unit points are non-collinear over
the reals. Source auxiliary: arXiv:2411.04551v3, §3.2,
the unit directions used by protected point matching. -/
theorem quadrant_unit_noncollinear (i : Fin d) (p m : SSphere d)
    (hp : p ∈ positiveQuadrant d) (hm : m ∈ positiveQuadrant d) (hne : m ≠ p) :
    ∀ α : ℝ, (m : EucSpace d) ≠ α • (p : EucSpace d) := by
  intro α he
  have hcoord := congrArg (fun v : EucSpace d => v i) he
  change (m : EucSpace d) i = α * (p : EucSpace d) i at hcoord
  have hp' : 0 < (p : EucSpace d) i := hp i
  have hm' : 0 < (m : EucSpace d) i := hm i
  have hα : 0 < α := by nlinarith
  have hn : 1 = α := by
    calc
      1 = ‖(m : EucSpace d)‖ := (mem_sphere_zero_iff_norm.mp m.2).symm
      _ = ‖α • (p : EucSpace d)‖ := congrArg norm he
      _ = α := by rw [norm_smul, Real.norm_of_nonneg hα.le, mem_sphere_zero_iff_norm.mp p.2, mul_one]
  rw [← hn, one_smul] at he
  exact hne (Subtype.ext he)

/-- Genuine coordinate planes in dimension three meet every properness
and dimension hypothesis; distinct coordinate vectors also realize the
reverse-plane exclusion assumptions. -/
example : ∃ (P : Fin 1 → Submodule ℝ (EucSpace 3)) (p q m : EucSpace 3),
    1 ≤ 3 ∧ 3 ≤ 3 ∧ (∀ i, P i ≠ ⊤) ∧
    (∀ α : ℝ, m ≠ α • p) ∧ (∀ α β : ℝ, q ≠ α • p + β • m) := by
  let p : EucSpace 3 := PiLp.single 2 (0 : Fin 3) 1
  let q : EucSpace 3 := PiLp.single 2 (1 : Fin 3) 1
  let m : EucSpace 3 := PiLp.single 2 (2 : Fin 3) 1
  refine ⟨(fun _ => Submodule.span ℝ ({p, q} : Set (EucSpace 3))), p, q, m,
    by decide, le_rfl, fun _ => span_pair_ne_top_of_three_le le_rfl p q, ?_, ?_⟩
  · intro α he
    have hh := congrArg (fun v : EucSpace 3 => v (2 : Fin 3)) he
    norm_num [m, p] at hh
  · intro α β he
    have hh := congrArg (fun v : EucSpace 3 => v (1 : Fin 3)) he
    norm_num [q, p, m] at hh

/-- Genuine distinct positive unit points realize the non-collinearity
hypotheses with all coordinates strictly positive. -/
example : ∃ p m : SSphere 3, p ∈ positiveQuadrant 3 ∧ m ∈ positiveQuadrant 3 ∧ m ≠ p := by
  exact ⟨quadrantThreePoint 0, quadrantThreePoint 1,
    quadrantThreePoint_mem 0, quadrantThreePoint_mem 1,
    fun he => (by decide : (1 : Fin 3) ≠ 0) (quadrantThreePoint_injective he)⟩

end Transformer.Interpolation
