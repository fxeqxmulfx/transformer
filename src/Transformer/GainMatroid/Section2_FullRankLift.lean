module

public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.LinearCombination
public import Transformer.GainMatroid.Section2_ExchangeForms
public import Transformer.GainMatroid.Section4_Switching

/-!
# Full-rank canonical lift matrices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.6: "Choosing one vertex in each component of `G`,
and deleting the rows of `A_L(G, γ)` indexed by these vertices yields a matrix representation of
`L(G, B_γ)` that is of full rank. […] the oriented incidence matrix of `G` is recovered from the
resulting matrix by appending a row equal to the negation of the sum of all rows but `v₀`. […]
Then `A_L^{-v}(G, B_γ)` is a full-rank matrix representing `L(G, B_γ)`. Clearly, for any vertex
`v ∈ V(G)` the matrices `A_L^{-v}(G, γ)` and `A_L(G, γ)` are projectively equivalent, and for any
pair of vertices `u, v ∈ V(G)`, the matrices `A_L^{-u}(G, γ)` and `A_L^{-v}(G, γ)` are projectively
equivalent."

`liftMatrixDel G ψ S` deletes the rows of `S`, and `A_L^{-v}` is the case `S = {v}`. **Full rank
needs an unbalanced cycle**: `fullRank_fails_of_balanced` has a connected balanced gain graph whose
matrix has two rows, three columns and rank one. `linearIndependent_liftMatrixDel` proves that the
rows are independent when `G` has an unbalanced cycle.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {V E F : Type*} [Field F]

/-- `S` consists of one vertex in each component of `G`. -/
def OrientedGraph.IsRootSet (G : OrientedGraph V E) (S : Set V) : Prop :=
  ∀ x, ∃! s, s ∈ S ∧ G.ReachableIn Set.univ x s

/-- A vertex of a connected graph is a root set. -/
theorem OrientedGraph.isRootSet_singleton {G : OrientedGraph V E}
    (hG : ∀ x y, Nonempty (G.Walk x y)) (v : V) : G.IsRootSet {v} := fun x => by
  obtain ⟨p⟩ := hG x v
  exact ⟨v, ⟨rfl, p, fun _ _ => Set.mem_univ _⟩, fun y hy => hy.1⟩

variable [DecidableEq V]

/-- `A_L(G, γ)` with the rows of the vertices in `S` deleted. `A_L^{-v}(G, γ)` is `S = {v}`. -/
def liftMatrixDel (G : OrientedGraph V E) (ψ : E → F) (S : Set V) :
    Matrix {r : Option V // ∀ s ∈ S, r ≠ some s} E F :=
  (liftMatrix G ψ).submatrix Subtype.val id

variable [Fintype V]

/-- The rows of a union `K` of components of `G` sum to zero, so the row of `v ∈ K` is the negation
of the sum of the other rows of `K`. -/
theorem liftMatrix_some_eq_neg_sum_of_closed (G : OrientedGraph V E) (ψ : E → F) {K : Finset V}
    (hK : ∀ e, G.tail e ∈ K ↔ G.head e ∈ K) {v : V} (hv : v ∈ K) :
    liftMatrix G ψ (some v) = -∑ w ∈ K.erase v, liftMatrix G ψ (some w) := by
  funext e
  have h := sum_mul_liftMatrix_some G (fun w => if w ∈ K then (1 : F) else 0) ψ e
  simp only [ite_mul, one_mul, zero_mul, hK e, sub_self, Finset.sum_ite_mem, Finset.univ_inter] at h
  rw [← Finset.add_sum_erase _ _ hv] at h
  simpa only [Pi.neg_apply, Finset.sum_apply] using eq_neg_of_add_eq_zero_left h

/-- **§2.6.** The row of `v` is the negation of the sum of the other vertex rows. -/
theorem liftMatrix_some_eq_neg_sum (G : OrientedGraph V E) (ψ : E → F) (v : V) :
    liftMatrix G ψ (some v) = -∑ w ∈ Finset.univ.erase v, liftMatrix G ψ (some w) :=
  liftMatrix_some_eq_neg_sum_of_closed G ψ (fun _ => by simp) (Finset.mem_univ v)

/-- Deleting the rows of one vertex in each component keeps the row space. -/
theorem rowSpace_liftMatrixDel {G : OrientedGraph V E} (ψ : E → F) {S : Set V}
    (hS : G.IsRootSet S) : rowSpace (liftMatrixDel G ψ S) = rowSpace (liftMatrix G ψ) := by
  classical
  refine le_antisymm (rowSpace_le_of_forall_mem fun r => row_mem_rowSpace (liftMatrix G ψ) r.1)
    (rowSpace_le_of_forall_mem fun r => ?_)
  by_cases hr : ∀ s ∈ S, r ≠ some s
  · exact row_mem_rowSpace (liftMatrixDel G ψ S) ⟨r, hr⟩
  push Not at hr
  obtain ⟨s, hs, rfl⟩ := hr
  let K : Finset V := Finset.univ.filter fun w => G.ReachableIn Set.univ w s
  have hsK : s ∈ K := Finset.mem_filter.2 ⟨Finset.mem_univ s, reachableIn_refl _ s⟩
  have hK (e : E) : G.tail e ∈ K ↔ G.head e ∈ K := by
    simp only [K, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨fun h' => ((reachableIn_refl _ _).step (Set.mem_univ e) (Or.inr ⟨rfl, rfl⟩)).trans h',
      fun h' => ((reachableIn_refl _ _).step (Set.mem_univ e) (Or.inl ⟨rfl, rfl⟩)).trans h'⟩
  change liftMatrix G ψ (some s) ∈ _
  rw [liftMatrix_some_eq_neg_sum_of_closed G ψ hK hsK]
  refine Submodule.neg_mem _ (Submodule.sum_mem _ fun w hw => ?_)
  obtain ⟨hws, hwK⟩ := Finset.mem_erase.1 hw
  have hreach : G.ReachableIn Set.univ w s := (Finset.mem_filter.1 hwK).2
  have hnd : ∀ t ∈ S, some w ≠ some t := fun t ht heq => by
    obtain rfl : w = t := Option.some.inj heq
    exact hws ((hS w).unique ⟨ht, reachableIn_refl _ _⟩ ⟨hs, hreach⟩)
  exact row_mem_rowSpace (liftMatrixDel G ψ S) ⟨some w, hnd⟩

/-- **§2.6.** `A_L(G, γ)` and the matrix without one row per component are projectively
equivalent. -/
theorem projEquiv_liftMatrixDel {G : OrientedGraph V E} (ψ : E → F) {S : Set V}
    (hS : G.IsRootSet S) : ProjEquiv (liftMatrix G ψ) (liftMatrixDel G ψ S) :=
  projEquiv_of_rowSpace_eq (rowSpace_liftMatrixDel ψ hS).symm

/-- **§2.6.** For a connected graph, `A_L^{-u}(G, γ)` and `A_L^{-v}(G, γ)` are projectively
equivalent. -/
theorem projEquiv_liftMatrixDel_pair {G : OrientedGraph V E} (ψ : E → F)
    (hG : ∀ x y, Nonempty (G.Walk x y)) (u v : V) :
    ProjEquiv (liftMatrixDel G ψ {u}) (liftMatrixDel G ψ {v}) :=
  (projEquiv_liftMatrixDel ψ (isRootSet_singleton hG u)).symm.trans
    (projEquiv_liftMatrixDel ψ (isRootSet_singleton hG v))

/-- **§2.6.** The matrix without one row per component represents the same matroid. -/
theorem IsMatrixRepresentation.liftMatrixDel [Finite E] {M : Matroid E} {G : OrientedGraph V E}
    {ψ : E → F} (hM : IsMatrixRepresentation M (liftMatrix G ψ)) {S : Set V}
    (hS : G.IsRootSet S) : IsMatrixRepresentation M (liftMatrixDel G ψ S) :=
  hM.of_projEquiv (projEquiv_liftMatrixDel ψ hS)

/-- **§2.6, corrected.** With an unbalanced cycle, the rows left are independent. -/
theorem linearIndependent_liftMatrixDel {G : OrientedGraph V E} {ψ : E → F} {S : Set V}
    (hS : G.IsRootSet S) {C : Set E} (hC : G.IsCycle C) (hun : C ∉ G.addBalancedCycles ψ) :
    LinearIndependent F (liftMatrixDel G ψ S).row := by
  classical
  let P : Option V → Prop := fun r => ∀ s ∈ S, r ≠ some s
  rw [Fintype.linearIndependent_iff]
  intro g hg
  let g' : Option V → F := Function.extend Subtype.val g (0 : Option V → F)
  have hg'v (i : {r // P r}) : g' i.1 = g i :=
    Subtype.val_injective.extend_apply g (0 : Option V → F) i
  have hg'0 (r : Option V) (hr : ¬P r) : g' r = 0 :=
    Function.extend_apply' g (0 : Option V → F) r fun ⟨i, hi⟩ => hr (hi ▸ i.2)
  have heq (e : E) : g' none * ψ e + (g' (some (G.tail e)) - g' (some (G.head e))) = 0 := by
    have h0 := congrFun hg e
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at h0
    have h1 : ∑ r ∈ Finset.univ.filter P, g' r * liftMatrix G ψ r e = 0 := by
      refine (Finset.sum_subtype (p := P) (F := inferInstance) _ (fun x => by simp [P]) _).trans ?_
      refine Eq.trans (Finset.sum_congr rfl fun i _ => ?_) h0
      rw [hg'v]
      rfl
    rw [Finset.sum_filter_of_ne fun r _ hr =>
      by_contra fun hP => hr (by rw [hg'0 r hP, zero_mul]), Fintype.sum_option,
      sum_mul_liftMatrix_some G (fun w => g' (some w)) ψ e] at h1
    simpa only [liftMatrix_none] using h1
  have hnone : g' none = 0 := by
    by_contra hc
    refine hun (mem_addBalancedCycles_iff.2
      ⟨hC, fun w => -(g' (some w) / g' none), fun e _ => ?_⟩)
    rw [addSwitch_apply]
    field_simp
    linear_combination heq e
  have hstep (e : E) : g' (some (G.tail e)) = g' (some (G.head e)) := by
    have h := heq e
    rw [hnone] at h
    linear_combination h
  have hconst {x y : V} (p : G.Walk x y) : g' (some x) = g' (some y) := by
    induction p with
    | nil => rfl
    | cons e hj _ ih =>
      rcases hj with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · exact (hstep e).trans ih
      · exact (hstep e).symm.trans ih
  intro i
  rw [← hg'v i]
  rcases i with ⟨_ | w, hi⟩
  · exact hnone
  · obtain ⟨s, ⟨hs, p, -⟩, -⟩ := hS w
    exact (hconst p).trans (hg'0 _ fun h => h s hs rfl)

/-- **The full-rank claim of §2.6 fails for a balanced gain graph.** Three parallel links with gain
`0`: the matrix `A_L^{-0}` has two rows and three columns, and its gain row is zero. -/
theorem fullRank_fails_of_balanced :
    ∃ (G : OrientedGraph (Fin 2) (Fin 3)) (ψ : Fin 3 → ℚ) (v : Fin 2),
      (∀ x y, Nonempty (G.Walk x y)) ∧ (∀ C, G.IsCycle C → C ∈ G.addBalancedCycles ψ) ∧
        ¬LinearIndependent ℚ (liftMatrixDel G ψ {v}).row :=
  ⟨pumpkin, 0, 0, pumpkin_connected,
    fun _ hC => mem_addBalancedCycles_iff.2 ⟨hC, 0, fun e _ => by simp [addSwitch_apply]⟩,
    fun h => h.ne_zero ⟨none, by simp⟩ (by funext e; simp [liftMatrixDel, Matrix.row, liftMatrix])⟩

example := liftMatrix_some_eq_neg_sum_of_closed pumpkin (0 : Fin 3 → ℚ) (K := Finset.univ)
  (fun _ => by simp) (Finset.mem_univ 0)
example := liftMatrix_some_eq_neg_sum pumpkin (0 : Fin 3 → ℚ) 0
example := rowSpace_liftMatrixDel (0 : Fin 3 → ℚ) (isRootSet_singleton pumpkin_connected 0)
example := projEquiv_liftMatrixDel (0 : Fin 3 → ℚ) (isRootSet_singleton pumpkin_connected 0)
example := projEquiv_liftMatrixDel_pair (0 : Fin 3 → ℚ) pumpkin_connected 0 1
example := linearIndependent_liftMatrixDel (ψ := (![1, 0, 0] : Fin 3 → ℚ))
  (isRootSet_singleton pumpkin_connected 0) pumpkin_pair_isCycle pumpkin_pair_unbalanced

set_option maxRecDepth 100000 in
/-- Over `𝔽₂` with zero gains, `A_L^{-0}` of three parallel links represents `U_{1,3}`. -/
example : IsMatrixRepresentation (finsetUniform (Fin 3) 1)
    (liftMatrixDel pumpkin (0 : Fin 3 → ZMod 2) {0}) := by
  refine IsMatrixRepresentation.liftMatrixDel ⟨rfl, fun I => ?_⟩
    (isRootSet_singleton pumpkin_connected 0)
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (finsetUniform_indep 1 I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

end Transformer.GainMatroid
