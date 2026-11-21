module

public import Mathlib.Tactic.FieldSimp
public import Transformer.GainMatroid.Section1_ThreeVertices

/-!
# Columns of loops, and cycles on two vertices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4.1. The frame or lift column of a loop with
non-identity gain is a nonzero multiple of a unit vector, whatever the gain. So gain functions that
differ only on such loops have projectively equivalent canonical matrices. This is the step
"up to scaling columns `e₁` and `e₂`" in the proofs of Lemmas 4.11 and 4.12.

A cycle on two vertices has one or two edges, so a gain function realizes the contrabalanced biased
graph there as soon as its loops and its pairs of parallel links are unbalanced. The graphs `U₂` and
`U₃` of Figure `U24` are defined here, with the labels and orientations of that figure.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {R V E F Γ : Type*} [Field F]

/-- Matrices whose columns are nonzero multiples of one another are projectively equivalent. -/
theorem projEquiv_of_forall_column {A B : Matrix R E F}
    (h : ∀ e, ∃ c : Fˣ, ∀ r, A r e * c = B r e) : ProjEquiv A B := by
  choose s hs using h
  exact ⟨s, congrArg rowSpace (Matrix.ext fun r e => hs e r)⟩

variable [DecidableEq V]

/-- Gain functions that agree off loops with non-identity gains have projectively equivalent frame
matrices. -/
theorem frame_projEquiv_of_eq_off_loops {G : OrientedGraph V E} {φ ψ : E → Fˣ}
    (h : ∀ e, (G.tail e = G.head e ∧ φ e ≠ 1 ∧ ψ e ≠ 1) ∨ φ e = ψ e) :
    ProjEquiv (frameMatrix G φ) (frameMatrix G ψ) := by
  refine projEquiv_of_forall_column fun e => ?_
  rcases h e with ⟨hloop, h₁, h₂⟩ | heq
  · have h₁' : (1 : F) - φ e ≠ 0 := sub_ne_zero.2 fun hc => h₁ (Units.ext hc.symm)
    have h₂' : (1 : F) - ψ e ≠ 0 := sub_ne_zero.2 fun hc => h₂ (Units.ext hc.symm)
    refine ⟨Units.mk0 ((1 - ψ e) / (1 - φ e)) (div_ne_zero h₂' h₁'), fun r => ?_⟩
    simp only [frameMatrix_apply, ← hloop, Units.val_mk0]
    by_cases hr : r = G.tail e
    · simp only [hr, ite_true, mul_one]
      field_simp
    · simp [hr]
  · exact ⟨1, fun r => by simp [frameMatrix_apply, heq]⟩

/-- Gain functions that agree off loops with nonzero gains have projectively equivalent lift
matrices. -/
theorem lift_projEquiv_of_eq_off_loops {G : OrientedGraph V E} {φ ψ : E → F}
    (h : ∀ e, (G.tail e = G.head e ∧ φ e ≠ 0 ∧ ψ e ≠ 0) ∨ φ e = ψ e) :
    ProjEquiv (liftMatrix G φ) (liftMatrix G ψ) := by
  refine projEquiv_of_forall_column fun e => ?_
  rcases h e with ⟨hloop, h₁, h₂⟩ | heq
  · refine ⟨Units.mk0 (ψ e / φ e) (div_ne_zero h₂ h₁), fun r => ?_⟩
    cases r with
    | none =>
      simp only [liftMatrix_none, Units.val_mk0]
      field_simp
    | some w => simp [liftMatrix_loop G φ hloop, liftMatrix_loop G ψ hloop]
  · refine ⟨1, fun r => ?_⟩
    cases r with
    | none => simp [liftMatrix_none, heq]
    | some w => simp [liftMatrix_some]

namespace OrientedGraph

/-- A realization of a contrabalanced biased graph gives every loop a non-identity gain. -/
@[to_additive contrabalanced_addRealizes_loop_ne
  /-- A realization of a contrabalanced biased graph gives every loop a nonzero gain. -/]
theorem contrabalanced_realizes_loop_ne [Group Γ] {G : OrientedGraph V E} {γ : E → Γ}
    (h : G.contrabalanced.Realizes γ) {e : E} (he : G.tail e = G.head e) : γ e ≠ 1 := fun h1 => by
  have hmem := (singleton_mem_balancedCycles_iff γ he).2 h1
  rw [show G.balancedCycles γ = ∅ from h] at hmem
  exact Set.notMem_empty _ hmem

omit [DecidableEq V] in
/-- A cycle of one edge is a loop. -/
theorem IsCycle.exists_loop {G : OrientedGraph V E} {C : Set E} (hC : G.IsCycle C)
    (h1 : C.encard = 1) : ∃ e, G.tail e = G.head e ∧ C = {e} := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  rw [hW.encard_edgeSet] at h1
  have hlen : W.edges.length = 1 := by exact_mod_cast h1
  obtain ⟨e, he, hj⟩ := W.exists_edge_of_length_one hlen
  refine ⟨e, ?_, ?_⟩
  · rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> exact h₁.trans h₂.symm
  · ext g
    simp [Walk.edgeSet, he]

section Two

variable {E : Type*} {G : OrientedGraph (Fin 2) E}

/-- A cycle on two vertices has one or two edges. -/
theorem IsCycle.encard_eq_one_or_two {C : Set E} (hC : G.IsCycle C) :
    C.encard = 1 ∨ C.encard = 2 := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  rw [hW.encard_edgeSet]
  have hle : W.edges.length ≤ 2 := by
    have h := hW.support_tail_nodup.length_le_card
    simpa [Walk.length_support] using h
  have hpos : W.edges.length ≠ 0 := fun h => hW.edges_ne_nil (List.length_eq_zero_iff.1 h)
  have h : W.edges.length = 1 ∨ W.edges.length = 2 := by omega
  rcases h with h | h <;> simp [h]

/-- On two vertices, a gain function with unbalanced loops and pairs of parallel links realizes the
contrabalanced biased graph. -/
@[to_additive contrabalanced_addRealizes_of_two
  /-- The additive form of `contrabalanced_realizes_of_two`. -/]
theorem contrabalanced_realizes_of_two [Group Γ] {γ : E → Γ}
    (hloop : ∀ e, G.tail e = G.head e → γ e ≠ 1)
    (hpair : ∀ e f u v, e ≠ f → u ≠ v → G.Joins e u v → G.Joins f u v →
      G.dart γ e u ≠ G.dart γ f u) :
    G.contrabalanced.Realizes γ := by
  refine Set.eq_empty_of_forall_notMem fun C hC => ?_
  have hC' : C ∈ G.balancedCycles γ := hC
  have hcyc : G.IsCycle C := (mem_balancedCycles_iff.1 hC').1
  rcases hcyc.encard_eq_one_or_two with h | h
  · obtain ⟨e, he, rfl⟩ := hcyc.exists_loop h
    exact hloop e he ((singleton_mem_balancedCycles_iff γ he).1 hC')
  · exact encard_ne_two_of_parallel hpair C hC' h

end Two

/-- The graph `U₂` of Figure `U24`: `e₁`, `e₂` are loops at `v₁`, `v₂`, then `e₃ : v₁ → v₂` and
`e₄ : v₂ → v₁`. -/
def U2 : OrientedGraph (Fin 2) (Fin 4) := ⟨![0, 1, 0, 1], ![0, 1, 1, 0]⟩

/-- The graph `U₃` of Figure `U24`: `e₁` is a loop at `v₁`, then `e₂, e₃ : v₁ → v₂` and
`e₄ : v₂ → v₁`. -/
def U3 : OrientedGraph (Fin 2) (Fin 4) := ⟨![0, 0, 0, 1], ![0, 1, 1, 0]⟩

/-- An `𝔽₃`-realization of `𝖴₂`. -/
def U2Gain₁ : Fin 4 → (ZMod 3)ˣ := ![-1, -1, 1, -1]

/-- An `𝔽₃`-realization of `𝖴₂` with the same gain on `e₃e₄` as `U2Gain₁`. -/
def U2Gain₂ : Fin 4 → (ZMod 3)ˣ := ![-1, -1, -1, 1]

/-- An additive `𝔽₃`-realization of `𝖴₃`. -/
def U3Gain₁ : Fin 4 → ZMod 3 := ![1, 0, 1, 1]

/-- Twice `U3Gain₁`, another additive `𝔽₃`-realization of `𝖴₃`. -/
def U3Gain₂ : Fin 4 → ZMod 3 := ![2, 0, 2, 2]

theorem U2Gain₁_realizes : U2.contrabalanced.Realizes U2Gain₁ :=
  contrabalanced_realizes_of_two (by simp only [U2, U2Gain₁]; decide)
    (by simp only [Joins, dart, U2, U2Gain₁]; decide)

theorem U2Gain₂_realizes : U2.contrabalanced.Realizes U2Gain₂ :=
  contrabalanced_realizes_of_two (by simp only [U2, U2Gain₂]; decide)
    (by simp only [Joins, dart, U2, U2Gain₂]; decide)

theorem U3Gain₁_realizes : U3.contrabalanced.AddRealizes U3Gain₁ :=
  contrabalanced_addRealizes_of_two (by simp only [U3, U3Gain₁]; decide)
    (by simp only [Joins, addDart, U3, U3Gain₁]; decide)

theorem U3Gain₂_realizes : U3.contrabalanced.AddRealizes U3Gain₂ :=
  contrabalanced_addRealizes_of_two (by simp only [U3, U3Gain₂]; decide)
    (by simp only [Joins, addDart, U3, U3Gain₂]; decide)

example : U2Gain₁ 0 ≠ 1 := contrabalanced_realizes_loop_ne U2Gain₁_realizes (e := 0) rfl

example : U3Gain₁ 0 ≠ 0 := contrabalanced_addRealizes_loop_ne U3Gain₁_realizes (e := 0) rfl

example : ProjEquiv (frameMatrix U2 U2Gain₁) (frameMatrix U2 U2Gain₁) :=
  projEquiv_of_forall_column fun _ => ⟨1, fun _ => by simp⟩

example : ProjEquiv (frameMatrix U2 U2Gain₁) (frameMatrix U2 U2Gain₁) :=
  frame_projEquiv_of_eq_off_loops fun _ => Or.inr rfl

/-- The loop of `U₃` may change its nonzero gain. -/
example : ProjEquiv (liftMatrix U3 U3Gain₁) (liftMatrix U3 ![2, 0, 1, 1]) :=
  lift_projEquiv_of_eq_off_loops (by simp only [U3, U3Gain₁]; decide)

example : ∃ e, U2.tail e = U2.head e ∧ ({0} : Set (Fin 4)) = {e} :=
  (isCycle_singleton_of_loop (G := U2) (e := 0) rfl).exists_loop (by simp)

example : ({0} : Set (Fin 4)).encard = 1 ∨ ({0} : Set (Fin 4)).encard = 2 :=
  (isCycle_singleton_of_loop (G := U2) (e := 0) rfl).encard_eq_one_or_two

end OrientedGraph

end Transformer.GainMatroid
