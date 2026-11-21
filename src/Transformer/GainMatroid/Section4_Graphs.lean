module

public import Mathlib.Tactic.FinCases
public import Transformer.GainMatroid.Section4_Rows

/-!
# The graphs `2C₃`, `K₄` and the tube

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2 and §4. `twoC3` is a 3-cycle with each edge
doubled, with the labels and orientations of the matrix in the proof of Lemma 4.3:
`e₁, e₂ : v₁ → v₂`, `e₃, e₄ : v₂ → v₃`, `e₅, e₆ : v₃ → v₁`. `tube` is the graph `2C₄''` of the
source's 4-tube figure: `e₁, e₂ : v₁ → v₂`, `e₃ : v₁ → v₃`, `e₄ : v₂ → v₄`, `e₅, e₆ : v₃ → v₄`.
`K4` is the complete graph on four vertices with each edge directed from the smaller vertex to the
larger.

A canonical matrix depends on an orientation that the source chooses freely, so the lemmas of §4
are stated for every reorientation of these models (`IsReorientationOf`). Off each vertex, these
graphs keep a star (`connectedOff_of_star`) and a pair of parallel links or a triangle.
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E Γ : Type*}

namespace OrientedGraph

variable {G H : OrientedGraph V E}

/-- `G` has the ends of `H`, with some edges reversed. -/
def IsReorientationOf (G H : OrientedGraph V E) : Prop := ∀ e, G.Joins e (H.tail e) (H.head e)

theorem IsReorientationOf.refl (G : OrientedGraph V E) : G.IsReorientationOf G :=
  fun _ => Or.inl ⟨rfl, rfl⟩

theorem IsReorientationOf.joins (h : G.IsReorientationOf H) {e : E} {u v : V}
    (hj : H.Joins e u v) : G.Joins e u v := by
  rcases hj with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · exact h e
  · exact (h e).symm

theorem IsReorientationOf.loopless (h : G.IsReorientationOf H) (hH : H.Loopless) : G.Loopless := by
  intro e
  change G.tail e ≠ G.head e
  rcases h e with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · rw [h₁, h₂]
    exact hH e
  · rw [h₁, h₂]
    exact (hH e).symm

/-- If off each vertex some center is adjacent to every other vertex, then the graph stays
connected off each vertex. -/
theorem connectedOff_of_star (h : ∀ w, ∃ c, c ≠ w ∧ ∀ x, x ≠ w → x = c ∨ ∃ e, G.Joins e x c) :
    G.ConnectedOff := by
  intro w x y _ _ hxw hyw
  obtain ⟨c, hcw, hs⟩ := h w
  have key : ∀ z, z ≠ w → ∃ p : G.Walk z c, w ∉ p.support := by
    intro z hz
    rcases hs z hz with hzc | ⟨e, he⟩
    · subst hzc
      exact ⟨.nil _, by simp [Walk.support, hz.symm]⟩
    · exact ⟨.cons e he (.nil c), by simp [Walk.support, hz.symm, hcw.symm]⟩
  obtain ⟨p, hp⟩ := key x hxw
  obtain ⟨q, hq⟩ := key y hyw
  refine ⟨p.append q.reverse, ?_⟩
  rw [Walk.support_append, List.mem_append, not_or]
  refine ⟨hp, fun hmem => hq ?_⟩
  have := List.mem_of_mem_tail hmem
  rwa [Walk.support_reverse, List.mem_reverse] at this

/-- A reorientation of a graph with a star off each vertex stays connected off each vertex. -/
theorem IsReorientationOf.connectedOff (h : G.IsReorientationOf H)
    (hs : ∀ w, ∃ c, c ≠ w ∧ ∀ x, x ≠ w → x = c ∨ ∃ e, H.Joins e x c) : G.ConnectedOff :=
  connectedOff_of_star fun w => by
    obtain ⟨c, hc, hx⟩ := hs w
    refine ⟨c, hc, fun x hxw => ?_⟩
    rcases hx x hxw with hxc | ⟨e, he⟩
    · exact Or.inl hxc
    · exact Or.inr ⟨e, h.joins he⟩

/-- The graph `2C₃`, with the labels and orientations of the matrix in the proof of Lemma 4.3. -/
def twoC3 : OrientedGraph (Fin 3) (Fin 6) := ⟨![0, 0, 1, 1, 2, 2], ![1, 1, 2, 2, 0, 0]⟩

/-- The complete graph `K₄`, with each edge directed from the smaller vertex to the larger. -/
def K4 : OrientedGraph (Fin 4) (Fin 6) := ⟨![0, 0, 0, 1, 1, 2], ![1, 2, 3, 2, 3, 3]⟩

/-- The tube `2C₄''`, with the labels and orientations of the source's 4-tube figure. -/
def tube : OrientedGraph (Fin 4) (Fin 6) := ⟨![0, 0, 0, 1, 2, 2], ![1, 1, 2, 3, 3, 3]⟩

theorem twoC3_loopless : twoC3.Loopless := by
  intro e
  fin_cases e <;> simp [twoC3]

theorem K4_loopless : K4.Loopless := by
  intro e
  fin_cases e <;> simp [K4]

theorem tube_loopless : tube.Loopless := by
  intro e
  fin_cases e <;> simp [tube]

theorem twoC3_star : ∀ w, ∃ c, c ≠ w ∧ ∀ x, x ≠ w → x = c ∨ ∃ e, twoC3.Joins e x c := by
  simp only [Joins, twoC3]
  decide

theorem K4_star : ∀ w, ∃ c, c ≠ w ∧ ∀ x, x ≠ w → x = c ∨ ∃ e, K4.Joins e x c := by
  simp only [Joins, K4]
  decide

theorem tube_star : ∀ w, ∃ c, c ≠ w ∧ ∀ x, x ≠ w → x = c ∨ ∃ e, tube.Joins e x c := by
  simp only [Joins, tube]
  decide

/-- With no balanced 2-cycle, no vertex of a biased `2C₃` balances the gain function. -/
@[to_additive hasNoAddBalancingVertex_of_twoC3
  /-- With no balanced 2-cycle, no vertex of a biased `2C₃` balances the additive gain function. -/]
theorem hasNoBalancingVertex_of_twoC3 [Group Γ] {G : OrientedGraph (Fin 3) (Fin 6)}
    (hG : G.IsReorientationOf twoC3) {γ : Fin 6 → Γ}
    (h : ∀ C ∈ G.balancedCycles γ, C.encard ≠ 2) : G.HasNoBalancingVertex γ := by
  have h23 : G.Joins 2 1 2 := hG 2
  have h33 : G.Joins 3 1 2 := hG 3
  have h45 : G.Joins 4 2 0 := hG 4
  have h55 : G.Joins 5 2 0 := hG 5
  have h01 : G.Joins 0 0 1 := hG 0
  have h11 : G.Joins 1 0 1 := hG 1
  intro w
  fin_cases w
  · exact exists_off_of_pair h h23 h33 (by decide) (by decide) (by decide) (by decide)
  · exact exists_off_of_pair h h45 h55 (by decide) (by decide) (by decide) (by decide)
  · exact exists_off_of_pair h h01 h11 (by decide) (by decide) (by decide) (by decide)

/-- With no balanced 2-cycle, no vertex of a biased tube balances the gain function. -/
@[to_additive hasNoAddBalancingVertex_of_tube
  /-- With no balanced 2-cycle, no vertex of a biased tube balances the additive gain function. -/]
theorem hasNoBalancingVertex_of_tube [Group Γ] {G : OrientedGraph (Fin 4) (Fin 6)}
    (hG : G.IsReorientationOf tube) {γ : Fin 6 → Γ}
    (h : ∀ C ∈ G.balancedCycles γ, C.encard ≠ 2) : G.HasNoBalancingVertex γ := by
  have h4 : G.Joins 4 2 3 := hG 4
  have h5 : G.Joins 5 2 3 := hG 5
  have h0 : G.Joins 0 0 1 := hG 0
  have h1 : G.Joins 1 0 1 := hG 1
  intro w
  fin_cases w
  · exact exists_off_of_pair h h4 h5 (by decide) (by decide) (by decide) (by decide)
  · exact exists_off_of_pair h h4 h5 (by decide) (by decide) (by decide) (by decide)
  · exact exists_off_of_pair h h0 h1 (by decide) (by decide) (by decide) (by decide)
  · exact exists_off_of_pair h h0 h1 (by decide) (by decide) (by decide) (by decide)

/-- With no balanced triangle, no vertex of a biased `K₄` balances the gain function. -/
@[to_additive hasNoAddBalancingVertex_of_K4
  /-- With no balanced triangle, no vertex of a biased `K₄` balances the additive gain function. -/]
theorem hasNoBalancingVertex_of_K4 [Group Γ] {G : OrientedGraph (Fin 4) (Fin 6)}
    (hG : G.IsReorientationOf K4) {γ : Fin 6 → Γ}
    (h : ∀ C ∈ G.balancedCycles γ, C.encard ≠ 3) : G.HasNoBalancingVertex γ := by
  have j0 : G.Joins 0 0 1 := hG 0
  have j1 : G.Joins 1 0 2 := hG 1
  have j2 : G.Joins 2 0 3 := hG 2
  have j3 : G.Joins 3 1 2 := hG 3
  have j4 : G.Joins 4 1 3 := hG 4
  have j5 : G.Joins 5 2 3 := hG 5
  intro w
  fin_cases w
  · exact exists_off_of_triangle h j3 j5 j4.symm (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
  · exact exists_off_of_triangle h j1 j5 j2.symm (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
  · exact exists_off_of_triangle h j0 j4 j2.symm (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
  · exact exists_off_of_triangle h j0 j3 j1.symm (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)

end OrientedGraph

end Transformer.GainMatroid
