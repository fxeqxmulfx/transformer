module

public import Transformer.GainMatroid.Section2_DeletionCycles
public import Transformer.GainMatroid.Section2_SpanningForests

/-!
# Cycles of the frame-type contraction of a joint

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1, contract a joint `e` at `v` of frame type by
"replacing each link `f` incident to `v` with a joint incident to its other endpoint"
(`OrientedGraph.frameContract`). Away from `v` an edge keeps its ends (`frameContract_ends`), a link
from `x` to `v` becomes a loop at `x` (`frameContract_ends_of_joins`), and a loop at `v` stays there
(`frameContract_ends_of_loop`).

Hence a cycle of the contraction is one of its loops or a cycle of `G` avoiding `v`
(`IsCycle.of_frameContract`, `IsCycle.frameContract`), and two vertices are linked in it along `S`
exactly when they are linked in `G` along the edges of `S` away from `v`
(`reachableIn_frameContract_iff`). A walk from a vertex other than `v` stays away from `v`, or first
reaches `v` along an edge from a vertex linked to its start away from `v`
(`Walk.reachableIn_awayFrom_or`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E} {v : V}

/-- The edges of `S` with no end at `v`. -/
def awayFrom (G : OrientedGraph V E) (S : Set E) (v : V) : Set E :=
  {f ∈ S | G.tail f ≠ v ∧ G.head f ≠ v}

theorem mem_awayFrom {S : Set E} {f : E} :
    f ∈ G.awayFrom S v ↔ f ∈ S ∧ G.tail f ≠ v ∧ G.head f ≠ v :=
  Iff.rfl

/-- Away from `v`, the frame-type contraction at `v` keeps the ends of an edge. -/
theorem frameContract_ends {f : E} (ht : G.tail f ≠ v) (hh : G.head f ≠ v) :
    (G.frameContract v).tail f = G.tail f ∧ (G.frameContract v).head f = G.head f := by
  simp [frameContract, ht, hh]

/-- A link from `x ≠ v` to `v` becomes a loop at `x`. -/
theorem frameContract_ends_of_joins {f : E} {x : V} (hf : G.Joins f x v) (hx : x ≠ v) :
    (G.frameContract v).tail f = x ∧ (G.frameContract v).head f = x := by
  rcases hf with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> simp [frameContract, h₁, h₂, hx]

/-- A loop at `v` stays a loop at `v`. -/
theorem frameContract_ends_of_loop {f : E} (ht : G.tail f = v) (hh : G.head f = v) :
    (G.frameContract v).tail f = v ∧ (G.frameContract v).head f = v := by
  simp [frameContract, ht, hh]

/-- An edge at `v` is a loop of the frame-type contraction at `v`. -/
theorem frameContract_tail_eq_head {f : E} (hf : G.tail f = v ∨ G.head f = v) :
    (G.frameContract v).tail f = (G.frameContract v).head f := by
  by_cases ht : G.tail f = v
  · by_cases hh : G.head f = v
    · rw [(frameContract_ends_of_loop ht hh).1, (frameContract_ends_of_loop ht hh).2]
    · have hj : G.Joins f (G.head f) v := Or.inr ⟨ht, rfl⟩
      rw [(frameContract_ends_of_joins hj hh).1, (frameContract_ends_of_joins hj hh).2]
  · have hj : G.Joins f (G.tail f) v := Or.inl ⟨rfl, hf.resolve_left ht⟩
    rw [(frameContract_ends_of_joins hj ht).1, (frameContract_ends_of_joins hj ht).2]

/-- An edge at `v` other than a loop at `v` joins `v` to the vertex carrying it after the
frame-type contraction. -/
theorem joins_frameContract_tail {f : E} (hf : G.tail f = v ∨ G.head f = v)
    (hl : ¬(G.tail f = v ∧ G.head f = v)) :
    G.Joins f ((G.frameContract v).tail f) v ∧ (G.frameContract v).tail f ≠ v := by
  rcases hf with ht | hh
  · have hh : G.head f ≠ v := fun hh => hl ⟨ht, hh⟩
    rw [(frameContract_ends_of_joins (Or.inr ⟨ht, rfl⟩) hh).1]
    exact ⟨Or.inr ⟨ht, rfl⟩, hh⟩
  · have ht : G.tail f ≠ v := fun ht => hl ⟨ht, hh⟩
    rw [(frameContract_ends_of_joins (Or.inl ⟨rfl, hh⟩) ht).1]
    exact ⟨Or.inl ⟨rfl, hh⟩, ht⟩

/-- Reachability along `S` in `H` gives reachability along `T` in `G` when the ends in `H` of each
edge of `S` are linked in `G` along `T`. -/
theorem ReachableIn.of_forall_graph {E' : Type*} {H : OrientedGraph V E'} {S : Set E'}
    {T : Set E} (hS : ∀ f ∈ S, G.ReachableIn T (H.tail f) (H.head f)) {u w : V}
    (h : H.ReachableIn S u w) : G.ReachableIn T u w :=
  ReachableIn.rel (r := G.ReachableIn T) (reachableIn_refl T) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
    (fun f hf => ⟨hS f hf, (hS f hf).symm⟩) h

/-- Two vertices are linked along `S` after the frame-type contraction at `v` exactly when they are
linked along the edges of `S` away from `v`. -/
theorem reachableIn_frameContract_iff {S : Set E} {u w : V} :
    (G.frameContract v).ReachableIn S u w ↔ G.ReachableIn (G.awayFrom S v) u w := by
  constructor
  · refine ReachableIn.of_forall_graph fun f hf => ?_
    by_cases hfv : G.tail f ≠ v ∧ G.head f ≠ v
    · rw [(frameContract_ends hfv.1 hfv.2).1, (frameContract_ends hfv.1 hfv.2).2]
      exact reachableIn_of_mem (mem_awayFrom.2 ⟨hf, hfv⟩)
    · rw [frameContract_tail_eq_head ((not_and_or.1 hfv).imp not_not.1 not_not.1)]
      exact reachableIn_refl _ _
  · refine ReachableIn.of_forall_graph fun f hf => ?_
    obtain ⟨hfS, ht, hh⟩ := mem_awayFrom.1 hf
    rw [← (frameContract_ends ht hh).1, ← (frameContract_ends ht hh).2]
    exact reachableIn_of_mem hfS

/-- A walk along `S` from `u ≠ v` stays away from `v`, or first reaches `v` along an edge `g` of the
walk from a vertex `x ≠ v` linked to `u` along the edges of `S` away from `v`. -/
theorem Walk.reachableIn_awayFrom_or {S : Set E} {u w : V} (p : G.Walk u w)
    (hp : ∀ f ∈ p.edges, f ∈ S) (hu : u ≠ v) :
    (G.ReachableIn (G.awayFrom S v) u w ∧ w ≠ v) ∨
      ∃ g ∈ p.edges, ∃ x, x ≠ v ∧ G.Joins g x v ∧ G.ReachableIn (G.awayFrom S v) u x := by
  induction p with
  | nil u => exact Or.inl ⟨reachableIn_refl _ _, hu⟩
  | @cons a b c g hg p ih =>
    have hgS : g ∈ S := hp g List.mem_cons_self
    by_cases hgv : G.tail g ≠ v ∧ G.head g ≠ v
    · have hb : b ≠ v := by
        rcases hg with ⟨-, h⟩ | ⟨h, -⟩
        · exact h ▸ hgv.2
        · exact h ▸ hgv.1
      have hab : G.ReachableIn (G.awayFrom S v) a b :=
        (reachableIn_refl _ a).step (mem_awayFrom.2 ⟨hgS, hgv⟩) hg
      rcases ih (fun f hf => hp f (List.mem_cons_of_mem _ hf)) hb with
        ⟨h₁, h₂⟩ | ⟨g', hg', x, hx, hj, h₁⟩
      · exact Or.inl ⟨hab.trans h₁, h₂⟩
      · exact Or.inr ⟨g', List.mem_cons_of_mem _ hg', x, hx, hj, hab.trans h₁⟩
    · refine Or.inr ⟨g, List.mem_cons_self, a, hu, ?_, reachableIn_refl _ _⟩
      rcases hg with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
      · exact Or.inl ⟨h₁, not_not.1 fun hh => hgv ⟨fun ht => hu (h₁.symm.trans ht), hh⟩⟩
      · exact Or.inr ⟨not_not.1 fun ht => hgv ⟨ht, fun hh => hu (h₂.symm.trans hh)⟩, h₂⟩

/-- A cycle of the frame-type contraction at `v` is one of its loops, or a cycle of `G` avoiding
`v`. -/
theorem IsCycle.of_frameContract {C : Set E} (hC : (G.frameContract v).IsCycle C) :
    (∃ f, (G.frameContract v).tail f = (G.frameContract v).head f ∧ C = {f}) ∨
      (G.IsCycle C ∧ v ∉ G.verts C) := by
  by_cases hl : ∃ f ∈ C, (G.frameContract v).tail f = (G.frameContract v).head f
  · obtain ⟨f, hf, hloop⟩ := hl
    exact Or.inl ⟨f, hloop, hC.eq_singleton_of_loop hf hloop⟩
  · push Not at hl
    have haway : ∀ f ∈ C, G.tail f ≠ v ∧ G.head f ≠ v := fun f hf =>
      not_or.1 fun h => hl f hf (frameContract_tail_eq_head h)
    refine Or.inr ⟨hC.map_vertex Function.injective_id fun f hf =>
      ⟨(frameContract_ends (haway f hf).1 (haway f hf).2).1.symm,
        (frameContract_ends (haway f hf).1 (haway f hf).2).2.symm⟩, fun hv => ?_⟩
    obtain ⟨f, hf, h⟩ := mem_verts.1 hv
    exact h.elim (haway f hf).1 (haway f hf).2

/-- A cycle of `G` avoiding `v` is a cycle of the frame-type contraction at `v`. -/
theorem IsCycle.frameContract {C : Set E} (hC : G.IsCycle C) (hv : v ∉ G.verts C) :
    (G.frameContract v).IsCycle C :=
  hC.map_vertex Function.injective_id fun f hf =>
    frameContract_ends (fun h => hv (mem_verts.2 ⟨f, hf, Or.inl h⟩))
      fun h => hv (mem_verts.2 ⟨f, hf, Or.inr h⟩)

/-- A set of edges avoiding `v` has the same vertices after the frame-type contraction at `v`. -/
theorem verts_frameContract {C : Set E} (hv : v ∉ G.verts C) :
    (G.frameContract v).verts C = G.verts C := by
  ext x
  simp only [mem_verts]
  refine exists_congr fun f => and_congr_right fun hf => ?_
  have h := frameContract_ends (fun h => hv (mem_verts.2 ⟨f, hf, Or.inl h⟩))
    fun h => hv (mem_verts.2 ⟨f, hf, Or.inr h⟩)
  rw [h.1, h.2]

/-- The vertex `1` of `loopLink` does not meet its loop `0`. -/
theorem loopLink_one_notMem : (1 : Fin 2) ∉ loopLink.verts {0} := fun h => by
  obtain ⟨e, rfl, he⟩ := mem_verts.1 h
  rcases he with h | h <;> exact absurd h (by decide)

example := frameContract_ends (G := pathThree) (v := 2) (f := 0) (by decide) (by decide)

example := frameContract_ends_of_joins (G := pumpkin) (f := 0) (x := 0) (v := 1)
  (Or.inl ⟨rfl, rfl⟩) (by decide)

example := frameContract_ends_of_loop (G := twoLoops) (f := true) (v := ()) rfl rfl

example := frameContract_tail_eq_head (G := pumpkin) (f := 0) (v := 1) (Or.inr rfl)

example := joins_frameContract_tail (G := pumpkin) (f := 0) (v := 1) (Or.inr rfl)
  fun h => absurd h.1 (by decide)

example := ReachableIn.of_forall_graph (G := pumpkin) (H := pumpkin) (S := {0}) (T := {0})
  (fun _ hf => reachableIn_of_mem hf) (reachableIn_of_mem (Set.mem_singleton 0))

example := reachableIn_frameContract_iff (G := pathThree) (v := 2) (S := Set.univ) (u := 0) (w := 1)

example := pathThreeFirst.reachableIn_awayFrom_or (v := 2) (S := Set.univ) (fun _ _ => trivial)
  (by decide)

example := IsCycle.of_frameContract (G := pumpkin) (v := 1) (C := {0})
  (isCycle_singleton_of_loop (frameContract_tail_eq_head (Or.inr rfl)))

example := (isCycle_singleton_of_loop (G := loopLink) (e := 0) rfl).frameContract
  loopLink_one_notMem

example := verts_frameContract loopLink_one_notMem

end OrientedGraph

end Transformer.GainMatroid
