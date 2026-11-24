module

public import Transformer.GainMatroid.Section2_ExchangeGraphs
public import Transformer.GainMatroid.Section2_ExchangeWalks

/-!
# Paths across an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.6. Let `T`, `S` be
an exchange pair on `X = {a, b, c}` (`IsExchange`), with the triangle `X` in `T` and the star `X` at
`v` in `S`. A path `P` of `S` between vertices of `T` other than `v` becomes a path `P'` of `T`
(`IsExchange.exists_isTransfer`): `P` itself if it misses `v`, and otherwise `P` with its two edges
at `v` replaced by the third edge of `X`, so that `P'` has the edges of `P ∆ X` (`IsTransfer`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V W E : Type*} {T : OrientedGraph V E} {S : OrientedGraph W E}

/-- A path `P'` of `T` from `s` to `t` is a **transfer** of a path `P` of `S` from `ι s` to `ι t`
across `X` and `v` (§2.4): its vertices map to vertices of `P`, and it has the edges of `P` if `P`
misses `v`, and those of `P ∆ X` otherwise. -/
structure IsTransfer (X : Set E) (ι : V → W) (v : W) {s t : V} (P : S.Walk (ι s) (ι t))
    (P' : T.Walk s t) : Prop where
  isPath : P'.IsPath
  support : ∀ u ∈ P'.support, ι u ∈ P.support
  edgeSet_of_not_mem : v ∉ P.support → P'.edgeSet = P.edgeSet
  edgeSet_of_mem : v ∈ P.support → P'.edgeSet = P.edgeSet ∆ X

/-- An edge of a transfer off `X` is an edge of the path. -/
theorem IsTransfer.mem_edges {X : Set E} {ι : V → W} {v : W} {s t : V} {P : S.Walk (ι s) (ι t)}
    {P' : T.Walk s t} (hP : IsTransfer X ι v P P') {g : E} (hg : g ∈ P'.edges) (hgX : g ∉ X) :
    g ∈ P.edges := by
  by_cases hv : v ∈ P.support
  · have hg' : g ∈ P'.edgeSet := hg
    rw [hP.edgeSet_of_mem hv, Set.mem_symmDiff] at hg'
    exact (hg'.resolve_right fun h => hgX h.1).1
  · have hg' : g ∈ P'.edgeSet := hg
    rwa [hP.edgeSet_of_not_mem hv] at hg'

namespace IsExchange

variable {ι : V → W} {v : W} {a b c : E} {x y z : V}

/-- A path of `S` that misses `v` is its own transfer. -/
theorem exists_isTransfer_of_not_mem (h : IsExchange T S ι v a b c x y z) {s t : V}
    {P : S.Walk (ι s) (ι t)} (hP : P.IsPath) (hv : v ∉ P.support) :
    ∃ P' : T.Walk s t, IsTransfer {a, b, c} ι v P P' := by
  obtain ⟨u, P', hu, hP₁, hP₂⟩ := h.exists_lift_of_not_mem P hv
  obtain rfl : u = t := h.inj hu
  refine ⟨P', List.Nodup.of_map ι (by rw [hP₂]; exact hP), fun w hw => ?_, fun _ => ?_,
    fun hv' => absurd hv' hv⟩
  · rw [← hP₂]
    exact List.mem_map_of_mem hw
  · unfold Walk.edgeSet
    rw [hP₁]

/-- A path of `S` between vertices of `T` through `v` becomes a path of `T` with the edges of
`P ∆ X`: its two edges at `v` give way to the third edge of `X`. -/
theorem exists_isTransfer_of_mem (h : IsExchange T S ι v a b c x y z) {s t : V}
    {P : S.Walk (ι s) (ι t)} (hP : P.IsPath) (hs : ι s ≠ v) (ht : ι t ≠ v) (hv : v ∈ P.support) :
    ∃ P' : T.Walk s t, IsTransfer {a, b, c} ι v P P' := by
  obtain ⟨A, B, rfl⟩ := P.exists_eq_append_of_mem_support hv
  obtain ⟨w₁, e₁, h₁, A₁, hA⟩ := A.reverse.exists_eq_cons_of_ne hs.symm
  obtain ⟨w₂, e₂, h₂, B₁, rfl⟩ := B.exists_eq_cons_of_ne ht.symm
  have hsA : A.support = A₁.support.reverse ++ [v] := by
    have := A.support_reverse
    rw [hA] at this
    rw [← List.reverse_reverse A.support, ← this]
    simp [Walk.support]
  have heA : A.edges = A₁.edges.reverse ++ [e₁] := by
    have := A.edges_reverse
    rw [hA] at this
    rw [← List.reverse_reverse A.edges, ← this]
    simp [Walk.edges]
  have hsP : (A.append (.cons e₂ h₂ B₁)).support = A₁.support.reverse ++ v :: B₁.support := by
    rw [Walk.support_append, hsA]
    simp [Walk.support]
  have hnd : (A₁.support.reverse ++ v :: B₁.support).Nodup := hsP ▸ hP
  have hvA : v ∉ A₁.support := fun hv' =>
    (List.nodup_append.1 hnd).2.2 v (List.mem_reverse.2 hv') v List.mem_cons_self rfl
  have hvB : v ∉ B₁.support := (List.nodup_cons.1 (List.nodup_append.1 hnd).2.1).1
  have he₁ : e₁ ∈ ({a, b, c} : Set E) := h.mem_of_incident h₁.tail_or_head
  have he₂ : e₂ ∈ ({a, b, c} : Set E) := h.mem_of_incident h₂.tail_or_head
  have hne : e₁ ≠ e₂ := fun he => hP.not_mem_of_mem (e := e₁) (by rw [heA]; simp)
    (by rw [he]; simp [Walk.edges])
  obtain ⟨e₀, x', y', z', h', hX⟩ := h.exists_relabel he₁ he₂ hne
  obtain rfl : w₁ = ι y' := h₁.eq_of_joins h'.joins_b
  obtain rfl : w₂ = ι z' := h₂.eq_of_joins h'.joins_c
  obtain ⟨u₁, p₁, hu₁, hp₁e, hp₁s⟩ := h.exists_lift_of_not_mem A₁ hvA
  obtain ⟨u₂, p₂, hu₂, hp₂e, hp₂s⟩ := h.exists_lift_of_not_mem B₁ hvB
  obtain rfl : u₁ = s := h.inj hu₁
  obtain rfl : u₂ = t := h.inj hu₂
  have hsub : ((p₁.reverse.append (.cons e₀ h'.triangle.joins_a p₂)).support.map ι).Sublist
      (A.append (.cons e₂ h₂ B₁)).support := by
    rw [hsP, Walk.support_append, Walk.support_reverse]
    simp only [Walk.support, List.tail_cons, List.map_append, List.map_reverse, hp₁s, hp₂s]
    exact (List.sublist_cons_self v B₁.support).append_left _
  have hR : ∀ g ∈ A₁.edgeSet ∪ B₁.edgeSet, g ∉ ({e₀, e₁, e₂} : Set E) := by
    rintro g (hg | hg)
    · exact h'.not_mem_of_not_mem_support hvA hg
    · exact h'.not_mem_of_not_mem_support hvB hg
  refine ⟨p₁.reverse.append (.cons e₀ h'.triangle.joins_a p₂),
    ⟨List.Nodup.of_map ι (hsub.nodup hP), fun u hu => hsub.subset (List.mem_map_of_mem hu),
      fun hv' => absurd hv hv', fun _ => ?_⟩⟩
  have hPe : (A.append (.cons e₂ h₂ B₁)).edgeSet =
      insert e₁ (insert e₂ (A₁.edgeSet ∪ B₁.edgeSet)) := by
    ext g
    simp only [Walk.edgeSet, Set.mem_ofPred_eq, Walk.edges_append, heA, Walk.edges,
      List.mem_append, List.mem_reverse, List.mem_cons, List.not_mem_nil,
      Set.mem_insert_iff, Set.mem_union, or_false]
    tauto
  have hP'e : (p₁.reverse.append (.cons e₀ h'.triangle.joins_a p₂)).edgeSet =
      insert e₀ (A₁.edgeSet ∪ B₁.edgeSet) := by
    ext g
    simp only [Walk.edgeSet, Set.mem_ofPred_eq, Walk.edges_append, Walk.edges_reverse, hp₁e,
      hp₂e, Walk.edges, List.mem_append, List.mem_reverse, List.mem_cons, Set.mem_insert_iff,
      Set.mem_union]
    tauto
  rw [hPe, hP'e, ← hX, insert_insert_symmDiff_triple hR h'.triangle.ne_ab h'.triangle.ne_ac
    h'.triangle.ne_bc]

/-- A path of `S` between vertices of `T` other than `v` has a transfer. -/
theorem exists_isTransfer (h : IsExchange T S ι v a b c x y z) {s t : V} {P : S.Walk (ι s) (ι t)}
    (hP : P.IsPath) (hs : ι s ≠ v) (ht : ι t ≠ v) :
    ∃ P' : T.Walk s t, IsTransfer {a, b, c} ι v P P' := by
  by_cases hv : v ∈ P.support
  · exact h.exists_isTransfer_of_mem hP hs ht hv
  · exact h.exists_isTransfer_of_not_mem hP hv

end IsExchange

end OrientedGraph

open OrientedGraph

theorem clawEdge12_isTransfer : IsTransfer {0, 1, 2} id 0 clawEdge12 nablaEdge12 :=
  ⟨by unfold Walk.IsPath; decide, by decide, fun _ => rfl, fun h => absurd h (by decide)⟩

theorem clawStar12_isTransfer : IsTransfer {0, 1, 2} id 0 clawStar12 nablaTriangle12 := by
  refine ⟨by unfold Walk.IsPath; decide, by decide, fun h => absurd (by decide) h, fun _ => ?_⟩
  ext g
  fin_cases g <;> simp [Walk.edgeSet, Walk.edges, clawStar12, nablaTriangle12, Set.mem_symmDiff]

example := clawEdge12_isTransfer.mem_edges (g := 3) (by simp [nablaEdge12, Walk.edges]) (by simp)

example := clawNabla_isExchange.exists_isTransfer_of_not_mem clawEdge12_isPath (by decide)

example := clawNabla_isExchange.exists_isTransfer_of_mem clawStar12_isPath (by decide) (by decide)
  (by decide)

example := clawNabla_isExchange.exists_isTransfer clawStar12_isPath (by decide) (by decide)

end Transformer.GainMatroid
