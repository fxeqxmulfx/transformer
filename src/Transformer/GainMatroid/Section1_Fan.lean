module

public import Transformer.GainMatroid.Section1_ThreeVertices

/-!
# A frame representation of `U_{3,5}` on three vertices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2. Let `fanGraph` have the vertices `x = 0`,
`y = 1`, `u = 2`, a link `0` from `x` to `y`, two links `1`, `2` from `x` to `u` and two links `3`,
`4` from `y` to `u`. With no balanced cycle, its frame matroid is the uniform matroid `U_{3,5}`. The
circuits are the five sets of four edges: one tight handcuff at `u` and four thetas.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

section Theta

variable {E : Type*} {G : OrientedGraph (Fin 3) E}

private theorem third_vertex_unique : ∀ x y m m' : Fin 3, x ≠ y → m ≠ x → m ≠ y → m' ≠ x →
    m' ≠ y → m = m' := by
  decide

/-- On three vertices with no three parallel links, a theta has exactly four edges. -/
theorem IsThetaSet.encard_eq_four
    (hpar : ∀ (e f g : E) (x y : Fin 3), e ≠ f → e ≠ g → f ≠ g → x ≠ y → G.Joins e x y →
      G.Joins f x y → G.Joins g x y → False)
    {X : Set E} (h : G.IsThetaSet X) : X.encard = 4 := by
  obtain ⟨x, y, P, Q, R, hT, rfl⟩ := h
  have hlen : ∀ p : G.Walk x y, p.IsPath → p.edges.length = 1 ∨ p.edges.length = 2 :=
    fun p hp => by
      have hle := (show p.support.Nodup from hp).length_le_card
      rw [Walk.length_support, Fintype.card_fin] at hle
      have hpos : p.edges.length ≠ 0 := fun h0 =>
        Walk.edges_ne_nil_of_ne hT.ne p (List.length_eq_zero_iff.1 h0)
      omega
  have hmid : ∀ {p q : G.Walk x y}, Walk.Openly p q → p.IsPath → q.IsPath →
      ¬(p.edges.length = 2 ∧ q.edges.length = 2) := by
    rintro p q hO hp hq ⟨hp2, hq2⟩
    obtain ⟨m, hm, hmx, hmy⟩ := hp.exists_interior hp2
    obtain ⟨m', hm', hmx', hmy'⟩ := hq.exists_interior hq2
    obtain rfl := third_vertex_unique x y m m' hT.ne hmx hmy hmx' hmy'
    rcases hO.verts m hm hm' with h | h
    exacts [hmx h, hmy h]
  have hnot3 : ¬(P.edges.length = 1 ∧ Q.edges.length = 1 ∧ R.edges.length = 1) := by
    rintro ⟨h₁, h₂, h₃⟩
    obtain ⟨e, he, hje⟩ := P.exists_edge_of_length_one h₁
    obtain ⟨f, hf, hjf⟩ := Q.exists_edge_of_length_one h₂
    obtain ⟨g, hg, hjg⟩ := R.exists_edge_of_length_one h₃
    exact hpar e f g x y (fun hc => hT.openPQ.edges e (by simp [he]) (by simp [hf, hc]))
      (fun hc => hT.openPR.edges e (by simp [he]) (by simp [hg, hc]))
      (fun hc => hT.openQR.edges f (by simp [hf]) (by simp [hg, hc])) hT.ne hje hjf hjg
  have h₁ := hlen P hT.pathP
  have h₂ := hlen Q hT.pathQ
  have h₃ := hlen R hT.pathR
  have hPQ := hmid hT.openPQ hT.pathP hT.pathQ
  have hPR := hmid hT.openPR hT.pathP hT.pathR
  have hQR := hmid hT.openQR hT.pathQ hT.pathR
  have hsum : P.edges.length + Q.edges.length + R.edges.length = 4 := by omega
  have hd₁ : Disjoint P.edgeSet Q.edgeSet := Set.disjoint_left.2 fun e he => hT.openPQ.edges e he
  have hd₂ : Disjoint (P.edgeSet ∪ Q.edgeSet) R.edgeSet := by
    rw [Set.disjoint_union_left]
    exact ⟨Set.disjoint_left.2 fun e he => hT.openPR.edges e he,
      Set.disjoint_left.2 fun e he => hT.openQR.edges e he⟩
  rw [Set.encard_union_eq hd₂, Set.encard_union_eq hd₁,
    Walk.encard_edgeSet_of_nodup hT.pathP.edges_nodup,
    Walk.encard_edgeSet_of_nodup hT.pathQ.edges_nodup,
    Walk.encard_edgeSet_of_nodup hT.pathR.edges_nodup]
  exact_mod_cast hsum

end Theta

/-- The graph of the counterexample to Theorem 1 (i): `x = 0`, `y = 1`, `u = 2`, with a link `x → y`
and two parallel links `x → u` and `y → u`. -/
def fanGraph : OrientedGraph (Fin 3) (Fin 5) := ⟨![0, 0, 0, 1, 1], ![1, 2, 2, 2, 2]⟩

theorem fanGraph_loopless : fanGraph.Loopless := by
  simp only [Loopless, IsLink, fanGraph]
  decide

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
theorem fanGraph_noThree : ∀ (e f g : Fin 5) (x y : Fin 3), e ≠ f → e ≠ g → f ≠ g → x ≠ y →
    fanGraph.Joins e x y → fanGraph.Joins f x y → fanGraph.Joins g x y → False := by
  simp only [Joins, fanGraph]
  decide +kernel

private theorem fan_tight : fanGraph.IsTightHandcuff ({0}ᶜ : Set (Fin 5)) := by
  refine ⟨{1, 2}, {3, 4}, 2,
    isCycle_pair (u := 0) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide),
    isCycle_pair (u := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide),
    ?_, ?_, ?_⟩
  · rw [Set.disjoint_left]
    intro a
    fin_cases a <;> simp
  · ext w
    fin_cases w <;> simp [mem_verts, fanGraph]
  · ext a
    fin_cases a <;> simp

private theorem fan_theta₁ : fanGraph.IsThetaSet ({1}ᶜ : Set (Fin 5)) := by
  have h : ({1}ᶜ : Set (Fin 5)) = {3, 4, 0, 2} := by
    ext a
    fin_cases a <;> simp
  rw [h]
  exact isThetaSet_of_parallel (x := 1) (m := 0) (y := 2) (by decide) (by decide) (by decide)
    (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (by decide) (by decide) (by decide) (by decide) (by decide)

private theorem fan_theta₂ : fanGraph.IsThetaSet ({2}ᶜ : Set (Fin 5)) := by
  have h : ({2}ᶜ : Set (Fin 5)) = {3, 4, 0, 1} := by
    ext a
    fin_cases a <;> simp
  rw [h]
  exact isThetaSet_of_parallel (x := 1) (m := 0) (y := 2) (by decide) (by decide) (by decide)
    (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (by decide) (by decide) (by decide) (by decide) (by decide)

private theorem fan_theta₃ : fanGraph.IsThetaSet ({3}ᶜ : Set (Fin 5)) := by
  have h : ({3}ᶜ : Set (Fin 5)) = {1, 2, 0, 4} := by
    ext a
    fin_cases a <;> simp
  rw [h]
  exact isThetaSet_of_parallel (x := 0) (m := 1) (y := 2) (by decide) (by decide) (by decide)
    (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (by decide) (by decide) (by decide) (by decide) (by decide)

private theorem fan_theta₄ : fanGraph.IsThetaSet ({4}ᶜ : Set (Fin 5)) := by
  have h : ({4}ᶜ : Set (Fin 5)) = {1, 2, 0, 3} := by
    ext a
    fin_cases a <;> simp
  rw [h]
  exact isThetaSet_of_parallel (x := 0) (m := 1) (y := 2) (by decide) (by decide) (by decide)
    (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
    (by decide) (by decide) (by decide) (by decide) (by decide)

/-- The complement of each edge of `fanGraph` is a tight handcuff or a theta. -/
theorem fan_compl_shape (e : Fin 5) :
    fanGraph.IsTightHandcuff ({e}ᶜ : Set (Fin 5)) ∨ fanGraph.IsThetaSet ({e}ᶜ : Set (Fin 5)) := by
  fin_cases e
  · exact Or.inl fan_tight
  · exact Or.inr fan_theta₁
  · exact Or.inr fan_theta₂
  · exact Or.inr fan_theta₃
  · exact Or.inr fan_theta₄

/-- With no balanced cycle, the frame matroid of `fanGraph` is `U_{3,5}`. -/
theorem fan_isFrameMatroid :
    fanGraph.contrabalanced.IsFrameMatroid (uniformMatroid (Fin 5) 3) := by
  refine ⟨rfl, fun X => ?_⟩
  rw [uniformMatroid_isCircuit_iff]
  constructor
  · intro h
    have hc : Xᶜ.ncard = 1 := by
      have h' := Set.ncard_add_ncard_compl X (Set.toFinite X) (Set.toFinite Xᶜ)
      simp only [Nat.card_eq_fintype_card, Fintype.card_fin] at h'
      omega
    obtain ⟨e, he⟩ := Set.ncard_eq_one.1 hc
    have hX : X = {e}ᶜ := by rw [← he, compl_compl]
    refine Or.inr ⟨?_, fun _ _ _ hmem => Set.notMem_empty _ hmem⟩
    rw [hX]
    rcases fan_compl_shape e with h' | h'
    · exact Or.inl h'
    · exact Or.inr (Or.inr h')
  · have hcast : X.encard = 4 → X.ncard = 3 + 1 := fun h => by
      have hn := (Set.toFinite X).cast_ncard_eq
      rw [h] at hn
      exact_mod_cast hn
    rintro (h | ⟨h | h | h, -⟩)
    · exact absurd h (Set.notMem_empty _)
    · exact hcast (IsTightHandcuff.encard_eq_four fanGraph_loopless h)
    · exact absurd h (not_isLooseHandcuff fanGraph_loopless X)
    · exact hcast (IsThetaSet.encard_eq_four fanGraph_noThree h)

example : ({0}ᶜ : Set (Fin 5)).encard = 4 :=
  IsTightHandcuff.encard_eq_four fanGraph_loopless fan_tight

example : ({1}ᶜ : Set (Fin 5)).encard = 4 :=
  IsThetaSet.encard_eq_four fanGraph_noThree fan_theta₁

example : ¬fanGraph.IsLooseHandcuff Set.univ := not_isLooseHandcuff fanGraph_loopless _

end OrientedGraph

end Transformer.GainMatroid
