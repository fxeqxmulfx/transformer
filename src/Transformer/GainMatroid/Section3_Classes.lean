module

public import Transformer.GainMatroid.Section4_SmallExamples

/-!
# The properly unbalanced biased `2C₃`, tubes and `K₄`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §3.1. A biased `2C₃` or a biased tube is properly
unbalanced exactly when it has no balanced 2-cycle, and a biased `K₄` exactly when it has no
balanced triangle. The source asserts these without proof: `K₄` before Proposition 3.1, `2C₃` in
the proof of that proposition and the tube after it.

If a 2-cycle or triangle `C` is balanced, then a vertex `w` off `C` lies on every other cycle of at
least two edges, so `w` balances the biased graph (`not_isProperlyUnbalanced_of_off`). On the tube,
the edges off `w` are `C` and one pendant edge, which lies on no cycle
(`IsCycle.not_mem_of_pendant`).
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

namespace OrientedGraph

variable {G H : OrientedGraph V E}

theorem IsReorientationOf.symm (h : G.IsReorientationOf H) : H.IsReorientationOf G := fun e => by
  rcases h e with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · exact Or.inl ⟨h₁.symm, h₂.symm⟩
  · exact Or.inr ⟨h₂.symm, h₁.symm⟩

theorem IsReorientationOf.ends_ne (h : G.IsReorientationOf H) {g : E} {w : V}
    (h₁ : G.tail g ≠ w) (h₂ : G.head g ≠ w) : H.tail g ≠ w ∧ H.head g ≠ w := by
  rcases h g with ⟨ht, hh⟩ | ⟨ht, hh⟩
  · exact ⟨fun hc => h₁ (ht.trans hc), fun hc => h₂ (hh.trans hc)⟩
  · exact ⟨fun hc => h₂ (hh.trans hc), fun hc => h₁ (ht.trans hc)⟩

/-- A cycle of two edges is a pair of parallel links. -/
theorem IsCycle.exists_pair {C : Set E} (hC : G.IsCycle C) (h2 : C.encard = 2) :
    ∃ e f u v, e ≠ f ∧ u ≠ v ∧ G.Joins e u v ∧ G.Joins f u v ∧ C = {e, f} := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  rw [hW.encard_edgeSet] at h2
  have hlen : W.edges.length = 2 := by exact_mod_cast h2
  cases W with
  | nil => simp [Walk.edges] at hlen
  | cons e he p =>
    rename_i v
    cases p with
    | nil => simp [Walk.edges] at hlen
    | cons f hf q =>
      cases q with
      | cons g hg r => simp [Walk.edges] at hlen
      | nil =>
        have hne : e ≠ f := by simpa [Walk.edges] using hW.edges_nodup
        have huv : u ≠ v := by
          have := hW.support_tail_nodup
          simp only [Walk.support, List.tail_cons, List.nodup_cons, List.mem_singleton] at this
          exact fun h => this.1 h.symm
        refine ⟨e, f, u, v, hne, huv, he, hf.symm, ?_⟩
        ext g
        simp [Walk.edgeSet, Walk.edges]

/-- A cycle of three edges is a triangle. -/
theorem IsCycle.exists_triangle {C : Set E} (hC : G.IsCycle C) (h3 : C.encard = 3) :
    ∃ a b c x y z, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ G.Joins a x y ∧
      G.Joins b y z ∧ G.Joins c z x ∧ C = {a, b, c} := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  rw [hW.encard_edgeSet] at h3
  have hlen : W.edges.length = 3 := by exact_mod_cast h3
  cases W with
  | nil => simp [Walk.edges] at hlen
  | cons a ha p =>
    cases p with
    | nil => simp [Walk.edges] at hlen
    | cons b hb q =>
      cases q with
      | nil => simp [Walk.edges] at hlen
      | cons c hc r =>
        cases r with
        | cons d hd s => simp [Walk.edges] at hlen
        | nil =>
          have hE := hW.edges_nodup
          simp only [Walk.edges, List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false,
            not_or] at hE
          have hV := hW.support_tail_nodup
          simp only [Walk.support, List.tail_cons, List.nodup_cons, List.mem_cons,
            List.not_mem_nil, or_false, not_or] at hV
          refine ⟨a, b, c, u, _, _, hE.1.1, hE.1.2, hE.2.1, Ne.symm hV.1.2, Ne.symm hV.2.1, hV.1.1,
            ha, hb, hc, ?_⟩
          ext g
          simp [Walk.edgeSet, Walk.edges]

/-- A link with an end `p` that no other edge of the cycle `D` meets is not on `D`. -/
theorem IsCycle.not_mem_of_pendant {D : Set E} (hD : G.IsCycle D) {g : E} {p : V}
    (hp : G.tail g = p ∨ G.head g = p) (hlink : G.tail g ≠ G.head g)
    (honly : ∀ f ∈ D, f ≠ g → G.tail f ≠ p ∧ G.head f ≠ p) : g ∉ D := by
  intro hg
  obtain ⟨u, W, hW, rfl⟩ := hD
  have h := hW.eq_of_forall_ne (e := g) hg (fun v => v = p) fun f hf hfg => by
    obtain ⟨h₁, h₂⟩ := honly f hf hfg
    exact propext ⟨fun h => absurd h h₁, fun h => absurd h h₂⟩
  have h' : G.tail g = p ↔ G.head g = p := Iff.of_eq h
  rcases hp with hp | hp
  · exact hlink (hp.trans (h'.1 hp).symm)
  · exact hlink ((h'.2 hp).trans hp.symm)

/-- In a reorientation of `K₄`, every cycle has at least three edges. -/
theorem three_le_encard_of_K4 {G : OrientedGraph (Fin 4) (Fin 6)} (hG : G.IsReorientationOf K4)
    {D : Set (Fin 6)} (hD : G.IsCycle D) : 3 ≤ D.encard := by
  have hnp : ∀ (e f : Fin 6) (u v : Fin 4), e ≠ f → u ≠ v → K4.Joins e u v → K4.Joins f u v →
      False := by
    simp only [Joins, K4]
    decide
  obtain ⟨u, W, hW, rfl⟩ := hD
  rw [hW.encard_edgeSet]
  cases W with
  | nil => exact absurd rfl hW.edges_ne_nil
  | cons e he p =>
    rename_i v
    have huv : u ≠ v := fun h => by
      subst h
      exact Joins.ne_of_loopless (hG.loopless K4_loopless) he rfl
    cases p with
    | nil => exact absurd rfl huv
    | cons f hf q =>
      cases q with
      | nil =>
        have hne : e ≠ f := by simpa [Walk.edges] using hW.edges_nodup
        exact (hnp e f u v hne huv (hG.symm.joins he) (hG.symm.joins hf.symm)).elim
      | cons g hg r =>
        have : 3 ≤ (Walk.cons e he (Walk.cons f hf (Walk.cons g hg r))).edges.length := by
          simp [Walk.edges]
        exact_mod_cast this

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E}

/-- If every cycle of at least two edges off `w` is the balanced cycle `C`, then `w` balances `Ω`
apart from its joints, so `Ω` is not properly unbalanced. -/
theorem not_isProperlyUnbalanced_of_off {C : Set E} (hC : C ∈ Ω.balanced) {w : V}
    (h : ∀ D, Ω.graph.IsCycle D → 2 ≤ D.encard → w ∉ Ω.graph.verts D → D = C) :
    ¬Ω.IsProperlyUnbalanced := by
  rintro ⟨hnb, hna⟩
  refine hna ⟨hnb, w, fun D hD h2 => ?_⟩
  by_contra hw
  exact hD.2 ((h D hD.1 h2 hw).symm ▸ hC)

/-- If no balanced cycle has `k ≥ 2` edges, and each vertex is missed by a cycle of `k` edges, then
`Ω` is properly unbalanced. -/
theorem isProperlyUnbalanced_of_walks [Nonempty V] {k : ℕ} (hk : 2 ≤ k)
    (h : ∀ C ∈ Ω.balanced, C.encard ≠ k)
    (hoff : ∀ w, ∃ (u : V) (W : Ω.graph.Walk u u),
      W.IsCycleWalk ∧ W.edges.length = k ∧ w ∉ W.support) :
    Ω.IsProperlyUnbalanced := by
  have hkey : ∀ w, ∃ D, Ω.IsUnbalancedCycle D ∧ 2 ≤ D.encard ∧ w ∉ Ω.graph.verts D := fun w => by
    obtain ⟨u, W, hW, hlen, hw⟩ := hoff w
    have hcard : W.edgeSet.encard = k := by rw [hW.encard_edgeSet, hlen]
    refine ⟨W.edgeSet, ⟨⟨u, W, hW, rfl⟩, fun hmem => h _ hmem hcard⟩, ?_, fun hv => hw ?_⟩
    · rw [hcard]
      exact_mod_cast hk
    · obtain ⟨e, he, h₁ | h₁⟩ := mem_verts.1 hv
      · exact h₁ ▸ (W.ends_mem_support he).1
      · exact h₁ ▸ (W.ends_mem_support he).2
  refine ⟨fun hB => ?_, fun ⟨_, v, hv⟩ => ?_⟩
  · obtain ⟨w⟩ := ‹Nonempty V›
    obtain ⟨D, hD, -⟩ := hkey w
    exact hD.2 (hB D hD.1)
  · obtain ⟨D, hD, h2, hw⟩ := hkey v
    exact hw (hv D hD h2)

end BiasedGraph

end Transformer.GainMatroid
