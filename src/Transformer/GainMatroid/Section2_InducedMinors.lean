module

public import Transformer.GainMatroid.Section2_InducedRealization

/-!
# Proposition 2.4

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), Proposition 2.4: a gain function induced on a minor
realizes that minor. `LinkMinorMap.realizes_of_isInducedGain` is the case of a link minor, and
`IsFrameContraction.realizes` the frame-type contraction of a joint at `v`, where a cycle of the
contraction is either a loop made from an edge at `v` or a cycle of `Ω` avoiding `v`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

universe uV uE uV' uE'

variable {V : Type uV} {E : Type uE} {V' : Type uV'} {E' : Type uE'} {Γ : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E} {v : V} {f : E}

theorem frameContract_of_off (h₁ : G.tail f ≠ v) (h₂ : G.head f ≠ v) :
    (G.frameContract v).tail f = G.tail f ∧ (G.frameContract v).head f = G.head f := by
  simp [frameContract, h₁, h₂]

/-- An edge at `v` is a loop of the frame-type contraction. -/
theorem frameContract_loop_of_touch (h : G.tail f = v ∨ G.head f = v) :
    (G.frameContract v).tail f = (G.frameContract v).head f := by
  by_cases h₁ : G.tail f = v
  · by_cases h₂ : G.head f = v <;> simp [frameContract, h₁, h₂]
  · have h₂ : G.head f = v := h.resolve_left h₁
    simp [frameContract, h₁, h₂]

/-- A cycle walk through a loop has no other edge. -/
theorem Walk.IsCycleWalk.edgeSet_eq_singleton_of_loop {u : V} {W : G.Walk u u}
    (hW : W.IsCycleWalk) {l : E} (hl : l ∈ W.edges) (hloop : G.tail l = G.head l) :
    W.edgeSet = {l} := by
  obtain ⟨s, t, hj, A, B, rfl⟩ := W.exists_append_cons_of_mem hl
  obtain rfl : s = t := by
    rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
    · exact h₁.symm.trans (hloop.trans h₂)
    · exact h₂.symm.trans (hloop.symm.trans h₁)
  have hnd := hW.support_tail_nodup
  rw [Walk.support_append] at hnd
  cases A with
  | nil =>
    have hB : B.support.Nodup := hnd
    cases B with
    | nil =>
      ext f
      simp [Walk.edgeSet, Walk.append, Walk.edges]
    | cons g hg B' =>
      exact absurd B'.end_mem_support (List.nodup_cons.1 (show (_ :: B'.support).Nodup from hB)).1
  | cons a ha A' =>
    have hdis := List.nodup_append.1 (show (A'.support ++ B.support).Nodup from hnd)
    exact absurd rfl (hdis.2.2 s A'.end_mem_support s B.start_mem_support)

/-- A walk whose edges have the same ends in `G`, through an edge map, is a walk of `G`. -/
theorem Walk.exists_map {E₂ : Type*} {H : OrientedGraph V E₂} (ε : E₂ → E) {x y : V}
    (p : H.Walk x y) (h : ∀ g ∈ p.edges, G.tail (ε g) = H.tail g ∧ G.head (ε g) = H.head g) :
    ∃ q : G.Walk x y, q.edges = p.edges.map ε ∧ q.support = p.support := by
  induction p with
  | nil x => exact ⟨.nil x, rfl, rfl⟩
  | @cons a b c g hg p ih =>
    obtain ⟨q, hq₁, hq₂⟩ := ih fun g' hg' => h g' (by simp [Walk.edges, hg'])
    obtain ⟨h₁, h₂⟩ := h g (by simp [Walk.edges])
    have hj : G.Joins (ε g) a b := by
      rcases hg with ⟨ha, hb⟩ | ⟨ha, hb⟩
      · exact Or.inl ⟨h₁.trans ha, h₂.trans hb⟩
      · exact Or.inr ⟨h₁.trans ha, h₂.trans hb⟩
    exact ⟨.cons (ε g) hj q, by simp [Walk.edges, hq₁], by simp [Walk.support, hq₂]⟩

/-- A cycle whose edges have the same ends in `G`, through an injective edge map, is a cycle of
`G`. -/
theorem IsCycle.map {E₂ : Type*} {H : OrientedGraph V E₂} {C : Set E₂} (hC : H.IsCycle C)
    (ε : E₂ → E) (hε : Function.Injective ε)
    (h : ∀ g ∈ C, G.tail (ε g) = H.tail g ∧ G.head (ε g) = H.head g) : G.IsCycle (ε '' C) := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  obtain ⟨q, h₁, h₂⟩ := W.exists_map (G := G) ε h
  refine ⟨u, q, ⟨?_, ?_, ?_⟩, ?_⟩
  · rw [h₁]
    simpa using hW.edges_ne_nil
  · rw [h₁]
    exact hW.edges_nodup.map hε
  · rw [h₂]
    exact hW.support_tail_nodup
  · ext e
    simp [Walk.edgeSet, h₁]

end OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {Ω' : BiasedGraph V' E'} {e : E} {v : V}

/-- **Proposition 2.4** for link minors: a gain function induced on a link minor realizes it. -/
theorem LinkMinorMap.realizes_of_isInducedGain [DecidableEq V] [DecidableEq V'] [Group Γ]
    (m : Ω.LinkMinorMap Ω') {γ : E → Γ} {ι : E' → Γ} (hγ : Ω.Realizes γ)
    (hι : m.IsInducedGain γ ι) : Ω'.Realizes ι := by
  obtain ⟨η, hη, hι⟩ := hι
  have hoff : ∀ (ζ : Quotient (Ω.graph.reachableSetoid m.contractSet) → Γ), ∀ f ∈ m.contractSet,
      switch Ω.graph (fun v => ζ (Quotient.mk _ v)) (switch Ω.graph η γ) f = 1 := fun ζ f hf => by
    rw [switch_apply, hη f hf, mul_one, contract_class_eq hf, inv_mul_cancel]
  have hbal : ∀ ζ : V → Γ,
      Ω.graph.balancedCycles (switch Ω.graph ζ (switch Ω.graph η γ)) = Ω.balanced := fun ζ => by
    rw [balancedCycles_switch, balancedCycles_switch]
    exact hγ
  have hext : ∀ ξ : V' → Γ, ∃ ζ : Quotient (Ω.graph.reachableSetoid m.contractSet) → Γ,
      (fun x => ζ (m.vertex x)) = ξ := fun ξ =>
    ⟨Function.extend m.vertex ξ 1, funext fun x => m.vertex_injective.extend_apply ξ 1 x⟩
  change Ω'.graph.balancedCycles ι = Ω'.balanced
  ext C'
  constructor
  · intro hC'
    obtain ⟨hcyc, ξ, hξ⟩ := mem_balancedCycles_iff.1 hC'
    obtain ⟨K₁, hK₁, hlift⟩ := m.exists_isCycle_lift hcyc
    obtain ⟨ζ, rfl⟩ := hext ξ
    refine (m.balanced_iff C' hcyc).2 ⟨K₁, hK₁, ?_⟩
    rw [← hbal fun v => ζ (Quotient.mk _ v)]
    refine mem_balancedCycles_iff.2 ⟨hlift, 1, fun f hf => ?_⟩
    rw [switch_one]
    rcases hf with ⟨e', he', rfl⟩ | hf
    · exact (m.switch_eq_one_iff hι ζ e').1 (hξ e' he')
    · exact hoff ζ f (hK₁ hf)
  · intro hC'
    have hcyc := Ω'.isCycle_of_mem _ hC'
    obtain ⟨K₀, hK₀, hb⟩ := (m.balanced_iff C' hcyc).1 hC'
    obtain ⟨u, W', hW', rfl⟩ := hcyc
    cases W' with
    | nil => exact absurd rfl hW'.edges_ne_nil
    | cons e₀ h p =>
      have hp : p.IsPath := hW'.support_tail_nodup
      obtain ⟨ξ, -, -, hξ⟩ := Walk.exists_potential_of_isPath ι p hp 1
      obtain ⟨ζ, rfl⟩ := hext ξ
      rw [← hbal fun v => ζ (Quotient.mk _ v)] at hb
      obtain ⟨w, W, hW, hWe⟩ := (mem_balancedCycles_iff.1 hb).1
      have hmemW : ∀ f, f ∈ W.edges ↔
          f ∈ (fun e' => (m.edge e' : E)) '' (Walk.cons e₀ h p).edgeSet ∪ K₀ :=
        fun f => Set.ext_iff.1 hWe f
      rw [← hWe] at hb
      have h₁ := hW.eq_one_of_forall_ne hb
        ((hmemW _).2 (Or.inl ⟨e₀, by simp [Walk.edgeSet, Walk.edges], rfl⟩))
        (fun f hf hne => by
          rcases (hmemW f).1 hf with ⟨g, hg, rfl⟩ | hK
          · have hg' : g ∈ p.edges := by
              rcases List.mem_cons.1 (show g ∈ e₀ :: p.edges from hg) with hge | hg'
              · exact absurd (by rw [hge]) hne
              · exact hg'
            exact (m.switch_eq_one_iff hι ζ g).1 (hξ g hg')
          · exact hoff ζ f (hK₀ hK))
      refine mem_balancedCycles_iff.2 ⟨⟨u, _, hW', rfl⟩, fun x => ζ (m.vertex x), fun g hg => ?_⟩
      rcases List.mem_cons.1 (show g ∈ e₀ :: p.edges from hg) with hge | hg
      · rw [hge]
        exact (m.switch_eq_one_iff hι ζ e₀).2 h₁
      · exact hξ g hg

end BiasedGraph

end Transformer.GainMatroid
