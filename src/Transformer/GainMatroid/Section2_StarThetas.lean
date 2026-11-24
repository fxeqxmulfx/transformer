module

public import Transformer.GainMatroid.Section2_BalancedTripods
public import Transformer.GainMatroid.Section2_TransferCycles

/-!
# Thetas at the centre of the star

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.6. Let `X` be a
balanced triangle of a biased graph, and `P`, `Q`, `R` a tripod off `X` (`IsTripod`). If the cycles
`c ∪ P ∪ Q` and `b ∪ P ∪ R` are balanced, so is `a ∪ Q ∪ R` (`BiasedGraph.mem_balanced_of_tripod`):
for a tripod ending at the corner `z` the labels `b`, `y` and `c`, `z` exchange.

Let `T`, `S` be an exchange pair on `X = {a, b, c}` (`IsExchange`), with the star `X` at `v` in
`S`, and `X` a balanced triangle of a biased graph `Ω` on `T`. The three paths of a theta of `S`
from `v` start with the three edges of `X`. Up to the labels of the triangle they are `a`, `b`, `c`
followed by a tripod of `T` from `x`, `y`, `z`. Each cycle of the theta meets `X` in two edges
(`insert_union_insert_inter`), and the symmetric differences with `X` are `c ∪ P ∪ Q`, `b ∪ P ∪ R`
and `a ∪ Q ∪ R` (`insert_union_insert_symmDiff`). So the theta property of `Δ_X B` at `v` is that
of `B` on a subdivided `K₄` (`IsExchange.theta_of_eq`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

section Sets

variable {E : Type*} {X P Q : Set E} {a b c : E}

/-- `(insert a P ∪ insert b Q) ∩ X = {a, b}` for `a`, `b` in `X` and `P`, `Q` off `X`. -/
theorem insert_union_insert_inter (ha : a ∈ X) (hb : b ∈ X) (hP : ∀ g ∈ P, g ∉ X)
    (hQ : ∀ g ∈ Q, g ∉ X) : (insert a P ∪ insert b Q) ∩ X = {a, b} := by
  ext g
  simp only [Set.mem_inter_iff, Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff]
  constructor
  · rintro ⟨(rfl | hg) | (rfl | hg), hgX⟩
    · exact Or.inl rfl
    · exact absurd hgX (hP g hg)
    · exact Or.inr rfl
    · exact absurd hgX (hQ g hg)
  · rintro (rfl | rfl)
    · exact ⟨Or.inl (Or.inl rfl), ha⟩
    · exact ⟨Or.inr (Or.inl rfl), hb⟩

/-- `(insert a P ∪ insert b Q) ∆ X = insert c (P ∪ Q)` for `X = {a, b, c}` with `c ≠ a, b`, and
`P`, `Q` off `X`. -/
theorem insert_union_insert_symmDiff (hX : X = {a, b, c}) (hca : c ≠ a) (hcb : c ≠ b)
    (hP : ∀ g ∈ P, g ∉ X) (hQ : ∀ g ∈ Q, g ∉ X) :
    (insert a P ∪ insert b Q) ∆ X = insert c (P ∪ Q) := by
  subst hX
  ext g
  have hP' := hP g
  have hQ' := hQ g
  have h₁ : g = c → g ≠ a := fun h h' => hca (h.symm.trans h')
  have h₂ : g = c → g ≠ b := fun h h' => hcb (h.symm.trans h')
  simp only [Set.mem_symmDiff, Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff]
    at hP' hQ' ⊢
  tauto

end Sets

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {a b c : E} {x y z : V}

/-- §2.4: if `X = {a, b, c}` is a balanced triangle and `P`, `Q`, `R` are paths off `X` from `x`,
`y`, `z` to `u` that meet only at `u`, and `c ∪ P ∪ Q` and `b ∪ P ∪ R` are balanced, so is
`a ∪ Q ∪ R`. -/
theorem mem_balanced_of_tripod (hX : Ω.graph.IsLabelledTriangle a b c x y z)
    (hbal : ({a, b, c} : Set E) ∈ Ω.balanced) {u : V} {P : Ω.graph.Walk x u}
    {Q : Ω.graph.Walk y u} {R : Ω.graph.Walk z u} (hT : IsTripod {a, b, c} P Q R)
    (hc : insert c (P.edgeSet ∪ Q.edgeSet) ∈ Ω.balanced)
    (hb : insert b (P.edgeSet ∪ R.edgeSet) ∈ Ω.balanced) :
    insert a (Q.edgeSet ∪ R.edgeSet) ∈ Ω.balanced := by
  by_cases huz : u = z
  · have hT' : IsTripod {a, c, b} P R Q := by
      rw [Set.pair_comm]
      exact hT.swap₂₃
    have := mem_balanced_of_tripod_of_ne hX.swap₂₃ (by rwa [Set.pair_comm]) hT'
      (huz ▸ hX.ne_yz.symm) hb hc
    rwa [Set.union_comm] at this
  · exact mem_balanced_of_tripod_of_ne hX hbal hT huz hc hb

end BiasedGraph

namespace OrientedGraph.IsExchange

variable {V W E : Type*} {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E} {x y z : V}

/-- §2.4, Proposition 2.6 at the centre of the star: for `X` balanced in `Ω`, a theta of `S` from
`v` with two of its cycles in `Δ_X B` has the third. -/
theorem theta_of_eq {Ω : BiasedGraph V E} (h : IsExchange Ω.graph S ι v a b c x y z)
    (hbal : ({a, b, c} : Set E) ∈ Ω.balanced) {t : W} {P Q R : S.Walk v t} (hθ : IsTheta P Q R)
    (hPQ : P.edgeSet ∪ Q.edgeSet ∈ deltaBalanced Ω.balanced {a, b, c})
    (hPR : P.edgeSet ∪ R.edgeSet ∈ deltaBalanced Ω.balanced {a, b, c}) :
    Q.edgeSet ∪ R.edgeSet ∈ deltaBalanced Ω.balanced {a, b, c} := by
  obtain ⟨wP, eP, hjP, P₁, rfl⟩ := P.exists_eq_cons_of_ne hθ.ne
  obtain ⟨wQ, eQ, hjQ, Q₁, rfl⟩ := Q.exists_eq_cons_of_ne hθ.ne
  obtain ⟨wR, eR, hjR, R₁, rfl⟩ := R.exists_eq_cons_of_ne hθ.ne
  have hePX := h.mem_of_incident hjP.tail_or_head
  have hPQe : eP ≠ eQ := fun he =>
    hθ.openPQ.edges eP List.mem_cons_self (by rw [he]; exact List.mem_cons_self)
  have hPRe : eP ≠ eR := fun he =>
    hθ.openPR.edges eP List.mem_cons_self (by rw [he]; exact List.mem_cons_self)
  have hQRe : eQ ≠ eR := fun he =>
    hθ.openQR.edges eQ List.mem_cons_self (by rw [he]; exact List.mem_cons_self)
  obtain ⟨e₀, x', y', z', h', hX⟩ := h.exists_relabel (h.mem_of_incident hjQ.tail_or_head)
    (h.mem_of_incident hjR.tail_or_head) hQRe
  obtain rfl : eP = e₀ := by
    rw [← hX] at hePX
    rcases hePX with he | he | he
    · exact he
    · exact absurd he hPQe
    · exact absurd he hPRe
  rw [← hX] at hbal hPQ hPR ⊢
  obtain rfl : wP = ι x' := hjP.eq_of_joins h'.joins_a
  obtain rfl : wQ = ι y' := hjQ.eq_of_joins h'.joins_b
  obtain rfl : wR = ι z' := hjR.eq_of_joins h'.joins_c
  obtain ⟨hvP, hP₁⟩ := Walk.isPath_cons.1 hθ.pathP
  obtain ⟨hvQ, hQ₁⟩ := Walk.isPath_cons.1 hθ.pathQ
  obtain ⟨hvR, hR₁⟩ := Walk.isPath_cons.1 hθ.pathR
  obtain ⟨u, P₀, rfl, hP₀e, hP₀s⟩ := h'.exists_lift_of_not_mem P₁ hvP
  obtain ⟨uQ, Q₀, huQ, hQ₀e, hQ₀s⟩ := h'.exists_lift_of_not_mem Q₁ hvQ
  obtain ⟨uR, R₀, huR, hR₀e, hR₀s⟩ := h'.exists_lift_of_not_mem R₁ hvR
  obtain rfl : u = uQ := (h.inj huQ).symm
  obtain rfl : u = uR := (h.inj huR).symm
  have hmem : ∀ {s₀ : V} {s : W} {p : Ω.graph.Walk s₀ u} {q : S.Walk s (ι u)},
      p.support.map ι = q.support → ∀ w ∈ p.support, ι w ∈ q.support := fun hpq w hw => by
    rw [← hpq]
    exact List.mem_map_of_mem hw
  have hend : ∀ {w : V} {s : W} {q : S.Walk s (ι u)}, v ∉ q.support → ι w ∈ q.support →
      ι w = v ∨ ι w = ι u → w = u := fun hv hw h₀ =>
    h.inj (h₀.resolve_left fun he => hv (he ▸ hw))
  have hT : IsTripod {eP, eQ, eR} P₀ Q₀ R₀ :=
    ⟨List.Nodup.of_map ι (by rw [hP₀s]; exact hP₁), List.Nodup.of_map ι (by rw [hQ₀s]; exact hQ₁),
      List.Nodup.of_map ι (by rw [hR₀s]; exact hR₁),
      fun g hg => h'.not_mem_of_not_mem_support hvP (hP₀e ▸ hg),
      fun g hg => h'.not_mem_of_not_mem_support hvQ (hQ₀e ▸ hg),
      fun g hg => h'.not_mem_of_not_mem_support hvR (hR₀e ▸ hg),
      fun g hg hg' => hθ.openPQ.edges g (List.mem_cons_of_mem _ (hP₀e ▸ hg))
        (List.mem_cons_of_mem _ (hQ₀e ▸ hg')),
      fun g hg hg' => hθ.openPR.edges g (List.mem_cons_of_mem _ (hP₀e ▸ hg))
        (List.mem_cons_of_mem _ (hR₀e ▸ hg')),
      fun g hg hg' => hθ.openQR.edges g (List.mem_cons_of_mem _ (hQ₀e ▸ hg))
        (List.mem_cons_of_mem _ (hR₀e ▸ hg')),
      fun w hw hw' => hend hvP (hmem hP₀s w hw) (hθ.openPQ.verts (ι w)
        (List.mem_cons_of_mem _ (hmem hP₀s w hw)) (List.mem_cons_of_mem _ (hmem hQ₀s w hw'))),
      fun w hw hw' => hend hvP (hmem hP₀s w hw) (hθ.openPR.verts (ι w)
        (List.mem_cons_of_mem _ (hmem hP₀s w hw)) (List.mem_cons_of_mem _ (hmem hR₀s w hw'))),
      fun w hw hw' => hend hvQ (hmem hQ₀s w hw) (hθ.openQR.verts (ι w)
        (List.mem_cons_of_mem _ (hmem hQ₀s w hw)) (List.mem_cons_of_mem _ (hmem hR₀s w hw')))⟩
  have eP' : (Walk.cons eP hjP P₁).edgeSet = insert eP P₀.edgeSet := by
    ext g
    simp [Walk.edgeSet, Walk.edges, hP₀e]
  have eQ' : (Walk.cons eQ hjQ Q₁).edgeSet = insert eQ Q₀.edgeSet := by
    ext g
    simp [Walk.edgeSet, Walk.edges, hQ₀e]
  have eR' : (Walk.cons eR hjR R₁).edgeSet = insert eR R₀.edgeSet := by
    ext g
    simp [Walk.edgeSet, Walk.edges, hR₀e]
  rw [eP', eQ'] at hPQ
  rw [eP', eR'] at hPR
  rw [eQ', eR', h'.mem_deltaBalanced_iff_of_pair hbal (Or.inr (Or.inl rfl)) (Or.inr (Or.inr rfl))
    hQRe (insert_union_insert_inter (Or.inr (Or.inl rfl)) (Or.inr (Or.inr rfl)) hT.offQ hT.offR),
    insert_union_insert_symmDiff (by rw [Set.insert_comm eP eQ, Set.pair_comm eP eR]) hPQe hPRe
      hT.offQ hT.offR]
  rw [h'.mem_deltaBalanced_iff_of_pair hbal (Or.inl rfl) (Or.inr (Or.inl rfl)) hPQe
    (insert_union_insert_inter (Or.inl rfl) (Or.inr (Or.inl rfl)) hT.offP hT.offQ),
    insert_union_insert_symmDiff rfl hPRe.symm hQRe.symm hT.offP hT.offQ] at hPQ
  rw [h'.mem_deltaBalanced_iff_of_pair hbal (Or.inl rfl) (Or.inr (Or.inr rfl)) hPRe
    (insert_union_insert_inter (Or.inl rfl) (Or.inr (Or.inr rfl)) hT.offP hT.offR),
    insert_union_insert_symmDiff (by rw [Set.pair_comm eQ eR]) hPQe.symm hQRe hT.offP
      hT.offR] at hPR
  exact Ω.mem_balanced_of_tripod h'.triangle hbal hT hPQ hPR

end OrientedGraph.IsExchange

example := insert_union_insert_inter (X := ({0, 1, 2} : Set (Fin 5))) (P := {3}) (Q := {4})
  (a := 0) (b := 1) (by simp) (by simp) (by simp) (by simp)

example := insert_union_insert_symmDiff (X := ({0, 1, 2} : Set (Fin 5))) (P := {3}) (Q := {4}) rfl
  (by decide) (by decide) (by simp) (by simp)

example := (K4.gainBiased (1 : Fin 6 → ℚˣ)).mem_balanced_of_tripod K4_isLabelledTriangle
  (gainBiased_one_isBalanced K4 _ K4_isLabelledTriangle.isCycle) K4_isTripod
  (gainBiased_one_isBalanced K4 _ K4_isCycle_spokesXY)
  (gainBiased_one_isBalanced K4 _ K4_isCycle_spokesXZ)

end Transformer.GainMatroid
