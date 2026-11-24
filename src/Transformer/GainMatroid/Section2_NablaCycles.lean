module

public import Transformer.GainMatroid.Section2_ExchangeWalks

/-!
# Cycles of `∇_Y G` in `∇_Y B`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.7. Let `T`, `S` be
an exchange pair on `Y = {a, b, c}` (`IsExchange`), with the triangle `Y` in `T`, the star `Y` at
`v` in `S`, and `B` the balanced cycles of a biased graph on `S`. A cycle `D ≠ Y` of `T` in the
family `nablaFamily B Y` is a minimal nonempty member of it
(`IsExchange.minimal_of_mem_nablaFamily`), so the cycles of `T` in `∇_Y B` are `Y` and those in the
family (`IsExchange.mem_nablaBalanced_iff`). A cycle of `T` meeting `Y` in no edge or in two edges
is in `∇_Y B` exactly when it is in `B`; one meeting `Y` in one edge, exactly when its symmetric
difference with `Y` is in `B`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

open OrientedGraph

section Sets

variable {E : Type*} {B : Set (Set E)} {a b c : E} {D : Set E}

/-- A set off `Y = {a, b, c}` is in `nablaFamily B Y` exactly when it is in `B`. -/
theorem mem_nablaFamily_iff_of_inter_eq_empty (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (hD : D ∩ {a, b, c} = ∅) : D ∈ nablaFamily B {a, b, c} ↔ D ∈ B := by
  refine ⟨?_, fun hDB => Or.inl ⟨hDB, Or.inl (by rw [hD, Set.ncard_empty])⟩⟩
  rintro (⟨hDB, -⟩ | ⟨C, -, hC, rfl⟩)
  · exact hDB
  · have hYC : ({a, b, c} : Set E) ⊆ C := fun g hg => by
      by_contra hgC
      have hg' : g ∈ (C ∆ {a, b, c}) ∩ {a, b, c} := ⟨Set.mem_symmDiff.2 (Or.inr ⟨hg, hgC⟩), hg⟩
      rw [hD] at hg'
      exact hg'
    rw [Set.inter_eq_right.2 hYC, Set.ncard_eq_three.2 ⟨a, b, c, hab, hac, hbc, rfl⟩] at hC
    omega

/-- A set meeting `Y = {a, b, c}` in `b` and `c` alone is in `nablaFamily B Y` exactly when it is
in `B`. -/
theorem mem_nablaFamily_iff_of_inter_eq_pair (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (hD : D ∩ {a, b, c} = {b, c}) : D ∈ nablaFamily B {a, b, c} ↔ D ∈ B := by
  refine ⟨?_, fun hDB => Or.inl ⟨hDB, Or.inr (by rw [hD, Set.ncard_pair hbc])⟩⟩
  rintro (⟨hDB, -⟩ | ⟨C, -, hC, rfl⟩)
  · exact hDB
  · have h1 := symmDiff_inter_of_inter_eq_pair hab hac hD
    rw [symmDiff_symmDiff_cancel_right] at h1
    rw [h1, Set.ncard_singleton] at hC
    omega

/-- A set meeting `Y = {a, b, c}` in `a` alone is in `nablaFamily B Y` exactly when its symmetric
difference with `Y` is in `B`. -/
theorem mem_nablaFamily_iff_of_inter_eq_singleton (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (hD : D ∩ {a, b, c} = {a}) : D ∈ nablaFamily B {a, b, c} ↔ D ∆ {a, b, c} ∈ B := by
  refine ⟨?_, fun hDB => Or.inr ⟨_, hDB, ?_, (symmDiff_symmDiff_cancel_right _ _).symm⟩⟩
  · rintro (⟨-, h0 | h2⟩ | ⟨C, hCB, -, rfl⟩)
    · rw [hD, Set.ncard_singleton] at h0
      omega
    · rw [hD, Set.ncard_singleton] at h2
      omega
    · rwa [symmDiff_symmDiff_cancel_right]
  · rw [symmDiff_inter_of_inter_eq_singleton hab hac hD, Set.ncard_pair hbc]

end Sets

/-- `Y` is in `∇_Y B`. -/
theorem OrientedGraph.IsLabelledTriangle.mem_nablaBalanced {V E : Type*} {T : OrientedGraph V E}
    {a b c : E} {x y z : V} (h : T.IsLabelledTriangle a b c x y z) (B : Set (Set E)) :
    ({a, b, c} : Set E) ∈ nablaBalanced T B {a, b, c} :=
  ⟨h.isCycle, Or.inl rfl⟩

namespace OrientedGraph.IsExchange

variable {V W E : Type*} {T : OrientedGraph V E} {ι : V → W} {v : W} {a b c : E} {x y z : V}
  {Ω : BiasedGraph W E}

/-- §2.4, Proposition 2.7: a cycle `D ≠ Y` of `T` in `nablaFamily B Y` is a minimal nonempty member.
A member inside `D` is a cycle of `T`, or a cycle of `S` meeting `Y` in two edges of `D`, which
makes `D` a cycle of `S` too. -/
theorem minimal_of_mem_nablaFamily (h : IsExchange T Ω.graph ι v a b c x y z) {D : Set E}
    (hD : T.IsCycle D) (hDY : D ≠ {a, b, c}) (hF : D ∈ nablaFamily Ω.balanced {a, b, c}) :
    Minimal (fun C => C ∈ nablaFamily Ω.balanced {a, b, c} ∧ C.Nonempty) D := by
  refine ⟨⟨hF, hD.nonempty⟩, fun M hM hMD => le_of_eq ?_⟩
  rcases hM.1 with ⟨hMB, -⟩ | ⟨C, hCB, hC, rfl⟩
  · have hMS := Ω.isCycle_of_mem M hMB
    rcases h.inter_cases_of_isCycle hMS with h0 | ⟨a', b', c', x', y', z', h', hX, h2⟩
    · exact (hD.eq_of_subset ((h.isCycle_iff_of_disjoint
        (Set.disjoint_iff_inter_eq_empty.2 h0)).2 hMS) hMD).symm
    rw [← hX] at hDY h2
    have hb : b' ∈ M := ((Set.ext_iff.1 h2 b').2 (Or.inl rfl)).1
    have hc : c' ∈ M := ((Set.ext_iff.1 h2 c').2 (Or.inr rfl)).1
    have ha : a' ∉ D := fun ha => hDY (h'.triangle.eq_of_subset hD (Set.insert_subset ha
      (Set.insert_subset (hMD hb) (Set.singleton_subset_iff.2 (hMD hc)))))
    have hDX : D ∩ {a', b', c'} = {b', c'} := by
      ext g
      simp only [Set.mem_inter_iff, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro ⟨hg, rfl | hg' | hg'⟩
        · exact absurd hg ha
        · exact Or.inl hg'
        · exact Or.inr hg'
      · rintro (rfl | rfl)
        · exact ⟨hMD hb, Or.inr (Or.inl rfl)⟩
        · exact ⟨hMD hc, Or.inr (Or.inr rfl)⟩
    exact ((h'.isCycle_of_inter_eq_pair hD hDX).eq_of_subset hMS hMD).symm
  · have hCS := Ω.isCycle_of_mem C hCB
    rcases h.inter_cases_of_isCycle hCS with h0 | ⟨a', b', c', x', y', z', h', hX, h2⟩
    · rw [h0, Set.ncard_empty] at hC
      omega
    rw [← hX] at h2 hMD ⊢
    exact (hD.eq_of_subset (h'.isCycle_symmDiff hCS h2) hMD).symm

/-- §2.4: a cycle `D ≠ Y` of `T` is in `∇_Y B` exactly when it is in `nablaFamily B Y`. -/
theorem mem_nablaBalanced_iff (h : IsExchange T Ω.graph ι v a b c x y z) {D : Set E}
    (hD : T.IsCycle D) (hDY : D ≠ {a, b, c}) :
    D ∈ nablaBalanced T Ω.balanced {a, b, c} ↔ D ∈ nablaFamily Ω.balanced {a, b, c} :=
  ⟨fun hD' => hD'.2.elim (fun h' => absurd h' hDY) fun hm => hm.prop.1,
    fun hF => ⟨hD, Or.inr (h.minimal_of_mem_nablaFamily hD hDY hF)⟩⟩

/-- A cycle of `T` off `Y` is in `∇_Y B` exactly when it is in `B`. -/
theorem mem_nablaBalanced_iff_of_disjoint (h : IsExchange T Ω.graph ι v a b c x y z) {D : Set E}
    (hD : T.IsCycle D) (hDY : D ∩ {a, b, c} = ∅) :
    D ∈ nablaBalanced T Ω.balanced {a, b, c} ↔ D ∈ Ω.balanced := by
  have hne : D ≠ {a, b, c} := fun h' => by
    rw [h', Set.inter_self] at hDY
    exact (Set.ext_iff.1 hDY a).1 (Or.inl rfl)
  rw [h.mem_nablaBalanced_iff hD hne]
  exact mem_nablaFamily_iff_of_inter_eq_empty h.triangle.ne_ab h.triangle.ne_ac h.triangle.ne_bc hDY

/-- A cycle of `T` meeting `Y` in one edge is in `∇_Y B` exactly when its symmetric difference with
`Y` is in `B`. -/
theorem mem_nablaBalanced_iff_of_singleton (h : IsExchange T Ω.graph ι v a b c x y z) {D : Set E}
    (hD : T.IsCycle D) {e : E} (he : e ∈ ({a, b, c} : Set E)) (hDe : D ∩ {a, b, c} = {e}) :
    D ∈ nablaBalanced T Ω.balanced {a, b, c} ↔ D ∆ {a, b, c} ∈ Ω.balanced := by
  obtain ⟨b', c', x', y', z', h', hX⟩ : ∃ (b' c' : E) (x' y' z' : V),
      IsExchange T Ω.graph ι v e b' c' x' y' z' ∧ ({e, b', c'} : Set E) = {a, b, c} := by
    rcases he with rfl | rfl | rfl
    · exact ⟨_, _, _, _, _, h, rfl⟩
    · exact ⟨_, _, _, _, _, h.swap₁₂, Set.insert_comm _ _ _⟩
    · refine ⟨_, _, _, _, _, h.swap₂₃.swap₁₂, ?_⟩
      ext g
      simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
      tauto
  rw [← hX] at hDe ⊢
  have hDY : D ≠ {e, b', c'} := fun hDY => by
    rw [hDY, Set.inter_self] at hDe
    exact h'.triangle.ne_ab ((Set.ext_iff.1 hDe b').1 (Or.inr (Or.inl rfl))).symm
  rw [h'.mem_nablaBalanced_iff hD hDY]
  exact mem_nablaFamily_iff_of_inter_eq_singleton h'.triangle.ne_ab h'.triangle.ne_ac
    h'.triangle.ne_bc hDe

/-- A cycle of `T` meeting `Y` in two edges is in `∇_Y B` exactly when it is in `B`. -/
theorem mem_nablaBalanced_iff_of_pair (h : IsExchange T Ω.graph ι v a b c x y z) {D : Set E}
    (hD : T.IsCycle D) {e₁ e₂ : E} (he₁ : e₁ ∈ ({a, b, c} : Set E))
    (he₂ : e₂ ∈ ({a, b, c} : Set E)) (hne : e₁ ≠ e₂) (hDe : D ∩ {a, b, c} = {e₁, e₂}) :
    D ∈ nablaBalanced T Ω.balanced {a, b, c} ↔ D ∈ Ω.balanced := by
  obtain ⟨e₀, x', y', z', h', hX⟩ := h.exists_relabel he₁ he₂ hne
  rw [← hX] at hDe ⊢
  have hDY : D ≠ {e₀, e₁, e₂} := fun hDY => by
    rw [hDY, Set.inter_self] at hDe
    rcases (Set.ext_iff.1 hDe e₀).1 (Or.inl rfl) with h1 | h1
    · exact h'.triangle.ne_ab h1
    · exact h'.triangle.ne_ac h1
  rw [h'.mem_nablaBalanced_iff hD hDY]
  exact mem_nablaFamily_iff_of_inter_eq_pair h'.triangle.ne_ab h'.triangle.ne_ac
    h'.triangle.ne_bc hDe

/-- §2.4, Proposition 2.7 for a theta through the triangle: two cycles `D ≠ Y` and `D ∆ Y` of `T`
are both in `∇_Y B` or both not. One meets `Y` in one edge and the other in the other two. -/
theorem mem_nablaBalanced_iff_symmDiff (h : IsExchange T Ω.graph ι v a b c x y z) {D : Set E}
    (hD : T.IsCycle D) (hD' : T.IsCycle (D ∆ {a, b, c})) (hDY : D ≠ {a, b, c}) :
    D ∈ nablaBalanced T Ω.balanced {a, b, c} ↔
      D ∆ {a, b, c} ∈ nablaBalanced T Ω.balanced {a, b, c} := by
  rcases h.inter_cases D with h0 | hXD | ⟨a', b', c', x', y', z', h', hX, h1 | h2⟩
  · obtain ⟨g, hg⟩ := hD.nonempty
    have hgY : g ∉ ({a, b, c} : Set E) := fun hgY => (Set.ext_iff.1 h0 g).1 ⟨hg, hgY⟩
    have hYD := h.triangle.eq_of_subset hD' fun e he =>
      Set.mem_symmDiff.2 (Or.inr ⟨he, fun heD => (Set.ext_iff.1 h0 e).1 ⟨heD, he⟩⟩)
    exact (hgY (hYD ▸ Set.mem_symmDiff.2 (Or.inl ⟨hg, hgY⟩))).elim
  · exact absurd (h.triangle.eq_of_subset hD hXD) hDY
  · rw [← hX] at h1 hD' ⊢
    rw [h'.mem_nablaBalanced_iff_of_singleton hD (Or.inl rfl) h1,
      h'.mem_nablaBalanced_iff_of_pair hD' (Or.inr (Or.inl rfl)) (Or.inr (Or.inr rfl))
        h'.triangle.ne_bc (symmDiff_inter_of_inter_eq_singleton h'.triangle.ne_ab
          h'.triangle.ne_ac h1)]
  · rw [← hX] at h2 hD' ⊢
    rw [h'.mem_nablaBalanced_iff_of_pair hD (Or.inr (Or.inl rfl)) (Or.inr (Or.inr rfl))
        h'.triangle.ne_bc h2,
      h'.mem_nablaBalanced_iff_of_singleton hD' (Or.inl rfl) (symmDiff_inter_of_inter_eq_pair
        h'.triangle.ne_ab h'.triangle.ne_ac h2), symmDiff_symmDiff_cancel_right]

end OrientedGraph.IsExchange

end Transformer.GainMatroid
