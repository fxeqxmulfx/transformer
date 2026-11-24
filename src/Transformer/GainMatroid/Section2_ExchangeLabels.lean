module

public import Transformer.GainMatroid.Section2_ExchangeCycles

/-!
# The cycles of an exchange pair through the triangle and the star

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. For an exchange pair `T`, `S` on
`X = {a, b, c}` (`IsExchange`), with the triangle `X` in `T` and the star `X` at `v` in `S`:

* for a cycle `C` of `T` meeting `X` in `a` alone, `C ∆ X` is a cycle of `S`
  (`IsExchange.isCycle_symmDiff_of_inter_eq_singleton`), and a cycle of `T` meeting `X` in `b` and
  `c` alone is a cycle of `S` (`IsExchange.isCycle_of_inter_eq_pair`);
* for a cycle `D` of `S` meeting `X` in `b` and `c` alone, `D ∆ X` is a cycle of `T`
  (`IsExchange.isCycle_symmDiff`).

Up to the labels of the triangle, a set of edges meets `X` in no edge, in `a` alone, in `b` and `c`
alone, or in all of `X` (`IsExchange.inter_cases`), and a cycle of `S` meets `X` in no edge or in
`b` and `c` alone (`IsExchange.inter_cases_of_isCycle`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

namespace OrientedGraph

variable {V W E : Type*}


/-- `(insert a R) ∆ {a, b, c} = {b, c} ∪ R` for `R` off `{a, b, c}`. -/
theorem insert_symmDiff_triple {a b c : E} {R : Set E} (hR : ∀ g ∈ R, g ∉ ({a, b, c} : Set E))
    (hab : a ≠ b) (hac : a ≠ c) : insert a R ∆ {a, b, c} = {b, c} ∪ R := by
  ext g
  by_cases hg : g ∈ R
  · have h' := hR g hg
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at h'
    simp [Set.mem_symmDiff, hg, h'.1, h'.2.1, h'.2.2]
  · rcases eq_or_ne g a with rfl | h₁
    · simp [Set.mem_symmDiff, hab, hac, hg]
    · simp [Set.mem_symmDiff, hg, h₁]

/-- `(insert b (insert c P)) ∆ {a, b, c} = insert a P` for `P` off `{a, b, c}`. -/
theorem insert_insert_symmDiff_triple {a b c : E} {P : Set E}
    (hP : ∀ g ∈ P, g ∉ ({a, b, c} : Set E)) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    insert b (insert c P) ∆ {a, b, c} = insert a P := by
  ext g
  by_cases hg : g ∈ P
  · have h' := hP g hg
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at h'
    simp [Set.mem_symmDiff, hg, h'.1, h'.2.1, h'.2.2]
  · rcases eq_or_ne g a with rfl | h₁
    · simp [Set.mem_symmDiff, hab, hac, hg]
    rcases eq_or_ne g b with rfl | h₂
    · simp [Set.mem_symmDiff, hbc, h₁, hg]
    rcases eq_or_ne g c with rfl | h₃
    · simp [Set.mem_symmDiff, h₁, hg]
    · simp [Set.mem_symmDiff, hg, h₁, h₂, h₃]

/-- If `C` meets `{a, b, c}` in `a` alone, then `C ∆ {a, b, c}` meets it in `b` and `c` alone. -/
theorem symmDiff_inter_of_inter_eq_singleton {a b c : E} {C : Set E} (hab : a ≠ b) (hac : a ≠ c)
    (h : C ∩ {a, b, c} = {a}) : (C ∆ {a, b, c}) ∩ {a, b, c} = {b, c} := by
  rw [Set.inter_symmDiff_distrib_right, h, Set.inter_self]
  ext g
  rcases eq_or_ne g a with rfl | h₁
  · simp [Set.mem_symmDiff, hab, hac]
  · simp [Set.mem_symmDiff, h₁]

/-- If `D` meets `{a, b, c}` in `b` and `c` alone, then `D ∆ {a, b, c}` meets it in `a` alone. -/
theorem symmDiff_inter_of_inter_eq_pair {a b c : E} {D : Set E} (hab : a ≠ b) (hac : a ≠ c)
    (h : D ∩ {a, b, c} = {b, c}) : (D ∆ {a, b, c}) ∩ {a, b, c} = {a} := by
  rw [Set.inter_symmDiff_distrib_right, h, Set.inter_self]
  ext g
  rcases eq_or_ne g a with rfl | h₁
  · simp [Set.mem_symmDiff, hab, hac]
  · simp [Set.mem_symmDiff, h₁]
    tauto

namespace IsExchange

variable {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E}
  {x y z : V}

/-- A cycle `C` of `T` meeting `X` in `a` alone gives the cycle `C ∆ X` of `S`. -/
theorem isCycle_symmDiff_of_inter_eq_singleton (h : IsExchange T S ι v a b c x y z) {C : Set E}
    (hC : T.IsCycle C) (hCX : C ∩ {a, b, c} = {a}) : S.IsCycle (C ∆ {a, b, c}) := by
  have haC : a ∈ C := ((Set.ext_iff.1 hCX a).2 (Set.mem_singleton a)).1
  obtain ⟨R, hR, haR, rfl⟩ := hC.exists_isPath_insert_of_joins h.triangle.joins_a haC
  have hRX : ∀ g ∈ R.edges, g ∉ ({a, b, c} : Set E) := fun g hg hgX => by
    have hg' := (Set.ext_iff.1 hCX g).1 ⟨Or.inr hg, hgX⟩
    rw [Set.mem_singleton_iff] at hg'
    exact haR (hg' ▸ hg)
  obtain ⟨q, hq, hvq, hqR⟩ := h.exists_map_path hR hRX (h.ne_of_joins h.triangle.joins_a)
  have hS := isCycle_pair_union h.joins_b.symm h.joins_c.symm h.triangle.ne_bc hq hvq
  rwa [hqR, ← insert_symmDiff_triple (fun g hg => hRX g hg) h.triangle.ne_ab
    h.triangle.ne_ac] at hS

/-- A cycle of `T` meeting `X` in `b` and `c` alone is a cycle of `S`. -/
theorem isCycle_of_inter_eq_pair (h : IsExchange T S ι v a b c x y z) {C : Set E}
    (hC : T.IsCycle C) (hCX : C ∩ {a, b, c} = {b, c}) : S.IsCycle C := by
  obtain ⟨P, hP, -, hPX, rfl⟩ := h.triangle.exists_isPath_of_inter_eq_pair hC hCX
  obtain ⟨q, hq, hvq, hqP⟩ := h.exists_map_path hP hPX (h.ne_of_joins h.triangle.joins_b.symm)
  have hS := isCycle_pair_union h.joins_c.symm h.joins_b.symm h.triangle.ne_bc.symm hq hvq
  rwa [hqP, Set.insert_union, Set.singleton_union, Set.insert_comm] at hS

/-- A cycle `D` of `S` meeting `X` in `b` and `c` alone gives the cycle `D ∆ X` of `T`. -/
theorem isCycle_symmDiff (h : IsExchange T S ι v a b c x y z) {D : Set E} (hD : S.IsCycle D)
    (hDX : D ∩ {a, b, c} = {b, c}) : T.IsCycle (D ∆ {a, b, c}) := by
  have hbD : b ∈ D := ((Set.ext_iff.1 hDX b).2 (Or.inl rfl)).1
  have hcD : c ∈ D := ((Set.ext_iff.1 hDX c).2 (Or.inr rfl)).1
  obtain ⟨P, hP, hvP, rfl⟩ := hD.exists_isPath_of_joins h.joins_b.symm h.joins_c.symm
    h.triangle.ne_bc hbD hcD
  have hPX : ∀ g ∈ P.edges, g ∉ ({a, b, c} : Set E) := fun g hg hgX =>
    hvP (Walk.mem_support_of_mem_edges hg (h.incident_of_mem hgX))
  obtain ⟨p, hp, hpP⟩ := h.exists_lift_path hP hPX
  have haP : a ∉ p.edges := fun ha =>
    hPX a (show a ∈ P.edgeSet by rw [← hpP]; exact ha) (Or.inl rfl)
  have hT := isCycle_insert_of_isPath h.triangle.joins_a.symm hp haP
  rwa [hpP, ← insert_insert_symmDiff_triple (fun g hg => hPX g hg) h.triangle.ne_ab
    h.triangle.ne_ac h.triangle.ne_bc] at hT

/-- Up to the labels of the triangle, a set of edges meets `X` in no edge, in `a` alone, in `b`
and `c` alone, or in all of `X`. -/
theorem inter_cases (h : IsExchange T S ι v a b c x y z) (C : Set E) :
    C ∩ {a, b, c} = ∅ ∨ {a, b, c} ⊆ C ∨
      ∃ (a' b' c' : E) (x' y' z' : V), IsExchange T S ι v a' b' c' x' y' z' ∧
        ({a', b', c'} : Set E) = {a, b, c} ∧
          (C ∩ {a, b, c} = {a'} ∨ C ∩ {a, b, c} = {b', c'}) := by
  have hab := h.triangle.ne_ab
  have hac := h.triangle.ne_ac
  have hbc := h.triangle.ne_bc
  have e₁ : ({b, a, c} : Set E) = {a, b, c} := Set.insert_comm b a {c}
  have e₂ : ({c, a, b} : Set E) = {a, b, c} := by
    ext g
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
    tauto
  by_cases ha : a ∈ C <;> by_cases hb : b ∈ C <;> by_cases hc : c ∈ C
  · exact Or.inr (Or.inl (by rintro g (rfl | rfl | rfl) <;> assumption))
  · exact Or.inr (Or.inr ⟨c, a, b, z, x, y, h.swap₂₃.swap₁₂, e₂, Or.inr (by ext g; aesop)⟩)
  · exact Or.inr (Or.inr ⟨b, a, c, y, x, z, h.swap₁₂, e₁, Or.inr (by ext g; aesop)⟩)
  · exact Or.inr (Or.inr ⟨a, b, c, x, y, z, h, rfl, Or.inl (by ext g; aesop)⟩)
  · exact Or.inr (Or.inr ⟨a, b, c, x, y, z, h, rfl, Or.inr (by ext g; aesop)⟩)
  · exact Or.inr (Or.inr ⟨b, a, c, y, x, z, h.swap₁₂, e₁, Or.inl (by ext g; aesop)⟩)
  · exact Or.inr (Or.inr ⟨c, a, b, z, x, y, h.swap₂₃.swap₁₂, e₂, Or.inl (by ext g; aesop)⟩)
  · exact Or.inl (by ext g; aesop)

/-- A cycle of `S` meets `X` in no edge or, up to the labels of the triangle, in `b` and `c`
alone: its edges at `v` are two edges of `X`. -/
theorem inter_cases_of_isCycle (h : IsExchange T S ι v a b c x y z) {D : Set E}
    (hD : S.IsCycle D) :
    D ∩ {a, b, c} = ∅ ∨ ∃ (a' b' c' : E) (x' y' z' : V), IsExchange T S ι v a' b' c' x' y' z' ∧
      ({a', b', c'} : Set E) = {a, b, c} ∧ D ∩ {a, b, c} = {b', c'} := by
  rcases h.inter_cases D with h0 | hXD | ⟨a', b', c', x', y', z', h', hX, h1 | h2⟩
  · exact Or.inl h0
  · exfalso
    obtain ⟨g, -, -, -, hall⟩ := hD.exists_partner (hXD (Or.inl rfl))
      (h.isLink_of_mem (Or.inl rfl)) (h.incident_of_mem (Or.inl rfl))
    rcases hall b (hXD (Or.inr (Or.inl rfl))) (h.incident_of_mem (Or.inr (Or.inl rfl)))
      with hb | hb
    · exact h.triangle.ne_ab hb.symm
    rcases hall c (hXD (Or.inr (Or.inr rfl))) (h.incident_of_mem (Or.inr (Or.inr rfl)))
      with hc | hc
    · exact h.triangle.ne_ac hc.symm
    · exact h.triangle.ne_bc (hb.trans hc.symm)
  · exfalso
    have ha' : a' ∈ D ∩ {a, b, c} := by
      rw [h1]
      exact Set.mem_singleton a'
    obtain ⟨g, hgD, hga, hgv, -⟩ := hD.exists_partner ha'.1 (h'.isLink_of_mem (Or.inl rfl))
      (h'.incident_of_mem (Or.inl rfl))
    have hgX := h'.mem_of_incident hgv
    rw [hX] at hgX
    exact hga ((Set.ext_iff.1 h1 g).1 ⟨hgD, hgX⟩)
  · exact Or.inr ⟨a', b', c', x', y', z', h', hX, h2⟩

end IsExchange

example := insert_symmDiff_triple (a := (5 : Fin 6)) (b := 4) (c := 3) (R := ∅) (by simp)
  (by decide) (by decide)
example := insert_insert_symmDiff_triple (a := (5 : Fin 6)) (b := 4) (c := 3) (P := ∅) (by simp)
  (by decide) (by decide) (by decide)
example := symmDiff_inter_of_inter_eq_singleton (b := 4) (c := 3) (by decide) (by decide)
  K4_inter_125
example := symmDiff_inter_of_inter_eq_pair (a := 5) (by decide) (by decide) K4_inter_1243
example := K4_isLabelledTriangle.isExchange.isCycle_symmDiff_of_inter_eq_singleton K4_isCycle_125
  K4_inter_125
example := K4_isLabelledTriangle.isExchange.isCycle_of_inter_eq_pair K4_isCycle_1243 K4_inter_1243
example := K4_isLabelledTriangle.isExchange.isCycle_symmDiff
  (K4_isLabelledTriangle.isExchange.isCycle_of_inter_eq_pair K4_isCycle_1243 K4_inter_1243)
  K4_inter_1243
example := K4_isLabelledTriangle.isExchange.inter_cases ∅
example := K4_isLabelledTriangle.isExchange.inter_cases_of_isCycle
  (K4_isLabelledTriangle.isExchange.isCycle_of_inter_eq_pair K4_isCycle_1243 K4_inter_1243)

end OrientedGraph

end Transformer.GainMatroid
