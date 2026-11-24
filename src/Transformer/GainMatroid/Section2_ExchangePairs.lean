module

public import Transformer.GainMatroid.Section2_CycleEars

/-!
# Exchange pairs

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. A Δ-Y exchange replaces the triangle
`X = {a, b, c}` of a graph, with `a = yz`, `b = xz` and `c = xy`, by a star at a new vertex `v`,
with `a = vx`, `b = vy` and `c = vz`; a Y-Δ exchange is the reverse.
`IsExchange T S ι v a b c x y z` says that the triangle graph `T` and the star graph `S` are such a
pair: the vertices of `T` embed in those of `S` by `ι`, every edge outside `X` keeps its ends, and
`v` is no end of an edge of `T`.

Both exchanges of §2.4 give exchange pairs: `G` and `Δ_X G` (`IsLabelledTriangle.isExchange`), and
`∇_Y G` and `G` (`IsLabelledClaw.isExchange`), where the centre of the star stays in `∇_Y G` as an
isolated vertex. The pair is symmetric in the labels of the triangle (`IsExchange.swap₁₂`,
`IsExchange.swap₂₃`), so a statement about one or two edges of `X` need only be proved for one
choice of them.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V W E : Type*}

section Triangle

variable {G : OrientedGraph V E} {a b c : E} {x y z : V}

/-- Exchanging the labels `a, x` with `b, y` of a labelled triangle. -/
theorem IsLabelledTriangle.swap₁₂ (h : G.IsLabelledTriangle a b c x y z) :
    G.IsLabelledTriangle b a c y x z :=
  ⟨h.joins_b, h.joins_a, h.joins_c.symm, h.ne_ab.symm, h.ne_bc, h.ne_ac, h.ne_xy.symm, h.ne_yz,
    h.ne_xz⟩

/-- Exchanging the labels `b, y` with `c, z` of a labelled triangle. -/
theorem IsLabelledTriangle.swap₂₃ (h : G.IsLabelledTriangle a b c x y z) :
    G.IsLabelledTriangle a c b x z y :=
  ⟨h.joins_a.symm, h.joins_c, h.joins_b, h.ne_ac, h.ne_ab, h.ne_bc.symm, h.ne_xz, h.ne_xy,
    h.ne_yz.symm⟩

end Triangle

/-- The triangle graph `T` and the star graph `S` of a Δ-Y or Y-Δ exchange on `X = {a, b, c}`
(§2.4): `a = yz`, `b = xz` and `c = xy` in `T`, and `a = vx`, `b = vy` and `c = vz` in `S`. -/
structure IsExchange (T : OrientedGraph V E) (S : OrientedGraph W E) (ι : V → W) (v : W)
    (a b c : E) (x y z : V) : Prop where
  triangle : T.IsLabelledTriangle a b c x y z
  inj : Function.Injective ι
  joins_a : S.Joins a v (ι x)
  joins_b : S.Joins b v (ι y)
  joins_c : S.Joins c v (ι z)
  /-- The edges outside `X` keep their ends. -/
  agree : ∀ e ∉ ({a, b, c} : Set E), S.tail e = ι (T.tail e) ∧ S.head e = ι (T.head e)
  /-- `v` is no tail of an edge of `T`. -/
  tail_ne : ∀ e, ι (T.tail e) ≠ v
  /-- `v` is no head of an edge of `T`. -/
  head_ne : ∀ e, ι (T.head e) ≠ v

namespace IsExchange

variable {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E}
  {x y z : V}

/-- Exchanging the labels `a, x` with `b, y` of an exchange pair. -/
theorem swap₁₂ (h : IsExchange T S ι v a b c x y z) : IsExchange T S ι v b a c y x z :=
  ⟨h.triangle.swap₁₂, h.inj, h.joins_b, h.joins_a, h.joins_c,
    fun e he => h.agree e fun h' => he (by rwa [Set.insert_comm]), h.tail_ne, h.head_ne⟩

/-- Exchanging the labels `b, y` with `c, z` of an exchange pair. -/
theorem swap₂₃ (h : IsExchange T S ι v a b c x y z) : IsExchange T S ι v a c b x z y :=
  ⟨h.triangle.swap₂₃, h.inj, h.joins_a, h.joins_c, h.joins_b,
    fun e he => h.agree e fun h' => he (by rwa [Set.pair_comm]), h.tail_ne, h.head_ne⟩

/-- An end of an edge of `T` is not mapped to `v`. -/
theorem ne_of_joins (h : IsExchange T S ι v a b c x y z) {e : E} {s t : V} (he : T.Joins e s t) :
    ι s ≠ v := by
  rcases he with ⟨h₁, -⟩ | ⟨-, h₂⟩
  · exact h₁ ▸ h.tail_ne e
  · exact h₂ ▸ h.head_ne e

/-- No vertex of a walk of `T` from a vertex not mapped to `v` is mapped to `v`. -/
theorem ne_of_mem_support (h : IsExchange T S ι v a b c x y z) {s t u : V} (p : T.Walk s t)
    (hs : ι s ≠ v) (hu : u ∈ p.support) : ι u ≠ v := by
  by_cases hus : u = s
  · rwa [hus]
  obtain ⟨g, -, hg | hg⟩ := p.exists_mem_edges_of_mem_support hu hus
  · exact hg ▸ h.tail_ne g
  · exact hg ▸ h.head_ne g

/-- An edge of `S` at `v` is an edge of `X`. -/
theorem mem_of_incident (h : IsExchange T S ι v a b c x y z) {e : E}
    (he : S.tail e = v ∨ S.head e = v) : e ∈ ({a, b, c} : Set E) := by
  by_contra hX
  obtain ⟨h₁, h₂⟩ := h.agree e hX
  rcases he with he | he
  · exact h.tail_ne e (h₁.symm.trans he)
  · exact h.head_ne e (h₂.symm.trans he)

/-- The edges of `X` are links of `S`. -/
theorem isLink_of_mem (h : IsExchange T S ι v a b c x y z) {e : E}
    (he : e ∈ ({a, b, c} : Set E)) : S.IsLink e := by
  have hj : ∀ {s : V} {e' : E}, S.Joins e' v (ι s) → ι s ≠ v → S.IsLink e' := by
    rintro s e' (⟨h₁, h₂⟩ | ⟨h₁, h₂⟩) hs hl
    · exact hs (h₂.symm.trans (hl.symm.trans h₁))
    · exact hs (h₁.symm.trans (hl.trans h₂))
  rcases he with rfl | rfl | rfl
  · exact hj h.joins_a (h.ne_of_joins h.triangle.joins_b)
  · exact hj h.joins_b (h.ne_of_joins h.triangle.joins_a)
  · exact hj h.joins_c (h.ne_of_joins h.triangle.joins_a.symm)

/-- The edges of `X` are at `v` in `S`. -/
theorem incident_of_mem (h : IsExchange T S ι v a b c x y z) {e : E}
    (he : e ∈ ({a, b, c} : Set E)) : S.tail e = v ∨ S.head e = v := by
  rcases he with rfl | rfl | rfl
  · exact h.joins_a.tail_or_head
  · exact h.joins_b.tail_or_head
  · exact h.joins_c.tail_or_head

end IsExchange

open Classical in
/-- `G` and `Δ_X G` are an exchange pair (§2.4), with the new vertex `none`. -/
theorem IsLabelledTriangle.isExchange {G : OrientedGraph V E} {a b c : E} {x y z : V}
    (h : G.IsLabelledTriangle a b c x y z) :
    IsExchange G (G.deltaGraph a b c x y z) some none a b c x y z := by
  refine ⟨h, Option.some_injective _, Or.inr ⟨?_, ?_⟩, Or.inr ⟨?_, ?_⟩, Or.inr ⟨?_, ?_⟩,
    fun e he => ?_, fun _ => Option.some_ne_none _, fun _ => Option.some_ne_none _⟩
  · simp [deltaGraph]
  · simp [deltaGraph]
  · simp [deltaGraph, h.ne_ab.symm]
  · simp [deltaGraph]
  · simp [deltaGraph, h.ne_ac.symm, h.ne_bc.symm]
  · simp [deltaGraph]
  · simp only [Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at he
    simp [deltaGraph, he.1, he.2.1, he.2.2]

open Classical in
/-- `∇_Y G` and `G` are an exchange pair (§2.4): the centre `v` of the star stays in `∇_Y G` as an
isolated vertex. -/
theorem IsLabelledClaw.isExchange {G : OrientedGraph V E} {v : V} {a b c : E} {x y z : V}
    (h : G.IsLabelledClaw v a b c x y z) :
    IsExchange (G.nablaGraph a b c x y z) G id v a b c x y z := by
  have hoff : ∀ e ∉ ({a, b, c} : Set E), G.tail e ≠ v ∧ G.head e ≠ v := fun e he =>
    ⟨fun ht => he (h.degree e (Or.inl ht)), fun hh => he (h.degree e (Or.inr hh))⟩
  have hab := h.ne_ab
  have hac := h.ne_ac
  have hbc := h.ne_bc
  refine ⟨⟨Or.inl ⟨?_, ?_⟩, Or.inl ⟨?_, ?_⟩, Or.inl ⟨?_, ?_⟩, hab, hac, hbc, h.ne_xy, h.ne_xz,
    h.ne_yz⟩, Function.injective_id, h.joins_a, h.joins_b, h.joins_c, fun e he => ?_,
    fun e => ?_, fun e => ?_⟩
  · simp [nablaGraph]
  · simp [nablaGraph]
  · simp [nablaGraph, hab.symm]
  · simp [nablaGraph, hab.symm]
  · simp [nablaGraph, hac.symm, hbc.symm]
  · simp [nablaGraph, hac.symm, hbc.symm]
  · simp only [Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at he
    simp [nablaGraph, he.1, he.2.1, he.2.2]
  · by_cases hX : e ∈ ({a, b, c} : Set E)
    · rcases hX with rfl | rfl | rfl
      · simpa [nablaGraph] using h.ne_vy.symm
      · simpa [nablaGraph, hab.symm] using h.ne_vx.symm
      · simpa [nablaGraph, hac.symm, hbc.symm] using h.ne_vx.symm
    · simp only [Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at hX
      simpa [nablaGraph, hX.1, hX.2.1, hX.2.2] using
        (hoff e (by simp [hX.1, hX.2.1, hX.2.2])).1
  · by_cases hX : e ∈ ({a, b, c} : Set E)
    · rcases hX with rfl | rfl | rfl
      · simpa [nablaGraph] using h.ne_vz.symm
      · simpa [nablaGraph, hab.symm] using h.ne_vz.symm
      · simpa [nablaGraph, hac.symm, hbc.symm] using h.ne_vy.symm
    · simp only [Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at hX
      simpa [nablaGraph, hX.1, hX.2.1, hX.2.2] using
        (hoff e (by simp [hX.1, hX.2.1, hX.2.2])).2

example := triangle_isLabelledTriangle.swap₁₂
example := triangle_isLabelledTriangle.swap₂₃
example := triangle_isLabelledTriangle.isExchange.swap₁₂
example := triangle_isLabelledTriangle.isExchange.swap₂₃
example := triangle_isLabelledTriangle.isExchange.ne_of_joins triangle_isLabelledTriangle.joins_a
example := triangle_isLabelledTriangle.isExchange.ne_of_mem_support (u := 0) (Walk.nil 0)
  (Option.some_ne_none 0) (by simp [Walk.support])
example := triangle_isLabelledTriangle.isExchange.mem_of_incident (e := 1)
  (Or.inr (by simp [deltaGraph]))
example := triangle_isLabelledTriangle.isExchange.isLink_of_mem (Or.inl rfl)
example := triangle_isLabelledTriangle.isExchange.incident_of_mem (Or.inl rfl)
example := clawGraph_isLabelledClaw.isExchange

end OrientedGraph

end Transformer.GainMatroid
