module

public import Transformer.GainMatroid.Section2_BalancedTripods
public import Transformer.GainMatroid.Section2_NablaDetours

/-!
# Thetas of `∇_Y G` matched by thetas of `G`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.7. A theta of `T`
**matches** a theta of a biased graph `Ω` under `B'` (`MatchesTheta`) if each of its cycles is in
`B'` exactly when the corresponding cycle of the theta of `Ω` is balanced. Then `B'` has the theta
property on it (`MatchesTheta.theta`), and matching survives permuting and reversing the paths.

Let `T`, `S` be an exchange pair on `Y = {a, b, c}` (`IsExchange`), with the triangle `Y` in `T` and
the star `Y` at `v` in `S`. A tripod of `T` off `Y` from the corners `x`, `y`, `z` gives a theta of
`S` between `v` and its centre through `a`, `b` and `c` (`IsExchange.exists_isTheta_of_isTripod`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V W E : Type*} {T : OrientedGraph V E}

/-- A theta `P`, `Q`, `R` of `T` **matches** a theta of `Ω` under `B'` (§2.4): each of its cycles
is in `B'` exactly when the corresponding cycle of the theta of `Ω` is balanced. -/
def MatchesTheta (B' : Set (Set E)) (Ω : BiasedGraph W E) {s t : V} (P Q R : T.Walk s t) :
    Prop :=
  ∃ (s' t' : W) (P' Q' R' : Ω.graph.Walk s' t'), IsTheta P' Q' R' ∧
    (P.edgeSet ∪ Q.edgeSet ∈ B' ↔ P'.edgeSet ∪ Q'.edgeSet ∈ Ω.balanced) ∧
    (P.edgeSet ∪ R.edgeSet ∈ B' ↔ P'.edgeSet ∪ R'.edgeSet ∈ Ω.balanced) ∧
    (Q.edgeSet ∪ R.edgeSet ∈ B' ↔ Q'.edgeSet ∪ R'.edgeSet ∈ Ω.balanced)

namespace MatchesTheta

variable {B' : Set (Set E)} {Ω : BiasedGraph W E} {s t : V} {P Q R : T.Walk s t}

/-- `B'` has the theta property on a theta that matches one of `Ω`. -/
theorem theta (h : MatchesTheta B' Ω P Q R) (hPQ : P.edgeSet ∪ Q.edgeSet ∈ B')
    (hPR : P.edgeSet ∪ R.edgeSet ∈ B') : Q.edgeSet ∪ R.edgeSet ∈ B' := by
  obtain ⟨s', t', P', Q', R', hθ, h₁, h₂, h₃⟩ := h
  exact h₃.2 (Ω.theta hθ (h₁.1 hPQ) (h₂.1 hPR))

/-- Exchanging the first two paths keeps a match. -/
theorem swap₁₂ (h : MatchesTheta B' Ω P Q R) : MatchesTheta B' Ω Q P R := by
  obtain ⟨s', t', P', Q', R', hθ, h₁, h₂, h₃⟩ := h
  refine ⟨s', t', Q', P', R', hθ.swap₁₂, ?_, h₃, h₂⟩
  rw [Set.union_comm Q.edgeSet, Set.union_comm Q'.edgeSet]
  exact h₁

/-- Exchanging the last two paths keeps a match. -/
theorem swap₂₃ (h : MatchesTheta B' Ω P Q R) : MatchesTheta B' Ω P R Q := by
  obtain ⟨s', t', P', Q', R', hθ, h₁, h₂, h₃⟩ := h
  refine ⟨s', t', P', R', Q', hθ.swap₂₃, h₂, h₁, ?_⟩
  rw [Set.union_comm R.edgeSet, Set.union_comm R'.edgeSet]
  exact h₃

/-- A match of the reversed paths is a match of the paths. -/
theorem of_reverse (h : MatchesTheta B' Ω P.reverse Q.reverse R.reverse) :
    MatchesTheta B' Ω P Q R := by
  simpa only [MatchesTheta, Walk.edgeSet_reverse] using h

end MatchesTheta

namespace IsExchange

variable {S : OrientedGraph W E} {ι : V → W} {v : W} {a b c : E} {x y z : V}

/-- A path of `T` from a vertex not mapped to `v`, carried to `S` after an edge from `v`, is a path
of `S`. -/
theorem isPath_cons_of_map (h : IsExchange T S ι v a b c x y z) {s t : V} {P : T.Walk s t}
    (hP : P.IsPath) (hs : ι s ≠ v) {e : E} {he : S.Joins e v (ι s)} {P' : S.Walk (ι s) (ι t)}
    (hP' : P'.support = P.support.map ι) : (Walk.cons e he P').IsPath := by
  refine Walk.isPath_cons.2 ⟨fun hv => ?_, ?_⟩
  · rw [hP'] at hv
    obtain ⟨u, hu, hvu⟩ := List.mem_map.1 hv
    exact h.ne_of_mem_support P hs hu hvu
  · unfold Walk.IsPath
    rw [hP']
    exact List.Nodup.map h.inj hP

/-- Paths of `T` meeting only at their common end, carried to `S` after distinct edges from `v`
that are not on the other path, are openly disjoint. -/
theorem openly_cons_of_map (h : IsExchange T S ι v a b c x y z) {p q u : V} {P : T.Walk p u}
    {Q : T.Walk q u} {e f : E} {he : S.Joins e v (ι p)} {hf : S.Joins f v (ι q)}
    {P' : S.Walk (ι p) (ι u)} {Q' : S.Walk (ι q) (ι u)} (hP'e : P'.edges = P.edges)
    (hP's : P'.support = P.support.map ι) (hQ'e : Q'.edges = Q.edges)
    (hQ's : Q'.support = Q.support.map ι) (hef : e ≠ f) (heQ : e ∉ Q.edges) (hfP : f ∉ P.edges)
    (hPQ : ∀ g ∈ P.edges, g ∉ Q.edges) (hPQu : ∀ w ∈ P.support, w ∈ Q.support → w = u) :
    Walk.Openly (Walk.cons e he P') (Walk.cons f hf Q') := by
  refine ⟨fun g hg hg' => ?_, fun w hw hw' => ?_⟩
  · simp only [Walk.edges, hP'e, hQ'e, List.mem_cons] at hg hg'
    rcases hg with rfl | hg <;> rcases hg' with rfl | hg'
    · exact hef rfl
    · exact heQ hg'
    · exact hfP hg
    · exact hPQ g hg hg'
  · simp only [Walk.support, hP's, hQ's, List.mem_cons] at hw hw'
    rcases hw with rfl | hw
    · exact Or.inl rfl
    rcases hw' with rfl | hw'
    · exact Or.inl rfl
    obtain ⟨u₁, hu₁, rfl⟩ := List.mem_map.1 hw
    obtain ⟨u₂, hu₂, he₂⟩ := List.mem_map.1 hw'
    obtain rfl := h.inj he₂
    exact Or.inr (congrArg ι (hPQu u₂ hu₁ hu₂))

/-- §2.4: a tripod of `T` off `Y` from `x`, `y`, `z` to `u` gives a theta of `S` between `v` and
`u`, through `a`, `b` and `c`. -/
theorem exists_isTheta_of_isTripod (h : IsExchange T S ι v a b c x y z) {u : V}
    {A : T.Walk x u} {B : T.Walk y u} {C : T.Walk z u} (hT : IsTripod {a, b, c} A B C) :
    ∃ A' B' C' : S.Walk v (ι u), IsTheta A' B' C' ∧ A'.edgeSet = insert a A.edgeSet ∧
      B'.edgeSet = insert b B.edgeSet ∧ C'.edgeSet = insert c C.edgeSet := by
  have hx := h.ne_of_joins h.triangle.joins_b
  have hy := h.ne_of_joins h.triangle.joins_a
  have hz := h.ne_of_joins h.triangle.joins_a.symm
  obtain ⟨A₀, hAe, hAs⟩ := A.exists_map_vertex ι fun g hg => h.agree g (hT.offP g hg)
  obtain ⟨B₀, hBe, hBs⟩ := B.exists_map_vertex ι fun g hg => h.agree g (hT.offQ g hg)
  obtain ⟨C₀, hCe, hCs⟩ := C.exists_map_vertex ι fun g hg => h.agree g (hT.offR g hg)
  refine ⟨.cons a h.joins_a A₀, .cons b h.joins_b B₀, .cons c h.joins_c C₀,
    ⟨(h.ne_of_mem_support A hx A.end_mem_support).symm, h.isPath_cons_of_map hT.pathP hx hAs,
      h.isPath_cons_of_map hT.pathQ hy hBs, h.isPath_cons_of_map hT.pathR hz hCs,
      h.openly_cons_of_map hAe hAs hBe hBs h.triangle.ne_ab (fun ha => hT.offQ a ha (by simp))
        (fun hb => hT.offP b hb (by simp)) hT.edgesPQ hT.vertsPQ,
      h.openly_cons_of_map hAe hAs hCe hCs h.triangle.ne_ac (fun ha => hT.offR a ha (by simp))
        (fun hc => hT.offP c hc (by simp)) hT.edgesPR hT.vertsPR,
      h.openly_cons_of_map hBe hBs hCe hCs h.triangle.ne_bc (fun hb => hT.offR b hb (by simp))
        (fun hc => hT.offQ c hc (by simp)) hT.edgesQR hT.vertsQR⟩, ?_, ?_, ?_⟩
  all_goals
    rw [Walk.edgeSet_cons]
    simp only [Walk.edgeSet, hAe, hBe, hCe]

end IsExchange

end OrientedGraph

end Transformer.GainMatroid
