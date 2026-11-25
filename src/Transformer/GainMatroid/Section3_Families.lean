module

public import Transformer.GainMatroid.Section2_LinkMinors
public import Transformer.GainMatroid.Section3_Prisms
public import Transformer.GainMatroid.Section3_SmallExamples

/-!
# The families `𝒢₀` and `𝒯₀`, isomorphisms and subdivisions

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1 and §3. An isomorphism of biased graphs
(`BiasedGraph.Iso`) is a graph isomorphism taking `B` to `S`. The set `𝒢₀` consists of "the four
biased `K₄`'s with no balanced triangle, the six biased `2C₃`'s with no balanced 2-cycle, and the
three biased tubes with no balanced 2-cycle" (§1.2). Link minors and subdivisions are taken up to
isomorphism, so membership is a condition on a biased graph whose graph is one of the models
`twoC3`, `tube`, `K4` of §4, in some orientation (`IsG0TwoC3`, `IsG0Tube`, `IsG0K4`). `𝒯₀` adds
`Pr`, `Pr₁` and `Pr₂` (`Section3_Prisms`).

A subdivision of `(G, B)` "is a biased graph `(H, S)` in which `H` is a subdivision of `G` and a
cycle `C` of `H` is in `S` if and only if its corresponding cycle `C'` of `G` is in `B`" (§3.1).
`SubdivisionMap S Ω` presents a biased subgraph of `Ω` that is a subdivision of `S`; it is all of
`Ω` when it is `IsSpanning`.

`2C₃ \ e` (`twoC3Minus`) and `D_{1,0}` (`IsD10`) are the graphs of Proposition 3.5: `𝖡₀'`, `𝖡₁'`,
`𝖡₂'` are the biased `2C₃ \ e` without a balanced 2-cycle (`IsBPrime`), and `D_{1,0}` is the biased
`K₄` with exactly one balanced triangle and no balanced 4-cycle.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

universe uV uE uV' uE'

variable {V : Type uV} {E : Type uE} {V' : Type uV'} {E' : Type uE'}

/-- `2C₃ \ e`: the graph `twoC3` without its last edge. -/
def OrientedGraph.twoC3Minus : OrientedGraph (Fin 3) (Fin 5) := ⟨![0, 0, 1, 1, 2], ![1, 1, 2, 2, 0]⟩

namespace BiasedGraph

/-- An **isomorphism** of biased graphs (§2.1): bijections of the vertices and of the edges under
which every edge keeps its ends, taking the balanced cycles of `Ω` onto those of `Ω'`. -/
structure Iso (Ω : BiasedGraph V E) (Ω' : BiasedGraph V' E') where
  /-- The bijection of vertices. -/
  vertex : V ≃ V'
  /-- The bijection of edges. -/
  edge : E ≃ E'
  /-- Each edge keeps its ends. -/
  joins : ∀ e, Ω'.graph.Joins (edge e) (vertex (Ω.graph.tail e)) (vertex (Ω.graph.head e))
  /-- `B` goes to `S`. -/
  balanced_iff : ∀ C, C ∈ Ω.balanced ↔ edge '' C ∈ Ω'.balanced

/-- `Ω` and `Ω'` are isomorphic. -/
def IsIsomorphic (Ω : BiasedGraph V E) (Ω' : BiasedGraph V' E') : Prop := Nonempty (Ω.Iso Ω')

/-- Every biased graph is isomorphic to itself. -/
def Iso.refl (Ω : BiasedGraph V E) : Ω.Iso Ω where
  vertex := Equiv.refl V
  edge := Equiv.refl E
  joins _ := Or.inl ⟨rfl, rfl⟩
  balanced_iff C := by simp

/-- A biased `2C₃` with no balanced 2-cycle (§3.1): one of six members of `𝒢₀` up to isomorphism. -/
def IsG0TwoC3 (S : BiasedGraph (Fin 3) (Fin 6)) : Prop :=
  S.graph.IsReorientationOf twoC3 ∧ ∀ C ∈ S.balanced, C.encard ≠ 2

/-- A biased tube with no balanced 2-cycle (§3.1): `𝖡₀`, `𝖡₁` or `𝖡₂` up to isomorphism. -/
def IsG0Tube (S : BiasedGraph (Fin 4) (Fin 6)) : Prop :=
  S.graph.IsReorientationOf tube ∧ ∀ C ∈ S.balanced, C.encard ≠ 2

/-- A biased `K₄` with no balanced triangle (§3.1): `D_{0,0}`, …, `D_{0,3}` up to isomorphism. -/
def IsG0K4 (S : BiasedGraph (Fin 4) (Fin 6)) : Prop :=
  S.graph.IsReorientationOf K4 ∧ ∀ C ∈ S.balanced, C.encard ≠ 3

/-- A biased `2C₃ \ e` with no balanced 2-cycle (§3.2): `𝖡₀'`, `𝖡₁'` or `𝖡₂'` up to isomorphism,
with `0`, `1` or `2` balanced triangles. -/
def IsBPrime (S : BiasedGraph (Fin 3) (Fin 5)) : Prop :=
  S.graph.IsReorientationOf twoC3Minus ∧ ∀ C ∈ S.balanced, C.encard ≠ 2

/-- `D_{1,0}` (§3.1): the biased `K₄` with exactly one balanced triangle and no balanced 4-cycle. -/
def IsD10 (S : BiasedGraph (Fin 4) (Fin 6)) : Prop :=
  S.graph.IsReorientationOf K4 ∧ {C | C ∈ S.balanced ∧ C.encard = 3}.ncard = 1 ∧
    ∀ C ∈ S.balanced, C.encard ≠ 4

/-- `Ω` has a member of `𝒢₀` as a link minor (§1.2, §3.1). -/
def HasG0LinkMinor (Ω : BiasedGraph V E) : Prop :=
  (∃ S, IsG0TwoC3 S ∧ Ω.HasLinkMinor S) ∨ (∃ S, IsG0Tube S ∧ Ω.HasLinkMinor S) ∨
    ∃ S, IsG0K4 S ∧ Ω.HasLinkMinor S

/-- A **subdivision** of `S` in `Ω` (§3.1): a biased subgraph of `Ω` that is a subdivision of `S`.
The vertices of `S` go to distinct vertices of `Ω`, a link to a path between the images of its
ends, and a loop to a cycle through the image of its vertex. Two of these walks share no edge and
meet only in images of vertices of `S`, and a walk meets no image other than its ends. A cycle of
`S` is balanced exactly when the union of the walks of its edges is balanced in `Ω`. -/
structure SubdivisionMap (S : BiasedGraph V' E') (Ω : BiasedGraph V E) where
  /-- The branch vertex of each vertex of `S`. -/
  vertex : V' → V
  vertex_injective : Function.Injective vertex
  /-- The walk replacing each edge of `S`. -/
  path : ∀ e', Ω.graph.Walk (vertex (S.graph.tail e')) (vertex (S.graph.head e'))
  isPath : ∀ e', S.graph.IsLink e' → (path e').IsPath
  isLoop : ∀ e', ¬S.graph.IsLink e' →
    (path e').edges ≠ [] ∧ (path e').edges.Nodup ∧ (path e').support.tail.Nodup
  ends : ∀ e', ∀ v ∈ (path e').support, v ∈ Set.range vertex →
    v = vertex (S.graph.tail e') ∨ v = vertex (S.graph.head e')
  disjoint : ∀ e₁ e₂, e₁ ≠ e₂ → ∀ v ∈ (path e₁).support, v ∈ (path e₂).support →
    v ∈ Set.range vertex
  edge_disjoint : ∀ e₁ e₂, e₁ ≠ e₂ → ∀ f ∈ (path e₁).edges, f ∉ (path e₂).edges
  balanced_iff : ∀ C', S.graph.IsCycle C' →
    (C' ∈ S.balanced ↔ (⋃ e' ∈ C', (path e').edgeSet) ∈ Ω.balanced)

/-- The subdivision is all of `Ω`: `Ω` **is** a subdivision of `S` (§5.2). -/
structure SubdivisionMap.IsSpanning {S : BiasedGraph V' E'} {Ω : BiasedGraph V E}
    (m : S.SubdivisionMap Ω) : Prop where
  edges : ∀ f, ∃ e', f ∈ (m.path e').edges
  verts : ∀ v, v ∈ Set.range m.vertex ∨ ∃ e', v ∈ (m.path e').support

/-- `Ω` contains a subdivision of `S` as a biased subgraph (§3.1). -/
def HasSubdivision (Ω : BiasedGraph V E) (S : BiasedGraph V' E') : Prop :=
  Nonempty (S.SubdivisionMap Ω)

/-- `Ω` contains a subdivision of a member of `𝒢₀` as a biased subgraph (§3.2). -/
def HasG0Subdivision (Ω : BiasedGraph V E) : Prop :=
  (∃ S, IsG0TwoC3 S ∧ Ω.HasSubdivision S) ∨ (∃ S, IsG0Tube S ∧ Ω.HasSubdivision S) ∨
    ∃ S, IsG0K4 S ∧ Ω.HasSubdivision S

/-- `Ω` contains a subdivision of a member of `𝒯₀ = 𝒢₀ ∪ {Pr, Pr₁, Pr₂}` as a biased subgraph
(§1.2, §3.2). -/
def HasT0Subdivision (Ω : BiasedGraph V E) : Prop :=
  Ω.HasG0Subdivision ∨ Ω.HasSubdivision prismBiased ∨ Ω.HasSubdivision prismOneBiased ∨
    Ω.HasSubdivision prismTwoBiased

end BiasedGraph

open BiasedGraph

theorem twoC3Gain_isG0TwoC3 : IsG0TwoC3 (twoC3.gainBiased twoC3Gain) :=
  ⟨IsReorientationOf.refl _, twoC3Gain_noPair⟩

theorem tubeGain_isG0Tube : IsG0Tube (tube.gainBiased tubeGain) :=
  ⟨IsReorientationOf.refl _, tubeGain_noPair⟩

theorem K4Gain_isG0K4 : IsG0K4 (K4.gainBiased K4Gain) :=
  ⟨IsReorientationOf.refl _, K4Gain_noTriangle⟩

theorem twoC3Gain_hasG0LinkMinor : (twoC3.gainBiased twoC3Gain).HasG0LinkMinor :=
  Or.inl ⟨_, twoC3Gain_isG0TwoC3, ⟨LinkMinorMap.refl _⟩⟩

example : (twoC3.gainBiased twoC3Gain).IsIsomorphic (twoC3.gainBiased twoC3Gain) :=
  ⟨Iso.refl _⟩

end Transformer.GainMatroid
