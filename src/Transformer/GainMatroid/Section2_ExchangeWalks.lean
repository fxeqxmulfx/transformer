module

public import Transformer.GainMatroid.Section2_ExchangeBalance
public import Transformer.GainMatroid.Section2_ExchangeGraphs

/-!
# Walks across an exchange pair

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, the proof of Proposition 2.6. For an
exchange pair `T`, `S` on `X = {a, b, c}` (`IsExchange`), with the triangle `X` in `T` and the star
`X` at `v` in `S`:

* an end other than `v` of an edge of `S` is a vertex of `T` (`IsExchange.exists_eq_of_incident`),
  and so are the ends other than `v` of a walk of `S` (`IsExchange.exists_eq_of_walk`);
* any two edges of `X` can be labelled `b` and `c` (`IsExchange.exists_relabel`), so a set of edges
  meeting a balanced triangle `X` in two edges is in `Δ_X B` exactly when its symmetric difference
  with `X` is in `B` (`IsExchange.mem_deltaBalanced_iff_of_pair`);
* a walk of `S` that misses `v` is a walk of `T` with the same edges
  (`IsExchange.exists_lift_of_not_mem`).

A walk between distinct vertices starts with an edge (`OrientedGraph.Walk.exists_eq_cons_of_ne`).
The examples are on `clawGraph` and `clawNabla`, an exchange pair with the star at `0`
(`clawNabla_isExchange`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open scoped symmDiff

/-- A walk between distinct vertices starts with an edge. -/
theorem OrientedGraph.Walk.exists_eq_cons_of_ne {V E : Type*} {G : OrientedGraph V E} {u w : V}
    (p : G.Walk u w) (h : u ≠ w) :
    ∃ (u' : V) (e : E) (he : G.Joins e u u') (q : G.Walk u' w), p = .cons e he q := by
  cases p with
  | nil => exact absurd rfl h
  | cons e he q => exact ⟨_, e, he, q, rfl⟩

namespace OrientedGraph.IsExchange

variable {V W E : Type*} {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W}
  {a b c : E} {x y z : V}

/-- An end other than `v` of an edge of `S` is the image of a vertex of `T`. -/
theorem exists_eq_of_incident (h : IsExchange T S ι v a b c x y z) {e : E} {w : W}
    (hw : S.tail e = w ∨ S.head e = w) (hv : w ≠ v) : ∃ u, ι u = w := by
  by_cases he : e ∈ ({a, b, c} : Set E)
  · rcases he with rfl | rfl | rfl
    · exact ⟨x, ((h.joins_a.eq_or_eq hw).resolve_left hv).symm⟩
    · exact ⟨y, ((h.joins_b.eq_or_eq hw).resolve_left hv).symm⟩
    · exact ⟨z, ((h.joins_c.eq_or_eq hw).resolve_left hv).symm⟩
  · obtain ⟨h₁, h₂⟩ := h.agree e he
    rcases hw with rfl | rfl
    · exact ⟨_, h₁.symm⟩
    · exact ⟨_, h₂.symm⟩

/-- The start of a walk of `S` between distinct vertices, if not `v`, is a vertex of `T`. -/
theorem exists_eq_of_walk (h : IsExchange T S ι v a b c x y z) {s t : W} (P : S.Walk s t)
    (hst : s ≠ t) (hs : s ≠ v) : ∃ u, ι u = s := by
  obtain ⟨g, -, hg⟩ := P.exists_mem_edges_start hst
  exact h.exists_eq_of_incident hg hs

/-- Any two distinct edges of `X` can be labelled `b` and `c`. -/
theorem exists_relabel (h : IsExchange T S ι v a b c x y z) {e₁ e₂ : E}
    (he₁ : e₁ ∈ ({a, b, c} : Set E)) (he₂ : e₂ ∈ ({a, b, c} : Set E)) (hne : e₁ ≠ e₂) :
    ∃ (e₀ : E) (x' y' z' : V), IsExchange T S ι v e₀ e₁ e₂ x' y' z' ∧
      ({e₀, e₁, e₂} : Set E) = {a, b, c} := by
  rcases he₁ with rfl | rfl | rfl <;> rcases he₂ with rfl | rfl | rfl <;>
    first
    | exact absurd rfl hne
    | exact ⟨_, _, _, _, h, rfl⟩
    | refine ⟨_, _, _, _, h.swap₁₂, ?_⟩
    | refine ⟨_, _, _, _, h.swap₂₃, ?_⟩
    | refine ⟨_, _, _, _, h.swap₁₂.swap₂₃, ?_⟩
    | refine ⟨_, _, _, _, h.swap₂₃.swap₁₂, ?_⟩
    | refine ⟨_, _, _, _, h.swap₁₂.swap₂₃.swap₁₂, ?_⟩
  all_goals
    ext g
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
    tauto

/-- A set of edges meeting the balanced triangle `X` in two edges is in `Δ_X B` exactly when its
symmetric difference with `X` is in `B`. -/
theorem mem_deltaBalanced_iff_of_pair {Ω : BiasedGraph V E}
    (h : IsExchange Ω.graph S ι v a b c x y z) (hbal : ({a, b, c} : Set E) ∈ Ω.balanced)
    {D : Set E} {e₁ e₂ : E} (he₁ : e₁ ∈ ({a, b, c} : Set E)) (he₂ : e₂ ∈ ({a, b, c} : Set E))
    (hne : e₁ ≠ e₂) (hD : D ∩ {a, b, c} = {e₁, e₂}) :
    D ∈ deltaBalanced Ω.balanced {a, b, c} ↔ D ∆ {a, b, c} ∈ Ω.balanced := by
  obtain ⟨e₀, x', y', z', h', hX⟩ := h.exists_relabel he₁ he₂ hne
  rw [← hX] at hbal hD ⊢
  exact Ω.mem_deltaBalanced_iff_of_inter_eq_pair h'.triangle hbal hD

/-- A walk of `S` from `ι s` that misses `v` is a walk of `T` from `s`, with the same edges. -/
theorem exists_lift_of_not_mem (h : IsExchange T S ι v a b c x y z) {s : V} {t : W}
    (q : S.Walk (ι s) t) (hv : v ∉ q.support) :
    ∃ (u : V) (p : T.Walk s u), ι u = t ∧ p.edges = q.edges ∧ p.support.map ι = q.support :=
  q.exists_lift h.inj (fun g hg => h.agree g fun hX =>
    hv (Walk.mem_support_of_mem_edges hg (h.incident_of_mem hX))) rfl

/-- A walk of `S` that misses `v` has no edge of `X`. -/
theorem not_mem_of_not_mem_support (h : IsExchange T S ι v a b c x y z) {s t : W}
    {q : S.Walk s t} (hv : v ∉ q.support) {g : E} (hg : g ∈ q.edges) :
    g ∉ ({a, b, c} : Set E) :=
  fun hX => hv (Walk.mem_support_of_mem_edges hg (h.incident_of_mem hX))

end OrientedGraph.IsExchange

open OrientedGraph

/-- The edge `3` of `clawGraph`, from `2` to `1`, as a walk: it misses the centre `0`. -/
def clawWalk21 : clawGraph.Walk 2 1 := .cons 3 (Or.inl ⟨rfl, rfl⟩) (.nil 1)

/-- `clawNabla` and `clawGraph` are an exchange pair on `{0, 1, 2}`, with the star at `0`. -/
theorem clawNabla_isExchange : IsExchange clawNabla clawGraph id 0 0 1 2 1 2 3 :=
  clawGraph_nablaGraph ▸ clawGraph_isLabelledClaw.isExchange

/-- The edge `3` of `clawGraph` from `1` to `2`, off the star. -/
def clawEdge12 : clawGraph.Walk 1 2 := .cons 3 (Or.inr ⟨rfl, rfl⟩) (.nil 2)

/-- The path `1, 0, 2` of `clawGraph` along the star. -/
def clawStar12 : clawGraph.Walk 1 2 :=
  .cons 0 (Or.inr ⟨rfl, rfl⟩) (.cons 1 (Or.inl ⟨rfl, rfl⟩) (.nil 2))

/-- The edge `3` of `clawNabla` from `1` to `2`. -/
def nablaEdge12 : clawNabla.Walk 1 2 := .cons 3 (Or.inr ⟨rfl, rfl⟩) (.nil 2)

/-- The edge `2` of `clawNabla` from `1` to `2`, of the triangle. -/
def nablaTriangle12 : clawNabla.Walk 1 2 := .cons 2 (Or.inl ⟨rfl, rfl⟩) (.nil 2)

theorem clawEdge12_isPath : clawEdge12.IsPath := by unfold Walk.IsPath; decide

theorem clawStar12_isPath : clawStar12.IsPath := by unfold Walk.IsPath; decide

example := Walk.exists_eq_cons_of_ne clawStar12 (by decide)

example := clawGraph_isLabelledClaw.isExchange.exists_eq_of_incident (e := 3) (w := 2) (Or.inl rfl)
  (by decide)

example := clawGraph_isLabelledClaw.isExchange.exists_eq_of_walk clawWalk21 (by decide) (by decide)

example := K4_isLabelledTriangle.isExchange.exists_relabel (e₁ := 4) (e₂ := 3) (Or.inr (Or.inl rfl))
  (Or.inr (Or.inr rfl)) (by decide)

example := K4_isLabelledTriangle.isExchange.mem_deltaBalanced_iff_of_pair
  (Ω := K4.gainBiased (1 : Fin 6 → ℚˣ))
  (gainBiased_one_isBalanced K4 _ K4_isLabelledTriangle.isCycle)
  (D := {4, 3}) (Or.inr (Or.inl rfl)) (Or.inr (Or.inr rfl)) (by decide) (by
    ext g
    simp only [Set.mem_inter_iff, Set.mem_insert_iff, Set.mem_singleton_iff]
    tauto)

example := clawGraph_isLabelledClaw.isExchange.exists_lift_of_not_mem (s := 2) clawWalk21
  (by decide)

example := clawGraph_isLabelledClaw.isExchange.not_mem_of_not_mem_support (q := clawWalk21)
  (by decide) (g := 3) (by simp [clawWalk21, Walk.edges])

end Transformer.GainMatroid
