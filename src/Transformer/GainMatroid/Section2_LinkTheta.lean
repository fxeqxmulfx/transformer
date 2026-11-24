module

public import Transformer.GainMatroid.Section2_RerouteBias

/-!
# Three links at a vertex on a theta

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5. Let `e`, `f` and `g` be distinct links at
`u`, `P` a path from the end of `e` to the end of `f`, and `Q` a path from there to the end of `g`,
both missing `u`. Cutting `P` at its first vertex `w` on `Q` gives a theta between `u` and `w` whose
paths start with `e`, `f` and `g` (`OrientedGraph.exists_isTheta_of_links`). So if all cycles
through `e` and `f` and all cycles through `f` and `g` are balanced, a cycle through `e` and `g` is
balanced (`BiasedGraph.exists_balanced_of_links`): the step of the transitivity of `∼`.
`OrientedGraph.not_mem_verts_pair` says that two edges between `x` and `y` miss other vertices.
-/

@[expose] public section

namespace Transformer.GainMatroid

universe uV uE

variable {V : Type uV} {E : Type uE}

namespace OrientedGraph

variable {G : OrientedGraph V E}

open Walk

/-- An edge between `x` and `y` misses every other vertex. -/
theorem Joins.ends_ne {c : E} {x y v : V} (hc : G.Joins c x y) (hx : v ≠ x) (hy : v ≠ y) :
    G.tail c ≠ v ∧ G.head c ≠ v := by
  rcases hc with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rw [h₁, h₂]
  · exact ⟨hx.symm, hy.symm⟩
  · exact ⟨hy.symm, hx.symm⟩

/-- Two edges between `x` and `y` miss every other vertex. -/
theorem not_mem_verts_pair {c d : E} {x y v : V} (hc : G.Joins c x y) (hd : G.Joins d x y)
    (hx : v ≠ x) (hy : v ≠ y) : v ∉ G.verts {c, d} := by
  rw [mem_verts]
  rintro ⟨f, (rfl | rfl), hv | hv⟩
  · exact (hc.ends_ne hx hy).1 hv
  · exact (hc.ends_ne hx hy).2 hv
  · exact (hd.ends_ne hx hy).1 hv
  · exact (hd.ends_ne hx hy).2 hv

/-- Links `e`, `f` and `g` from `u` to the start `a` and the end `b` of a path `P` and to the end
`c` of a path `Q` from `b`, where `P` and `Q` miss `u`, lie on a theta whose paths from `u` start
with `e`, `f` and `g`. -/
theorem exists_isTheta_of_links {u a b c : V} {e f g : E} (hef : e ≠ f) (hfg : f ≠ g)
    (heg : e ≠ g) (he : G.Joins e u a) (hf : G.Joins f u b) (hg : G.Joins g u c)
    {P : G.Walk a b} {Q : G.Walk b c} (hP : P.IsPath) (hQ : Q.IsPath) (huP : u ∉ P.support)
    (huQ : u ∉ Q.support) :
    ∃ (w : V) (A F B : G.Walk u w), IsTheta F A B ∧ e ∈ A.edgeSet ∧ f ∈ F.edgeSet ∧
      g ∈ B.edgeSet := by
  obtain ⟨w, P₁, P₂, hwQ, rfl, hP₁⟩ :=
    P.exists_eq_append_first (S := {z | z ∈ Q.support}) Q.start_mem_support
  obtain ⟨Q₁, Q₂, rfl⟩ := Q.exists_eq_append_of_mem_support hwQ
  have hPQ : ∀ z ∈ P₁.support, z ∈ (Q₁.append Q₂).support → z = w := hP₁
  have hP₁p : P₁.IsPath := hP.of_append_left
  have huP₁ : u ∉ P₁.support := fun h => huP (mem_support_append.2 (Or.inl h))
  have huQ₁ : u ∉ Q₁.support := fun h => huQ (mem_support_append.2 (Or.inl h))
  have huQ₂ : u ∉ Q₂.support := fun h => huQ (mem_support_append.2 (Or.inr h))
  have hPe : ∀ d ∈ P₁.edges, d ∉ (Q₁.append Q₂).edges := fun d hd hdQ =>
    hP₁p.isLink_of_mem hd
      ((hPQ _ (P₁.ends_mem_support hd).1 ((Q₁.append Q₂).ends_mem_support hdQ).1).trans
        (hPQ _ (P₁.ends_mem_support hd).2 ((Q₁.append Q₂).ends_mem_support hdQ).2).symm)
  have hu : ∀ {d : E} {x y : V} {R : G.Walk x y}, u ∉ R.support →
      (G.tail d = u ∨ G.head d = u) → d ∉ R.edges := fun hR hd hdR =>
    hR (mem_support_of_mem_edges hdR hd)
  refine ⟨w, cons e he P₁, cons f hf Q₁, cons g hg Q₂.reverse,
    ⟨fun h => huQ (mem_support_append.2 (Or.inr (h ▸ Q₂.start_mem_support))),
      isPath_cons.2 ⟨huQ₁, hQ.of_append_left⟩, isPath_cons.2 ⟨huP₁, hP₁p⟩,
      isPath_cons.2 ⟨fun h => huQ₂ (mem_support_reverse.1 h), hQ.of_append_right.reverse⟩,
      ⟨fun d hdF hdA => ?_, fun z hzF hzA => ?_⟩, ⟨fun d hdF hdB => ?_, fun z hzF hzB => ?_⟩,
      ⟨fun d hdA hdB => ?_, fun z hzA hzB => ?_⟩⟩,
    List.mem_cons_self, List.mem_cons_self, List.mem_cons_self⟩
  · rcases List.mem_cons.1 hdF with rfl | hdF <;> rcases List.mem_cons.1 hdA with rfl | hdA
    · exact hef rfl
    · exact hu huP₁ hf.tail_or_head hdA
    · exact hu huQ₁ he.tail_or_head hdF
    · exact hPe d hdA (mem_edges_append.2 (Or.inl hdF))
  · rcases List.mem_cons.1 hzF with rfl | hzF
    · exact Or.inl rfl
    rcases List.mem_cons.1 hzA with rfl | hzA
    · exact Or.inl rfl
    exact Or.inr (hPQ z hzA (mem_support_append.2 (Or.inl hzF)))
  · rcases List.mem_cons.1 hdF with rfl | hdF <;> rcases List.mem_cons.1 hdB with rfl | hdB
    · exact hfg rfl
    · exact hu huQ₂ hf.tail_or_head (mem_edges_reverse.1 hdB)
    · exact hu huQ₁ hg.tail_or_head hdF
    · exact hQ.not_mem_of_mem hdF (mem_edges_reverse.1 hdB)
  · rcases List.mem_cons.1 hzF with rfl | hzF
    · exact Or.inl rfl
    rcases List.mem_cons.1 hzB with rfl | hzB
    · exact Or.inl rfl
    exact Or.inr (hQ.eq_of_mem_of_mem hzF (mem_support_reverse.1 hzB))
  · rcases List.mem_cons.1 hdA with rfl | hdA <;> rcases List.mem_cons.1 hdB with rfl | hdB
    · exact heg rfl
    · exact hu huQ₂ he.tail_or_head (mem_edges_reverse.1 hdB)
    · exact hu huP₁ hg.tail_or_head hdA
    · exact hPe d hdA (mem_edges_append.2 (Or.inr (mem_edges_reverse.1 hdB)))
  · rcases List.mem_cons.1 hzA with rfl | hzA
    · exact Or.inl rfl
    rcases List.mem_cons.1 hzB with rfl | hzB
    · exact Or.inl rfl
    exact Or.inr (hPQ z hzA (mem_support_append.2 (Or.inr (mem_support_reverse.1 hzB))))

end OrientedGraph

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E}

/-- Let `e`, `f` and `g` be distinct edges at `u`, with a cycle `C` through `e` and `f` and a cycle
`D` through `f` and `g`. If all cycles through `e` and `f` and all cycles through `f` and `g` are
balanced, some balanced cycle contains `e` and `g`. -/
theorem exists_balanced_of_links {u : V} {e f g : E} (hef : e ≠ f) (hfg : f ≠ g) (heg : e ≠ g)
    (he : Ω.graph.tail e = u ∨ Ω.graph.head e = u) (hf : Ω.graph.tail f = u ∨ Ω.graph.head f = u)
    (hg : Ω.graph.tail g = u ∨ Ω.graph.head g = u) {C D : Set E} (hC : Ω.graph.IsCycle C)
    (hD : Ω.graph.IsCycle D) (heC : e ∈ C) (hfC : f ∈ C) (hfD : f ∈ D) (hgD : g ∈ D)
    (hEF : ∀ B, Ω.graph.IsCycle B → e ∈ B → f ∈ B → B ∈ Ω.balanced)
    (hFG : ∀ B, Ω.graph.IsCycle B → f ∈ B → g ∈ B → B ∈ Ω.balanced) :
    ∃ B ∈ Ω.balanced, e ∈ B ∧ g ∈ B := by
  obtain ⟨P, hP, huP, -⟩ := hC.exists_isPath_of_mem hef heC hfC he hf
  obtain ⟨Q, hQ, huQ, -⟩ := hD.exists_isPath_of_mem hfg hfD hgD hf hg
  obtain ⟨w, A, F, B, hθ, heA, hfF, hgB⟩ := exists_isTheta_of_links hef hfg heg
    (joins_otherEnd he).symm (joins_otherEnd hf).symm (joins_otherEnd hg).symm hP hQ huP huQ
  obtain ⟨hFA, hFB, -⟩ := hθ.isCycle
  exact ⟨_, Ω.theta hθ (hEF _ hFA (Or.inr heA) (Or.inl hfF)) (hFG _ hFB (Or.inl hfF) (Or.inr hgB)),
    Or.inl heA, Or.inr hgB⟩

end BiasedGraph

example := Joins.ends_ne (G := pathThree) (c := 0) (x := 0) (y := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩)
  (by decide) (by decide)

example := not_mem_verts_pair (G := rerouteGraph) (c := 0) (d := 1) (x := 0) (y := 1) (v := 2)
  (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

example := exists_isTheta_of_links (G := rerouteGraph) (u := 0) (e := 0) (f := 1) (g := 2)
  (by decide) (by decide) (by decide) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
  (P := .nil 1) (Q := .cons 3 (Or.inr ⟨rfl, rfl⟩) (.nil 2)) (List.nodup_singleton 1)
  (by unfold Walk.IsPath; decide) (by decide) (by decide)

example := BiasedGraph.exists_balanced_of_links (Ω := pumpkin.gainBiased (1 : Fin 3 → ℚˣ)) (u := 0)
  (e := 0) (f := 1) (g := 2) (by decide) (by decide) (by decide) (Or.inl rfl) (Or.inl rfl)
  (Or.inl rfl)
  (isCycle_pair (G := pumpkin) (e := 0) (f := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
    (by decide))
  (isCycle_pair (G := pumpkin) (e := 1) (f := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
    (by decide))
  (by simp) (by simp) (by simp) (by simp)
  (fun _ hB _ _ => mem_balancedCycles_iff.2 ⟨hB, 1, by simp [switch]⟩)
  (fun _ hB _ _ => mem_balancedCycles_iff.2 ⟨hB, 1, by simp [switch]⟩)

end Transformer.GainMatroid
