module

public import Transformer.GainMatroid.Section2_CycleBicycles

/-!
# Balanced fundamental cycles

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1, define a biased graph by the theta property
and cite Zaslavsky for its matroids (§2.2). The exchange property of the lift matroid rests on the
following consequence of the theta property (`BiasedGraph.mem_balanced_of_fundamental`).

Let `I` be a forest, and let every edge `e` of `S` have its ends linked along `I`, so that `e` and
the path of `I` between its ends form a cycle. If all these cycles are balanced, then every cycle
inside `I ∪ S` is balanced. The proof is by induction on the number of edges of the cycle `C` off
`I`. If `C` has two of them, the cycle `D` of one of them does not lie in `C`, so it has an ear on
`C`. The ear lies in `I`, and with the two arcs of `C` between its ends it forms a theta whose other
two cycles have fewer edges off `I`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*} [Finite E] (Ω : BiasedGraph V E)

/-- **Balanced fundamental cycles.** If `I` is a forest, every edge `e` of `S` has its ends linked
along `I`, and every cycle inside `insert e I` is balanced, then every cycle inside `I ∪ S` is
balanced. -/
theorem mem_balanced_of_fundamental {I S : Set E} (hI : Ω.graph.IsForest I)
    (hR : ∀ e ∈ S, Ω.graph.ReachableIn I (Ω.graph.tail e) (Ω.graph.head e))
    (hB : ∀ e ∈ S, ∀ D, Ω.graph.IsCycle D → D ⊆ insert e I → D ∈ Ω.balanced)
    {C : Set E} (hC : Ω.graph.IsCycle C) (hCS : C ⊆ I ∪ S) : C ∈ Ω.balanced := by
  suffices key : ∀ n, ∀ C : Set E, (C \ I).ncard = n → Ω.graph.IsCycle C → C ⊆ I ∪ S →
      C ∈ Ω.balanced from key _ C rfl hC hCS
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro C hn hC hCS
  obtain ⟨e₁, he₁C, he₁I⟩ : ∃ e ∈ C, e ∉ I := by
    by_contra! h
    exact hI C hC h
  have he₁S : e₁ ∈ S := (hCS he₁C).resolve_left he₁I
  by_cases hCe : C ⊆ insert e₁ I
  · exact hB e₁ he₁S C hC hCe
  obtain ⟨p, hp⟩ := hR e₁ he₁S
  obtain ⟨Q, hQ, hQp⟩ := p.exists_isPath
  have hQI : ∀ f ∈ Q.edges, f ∈ I := fun f hf => hp f (hQp f hf)
  have he₁Q : e₁ ∉ Q.edges := fun h => he₁I (hQI e₁ h)
  have hDI : insert e₁ Q.edgeSet ⊆ insert e₁ I := Set.insert_subset_insert fun f hf => hQI f hf
  have hD : Ω.graph.IsCycle (insert e₁ Q.edgeSet) :=
    isCycle_insert_of_isPath (Or.inr ⟨rfl, rfl⟩) hQ he₁Q
  have hDC : ¬insert e₁ Q.edgeSet ⊆ C := fun h => hCe (by rw [← hC.eq_of_subset hD h]; exact hDI)
  obtain ⟨u, a, y, g, hg, P, hu, hgD, hgC, hP, hgP, hPD, hy, hPC⟩ :=
    hD.exists_ear hDC (mem_verts.2 ⟨e₁, he₁C, Or.inl rfl⟩)
      (mem_verts.2 ⟨e₁, Set.mem_insert _ _, Or.inl rfl⟩)
  have hPC' : ∀ f ∈ P.edges, f ∉ C := fun f hf => hP.notMem_of_forall hPC hf
  have hoff : ∀ f ∈ insert e₁ Q.edgeSet, f ∉ C → f ∈ I := fun f hf hfC => by
    rcases hf with rfl | hf
    · exact absurd he₁C hfC
    · exact hQI f hf
  have hgI : g ∈ I := hoff g hgD hgC
  by_cases hyu : y = u
  · subst hyu
    have hEq := hD.eq_of_subset (isCycle_insert_of_isPath hg hP hgP)
      (Set.insert_subset hgD fun f hf => hPD f hf)
    have he₁ : e₁ ∈ insert g P.edgeSet := hEq ▸ Set.mem_insert _ _
    rcases he₁ with rfl | he₁
    · exact absurd he₁C hgC
    · exact absurd he₁C (hPC' e₁ he₁)
  have huP : u ∉ P.support := fun h => hyu (hPC u h hu).symm
  let R : Ω.graph.Walk u y := .cons g hg P
  have hR : R.IsPath := Walk.isPath_cons.2 ⟨huP, hP⟩
  have hRC : ∀ f ∈ R.edges, f ∉ C := fun f hf => by
    rcases List.mem_cons.1 hf with rfl | hf
    · exact hgC
    · exact hPC' f hf
  have hRI : R.edgeSet ⊆ I := fun f hf => by
    rcases List.mem_cons.1 hf with rfl | hf
    · exact hgI
    · exact hoff f (hPD f hf) (hPC' f hf)
  obtain ⟨P₁, P₂, hP₁, hP₂, h₁₂, rfl⟩ := hC.exists_openly (Ne.symm hyu) hu hy
  have hopen (T : Ω.graph.Walk u y) (hT : T.edgeSet ⊆ P₁.edgeSet ∪ P₂.edgeSet) :
      Walk.Openly R T := by
    refine ⟨fun f hfR hfT => hRC f hfR (hT hfT), fun z hzR hzT => ?_⟩
    rcases List.mem_cons.1 hzR with rfl | hzP
    · exact Or.inl rfl
    · exact Or.inr (hPC z hzP (verts_mono hT
        (T.mem_verts_of_mem_support (Walk.edges_ne_nil_of_ne (Ne.symm hyu) T) hzT)))
  have hθ : IsTheta R P₁ P₂ := ⟨Ne.symm hyu, hR, hP₁, hP₂, hopen P₁ Set.subset_union_left,
    hopen P₂ Set.subset_union_right, h₁₂⟩
  have hoffI (T : Ω.graph.Walk u y) (hT : Ω.graph.IsCycle (R.edgeSet ∪ T.edgeSet)) :
      ∃ f ∈ T.edgeSet, f ∉ I := by
    by_contra! h
    exact hI _ hT (Set.union_subset hRI h)
  have hsub (T : Ω.graph.Walk u y) (hT : T.edgeSet ⊆ P₁.edgeSet ∪ P₂.edgeSet) :
      R.edgeSet ∪ T.edgeSet ⊆ I ∪ S :=
    Set.union_subset (hRI.trans Set.subset_union_left) (hT.trans hCS)
  have hlt (T T' : Ω.graph.Walk u y) (hTT' : T.edgeSet ∪ T'.edgeSet = P₁.edgeSet ∪ P₂.edgeSet)
      (hdis : ∀ f ∈ T'.edges, f ∉ T.edges) (hT' : Ω.graph.IsCycle (R.edgeSet ∪ T'.edgeSet)) :
      ((R.edgeSet ∪ T.edgeSet) \ I).ncard < n := by
    rw [← hn, ← hTT']
    refine Set.ncard_lt_ncard (ssubset_of_subset_not_superset ?_ fun hle => ?_)
    · rintro f ⟨hf | hf, hfI⟩
      · exact absurd (hRI hf) hfI
      · exact ⟨Or.inl hf, hfI⟩
    · obtain ⟨f, hf, hfI⟩ := hoffI T' hT'
      rcases (hle ⟨Or.inr hf, hfI⟩).1 with hf' | hf'
      · exact absurd (hRI hf') hfI
      · exact hdis f hf hf'
  have hB₁ := ih _ (hlt P₁ P₂ rfl (fun f hf hf' => h₁₂.edges f hf' hf) hθ.isCycle.2.1) _ rfl
    hθ.isCycle.1 (hsub P₁ Set.subset_union_left)
  have hB₂ := ih _ (hlt P₂ P₁ (Set.union_comm _ _) (fun f hf hf' => h₁₂.edges f hf hf')
    hθ.isCycle.1) _ rfl hθ.isCycle.2.1 (hsub P₂ Set.subset_union_right)
  exact Ω.theta hθ hB₁ hB₂

example := (pumpkin.gainBiased (1 : Fin 3 → ℚˣ)).mem_balanced_of_fundamental (I := {0})
  (S := {1, 2}) pumpkin_isForest
  (fun _ _ => reachableIn_of_mem (e := (0 : Fin 3)) (Set.mem_singleton 0))
  (fun _ _ _ hD _ => mem_balancedCycles_iff.2 ⟨hD, 1, fun f _ => by simp [switch]⟩)
  pumpkin_pair_isCycle fun x hx => by rcases hx with rfl | rfl <;> simp

end BiasedGraph

end Transformer.GainMatroid
