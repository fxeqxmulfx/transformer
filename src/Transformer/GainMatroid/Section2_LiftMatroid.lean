module

public import Transformer.GainMatroid.Section2_LiftIndep

/-!
# The lift matroid of a biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2: "A subset `C ⊆ E(G)` is a circuit of the
lifted-graphic matroid `L(G, B)` precisely when `C ∈ B` or `C` induces a subdivision of one of the
subgraphs shown in Figure 2 containing no balanced cycle", citing Zaslavsky for the fact that these
are the circuits of a matroid. That fact is proved here (`BiasedGraph.exists_isLiftMatroid`).

A family of nonempty sets on a finite type, none containing another, is the family of circuits of a
matroid when the sets containing no member satisfy the exchange axiom
(`exists_matroid_isCircuit_iff`). For the lift circuits these conditions are
`BiasedGraph.IsLiftCircuit.eq_of_subset` and `BiasedGraph.IsLiftIndep.exists_insert`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

/-- A family `P` of nonempty sets on a finite type, none of which contains another, is the family of
circuits of a matroid on the whole type when the sets containing no member of `P` satisfy the
exchange axiom. -/
theorem exists_matroid_isCircuit_iff {α : Type*} [Finite α] (P : Set α → Prop) (hne : ¬P ∅)
    (hanti : ∀ X Y, P X → P Y → Y ⊆ X → Y = X)
    (haug : ∀ I J, (∀ X ⊆ I, ¬P X) → (∀ X ⊆ J, ¬P X) → I.ncard < J.ncard →
      ∃ e ∈ J, e ∉ I ∧ ∀ X ⊆ insert e I, ¬P X) :
    ∃ M : Matroid α, M.E = Set.univ ∧ ∀ X, M.IsCircuit X ↔ P X := by
  let M := (IndepMatroid.ofFinite Set.finite_univ (fun I => ∀ X ⊆ I, ¬P X)
    (fun X hX => by rw [Set.subset_empty_iff.1 hX]; exact hne)
    (fun _ _ hJ hIJ X hXI => hJ X (hXI.trans hIJ)) (fun I J hI hJ hlt => haug I J hI hJ hlt)
    (fun I _ => Set.subset_univ I)).matroid
  have hdep (D : Set α) : M.Dep D ↔ ∃ Y ⊆ D, P Y := by
    rw [Matroid.dep_iff]
    simp [M]
  refine ⟨M, rfl, fun X => ?_⟩
  rw [Matroid.isCircuit_iff]
  constructor
  · rintro ⟨hX, hmin⟩
    obtain ⟨Y, hYX, hY⟩ := (hdep X).1 hX
    rwa [← hmin ((hdep Y).2 ⟨Y, le_rfl, hY⟩) hYX]
  · intro hX
    refine ⟨(hdep X).2 ⟨X, le_rfl, hX⟩, fun D hD hDX => ?_⟩
    obtain ⟨Y, hYD, hY⟩ := (hdep D).1 hD
    exact hDX.antisymm ((hanti X Y hX hY (hYD.trans hDX)) ▸ hYD)

-- The single loop on `Fin 1`.
example : ∃ M : Matroid (Fin 1), M.E = Set.univ ∧ ∀ X, M.IsCircuit X ↔ X = Set.univ :=
  exists_matroid_isCircuit_iff (fun X => X = Set.univ) Set.empty_ne_univ
    (fun _ _ hX hY _ => hY.trans hX.symm) fun I J _ hJ hlt => by
      have hJe : J = ∅ := by
        by_contra hne
        exact hJ J le_rfl (Set.nonempty_iff_ne_empty.2 hne).eq_univ
      simp [hJe] at hlt

namespace BiasedGraph

variable {V E : Type*}

/-- **§2.2** (Zaslavsky). The lift circuits of a finite biased graph are the circuits of a matroid,
`L(G, B)`. -/
theorem exists_isLiftMatroid [Finite V] [Finite E] (Ω : BiasedGraph V E) :
    ∃ M, Ω.IsLiftMatroid M :=
  exists_matroid_isCircuit_iff Ω.IsLiftCircuit
    (fun h => by
      obtain ⟨C, hC, hCyc⟩ := h.exists_isCycle
      obtain ⟨e, he⟩ := hCyc.nonempty
      exact hC he)
    (fun _ _ hX hY hYX => hX.eq_of_subset hY hYX)
    (fun _ _ hI hJ hlt => IsLiftIndep.exists_insert hI hJ hlt)

end BiasedGraph

open BiasedGraph

example := triangleBiased.exists_isLiftMatroid

example : ∃ M : Matroid Bool, M.contract {true} = M.delete {true} := by
  obtain ⟨M, hM⟩ := (twoLoops.gainBiased (1 : Bool → ℚˣ)).exists_isLiftMatroid
  exact ⟨M, hM.contract_eq_delete ((singleton_mem_balancedCycles_iff _ rfl).2 rfl)⟩

example : ∃ M, ((twoLoops.delete {true}).gainBiased
    (1 : ↥({true}ᶜ : Set Bool) → ℚˣ)).IsLiftMatroid M := by
  obtain ⟨M, hM⟩ := twoLoops.contrabalanced.exists_isLiftMatroid
  exact ⟨_, IsLiftContraction.isLiftMatroid (Ω := twoLoops.contrabalanced) (e := true)
    ⟨⟨rfl, Set.notMem_empty _⟩, rfl,
      fun _ hC => mem_balancedCycles_iff.2 ⟨hC, 1, fun f _ => by simp [switch]⟩⟩ hM⟩

end Transformer.GainMatroid
