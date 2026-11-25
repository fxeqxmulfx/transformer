module

public import Transformer.GainMatroid.Section2_FrameAugment
public import Transformer.GainMatroid.Section2_FrameCircuits
public import Transformer.GainMatroid.Section2_LiftMatroid

/-!
# The frame matroid of a biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2: "A subset `C ⊆ E(G)` is a circuit of the
frame matroid `F(G, B)` precisely when `C ∈ B` or `C` induces a subdivision of one of the graphs
shown in Figure 1 containing no balanced cycle", citing Zaslavsky for the fact that these are the
circuits of a matroid. That fact is proved here (`BiasedGraph.exists_isFrameMatroid`) by
`exists_matroid_isCircuit_iff`: no frame circuit lies in another
(`BiasedGraph.IsFrameCircuit.eq_of_subset`), and the exchange axiom holds for the sets containing
none (`BiasedGraph.IsFrameIndep.exists_insert`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*}

/-- **§2.2** (Zaslavsky). The frame circuits of a finite biased graph are the circuits of a
matroid, `F(G, B)`. -/
theorem exists_isFrameMatroid [Finite V] [Finite E] (Ω : BiasedGraph V E) :
    ∃ M, Ω.IsFrameMatroid M :=
  exists_matroid_isCircuit_iff Ω.IsFrameCircuit
    (fun h => by
      obtain ⟨C, hC, hCyc⟩ := h.exists_isCycle
      obtain ⟨e, he⟩ := hCyc.nonempty
      exact hC he)
    (fun _ _ hX hY hYX => hX.eq_of_subset hY hYX)
    (fun _ _ hI hJ hlt => IsFrameIndep.exists_insert hI hJ hlt)

end BiasedGraph

open BiasedGraph

/-- The first link of `pumpkin` is independent in its frame matroid with every cycle balanced. -/
theorem pumpkin_isFrameIndep_zero : (pumpkin.gainBiased (1 : Fin 3 → ℚˣ)).IsFrameIndep {0} :=
  isFrameIndep_iff.2 ⟨fun C hC hCyc _ => pumpkin_isForest C hCyc hC,
    fun C _ hC _ hCK _ _ _ _ _ _ => absurd hCK (pumpkin_isForest C hC)⟩

/-- Its first two links form a balanced cycle. -/
theorem pumpkin_not_isFrameIndep_pair :
    ¬(pumpkin.gainBiased (1 : Fin 3 → ℚˣ)).IsFrameIndep {1, 0} :=
  fun h => h {1, 0} le_rfl (Or.inl (mem_balancedCycles_iff.2
    ⟨isCycle_pair (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide), 1,
      fun f _ => by simp [switch]⟩))

example := reachableIn_of_not_isFrameIndep_insert pumpkin_isFrameIndep_zero
  pumpkin_not_isFrameIndep_pair (Or.inl fun ⟨C, hCK, hC, _⟩ => pumpkin_isForest C hC hCK)

example := mem_balanced_of_not_isFrameIndep_insert pumpkin_isFrameIndep_zero
  pumpkin_not_isFrameIndep_pair (Or.inl fun ⟨C, hCK, hC, _⟩ => pumpkin_isForest C hC hCK)
  (D := {1, 0}) (isCycle_pair (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide))
  (Set.mem_insert 1 {0}) le_rfl

example := IsFrameIndep.exists_insert (I := ∅) (J := {0})
  (fun X hX hXc => by
    obtain ⟨C, hCX, hC⟩ := hXc.exists_isCycle
    obtain ⟨e, he⟩ := hC.nonempty
    exact hX (hCX he))
  pumpkin_isFrameIndep_zero (by simp)

example := triangleBiased.exists_isFrameMatroid

example : ∃ M : Matroid Bool, M.contract {true} = M.delete {true} := by
  obtain ⟨M, hM⟩ := (twoLoops.gainBiased (1 : Bool → ℚˣ)).exists_isFrameMatroid
  exact ⟨M, hM.contract_eq_delete ((singleton_mem_balancedCycles_iff _ rfl).2 rfl)⟩

example := digonJoint_isRollable.balancing.isCircuit_or_indep
  (Classical.choose_spec digonJoint.contrabalanced.exists_isFrameMatroid)
  (Or.inl digonJoint_link₀) (Or.inl digonJoint_link₁) (by decide)

end Transformer.GainMatroid
