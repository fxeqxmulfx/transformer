module

public import Mathlib.Combinatorics.Matroid.Minor.Contract
public import Transformer.GainMatroid.Section2_ExchangeMatroids
public import Transformer.GainMatroid.Section2_JointContractions
public import Transformer.GainMatroid.Section2_Representable

/-!
# The frame and lift matroids of a biased graph

Funk, Pivotto and Slilaty (arXiv:1609.05574v5). §1: "When `M` is frame or lifted-graphic, the biased
graph `(G, B)` completely determines `M`", and "Zaslavsky showed that every such gain graph yields a
canonical matrix representation `A` for `M` over `𝔽`." §2.2 gives the circuits of `F(G, B)` and
`L(G, B)` after Zaslavsky and continues: "Minors of biased graphs and their matroids agree: for any
edge `e` of a biased graph `(G, B)`, `F(G, B) \ e = F((G, B) \ e)`, `F(G, B) / e = F((G, B) / e)`,
`L(G, B) \ e = L((G, B) \ e)`, for any link `e`, `L(G, B) / e = L((G, B) / e)`, and if `e` is a
joint then `L(G, B) / e = M(G \ e)`." It adds: "Evidently whenever `(G, B)` is a biased graph with
no two vertex-disjoint unbalanced cycles, `F(G, B) = L(G, B)`."

The source follows Oxley's text, whose matroids are finite, so the statements asserting a matroid
take `V` and `E` finite. Zaslavsky's papers are not among the local sources: the facts cited from
them are stated here and left unproved, except that the lift circuits are the circuits of a matroid,
which `Section2_LiftMatroid` proves (`exists_isLiftMatroid`). `LinkMinorMap` covers deletion and
the contraction of a link, `IsFrameContraction` and `IsLiftContraction` the two contractions of a
joint, and `IsFrameMatroid.contract_eq_delete` a balanced loop, for which the source sets
`/ e = \ e`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E V' E' : Type*} {Ω : BiasedGraph V E} {M M' : Matroid E}

/-- **§1.** The biased graph determines its frame matroid. -/
theorem IsFrameMatroid.unique (h : Ω.IsFrameMatroid M) (h' : Ω.IsFrameMatroid M') : M = M' :=
  Matroid.ext_isCircuit (h.1.trans h'.1.symm) fun C _ => (h.2 C).trans (h'.2 C).symm

/-- **§1.** The biased graph determines its lift matroid. -/
theorem IsLiftMatroid.unique (h : Ω.IsLiftMatroid M) (h' : Ω.IsLiftMatroid M') : M = M' :=
  Matroid.ext_isCircuit (h.1.trans h'.1.symm) fun C _ => (h.2 C).trans (h'.2 C).symm

/-- **§2.2.** With no two vertex-disjoint unbalanced cycles, the frame and lift circuits agree. -/
theorem isFrameCircuit_iff_isLiftCircuit
    (h : ∀ C D, Ω.IsUnbalancedCycle C → Ω.IsUnbalancedCycle D →
      ¬Disjoint (Ω.graph.verts C) (Ω.graph.verts D)) (X : Set E) :
    Ω.IsFrameCircuit X ↔ Ω.IsLiftCircuit X := by
  constructor
  · rintro (hb | ⟨ht | hl | hth, hn⟩)
    · exact Or.inl hb
    · exact Or.inr ⟨Or.inl ht, hn⟩
    · obtain ⟨C, D, x, y, P, hC, hD, hd, -, -, -, -, -, rfl⟩ := hl
      exact (h C D ⟨hC, hn C (fun e he => Or.inl (Or.inl he)) hC⟩
        ⟨hD, hn D (fun e he => Or.inl (Or.inr he)) hD⟩ hd).elim
    · exact Or.inr ⟨Or.inr (Or.inr hth), hn⟩
  · rintro (hb | ⟨ht | hl | hth, hn⟩)
    · exact Or.inl hb
    · exact Or.inr ⟨Or.inl ht, hn⟩
    · obtain ⟨C, D, hC, hD, hd, rfl⟩ := hl
      exact (h C D ⟨hC, hn C Set.subset_union_left hC⟩
        ⟨hD, hn D Set.subset_union_right hD⟩ hd).elim
    · exact Or.inr ⟨Or.inr (Or.inr hth), hn⟩

/-- **§2.2.** "Evidently whenever `(G, B)` is a biased graph with no two vertex-disjoint unbalanced
cycles, `F(G, B) = L(G, B)`." -/
theorem isFrameMatroid_iff_isLiftMatroid
    (h : ∀ C D, Ω.IsUnbalancedCycle C → Ω.IsUnbalancedCycle D →
      ¬Disjoint (Ω.graph.verts C) (Ω.graph.verts D)) :
    Ω.IsFrameMatroid M ↔ Ω.IsLiftMatroid M :=
  ⟨fun hM => ⟨hM.1, fun X => (hM.2 X).trans (isFrameCircuit_iff_isLiftCircuit h X)⟩,
    fun hM => ⟨hM.1, fun X => (hM.2 X).trans (isFrameCircuit_iff_isLiftCircuit h X).symm⟩⟩

/-- **§2.2.** A balanced loop is a loop of `F(G, B)`, so contracting it is deleting it. -/
theorem IsFrameMatroid.contract_eq_delete (hM : Ω.IsFrameMatroid M) {e : E}
    (he : ({e} : Set E) ∈ Ω.balanced) : M.contract {e} = M.delete {e} :=
  Matroid.contract_eq_delete_of_subset_loops (Set.singleton_subset_iff.2
    (Matroid.isLoop_iff.1 (Matroid.singleton_isCircuit.1 ((hM.2 _).2 (Or.inl he)))))

/-- **§2.2.** A balanced loop is a loop of `L(G, B)`, so contracting it is deleting it. -/
theorem IsLiftMatroid.contract_eq_delete (hM : Ω.IsLiftMatroid M) {e : E}
    (he : ({e} : Set E) ∈ Ω.balanced) : M.contract {e} = M.delete {e} :=
  Matroid.contract_eq_delete_of_subset_loops (Set.singleton_subset_iff.2
    (Matroid.isLoop_iff.1 (Matroid.singleton_isCircuit.1 ((hM.2 _).2 (Or.inl he)))))

/-- **§2.2** (Zaslavsky). The frame circuits of a finite biased graph are the circuits of a
matroid, `F(G, B)`. -/
theorem exists_isFrameMatroid [Finite V] [Finite E] (Ω : BiasedGraph V E) :
    ∃ M, Ω.IsFrameMatroid M := by
  sorry

/-- **§1** (Zaslavsky). The canonical frame matrix of a realization represents `F(G, B)`. -/
theorem Realizes.exists_isFrameMatroid_frameMatrix [DecidableEq V] [Finite V] [Finite E]
    {F : Type*} [Field F] {φ : E → Fˣ} (h : Ω.Realizes φ) :
    ∃ M, Ω.IsFrameMatroid M ∧ IsMatrixRepresentation M (frameMatrix Ω.graph φ) := by
  sorry

/-- **§1** (Zaslavsky). The canonical lift matrix of a realization represents `L(G, B)`. -/
theorem AddRealizes.exists_isLiftMatroid_liftMatrix [DecidableEq V] [Finite V] [Finite E]
    {F : Type*} [Field F] {ψ : E → F} (h : Ω.AddRealizes ψ) :
    ∃ M, Ω.IsLiftMatroid M ∧ IsMatrixRepresentation M (liftMatrix Ω.graph ψ) := by
  sorry

/-- **§2.2** (Zaslavsky). Deleting and contracting links in `(G, B)` is deleting and contracting
them in `F(G, B)`. -/
theorem LinkMinorMap.isFrameMatroid [Finite V] [Finite E] {Ω' : BiasedGraph V' E'}
    (m : Ω.LinkMinorMap Ω') (hM : Ω.IsFrameMatroid M) :
    Ω'.IsFrameMatroid
      (((M.contract m.contractSet).delete m.deleteSet).comap fun e' => (m.edge e' : E)) := by
  sorry

/-- **§2.2** (Zaslavsky). Deleting and contracting links in `(G, B)` is deleting and contracting
them in `L(G, B)`. -/
theorem LinkMinorMap.isLiftMatroid [Finite V] [Finite E] {Ω' : BiasedGraph V' E'}
    (m : Ω.LinkMinorMap Ω') (hM : Ω.IsLiftMatroid M) :
    Ω'.IsLiftMatroid
      (((M.contract m.contractSet).delete m.deleteSet).comap fun e' => (m.edge e' : E)) := by
  sorry

/-- **§2.2** (Zaslavsky). `F(G, B) / e = F((G, B) / e)` for a joint `e`, by the frame-type
contraction. -/
theorem IsFrameContraction.isFrameMatroid [Finite V] [Finite E] {e : E} {v : V}
    {Ω₂ : BiasedGraph V ↥({e}ᶜ : Set E)} (h : Ω.IsFrameContraction e v Ω₂)
    (hM : Ω.IsFrameMatroid M) :
    Ω₂.IsFrameMatroid ((M.contract {e}).comap (Subtype.val : ↥({e}ᶜ : Set E) → E)) := by
  sorry

/-- **§2.2** (Zaslavsky). `L(G, B) / e = M(G \ e)` for a joint `e`: the lift-type contraction has
graph `G \ e` and all its cycles balanced. -/
theorem IsLiftContraction.isLiftMatroid [Finite V] [Finite E] {e : E}
    {Ω₂ : BiasedGraph V ↥({e}ᶜ : Set E)} (h : Ω.IsLiftContraction e Ω₂)
    (hM : Ω.IsLiftMatroid M) :
    Ω₂.IsLiftMatroid ((M.contract {e}).comap (Subtype.val : ↥({e}ᶜ : Set E) → E)) := by
  sorry

end BiasedGraph

open BiasedGraph

theorem triangleBiased_isFrameMatroid : triangleBiased.IsFrameMatroid twoOfThree :=
  triangleBiased_isBalanced.isFrameMatroid rfl twoOfThree_isCircuit_iff

theorem triangleBiased_isLiftMatroid : triangleBiased.IsLiftMatroid twoOfThree :=
  triangleBiased_isBalanced.isLiftMatroid rfl twoOfThree_isCircuit_iff

/-- The two loops at the one vertex are unbalanced and share it. -/
theorem twoLoops_not_disjoint (C D : Set Bool) (hC : twoLoops.contrabalanced.IsUnbalancedCycle C)
    (hD : twoLoops.contrabalanced.IsUnbalancedCycle D) :
    ¬Disjoint (twoLoops.contrabalanced.graph.verts C) (twoLoops.contrabalanced.graph.verts D) := by
  intro hd
  obtain ⟨c, hc⟩ := hC.1.nonempty
  obtain ⟨d, hd'⟩ := hD.1.nonempty
  exact Set.disjoint_left.1 hd (mem_verts.2 ⟨c, hc, Or.inl rfl⟩) (mem_verts.2 ⟨d, hd', Or.inl rfl⟩)

example := triangleBiased_isFrameMatroid.unique triangleBiased_isFrameMatroid
example := triangleBiased_isLiftMatroid.unique triangleBiased_isLiftMatroid

example := isFrameCircuit_iff_isLiftCircuit twoLoops_not_disjoint {true}

example (M : Matroid Bool) := isFrameMatroid_iff_isLiftMatroid (M := M) twoLoops_not_disjoint

example := triangleBiased.exists_isFrameMatroid

example := twoLoops_realizes.exists_isFrameMatroid_frameMatrix

example := AddRealizes.exists_isLiftMatroid_liftMatrix
  (Ω := triangle.addGainBiased (![1, 0, 1] : Fin 3 → ℚ)) rfl

example := (LinkMinorMap.refl triangleBiased).isFrameMatroid triangleBiased_isFrameMatroid

example := (LinkMinorMap.refl triangleBiased).isLiftMatroid triangleBiased_isLiftMatroid

example : ∃ M : Matroid Bool, M.contract {true} = M.delete {true} := by
  obtain ⟨M, hM⟩ := (twoLoops.gainBiased (1 : Bool → ℚˣ)).exists_isFrameMatroid
  exact ⟨M, hM.contract_eq_delete ((singleton_mem_balancedCycles_iff _ rfl).2 rfl)⟩

example : ∃ M, twoLoopsContract.IsFrameMatroid M := by
  obtain ⟨M, hM⟩ := twoLoops.contrabalanced.exists_isFrameMatroid
  exact ⟨_, twoLoops_isFrameContraction.isFrameMatroid hM⟩

end Transformer.GainMatroid
