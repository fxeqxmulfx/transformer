module

public import Transformer.GainMatroid.Section2_BiasedMatroids
public import Transformer.GainMatroid.Section2_ExchangeForms
public import Transformer.GainMatroid.Section2_FrameContractIndep
public import Transformer.GainMatroid.Section2_LiftColumns
public import Transformer.GainMatroid.Section2_LiftContraction
public import Transformer.GainMatroid.Section2_RealizedIndep
public import Transformer.GainMatroid.Section4_Switching

/-!
# The canonical matrices represent the frame and lift matroids

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1: "Zaslavsky showed that every such gain graph
yields a canonical matrix representation `A` for `M` over `𝔽`": the canonical frame matrix of a
gain function realizing `(G, B)` represents `F(G, B)`, and the canonical lift matrix of an
additive one represents `L(G, B)` (`BiasedGraph.Realizes.exists_isFrameMatroid_frameMatrix`,
`BiasedGraph.AddRealizes.exists_isLiftMatroid_liftMatrix`).

For a set `I` of edges, take a forest `T ⊆ I` linking the ends of every edge of `I`
(`OrientedGraph.exists_spanning_forest`) and switch the gains to one, or zero, on `T`
(Proposition 2.1). Switching keeps the realization and, up to projective equivalence, the matrix,
so its independent sets of columns. Then the columns of `I` are independent
(`OrientedGraph.linearIndepOn_frameMatrix_iff`, `OrientedGraph.linearIndepOn_liftMatrix_iff`)
exactly when `I` is independent in the matroid of `(G, B)`
(`BiasedGraph.Realizes.isFrameIndep_iff_of_isForest`,
`BiasedGraph.AddRealizes.isLiftIndep_iff_of_isForest`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E}

/-- **§1** (Zaslavsky). The canonical frame matrix of a realization represents `F(G, B)`. -/
theorem Realizes.exists_isFrameMatroid_frameMatrix [DecidableEq V] [Finite V] [Finite E]
    {F : Type*} [Field F] {φ : E → Fˣ} (h : Ω.Realizes φ) :
    ∃ M, Ω.IsFrameMatroid M ∧ IsMatrixRepresentation M (frameMatrix Ω.graph φ) := by
  obtain ⟨M, hM⟩ := Ω.exists_isFrameMatroid
  refine ⟨M, hM, hM.1, fun I => ?_⟩
  obtain ⟨T, hTI, hT, hr⟩ := exists_spanning_forest (G := Ω.graph) I
  obtain ⟨η, hη⟩ := exists_switch_eq_one_of_isForest hT φ
  have := Fintype.ofFinite V
  have := Fintype.ofFinite E
  classical
  rw [hM.indep_iff, (projEquiv_frameMatrix_of_switchEquiv (G := Ω.graph) (φ := φ)
      ⟨η, rfl⟩).linearIndepOn_iff I,
    linearIndepOn_frameMatrix_iff hTI hT hr hη,
    (h.switch η).isFrameIndep_iff_of_isForest hTI hT hr hη]

/-- **§1** (Zaslavsky). The canonical lift matrix of a realization represents `L(G, B)`. -/
theorem AddRealizes.exists_isLiftMatroid_liftMatrix [DecidableEq V] [Finite V] [Finite E]
    {F : Type*} [Field F] {ψ : E → F} (h : Ω.AddRealizes ψ) :
    ∃ M, Ω.IsLiftMatroid M ∧ IsMatrixRepresentation M (liftMatrix Ω.graph ψ) := by
  obtain ⟨M, hM⟩ := Ω.exists_isLiftMatroid
  refine ⟨M, hM, hM.1, fun I => ?_⟩
  obtain ⟨T, hTI, hT, hr⟩ := exists_spanning_forest (G := Ω.graph) I
  obtain ⟨η, hη⟩ := exists_addSwitch_eq_zero_of_isForest hT ψ
  have := Fintype.ofFinite V
  have := Fintype.ofFinite E
  classical
  rw [hM.indep_iff, (projEquiv_liftMatrix_of_switchScaleEquiv (G := Ω.graph) (φ := ψ)
      (ψ := GainMatroid.addSwitch Ω.graph η ψ)
      ⟨η, 1, by rw [Units.val_one, one_smul]⟩).linearIndepOn_iff I,
    linearIndepOn_liftMatrix_iff hTI hT hr hη,
    (h.addSwitch η).isLiftIndep_iff_of_isForest hTI hT hr hη]

end BiasedGraph

open BiasedGraph

example := twoLoops_realizes.exists_isFrameMatroid_frameMatrix

example := AddRealizes.exists_isLiftMatroid_liftMatrix
  (Ω := triangle.addGainBiased (![1, 0, 1] : Fin 3 → ℚ)) rfl

end Transformer.GainMatroid
