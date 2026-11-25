module

public import Transformer.GainMatroid.Section2_ConnectionExamples
public import Transformer.GainMatroid.Section2_DeltaYMatroids
public import Transformer.GainMatroid.Section2_ExchangeAgree
public import Transformer.GainMatroid.Section2_ExchangeDelta
public import Transformer.GainMatroid.Section2_NablaExamples

/-!
# Examples for the exchange pairs of Proposition 2.8

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. The lemmas on exchange pairs applied to the
`Δ_X` of the balanced triangle (`triangle_isDeltaExchange`), to the `∇_Y` of the balanced claw
(`claw_isNablaExchange`), whose frame matroids give `nablaMatroid_isDeltaMatroid`, and to the
`∇_Y` of the balanced `clawPlus` (`clawPlusNablaBiased_isNablaExchange`), whose cycles `{3, 4}` off
`Y` and `{1, 0, 3}` off `c = 2` carry the lemmas on cycles off `X` and off `c`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

/-- `∇_Y` of the balanced `clawPlus`. -/
noncomputable def clawPlusNablaBiased : BiasedGraph (Fin 4) (Fin 7) :=
  ((clawPlus.gainBiased (1 : Fin 7 → ℚˣ)).exists_isNablaExchange clawPlus_isLabelledClaw).choose

theorem clawPlusNablaBiased_isNablaExchange :
    (clawPlus.gainBiased (1 : Fin 7 → ℚˣ)).IsNablaExchange 0 0 1 2 1 2 3 clawPlusNablaBiased :=
  ((clawPlus.gainBiased (1 : Fin 7 → ℚˣ)).exists_isNablaExchange
    clawPlus_isLabelledClaw).choose_spec

theorem plusEx : IsExchange clawPlusNablaBiased.graph (clawPlus.gainBiased (1 : Fin 7 → ℚˣ)).graph
    id 0 0 1 2 1 2 3 :=
  clawPlusNablaBiased_isNablaExchange.isExchange

theorem plusAgree :
    clawPlusNablaBiased.BalanceAgree (clawPlus.gainBiased (1 : Fin 7 → ℚˣ)) {0, 1, 2} :=
  clawPlusNablaBiased_isNablaExchange.balanceAgree

theorem plusT_isCycle₃₄ : clawPlusNablaBiased.graph.IsCycle {3, 4} := by
  rw [clawPlusNablaBiased_isNablaExchange.graph_eq]
  exact clawPlus_nablaGraph ▸ plusNabla_isCycle₃₄

theorem plusOff_isCycle₀ : (plusEx.offGraph ({0, 1, 2} : Set (Fin 7))ᶜ).IsCycle
    (Subtype.val ⁻¹' {3, 4}) :=
  plusEx.isCycle_offGraph (by rintro e (rfl | rfl) <;> simp) plusT_isCycle₃₄

theorem plusOff_isCycle : (plusEx.offGraph ({2} : Set (Fin 7))ᶜ).IsCycle (Subtype.val ⁻¹' {3, 4}) :=
  plusEx.isCycle_offGraph (by rintro e (rfl | rfl) <;> simp) plusT_isCycle₃₄

theorem triangleStar_isFrameMatroid :
    (triangle.deltaGraph 1 2 0 0 1 2).contrabalanced.IsFrameMatroid freeThree :=
  triangleStar_isBalanced.isFrameMatroid rfl freeThree_isCircuit_iff'

theorem clawNabla_isFrameMatroid :
    (clawNabla.gainBiased (1 : Fin 5 → ℚˣ)).IsFrameMatroid nablaMatroid :=
  (gainBiased_one_isBalanced clawNabla).isFrameMatroid rfl nablaMatroid_isCircuit_iff

theorem clawGraph_isFrameMatroid :
    (clawGraph.gainBiased (1 : Fin 5 → ℚˣ)).IsFrameMatroid clawMatroid :=
  (gainBiased_one_isBalanced clawGraph).isFrameMatroid rfl clawMatroid_isCircuit_iff

/-- The frame matroid of the balanced claw is `Δ_Y` of that of its `∇_Y`. -/
theorem nablaMatroid_isDeltaMatroid : IsDeltaMatroid nablaMatroid 0 1 2 clawMatroid :=
  claw_isNablaExchange.isExchange.isDeltaMatroid_frame claw_isNablaExchange.balanceAgree
    claw_isNablaExchange.mem_balanced clawNabla_isFrameMatroid clawGraph_isFrameMatroid

example := ReachableIn.eq_of_empty (reachableIn_refl (G := triangle) ∅ (0 : Fin 3))

example := ReachableIn.mem_verts_of_ne (reachableIn_of_mem (G := triangle) (Set.mem_singleton 0))
  (by decide)

example := Joins.contract_mk (K := (∅ : Set (Fin 3)))
  (Or.inl ⟨rfl, rfl⟩ : triangle.Joins 0 (triangle.tail 0) (triangle.head 0))
  ⟨0, Set.notMem_empty 0⟩ rfl

example := swap_ne (a := (0 : Fin 3)) (b := 1) (c := 2) (e := 0) (by decide) (by decide) (by decide)

example := plusEx.isCycle_val plusOff_isCycle

example := plusEx.isCycle_offGraph (D := Set.univ) (Set.subset_univ _) plusT_isCycle₃₄

example := plusEx.iota_z_ne

example := plusEx.eq_of_mem_verts_c (p := 3) (Or.inr ⟨2, Set.mem_singleton 2, rfl⟩) (by decide)

example := plusEx.isForest_c

example := plusEx.class_v_eq

example := plusEx.tEmbedding_image Set.univ ∅

example := plusEx.tEmbedding_balanced_iff Set.univ ∅

example := plusEx.tEmbedding_carry Set.univ ∅

example := plusEx.sEmbedding₀_image ∅

example := plusEx.sEmbedding₀_balanced_iff ∅

example := plusEx.sEmbedding₀_carry ∅

example := plusEx.joins_swap ⟨3, by simp⟩ ⟨3, by simp⟩
  (Equiv.swap_apply_of_ne_of_ne (by decide) (by decide)).symm

example := plusEx.sEmbedding_image ∅

example := plusEx.sEmbedding_carry ∅

example := image_swap_eq_self (a := (0 : Fin 3)) (b := 1) (D := ∅)
  (iff_of_false (Set.notMem_empty _) (Set.notMem_empty _))

example := plusEx.ncard_inter_of_isCycle clawPlus_isCycle₃₄

example := plusEx.exists_swap_mem_iff plusAgree plusT_isCycle₃₄ (by simp)

example := plusEx.sEmbedding₀_balanced_iff_tEmbedding plusAgree plusOff_isCycle₀

example := plusEx.sEmbedding_balanced_iff_tEmbedding plusAgree plusOff_isCycle

example := OrientedGraph.IsExchange.image_preimage_of_disjoint (a := (0 : Fin 3)) (b := 1) (c := 2)
  (Set.empty_disjoint _)

example := OrientedGraph.IsExchange.image_preimage_of_notMem (c := (2 : Fin 3)) (Set.notMem_empty 2)

example := plusEx.isFrameIndep_iff_of_disjoint plusAgree (Set.empty_disjoint _)

example := plusEx.isLiftIndep_iff_of_disjoint plusAgree (Set.empty_disjoint _)

example := plusEx.isFrameIndep_insert_swap_iff plusAgree (Set.notMem_empty 2)

example := plusEx.isLiftIndep_insert_swap_iff plusAgree (Set.notMem_empty 2)

example := triangle_isDeltaExchange.isExchange.isDeltaMatroid_frame
  triangle_isDeltaExchange.balanceAgree triangle_isDeltaExchange.balanced
  triangleBiased_isFrameMatroid triangleStar_isFrameMatroid

example := triangle_isDeltaExchange.isExchange.isDeltaMatroid_lift
  triangle_isDeltaExchange.balanceAgree triangle_isDeltaExchange.balanced
  triangleBiased_isLiftMatroid (triangleStar_isBalanced.isLiftMatroid rfl freeThree_isCircuit_iff')

example := triangle_isDeltaExchange.isExchange

example := triangle_isDeltaExchange.balanceAgree

example := claw_isNablaExchange.isExchange

example := claw_isNablaExchange.balanced_eq_nablaBalanced

example := claw_isNablaExchange.mem_balanced

example := claw_isNablaExchange.balanceAgree

example := isDeltaMatroid_of_indep_iff
  ((triangleBiased_isFrameMatroid.2 _).2 (Or.inl triangle_isDeltaExchange.balanced))
  (by decide) (by decide) (by decide) (M' := freeThree) rfl
  (fun _ hI => by
    rw [triangleStar_isFrameMatroid.indep_iff, triangleBiased_isFrameMatroid.indep_iff]
    exact triangle_isDeltaExchange.isExchange.isFrameIndep_iff_of_disjoint
      triangle_isDeltaExchange.balanceAgree hI)
  (fun _ hI => by
    rw [triangleStar_isFrameMatroid.indep_iff, triangleBiased_isFrameMatroid.indep_iff]
    exact triangle_isDeltaExchange.isExchange.isFrameIndep_insert_swap_iff
      triangle_isDeltaExchange.balanceAgree hI)
  (fun _ hI => by
    rw [triangleStar_isFrameMatroid.indep_iff, triangleBiased_isFrameMatroid.indep_iff]
    exact triangle_isDeltaExchange.isExchange.swap₂₃.isFrameIndep_insert_swap_iff
      (by rw [triple_acb_eq]; exact triangle_isDeltaExchange.balanceAgree) hI)
  (fun _ hI => by
    rw [triangleStar_isFrameMatroid.indep_iff, triangleBiased_isFrameMatroid.indep_iff]
    exact triangle_isDeltaExchange.isExchange.swap₁₂.swap₂₃.isFrameIndep_insert_swap_iff
      (by rw [triple_bca_eq]; exact triangle_isDeltaExchange.balanceAgree) hI)

example := nablaMatroid_isDeltaMatroid.eq_comap

example := nablaMatroid_isDeltaMatroid.natRk_eq (Set.empty_subset _)

example := nablaMatroid_isDeltaMatroid.dual_indep_iff_of_disjoint (Set.empty_disjoint _)

example := nablaMatroid_isDeltaMatroid.dual_indep_insert_swap_iff (s := 2) (u := 0) (w := 1) rfl
  (by decide) (by decide) (by decide) (Set.notMem_empty 2)

example := nablaMatroid_isDeltaMatroid.dual clawMatroid_triad

example := nablaMatroid_isDeltaMatroid.isNablaMatroid clawMatroid_triad

end Transformer.GainMatroid
