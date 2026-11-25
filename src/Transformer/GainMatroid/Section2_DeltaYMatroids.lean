module

public import Transformer.GainMatroid.Section2_DeltaDual
public import Transformer.GainMatroid.Section2_ExchangeAgree
public import Transformer.GainMatroid.Section2_ExchangeDelta
public import Transformer.GainMatroid.Section2_ExchangeMatroids

/-!
# Proposition 2.8: exchanges of biased graphs and of their matroids

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, Proposition 2.8. The source: "(i) If `X` is a
balanced triangle of `(G, B)`, then `F(Δ_X(G, B)) = Δ_X F(G, B)` and `L(Δ_X(G, B)) = Δ_X L(G, B)`.
(ii) Let `Y` be a `K_{1,3}`-subgraph of `G` for which, if `(G, B)` is unbalanced, `(G, B) − E(Y)`
remains unbalanced. Then `F(∇_Y(G, B)) = ∇_Y F(G, B)` and `L(∇_Y(G, B)) = ∇_Y L(G, B)`."

The source defines `∇_Y M` only for a triad `Y` of `M`. The printed hypothesis of (ii) does not make
`Y` a triad (`nabla_hypothesis_not_triad`), so the statements of (ii) add that hypothesis; given
it, the printed hypothesis on `(G, B) − E(Y)` is not needed, and the statements of (ii) drop it. The
source assumes matroids as in Oxley's text (§2), which are finite, so these statements take
`Finite V` and `Finite E`.

The proofs: `Ω` and `Δ_X Ω`, and `∇_Y Ω` and `Ω`, are biased graphs of exchange pairs whose balanced
cycles agree across the exchange (`IsDeltaExchange.balanceAgree`, `IsNablaExchange.balanceAgree`),
so their frame and lift matroids differ by `Δ` of the balanced triangle
(`IsExchange.isDeltaMatroid_frame`, `IsExchange.isDeltaMatroid_lift`). For (ii) this gives
`F(G, B) = Δ_Y F(∇_Y(G, B))`, which `∇_Y` undoes at a triad (`IsDeltaMatroid.isNablaMatroid`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

open scoped symmDiff

namespace BiasedGraph

variable {V E : Type*} [DecidableEq E] [Finite V] [Finite E] {Ω : BiasedGraph V E} {a b c : E}
  {x y z : V} {M M' : Matroid E}

/-- **Proposition 2.8 (i)**, frame case: `F(Δ_X(G, B)) = Δ_X F(G, B)`. -/
theorem IsDeltaExchange.isDeltaMatroid_frame {Ω' : BiasedGraph (Option V) E}
    (h : Ω.IsDeltaExchange a b c x y z Ω') (hM : Ω.IsFrameMatroid M)
    (hM' : Ω'.IsFrameMatroid M') : IsDeltaMatroid M a b c M' :=
  h.isExchange.isDeltaMatroid_frame h.balanceAgree h.balanced hM hM'

/-- **Proposition 2.8 (i)**, lift case: `L(Δ_X(G, B)) = Δ_X L(G, B)`. -/
theorem IsDeltaExchange.isDeltaMatroid_lift {Ω' : BiasedGraph (Option V) E}
    (h : Ω.IsDeltaExchange a b c x y z Ω') (hM : Ω.IsLiftMatroid M)
    (hM' : Ω'.IsLiftMatroid M') : IsDeltaMatroid M a b c M' :=
  h.isExchange.isDeltaMatroid_lift h.balanceAgree h.balanced hM hM'

/-- **Proposition 2.8 (ii)**, frame case: `F(∇_Y(G, B)) = ∇_Y F(G, B)`. The hypothesis `htriad` is
added, since the source defines `∇_Y M` only for a triad (`nabla_hypothesis_not_triad`). Given it,
the source's hypothesis that `(G, B) − E(Y)` remains unbalanced if `(G, B)` is unbalanced is not
needed, and is dropped. -/
theorem IsNablaExchange.isNablaMatroid_frame {v : V} {Ω' : BiasedGraph V E}
    (h : Ω.IsNablaExchange v a b c x y z Ω')
    (hM : Ω.IsFrameMatroid M) (hM' : Ω'.IsFrameMatroid M') (htriad : M✶.IsCircuit {a, b, c}) :
    IsNablaMatroid M a b c M' :=
  (h.isExchange.isDeltaMatroid_frame h.balanceAgree h.mem_balanced hM' hM).isNablaMatroid htriad

/-- **Proposition 2.8 (ii)**, lift case: `L(∇_Y(G, B)) = ∇_Y L(G, B)`. The hypothesis `htriad` is
added, since the source defines `∇_Y M` only for a triad (`nabla_hypothesis_not_triad`). Given it,
the source's hypothesis that `(G, B) − E(Y)` remains unbalanced if `(G, B)` is unbalanced is not
needed, and is dropped. -/
theorem IsNablaExchange.isNablaMatroid_lift {v : V} {Ω' : BiasedGraph V E}
    (h : Ω.IsNablaExchange v a b c x y z Ω')
    (hM : Ω.IsLiftMatroid M) (hM' : Ω'.IsLiftMatroid M') (htriad : M✶.IsCircuit {a, b, c}) :
    IsNablaMatroid M a b c M' :=
  (h.isExchange.isDeltaMatroid_lift h.balanceAgree h.mem_balanced hM' hM).isNablaMatroid htriad

end BiasedGraph

/-- The star of `K_{1,3}` is not a triad of `U_{3,3}`: every element is a coloop. -/
theorem freeThree_not_triad : ¬freeThree✶.IsCircuit {1, 2, 0} := by
  have h : ({1, 2, 0} : Set (Fin 3)) = (({1, 2, 0} : Finset (Fin 3)) : Set (Fin 3)) := by simp
  rw [h]
  intro hc
  have := (matroidOfFinset_dual_isCircuit_iff _).1 hc
  revert this
  decide +kernel

/-- The printed hypothesis of **Proposition 2.8 (ii)** does not make `Y` a triad. On the balanced
`K_{1,3}` it holds vacuously, and the frame and lift matroid `U_{3,3}` has no triad, so the
source's `∇_Y M` is not defined there. -/
theorem nabla_hypothesis_not_triad :
    ∃ (Ω : BiasedGraph (Option (Fin 3)) (Fin 3)) (M : Matroid (Fin 3)),
      Ω.graph.IsLabelledClaw none 1 2 0 (some 0) (some 1) (some 2) ∧ Ω.IsBalanced ∧
        Ω.IsFrameMatroid M ∧ Ω.IsLiftMatroid M ∧ ¬M✶.IsCircuit {1, 2, 0} := by
  have hb : triangleStar.contrabalanced.IsBalanced := fun C hC =>
    (triangleStar_not_isCycle C hC).elim
  refine ⟨triangleStar.contrabalanced, freeThree, ?_, hb,
    hb.isFrameMatroid rfl freeThree_isCircuit_iff, hb.isLiftMatroid rfl freeThree_isCircuit_iff,
    freeThree_not_triad⟩
  refine ⟨Or.inr ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, Or.inr ⟨rfl, rfl⟩, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, fun e _ => ?_⟩
  revert e
  decide

/-- The members of `nablaFamily` for the balanced `clawGraph`. -/
def clawFamily : Finset (Finset (Fin 5)) :=
  {{0, 1, 3}, {0, 2, 4}, {1, 3, 4, 2}, {2, 3}, {1, 4}, {0, 3, 4}}

set_option maxRecDepth 100000 in
set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
private theorem claw_family_facts :
    (∀ C ∈ nablaCycles, C = ({0, 1, 2} : Finset (Fin 5)) ∨
      (C ∈ clawFamily ∧ ∀ D ∈ clawFamily, D ⊆ C → C ⊆ D)) ∧
    (∀ C ∈ clawFamily, (C ∈ clawCycles ∧ (C ∩ {0, 1, 2}).card = 2) ∨
      ∃ K ∈ clawCycles, (K ∩ {0, 1, 2}).card = 2 ∧ C = symmDiff K {0, 1, 2}) ∧
    (∀ K ∈ clawCycles, K ∈ clawFamily ∧ symmDiff K {0, 1, 2} ∈ clawFamily) := by
  decide +kernel

private theorem ncard_inter_coe {α : Type*} [DecidableEq α] (C Y : Finset α) :
    ((C : Set α) ∩ (Y : Set α)).ncard = (C ∩ Y).card := by
  rw [← Finset.coe_inter, Set.ncard_coe_finset]

private theorem coe_012 :
    ({0, 1, 2} : Set (Fin 5)) = (({0, 1, 2} : Finset (Fin 5)) : Set (Fin 5)) := by
  simp

private theorem coe_mem_nablaFamily {C : Finset (Fin 5)} (hC : C ∈ clawFamily) :
    (C : Set (Fin 5)) ∈ nablaFamily (clawGraph.balancedCycles (1 : Fin 5 → ℚˣ)) {0, 1, 2} := by
  rw [coe_012]
  rcases claw_family_facts.2.1 C hC with ⟨hK, hc⟩ | ⟨K, hK, hc, rfl⟩
  · exact Or.inl ⟨mem_balancedCycles_one.2 ((clawGraph_isCycle_iff C).2 hK),
      Or.inr (by rw [ncard_inter_coe, hc])⟩
  · refine Or.inr ⟨K, mem_balancedCycles_one.2 ((clawGraph_isCycle_iff K).2 hK), ?_, ?_⟩
    · rw [ncard_inter_coe, hc]
    · exact Finset.coe_symmDiff

private theorem exists_coe_of_mem_nablaFamily {D : Set (Fin 5)}
    (hD : D ∈ nablaFamily (clawGraph.balancedCycles (1 : Fin 5 → ℚˣ)) {0, 1, 2}) :
    ∃ D' ∈ clawFamily, D = (D' : Set (Fin 5)) := by
  rcases hD with ⟨hD, -⟩ | ⟨K, hK, -, rfl⟩
  · obtain ⟨D', rfl⟩ := (Set.toFinite D).exists_finset_coe
    exact ⟨D', (claw_family_facts.2.2 D'
      ((clawGraph_isCycle_iff D').1 (mem_balancedCycles_one.1 hD))).1, rfl⟩
  · obtain ⟨K, rfl⟩ := (Set.toFinite K).exists_finset_coe
    refine ⟨symmDiff K {0, 1, 2}, (claw_family_facts.2.2 K
      ((clawGraph_isCycle_iff K).1 (mem_balancedCycles_one.1 hK))).2, ?_⟩
    rw [coe_012, Finset.coe_symmDiff]

/-- For the balanced `clawGraph`, `∇_Y B` is the set of all six cycles of `clawNabla`. -/
theorem clawNabla_balanced_eq :
    clawNabla.balancedCycles (1 : Fin 5 → ℚˣ) =
      nablaBalanced clawNabla (clawGraph.balancedCycles (1 : Fin 5 → ℚˣ)) {0, 1, 2} := by
  ext C
  refine ⟨fun hC => ?_, fun hC => mem_balancedCycles_one.2 hC.1⟩
  have hcyc : clawNabla.IsCycle C := mem_balancedCycles_one.1 hC
  refine ⟨hcyc, ?_⟩
  obtain ⟨C, rfl⟩ := (Set.toFinite C).exists_finset_coe
  rcases claw_family_facts.1 C ((clawNabla_isCycle_iff C).1 hcyc) with rfl | ⟨hf, hmin⟩
  · exact Or.inl (by simp)
  · refine Or.inr ⟨⟨coe_mem_nablaFamily hf, hcyc.nonempty⟩, ?_⟩
    rintro D ⟨hD, -⟩ hle
    obtain ⟨D', hD', rfl⟩ := exists_coe_of_mem_nablaFamily hD
    exact Finset.coe_subset.2 (hmin D' hD' (Finset.coe_subset.1 hle))

/-- The balanced `clawNabla` is `∇_Y` of the balanced `clawGraph`. -/
theorem claw_isNablaExchange :
    (clawGraph.gainBiased (1 : Fin 5 → ℚˣ)).IsNablaExchange 0 0 1 2 1 2 3
      (clawNabla.gainBiased (1 : Fin 5 → ℚˣ)) := by
  refine ⟨clawGraph_isLabelledClaw, clawGraph_nablaGraph.symm, ?_⟩
  change clawNabla.balancedCycles _ = nablaBalanced (clawGraph.nablaGraph 0 1 2 1 2 3) _ _
  rw [clawGraph_nablaGraph]
  exact clawNabla_balanced_eq

example := triangle_isDeltaExchange.isDeltaMatroid_frame
  (triangleBiased_isBalanced.isFrameMatroid rfl twoOfThree_isCircuit_iff)
  (triangleStar_isBalanced.isFrameMatroid rfl freeThree_isCircuit_iff')

example := triangle_isDeltaExchange.isDeltaMatroid_lift
  (triangleBiased_isBalanced.isLiftMatroid rfl twoOfThree_isCircuit_iff)
  (triangleStar_isBalanced.isLiftMatroid rfl freeThree_isCircuit_iff')

example := claw_isNablaExchange.isNablaMatroid_frame
  ((gainBiased_one_isBalanced clawGraph).isFrameMatroid rfl clawMatroid_isCircuit_iff)
  ((gainBiased_one_isBalanced clawNabla).isFrameMatroid rfl nablaMatroid_isCircuit_iff)
  clawMatroid_triad

example := claw_isNablaExchange.isNablaMatroid_lift
  ((gainBiased_one_isBalanced clawGraph).isLiftMatroid rfl clawMatroid_isCircuit_iff)
  ((gainBiased_one_isBalanced clawNabla).isLiftMatroid rfl nablaMatroid_isCircuit_iff)
  clawMatroid_triad

end Transformer.GainMatroid
