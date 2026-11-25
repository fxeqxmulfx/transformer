module

public import Transformer.GainMatroid.Section2_FrameContractIndep
public import Transformer.GainMatroid.Section2_LinkMinorIndep

/-!
# Link minors of frame and lift matroids

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2: "Minors of biased graphs and their matroids
agree: for any edge `e` of a biased graph `(G, B)`, `F(G, B) \ e = F((G, B) \ e)`,
`F(G, B)/e = F((G, B)/e)`, `L(G, B) \ e = L((G, B) \ e)`, for any link `e`,
`L(G, B)/e = L((G, B)/e)`", citing Zaslavsky. For a link minor `Ω / K \ D`, deleting `D` and
contracting the forest `K` of links at once, this is proved here
(`BiasedGraph.LinkMinorMap.isFrameMatroid`, `BiasedGraph.LinkMinorMap.isLiftMatroid`).

The forest `K` is independent in both matroids, so a set `I` of edges of the minor is independent in
`F(G, B) / K \ D` when `I ∪ K` is independent in `F(G, B)`, which says that `I` is independent in
the frame matroid of the minor (`BiasedGraph.LinkMinorMap.isFrameIndep_iff`); likewise for the lift
matroid (`BiasedGraph.LinkMinorMap.isLiftIndep_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E V' E' : Type*} {Ω : BiasedGraph V E} {M : Matroid E}

namespace LinkMinorMap

variable {Ω' : BiasedGraph V' E'} (m : Ω.LinkMinorMap Ω')

/-- The edges of a link minor are carried outside the contracted and the deleted edges. -/
theorem edge_notMem (e' : E') :
    (m.edge e' : E) ∉ m.contractSet ∧ (m.edge e' : E) ∉ m.deleteSet := by
  have h : (m.edge e' : E) ∈ (m.contractSet ∪ m.deleteSet)ᶜ := m.range_edge ▸ ⟨e', rfl⟩
  exact ⟨fun hK => h (Or.inl hK), fun hD => h (Or.inr hD)⟩

/-- The minor of a matroid on the edges of `Ω` along a link minor has every edge of `Ω'`. -/
theorem comap_ground_eq (hM : M.E = Set.univ) :
    (((M.contract m.contractSet).delete m.deleteSet).comap fun e' => (m.edge e' : E)).E =
      Set.univ := by
  rw [Matroid.comap_ground_eq, Matroid.delete_ground, Matroid.contract_ground, hM]
  ext e'
  simpa using m.edge_notMem e'

/-- A set of edges of `Ω'` is independent in the minor of `M` exactly when its image together
with `K` is independent in `M`, when `K` is. -/
theorem comap_indep_iff (hK : M.Indep m.contractSet) {I : Set E'} :
    (((M.contract m.contractSet).delete m.deleteSet).comap fun e' => (m.edge e' : E)).Indep I ↔
      M.Indep ((fun e' => (m.edge e' : E)) '' I ∪ m.contractSet) := by
  have hinj : Set.InjOn (fun e' => (m.edge e' : E)) I :=
    (Subtype.val_injective.comp m.edge_injective).injOn
  have hdK : Disjoint ((fun e' => (m.edge e' : E)) '' I) m.contractSet := by
    rw [Set.disjoint_left]
    rintro _ ⟨e', -, rfl⟩
    exact (m.edge_notMem e').1
  have hdD : Disjoint ((fun e' => (m.edge e' : E)) '' I) m.deleteSet := by
    rw [Set.disjoint_left]
    rintro _ ⟨e', -, rfl⟩
    exact (m.edge_notMem e').2
  rw [Matroid.comap_indep_iff, Matroid.delete_indep_iff, hK.contract_indep_iff, and_iff_left hinj,
    and_iff_left hdD, and_iff_right hdK]

/-- The contracted forest of a link minor is frame-independent. -/
theorem isFrameIndep_contractSet : Ω.IsFrameIndep m.contractSet :=
  BiasedGraph.isFrameIndep_iff.2 ⟨fun C hCK hC => absurd hCK (m.isForest C hC),
    fun C _ hC _ hCK _ _ _ _ _ _ => absurd hCK (m.isForest C hC)⟩

/-- The contracted forest of a link minor is lift-independent. -/
theorem isLiftIndep_contractSet [Finite V] [Finite E] : Ω.IsLiftIndep m.contractSet :=
  BiasedGraph.isLiftIndep_iff.2 ⟨fun C hCK hC => absurd hCK (m.isForest C hC), by
    rw [nullity_eq_zero_iff.2 m.isForest]
    exact zero_le_one⟩

end LinkMinorMap

/-- **§2.2** (Zaslavsky). Deleting and contracting links in `(G, B)` is deleting and contracting
them in `F(G, B)`. -/
theorem LinkMinorMap.isFrameMatroid [Finite V] [Finite E] {Ω' : BiasedGraph V' E'}
    (m : Ω.LinkMinorMap Ω') (hM : Ω.IsFrameMatroid M) :
    Ω'.IsFrameMatroid
      (((M.contract m.contractSet).delete m.deleteSet).comap fun e' => (m.edge e' : E)) := by
  have : Finite V' := Finite.of_injective _ m.vertex_injective
  have : Finite E' := Finite.of_injective _ m.edge_injective
  obtain ⟨N, hN⟩ := Ω'.exists_isFrameMatroid
  convert hN using 1
  refine Matroid.ext_indep ((m.comap_ground_eq hM.1).trans hN.1.symm) fun I _ => ?_
  rw [m.comap_indep_iff (hM.indep_iff.2 m.isFrameIndep_contractSet), hM.indep_iff, hN.indep_iff,
    m.isFrameIndep_iff]

/-- **§2.2** (Zaslavsky). Deleting and contracting links in `(G, B)` is deleting and contracting
them in `L(G, B)`. -/
theorem LinkMinorMap.isLiftMatroid [Finite V] [Finite E] {Ω' : BiasedGraph V' E'}
    (m : Ω.LinkMinorMap Ω') (hM : Ω.IsLiftMatroid M) :
    Ω'.IsLiftMatroid
      (((M.contract m.contractSet).delete m.deleteSet).comap fun e' => (m.edge e' : E)) := by
  have : Finite V' := Finite.of_injective _ m.vertex_injective
  have : Finite E' := Finite.of_injective _ m.edge_injective
  obtain ⟨N, hN⟩ := Ω'.exists_isLiftMatroid
  convert hN using 1
  refine Matroid.ext_indep ((m.comap_ground_eq hM.1).trans hN.1.symm) fun I _ => ?_
  rw [m.comap_indep_iff (hM.indep_iff.2 m.isLiftIndep_contractSet), hM.indep_iff, hN.indep_iff,
    m.isLiftIndep_iff]

end BiasedGraph

open BiasedGraph

example := pumpkin_hasLinkMinor_twoLoops.some.edge_notMem true

example := pumpkin_hasLinkMinor_twoLoops.some.comap_ground_eq
  (Classical.choose_spec pumpkin.contrabalanced.exists_isFrameMatroid).1

example := pumpkin_hasLinkMinor_twoLoops.some.comap_indep_iff (I := {true})
  ((Classical.choose_spec pumpkin.contrabalanced.exists_isFrameMatroid).indep_iff.2
    pumpkin_hasLinkMinor_twoLoops.some.isFrameIndep_contractSet)

example := pumpkin_hasLinkMinor_twoLoops.some.isFrameIndep_contractSet

example := pumpkin_hasLinkMinor_twoLoops.some.isLiftIndep_contractSet

example := (LinkMinorMap.refl triangleBiased).isFrameMatroid triangleBiased_isFrameMatroid

example := (LinkMinorMap.refl triangleBiased).isLiftMatroid triangleBiased_isLiftMatroid

example := pumpkin_hasLinkMinor_twoLoops.some.isFrameMatroid
  (Classical.choose_spec pumpkin.contrabalanced.exists_isFrameMatroid)

example := pumpkin_hasLinkMinor_twoLoops.some.isLiftMatroid
  (Classical.choose_spec pumpkin.contrabalanced.exists_isLiftMatroid)

end Transformer.GainMatroid
