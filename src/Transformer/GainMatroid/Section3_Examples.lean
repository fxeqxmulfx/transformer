module

public import Transformer.GainMatroid.Section1_Theorem1
public import Transformer.GainMatroid.Section2_TwoConnected
public import Transformer.GainMatroid.Section3_Families

/-!
# Concrete biased graphs for the statements of §3

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §3. These meet the hypotheses of the statements of
§3. Two cycles of a loopless graph on three vertices meet (`OrientedGraph.not_disjoint_verts`),
so `2C₃` with the gains `twoC3Gain` is tangled (`twoC3Gain_isTangled`); its three 2-cycles meet
pairwise in one vertex. The tube has two vertex-disjoint 2-cycles. The contrabalanced `2C₃` has
realizations over `𝔽₇ˣ` and over `𝔽₇⁺` that are not switching (switching-and-scaling) equivalent,
as the gains of its 2-cycles show.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

local instance : Fact (Nat.Prime 7) := ⟨by decide⟩

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- A cycle of a loopless graph has at least two vertices. -/
theorem IsCycle.two_le_encard_verts (hG : G.Loopless) {C : Set E} (hC : G.IsCycle C) :
    2 ≤ (G.verts C).encard := by
  obtain ⟨u, W, hW, rfl⟩ := hC
  obtain ⟨e, he⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
  have h₁ : G.tail e ∈ G.verts W.edgeSet := mem_verts.2 ⟨e, he, Or.inl rfl⟩
  have h₂ : G.head e ∈ G.verts W.edgeSet := mem_verts.2 ⟨e, he, Or.inr rfl⟩
  rw [← Set.encard_pair (hG e)]
  exact Set.encard_le_encard (Set.insert_subset h₁ (Set.singleton_subset_iff.2 h₂))

/-- In a loopless graph on at most three vertices, any two cycles meet. -/
theorem not_disjoint_verts (hV : (Set.univ : Set V).encard ≤ 3) (hG : G.Loopless) {C D : Set E}
    (hC : G.IsCycle C) (hD : G.IsCycle D) : ¬Disjoint (G.verts C) (G.verts D) := fun h => by
  have h4 : (4 : ℕ∞) ≤ (G.verts C ∪ G.verts D).encard := by
    rw [Set.encard_union_eq h]
    exact le_trans (by norm_num)
      (add_le_add (hC.two_le_encard_verts hG) (hD.two_le_encard_verts hG))
  have h43 := (h4.trans (Set.encard_le_encard (Set.subset_univ _))).trans hV
  norm_num at h43

end OrientedGraph

/-- `2C₃` with the gains `twoC3Gain` is tangled. -/
theorem twoC3Gain_isTangled : (twoC3.gainBiased twoC3Gain).IsTangled :=
  ⟨twoC3Gain_isProperlyUnbalanced, fun _ _ hC hD =>
    not_disjoint_verts (by simp [Set.encard_univ]) twoC3_loopless hC.1 hD.1⟩

theorem twoC3_isCycle₀₁ : twoC3.IsCycle {0, 1} :=
  isCycle_pair (u := 0) (v := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

theorem twoC3_isCycle₂₃ : twoC3.IsCycle {2, 3} :=
  isCycle_pair (u := 1) (v := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

theorem twoC3_isCycle₄₅ : twoC3.IsCycle {4, 5} :=
  isCycle_pair (u := 2) (v := 0) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

/-- The 2-cycles `{0, 1}` and `{2, 3}` of `2C₃` meet in the vertex `1`. -/
theorem twoC3_verts₀₁_inter₂₃ : twoC3.verts {0, 1} ∩ twoC3.verts {2, 3} = {1} := by
  ext v
  simp only [Set.mem_inter_iff, mem_verts, Set.mem_insert_iff, Set.mem_singleton_iff]
  revert v
  decide

/-- The 2-cycles `{0, 1}` and `{4, 5}` of `2C₃` meet in the vertex `0`. -/
theorem twoC3_verts₀₁_inter₄₅ : twoC3.verts {0, 1} ∩ twoC3.verts {4, 5} = {0} := by
  ext v
  simp only [Set.mem_inter_iff, mem_verts, Set.mem_insert_iff, Set.mem_singleton_iff]
  revert v
  decide

theorem tube_isCycle₀₁ : tube.IsCycle {0, 1} :=
  isCycle_pair (u := 0) (v := 1) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

theorem tube_isCycle₄₅ : tube.IsCycle {4, 5} :=
  isCycle_pair (u := 2) (v := 3) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

/-- The two doubled links of the tube share no vertex. -/
theorem tube_disjoint₀₁₄₅ : Disjoint (tube.verts {0, 1}) (tube.verts {4, 5}) := by
  rw [Set.disjoint_left]
  intro v hv hw
  simp only [mem_verts, Set.mem_insert_iff, Set.mem_singleton_iff] at hv hw
  revert hv hw
  revert v
  decide

/-- A gain function on `2C₃` with no balanced 2-cycle and no balanced triangle realizes the
contrabalanced `2C₃`, because every cycle has two or three edges. -/
@[to_additive twoC3_addRealizes_of_small
  /-- An additive gain function on `2C₃` with no balanced 2-cycle and no balanced triangle
  realizes the contrabalanced `2C₃`. -/]
theorem twoC3_realizes_of_small {Γ : Type*} [Group Γ] {γ : Fin 6 → Γ}
    (h2 : ∀ C ∈ twoC3.balancedCycles γ, C.encard ≠ 2)
    (h3 : ∀ C ∈ twoC3.balancedCycles γ, C.encard ≠ 3) : twoC3.contrabalanced.Realizes γ := by
  refine Set.eq_empty_of_forall_notMem fun C hC => ?_
  rcases (mem_balancedCycles_iff.1 hC).1.encard_eq_two_or_three twoC3_loopless with h | h
  · exact h2 C hC h
  · exact h3 C hC h

/-- The gains `(1, 3, 1, 3, 2, 3)` on `2C₃` over `𝔽₇ˣ`. -/
def twoC3T0Gain₁ : Fin 6 → (ZMod 7)ˣ := ![1, fanThree, 1, fanThree, fanTwo, fanThree]

/-- The gains `(1, 2, 1, 3, 2, 3)` on `2C₃` over `𝔽₇ˣ`. -/
def twoC3T0Gain₂ : Fin 6 → (ZMod 7)ˣ := ![1, fanTwo, 1, fanThree, fanTwo, fanThree]

/-- The gains `(0, 1, 0, 1, 1, 2)` on `2C₃` over `𝔽₇⁺`. -/
def twoC3T0AddGain₁ : Fin 6 → ZMod 7 := ![0, 1, 0, 1, 1, 2]

/-- The gains `(0, 1, 0, 2, 1, 2)` on `2C₃` over `𝔽₇⁺`. -/
def twoC3T0AddGain₂ : Fin 6 → ZMod 7 := ![0, 1, 0, 2, 1, 2]

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
theorem twoC3T0Gain₁_realizes : twoC3.contrabalanced.Realizes twoC3T0Gain₁ :=
  twoC3_realizes_of_small
    (encard_ne_two_of_parallel (by simp only [Joins, dart, twoC3, twoC3T0Gain₁]; decide +kernel))
    (encard_ne_three_of_triangle (by simp only [Joins, dart, twoC3, twoC3T0Gain₁]; decide +kernel))

set_option synthInstance.maxSize 4096 in
set_option synthInstance.maxHeartbeats 400000 in
theorem twoC3T0Gain₂_realizes : twoC3.contrabalanced.Realizes twoC3T0Gain₂ :=
  twoC3_realizes_of_small
    (encard_ne_two_of_parallel (by simp only [Joins, dart, twoC3, twoC3T0Gain₂]; decide +kernel))
    (encard_ne_three_of_triangle (by simp only [Joins, dart, twoC3, twoC3T0Gain₂]; decide +kernel))

theorem twoC3T0AddGain₁_addRealizes : twoC3.contrabalanced.AddRealizes twoC3T0AddGain₁ :=
  twoC3_addRealizes_of_small
    (encard_ne_two_of_addParallel (by
      simp only [Joins, addDart, twoC3, twoC3T0AddGain₁]; decide +kernel))
    (encard_ne_three_of_addTriangle (by
      simp only [Joins, addDart, twoC3, twoC3T0AddGain₁]; decide +kernel))

theorem twoC3T0AddGain₂_addRealizes : twoC3.contrabalanced.AddRealizes twoC3T0AddGain₂ :=
  twoC3_addRealizes_of_small
    (encard_ne_two_of_addParallel (by
      simp only [Joins, addDart, twoC3, twoC3T0AddGain₂]; decide +kernel))
    (encard_ne_three_of_addTriangle (by
      simp only [Joins, addDart, twoC3, twoC3T0AddGain₂]; decide +kernel))

/-- A switching keeps the gain `3` of the 2-cycle `{0, 1}`, while the second gains give it `2`. -/
theorem twoC3T0Gain_not_switchEquiv : ¬SwitchEquiv twoC3 twoC3T0Gain₁ twoC3T0Gain₂ := by
  rintro ⟨η, h⟩
  have h₀ : (η 0)⁻¹ * 1 * η 1 = 1 := congrFun h 0
  have h₁ : (η 0)⁻¹ * fanThree * η 1 = fanTwo := congrFun h 1
  have key : fanThree = fanTwo := by
    calc fanThree = fanThree * ((η 0)⁻¹ * 1 * η 1) := by rw [h₀, mul_one]
      _ = (η 0)⁻¹ * fanThree * η 1 := by ac_rfl
      _ = fanTwo := h₁
  exact absurd key (by decide)

/-- A switching and scaling by `a` would give the 2-cycles `{0, 1}` and `{2, 3}` the gains `a` and
`a`, while the second gains give them `1` and `2`. -/
theorem twoC3T0AddGain_not_switchScaleEquiv :
    ¬SwitchScaleEquiv twoC3 twoC3T0AddGain₁ twoC3T0AddGain₂ := by
  rintro ⟨η, a, h⟩
  have h₀ : (a : ZMod 7) * (-η 0 + 0 + η 1) = 0 := congrFun h 0
  have h₁ : (a : ZMod 7) * (-η 0 + 1 + η 1) = 1 := congrFun h 1
  have h₂ : (a : ZMod 7) * (-η 1 + 0 + η 2) = 0 := congrFun h 2
  have h₃ : (a : ZMod 7) * (-η 1 + 1 + η 2) = 2 := congrFun h 3
  have key : (1 : ZMod 7) = 2 := by linear_combination h₃ - h₂ - h₁ + h₀
  exact absurd key (by decide)

end Transformer.GainMatroid
