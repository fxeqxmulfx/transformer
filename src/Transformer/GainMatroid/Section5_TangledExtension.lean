module

public import Transformer.GainMatroid.Section2_CanonicalMatrices
public import Transformer.GainMatroid.Section2_EdgeImages
public import Transformer.GainMatroid.Section2_LiftMatroid
public import Transformer.GainMatroid.Section3_Families

/-!
# Lemma 5.8: a tangled canonical matrix does not extend across a joint

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §5.2. Let `(G, B)` have a joint `e` with
`(G, B) \ e` a biased `2C₃` with no balanced 2-cycle or a biased `K₄` with no balanced triangle.
A canonical frame matrix of `(G, B) \ e` representing its matroid gains no column making it
represent `L(G, B)`, and a canonical lift matrix gains no column making it represent `F(G, B)`.

Here `G` has the edges `Option (Fin 6)`: the joint is `none`, and `(G, B) \ e` is a biased graph `S`
on the edges `Fin 6` whose edge `i` has the ends of `some i` and whose balanced cycles are those of
`(G, B)` avoiding `none`. The examples add a joint to `2C₃` and `K₄` with `G.withLoop v`, whose
balanced cycles avoiding `none` are those of `G` (`OrientedGraph.mem_balancedCycles_image_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {F : Type*} [Field F]

/-- **Lemma 5.8 (i)** for `2C₃`. "Let `(G, B)` be a connected biased graph with a joint `e` such
that `(G, B) \ e` is a biased `2C₃` with no balanced 2-cycle … Let `𝔽` be a field, and let `φ` be an
`𝔽ˣ`- or `𝔽⁺`-gain function on `G`. (i) If `A_F(G \ e, φ)` represents `M((G, B) \ e)` then
`A_F(G \ e, φ)` does not extend to an `𝔽`-representation of `L(G, B)`." -/
theorem IsG0TwoC3.frameMatrix_not_extend {Ω : BiasedGraph (Fin 3) (Option (Fin 6))}
    {S : BiasedGraph (Fin 3) (Fin 6)} (hS : IsG0TwoC3 S) (he : Ω.IsJoint none)
    (hG : ∀ i, Ω.graph.tail (some i) = S.graph.tail i ∧ Ω.graph.head (some i) = S.graph.head i)
    (hB : ∀ C, C ∈ S.balanced ↔ some '' C ∈ Ω.balanced) {φ : Fin 6 → Fˣ} {M : Matroid (Fin 6)}
    (hM : S.Represents M) (hA : IsMatrixRepresentation M (frameMatrix S.graph φ))
    {L : Matroid (Option (Fin 6))} (hL : Ω.IsLiftMatroid L) :
    ¬∃ c : Fin 3 → F, IsMatrixRepresentation L fun r x =>
      x.elim (c r) fun i => frameMatrix S.graph φ r i := by
  sorry

/-- **Lemma 5.8 (ii)** for `2C₃`. "If `A_L(G \ e, φ)` represents `M((G, B) \ e)` then
`A_L(G \ e, φ)` does not extend to an `𝔽`-representation of `F(G, B)`." -/
theorem IsG0TwoC3.liftMatrix_not_extend {Ω : BiasedGraph (Fin 3) (Option (Fin 6))}
    {S : BiasedGraph (Fin 3) (Fin 6)} (hS : IsG0TwoC3 S) (he : Ω.IsJoint none)
    (hG : ∀ i, Ω.graph.tail (some i) = S.graph.tail i ∧ Ω.graph.head (some i) = S.graph.head i)
    (hB : ∀ C, C ∈ S.balanced ↔ some '' C ∈ Ω.balanced) {φ : Fin 6 → F} {M : Matroid (Fin 6)}
    (hM : S.Represents M) (hA : IsMatrixRepresentation M (liftMatrix S.graph φ))
    {N : Matroid (Option (Fin 6))} (hN : Ω.IsFrameMatroid N) :
    ¬∃ c : Option (Fin 3) → F, IsMatrixRepresentation N fun r x =>
      x.elim (c r) fun i => liftMatrix S.graph φ r i := by
  sorry

/-- **Lemma 5.8 (i)** for `K₄`: "… or a biased `K₄` with no balanced triangle." -/
theorem IsG0K4.frameMatrix_not_extend {Ω : BiasedGraph (Fin 4) (Option (Fin 6))}
    {S : BiasedGraph (Fin 4) (Fin 6)} (hS : IsG0K4 S) (he : Ω.IsJoint none)
    (hG : ∀ i, Ω.graph.tail (some i) = S.graph.tail i ∧ Ω.graph.head (some i) = S.graph.head i)
    (hB : ∀ C, C ∈ S.balanced ↔ some '' C ∈ Ω.balanced) {φ : Fin 6 → Fˣ} {M : Matroid (Fin 6)}
    (hM : S.Represents M) (hA : IsMatrixRepresentation M (frameMatrix S.graph φ))
    {L : Matroid (Option (Fin 6))} (hL : Ω.IsLiftMatroid L) :
    ¬∃ c : Fin 4 → F, IsMatrixRepresentation L fun r x =>
      x.elim (c r) fun i => frameMatrix S.graph φ r i := by
  sorry

/-- **Lemma 5.8 (ii)** for `K₄`. -/
theorem IsG0K4.liftMatrix_not_extend {Ω : BiasedGraph (Fin 4) (Option (Fin 6))}
    {S : BiasedGraph (Fin 4) (Fin 6)} (hS : IsG0K4 S) (he : Ω.IsJoint none)
    (hG : ∀ i, Ω.graph.tail (some i) = S.graph.tail i ∧ Ω.graph.head (some i) = S.graph.head i)
    (hB : ∀ C, C ∈ S.balanced ↔ some '' C ∈ Ω.balanced) {φ : Fin 6 → F} {M : Matroid (Fin 6)}
    (hM : S.Represents M) (hA : IsMatrixRepresentation M (liftMatrix S.graph φ))
    {N : Matroid (Option (Fin 6))} (hN : Ω.IsFrameMatroid N) :
    ¬∃ c : Option (Fin 4) → F, IsMatrixRepresentation N fun r x =>
      x.elim (c r) fun i => liftMatrix S.graph φ r i := by
  sorry

end BiasedGraph

open BiasedGraph

/-- `2C₃` with a joint `none` at `0` of gain `-1`, over `𝔽₃ˣ`. -/
def twoC3JointGain : Option (Fin 6) → (ZMod 3)ˣ := fun e => e.elim (-1) twoC3Gain
/-- `2C₃` with a joint `none` at `0` of gain `1`, over `𝔽₃⁺`. -/
def twoC3JointAddGain : Option (Fin 6) → ZMod 3 := fun e => e.elim 1 twoC3AddGain
/-- `K₄` with a joint `none` at `0` of gain `-1`, over `𝔽₃ˣ`. -/
def K4JointGain : Option (Fin 6) → (ZMod 3)ˣ := fun e => e.elim (-1) K4Gain
/-- `K₄` with a joint `none` at `0` of gain `1`, over `𝔽₃⁺`. -/
def K4JointAddGain : Option (Fin 6) → ZMod 3 := fun e => e.elim 1 K4AddGain

/-- The hypotheses of Lemma 5.8 (i) hold for `2C₃` with the gains `twoC3Gain` and a joint. -/
example : IsG0TwoC3 (twoC3.gainBiased twoC3Gain) ∧
    ((twoC3.withLoop 0).gainBiased twoC3JointGain).IsJoint none ∧
    (∀ C, C ∈ (twoC3.gainBiased twoC3Gain).balanced ↔
      some '' C ∈ ((twoC3.withLoop 0).gainBiased twoC3JointGain).balanced) ∧
    (∃ M, (twoC3.gainBiased twoC3Gain).Represents M ∧
      IsMatrixRepresentation M (frameMatrix twoC3 twoC3Gain)) ∧
    ∃ L, ((twoC3.withLoop 0).gainBiased twoC3JointGain).IsLiftMatroid L := by
  obtain ⟨M, hM, hA⟩ := Realizes.exists_isFrameMatroid_frameMatrix
    (Ω := twoC3.gainBiased twoC3Gain) (φ := twoC3Gain) rfl
  exact ⟨twoC3Gain_isG0TwoC3, (isJoint_gainBiased_iff rfl).2 (by decide),
    fun _ => mem_balancedCycles_image_iff (G := twoC3.withLoop 0) (H := twoC3)
      (γ := twoC3Gain) (γ' := twoC3JointGain) (Option.some_injective _) (fun _ => ⟨rfl, rfl⟩)
      (fun _ => rfl),
    ⟨M, Or.inl hM, hA⟩, exists_isLiftMatroid _⟩

/-- The hypotheses of Lemma 5.8 (ii) hold for `2C₃` with the gains `twoC3AddGain` and a joint. -/
example : IsG0TwoC3 (twoC3.addGainBiased twoC3AddGain) ∧
    ((twoC3.withLoop 0).addGainBiased twoC3JointAddGain).IsJoint none ∧
    (∀ C, C ∈ (twoC3.addGainBiased twoC3AddGain).balanced ↔
      some '' C ∈ ((twoC3.withLoop 0).addGainBiased twoC3JointAddGain).balanced) ∧
    (∃ M, (twoC3.addGainBiased twoC3AddGain).Represents M ∧
      IsMatrixRepresentation M (liftMatrix twoC3 twoC3AddGain)) ∧
    ∃ N, ((twoC3.withLoop 0).addGainBiased twoC3JointAddGain).IsFrameMatroid N := by
  obtain ⟨M, hM, hA⟩ := AddRealizes.exists_isLiftMatroid_liftMatrix
    (Ω := twoC3.addGainBiased twoC3AddGain) (ψ := twoC3AddGain) rfl
  exact ⟨⟨IsReorientationOf.refl _, twoC3AddGain_noPair⟩, (isJoint_addGainBiased_iff rfl).2
    (by decide), fun _ => mem_addBalancedCycles_image_iff (G := twoC3.withLoop 0) (H := twoC3)
      (γ := twoC3AddGain) (γ' := twoC3JointAddGain) (Option.some_injective _)
      (fun _ => ⟨rfl, rfl⟩) (fun _ => rfl), ⟨M, Or.inr hM, hA⟩, exists_isFrameMatroid _⟩

/-- The hypotheses of Lemma 5.8 (i) hold for `K₄` with the gains `K4Gain` and a joint. -/
example : IsG0K4 (K4.gainBiased K4Gain) ∧ ((K4.withLoop 0).gainBiased K4JointGain).IsJoint none ∧
    (∀ C, C ∈ (K4.gainBiased K4Gain).balanced ↔
      some '' C ∈ ((K4.withLoop 0).gainBiased K4JointGain).balanced) ∧
    (∃ M, (K4.gainBiased K4Gain).Represents M ∧ IsMatrixRepresentation M (frameMatrix K4 K4Gain)) ∧
    ∃ L, ((K4.withLoop 0).gainBiased K4JointGain).IsLiftMatroid L := by
  obtain ⟨M, hM, hA⟩ := Realizes.exists_isFrameMatroid_frameMatrix
    (Ω := K4.gainBiased K4Gain) (φ := K4Gain) rfl
  exact ⟨K4Gain_isG0K4, (isJoint_gainBiased_iff rfl).2 (by decide),
    fun _ => mem_balancedCycles_image_iff (G := K4.withLoop 0) (H := K4) (γ := K4Gain)
      (γ' := K4JointGain) (Option.some_injective _) (fun _ => ⟨rfl, rfl⟩) (fun _ => rfl),
    ⟨M, Or.inl hM, hA⟩, exists_isLiftMatroid _⟩

/-- The hypotheses of Lemma 5.8 (ii) hold for `K₄` with the gains `K4AddGain` and a joint. -/
example : IsG0K4 (K4.addGainBiased K4AddGain) ∧
    ((K4.withLoop 0).addGainBiased K4JointAddGain).IsJoint none ∧
    (∀ C, C ∈ (K4.addGainBiased K4AddGain).balanced ↔
      some '' C ∈ ((K4.withLoop 0).addGainBiased K4JointAddGain).balanced) ∧
    (∃ M, (K4.addGainBiased K4AddGain).Represents M ∧
      IsMatrixRepresentation M (liftMatrix K4 K4AddGain)) ∧
    ∃ N, ((K4.withLoop 0).addGainBiased K4JointAddGain).IsFrameMatroid N := by
  obtain ⟨M, hM, hA⟩ := AddRealizes.exists_isLiftMatroid_liftMatrix
    (Ω := K4.addGainBiased K4AddGain) (ψ := K4AddGain) rfl
  exact ⟨⟨IsReorientationOf.refl _, K4AddGain_noTriangle⟩, (isJoint_addGainBiased_iff rfl).2
    (by decide), fun _ => mem_addBalancedCycles_image_iff (G := K4.withLoop 0) (H := K4)
      (γ := K4AddGain) (γ' := K4JointAddGain) (Option.some_injective _)
      (fun _ => ⟨rfl, rfl⟩) (fun _ => rfl), ⟨M, Or.inr hM, hA⟩, exists_isFrameMatroid _⟩

end Transformer.GainMatroid
