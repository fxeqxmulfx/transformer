module

public import Transformer.GainMatroid.Section1_Particular
public import Transformer.GainMatroid.Section2_LiftMatroid
public import Transformer.GainMatroid.Section2_RollUps

/-!
# All representations of the unavoidable minors are canonical

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4.2: "every `𝔽`-matrix representation of a frame
or lift matroid arising from a biased graph in `𝒯₀` is projectively equivalent to a canonical
representation particular to that biased graph".

The biased `2C₃` (Lemma 4.13), the prisms (Lemma 4.14) and the biased `K₄` (Lemma 4.15) of `𝒯₀` have
no two vertex-disjoint unbalanced cycles, so their frame and lift matroids coincide, written `M(Ω)`
in the source (`BiasedGraph.isFrameMatroid_iff_isLiftMatroid`). Their statements take `M` to be
either (`BiasedGraph.Represents`), and a representation is then particular as a lift matrix or as a
frame matrix, but not both. For the tubes the two matroids differ, and each representation is
particular of its own kind (Lemmas 4.16 and 4.17). For `2C₃ \ e` (Lemma 4.18) a representation is
particular as a lift matrix, and over a field other than `GF(2)` as a frame matrix to the graph or
to one of its roll-ups from the vertex `1` of both doubled links.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

variable {R F : Type*} [Field F]

namespace BiasedGraph

/-- **Lemma 4.13.** "Let `(2C₃, B)` be a biased graph with no balanced 2-cycle and let `A` be an
`𝔽`-matrix representing `M(2C₃, B)`. Then `A` is projectively equivalent to a canonical lift matrix
particular to `(2C₃, B)` or to a canonical frame matrix particular to `(2C₃, B)`, but not both." -/
theorem IsG0TwoC3.xor_particular {Ω : BiasedGraph (Fin 3) (Fin 6)} (hΩ : IsG0TwoC3 Ω)
    {M : Matroid (Fin 6)} (hM : Ω.Represents M) {A : Matrix R (Fin 6) F}
    (hA : IsMatrixRepresentation M A) : Xor (Ω.IsLiftParticular A) (Ω.IsFrameParticular A) := by
  sorry

/-- **Lemma 4.14.** "Let `Ω ∈ {Pr, Pr₁, Pr₂}` and let `A` be an `𝔽`-matrix representing `M(Ω)`.
Then `A` is projectively equivalent to a canonical lift matrix particular to `Ω` or to a canonical
frame matrix particular to `Ω`, but not both." -/
theorem prisms_xor_particular :
    (∀ (M : Matroid (Fin 9)) (A : Matrix R (Fin 9) F), prismBiased.Represents M →
      IsMatrixRepresentation M A →
        Xor (prismBiased.IsLiftParticular A) (prismBiased.IsFrameParticular A)) ∧
    (∀ (M : Matroid (Fin 7)) (A : Matrix R (Fin 7) F), prismOneBiased.Represents M →
      IsMatrixRepresentation M A →
        Xor (prismOneBiased.IsLiftParticular A) (prismOneBiased.IsFrameParticular A)) ∧
    ∀ (M : Matroid (Fin 8)) (A : Matrix R (Fin 8) F), prismTwoBiased.Represents M →
      IsMatrixRepresentation M A →
        Xor (prismTwoBiased.IsLiftParticular A) (prismTwoBiased.IsFrameParticular A) := by
  sorry

/-- **Lemma 4.15.** "Let `(K₄, B)` be a biased graph with no balanced 3-cycle and let `A` be an
`𝔽`-matrix representing `M(K₄, B)`. Then `A` is projectively equivalent to a canonical lift matrix
particular to `(K₄, B)` or to a canonical frame matrix particular to `(K₄, B)`, but not both." -/
theorem IsG0K4.xor_particular {Ω : BiasedGraph (Fin 4) (Fin 6)} (hΩ : IsG0K4 Ω)
    {M : Matroid (Fin 6)} (hM : Ω.Represents M) {A : Matrix R (Fin 6) F}
    (hA : IsMatrixRepresentation M A) : Xor (Ω.IsLiftParticular A) (Ω.IsFrameParticular A) := by
  sorry

/-- **Lemma 4.16.** "Let `(2C₄'', B)` be a biased graph with no balanced 2-cycle and let `A` be an
`𝔽`-matrix representing `F(2C₄'', B)`. Then `A` is projectively equivalent to a canonical frame
matrix particular to `(2C₄'', B)`." -/
theorem IsG0Tube.isFrameParticular {Ω : BiasedGraph (Fin 4) (Fin 6)} (hΩ : IsG0Tube Ω)
    {M : Matroid (Fin 6)} (hM : Ω.IsFrameMatroid M) {A : Matrix R (Fin 6) F}
    (hA : IsMatrixRepresentation M A) : Ω.IsFrameParticular A := by
  sorry

/-- **Lemma 4.17.** "Let `(2C₄'', B)` be a biased graph with no balanced 2-cycle and let `A` be an
`𝔽`-matrix representing `L(2C₄'', B)`. Then `A` is projectively equivalent to a canonical lift
matrix particular to `(2C₄'', B)`." -/
theorem IsG0Tube.isLiftParticular {Ω : BiasedGraph (Fin 4) (Fin 6)} (hΩ : IsG0Tube Ω)
    {M : Matroid (Fin 6)} (hM : Ω.IsLiftMatroid M) {A : Matrix R (Fin 6) F}
    (hA : IsMatrixRepresentation M A) : Ω.IsLiftParticular A := by
  sorry

/-- **Lemma 4.18.** "Let `(2C₃ \ e, B)` be a biased graph with no balanced 2-cycle and let `A` be
an `𝔽`-matrix representing `M(2C₃ \ e, B)`. Then `A` is projectively equivalent to a canonical lift
matrix particular to `(2C₃ \ e, B)` and `A` is projectively equivalent to a canonical frame matrix
particular to `(2C₃ \ e, B)` or to a roll-up of `(2C₃ \ e, B)`." The roll-ups are from the vertex
`1`, which lies on both doubled links and so on every cycle of two or more edges.

The frame part assumes `𝔽 ≠ GF(2)`. The proof's roll-up has a joint, of gain other than `1`; over
`GF(2)` the gains `(0, 1, 0, 1, 0)` of `𝔽⁺` give a binary `M(2C₃ \ e, B)`, while `GF(2)ˣ` is
trivial and so realizes neither `(2C₃ \ e, B)` nor a roll-up. -/
theorem IsBPrime.isLiftParticular_and_isFrameParticular {Ω : BiasedGraph (Fin 3) (Fin 5)}
    (hΩ : IsBPrime Ω) {M : Matroid (Fin 5)} (hM : Ω.Represents M) {A : Matrix R (Fin 5) F}
    (hA : IsMatrixRepresentation M A) :
    Ω.IsLiftParticular A ∧ ((∃ a : Fˣ, a ≠ 1) → Ω.IsFrameParticular A ∨
      ∃ U ∈ Ω.unbalancingClasses 1, (Ω.rollUp 1 U).IsFrameParticular A) := by
  sorry

end BiasedGraph

/-- The gains `(1, -1, 1, -1, 1)` on `2C₃ \ e` over `𝔽₃ˣ`. -/
def twoC3MinusGain : Fin 5 → (ZMod 3)ˣ := ![1, -1, 1, -1, 1]

theorem twoC3MinusGain_noPair : ∀ C ∈ twoC3Minus.balancedCycles twoC3MinusGain, C.encard ≠ 2 :=
  encard_ne_two_of_parallel (by
    simp only [Joins, dart, twoC3Minus, twoC3MinusGain]
    decide)

/-- The hypotheses of Lemma 4.13 hold for the frame matrix of `twoC3Gain`. -/
example : IsG0TwoC3 (twoC3.gainBiased twoC3Gain) ∧ ∃ M, (twoC3.gainBiased twoC3Gain).Represents M ∧
    IsMatrixRepresentation M (frameMatrix twoC3 twoC3Gain) := by
  obtain ⟨M, hM, hA⟩ := Realizes.exists_isFrameMatroid_frameMatrix (Ω := twoC3.gainBiased twoC3Gain)
    (φ := twoC3Gain) rfl
  exact ⟨twoC3Gain_isG0TwoC3, M, Or.inl hM, hA⟩

/-- The matroids of the prisms exist; Lemma 4.14 asks for an `𝔽`-representation of one. -/
example : (∃ M, prismBiased.Represents M) ∧ (∃ M, prismOneBiased.Represents M) ∧
    ∃ M, prismTwoBiased.Represents M :=
  ⟨prismBiased.exists_isFrameMatroid.imp fun _ => Or.inl,
    prismOneBiased.exists_isFrameMatroid.imp fun _ => Or.inl,
    prismTwoBiased.exists_isFrameMatroid.imp fun _ => Or.inl⟩

/-- The hypotheses of Lemma 4.15 hold for the frame matrix of `K4Gain`. -/
example : IsG0K4 (K4.gainBiased K4Gain) ∧ ∃ M, (K4.gainBiased K4Gain).Represents M ∧
    IsMatrixRepresentation M (frameMatrix K4 K4Gain) := by
  obtain ⟨M, hM, hA⟩ := Realizes.exists_isFrameMatroid_frameMatrix (Ω := K4.gainBiased K4Gain)
    (φ := K4Gain) rfl
  exact ⟨K4Gain_isG0K4, M, Or.inl hM, hA⟩

/-- The hypotheses of Lemma 4.16 hold for the frame matrix of `tubeGain`. -/
example : IsG0Tube (tube.gainBiased tubeGain) ∧ ∃ M, (tube.gainBiased tubeGain).IsFrameMatroid M ∧
    IsMatrixRepresentation M (frameMatrix tube tubeGain) :=
  ⟨tubeGain_isG0Tube, Realizes.exists_isFrameMatroid_frameMatrix (φ := tubeGain) rfl⟩

/-- The hypotheses of Lemma 4.17 hold for the lift matrix of `tubeAddGain`. -/
example : IsG0Tube (tube.addGainBiased tubeAddGain) ∧
    ∃ M, (tube.addGainBiased tubeAddGain).IsLiftMatroid M ∧
      IsMatrixRepresentation M (liftMatrix tube tubeAddGain) :=
  ⟨⟨IsReorientationOf.refl _, tubeAddGain_noPair⟩,
    AddRealizes.exists_isLiftMatroid_liftMatrix (ψ := tubeAddGain) rfl⟩

/-- The hypotheses of Lemma 4.18 hold for the frame matrix of `twoC3MinusGain`. -/
example : IsBPrime (twoC3Minus.gainBiased twoC3MinusGain) ∧
    ∃ M, (twoC3Minus.gainBiased twoC3MinusGain).Represents M ∧
      IsMatrixRepresentation M (frameMatrix twoC3Minus twoC3MinusGain) := by
  obtain ⟨M, hM, hA⟩ := Realizes.exists_isFrameMatroid_frameMatrix
    (Ω := twoC3Minus.gainBiased twoC3MinusGain) (φ := twoC3MinusGain) rfl
  exact ⟨⟨IsReorientationOf.refl _, twoC3MinusGain_noPair⟩, M, Or.inl hM, hA⟩

end Transformer.GainMatroid
