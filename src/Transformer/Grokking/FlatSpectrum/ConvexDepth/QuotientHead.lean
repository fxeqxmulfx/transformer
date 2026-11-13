module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientRefinement
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.FourierCode

/-!
# The refined quotient is an actual bounded complex CE head

Source: `review/flat_spectrum.md`, CM31 and CFt5. Reindex the normalized
Fourier basis by categorical state names, then apply the actual conjugate
initializer to observed raw labels and declared latent representatives.
No input arithmetic, held-out labels or separately learned factors occur
in this fixed-feature convex head. Its size includes every observed state.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Matrix
open Landscape.Neural (crossEntropy)

/-- The normalized Fourier code on any nonempty finite set of categorical
names. The indexing equivalence uses only the state set, CM31. -/
noncomputable def categoricalFourierOn (I : Type*) [Fintype I] [Nonempty I] : Matrix I I ℂ := by
  letI : NeZero (Fintype.card I) := ⟨Fintype.card_ne_zero⟩
  let e : I ≃ ZMod (Fintype.card I) := Fintype.equivOfCardEq (by simp)
  exact (categoricalFourier (Fintype.card I)).submatrix e e

/-- Relabeling Fourier rows and columns preserves both actual Gram
identities, rather than merely assigning a flatness flag, CM31. -/
theorem categoricalFourierOn_unitary (I : Type*) [Fintype I] [Nonempty I] [DecidableEq I] :
    categoricalFourierOn I ∈ Matrix.unitaryGroup I ℂ := by
  let : NeZero (Fintype.card I) := ⟨Fintype.card_ne_zero⟩
  let e : I ≃ ZMod (Fintype.card I) := Fintype.equivOfCardEq (by simp)
  have hl : (categoricalFourier (Fintype.card I)).conjTranspose *
      categoricalFourier (Fintype.card I) = 1 :=
    Unitary.star_mul_self_of_mem (categoricalFourier_unitary _)
  have hr : categoricalFourier (Fintype.card I) *
      (categoricalFourier (Fintype.card I)).conjTranspose = 1 :=
    Unitary.mul_star_self_of_mem (categoricalFourier_unitary _)
  constructor
  · change ((categoricalFourier _).submatrix e e).conjTranspose *
      (categoricalFourier _).submatrix e e = 1
    rw [Matrix.conjTranspose_submatrix, Matrix.submatrix_mul_equiv, hl,
      Matrix.submatrix_one_equiv]
  · change (categoricalFourier _).submatrix e e *
      ((categoricalFourier _).submatrix e e).conjTranspose = 1
    rw [Matrix.conjTranspose_submatrix, Matrix.submatrix_mul_equiv, hr,
      Matrix.submatrix_one_equiv]

/-- The refined basis has singular values one. Flatness concerns the
fixed code, not the output head or the empirical target table, CM31. -/
theorem categoricalFourierOn_singularValues (I : Type*) [Fintype I] [Nonempty I]
    [DecidableEq I] (i : I) : singularValues (categoricalFourierOn I) i = 1 :=
  unitary_singularValues _ (categoricalFourierOn_unitary I) i

variable {X Z C : Type*} [DecidableEq X] [Fintype Z] [Nonempty Z] [DecidableEq Z]
  [Fintype C] [DecidableEq C]

/-- Both complex coordinates are initialized by conjugate multiplication;
the labels affect only this readout, CM31. -/
noncomputable def refinedComplexHead (observed : Finset X) (labels : ↥observed → C)
    (representative : Z → C) (a : ℝ) : Matrix (↥observed ⊕ Z) C ℂ :=
  complexInitializer (categoricalFourierOn (↥observed ⊕ Z))
    (refinedLabels observed labels representative) a

/-- Actual real/imaginary matrix logits evaluated at the refined input
state, with the full original class vocabulary, CM31. -/
noncomputable def refinedComplexLogits (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (a : ℝ) (x : X) : C → ℝ :=
  headLogits (complexFeatures (categoricalFourierOn (↥observed ⊕ Z)))
    (complexCoefficients (refinedComplexHead observed labels representative a))
    (refinedState observed latent x)

omit [Fintype C] in
/-- Fourier inversion realizes the actual refined raw assignment at every
input, including the explicit representative on unobserved inputs, CM31. -/
theorem refinedComplexLogits_eq (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (a : ℝ) (x : X) :
    refinedComplexLogits observed latent labels representative a x =
      fun c => if c = refinedPrediction observed latent labels representative x then a else 0 := by
  exact complexInitializer_logits _ (categoricalFourierOn_unitary _) _ _ _

/-- The refined head fits each recorded raw class with the same ordinary
CE formula as CFt5. A colliding quotient no longer forces log2 loss, CM31. -/
theorem refinedComplex_ce_seen (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (a : ℝ)
    {x : X} (hx : x ∈ observed) :
    crossEntropy (refinedComplexLogits observed latent labels representative a x) (labels ⟨x, hx⟩) =
      Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-a)) := by
  rw [refinedComplexLogits_eq, refinedPrediction_seen _ _ _ _ hx]
  exact fixed_margin_ce _ _

omit [Fintype C] in
/-- The declared prediction is the unique raw argmax for positive margin,
CM31. This states what the saved complex head computes, not unseen truth. -/
theorem refinedComplex_strict (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) {a : ℝ} (ha : 0 < a)
    (x : X) {c : C} (hc : c ≠ refinedPrediction observed latent labels representative x) :
    refinedComplexLogits observed latent labels representative a x c <
      refinedComplexLogits observed latent labels representative a x
        (refinedPrediction observed latent labels representative x) := by
  rw [refinedComplexLogits_eq]
  simpa [hc] using ha

/-- All real and imaginary initialized coordinates fit the same native
per-class budget with the refined state count charged explicitly, CM31. -/
theorem refinedComplex_budget (observed : Finset X) (labels : ↥observed → C)
    (representative : Z → C) :
    complexCoefficients (refinedComplexHead observed labels representative 24) ∈
      headBudget (64 * ((observed.card : ℝ) + (Fintype.card Z : ℝ))) := by
  have h := complexInitializer_native_budget _ (categoricalFourierOn_unitary (↥observed ⊕ Z))
    (refinedLabels observed labels representative)
  simpa only [refinedComplexHead, Fintype.card_sum, Fintype.card_coe, Nat.cast_add] using h

omit [DecidableEq Z] [DecidableEq C] in
/-- With the extractor fixed, ordinary CE is convex in every learned
coefficient of the refined head, CM31. No discovery parameter is optimized. -/
theorem refinedComplex_loss_convex {S : Type*} [Fintype S]
    (observed : Finset X) (latent : X → Z) (inputs : S → X) (labels : S → C) (R : ℝ) :
    ConvexOn ℝ (headBudget R)
      (headLoss (complexFeatures (categoricalFourierOn (↥observed ⊕ Z)))
        (fun s => refinedState observed latent (inputs s)) labels) :=
  headLoss_convex _ _ _ _

omit [Fintype C] [DecidableEq C] in
/-- The refined flat basis bounds every feasible learned head's logits,
not only the initialized margins. This is the capacity role of flatness. -/
theorem refined_headLogits_bound (observed : Finset X) (latent : X → Z)
    {W : (↥observed ⊕ Z) × Bool → C → ℝ} {R : ℝ} (hW : W ∈ headBudget R)
    (x : X) (c : C) :
    |headLogits (complexFeatures (categoricalFourierOn (↥observed ⊕ Z))) W
      (refinedState observed latent x) c| ≤ R := by
  exact complex_headLogits_bound _ (categoricalFourierOn_unitary _) hW _ _

/-- The refined initializer pays squared energy in proportion to its full
state count. Observed-input refinement is not spectrally free, CM31. -/
theorem refinedComplex_frobenius (observed : Finset X) (labels : ↥observed → C)
    (representative : Z → C) (a : ℝ) :
    frobeniusSq (refinedComplexHead observed labels representative a) =
      ((observed.card : ℝ) + (Fintype.card Z : ℝ)) * a^2 := by
  rw [refinedComplexHead, complexInitializer_frobenius _ (categoricalFourierOn_unitary _)]
  simp only [Fintype.card_sum, Fintype.card_coe, Nat.cast_add]

example := categoricalFourierOn_unitary (Bool × Bool)

example := categoricalFourierOn_singularValues (Bool × Bool) (false, true)

example := refinedComplex_ce_seen ({false} : Finset Bool) id (fun x => x.val) id 24
  (x := false) (by simp)

example := refinedComplex_strict ({false} : Finset Bool) id (fun x => x.val) id
  (a := 24) (by norm_num) true (c := false) (by decide)

example := refinedComplex_budget ({false} : Finset Bool) (fun x => x.val) (id : Bool → Bool)

example := refined_headLogits_bound ({false} : Finset Bool) id
  (W := fun _ (_ : Bool) => 0) (R := 1) (zero_mem_headBudget (by norm_num)) true false

end Transformer.Grokking.FlatSpectrum.ConvexDepth
