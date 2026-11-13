module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ObservedRefinement

/-!
# Completeness of observed categorical fibers

Source: `review/flat_spectrum.md`, CM26. Same-label observations give a subset
of an actual decoder fiber. Injective phase coordinates and the fiber's actual
cardinality certify completeness. An observed multiplicity alone does not.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFibers

variable {I G C : Type*} [AddCommGroup G]

/-- The full latent preimage of an output label, CM26. It is defined using the
actual proposed decoder, not the subset visible in the training observations. -/
def fiber [Fintype G] [DecidableEq C] (d : G → C) (c : C) : Finset G :=
  Finset.univ.filter (fun z => d z = c)

/-- Translation preserves phase injectivity, CM26; coincident input phases
would invalidate a cardinality argument about their distinct latent images. -/
theorem shift_injective {r : I → G} (hr : Function.Injective r) (b : G) :
    Function.Injective (fun i => r i + b) := by
  intro i j h
  apply hr
  exact add_right_cancel h

variable [Fintype G] [DecidableEq G] [DecidableEq C]

/-- Actual same-label observations place their shifted phases inside the full
decoder fiber, CM26. There is no completeness conclusion at this step. -/
theorem observed_subset {r : I → G} {b : G} {F : Finset I} {d : G → C} {c : C}
    (hobs : ∀ i ∈ F, d (r i + b) = c) :
    F.image (fun i => r i + b) ⊆ fiber d c := by
  apply Finset.image_subset_iff.mpr
  intro i hi
  simp [fiber, hobs i hi]

/-- A same-label observed group is complete when its distinct phase image has
the actual fiber cardinality, CM26. Neither premise is inferred from counts. -/
theorem complete_image {r : I → G} {b : G} {F : Finset I} {d : G → C} {c : C}
    (hr : Function.Injective r) (hobs : ∀ i ∈ F, d (r i + b) = c)
    (hcard : F.card = (fiber d c).card) :
    F.image (fun i => r i + b) = fiber d c := by
  apply Finset.eq_of_subset_of_card_le (observed_subset hobs)
  simpa only [Finset.card_image_of_injective F (shift_injective hr b)] using hcard.ge

/-- Nonzero translation and a nontrivial finite phase space inhabit injectivity. -/
example : Function.Injective (fun i : ZMod 3 => i + 1) :=
  shift_injective (fun _ _ h => h) 1

/-- A constant categorical decoder has a genuinely many-to-one full fiber. -/
example : (Finset.univ : Finset (ZMod 3)).image (fun i => i + 1) ⊆
    fiber (fun (_ : ZMod 3) => ()) () :=
  observed_subset (by intros; rfl)

/-- All three distinct observed phases meet the actual three-element fiber. -/
example : (Finset.univ : Finset (ZMod 3)).image (fun i => i + 1) =
    fiber (fun (_ : ZMod 3) => ()) () := by
  apply complete_image (fun _ _ h => h) (by intros; rfl)
  simp [fiber]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFibers
