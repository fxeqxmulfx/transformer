module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.Factorization

/-!
# Conditional identification from the restricted complete kernel

Source: `review/flat_spectrum.md`, CM40. The extra observed-code equations
express a shared encoder across the supplied role alignment. Complete-basis
matching still has its actual universal meaning. Soundness requires the
unknown target to belong to this more restrictive family; observed separation
or a flat feature bound cannot establish that premise.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles

open KernelCodes

variable {R I J C A D K L S T : Type*} [CommRing R]
variable {cell : S → I × J} {recorded : S → C} {i0 : I} {c0 : C}
variable {align : T → I × J} {t0 : T}

/-- Every original observation still matches its recorded class in the
restricted complete basis, CM40. This is not a claim of class separation. -/
theorem observed_match (b : Module.Basis K R (sharedKernel cell recorded i0 c0 align t0))
    (t : S) : queryCode b (cell t).1 (cell t).2 = classCode b (recorded t) := by
  apply (code_match_iff b _ _ _).mpr
  intro w
  exact w.property.1.1 t

/-- Restricting the coefficient space cannot remove an existing full-code
match, CM40. It may add matches, as the strict witness demonstrates. -/
theorem match_of_unrestricted (old : Module.Basis K R (observedKernel cell recorded i0 c0))
    (b : Module.Basis L R (sharedKernel cell recorded i0 c0 align t0))
    (i : I) (j : J) (c : C) (h : queryCode old i j = classCode old c) :
    queryCode b i j = classCode b c := by
  apply (code_match_iff b i j c).mpr
  intro w
  exact (code_match_iff old i j c).mp h ⟨w.val, w.property.1⟩

/-- A shared-role additive target normalizes into the actual restricted
kernel, so a full-basis match identifies its code, CM40. Both the global
target family and its fit to original observations are explicit premises. -/
theorem matched_code_sound (b : Module.Basis K R (sharedKernel cell recorded i0 c0 align t0))
    (name : C → A) (e : A → R) (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = name (recorded t))
    (hshare : ∀ t, r (align t).1 - s (align t).2 = r (align t0).1 - s (align t0).2)
    (i : I) (j : J) (c : C) (hmatch : queryCode b i j = classCode b c) :
    e (y i j) = e (name c) := by
  let w := pack r s (e ∘ name)
  have hw : ∀ t, queryAction (cell t).1 (cell t).2 (recorded t) w = 0 := by
    intro t
    change r (cell t).1 + s (cell t).2 - e (name (recorded t)) = 0
    rw [← hobs t, hreal]
    ring
  have hs : w ∈ roleKernel align t0 := pack_mem_roleKernel align t0 r s _ hshare
  have hn := normalize_mem_shared cell recorded i0 c0 align t0 w hw hs
  have hz := (code_match_iff b i j c).mp hmatch ⟨normalize i0 c0 w, hn⟩
  rw [normalize_query] at hz
  change r i + s j - e (name c) = 0 at hz
  rw [hreal]
  exact sub_eq_zero.mp hz

/-- A compatible readout retains raw-label ambiguity while identifying
the declared answer from a checked shared-kernel match, CM40. -/
theorem matched_readout_sound (b : Module.Basis K R (sharedKernel cell recorded i0 c0 align t0))
    (name : C → A) (e : A → R) (q : A → D)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = name (recorded t))
    (hshare : ∀ t, r (align t).1 - s (align t).2 = r (align t0).1 - s (align t0).2)
    (i : I) (j : J) (c : C) (hmatch : queryCode b i j = classCode b c) :
    q (y i j) = q (name c) := by
  apply hcompat
  exact matched_code_sound b name e r s y hreal hobs hshare i j c hmatch

variable [Fintype R] [Fintype D]

/-- CM35's derived answer-code equivalence applies to the restricted
kernel as well, CM40. Independently compacted class names represent declared
answers, not unseen raw labels or a supplied target code. -/
theorem named_readout_sound {rawLabels : S → A} (q : A → D)
    (b : Module.Basis K R (sharedKernel cell recorded i0 c0 align t0))
    (name : C → D) (hrecorded : ∀ t, q (rawLabels t) = name (recorded t))
    (e : A → R) (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R)
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = rawLabels t)
    (hshare : ∀ t, r (align t).1 - s (align t).2 = r (align t0).1 - s (align t0).2)
    (i : I) (j : J) (c : C) (hmatch : queryCode b i j = classCode b c) :
    q (y i j) = name c := by
  obtain ⟨φ, hφ⟩ := ReadoutKernel.compatible_factor_equiv q e hsurj hcompat hcard
  have hreal' : ∀ i j, φ (q (y i j)) = r i + s j := by
    intro i j
    rw [← hφ, hreal]
  have hobs' : ∀ t, q (y (cell t).1 (cell t).2) = name (recorded t) := by
    intro t
    rw [hobs]
    exact hrecorded t
  exact φ.injective (matched_code_sound b name φ r s (fun i j => q (y i j))
    hreal' hobs' hshare i j c hmatch)

/-- The unchanged observed-class search identifies a declared answer
when it succeeds, CM40. Failure remains explicit; no global coverage or
unseen-answer assertion has been inserted into the search algorithm. -/
theorem choose_readout_sound {rawLabels : S → A} (q : A → D)
    (b : Module.Basis K R (sharedKernel cell recorded i0 c0 align t0))
    (name : C → D) (hrecorded : ∀ t, q (rawLabels t) = name (recorded t))
    (e : A → R) (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R)
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = rawLabels t)
    (hshare : ∀ t, r (align t).1 - s (align t).2 = r (align t0).1 - s (align t0).2)
    (i : I) (j : J) (bank : List C) {c : C}
    (hfound : chooseClass b (queryCode b i j) bank = some c) :
    q (y i j) = name c := by
  apply named_readout_sound q b name hrecorded e hsurj hcompat hcard r s y
    hreal hobs hshare i j c
  exact chooseClass_match b _ bank hfound

end Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles
