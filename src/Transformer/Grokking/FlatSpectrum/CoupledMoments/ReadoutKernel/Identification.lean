module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.Factorization
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Identification

/-!
# Complete-kernel matching after derived readout grouping

Source: `review/flat_spectrum.md`, CM35. Apply the declared readout to the
actual observed raw labels, and keep the complete coefficient kernel of those
grouped equations. The unknown compatible code factors through this grouping
by cardinality, so a checked match identifies the original answer. Neither
the unknown code nor its factorizing equivalence enters the kernel or search.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

open KernelCodes

variable {R I J A D K S : Type*} [CommRing R] [Fintype R] [Fintype D]
variable {cell : S → I × J} {labels : S → A} {i0 : I} {d0 : D}

/-- Independently compacted class names may represent the grouped
observations, CM35. The extra premise checks only the recorded label map;
the unknown raw target must still fit every original raw observation. -/
theorem named_readout_sound {C : Type*} {recorded : S → C} {c0 : C}
    (q : A → D) (b : Module.Basis K R (observedKernel cell recorded i0 c0))
    (name : C → D) (hrecorded : ∀ t, q (labels t) = name (recorded t))
    (e : A → R) (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R)
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (i : I) (j : J) (c : C) (hmatch : queryCode b i j = classCode b c) :
    q (y i j) = name c := by
  obtain ⟨φ, hφ⟩ := compatible_factor_equiv q e hsurj hcompat hcard
  have hreal' : ∀ i j, φ (q (y i j)) = r i + s j := by
    intro i j
    rw [← hφ, hreal]
  have hobs' : ∀ t, q (y (cell t).1 (cell t).2) = name (recorded t) := by
    intro t
    rw [hobs]
    exact hrecorded t
  exact φ.injective (KernelCodes.matched_code_sound b name φ r s
    (fun i j => q (y i j)) hreal' hobs' i j c hmatch)

/-- Deriving the answer-code equivalence makes every complete grouped
kernel match sound for the unchanged declared readout, CM35. The raw-label
fit and unknown-code additive representation remain explicit premises. -/
theorem matched_readout_sound (q : A → D)
    (b : Module.Basis K R (observedKernel cell (q ∘ labels) i0 d0))
    (e : A → R) (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R)
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (i : I) (j : J) (d : D) (hmatch : queryCode b i j = classCode b d) :
    q (y i j) = d := by
  exact named_readout_sound (labels := labels) q b id (fun _ => rfl) e hsurj hcompat
    hcard r s y hreal hobs i j d hmatch

/-- Finite observed-only search on grouped labels identifies the same
declared answer, CM35. A failure is retained; no full-coverage assumption
or correct-answer oracle is moved into the search definition. -/
theorem choose_readout_sound (q : A → D)
    (b : Module.Basis K R (observedKernel cell (q ∘ labels) i0 d0))
    (e : A → R) (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R)
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = labels t)
    (i : I) (j : J) (bank : List D) {d : D}
    (hfound : chooseClass b (queryCode b i j) bank = some d) :
    q (y i j) = d := by
  apply matched_readout_sound q b e hsurj hcompat hcard r s y hreal hobs i j d
  exact chooseClass_match b _ bank hfound

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel
