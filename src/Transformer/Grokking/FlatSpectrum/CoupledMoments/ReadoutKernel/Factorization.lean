module

public import Mathlib.Data.Fintype.EquivFin

/-!
# Compatible finite codes have exactly the declared readout fibers

Source: `review/flat_spectrum.md`, CM35. If a surjective declared readout has
the same finite cardinality as the unknown code alphabet, CM34's compatibility
premise forces a bijective factorization of that unknown code through the
readout. Factorization is a conclusion, not a definition or added hypothesis.
The raw vocabulary may contain multiple aliases for each answer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

variable {A D R : Type*} [Fintype D] [Fintype R]

/-- Equal finite code and answer cardinalities turn one-way compatibility
into an actual bijection on answers, CM35. No code value or inverse readout
is supplied to a learner; the equivalence exists in the soundness proof. -/
theorem compatible_factor_equiv (q : A → D) (e : A → R)
    (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R) :
    ∃ φ : D ≃ R, ∀ a, e a = φ (q a) := by
  classical
  obtain ⟨s, hs⟩ := hsurj.hasRightInverse
  let f : D → R := fun d => e (s d)
  have hinj : Function.Injective f := by
    intro d d' heq
    have h := hcompat (s d) (s d') heq
    simpa only [hs d, hs d'] using h
  have hbij : Function.Bijective f :=
    (Fintype.bijective_iff_injective_and_card f).mpr ⟨hinj, hcard⟩
  refine ⟨Equiv.ofBijective f hbij, ?_⟩
  intro a
  obtain ⟨d, hd⟩ := hbij.2 (e a)
  have hq : d = q a := by
    have h := hcompat (s d) a hd
    simpa only [hs d] using h
  change e a = f (q a)
  rw [← hq]
  exact hd.symm

/-- Under the same cardinality premise, all aliases of an answer have
equal unknown codes, and conversely, CM35. These are additional linear
class-coordinate equalities that the ungrouped observed system omitted. -/
theorem compatible_fibers_iff (q : A → D) (e : A → R)
    (hsurj : Function.Surjective q)
    (hcompat : ∀ a a', e a = e a' → q a = q a')
    (hcard : Fintype.card D = Fintype.card R) (a a' : A) :
    e a = e a' ↔ q a = q a' := by
  obtain ⟨φ, hφ⟩ := compatible_factor_equiv q e hsurj hcompat hcard
  constructor
  · exact hcompat a a'
  · intro h
    rw [hφ a, hφ a', h]

example : ∃ φ : Fin 2 ≃ Fin 2,
    ∀ a : Fin 2 × Bool, a.1 = φ a.1 :=
  compatible_factor_equiv Prod.fst Prod.fst (fun d => ⟨(d, false), rfl⟩)
    (fun _ _ h => h) rfl

example : ((1, false) : Fin 2 × Bool).1 = (1, true).1 ↔
    ((1, false) : Fin 2 × Bool).1 = (1, true).1 :=
  compatible_fibers_iff Prod.fst Prod.fst (fun d => ⟨(d, false), rfl⟩)
    (fun _ _ h => h) rfl (1, false) (1, true)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel
