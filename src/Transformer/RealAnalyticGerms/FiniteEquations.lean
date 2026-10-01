/-
# Finite bases for families of real analytic germ equations

Rückert's theorem gives a finite subfamily with the same generated ideal.
Equations after any ring-homomorphic pullback are therefore determined by
that subfamily. The conclusions concern germs; they do not assert a common
neighborhood for arbitrary infinitely many function representatives.
-/

import Transformer.RealAnalyticGerms.Noetherian

open Set

namespace Transformer.RealAnalyticGerms

/-- Every family of real analytic germ equations has a finite subfamily
with exactly the same ideal. Auxiliary for finite analytic elimination in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_finite_equations {n : ℕ} (S : Set (AnalyticGerm n)) :
    ∃ T : Finset (AnalyticGerm n), (T : Set (AnalyticGerm n)) ⊆ S ∧
      Ideal.span S = Ideal.span (T : Set (AnalyticGerm n)) := by
  exact (Submodule.fg_span_iff_fg_span_finset_subset S).mp
    (IsNoetherian.noetherian (Ideal.span S))

/-- One finite family controls the vanishing of all equations under every
ring-homomorphic pullback, including the analytic pullbacks used in curve
selection. The finite family is independent of the pullback. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_finite_pullback_equations {n : ℕ} (S : Set (AnalyticGerm n)) :
    ∃ T : Finset (AnalyticGerm n), (T : Set (AnalyticGerm n)) ⊆ S ∧
      ∀ {A : Type} [CommRing A] (pullback : AnalyticGerm n →+* A),
        (∀ f ∈ S, pullback f = 0) ↔ ∀ f ∈ T, pullback f = 0 := by
  obtain ⟨T, hTS, hspan⟩ := analyticGerm_finite_equations S
  refine ⟨T, hTS, ?_⟩
  intro A inst pullback
  constructor
  · intro h f hf
    exact h f (hTS hf)
  · intro h f hf
    have hTker : Ideal.span (T : Set (AnalyticGerm n)) ≤ RingHom.ker pullback := by
      apply Ideal.span_le.mpr
      intro g hg
      exact h g hg
    have hfspan : f ∈ Ideal.span S := Ideal.subset_span hf
    rw [hspan] at hfspan
    exact hTker hfspan

/-- An increasing chain of ideals of real analytic germs stabilizes.
Auxiliary for the finite analytic conditions underlying Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_increasing_ideals_stabilize {n : ℕ}
    (I : ℕ →o Ideal (AnalyticGerm n)) :
    ∃ k : ℕ, ∀ m : ℕ, k ≤ m → I k = I m := by
  exact monotone_stabilizes_iff_noetherian.mpr inferInstance I

end Transformer.RealAnalyticGerms
