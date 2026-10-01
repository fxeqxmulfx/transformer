/-
# Infinite nonconstant equation families with finite germ bases

All positive powers of the last coordinate generate one proper ideal.
The corresponding finite-prefix ideals form an increasing chain, and
nonlinear pullbacks are covered by the same finite basis. Auxiliary
examples for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.RealAnalyticGerms.BasicExamples

open Set

noncomputable section
namespace Transformer.RealAnalyticGerms

open AnalyticPreparation

/-- The infinite family `y,y²,y³,...` has a finite equation basis which
works for every ring-homomorphic and analytic pullback. The example
uses an actual infinite family of nonconstant germs. Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : let S : Set (AnalyticGerm 2) :=
      Set.range (fun k : ℕ => lastCoordinateGerm 1 ^ (k + 1))
    Ideal.span S = Ideal.span ({lastCoordinateGerm 1} : Set (AnalyticGerm 2)) ∧
      (∃ T : Finset (AnalyticGerm 2), (T : Set (AnalyticGerm 2)) ⊆ S ∧
        Ideal.span S = Ideal.span (T : Set (AnalyticGerm 2))) ∧
      (∃ T : Finset (AnalyticGerm 2), (T : Set (AnalyticGerm 2)) ⊆ S ∧
        ∀ {A : Type} [CommRing A] (pullback : AnalyticGerm 2 →+* A),
          (∀ f ∈ S, pullback f = 0) ↔ ∀ f ∈ T, pullback f = 0) ∧
      ∃ F : Finset (AnalyticGerm 2), (F : Set (AnalyticGerm 2)) ⊆ S ∧
        ∀ (m : ℕ) (T : Base m → Base 2) (hT : AnalyticAt ℝ T 0) (hT0 : T 0 = 0),
          (∀ f ∈ S, analyticPullbackHom T hT hT0 f = 0) ↔
            ∀ f ∈ F, analyticPullbackHom T hT hT0 f = 0 := by
  intro S
  have hspan : Ideal.span S = Ideal.span ({lastCoordinateGerm 1} : Set (AnalyticGerm 2)) := by
    apply le_antisymm
    · apply Ideal.span_le.mpr
      rintro f ⟨k, rfl⟩
      change lastCoordinateGerm 1 ^ (k + 1) ∈ _
      rw [pow_succ]
      exact Ideal.mul_mem_left _ _ (Ideal.subset_span (Set.mem_singleton _))
    · apply Ideal.span_le.mpr
      intro f hf
      obtain rfl := Set.mem_singleton_iff.mp hf
      apply Ideal.subset_span
      exact ⟨0, by simp⟩
  exact ⟨hspan, analyticGerm_finite_equations S,
    analyticGerm_finite_pullback_equations S, analyticGerm_finite_parameterized_equations S⟩

/-- Finite prefixes of the positive powers of `y` form an increasing
sequence of actual analytic-germ ideals. Rückert's theorem proves that
the sequence stabilizes. Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : ∃ I : ℕ →o Ideal (AnalyticGerm 2),
    (∀ n, I n = Ideal.span {g | ∃ k < n, g = lastCoordinateGerm 1 ^ (k + 1)}) ∧
      ∃ k : ℕ, ∀ m : ℕ, k ≤ m → I k = I m := by
  let I : ℕ →o Ideal (AnalyticGerm 2) :=
    { toFun := fun n => Ideal.span {g | ∃ k < n, g = lastCoordinateGerm 1 ^ (k + 1)}
      monotone' := by
        intro n m hnm
        apply Submodule.span_mono
        rintro g ⟨k, hk, hkg⟩
        exact ⟨k, hk.trans_le hnm, hkg⟩ }
  exact ⟨I, fun _ => rfl, analyticGerm_increasing_ideals_stabilize I⟩

end Transformer.RealAnalyticGerms
