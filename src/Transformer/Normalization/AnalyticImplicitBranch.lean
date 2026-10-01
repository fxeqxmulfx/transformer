/-
# Analytic branches of nonsingular systems

The regular step in analytic curve selection is an implicit-function
construction. This module proves that step for real Banach spaces, with
analyticity and local uniqueness of the branch. Singular systems still
require analytic preparation and ramification.
-/

import Mathlib.Analysis.Calculus.FDeriv.Analytic
import Mathlib.Analysis.Calculus.InverseFunctionTheorem.FDeriv
import Mathlib.Analysis.Calculus.FDeriv.Pow

open Filter

namespace Transformer.Normalization

variable {X Y : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X] [CompleteSpace X]
  [NormedAddCommGroup Y] [NormedSpace ℝ Y] [CompleteSpace Y]

/-- A real analytic system with an invertible derivative in its unknowns
has a locally unique analytic solution depending on the parameters. This
is the nonsingular branch construction used before treating singular
analytic curve selection for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. The invertibility hypothesis is explicit. -/
theorem analytic_implicit_branch (F : X × Y → Y)
    (hF : AnalyticAt ℝ F 0) (hF0 : F 0 = 0) (A : Y ≃L[ℝ] Y)
    (hpartial : ∀ y : Y, (fderiv ℝ F 0) (0, y) = A y) :
    ∃ g : X → Y, AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ x in nhds (0 : X), F (x, g x) = 0) ∧
      ∀ᶠ p in nhds (0 : X × Y), F p = 0 → p.2 = g p.1 := by
  let D := fderiv ℝ F 0
  let L : (X × Y) →L[ℝ] (X × Y) := (ContinuousLinearMap.fst ℝ X Y).prod D
  have hsplit (p : X × Y) : D p = D (p.1, 0) + A p.2 := by
    calc
      D p = D ((p.1, 0) + (0, p.2)) := by congr 1; ext <;> simp
      _ = D (p.1, 0) + D (0, p.2) := map_add _ _ _
      _ = D (p.1, 0) + A p.2 := by rw [hpartial]
  have hinj : Function.Injective L := by
    intro p q hpq
    have hx : p.1 = q.1 := by
      have hp : (L p).1 = (L q).1 := congrArg Prod.fst hpq
      change p.1 = q.1 at hp
      exact hp
    have hy : D p = D q := congrArg Prod.snd hpq
    rw [hsplit p, hsplit q, hx] at hy
    exact Prod.ext hx (A.injective (add_left_cancel hy))
  have hsurj : Function.Surjective L := by
    intro p
    refine ⟨(p.1, A.symm (p.2 - D (p.1, 0))), ?_⟩
    apply Prod.ext
    · rfl
    · change D (p.1, A.symm (p.2 - D (p.1, 0))) = p.2
      rw [hsplit, A.apply_symm_apply]
      simp
  let B : (X × Y) ≃L[ℝ] (X × Y) := ContinuousLinearEquiv.ofBijective L
    (LinearMap.ker_eq_bot.mpr hinj) (LinearMap.range_eq_top.mpr hsurj)
  have hB : B.toContinuousLinearMap = L := ContinuousLinearEquiv.coe_ofBijective _ _ _
  let phi : X × Y → X × Y := fun p => (p.1, F p)
  have hphi : AnalyticAt ℝ phi 0 := analyticAt_fst.prod hF
  have hdphi : HasFDerivAt phi L 0 :=
    hasFDerivAt_fst.prodMk hF.hasStrictFDerivAt.hasFDerivAt
  have hstrict : HasStrictFDerivAt phi B.toContinuousLinearMap 0 := by
    rw [hB, ← hdphi.fderiv]
    exact hphi.hasStrictFDerivAt
  let R := hstrict.toOpenPartialHomeomorph phi
  have hR : AnalyticAt ℝ R 0 := by
    simpa only [R, HasStrictFDerivAt.toOpenPartialHomeomorph_coe] using hphi
  have hdR : fderiv ℝ R 0 = B.toContinuousLinearMap := by
    simpa only [R, HasStrictFDerivAt.toOpenPartialHomeomorph_coe] using
      hstrict.hasFDerivAt.fderiv
  have hinverse : AnalyticAt ℝ R.symm (R 0) :=
    R.analyticAt_symm' hstrict.mem_toOpenPartialHomeomorph_source hR hdR
  have hphi0 : phi 0 = 0 := by simp [phi, hF0]
  have hR0 : R 0 = 0 := hphi0
  have hinverse0 : AnalyticAt ℝ R.symm 0 := hR0 ▸ hinverse
  have hleft0 : R.symm 0 = 0 := by
    simpa only [HasStrictFDerivAt.localInverse, hphi0, R] using
      hstrict.localInverse_apply_image
  let g : X → Y := fun x => (R.symm (x, 0)).2
  have haxis : AnalyticAt ℝ (fun x : X => (x, (0 : Y))) 0 :=
    analyticAt_id.prod analyticAt_const
  have hcomp : AnalyticAt ℝ (fun x : X => R.symm (x, 0)) 0 := by
    simpa only [Function.comp_def] using
      hinverse0.comp (f := fun x : X => (x, (0 : Y))) haxis
  refine ⟨g, analyticAt_snd.comp hcomp, ?_, ?_, ?_⟩
  · simpa only [g, ← Prod.zero_eq_mk, hleft0] using (Prod.snd_zero : (0 : X × Y).2 = 0)
  · have ht : Tendsto (fun x : X => (x, (0 : Y))) (nhds 0) (nhds 0) := by
      simpa only [← Prod.zero_eq_mk] using haxis.continuousAt.tendsto
    have hright : ∀ᶠ p in nhds (0 : X × Y), phi (R.symm p) = p := by
      simpa only [HasStrictFDerivAt.localInverse, hphi0, R] using
        hstrict.eventually_right_inverse
    filter_upwards [ht.eventually hright] with x hx
    have hx1 : (R.symm (x, 0)).1 = x := congrArg Prod.fst hx
    have hx2 : F (R.symm (x, 0)) = 0 := congrArg Prod.snd hx
    have hpair : R.symm (x, 0) = (x, g x) := Prod.ext hx1 rfl
    rwa [hpair] at hx2
  · filter_upwards [hstrict.eventually_left_inverse] with p hp hzero
    have heq : R.symm (p.1, (0 : Y)) = p := by
      simpa only [HasStrictFDerivAt.localInverse, R, phi, hzero] using hp
    exact (congrArg Prod.snd heq).symm

/-- The nonlinear equation `y + x² = 0` satisfies all implicit branch
hypotheses simultaneously. Auxiliary example for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ g : ℝ → ℝ, AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
    (∀ᶠ x in nhds (0 : ℝ), g x + x ^ 2 = 0) ∧
    ∀ᶠ p in nhds (0 : ℝ × ℝ), p.2 + p.1 ^ 2 = 0 → p.2 = g p.1 := by
  let F : ℝ × ℝ → ℝ := fun p => p.2 + p.1 ^ 2
  have hF : AnalyticAt ℝ F 0 := analyticAt_snd.add (analyticAt_fst.fun_pow 2)
  have hx : HasFDerivAt (fun p : ℝ × ℝ => p.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) 0 :=
    hasFDerivAt_fst
  have hy : HasFDerivAt (fun p : ℝ × ℝ => p.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) 0 :=
    hasFDerivAt_snd
  have hd : HasFDerivAt F (ContinuousLinearMap.snd ℝ ℝ ℝ) 0 := by
    convert hy.add (hx.pow 2) using 1
    simp
  exact analytic_implicit_branch F hF (by simp [F]) (ContinuousLinearEquiv.refl ℝ ℝ)
    (by intro y; rw [hd.fderiv]; rfl)

end Transformer.Normalization
