module

public import Transformer.Perceptron.Section6_BranchIdentity
public import Mathlib.MeasureTheory.Measure.Prod
public import Mathlib.MeasureTheory.Measure.Support
public import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
public import Mathlib.MeasureTheory.Measure.Typeclasses.NullSingletonClass

/-!
# Analytic zero sets under a measure-preserving family

ArXiv:2601.21366v2, §6, proof of `prop: unified.log`, part (a),
uses uniqueness of real-analytic functions from a positive-measure
spherical zero set. Here is the measure-theoretic step for that argument.

For a continuous family `R(t)` preserving the actual finite measure,
assume each actual orbit `t ↦ f(R(t)x)` is analytic on the real line,
and `R(0)=id`. Each orbit either vanishes identically or has finitely
many zeros in `[0,1]`. Fubini on the original product measure shows
that the zero set of `f` differs by a null set from the set of points
whose complete orbit vanishes. That latter set is closed. Therefore
every point in the support of the actual zero-set restriction has a
globally zero orbit, for every supplied measure-preserving family.

The argument keeps the original measure and its topological support.
No analytic-manifold structure or spherical density is assumed. Its
application to actual orthogonal sphere rotations is a separate step.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped ENNReal

namespace Transformer.Perceptron

/-- A nonzero analytic function has only finitely many zeros in a
compact real interval. Source: arXiv:2601.21366v2, §6, the identity
theorem in `prop: unified.log`. The interval endpoints are included,
so accumulation at an activation boundary is covered too. -/
theorem finite_zeros_analytic_Icc (f : ℝ → ℝ) (hf : AnalyticOnNhd ℝ f univ)
    (hne : ∃ t, f t ≠ 0) (a b : ℝ) : ({t | f t = 0} ∩ Icc a b).Finite := by
  by_contra hInf
  obtain ⟨t, ht⟩ := hne
  exact ht (eq_zero_of_analytic_infinite_zeros f hf (Icc a b) isCompact_Icc
    ({s | f s = 0} ∩ Icc a b) hInf (fun _ hs => hs.2) (fun _ hs => hs.1) t)

/-- The isolated-zero alternative is actually Lebesgue null on the
compact time interval. Source: arXiv:2601.21366v2, §6, the
positive-measure uniqueness argument of `prop: unified.log`.
This is used before integrating over the actual sphere measure. -/
theorem volume_zeros_analytic_Icc_eq_zero (f : ℝ → ℝ)
    (hf : AnalyticOnNhd ℝ f univ) (hne : ∃ t, f t ≠ 0) (a b : ℝ) :
    volume ({t | f t = 0} ∩ Icc a b) = 0 :=
  (finite_zeros_analytic_Icc f hf hne a b).measure_zero volume

/-- Both analytic regularity and nonzero-function inputs hold for
the actual identity function, with its genuine zero at the origin.
Source: arXiv:2601.21366v2, §6, the isolated-zero alternative. -/
example : AnalyticOnNhd ℝ (fun t : ℝ => t) univ ∧
    (∃ t : ℝ, (fun s : ℝ => s) t ≠ 0) :=
  ⟨fun _ _ => analyticAt_id, ⟨1, one_ne_zero⟩⟩

/-- The support of the actual zero-set restriction consists of complete
zero orbits of any continuous analytic measure-preserving family starting
at the identity. Source: arXiv:2601.21366v2, §6, the positive-measure
uniqueness step in part (a) of `prop: unified.log`.
The family is actual point maps, and its invariance is an equality of the
original pushforward measure. The finite measure need not have a density. -/
theorem support_zero_restrict_subset_zero_orbits
    {X : Type*} [TopologicalSpace X] [MeasurableSpace X] [BorelSpace X]
    (ν : Measure X) [IsFiniteMeasure ν] (f : X → ℝ) (hf : Continuous f)
    (R : ℝ → X → X) (hR : Continuous (fun p : ℝ × X => R p.1 p.2))
    (hR₀ : ∀ x, R 0 x = x) (hν : ∀ t, ν.map (R t) = ν)
    (hana : ∀ x, AnalyticOnNhd ℝ (fun t => f (R t x)) univ) :
    (ν.restrict {x | f x = 0}).support ⊆ {x | ∀ t, f (R t x) = 0} := by
  let S : Set X := {x | f x = 0}
  let M : Set X := {x | ∀ t, f (R t x) = 0}
  let Z : Set (ℝ × X) := {p | f (R p.1 p.2) = 0}
  let τ : Measure ℝ := volume.restrict (Icc (0 : ℝ) 1)
  have hτ : IsProbabilityMeasure τ := ⟨by simp [τ, Real.volume_Icc]⟩
  have hRt (t : ℝ) : Continuous (R t) := hR.comp (continuous_const.prodMk continuous_id)
  have hS : IsClosed S := isClosed_eq hf continuous_const
  have hM : IsClosed M := by
    simp only [M, ofPred_forall]
    exact isClosed_iInter (fun t => isClosed_eq (hf.comp (hRt t)) continuous_const)
  have hZ : MeasurableSet Z := (isClosed_eq (hf.comp hR) continuous_const).measurableSet
  have hMS : M ⊆ S := fun x hx => by
    have h := hx 0
    rwa [hR₀ x] at h
  have hmass (t : ℝ) : ν {x | f (R t x) = 0} = ν S := by
    have h := Measure.map_apply (μ := ν) (hRt t).measurable hS.measurableSet
    rw [hν t] at h
    exact h.symm
  have htime (x : X) : τ {t | f (R t x) = 0} = M.indicator (1 : X → ℝ≥0∞) x := by
    by_cases hx : x ∈ M
    · have heq : {t | f (R t x) = 0} = univ := by
        ext t
        exact iff_true_intro (hx t)
      rw [heq, measure_univ, indicator_of_mem hx]
      rfl
    · have hne : ∃ t, f (R t x) ≠ 0 := by
        simpa only [M, mem_ofPred_eq, not_forall] using hx
      change (volume.restrict (Icc (0 : ℝ) 1)) {t | f (R t x) = 0} = _
      rw [Measure.restrict_apply (isClosed_eq (hana x).continuous continuous_const).measurableSet,
        indicator_of_notMem hx]
      exact volume_zeros_analytic_Icc_eq_zero (fun t => f (R t x)) (hana x) hne 0 1
  have hfirst : τ.prod ν Z = ν S := by
    rw [Measure.prod_apply hZ]
    change (∫⁻ t, ν {x | f (R t x) = 0} ∂τ) = ν S
    simp only [hmass, lintegral_const, measure_univ, mul_one]
  have hsecond : τ.prod ν Z = ν M := by
    rw [Measure.prod_apply_symm hZ]
    change (∫⁻ x, τ {t | f (R t x) = 0} ∂ν) = ν M
    simp only [htime]
    exact lintegral_indicator_one hM.measurableSet
  have heq : ν M = ν S := hsecond.symm.trans hfirst
  have hdiff : ν (S \ M) = 0 := by
    rw [measure_sdiff hMS hM.measurableSet.nullMeasurableSet (measure_ne_top ν M), heq,
      tsub_self]
  apply Measure.support_subset_of_isClosed hM
  change ∀ᵐ x ∂ν.restrict S, x ∈ M
  rw [ae_iff]
  change (ν.restrict S) Mᶜ = 0
  rw [Measure.restrict_apply hM.measurableSet.compl]
  simpa only [sdiff_eq, inter_comm] using hdiff

/-- The finite measure, continuity, identity-at-zero, invariance and
analytic-orbit hypotheses all hold jointly for a nonzero Dirac measure,
the nonconstant real identity function, and the actual identity family.
Its zero restriction has the genuine support point `0`.
Source: arXiv:2601.21366v2, §6, the support argument in
`prop: unified.log`. No empty measure is used for this witness. -/
example : IsFiniteMeasure (Measure.dirac (0 : ℝ)) ∧
    Continuous (fun x : ℝ => x) ∧
    Continuous (fun p : ℝ × ℝ => (fun (_ : ℝ) x => x) p.1 p.2) ∧
    (∀ x : ℝ, (fun (_ : ℝ) y => y) 0 x = x) ∧
    (∀ t : ℝ, (Measure.dirac (0 : ℝ)).map (fun x : ℝ => (fun (_ : ℝ) y => y) t x) =
      Measure.dirac 0) ∧
    ∀ x : ℝ, AnalyticOnNhd ℝ (fun t : ℝ => (fun y : ℝ => y) ((fun (_ : ℝ) y => y) t x))
      univ :=
  ⟨inferInstance, continuous_id, continuous_snd, fun _ => rfl,
    fun _ => Measure.map_id, fun _ _ _ => analyticAt_const⟩

/-- The zero restriction in the preceding example has an actual
nonempty support. Source: arXiv:2601.21366v2, §6, choosing a
support point of the positive-measure zero restriction. -/
example : (0 : ℝ) ∈ ((Measure.dirac (0 : ℝ)).restrict {x : ℝ | x = 0}).support := by
  rw [restrict_dirac]
  simp only [mem_ofPred_eq, ite_true]
  apply (Measure.mem_support_iff_forall 0).mpr
  intro U hU
  have hx : (0 : ℝ) ∈ U := mem_of_mem_nhds hU
  change 0 < Measure.dirac (0 : ℝ) U
  simp [hx]

end Transformer.Perceptron
