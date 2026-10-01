/-
# Real analytic germs: Basic

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, Germs/Basic.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.AnalyticPreparation.Basic
import Mathlib.Topology.Germ
import Mathlib.RingTheory.LocalRing.MaximalIdeal.Basic

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- The ambient ring of all function germs at the origin of `ℝⁿ`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
abbrev FunctionGerm (n : ℕ) :=
  Filter.Germ (𝓝 (0 : Base n)) ℝ

/-- Function germs which have a representative analytic at the origin.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def analyticGermSubring (n : ℕ) : Subring (FunctionGerm n) where
  carrier := {φ | ∃ f : Base n → ℝ,
    AnalyticAt ℝ f 0 ∧ (f : FunctionGerm n) = φ}
  zero_mem' := ⟨0, analyticAt_const, rfl⟩
  one_mem' := ⟨1, analyticAt_const, rfl⟩
  add_mem' := by
    rintro φ ψ ⟨f, hf, rfl⟩ ⟨g, hg, rfl⟩
    exact ⟨f + g, hf.add hg, by simp⟩
  mul_mem' := by
    rintro φ ψ ⟨f, hf, rfl⟩ ⟨g, hg, rfl⟩
    exact ⟨f * g, hf.mul hg, by simp⟩
  neg_mem' := by
    rintro φ ⟨f, hf, rfl⟩
    exact ⟨-f, hf.neg, by simp⟩

/-- The commutative ring `𝒪_{ℝⁿ,0}` of analytic germs at the origin.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
abbrev AnalyticGerm (n : ℕ) := analyticGermSubring n

/-- Pass from an analytic representative to its analytic germ.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def AnalyticGerm.ofFunction {n : ℕ} (f : Base n → ℝ)
    (hf : AnalyticAt ℝ f 0) : AnalyticGerm n :=
  ⟨(f : FunctionGerm n), ⟨f, hf, rfl⟩⟩

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem AnalyticGerm.coe_ofFunction {n : ℕ} (f : Base n → ℝ)
    (hf : AnalyticAt ℝ f 0) :
    ((AnalyticGerm.ofFunction f hf : AnalyticGerm n) : FunctionGerm n) = f :=
  rfl

/-- Every analytic germ has an analytic representative.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem AnalyticGerm.exists_rep {n : ℕ} (φ : AnalyticGerm n) :
    ∃ f : Base n → ℝ,
      AnalyticAt ℝ f 0 ∧ (f : FunctionGerm n) = φ :=
  φ.property

/-- Evaluation at the origin, as a ring homomorphism.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def evalAtOriginHom (n : ℕ) : AnalyticGerm n →+* ℝ :=
  (Filter.Germ.valueRingHom : FunctionGerm n →+* ℝ).comp
    (analyticGermSubring n).subtype

/-- Evaluation of an analytic germ at the origin.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
abbrev evalAtOrigin {n : ℕ} (φ : AnalyticGerm n) : ℝ :=
  evalAtOriginHom n φ

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp]
theorem evalAtOrigin_ofFunction {n : ℕ} (f : Base n → ℝ)
    (hf : AnalyticAt ℝ f 0) :
    evalAtOrigin (AnalyticGerm.ofFunction f hf) = f 0 :=
  rfl

instance analyticGerm_nontrivial (n : ℕ) : Nontrivial (AnalyticGerm n) := by
  refine ⟨⟨0, 1, ?_⟩⟩
  intro h
  exact (zero_ne_one : (0 : ℝ) ≠ 1)
    (by simpa using congrArg (evalAtOriginHom n) h)

/-- An analytic germ is a unit exactly when its value at the origin is nonzero.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_isUnit_iff {n : ℕ} (φ : AnalyticGerm n) :
    IsUnit φ ↔ evalAtOrigin φ ≠ 0 := by
  constructor
  · intro hφ
    exact isUnit_iff_ne_zero.mp (hφ.map (evalAtOriginHom n))
  · intro hφ0
    obtain ⟨f, hf, hrep⟩ := φ.property
    have hf0 : f 0 ≠ 0 := by
      change Filter.Germ.value (φ : FunctionGerm n) ≠ 0 at hφ0
      rw [← hrep] at hφ0
      simpa using hφ0
    let ψ : AnalyticGerm n :=
      ⟨((f⁻¹ : Base n → ℝ) : FunctionGerm n),
        ⟨(f⁻¹ : Base n → ℝ), hf.inv hf0, rfl⟩⟩
    have hne : ∀ᶠ x in 𝓝 (0 : Base n), f x ≠ 0 :=
      hf.continuousAt.eventually_ne hf0
    have hmul : φ * ψ = 1 := by
      apply Subtype.ext
      change (φ : FunctionGerm n) *
        ((f⁻¹ : Base n → ℝ) : FunctionGerm n) = 1
      rw [← hrep, ← Filter.Germ.coe_mul, ← Filter.Germ.coe_one]
      apply Filter.Germ.coe_eq.mpr
      filter_upwards [hne] with x hx
      exact mul_inv_cancel₀ hx
    have hmul' : ψ * φ = 1 := by
      rw [mul_comm, hmul]
    exact ⟨⟨φ, ψ, hmul, hmul'⟩, rfl⟩

/-- The analytic germ ring is local.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_isLocalRing (n : ℕ) : IsLocalRing (AnalyticGerm n) := by
  apply IsLocalRing.of_isUnit_or_isUnit_one_sub_self
  intro φ
  by_cases hφ : evalAtOrigin φ = 0
  · right
    apply (analyticGerm_isUnit_iff (1 - φ)).2
    simp [hφ]
  · exact Or.inl ((analyticGerm_isUnit_iff φ).2 hφ)

instance analyticGerm_instIsLocalRing (n : ℕ) : IsLocalRing (AnalyticGerm n) :=
  analyticGerm_isLocalRing n

/-- The maximal ideal consists exactly of germs vanishing at the origin.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_maximalIdeal (n : ℕ) :
    IsLocalRing.maximalIdeal (AnalyticGerm n) =
      Ideal.comap (evalAtOriginHom n) ⊥ := by
  ext φ
  rw [IsLocalRing.mem_maximalIdeal]
  simp only [mem_nonunits_iff, Ideal.mem_comap, Ideal.mem_bot]
  simpa only [not_ne_iff] using not_congr (analyticGerm_isUnit_iff φ)

end Transformer.RealAnalyticGerms
