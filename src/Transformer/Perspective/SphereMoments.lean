/-
# Moments determine a measure on the sphere

A finite measure on `𝕊^{d-1}` is determined by its moments
`m_I(μ) = ∫ x_{I 0} ⋯ x_{I (k-1)} dμ(x)` (`measure_eq_of_integral_mono_eq`):
the coordinates separate the points of the sphere, so by Stone–Weierstrass the
algebra they generate is dense in `C(𝕊^{d-1})`, and a finite measure is
determined by its integrals against a dense algebra.  This is the last step of
every positive-definiteness argument for the kernel `e^{β x·y}`
(`PositiveDefinite`).

Source: arXiv:2601.21366v2, proof of `lem: quadpol` ("spherical polynomials
are uniformly dense in `C⁰(𝕊^{d-1})`").
-/

import Transformer.Perspective.Section2_EnergySeries
import Mathlib.MeasureTheory.Measure.FiniteMeasureExt

open scoped BigOperators BoundedContinuousFunction
open MeasureTheory

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-! ### Moments determine the measure -/

/-- The coordinate `x ↦ x_i` of the sphere, as a bounded continuous function. -/
noncomputable def coordBCF (i : Fin d) : SSphere d →ᵇ ℝ :=
  BoundedContinuousFunction.mkOfCompact ⟨fun x => (x : EucSpace d) i, by fun_prop⟩

@[simp] theorem coordBCF_apply (i : Fin d) (x : SSphere d) :
    coordBCF i x = (x : EucSpace d) i := rfl

/-- A product of coordinates is a monomial `x ↦ x_{I 0} ⋯ x_{I (k-1)}`. -/
theorem exists_mono_of_mem_closure {g : SSphere d →ᵇ ℝ}
    (hg : g ∈ Submonoid.closure (Set.range coordBCF ∪ star (Set.range coordBCF))) :
    ∃ k, ∃ I : Fin k → Fin d, ∀ x : SSphere d, g x = mono I (x : EucSpace d) := by
  induction hg using Submonoid.closure_induction with
  | mem g hg =>
    obtain ⟨i, rfl⟩ : ∃ i, coordBCF (d := d) i = g := by
      rcases hg with hg | hg
      · exact hg
      · obtain ⟨i, hi⟩ := Set.mem_star.1 hg
        exact ⟨i, by rw [hi]; ext x; simp⟩
    exact ⟨1, fun _ => i, fun x => by simp [mono]⟩
  | one => exact ⟨0, Fin.elim0, fun x => by simp [mono]⟩
  | mul f g _ _ hf hg =>
    obtain ⟨k, I, hI⟩ := hf
    obtain ⟨l, J, hJ⟩ := hg
    refine ⟨k + l, Fin.append I J, fun x => ?_⟩
    rw [BoundedContinuousFunction.mul_apply, hI, hJ, mono, mono,
      mono, Fin.prod_univ_add]
    simp [Fin.append_left, Fin.append_right]

/-- **A finite measure on `𝕊^{d-1}` is determined by its moments.**  If
`∫ x_{I 0} ⋯ x_{I (k-1)} dμ₁ = ∫ x_{I 0} ⋯ x_{I (k-1)} dμ₂` for every `k` and
every `I : Fin k → Fin d`, then `μ₁ = μ₂`.

The step the source takes as "spherical polynomials are uniformly dense in
`C⁰(𝕊^{d-1})`" and "uniqueness in the Riesz representation theorem": the
coordinates separate points, so the star algebra they generate satisfies the
hypothesis of Mathlib's Stone–Weierstrass uniqueness theorem for finite measures.

Source: arXiv:2601.21366v2, proof of `lem: quadpol` (injectivity). -/
theorem measure_eq_of_integral_mono_eq (μ₁ μ₂ : Measure (SSphere d))
    [IsFiniteMeasure μ₁] [IsFiniteMeasure μ₂]
    (h : ∀ k (I : Fin k → Fin d), ∫ x, mono I (x : EucSpace d) ∂μ₁ =
      ∫ x, mono I (x : EucSpace d) ∂μ₂) : μ₁ = μ₂ := by
  refine ext_of_forall_mem_subalgebra_integral_eq_of_pseudoEMetric_complete_countable
    (A := StarAlgebra.adjoin ℝ (Set.range (coordBCF (d := d)))) ?_ ?_
  · intro x y hxy
    obtain ⟨i, hi⟩ : ∃ i, (x : EucSpace d) i ≠ (y : EucSpace d) i := by
      by_contra hne
      push Not at hne
      exact hxy (Subtype.ext (PiLp.ext hne))
    refine ⟨_, ⟨BoundedContinuousFunction.toContinuousMapStarₐ ℝ (coordBCF i), ?_, rfl⟩, ?_⟩
    · exact StarSubalgebra.mem_map.2
        ⟨coordBCF i, StarAlgebra.subset_adjoin ℝ _ ⟨i, rfl⟩, rfl⟩
    · simpa using hi
  · intro g hg
    have hg' : g ∈ Subalgebra.toSubmodule
        (Algebra.adjoin ℝ (Set.range (coordBCF (d := d)) ∪ star (Set.range coordBCF))) := hg
    rw [Algebra.adjoin_eq_span] at hg'
    clear hg
    induction hg' using Submodule.span_induction with
    | mem g hg =>
      obtain ⟨k, I, hI⟩ := exists_mono_of_mem_closure hg
      simp_rw [hI]
      exact h k I
    | zero => simp
    | add f g _ _ hf hg =>
      simp only [BoundedContinuousFunction.coe_add, Pi.add_apply]
      rw [integral_add (f.integrable _) (g.integrable _),
        integral_add (f.integrable _) (g.integrable _), hf, hg]
    | smul a f _ hf =>
      simp only [BoundedContinuousFunction.coe_smul, smul_eq_mul]
      rw [integral_const_mul, integral_const_mul, hf]

/-- The hypothesis of `measure_eq_of_integral_mono_eq` is satisfiable: a
measure and itself. -/
example : ∀ k (I : Fin k → Fin 1),
    ∫ x, mono I (x : EucSpace 1) ∂(0 : Measure (SSphere 1)) =
      ∫ x, mono I (x : EucSpace 1) ∂(0 : Measure (SSphere 1)) :=
  fun _ _ => rfl

end Perspective
end Transformer
