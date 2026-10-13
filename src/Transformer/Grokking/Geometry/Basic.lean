import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic

/-!
# The actual finite-cell projection used by grokking probes

Source: Nanda et al., arXiv:2301.05217v1, section 5.1, restricted and
excluded logits. Deviation: lab.infrastructure.engine.grokking at 4436290
averages fixed division orbits instead of keeping frequencies selected
from the final network. Each cell may also contain only held-out points.

The projection is computed as an arithmetic mean, not defined by assuming
orthogonality or an energy identity. The theorems derive those properties
from finite sums. Empty cells have zero sum and mean under real division;
this convention gives no evidence of held-out coverage. The observer's
minimum-cell-size and positive-energy checks remain separate conditions.

These scalar identities apply coordinatewise to logits. Summing them over
classes and cells will justify the probe's energy decomposition even when
held-out cell sizes differ. No property of a learned classifier follows
from squared energy alone; target labels do not enter the projection.
-/

namespace Transformer.Grokking.Geometry

open scoped BigOperators

variable {I : Type*}

/-- Arithmetic cell mean. Source: the fixed-orbit adaptation of restricted
logits in arXiv:2301.05217v1, section 5.1, implemented at 4436290. -/
noncomputable def meanOver (s : Finset I) (f : I → ℝ) : ℝ :=
  (∑ i ∈ s, f i) / s.card

/-- Unnormalized squared energy on exactly the selected points. Source:
the energy observer at 4436290; unlike loss in arXiv:2301.05217v1,
section 5.1, this quantity has no correctness interpretation by itself. -/
noncomputable def energyOver (s : Finset I) (f : I → ℝ) : ℝ :=
  ∑ i ∈ s, (f i) ^ 2

/-- Energy remaining after subtracting the actual cell mean. Source:
the excluded-component adaptation of arXiv:2301.05217v1, section 5.1.
No orthogonality or desired energy equation is assumed in this definition. -/
noncomputable def residualOver (s : Finset I) (f : I → ℝ) : ℝ :=
  energyOver s (fun i => f i - meanOver s f)

/-- The computed mean recovers the cell sum, including the empty case.
Source: arithmetic averaging in the fixed-orbit adaptation of
arXiv:2301.05217v1, section 5.1, at 4436290. -/
theorem card_mul_meanOver (s : Finset I) (f : I → ℝ) :
    (s.card : ℝ) * meanOver s f = ∑ i ∈ s, f i := by
  classical
  by_cases hs : s = ∅
  · subst s
    simp [meanOver]
  · have hpos : 0 < s.card := Finset.card_pos.mpr (Finset.nonempty_iff_ne_empty.mpr hs)
    have hcard : (s.card : ℝ) ≠ 0 := by exact_mod_cast (by omega : s.card ≠ 0)
    unfold meanOver
    field_simp

/-- Subtracting this mean, rather than an arbitrary offset, makes the
finite residual sum exactly zero. Source: the projection adaptation of
arXiv:2301.05217v1, section 5.1, at 4436290. -/
theorem centered_sum_zero (s : Finset I) (f : I → ℝ) :
    (∑ i ∈ s, (f i - meanOver s f)) = 0 := by
  rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
    card_mul_meanOver]
  ring

/-- A nonempty cell's constant is preserved by the projection. Source:
the fixed-cell averaging adaptation of arXiv:2301.05217v1, section 5.1.
Nonemptiness is necessary: the empty-cell mean is zero for every input. -/
theorem meanOver_const (s : Finset I) (c : ℝ) (hs : s.Nonempty) :
    meanOver s (fun _ => c) = c := by
  have hpos : 0 < s.card := Finset.card_pos.mpr hs
  have hcard : (s.card : ℝ) ≠ 0 := by exact_mod_cast (by omega : s.card ≠ 0)
  unfold meanOver
  rw [Finset.sum_const, nsmul_eq_mul]
  field_simp

example : ({0, 1} : Finset ℕ).Nonempty := by
  exact ⟨0, by norm_num⟩

/-- The actual projection and its residual are orthogonal on the selected
cell. Source: the fixed-orbit adaptation of arXiv:2301.05217v1, section 5.1.
This is derived from the mean; it is not a premise of the energy theorem. -/
theorem projection_residual_orthogonal (s : Finset I) (f : I → ℝ) :
    (∑ i ∈ s, meanOver s f * (f i - meanOver s f)) = 0 := by
  rw [← Finset.mul_sum, centered_sum_zero]
  ring

/-- Squared energy splits exactly into cell-mean and residual energies.
Source: the orbit-energy observer at 4436290, adapted from restricted and
excluded components in arXiv:2301.05217v1, section 5.1. -/
theorem energy_decomposition (s : Finset I) (f : I → ℝ) :
    energyOver s f = (s.card : ℝ) * (meanOver s f) ^ 2 + residualOver s f := by
  have he : ∀ i, (f i) ^ 2 = (f i - meanOver s f) ^ 2 +
      2 * meanOver s f * (f i - meanOver s f) + (meanOver s f) ^ 2 := by
    intro i
    ring
  unfold residualOver energyOver
  simp_rw [he]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
    centered_sum_zero, Finset.sum_const, nsmul_eq_mul]
  ring

/-- Moving the fitted constant away from the mean adds precisely a
cardinality-weighted squared error. Source: the cell projection in the
adaptation of arXiv:2301.05217v1, section 5.1, at 4436290. -/
theorem constant_fit_error (s : Finset I) (f : I → ℝ) (c : ℝ) :
    energyOver s (fun i => f i - c) = residualOver s f +
      (s.card : ℝ) * (meanOver s f - c) ^ 2 := by
  have he : ∀ i, (f i - c) ^ 2 = (f i - meanOver s f) ^ 2 +
      2 * (meanOver s f - c) * (f i - meanOver s f) +
      (meanOver s f - c) ^ 2 := by
    intro i
    ring
  unfold residualOver energyOver
  simp_rw [he]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
    centered_sum_zero, Finset.sum_const, nsmul_eq_mul]
  ring

/-- The computed cell mean minimizes squared error among all constants.
Source: the projection interpretation of the orbit observer at 4436290,
adapted from arXiv:2301.05217v1, section 5.1. This is a functional fit,
not convexity of the transformer's training parameters. -/
theorem mean_minimizes_error (s : Finset I) (f : I → ℝ) (c : ℝ) :
    residualOver s f ≤ energyOver s (fun i => f i - c) := by
  rw [constant_fit_error]
  have hnonneg : 0 ≤ (s.card : ℝ) * (meanOver s f - c) ^ 2 := by positivity
  linarith

/-- On a nonempty cell, equality in the least-squares fit forces the
fitted constant to be the mean. Source: the projection adaptation of
arXiv:2301.05217v1, section 5.1, at 4436290. -/
theorem mean_unique_minimizer (s : Finset I) (f : I → ℝ) (c : ℝ)
    (hs : s.Nonempty) : energyOver s (fun i => f i - c) = residualOver s f ↔
      c = meanOver s f := by
  have hpos : (0 : ℝ) < s.card := by exact_mod_cast Finset.card_pos.mpr hs
  rw [constant_fit_error]
  constructor
  · intro heq
    have hz : (meanOver s f - c) ^ 2 = 0 := by nlinarith [sq_nonneg (meanOver s f - c)]
    nlinarith [sq_nonneg (meanOver s f - c)]
  · intro heq
    rw [heq]
    ring

example : ({false, true} : Finset Bool).Nonempty := by
  exact ⟨false, by norm_num⟩

/-- Cell averaging is additive, including empty cells. Source: the
functional projection adaptation of arXiv:2301.05217v1, section 5.1,
at 4436290; this is linearity in logits, not in model parameters. -/
theorem meanOver_add (s : Finset I) (f g : I → ℝ) :
    meanOver s (fun i => f i + g i) = meanOver s f + meanOver s g := by
  unfold meanOver
  rw [Finset.sum_add_distrib, add_div]

/-- Cell averaging commutes with a common real scale. Source: the
functional projection adaptation of arXiv:2301.05217v1, section 5.1,
at 4436290; positive scale is needed only for decision invariance. -/
theorem meanOver_scale (s : Finset I) (f : I → ℝ) (a : ℝ) :
    meanOver s (fun i => a * f i) = a * meanOver s f := by
  unfold meanOver
  rw [← Finset.mul_sum]
  ring

/-- Repeating the actual cell projection has no further effect. Source:
the fixed-cell averaging adaptation of arXiv:2301.05217v1, section 5.1,
at 4436290. The empty case follows from its zero mean explicitly. -/
theorem meanOver_idempotent (s : Finset I) (f : I → ℝ) :
    meanOver s (fun _ => meanOver s f) = meanOver s f := by
  classical
  by_cases hs : s = ∅
  · subst s
    simp [meanOver]
  · exact meanOver_const s _ (Finset.nonempty_iff_ne_empty.mpr hs)

end Transformer.Grokking.Geometry
