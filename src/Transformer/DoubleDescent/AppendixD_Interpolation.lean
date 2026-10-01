import Transformer.DoubleDescent.AppendixD_RandomFeatures
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.FiniteDimensional.Basic

/-!
# Exact interpolation in a fixed feature model

arXiv:1912.02292v1, Section 5, discussion, and Appendix D. Exact fitting
of *every* label vector requires at least as many features as examples.
Uniqueness at a square interpolation threshold additionally needs full
rank. These are exact linear-algebra statements, distinct from Definition 1's
distribution-dependent expected approximate interpolation.
-/

namespace Transformer.DoubleDescent

/-- Section 5 and Appendix D: evaluating the trained linear output layer
on the actual feature vectors gives a linear map from weights to labels. -/
noncomputable def featureDesign {X : Type*} {d n : ℕ}
    (features : X → Fin d → ℝ) (sample : Fin n → X) :
    (Fin d → ℝ) →ₗ[ℝ] (Fin n → ℝ) where
  toFun weight := fun i => featurePrediction features weight (sample i)
  map_add' w v := by
    funext i
    simp [featurePrediction, add_mul, Finset.sum_add_distrib]
  map_smul' c w := by
    funext i
    simp [featurePrediction, Finset.mul_sum, mul_assoc]

/-- Appendix D: exact interpolation of arbitrary labels requires `n ≤ d`.
This concerns surjectivity of a fixed linear design, not the paper's EMC. -/
theorem exact_interpolation_requires_dimension {d n : ℕ}
    (design : (Fin d → ℝ) →ₗ[ℝ] (Fin n → ℝ))
    (hfit : Function.Surjective design) : n ≤ d := by
  have h := LinearMap.finrank_le_finrank_of_surjective hfit
  simpa [Module.finrank_pi] using h

/-- Appendix D: the exact-fitting hypothesis is realized by the identity design. -/
example : Function.Surjective (LinearMap.id : (Fin 1 → ℝ) →ₗ[ℝ] (Fin 1 → ℝ)) :=
  fun y => ⟨y, rfl⟩

/-- Section 5, discussion: a square, full-rank design has a unique
interpolator for every label vector. The source's informal "only one model"
at threshold needs this full-rank premise in an exact linear formulation. -/
theorem square_full_rank_interpolation_unique {d : ℕ}
    (design : (Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ))
    (hfit : Function.Surjective design) (labels : Fin d → ℝ) :
    ∃! weight, design weight = labels := by
  obtain ⟨weight, hw⟩ := hfit labels
  refine ⟨weight, hw, ?_⟩
  intro other ho
  exact (LinearMap.injective_iff_surjective.mpr hfit) (ho.trans hw.symm)

/-- Section 5: a square full-rank design exists. -/
example : Function.Surjective (LinearMap.id : (Fin 1 → ℝ) →ₗ[ℝ] (Fin 1 → ℝ)) :=
  fun y => ⟨y, rfl⟩

/-- Section 5, discussion: a nonzero kernel direction produces a genuinely
different interpolator with the same labels. More parameters alone do not
guarantee good generalization of either interpolator. -/
theorem kernel_direction_gives_another_interpolator {d n : ℕ}
    (design : (Fin d → ℝ) →ₗ[ℝ] (Fin n → ℝ))
    (weight direction : Fin d → ℝ) (labels : Fin n → ℝ)
    (hfit : design weight = labels) (hker : design direction = 0) (hne : direction ≠ 0) :
    ∃ other, other ≠ weight ∧ design other = labels := by
  refine ⟨weight + direction, ?_, ?_⟩
  · intro he
    apply hne
    have hc := congrArg (fun v => -weight + v) he
    simpa [← add_assoc] using hc
  · simp [design.map_add, hfit, hker]

/-- Section 5: exact fitting and a nonzero kernel direction are simultaneously
satisfiable, e.g. for zero labels and the zero linear design. -/
example : ∃ (design : (Fin 1 → ℝ) →ₗ[ℝ] (Fin 1 → ℝ))
    (weight direction : Fin 1 → ℝ) (labels : Fin 1 → ℝ),
    design weight = labels ∧ design direction = 0 ∧ direction ≠ 0 := by
  refine ⟨0, 0, fun _ => 1, 0, by simp, by simp, ?_⟩
  intro he
  have hc := congrFun he (0 : Fin 1)
  norm_num at hc

end Transformer.DoubleDescent
