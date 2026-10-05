import Transformer.GPTMini.Sparsemax.OuterSensitivity
import Mathlib.LinearAlgebra.Span.Basic
import Mathlib.Algebra.Module.Submodule.Ker

/-!
# A structural restriction on active values

Derived from arXiv:1602.02068v2, §2.5, `sparsemax_gradient`, and the
linear value readout in `Attention.forward` at commit `73f8a0b`.
If the differences of active values span the entire output space, no
nonzero output derivative can annihilate every active score direction.
The condition concerns the values and support, not an attention target
or a derivative chosen for a particular training label.

This is a conditional guarantee for the actual sparsemax projection.
It preserves inactive zeros. It neither supplies an architecture that
enforces the span for every input nor guarantees that a query/key
parameter map can realize every raw score direction. In a scalar output
space, two distinct active values already suffice for the span condition.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Differences of values on the actual positive support of one row.
Source: arXiv:1602.02068v2, §2.5, active coordinates of the Jacobian,
composed with `Attention.forward` at `73f8a0b`. -/
def activeValueDifferences {T : ℕ} (scores : Fin T → ℝ) (i : Fin T)
    (values : Fin T → E) : Set E :=
  {v | ∃ j k, 0 < sparseWeights scores i j ∧ 0 < sparseWeights scores i k ∧
    v = values j - values k}

/-- Two distinct active scalar values span the scalar output space.
Source context: the derived value restriction for §2.5 of
arXiv:1602.02068v2; this is independent of the output target. -/
theorem activeScalarValues_span_of_distinct {T : ℕ} (scores : Fin T → ℝ)
    (i j k : Fin T) (values : Fin T → ℝ)
    (hj : 0 < sparseWeights scores i j) (hk : 0 < sparseWeights scores i k)
    (hv : values j ≠ values k) :
    Submodule.span ℝ (activeValueDifferences scores i values) = ⊤ := by
  apply Submodule.eq_top_iff'.mpr
  intro x
  have hd : values j - values k ≠ 0 := sub_ne_zero.mpr hv
  have hmem : values j - values k ∈
      Submodule.span ℝ (activeValueDifferences scores i values) :=
    Submodule.subset_span ⟨j, k, hj, hk, rfl⟩
  have h := (Submodule.span ℝ (activeValueDifferences scores i values)).smul_mem
    (x / (values j - values k)) hmem
  simpa only [smul_eq_mul, div_mul_cancel₀ _ hd] using h

/-- Distinct active scalars coexist with an actual inactive visible slot.
Source context: §2.2's bounded sparse example and the value restriction. -/
example : Submodule.span ℝ (activeValueDifferences twoActiveScores 2 (basis 0)) = ⊤ :=
  activeScalarValues_span_of_distinct _ 2 0 1 _
    (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])
    (by norm_num [basis])

/-- A full active-value span separates every nonzero output derivative
on some active pair. Source: arXiv:1602.02068v2, §2.5, with the frozen
linear value sum at `73f8a0b`; proved by the kernel of a linear map. -/
theorem activeValueSpan_separates_gradient {T : ℕ} (scores : Fin T → ℝ)
    (i : Fin T) (values : Fin T → E) (gradient : E →L[ℝ] ℝ)
    (hspan : Submodule.span ℝ (activeValueDifferences scores i values) = ⊤)
    (hg : gradient ≠ 0) :
    ∃ j k, j ≠ k ∧ 0 < sparseWeights scores i j ∧ 0 < sparseWeights scores i k ∧
      gradient (values j - values k) ≠ 0 := by
  classical
  by_contra hn
  have hkill (j k : Fin T) (hj : 0 < sparseWeights scores i j)
      (hk : 0 < sparseWeights scores i k) : gradient (values j - values k) = 0 := by
    by_cases he : j = k
    · subst k
      simp
    · by_contra hs
      exact hn ⟨j, k, he, hj, hk, hs⟩
  have hle : Submodule.span ℝ (activeValueDifferences scores i values) ≤
      LinearMap.ker gradient.toLinearMap := by
    apply Submodule.span_le.mpr
    rintro v ⟨j, k, hj, hk, rfl⟩
    exact LinearMap.mem_ker.mpr (hkill j k hj hk)
  apply hg
  apply ContinuousLinearMap.ext
  intro v
  have hv : v ∈ Submodule.span ℝ (activeValueDifferences scores i values) := by
    rw [hspan]
    exact Submodule.mem_top
  exact LinearMap.mem_ker.mp (hle hv)

/-- A sparse scalar row inhabits the full-span and nonzero-gradient
premises. Source context: the derived active-value restriction for §2.5. -/
example : ∃ j k : Fin 3, j ≠ k ∧ 0 < sparseWeights twoActiveScores 2 j ∧
    0 < sparseWeights twoActiveScores 2 k ∧
    (ContinuousLinearMap.id ℝ ℝ) (basis 0 j - basis 0 k) ≠ 0 := by
  apply activeValueSpan_separates_gradient twoActiveScores 2 (basis 0)
    (ContinuousLinearMap.id ℝ ℝ)
  · exact activeScalarValues_span_of_distinct _ 2 0 1 _
      (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])
      (by norm_num [basis])
  · intro h
    have he := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) h
    norm_num at he

/-- A full active-value span transmits every nonzero ordinary task
derivative into a nonzero raw score direction. Source: §2.5 of
arXiv:1602.02068v2 and `Attention.forward` at `73f8a0b`.
No prescribed attention route or dense companion is present. -/
theorem taskLoss_no_zero_score_derivative_of_active_value_span {T : ℕ}
    (scores : Fin T → ℝ) (i : Fin T) (values : Fin T → E)
    (loss : E → ℝ) (gradient : E →L[ℝ] ℝ)
    (hspan : Submodule.span ℝ (activeValueDifferences scores i values) = ⊤)
    (hg : gradient ≠ 0)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient
      (frozenValueReadout values (sparseWeights scores i))) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun z => loss (frozenValueReadout values (sparseWeights z i))) 0 scores := by
  obtain ⟨j, k, hne, hj, hk, hsep⟩ :=
    activeValueSpan_separates_gradient scores i values gradient hspan hg
  exact value_sensitivity_not_hasFDerivAt_zero scores i j k
    (fun row => loss (frozenValueReadout values row)) gradient values hne hj hk
    (valueReadoutLoss_hasFDerivAt values loss gradient _ hl) hsep

/-- The complete task-loss hypotheses hold on a sparse scalar readout.
Source context: §2.5's derived full-span condition, with a linear task. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun z : Fin 3 → ℝ => (ContinuousLinearMap.id ℝ ℝ)
      (frozenValueReadout (basis 0) (sparseWeights z 2))) 0 twoActiveScores := by
  apply taskLoss_no_zero_score_derivative_of_active_value_span twoActiveScores 2
    (basis 0) (ContinuousLinearMap.id ℝ ℝ) (ContinuousLinearMap.id ℝ ℝ)
  · exact activeScalarValues_span_of_distinct _ 2 0 1 _
      (by norm_num [twoActiveScores_projection]) (by norm_num [twoActiveScores_projection])
      (by norm_num [basis])
  · intro h
    have he := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) h
    norm_num at he
  · exact (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

/-- The previous plateau's active values fail the structural span
condition. Source: the frozen `(0, 0, 1)` readout at `73f8a0b`, with
the actual bounded sparse row from arXiv:1602.02068v2, §2.2. -/
theorem collapsed_active_values_not_spanning :
    Submodule.span ℝ (activeValueDifferences twoActiveScores 2 (basis 2)) ≠ ⊤ := by
  intro hspan
  have hg : (ContinuousLinearMap.id ℝ ℝ) ≠ 0 := by
    intro h
    have he := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) h
    norm_num at he
  obtain ⟨j, k, _, hj, hk, hsep⟩ :=
    activeValueSpan_separates_gradient twoActiveScores 2 (basis 2)
      (ContinuousLinearMap.id ℝ ℝ) hspan hg
  have hj2 : j ≠ 2 := by
    intro he
    subst j
    norm_num [twoActiveScores_projection] at hj
  have hk2 : k ≠ 2 := by
    intro he
    subst k
    norm_num [twoActiveScores_projection] at hk
  norm_num [basis, hj2, hk2] at hsep

end Transformer.GPTMini.Sparsemax
