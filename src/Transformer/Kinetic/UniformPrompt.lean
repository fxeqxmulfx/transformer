/-
# Kinetic theory for Transformers — uniform prompts

The most homogeneous baseline of arXiv:2605.09213v1, §1.3: the initial tokens of a prompt are
independent and uniformly distributed on `𝕋`.  Then `eq:init-conv` holds with `f_∘ ≡ 1`, the
mean-field solution stays `f ≡ 1`, and the mean term `E[e^{inθ_N(t)}] E[e^{-inθ_{i_*}(0)}]` of the
retrieval score vanishes for `n ≠ 0`.  This module states the uniformity: the one property of the
law of a single token that the theorems on the retrieval score need beyond independence.
-/

import Transformer.Kinetic.MeanField
import Mathlib.Analysis.Fourier.AddCircle

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **The initial tokens are uniformly distributed on `𝕋`.**  Every `θ_j(0)`, read modulo `2π`,
has the normalized Haar measure `dθ/(2π)` of the torus as its law.

Together with independence this is the source's "the initial tokens may even be taken iid and
uniformly distributed on `𝕋`, so that `eq:init-conv` holds with `f_∘ ≡ 1`".  It is *exact*
uniformity, not `eq:init-conv` with `f_∘ ≡ 1`, which only asks the characteristic functions to be
within `C N^{-δ} ⟨n⟩^γ` of those of the uniform law.

Source: arXiv:2605.09213v1, §1.3, after `eq:conv-rate-muNf`. -/
def IsUniformPrompt (P : Measure Ω) (N : ℕ) (ϑ : Ω → Idx N → ℝ) : Prop :=
  ∀ j : Idx N, P.map (fun ω => ((ϑ ω j : ℝ) : Torus)) = AddCircle.haarAddCircle

/-- The uniform prompt exists: the identity of `ℝ` on one period, with the uniform law, is a single
token uniformly distributed on `𝕋`. -/
theorem exists_isUniformPrompt :
    ∃ P : Measure ℝ, IsProbabilityMeasure P ∧ IsUniformPrompt P 1 (fun ω _ => ω) := by
  have hmp := AddCircle.measurePreserving_mk (T := 2 * π) 0
  have hpos : ENNReal.ofReal (2 * π) ≠ 0 := by simp [Real.pi_pos]
  have hne : ENNReal.ofReal (2 * π) ≠ ⊤ := ENNReal.ofReal_ne_top
  refine ⟨(ENNReal.ofReal (2 * π))⁻¹ • volume.restrict (Set.Ioc 0 (0 + 2 * π)), ⟨?_⟩,
    fun j => ?_⟩
  · rw [Measure.smul_apply, Measure.restrict_apply_univ, Real.volume_Ioc, smul_eq_mul, zero_add,
      sub_zero, ENNReal.inv_mul_cancel hpos hne]
  · show Measure.map (fun ω : ℝ => (ω : Torus))
        ((ENNReal.ofReal (2 * π))⁻¹ • volume.restrict (Set.Ioc 0 (0 + 2 * π))) = _
    rw [Measure.map_smul _ hmp.measurable.aemeasurable, hmp.map_eq,
      AddCircle.volume_eq_smul_haarAddCircle, smul_smul, ENNReal.inv_mul_cancel hpos hne, one_smul]

end Kinetic
end Transformer
