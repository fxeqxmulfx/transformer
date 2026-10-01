/-
# Uniform bounds for the actual alignment EMA

Formalization of arXiv:2602.15322v1, Section 3, eq:masking_prob and
Algorithm 1. Cosine bounds and a positive fixed temperature give a
uniform positive scale when the previous EMA scale is initialized in
the same invariant interval. This supplies genuine operator bounds
for the corrected Section 5 analysis.
-/

import Transformer.Magma.Section3_Algorithm
import Transformer.Magma.Section5_AlignmentBound

open scoped InnerProductSpace

noncomputable section

namespace Transformer.Magma

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Cauchy--Schwarz bounds the cosine, including the declared zero-vector
extension. Source: arXiv:2602.15322v1, Section 3, eq:masking_prob. -/
theorem cosine_abs_le_one (momentum g : E) : |cosine momentum g| ≤ 1 := by
  by_cases hzero : ‖momentum‖ * ‖g‖ = 0
  · simp [cosine, hzero]
  · have hpos : 0 < ‖momentum‖ * ‖g‖ :=
      lt_of_le_of_ne (mul_nonneg (norm_nonneg _) (norm_nonneg _)) (Ne.symm hzero)
    rw [cosine, abs_div, abs_of_pos hpos]
    apply (div_le_iff₀ hpos).mpr
    simpa using abs_real_inner_le_norm momentum g

/-- The printed EMA preserves a uniform positive lower scale. Unlike an
alignment event involving the unknown true gradient, this bound actually
applies to momentum-gradient cosine and its EMA. Source:
arXiv:2602.15322v1, Section 3 and Appendix A.3, corrected damping bound. -/
theorem damping_uniform_bounds (τ previous : ℝ) (hτ : 0 < τ) (momentum g : E)
    (hprev : previous ∈ Set.Icc (Real.sigmoid (-1 / τ)) 1) :
    damping τ previous momentum g ∈ Set.Icc (Real.sigmoid (-1 / τ)) 1 := by
  have hc := (abs_le.mp (cosine_abs_le_one momentum g)).1
  have hs := Real.sigmoid_le (div_le_div_of_nonneg_right hc hτ.le)
  have hu := Real.sigmoid_le_one (cosine momentum g / τ)
  constructor <;> unfold damping <;> nlinarith [hprev.1, hprev.2]

/-- Positive temperature and the EMA invariant have a joint witness
with previous scale 1/2. Source: arXiv:2602.15322v1, Section 3. -/
example : (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Set.Icc (Real.sigmoid (-1 / 1)) 1 := by
  refine ⟨by norm_num, ?_, by norm_num⟩
  have h := Real.sigmoid_le (by norm_num : (-1 : ℝ) ≤ 0)
  simpa using h

/-- A scalar block operator using the actual Magma EMA satisfies the
spectral lower bound and norm upper bound of the corrected analysis.
Source: arXiv:2602.15322v1, Sections 3 and 5, block-diagonal S_t. -/
theorem magma_scalar_operator_bounds (τ previous : ℝ) (hτ : 0 < τ) (momentum g : E)
    (hprev : previous ∈ Set.Icc (Real.sigmoid (-1 / τ)) 1) :
    DampingBounds (Real.sigmoid (-1 / τ)) 1
      (damping τ previous momentum g • ContinuousLinearMap.id ℝ E) := by
  have hb := damping_uniform_bounds τ previous hτ momentum g hprev
  have hpos := Real.sigmoid_pos (-1 / τ)
  intro u
  have hd : 0 ≤ damping τ previous momentum g := le_trans hpos.le hb.1
  constructor
  · simp only [FunLike.coe_smul, Pi.smul_apply, ContinuousLinearMap.id_apply,
      real_inner_smul_left, real_inner_self_eq_norm_sq]
    exact mul_le_mul_of_nonneg_right hb.1 (sq_nonneg ‖u‖)
  · simp only [FunLike.coe_smul, Pi.smul_apply, ContinuousLinearMap.id_apply,
      norm_smul, Real.norm_eq_abs, abs_of_nonneg hd, one_mul]
    simpa using mul_le_mul_of_nonneg_right hb.2 (norm_nonneg u)

/-- The operator theorem's temperature and scale conditions are
simultaneously satisfiable. Source: arXiv:2602.15322v1, Section 3. -/
example : (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Set.Icc (Real.sigmoid (-1 / 1)) 1 := by
  refine ⟨by norm_num, ?_, by norm_num⟩
  simpa using Real.sigmoid_le (by norm_num : (-1 : ℝ) ≤ 0)

end Transformer.Magma
