/-
# Valid algebraic properties of the proof's efficiency coefficients

Formalization of arXiv:2602.15322v1, Appendix A.3, intem4. The interval
bounds and square-root inequality are true as algebraic statements.
They do not establish the false momentum-alignment lower-bound lemma.
-/

import Transformer.Magma.Section3_DampingBounds
import Mathlib.Analysis.Real.Sqrt

noncomputable section

namespace Transformer.Magma

/-- The proof-defined alpha coefficient, retaining its linear use in
intem4. The lemma statement instead squares alpha through a norm.
Source: arXiv:2602.15322v1, Appendix A.3. -/
def efficiencyAlpha (τ γ e : ℝ) : ℝ :=
  Real.sigmoid (γ / τ) - (Real.sigmoid (γ / τ) - Real.sigmoid (-1 / τ)) *
    γ * Real.sqrt (1 - e)

/-- The proof-defined noise-descent coupling coefficient.
Source: arXiv:2602.15322v1, Appendix A.3, intem4. -/
def noiseCouplingCoefficient (τ γ e : ℝ) : ℝ :=
  (Real.sigmoid (γ / τ) - Real.sigmoid (-1 / τ)) * γ / 2 * Real.sqrt (1 - e)

/-- The source coefficient intervals hold for an admissible threshold,
event probability, and temperature. This proves only their numerical
ranges, not the claimed application to the momentum-based damping.
Source: arXiv:2602.15322v1, Appendix A.3, paragraph after intem4. -/
theorem efficiency_coefficient_ranges (τ γ e : ℝ) (hτ : 0 < τ)
    (hγ : γ ∈ Set.Ioc (0 : ℝ) 1) (he : e ∈ Set.Icc (0 : ℝ) 1) :
    efficiencyAlpha τ γ e ∈ Set.Icc (Real.sigmoid (-1 / τ)) (Real.sigmoid (1 / τ)) ∧
      noiseCouplingCoefficient τ γ e ∈ Set.Icc (0 : ℝ) (Real.sigmoid (1 / τ) / 2) := by
  have hlo : Real.sigmoid (-1 / τ) ≤ Real.sigmoid (γ / τ) :=
    Real.sigmoid_le (div_le_div_of_nonneg_right (by linarith [hγ.1]) hτ.le)
  have hhi := Real.sigmoid_le (div_le_div_of_nonneg_right hγ.2 hτ.le)
  have hsm := Real.sigmoid_nonneg (-1 / τ)
  have hsqrt := Real.sqrt_nonneg (1 - e)
  have hsqrt1 : Real.sqrt (1 - e) ≤ 1 := Real.sqrt_le_one.mpr (by linarith [he.1])
  have hfac : 0 ≤ γ * Real.sqrt (1 - e) := mul_nonneg hγ.1.le hsqrt
  have hfac1 : γ * Real.sqrt (1 - e) ≤ 1 := by nlinarith [hγ.1, hγ.2]
  have hprod := mul_le_mul_of_nonneg_left hfac1 (sub_nonneg.mpr hlo)
  have hprod0 := mul_nonneg (sub_nonneg.mpr hlo) hfac
  unfold efficiencyAlpha noiseCouplingCoefficient
  constructor <;> constructor <;> nlinarith

/-- The coefficient hypotheses are jointly satisfiable at tau=1,
gamma=1 and event probability 1. Source: arXiv:2602.15322v1, Appendix A.3. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ∈ Set.Ioc (0 : ℝ) 1 ∧
    (1 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := by norm_num

/-- The square-root inequality used in the source proof is valid.
Source: arXiv:2602.15322v1, Appendix A.3, paragraph before intem4. -/
theorem sqrt_noise_bound (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    Real.sqrt (a * (a + b)) ≤ a + b / 2 := by
  apply Real.sqrt_le_iff.mpr
  constructor
  · positivity
  · nlinarith [sq_nonneg b]

/-- A nonzero signal and nonzero noise meet the square-root hypotheses.
Source: arXiv:2602.15322v1, Appendix A.3. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 := by norm_num

end Transformer.Magma
