import Transformer.Modes.Section3_PowDeriv
import Transformer.Modes.Section3_ExpMoments

/-
# The number of modes of a Gaussian KDE — the moment generating function of one
real random variable

`Section3_ExpMoments.lean` differentiates `∫ X^i Y^j e^{uX+vY}` under the
integral sign in two parameters.  The third moment of a normalized sum, which
is what the moment identity of arXiv:2412.09080v3, §3.1, needs, is read off a
*one*-parameter moment generating function: the two-dimensional statement is
recovered from it afterwards by taking directional combinations
`s x₁ + t x₂`.  This module is that one-parameter case,

  `scalarMoment P f i θ = ∫ f^i e^{θ f} dP`,

obtained from the general lemmas with the second variable set to `0`, together
with the third derivative at the origin, `∫ f³ dP`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`.
-/

open Real MeasureTheory Filter
open scoped Topology

namespace Transformer
namespace Modes

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {f : Ω → ℝ} {E : ℝ}

/-- `f` has exponential moments on `(-E, E)` under `P`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
def HasScalarExpMomentsOn (P : Measure Ω) (f : Ω → ℝ) (E : ℝ) : Prop :=
  ∀ θ : ℝ, |θ| < E → Integrable (fun ω => exp (θ * f ω)) P

/-- The moment `∫ f^i e^{θ f} dP`.  For `i = 0` this is the moment generating
function of `f`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
noncomputable def scalarMoment (P : Measure Ω) (f : Ω → ℝ) (i : ℕ) (θ : ℝ) : ℝ :=
  ∫ ω, f ω ^ i * exp (θ * f ω) ∂P

/-- At the origin the moment `∫ f^i e^{θ f} dP` is the plain moment `∫ f^i dP`. -/
theorem scalarMoment_zero (P : Measure Ω) (f : Ω → ℝ) (i : ℕ) :
    scalarMoment P f i 0 = ∫ ω, f ω ^ i ∂P := by
  simp [scalarMoment]

/-- The moment generating function of a probability measure is `1` at the
origin. -/
theorem scalarMoment_zero_zero (P : Measure Ω) [IsProbabilityMeasure P] (f : Ω → ℝ) :
    scalarMoment P f 0 0 = 1 := by
  simp [scalarMoment]

/-- **A derivative in `θ` raises the power of `f`**, by differentiating under
the integral sign.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem hasDerivAt_scalarMoment (hf : Measurable f) (hE : HasScalarExpMomentsOn P f E)
    {θ : ℝ} (hθ : |θ| < E) (i : ℕ) :
    HasDerivAt (scalarMoment P f i) (scalarMoment P f (i + 1) θ) θ := by
  have hE0 : (0 : ℝ) < E := (abs_nonneg θ).trans_lt hθ
  have h0 : |(0 : ℝ)| < E := by simpa using hE0
  have hE' : ∀ u v : ℝ, |u| < E → |v| < E →
      Integrable (fun ω => exp (u * f ω + v * (0 : ℝ))) P := by
    intro u _ hu _
    simpa using hE u hu
  have key := hasDerivAt_integral_pow_mul_exp (X := f) (Y := fun _ => (0 : ℝ))
    hf measurable_const hE' hθ h0 i 0
  simp only [pow_zero, mul_one, mul_zero, add_zero] at key
  exact key

/-- The hypotheses of `hasDerivAt_scalarMoment` are satisfiable: the Dirac mass
at a point, where every function is integrable. -/
example : Measurable (id : ℝ → ℝ) ∧ HasScalarExpMomentsOn (Measure.dirac (0 : ℝ)) id 1 ∧
    |(0 : ℝ)| < 1 :=
  ⟨measurable_id, fun _ _ => integrable_dirac enorm_lt_top, by norm_num⟩

/-- **The third derivative of the moment generating function at the origin is
the third moment.**

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem iteratedDeriv_three_scalarMoment (hf : Measurable f) (hE : HasScalarExpMomentsOn P f E)
    (hE0 : 0 < E) :
    iteratedDeriv 3 (scalarMoment P f 0) 0 = ∫ ω, f ω ^ 3 ∂P := by
  have hball : Metric.ball (0 : ℝ) E ∈ 𝓝 (0 : ℝ) := Metric.ball_mem_nhds 0 hE0
  have habs : ∀ t ∈ Metric.ball (0 : ℝ) E, |t| < E := by
    intro t ht; simpa [Real.dist_eq] using ht
  have hchain := iteratedDeriv_three_of_chain (F₁ := scalarMoment P f 1)
    (F₂ := scalarMoment P f 2) (F₃ := scalarMoment P f 3)
    (by filter_upwards [hball] with t ht using hasDerivAt_scalarMoment hf hE (habs t ht) 0)
    (by filter_upwards [hball] with t ht using hasDerivAt_scalarMoment hf hE (habs t ht) 1)
    (hasDerivAt_scalarMoment hf hE (by simpa using hE0) 2)
  rw [hchain, scalarMoment_zero]

/-- The hypotheses of `iteratedDeriv_three_scalarMoment` are satisfiable. -/
example : Measurable (id : ℝ → ℝ) ∧ HasScalarExpMomentsOn (Measure.dirac (0 : ℝ)) id 1 ∧
    (0 : ℝ) < 1 :=
  ⟨measurable_id, fun _ _ => integrable_dirac enorm_lt_top, one_pos⟩

end Modes
end Transformer
