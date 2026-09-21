/-
# Mean-Field Dynamics — the global rate of clustering (§4 of 2512.01868v4)

* `Theorem thm: mfclust` — mean-field exponential rate for small `β`
                           (Chen–Lin–Polyanskiy 2025).

The statement is made against the objects the source names: the continuity
equation `eq: continuity`, whose velocity field is the unnormalized
`Proj_x ∫ e^{β⟨x,y⟩} y dμ(y)` (`Perspective.usaContinuityEquation`), an
initial measure with a density `f₀ ∈ L²` against the uniform law
(`uniformLaw`, which is `Causal.uniformSphere`), and the Wasserstein distance `Wasserstein.W2`.

Its hypotheses are satisfiable, and that is proved:
`MeanField.exists_density_mean_pos` exhibits such a density with `R₀ > 0`.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_GradientFlow
import Transformer.MeanField.UniformLaw
import Transformer.Wasserstein.Basic

open Real MeasureTheory

namespace Transformer
namespace MeanField

variable (d : ℕ)

/-- **Theorem (thm: mfclust).** *Mean-field exponential rate (small `β`).*

Let `d ≥ 2` and let `μ_t` evolve according to `eq: continuity` from an initial
measure `μ_0` with density `f_0 ∈ L²(𝕊^{d-1})` satisfying

  `R_0 := ‖∫_{𝕊^{d-1}} x dμ_0(x)‖² > 0`.

Then there exist `β_0, C_0, T_0 > 0`, depending on `μ_0`, such that for
`|β| < β_0` there is `x_∞ ∈ 𝕊^{d-1}` with
`W_2(μ_t, δ_{x_∞}) ≤ C_0 e^{-t/100}` for `t ≥ T_0`.

The density is taken against the uniform law `uniformLaw d`. It is
written as `μ_0 = f_0 · σ` with `f_0 ≥ 0`, the pointwise form of "admits the
density `f_0`".  Without the density the claim is false: `⅔δ_e + ⅓δ_{-e}` has
`R_0 = 1/9 > 0` and is stationary for every `β`.  The velocity field is the
one `eq:continuity` prints, `Proj_x ∫ e^{β⟨x,y⟩} y dμ(y)`, without the
partition function: `Perspective.usaContinuityEquation`, not
`Perspective.continuityEquation`.

Not proved here.

Source: arXiv:2512.01868v4, §4, `thm:mfclust` (Chen–Lin–Polyanskiy 2025);
`eq:continuity`. -/
theorem meanField_exponential_rate (hd : 2 ≤ d) (μ₀ : Perspective.ProbSphere d)
    (f₀ : SSphere d → ℝ) (hf₀ : ∀ x, 0 ≤ f₀ x)
    (hL2 : MemLp f₀ 2 (uniformLaw d))
    (hμ₀ : (μ₀ : Measure (SSphere d)) =
      (uniformLaw d).withDensity (fun x => ENNReal.ofReal (f₀ x)))
    (hR : 0 < ‖∫ x, (x : EucSpace d) ∂(μ₀ : Measure (SSphere d))‖ ^ 2) :
    ∃ β₀ C₀ T₀ : ℝ, 0 < β₀ ∧ 0 < C₀ ∧ 0 < T₀ ∧
      ∀ β : ℝ, |β| < β₀ →
        ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ →
          Perspective.usaContinuityEquation d β μ →
            ∃ x_inf : SSphere d, ∀ t : ℝ, T₀ ≤ t →
              Wasserstein.W2 (μ t : Measure (SSphere d)) (Measure.dirac x_inf)
                ≤ C₀ * Real.exp (-t / 100) := by
  sorry

/-- The hypotheses of `meanField_exponential_rate` are satisfiable: on the circle,
`exists_density_mean_pos` supplies `μ₀` and its `L²` density. -/
example : ∃ (μ₀ : Perspective.ProbSphere 2) (f₀ : SSphere 2 → ℝ),
    2 ≤ 2 ∧ (∀ x, 0 ≤ f₀ x) ∧ MemLp f₀ 2 (uniformLaw 2) ∧
    (μ₀ : Measure (SSphere 2)) =
      (uniformLaw 2).withDensity (fun x => ENNReal.ofReal (f₀ x)) ∧
    0 < ‖∫ x, (x : EucSpace 2) ∂(μ₀ : Measure (SSphere 2))‖ ^ 2 := by
  have := isProbabilityMeasure_uniformLaw 2 (by norm_num)
  obtain ⟨μ₀, f₀, h⟩ := exists_density_mean_pos 2 (uniformLaw 2)
  exact ⟨μ₀, f₀, le_rfl, h⟩

end MeanField
end Transformer
