/-
# Homogenized Transformers — the low-temperature setting

Formalization of `ass:low-temperature` of arXiv:2604.01978v1, *Homogenized
Transformers*, §3: the perturbative regime around the uniform distribution in
which the slow-motion theorem `thm:large_beta_meta` is proved.

The uniform measure `σ_d` the assumption is written against is
`UniformLaw.IsUniformAmbient`.
-/

import Transformer.Homogenized.UniformLaw
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.MeasureTheory.Measure.WithDensity

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- **Assumption (ass:low-temperature).**  For every `t ≥ 0` the law `μ(t)` is
absolutely continuous with respect to the uniform measure `σ_d`, and its
density `ρ(t,·) = dμ(t)/dσ_d` is `C¹` with

  `0 < ρ_min ≤ ρ(t,·) ≤ ρ_max`,  `‖∇ρ(t,·)‖_{L^∞(𝕊^{d-1})} ≤ L`.

**What the source says and how it is read here.**

* Absolute continuity together with the density is one condition, `density`:
  naming `ρ` and asking `μ(t) = ρ(t,·) σ_d` says both at once.
* `ρ(t,·) ∈ C¹(𝕊^{d-1})` is carried as `C¹` on the ambient `ℝ^d`, which is how
  the proof of `lemma:Laplace_method` uses it — it expands
  `ρ(Exp_u(v)) = ρ(u) + O(‖v‖)` against an ambient modulus.
* `∇ρ` is the Riemannian gradient, `proj_y ∇ρ̃(y)` by `lem:toolkit_geo_riem`,
  and the `L^∞(𝕊^{d-1})` norm is a bound at every unit vector.  The bounds on
  `ρ` itself are likewise imposed only on the sphere, where they are the
  source's, and are silent off it, where the density is irrelevant to `μ(t)`.
* The source adds that `ρ_min` and `ρ_max` are "independent of `d` and `β`".
  That is not a condition on one instance of this assumption but a uniformity
  in how `thm:large_beta_meta` quantifies; it is recorded there, by placing the
  constants before `d` and `β`.

Source: arXiv:2604.01978v1, `ass:low-temperature`. -/
structure IsLowTemperature (d : ℕ) (σ : Measure (EucSpace d))
    (μ : ℝ → Measure (EucSpace d)) (ρ : ℝ → EucSpace d → ℝ) (ρmin ρmax L : ℝ) : Prop where
  /-- `σ` is the uniform law on `𝕊^{d-1}`. -/
  unif : IsUniformAmbient d σ
  /-- The lower bound on the density is positive. -/
  pos : 0 < ρmin
  /-- `μ(t) ≪ σ_d` with density `ρ(t,·)`. -/
  density : ∀ t, 0 ≤ t → μ t = σ.withDensity fun y => ENNReal.ofReal (ρ t y)
  /-- `ρ(t,·) ∈ C¹`. -/
  smooth : ∀ t, 0 ≤ t → ContDiff ℝ 1 (ρ t)
  /-- `ρ_min ≤ ρ(t,·)` on the sphere. -/
  lower : ∀ t, 0 ≤ t → ∀ y : EucSpace d, ‖y‖ = 1 → ρmin ≤ ρ t y
  /-- `ρ(t,·) ≤ ρ_max` on the sphere. -/
  upper : ∀ t, 0 ≤ t → ∀ y : EucSpace d, ‖y‖ = 1 → ρ t y ≤ ρmax
  /-- `‖∇ρ(t,·)‖ ≤ L` on the sphere, for the Riemannian gradient. -/
  grad : ∀ t, 0 ≤ t → ∀ y : EucSpace d, ‖y‖ = 1 → ‖proj d y (gradient (ρ t) y)‖ ≤ L

/-- **`IsLowTemperature` is satisfiable**, in every dimension `d ≥ 1` and at a
stationary law: the uniform measure itself, whose density is `ρ ≡ 1`, with
`ρ_min = ρ_max = 1` and `L = 0`. -/
theorem isLowTemperature_uniformAmbient {d : ℕ} (hd : 0 < d) :
    IsLowTemperature d (uniformAmbient d) (fun _ => uniformAmbient d) (fun _ _ => 1) 1 1 0 := by
  refine ⟨isUniformAmbient_uniformAmbient hd, one_pos, ?_, ?_, ?_, ?_, ?_⟩
  · intro t _
    simp
  · intro _ _
    exact contDiff_const
  · intro _ _ _ _
    exact le_refl _
  · intro _ _ _ _
    exact le_refl _
  · intro _ _ y _
    simp [proj]

end Homogenized
end Transformer
