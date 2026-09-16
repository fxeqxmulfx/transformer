/-
# Measure-to-measure interpolation — Mass concentration estimates

Formalization of the two technical claims of arXiv:2411.04551v3 that turn a
Wasserstein bound into a statement about mass:

* `Lemma lem: mass.concentration.Q1` — a measure that puts `1 - η` of its mass
  in the positive quadrant is `O(η)`-close to a measure carried by it,
* `Claim cl: W.to.ball`             — a measure `W₂`-close to a Dirac mass puts
  almost all of its mass in a small ball.

Mathlib has no Wasserstein distance, so `W₂` is a parameter, as in
`Interpolation/Main.lean` and `Interpolation/Clustering.lean`; here it is a
family indexed by the dimension, so that the survey's *universal* constant `C`
can be quantified before the dimension and not after it.

Both claims are therefore properties *of* `W₂` and are `Prop`-valued
definitions, the way `Interpolation.Compression` and `Interpolation.Monge`
already are.  As theorems about an arbitrary `W₂` they would be false, not
merely unproved: `not_forall_wToBall` takes the constant family `W₂ ≡ 0` and
refutes `cl: W.to.ball` at a Dirac mass sitting on the antipode.
-/

import Transformer.Basic
import Transformer.Interpolation.Basic

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

/-- A Wasserstein distance on each sphere at once: the parameter the two
statements of this file share. -/
abbrev W2Family : Type :=
  ∀ d : ℕ, Measure (SSphere d) → Measure (SSphere d) → ℝ

/-- **Lemma (lem: mass.concentration.Q1).**

Let `ν_0^i ∈ 𝒫(𝕊^{d-1})` and `η > 0`, and let `φ_η : 𝕊^{d-1} → 𝕊^{d-1}` be
Lipschitz and invertible with

  `(φ_η)_# ν_0^i (ℚ_1^{d-1}) = 1 - η`   for every `i`.

Then for every `i` there is a `μ_0^i ∈ 𝒫(ℚ_1^{d-1})` with

  `W_2((φ_η)_# ν_0^i, μ_0^i) ≤ C η`,

`C > 0` universal.  "`μ_0^i ∈ 𝒫(ℚ_1^{d-1})`" is read as a probability measure
on the sphere giving no mass to the complement of the quadrant, since
`ℚ_1^{d-1}` is not carried as a type here.

A `Prop`-valued definition and not a theorem: it says nothing until `W₂` is
the Wasserstein distance, which is not available here.

Source: arXiv:2411.04551v3, Appendix "On condition (eq: assumption.hole)",
`lem: mass.concentration.Q1`. -/
def MassConcentrationQ1 (W₂ : W2Family) : Prop :=
  ∃ C : ℝ, 0 < C ∧
    ∀ (d N : ℕ) (ν : Idx N → Perspective.ProbSphere d) (η : ℝ), 0 < η →
      ∀ (φ : SSphere d → SSphere d) (L : NNReal), LipschitzWith L φ →
        Function.Bijective φ →
        (∀ i : Idx N,
          Measure.map φ (ν i : Measure (SSphere d)) (positiveQuadrant d)
            = ENNReal.ofReal (1 - η)) →
        ∀ i : Idx N, ∃ μ : Perspective.ProbSphere d,
          (μ : Measure (SSphere d)) (positiveQuadrant d)ᶜ = 0 ∧
          W₂ d (Measure.map φ (ν i : Measure (SSphere d)))
              (μ : Measure (SSphere d)) ≤ C * η

/-- **Claim (cl: W.to.ball).**

If `μ ∈ 𝒫(𝕊^{d-1})` and `x_0 ∈ 𝕊^{d-1}` satisfy `W_2(μ, δ_{x_0}) ≤ η_2`, then

  `1 - μ(B(x_0, η_3)) ≤ C η_2 / η_3`   for every `η_3 > 0`,

with `C > 0` universal.  The paper's proof goes through Kantorovich–Rubinstein
duality for `W_1`; none of it is formalized.

A `Prop`-valued definition and not a theorem, for the reason
`not_forall_wToBall` records: the claim is about the Wasserstein distance and
fails outright for other families `W₂`.

Source: arXiv:2411.04551v3, Appendix, `cl: W.to.ball`. -/
def WToBall (W₂ : W2Family) : Prop :=
  ∃ C : ℝ, 0 < C ∧
    ∀ (d : ℕ) (μ : Perspective.ProbSphere d) (x₀ : SSphere d) (η₂ : ℝ),
      W₂ d (μ : Measure (SSphere d)) (Measure.dirac x₀) ≤ η₂ →
      ∀ η₃ : ℝ, 0 < η₃ →
        1 - ((μ : Measure (SSphere d)) (Metric.ball x₀ η₃)).toReal
          ≤ C * η₂ / η₃

/-- The antipode of `basePoint 0` on `𝕊^0 ⊂ ℝ^1`: the second of the two points
of the one-dimensional sphere, at distance `2` from the first. -/
noncomputable def antipode : SSphere 1 :=
  ⟨-((basePoint 0 : SSphere 1) : EucSpace 1), by
    rw [mem_sphere_zero_iff_norm, norm_neg]
    exact mem_sphere_zero_iff_norm.mp (basePoint 0).2⟩

/-- The two points of `𝕊^0` are at distance `2`, so neither lies in the unit
ball around the other. -/
theorem dist_antipode : dist antipode (basePoint 0 : SSphere 1) = 2 := by
  rw [Subtype.dist_eq, dist_eq_norm, antipode]
  have h : -((basePoint 0 : SSphere 1) : EucSpace 1) - ((basePoint 0 : SSphere 1) : EucSpace 1)
      = (2 : ℝ) • -((basePoint 0 : SSphere 1) : EucSpace 1) := by
    module
  rw [h, norm_smul, norm_neg, mem_sphere_zero_iff_norm.mp (basePoint 0).2]
  norm_num

/-- **`cl: W.to.ball` is a claim about the Wasserstein distance, not about an
arbitrary family `W₂`.**

For the constant family `W₂ ≡ 0` every pair of measures is at distance `0`, so
`η₂ = 0` is admissible and the claim becomes `μ(B(x₀, η₃)) ≥ 1` for every
probability measure, every centre and every radius.  The Dirac mass at the
antipode of `x₀` refutes it at radius `1`, because the two points of `𝕊^0` are
at distance `2`. -/
theorem not_forall_wToBall : ¬ ∀ W₂ : W2Family, WToBall W₂ := by
  intro h
  obtain ⟨C, _, hmain⟩ := h (fun _ _ _ => 0)
  let μ₀ : Perspective.ProbSphere 1 := ⟨Measure.dirac antipode, inferInstance⟩
  have hnot : antipode ∉ Metric.ball (basePoint 0 : SSphere 1) 1 := by
    rw [Metric.mem_ball, dist_antipode]
    norm_num
  have hzero : (μ₀ : Measure (SSphere 1)) (Metric.ball (basePoint 0 : SSphere 1) 1) = 0 := by
    show Measure.dirac antipode (Metric.ball (basePoint 0 : SSphere 1) 1) = 0
    rw [Measure.dirac_apply, Set.indicator_of_notMem hnot]
  have hball := hmain 1 μ₀ (basePoint 0) 0 le_rfl 1 one_pos
  rw [hzero] at hball
  norm_num at hball

end Interpolation
end Transformer
