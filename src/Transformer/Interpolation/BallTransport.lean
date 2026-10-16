/-
# Measure-to-measure interpolation — Transporting mass through balls

Formalization of Appendix "Transporting mass through overlapping balls" of
arXiv:2411.04551v3:

* `Lemma lem: two.balls`             — pushing the mass of one ball into its
                                       overlap with a neighbour,
* `Lemma lem: tubular.mass.movement` — iterating along a chain of balls.

The proved two-ball statement is in `Section3_TwoBalls`. This module holds
the shared flow and collapse definitions and the remaining chain statement.

Both lemmas use only the perceptron part of `eq: vf`, which is `eq: vf` with
`𝐕 ≡ 0`; the `eq: neural.pde.sphere` of the source is therefore `cauchyPB`
at `neuralParams`, and its flow map is that of `eq: neural.ode.sphere`
(`IsNeuralFlow`).  The source's geodesic balls are ambient balls here: on the
unit sphere the two families coincide, only the radius being reparametrized.

**Quantifier order.**  The source chooses the parameters first, "such that for
any `μ_0 ∈ 𝒫(𝕊^{d-1})`", and its flow map `φ^t` is then fixed with them.  In
that order both lemmas are false whenever the mass must actually move
(`BallTransportFalse`): a bijection which is the identity off `𝓑_0` maps `𝓑_0`
onto itself, so Dirac data forces `𝓑_0 ⊆ 𝓑_1`.  The proof chooses its
radius `R - δ` after `μ_0`, and the lemmas are stated here in that order:
`μ_0` first, then the parameters.
-/

import Transformer.Basic
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Clustering
import Transformer.Interpolation.NeuralODE

open scoped BigOperators ENNReal
open Real MeasureTheory Set

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- `φ` is the flow map of `eq: neural.ode.sphere` for the parameters
`(𝐖, 𝐔, b)`: from every `x`, `t ↦ φ^t(x)` solves the equation, in integrated
form as `neuralODESphere`.

Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
def IsNeuralFlow (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (φ : ℝ → SSphere d → SSphere d) : Prop :=
  ∀ (x : SSphere d) (t : ℝ),
    IntervalIntegrable (fun s => neuralVF d W U b s (φ s x)) volume 0 t ∧
      (φ t x : EucSpace d) = x + ∫ s in (0 : ℝ)..t, neuralVF d W U b s (φ s x)

/-- At every time of `[0, T]` the map `φ^t` is Lipschitz, invertible, and the
identity outside `S`: the "moreover" clause shared by `lem: two.balls` and
`lem: tubular.mass.movement`.

Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
def IsTransportOutside (φ : ℝ → SSphere d → SSphere d) (S : Set (SSphere d)) (T : ℝ) :
    Prop :=
  ∀ t ∈ Set.Icc (0 : ℝ) T,
    (∃ L : NNReal, LipschitzWith L (φ t)) ∧ Function.Bijective (φ t) ∧
      ∀ x : SSphere d, x ∉ S → φ t x = x

/-- Constant parameters `(𝐕, 𝐁, 𝐖, 𝐔, b) ≡ (0, 0, 𝐖, 𝐔, b)`. -/
def constNeural (W U : ParamMatrix d) (b : EucSpace d) : TimeParams d :=
  neuralParams d (fun _ => W) (fun _ => U) (fun _ => b)

/-- The target of the last clause of `lem: two.balls`,
`α(A) = μ_0(𝓑_0) δ_ω(A) + μ_0(A ∖ 𝓑_0)`: the mass of `S` collapsed onto `ω`.

Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
noncomputable def collapseInto (μ₀ : Measure (SSphere d)) (S : Set (SSphere d))
    (ω : SSphere d) : Measure (SSphere d) :=
  μ₀ S • Measure.dirac ω + μ₀.restrict Sᶜ

section CollapseMaps

variable {d : ℕ}

/-- The measurable collapse used in the source's measure `α`.
Source: arXiv:2411.04551v3, Appendix, last clause of `lem: two.balls`. -/
noncomputable def collapseMap (S : Set (SSphere d)) (ω x : SSphere d) : SSphere d := by
  classical
  exact if x ∈ S then ω else x

/-- Collapsing a measurable region gives a measurable map.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem measurable_collapseMap (S : Set (SSphere d)) (hS : MeasurableSet S) (ω : SSphere d) :
    Measurable (collapseMap S ω) := by
  classical
  exact Measurable.ite hS measurable_const measurable_id

/-- The collapse pushforward is exactly the source's measure `α`, including
the untouched restricted measure. Source: arXiv:2411.04551v3,
Appendix, the final clause of `lem: two.balls`. -/
theorem map_collapseMap (μ : Measure (SSphere d)) (S : Set (SSphere d)) (hS : MeasurableSet S) (ω : SSphere d) :
    Measure.map (collapseMap S ω) μ = collapseInto d μ S ω := by
  classical
  ext A hA
  rw [Measure.map_apply (measurable_collapseMap S hS ω) hA, collapseInto,
    Measure.add_apply, Measure.smul_apply, Measure.restrict_apply hA, Measure.dirac_apply' ω hA]
  by_cases hω : ω ∈ A
  · have he : collapseMap S ω ⁻¹' A = S ∪ (A ∩ Sᶜ) := by
      ext x
      by_cases hx : x ∈ S <;> simp [collapseMap, hx, hω]
    rw [he, measure_union (by exact disjoint_left.mpr (fun x hx hx' => hx'.2 hx)) (hA.inter hS.compl)]
    simp [hω]
  · have he : collapseMap S ω ⁻¹' A = A ∩ Sᶜ := by
      ext x
      by_cases hx : x ∈ S <;> simp [collapseMap, hx, hω]
    rw [he]
    simp [hω]

/-- A genuine spherical ball realizes the measurable-collapse hypothesis. -/
example : MeasurableSet (Metric.ball (basePoint 1) 1 : Set (SSphere 2)) :=
  Metric.isOpen_ball.measurableSet

end CollapseMaps

/-- **Lemma (lem: tubular.mass.movement).**

For a chain `𝓑_0, …, 𝓑_K ⊆ 𝕊^{d-1}` of open balls in which consecutive balls
meet and balls two or more apart are disjoint, any `ε, T > 0` and any
`μ_0 ∈ 𝒫(𝕊^{d-1})`, there are piecewise-constant `(𝐖, 𝐔, b)` on `[0, T]`
with at most `K` switches such that the solution of `eq: neural.pde.sphere`
satisfies

  `μ(T, 𝓑_K) ≥ (1 - ε)^K μ_0(⋃_k 𝓑_k)`,

and `μ(T) = φ^T_# μ_0` for the flow map `φ^t` of `eq: neural.ode.sphere`,
Lipschitz, invertible, and the identity off `⋃_k 𝓑_k` for `t ∈ [0, T]`.

`K` switches are `K + 1` constant pieces, which is what `PiecewiseConstant`
counts.  The order of `μ_0` and the parameters, `d ≥ 2` and "the unique
solution" are changed or read as in `two_balls`; the source's order is refuted
by `not_tubular_mass_movement`.

Not proved here.

Source: arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem tubular_mass_movement (hd : 2 ≤ d)
    (K : ℕ) (z : ℕ → SSphere d) (R : ℕ → ℝ)
    (hadj : ∀ k : ℕ, 1 ≤ k → k ≤ K →
      (Metric.ball (z k) (R k) ∩ Metric.ball (z (k - 1)) (R (k - 1))).Nonempty)
    (hdisj : ∀ k k' : ℕ, k ≤ K → k' ≤ K → k + 2 ≤ k' →
      Disjoint (Metric.ball (z k) (R k)) (Metric.ball (z k') (R k')))
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) (μ₀ : Perspective.ProbSphere d) :
    ∃ (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) (φ : ℝ → SSphere d → SSphere d),
      PiecewiseConstant d (neuralParams d W U b) T (K + 1) ∧
      IsNeuralFlow d W U b φ ∧
      IsTransportOutside d φ (⋃ k ∈ Set.Iic K, Metric.ball (z k) (R k)) T ∧
      (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d (neuralParams d W U b) μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ → cauchyPB d (neuralParams d W U b) μ →
        (μ T : Measure (SSphere d)) = Measure.map (φ T) (μ₀ : Measure (SSphere d)) ∧
        ENNReal.ofReal ((1 - ε) ^ K) *
            (μ₀ : Measure (SSphere d)) (⋃ k ∈ Set.Iic K, Metric.ball (z k) (R k))
          ≤ (μ T : Measure (SSphere d)) (Metric.ball (z K) (R K)) := by
  sorry

/-- The hypotheses of `tubular_mass_movement` are satisfiable: on the circle, a
chain of two equal balls, `K = 1`, where consecutive balls meet and the
disjointness requirement is empty. -/
example :
    2 ≤ 2 ∧
    (∀ k : ℕ, 1 ≤ k → k ≤ 1 →
        (Metric.ball ((fun _ : ℕ => basePoint 1) k) ((fun _ : ℕ => (1 : ℝ)) k)
          ∩ Metric.ball ((fun _ : ℕ => basePoint 1) (k - 1))
              ((fun _ : ℕ => (1 : ℝ)) (k - 1)) : Set (SSphere 2)).Nonempty) ∧
      ∀ k k' : ℕ, k ≤ 1 → k' ≤ 1 → k + 2 ≤ k' →
        Disjoint (Metric.ball ((fun _ : ℕ => basePoint 1) k) ((fun _ : ℕ => (1 : ℝ)) k)
            : Set (SSphere 2))
          (Metric.ball ((fun _ : ℕ => basePoint 1) k') ((fun _ : ℕ => (1 : ℝ)) k')) := by
  refine ⟨le_rfl, fun k _ _ => ⟨basePoint 1, by simp, by simp⟩, fun k k' _ hk' hle => ?_⟩
  omega

end Interpolation
end Transformer
