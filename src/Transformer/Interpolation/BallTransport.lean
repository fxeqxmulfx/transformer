/-
# Measure-to-measure interpolation — Transporting mass through balls

Formalization of Appendix "Transporting mass through overlapping balls" of
arXiv:2411.04551v3:

* `Lemma lem: two.balls`             — pushing the mass of one ball into its
                                       overlap with a neighbour,
* `Lemma lem: tubular.mass.movement` — iterating along a chain of balls.

Both lemmas use only the perceptron part of `eq: vf`, which is `eq: vf` with
`𝐕 ≡ 0`; the `eq: neural.pde.sphere` of the survey is therefore `cauchyPB`
under that constraint, and no separate continuity equation is introduced.
The survey's geodesic balls are ambient balls here: on the unit sphere the two
families coincide, only the radius being reparametrized.
-/

import Transformer.Basic
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Clustering

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- A time-`t` transport map realizing `μ(t) = φ^t_# μ(0)`: Lipschitz,
bijective, and the identity outside `S`.

This is the "moreover" clause shared by `lem: two.balls` and
`lem: tubular.mass.movement`. -/
def IsTransportOutside
    (μ : ℝ → Perspective.ProbSphere d) (φ : ℝ → SSphere d → SSphere d)
    (S : Set (SSphere d)) (T : ℝ) : Prop :=
  ∀ t : ℝ, t ∈ Set.Icc 0 T →
    (∃ L : NNReal, LipschitzWith L (φ t)) ∧
    Function.Bijective (φ t) ∧
    (∀ x : SSphere d, x ∉ S → φ t x = x) ∧
    (μ t : Measure (SSphere d)) = Measure.map (φ t) (μ 0 : Measure (SSphere d))

/-- **Lemma (lem: two.balls).**

For two overlapping open balls `𝓑_0, 𝓑_1 ⊆ 𝕊^{d-1}` and any `ε, T > 0` there
are constant parameters `(𝐖, 𝐔, b)` such that every solution of
`eq: neural.pde.sphere` satisfies

  `μ(T, 𝓑_0 ∩ 𝓑_1) ≥ (1 - ε) μ_0(𝓑_0)`,

and `μ(T) = φ^T_# μ_0` for a Lipschitz, invertible `φ^t` which is the identity
off `𝓑_0`.

Not proved here.

Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem two_balls
    (z₀ z₁ : SSphere d) (R₀ R₁ : ℝ)
    (hmeet : (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁).Nonempty)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ θ : TimeParams d,
      (∀ s : ℝ, (θ s).V = 0) ∧ (∀ s : ℝ, (θ s).B = 0) ∧
      PiecewiseConstant d θ T 1 ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ →
        ∃ φ : ℝ → SSphere d → SSphere d,
          IsTransportOutside d μ φ (Metric.ball z₀ R₀) T ∧
          ENNReal.ofReal (1 - ε) *
              (μ 0 : Measure (SSphere d)) (Metric.ball z₀ R₀)
            ≤ (μ T : Measure (SSphere d))
                (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁) := by
  sorry

/-- The unit vector `e_0`, as a point of `𝕊^0`. -/
noncomputable def basePoint : SSphere 1 :=
  ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- The hypotheses of `two_balls` are satisfiable: two copies of the same ball
overlap. -/
example :
    ((Metric.ball basePoint 1 ∩ Metric.ball basePoint 1 : Set (SSphere 1))).Nonempty
      ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 :=
  ⟨⟨basePoint, by simp, by simp⟩, one_pos, one_pos⟩

/-- **Lemma (lem: tubular.mass.movement).**

For a chain `𝓑_0, …, 𝓑_K ⊆ 𝕊^{d-1}` of open balls in which consecutive balls
meet and non-consecutive ones are disjoint, and any `ε, T > 0`, there are
piecewise-constant parameters with at most `K` switches such that every
solution of `eq: neural.pde.sphere` satisfies

  `μ(T, 𝓑_K) ≥ (1 - ε)^K μ_0(⋃_k 𝓑_k)`,

and `μ(T) = φ^T_# μ_0` for a Lipschitz, invertible `φ^t` which is the identity
off `⋃_k 𝓑_k`.

`K` switches means `K + 1` constant pieces, which is what `PiecewiseConstant`
counts.  Not proved here.

Source: arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem tubular_mass_movement
    (K : ℕ) (z : ℕ → SSphere d) (R : ℕ → ℝ)
    (hadj : ∀ k : ℕ, 1 ≤ k → k ≤ K →
      (Metric.ball (z k) (R k) ∩ Metric.ball (z (k - 1)) (R (k - 1))).Nonempty)
    (hdisj : ∀ k k' : ℕ, k ≤ K → k' ≤ K → k + 2 ≤ k' →
      Disjoint (Metric.ball (z k) (R k)) (Metric.ball (z k') (R k')))
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∃ θ : TimeParams d,
      (∀ s : ℝ, (θ s).V = 0) ∧ (∀ s : ℝ, (θ s).B = 0) ∧
      PiecewiseConstant d θ T (K + 1) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ →
        ∃ φ : ℝ → SSphere d → SSphere d,
          IsTransportOutside d μ φ
            (⋃ k ∈ Set.Iic K, Metric.ball (z k) (R k)) T ∧
          ENNReal.ofReal ((1 - ε) ^ K) *
              (μ 0 : Measure (SSphere d))
                (⋃ k ∈ Set.Iic K, Metric.ball (z k) (R k))
            ≤ (μ T : Measure (SSphere d)) (Metric.ball (z K) (R K)) := by
  sorry

/-- The hypotheses of `tubular_mass_movement` are satisfiable: a chain of two
equal balls, `K = 1`, where consecutive balls meet and the disjointness
requirement is empty. -/
example :
    (∀ k : ℕ, 1 ≤ k → k ≤ 1 →
        (Metric.ball ((fun _ : ℕ => basePoint) k) ((fun _ : ℕ => (1 : ℝ)) k)
          ∩ Metric.ball ((fun _ : ℕ => basePoint) (k - 1))
              ((fun _ : ℕ => (1 : ℝ)) (k - 1))).Nonempty) ∧
      ∀ k k' : ℕ, k ≤ 1 → k' ≤ 1 → k + 2 ≤ k' →
        Disjoint (Metric.ball ((fun _ : ℕ => basePoint) k) ((fun _ : ℕ => (1 : ℝ)) k))
          (Metric.ball ((fun _ : ℕ => basePoint) k') ((fun _ : ℕ => (1 : ℝ)) k')) := by
  refine ⟨fun k _ _ => ⟨basePoint, by simp, by simp⟩, fun k k' _ hk' hle => ?_⟩
  omega

end Interpolation
end Transformer
