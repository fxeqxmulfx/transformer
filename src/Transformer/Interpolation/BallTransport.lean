/-
# Measure-to-measure interpolation — Transporting mass through balls

Formalization of Appendix "Transporting mass through overlapping balls" of
arXiv:2411.04551v3:

* `Lemma lem: two.balls`             — pushing the mass of one ball into its
                                       overlap with a neighbour,
* `Lemma lem: tubular.mass.movement` — iterating along a chain of balls.

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
open Real MeasureTheory

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

/-- **Lemma (lem: two.balls).**

For two overlapping open balls `𝓑_0, 𝓑_1 ⊆ 𝕊^{d-1}`, any `ε, T > 0` and any
`μ_0 ∈ 𝒫(𝕊^{d-1})` there are constant `𝐖, 𝐔 ∈ M_{d×d}(ℝ)`, `b ∈ ℝ^d` such
that the solution of `eq: neural.pde.sphere` satisfies

  `μ(T, 𝓑_0 ∩ 𝓑_1) ≥ (1 - ε) μ_0(𝓑_0)`,

and `μ(T) = φ^T_# μ_0` for the flow map `φ^t` of `eq: neural.ode.sphere`,
Lipschitz, invertible, and the identity off `𝓑_0` for `t ∈ [0, T]`.
Furthermore, for any `ω ∈ 𝓑_0` the parameters can be chosen so that
`W_2(μ(T), α) ≤ ε`, `α = μ_0(𝓑_0) δ_ω + μ_0⌊(𝕊^{d-1} ∖ 𝓑_0)`.

What is changed.  The source chooses `(𝐖, 𝐔, b)` before `μ_0`; that order is
false (`not_two_balls`), and `μ_0` comes first here, as in the proof.  `d ≥ 2`
is added: on `𝕊^0` every tangent projection vanishes, no mass moves, and the
claim fails for `𝓑_0 = 𝕊^0 ⊋ 𝓑_1`.  "The unique solution" is read as: one
exists, and every solution satisfies the conclusion.

Not proved here.

Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem two_balls (hd : 2 ≤ d)
    (z₀ z₁ : SSphere d) (R₀ R₁ : ℝ)
    (hmeet : (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁).Nonempty)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) (μ₀ : Perspective.ProbSphere d) :
    (∃ (W U : ParamMatrix d) (b : EucSpace d) (φ : ℝ → SSphere d → SSphere d),
        IsNeuralFlow d (fun _ => W) (fun _ => U) (fun _ => b) φ ∧
        IsTransportOutside d φ (Metric.ball z₀ R₀) T ∧
        (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d (constNeural d W U b) μ) ∧
        ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ → cauchyPB d (constNeural d W U b) μ →
          (μ T : Measure (SSphere d)) = Measure.map (φ T) (μ₀ : Measure (SSphere d)) ∧
          ENNReal.ofReal (1 - ε) * (μ₀ : Measure (SSphere d)) (Metric.ball z₀ R₀)
            ≤ (μ T : Measure (SSphere d)) (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁)) ∧
      ∀ ω ∈ Metric.ball z₀ R₀, ∃ (W U : ParamMatrix d) (b : EucSpace d),
        (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d (constNeural d W U b) μ) ∧
        ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ → cauchyPB d (constNeural d W U b) μ →
          W2 d (μ T) (collapseInto d μ₀ (Metric.ball z₀ R₀) ω) ≤ ε := by
  sorry

/-- The hypotheses of `two_balls` are satisfiable: two copies of the same ball
on the circle overlap. -/
example :
    2 ≤ 2 ∧ ((Metric.ball (basePoint 1) 1 ∩ Metric.ball (basePoint 1) 1 :
        Set (SSphere 2))).Nonempty ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 :=
  ⟨le_rfl, ⟨basePoint 1, by simp, by simp⟩, one_pos, one_pos⟩

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
