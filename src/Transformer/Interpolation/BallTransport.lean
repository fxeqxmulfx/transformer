module

public import Transformer.Interpolation.Section3_ChainControls
public import Transformer.Interpolation.Section3_TubularFactor

/-!
# Measure-to-measure interpolation — Transporting mass through balls

Formalization of Appendix "Transporting mass through overlapping balls" of
arXiv:2411.04551v3:

* `Lemma lem: two.balls`             — pushing the mass of one ball into its
                                       overlap with a neighbour,
* `Lemma lem: tubular.mass.movement` — iterating along a chain of balls.

The two-ball theorem is proved in `Section3_TwoBalls`; this module proves
the tubular theorem. Shared flow and collapse operations are in
`Section3_BallFlow`.

Both lemmas use only the perceptron part of `eq: vf`, which is `eq: vf` with
`𝐕 ≡ 0`; the `eq: neural.pde.sphere` of the source is therefore `cauchyPB`
at `neuralParams`, and its flow map is that of `eq: neural.ode.sphere`
(`IsNeuralFlow`). The source's geodesic balls are ambient balls here: on the
unit sphere the two families coincide, only the radius being reparametrized.

**Quantifier order.**  The source chooses the parameters first, "such that for
any `μ_0 ∈ 𝒫(𝕊^{d-1})`", and its flow map `φ^t` is then fixed with them. In
that order both lemmas are false whenever the mass must actually move
(`BallTransportFalse`): a bijection which is the identity off `𝓑_0` maps `𝓑_0`
onto itself, so Dirac data forces `𝓑_0 ⊆ 𝓑_1`. The proof chooses its
radius `R - δ` after `μ_0`, and the lemmas are stated here in that order:
`μ_0` first, then the parameters.

**Retained fraction.** The source allows all positive `ε`, but its real
power `(1 - ε)^K` is false for `ε = 3`, `K = 2`, even after the quantifier
correction. `exists_tubular_unclamped_retention_counterexample` proves
this on an explicit three-ball chain. The proved statement clamps the
nonnegative fraction before taking its power; for `ε ≤ 1` the corollary
`tubular_mass_movement_of_error_le_one` has the printed real power.
-/

@[expose] public section

open scoped BigOperators ENNReal
open Real MeasureTheory Set

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- **Lemma (lem: tubular.mass.movement).**

For a chain `𝓑_0, …, 𝓑_K ⊆ 𝕊^{d-1}` of open balls in which consecutive balls
meet and balls two or more apart are disjoint, any `ε, T > 0` and any
`μ_0 ∈ 𝒫(𝕊^{d-1})`, there are piecewise-constant `(𝐖, 𝐔, b)` on `[0, T]`
with at most `K` switches such that the solution of `eq: neural.pde.sphere`
satisfies

  `μ(T, 𝓑_K) ≥ max(1 - ε, 0)^K μ_0(⋃_k 𝓑_k)`,

and `μ(T) = φ^T_# μ_0` for the flow map `φ^t` of `eq: neural.ode.sphere`,
Lipschitz, invertible, and the identity off `⋃_k 𝓑_k` for `t ∈ [0, T]`.

`K` switches are `K + 1` constant pieces, which is what `PiecewiseConstant`
counts. The order of `μ_0` and the parameters, `d ≥ 2` and "the unique
solution" are changed or read as in `two_balls`; the source's order is refuted
by `not_tubular_mass_movement`.

The source prints `(1 - ε)^K` for every `ε > 0`. Its factor is false
when `ε = 3`, `K = 2`, as proved by
`exists_tubular_unclamped_retention_counterexample`. Clamping before the
power corrects that factor, agrees with the printed bound for `ε ≤ 1`,
and keeps the stated positive-error quantifier.

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
        ENNReal.ofReal (1 - ε) ^ K *
            (μ₀ : Measure (SSphere d)) (⋃ k ∈ Set.Iic K, Metric.ball (z k) (R k))
          ≤ (μ T : Measure (SSphere d)) (Metric.ball (z K) (R K)) := by
  let i : Fin d := ⟨0, by omega⟩
  let j : Fin d := ⟨1, by omega⟩
  have hij : i ≠ j := by
    intro he
    have hv := congrArg Fin.val he
    dsimp [i, j] at hv
    omega
  obtain ⟨p, u, b₀, q, v, c, W, U, b, φ, _, hpc, hflow, _, hout, hex, hmap, hmass⟩ :=
    exists_chain_neural_controls hd i j hij K z R hadj hdisj ε T hε hT μ₀
  refine ⟨W, U, b, φ, hpc, hflow, hout, hex μ₀, fun μ h0 hμ => ?_⟩
  exact ⟨by rw [hmap μ hμ T, h0], hmass μ h0 hμ⟩

/-- The hypotheses of `tubular_mass_movement` are satisfiable: on the circle, a
chain of two equal balls, `K = 1`, where consecutive balls meet and the
disjointness requirement is empty. -/
example :
    2 ≤ 2 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (∃ μ₀ : Perspective.ProbSphere 2, (μ₀ : Measure (SSphere 2)) univ = 1) ∧
    (∀ k : ℕ, 1 ≤ k → k ≤ 1 →
        (Metric.ball ((fun _ : ℕ => basePoint 1) k) ((fun _ : ℕ => (1 : ℝ)) k)
          ∩ Metric.ball ((fun _ : ℕ => basePoint 1) (k - 1))
              ((fun _ : ℕ => (1 : ℝ)) (k - 1)) : Set (SSphere 2)).Nonempty) ∧
      ∀ k k' : ℕ, k ≤ 1 → k' ≤ 1 → k + 2 ≤ k' →
        Disjoint (Metric.ball ((fun _ : ℕ => basePoint 1) k) ((fun _ : ℕ => (1 : ℝ)) k)
            : Set (SSphere 2))
          (Metric.ball ((fun _ : ℕ => basePoint 1) k') ((fun _ : ℕ => (1 : ℝ)) k')) := by
  refine ⟨le_rfl, by norm_num, by norm_num, zero_lt_one,
    ⟨Perspective.diracProb 2 (basePoint 1), IsProbabilityMeasure.measure_univ⟩,
    fun k _ _ => ⟨basePoint 1, by simp, by simp⟩, fun k k' _ hk' hle => ?_⟩
  omega

/-- The source's printed real-power bound holds for errors at most one,
with the corrected order of initial law and controls. The source states
only `ε > 0`; the added `ε ≤ 1` restricts its printed retained fraction
to the nonnegative range. Source: arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`. -/
theorem tubular_mass_movement_of_error_le_one (hd : 2 ≤ d)
    (K : ℕ) (z : ℕ → SSphere d) (R : ℕ → ℝ)
    (hadj : ∀ k : ℕ, 1 ≤ k → k ≤ K →
      (Metric.ball (z k) (R k) ∩ Metric.ball (z (k - 1)) (R (k - 1))).Nonempty)
    (hdisj : ∀ k k' : ℕ, k ≤ K → k' ≤ K → k + 2 ≤ k' →
      Disjoint (Metric.ball (z k) (R k)) (Metric.ball (z k') (R k')))
    (ε T : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) (hT : 0 < T) (μ₀ : Perspective.ProbSphere d) :
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
  simpa only [tubular_retention_of_error_le_one ε hε1 K] using
    tubular_mass_movement d hd K z R hadj hdisj ε T hε hT μ₀

/-- An error strictly between zero and one, a positive horizon and an
atomic initial law realize the additional range and data hypotheses. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    ∃ μ₀ : Perspective.ProbSphere 2, (μ₀ : Measure (SSphere 2)) univ = 1 := by
  exact ⟨by norm_num, by norm_num, zero_lt_one, Perspective.diracProb 2 (basePoint 1),
    IsProbabilityMeasure.measure_univ⟩

end Interpolation
end Transformer
