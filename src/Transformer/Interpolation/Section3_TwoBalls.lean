/-
# Transport through two overlapping spherical balls

Formalization of arXiv:2411.04551v3, Appendix, `lem: two.balls`.

The actual original constant controls are realized by two hidden
coordinates, with no change to the width or vector-field model. An inner
cap contains the source center and the requested target and stays strictly
inside the outer activation cap. Its boundary drift brings every active
particle into a region where the calibrated field contracts exponentially.
The complement is fixed pointwise by the same complete neural flow.

For a ball covering the whole sphere, the attracting direction is chosen
near the target with a repelling antipode of zero initial mass. The choice
is possible for every initial probability in dimension at least two.
Dominated convergence and an explicit quadratic transport coupling give
the mass and original Wasserstein bounds at a sufficiently late time.
Scaling the output matrices realizes that endpoint at the requested time.

The separate mass and collapse formulations below expose the two clauses
used independently in the source's subsequent constructions.
-/

import Transformer.Interpolation.Section3_BallControl

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Interpolation

variable (d : ℕ)

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

The source's activation sign is also corrected: its displayed activation
is positive outside the intended ball. The construction uses the inward
activation and a second actual neuron, since the source's single-neuron
argument does not establish attraction to arbitrary targets in large caps.
Both matrices remain constant and retain the original width `d`.

The source's final calculation is written for `W_1`. Here the exact
quadratic coupling cost is bounded directly, proving the stated `W_2`
conclusion for every initial probability, including atomic data.

Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem two_balls (hd : 2 ≤ d)
    (z₀ z₁ : SSphere d) (R₀ R₁ : ℝ)
    (hmeet : (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁).Nonempty)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) (μ₀ : Perspective.ProbSphere d) :
    (∃ (W U : ParamMatrix d) (b : EucSpace d)
        (φ : ℝ → SSphere d → SSphere d),
        IsNeuralFlow d (fun _ => W) (fun _ => U) (fun _ => b) φ ∧
        IsTransportOutside d φ (Metric.ball z₀ R₀) T ∧
        (∃ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ ∧ cauchyPB d (constNeural d W U b) μ) ∧
        ∀ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ → cauchyPB d (constNeural d W U b) μ →
          (μ T : Measure (SSphere d)) =
            Measure.map (φ T) (μ₀ : Measure (SSphere d)) ∧
          ENNReal.ofReal (1 - ε) * (μ₀ : Measure (SSphere d)) (Metric.ball z₀ R₀)
            ≤ (μ T : Measure (SSphere d)) (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁)) ∧
      ∀ ω ∈ Metric.ball z₀ R₀, ∃ (W U : ParamMatrix d) (b : EucSpace d),
        (∃ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ ∧ cauchyPB d (constNeural d W U b) μ) ∧
        ∀ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = μ₀ → cauchyPB d (constNeural d W U b) μ →
          W2 d (μ T) (collapseInto d μ₀ (Metric.ball z₀ R₀) ω) ≤ ε := by
  let i : Fin d := ⟨0, by omega⟩
  let j : Fin d := ⟨1, by omega⟩
  have hij : i ≠ j := Fin.ne_of_val_ne (by change (0 : ℕ) ≠ 1; omega)
  constructor
  · obtain ⟨ω, hω⟩ := hmeet
    obtain ⟨W, U, b, φ, hflow, hout, hex, hall⟩ :=
      ball_control hd i j hij z₀ R₀
      (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁) (Metric.isOpen_ball.inter Metric.isOpen_ball)
      inter_subset_left ω hω ε T hε hT.ne' μ₀
    refine ⟨W, U, b, φ, hflow, hout, hex, fun μ h0 hμ => ?_⟩
    exact ⟨(hall μ h0 hμ).1, (hall μ h0 hμ).2.1⟩
  · intro ω hω
    obtain ⟨W, U, b, _, _, _, hex, hall⟩ :=
      ball_control hd i j hij z₀ R₀
      (Metric.ball z₀ R₀) Metric.isOpen_ball Subset.rfl ω hω ε T hε hT.ne' μ₀
    exact ⟨W, U, b, hex, fun μ h0 hμ => (hall μ h0 hμ).2.2⟩

/-- The hypotheses of `two_balls` are satisfiable: two copies of the same ball
on the circle overlap. -/
example :
    2 ≤ 2 ∧ ((Metric.ball (basePoint 1) 1 ∩ Metric.ball (basePoint 1) 1 :
        Set (SSphere 2))).Nonempty ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 :=
  ⟨le_rfl, ⟨basePoint 1, by simp, by simp⟩, one_pos, one_pos⟩


/-- The mass-transport clause of the proved two-ball lemma, exposed for
iteration along chains of overlapping balls. The initial probability
precedes the controls, as in the corrected main statement.
Source: arXiv:2411.04551v3, Appendix, `eq: claim.statement.lemmaA1`. -/
theorem two_balls_mass_transport (hd : 2 ≤ d)
    (z₀ z₁ : SSphere d) (R₀ R₁ : ℝ)
    (hmeet : (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁).Nonempty)
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) (μ₀ : Perspective.ProbSphere d) :
    ∃ (W U : ParamMatrix d) (b : EucSpace d)
      (φ : ℝ → SSphere d → SSphere d),
      IsNeuralFlow d (fun _ => W) (fun _ => U) (fun _ => b) φ ∧
      IsTransportOutside d φ (Metric.ball z₀ R₀) T ∧
      (∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = μ₀ ∧ cauchyPB d (constNeural d W U b) μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = μ₀ → cauchyPB d (constNeural d W U b) μ →
        (μ T : Measure (SSphere d)) = Measure.map (φ T) (μ₀ : Measure (SSphere d)) ∧
        ENNReal.ofReal (1 - ε) * (μ₀ : Measure (SSphere d)) (Metric.ball z₀ R₀) ≤
          (μ T : Measure (SSphere d)) (Metric.ball z₀ R₀ ∩ Metric.ball z₁ R₁) :=
  (two_balls d hd z₀ z₁ R₀ R₁ hmeet ε T hε hT μ₀).1

/-- The collapse clause needs only one ball and a target inside it.
The target neighborhood in the mass clause can be chosen independently.
Source: arXiv:2411.04551v3, Appendix, last clause of `lem: two.balls`. -/
theorem collapse_ball_at_target (hd : 2 ≤ d) (z ω : SSphere d) (R : ℝ)
    (hω : ω ∈ Metric.ball z R) (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T)
    (μ₀ : Perspective.ProbSphere d) :
    ∃ (W U : ParamMatrix d) (b : EucSpace d),
      (∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = μ₀ ∧ cauchyPB d (constNeural d W U b) μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = μ₀ → cauchyPB d (constNeural d W U b) μ →
        W2 d (μ T) (collapseInto d μ₀ (Metric.ball z R) ω) ≤ ε :=
  (two_balls d hd z z R R ⟨ω, hω, hω⟩ ε T hε hT μ₀).2 ω hω

/-- The additional target hypothesis of the standalone collapse clause
holds for a genuine ball on the circle. -/
example : (basePoint 1 : SSphere 2) ∈ Metric.ball (basePoint 1) 1 := by
  simp

end Transformer.Interpolation
