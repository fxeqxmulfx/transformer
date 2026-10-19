module

public import Transformer.Interpolation.Section2_PerceptronControls
public import Transformer.Interpolation.Section3_MassConcentration
public import Transformer.Interpolation.Section3_CapAttractor

/-!
# Ball collapses as local perceptron phases

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, through the
ball moves of `lem: tubular.mass.movement` in the Appendix. A proper
ball `B(z, R)`, `R ≤ 2`, and a point `ω` inside it have a constant
two-neuron perceptron field whose flow sends every point of the ball
to `ω` and fixes every other point. At a clock chosen for one initial
law, the phase keeps all but a fraction `ε` of that law's mass in the
ball inside a prescribed open set around `ω`.

The phase is local in the ball for every initial law; only its clock
depends on the chosen law.
-/

@[expose] public section

open Real MeasureTheory Set Filter

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The constant two-neuron phase with its proved law. Source auxiliary:
arXiv:2411.04551v3, Appendix, the two-ridge fields of `lem: two.balls`. -/
noncomputable def twoRidgePhase (i j : Fin d) (p u : EucSpace d) (b : ℝ) (q v : EucSpace d)
    (c : ℝ) : AutonomousPhase d :=
  (twoRidgeParams i j p u b q v c, twoRidgeLaw p u b q v c)

/-- The two-neuron phase solves the original problem with its proved law.
Source auxiliary: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem isAutonomousPhase_twoRidge (i j : Fin d) (hij : i ≠ j) (p u : EucSpace d) (b : ℝ)
    (q v : EucSpace d) (c : ℝ) : IsAutonomousPhase (twoRidgePhase i j p u b q v c) :=
  ⟨twoRidgeLaw_zero p u b q v c, cauchyPB_twoRidgeLaw i j hij p u b q v c,
    fun μ hμ => cauchyPB_twoRidge_eq i j hij p u b q v c (μ 0) μ hμ rfl⟩

/-- A two-neuron flow at a nonzero clock, identity off `D`, is a local
phase in `D`. Source auxiliary: arXiv:2411.04551v3, §2.2. -/
theorem isLocalPhase_twoRidge (i j : Fin d) (hij : i ≠ j) (p u : EucSpace d) (b : ℝ)
    (q v : EucSpace d) (c τ : ℝ) (hτ : τ ≠ 0) (D : Set (SSphere d))
    (hfix : ∀ x : SSphere d, x ∉ D → ∀ t : ℝ, twoRidgeFlow p u b q v c x t = x) :
    IsLocalPhase (rescalePhase τ (twoRidgePhase i j p u b q v c)) D
      (fun x => twoRidgeFlow p u b q v c x τ) := by
  obtain ⟨L, hL⟩ := exists_lipschitzWith_twoRidgeFlow p u b q v c τ
  refine ⟨(isAutonomousPhase_twoRidge i j hij p u b q v c).rescale _ τ hτ, rfl,
    rescalePhase_V_zero _ τ rfl, hL.continuous, bijective_twoRidgeFlow p u b q v c τ,
    fun x hx => hfix x hx τ, fun ρ => ?_⟩
  change (twoRidgeLaw p u b q v c ρ (τ * 1) : Measure (SSphere d)) = _
  rw [mul_one, coe_twoRidgeLaw]

/-- **Ball collapse.** For a proper ball, a point `ω` in an open set `A`
inside it, an accuracy `ε ∈ (0, 1]` and an initial law `ν`, some phase
local in the ball sends at least the fraction `1 - ε` of the mass of
`ν` in the ball into `A`. Source auxiliary: arXiv:2411.04551v3, §2.2,
Step 3, and Appendix, `lem: two.balls`. -/
theorem exists_collapse_phase (i j : Fin d) (hij : i ≠ j) (z ω : SSphere d) (R : ℝ)
    (hR : 0 < R) (hR2 : R ≤ 2) (A : Set (SSphere d)) (hA : IsOpen A)
    (hAB : A ⊆ Metric.ball z R) (hω : ω ∈ A) (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1)
    (ν : Perspective.ProbSphere d) :
    ∃ (P : AutonomousPhase d) (Φ : SSphere d → SSphere d),
      IsLocalPhase P (Metric.ball z R) Φ ∧
      ENNReal.ofReal (1 - ε) * (ν : Measure (SSphere d)) (Metric.ball z R) ≤
        (P.2 ν 1 : Measure (SSphere d)) A := by
  obtain ⟨w, h, ℓ, hattract, hfixed⟩ := exists_twoRidge_ball_attractor z ω R hR hR2 (hAB hω)
  let c := 1 - R ^ 2 / 2
  let q := capTargetOutput z w ω c h ℓ
  let Φ (t : ℝ) (x : SSphere d) :=
    twoRidgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) q (-(w : EucSpace d)) (-h) x t
  have hΦ (t : ℝ) : Measurable (Φ t) := measurable_twoRidgeFlow_initial _ _ _ _ _ _ t
  have hm := eventually_map_mass_ge ν Φ hΦ (Metric.ball z R) A Metric.isOpen_ball.measurableSet
    hA hAB ω hω (Eventually.of_forall hattract) hfixed ε hε hε1
  obtain ⟨τ, hτm, hτ⟩ := (hm.and (eventually_gt_atTop (0 : ℝ))).exists
  have hloc := isLocalPhase_twoRidge i j hij (z : EucSpace d) (-(z : EucSpace d)) (-c) q
    (-(w : EucSpace d)) (-h) τ hτ.ne' (Metric.ball z R) hfixed
  refine ⟨_, _, hloc, ?_⟩
  rw [hloc.2.2.2.2.2.2 ν]
  exact hτm

/-- The hypotheses of `exists_collapse_phase` are satisfiable: two
distinct hidden coordinates, a ball of radius one around a point, the
ball itself as target set, and accuracy one half. -/
example : (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 ∧
    IsOpen (Metric.ball (basePoint 1 : SSphere 2) 1) ∧
    (basePoint 1 : SSphere 2) ∈ Metric.ball (basePoint 1 : SSphere 2) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 :=
  ⟨by decide, one_pos, by norm_num, Metric.isOpen_ball, Metric.mem_ball_self one_pos,
    by norm_num, by norm_num⟩

end Transformer.Interpolation
