module

public import Transformer.Homogenized.Section7_LogisticArcsinTests

/-!
# A globally bounded coordinate for logistic interval invariance

ArXiv:2604.01978v1, §7, `sec: sde.logistic`, the state interval used
in the angle argument for `eq: sde.logistic`.
The source uses u in [-1,1] before applying arcsin. To justify that
range from the original smooth-test martingale problem, use
w(u)=(1-u²)/(1+u²). Its sign encodes precisely the original interval,
and its absolute value is bounded by one on the entire real line.

The original logistic generator satisfies Lw=A w, and its actual
diffusion contribution is B² w². Both A and B² are globally bounded,
even though the original polynomial coefficients are unbounded.
These identities permit smooth negative-part tests without stopping
the original process or assuming the interval invariance being proved.
The bounds 20 and 8 are auxiliary proof constants, not source constants.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology
namespace Transformer.Homogenized

/-- A bounded coordinate whose nonnegative region is the original state interval.
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`, interval invariance. -/
noncomputable def logisticBoundaryCoordinate (u : ℝ) : ℝ := (1 - u ^ 2) / (1 + u ^ 2)

/-- The actual multiplicative drift of the bounded coordinate.
Source: arXiv:2604.01978v1, §7, the generator of `eq: sde.logistic`. -/
noncomputable def logisticBoundaryRate (u : ℝ) : ℝ :=
  4 * (-1 + 5 * u ^ 2 - 2 * u ^ 4) / (1 + u ^ 2) ^ 2

/-- The actual multiplicative variance of the bounded coordinate.
Source: arXiv:2604.01978v1, §7, the diffusion of `eq: sde.logistic`. -/
noncomputable def logisticBoundaryVariance (u : ℝ) : ℝ :=
  32 * u ^ 2 / (1 + u ^ 2) ^ 2

/-- The bounded coordinate is an admissible global smooth test.
Source: arXiv:2604.01978v1, §7, the smooth-test martingale problem. -/
theorem contDiff_logisticBoundaryCoordinate :
    ContDiff ℝ (⊤ : ℕ∞) logisticBoundaryCoordinate := by
  exact (contDiff_const.sub (contDiff_id.pow 2)).div
    (contDiff_const.add (contDiff_id.pow 2)) (fun u => by positivity)

/-- The coordinate is bounded globally and detects exactly [-1,1].
Source: arXiv:2604.01978v1, §7, `sec: sde.logistic`. -/
theorem logisticBoundaryCoordinate_bounds (u : ℝ) :
    |logisticBoundaryCoordinate u| ≤ 1 ∧
      (0 ≤ logisticBoundaryCoordinate u ↔ u ∈ Icc (-1 : ℝ) 1) := by
  have hp : 0 < 1 + u ^ 2 := by positivity
  constructor
  · rw [logisticBoundaryCoordinate, abs_div, abs_of_pos hp]
    apply (div_le_one hp).mpr
    exact abs_le.mpr ⟨by linarith [sq_nonneg u], by linarith [sq_nonneg u]⟩
  · rw [logisticBoundaryCoordinate, le_div_iff₀ hp]
    simp only [zero_mul]
    constructor
    · intro h
      have hab : |u| ≤ 1 := (sq_le_one_iff_abs_le_one u).mp (by linarith)
      exact abs_le.mp hab
    · intro h
      have hs : u ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one u).mpr (abs_le.mpr h)
      linarith

/-- The actual first derivative used in the original generator.
Source: arXiv:2604.01978v1, §7, `eq: sde.logistic`. -/
theorem hasDerivAt_logisticBoundaryCoordinate (u : ℝ) :
    HasDerivAt logisticBoundaryCoordinate (-4 * u / (1 + u ^ 2) ^ 2) u := by
  have hp : 1 + u ^ 2 ≠ 0 := by positivity
  have hd := (((hasDerivAt_id u).pow 2).const_sub 1).div
    (((hasDerivAt_id u).pow 2).const_add 1) hp
  change HasDerivAt logisticBoundaryCoordinate _ u at hd
  apply hd.congr_deriv
  dsimp
  ring

/-- The actual second derivative, including outside the state interval.
Source: arXiv:2604.01978v1, §7, `eq: sde.logistic`. -/
theorem hasDerivAt_logisticBoundaryDerivative (u : ℝ) :
    HasDerivAt (fun v : ℝ => -4 * v / (1 + v ^ 2) ^ 2)
      ((-4 + 12 * u ^ 2) / (1 + u ^ 2) ^ 3) u := by
  have hp : 1 + u ^ 2 ≠ 0 := by positivity
  have hd := ((hasDerivAt_id u).const_mul (-4)).div
    ((((hasDerivAt_id u).pow 2).const_add 1).pow 2) (pow_ne_zero 2 hp)
  apply hd.congr_deriv
  dsimp
  field_simp
  ring

/-- Both original drift and diffusion retain the vanishing boundary factor.
Source: arXiv:2604.01978v1, §7, `eq: sde.logistic`, interval invariance. -/
theorem logisticBoundaryCoordinate_generator (u : ℝ) :
    logisticGenerator logisticBoundaryCoordinate u =
      logisticBoundaryRate u * logisticBoundaryCoordinate u ∧
    2 * (1 - u ^ 2) ^ 2 * deriv logisticBoundaryCoordinate u ^ 2 =
      logisticBoundaryVariance u * logisticBoundaryCoordinate u ^ 2 := by
  have hp : 1 + u ^ 2 ≠ 0 := by positivity
  have hd : deriv logisticBoundaryCoordinate = fun v => -4 * v / (1 + v ^ 2) ^ 2 :=
    funext (fun v => (hasDerivAt_logisticBoundaryCoordinate v).deriv)
  constructor
  · rw [logisticGenerator, hd, (hasDerivAt_logisticBoundaryDerivative u).deriv]
    unfold logisticBoundaryRate logisticBoundaryCoordinate
    field_simp
    ring
  · rw [hd]
    unfold logisticBoundaryVariance logisticBoundaryCoordinate
    field_simp
    ring

/-- Global bounds for the computed multiplicative coefficients.
Source: arXiv:2604.01978v1, §7, interval invariance; 20 and 8 are auxiliary bounds. -/
theorem logisticBoundary_coefficients_bounds (u : ℝ) :
    |logisticBoundaryRate u| ≤ 20 ∧ logisticBoundaryVariance u ∈ Icc (0 : ℝ) 8 := by
  have hp : 0 < (1 + u ^ 2) ^ 2 := by positivity
  constructor
  · rw [logisticBoundaryRate, abs_div, abs_of_pos hp]
    apply (div_le_iff₀ hp).mpr
    apply abs_le.mpr
    constructor <;> nlinarith [sq_nonneg u, sq_nonneg (u ^ 2)]
  · constructor
    · unfold logisticBoundaryVariance
      positivity
    · rw [logisticBoundaryVariance]
      apply (div_le_iff₀ hp).mpr
      nlinarith [sq_nonneg (u ^ 2 - 1)]

/-- The chain rule for the original generator and actual smooth compositions.
Source: arXiv:2604.01978v1, §7, `def:ito.formula` applied to `eq: sde.logistic`. -/
theorem logisticGenerator_comp (φ ψ : ℝ → ℝ)
    (hφ : ContDiff ℝ (⊤ : ℕ∞) φ) (hψ : ContDiff ℝ (⊤ : ℕ∞) ψ) (u : ℝ) :
    logisticGenerator (fun v => φ (ψ v)) u =
      deriv φ (ψ u) * logisticGenerator ψ u +
        (1 - u ^ 2) ^ 2 * deriv (deriv φ) (ψ u) * deriv ψ u ^ 2 := by
  have hφd (v : ℝ) := (hφ.differentiable (by decide) v).hasDerivAt
  have hψd (v : ℝ) := (hψ.differentiable (by decide) v).hasDerivAt
  have hφdd (v : ℝ) := ((contDiff_infty_iff_deriv.mp hφ).2.differentiable (by decide) v).hasDerivAt
  have hψdd (v : ℝ) := ((contDiff_infty_iff_deriv.mp hψ).2.differentiable (by decide) v).hasDerivAt
  have hd : deriv (fun v => φ (ψ v)) = fun v => deriv φ (ψ v) * deriv ψ v :=
    funext (fun v => ((hφd (ψ v)).comp v (hψd v)).deriv)
  have hdd := (((hφdd (ψ u)).comp u (hψd u)).mul (hψdd u)).deriv
  change deriv (fun v => deriv φ (ψ v) * deriv ψ v) u =
    (deriv (deriv φ) (ψ u) * deriv ψ u) * deriv ψ u +
      deriv φ (ψ u) * deriv (deriv ψ) u at hdd
  simp only [logisticGenerator, hd, hdd]
  ring

/-- The coordinate has both signs at genuine states and is globally smooth.
Taking both chain-rule tests equal to it realizes both smoothness premises. -/
example : (1 / 2 : ℝ) ∈ Icc (-1 : ℝ) 1 ∧
    ContDiff ℝ (⊤ : ℕ∞) logisticBoundaryCoordinate ∧
    logisticBoundaryCoordinate (1 / 2) = 3 / 5 ∧
    logisticBoundaryCoordinate 2 = -3 / 5 ∧
    logisticBoundaryRate 2 = -52 / 25 ∧ logisticBoundaryVariance 2 = 128 / 25 := by
  refine ⟨by norm_num, contDiff_logisticBoundaryCoordinate, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [logisticBoundaryCoordinate, logisticBoundaryRate, logisticBoundaryVariance]

end Transformer.Homogenized
