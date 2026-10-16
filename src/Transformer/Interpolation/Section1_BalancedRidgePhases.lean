/-
# Actual original weak phases with balanced ridge clock costs

ArXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf`,
and §5, the parameter norm estimate for `thm: targets.atoms`.
Balanced original ridge parameters have the same full vector field
as scaling just the old output vector. Their actual probability law
therefore agrees with the original law at the scaled clock, by the
proved original weak uniqueness theorem.

This identity holds for every original probability, every real
observation time and every nonnegative clock factor, including zero.
It is retained by the actual autonomous phase representation.
At a positive prescribed physical horizon, a nonnegative intrinsic
clock is realized by one actual constant piece whose complete
parameter norm grows as sqrt(clock/horizon).
-/

import Transformer.Interpolation.Section1_RidgeParameterBounds
import Transformer.Interpolation.Section3_RidgeClassProgress
import Transformer.Interpolation.Section4_GatedTransfers

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Balanced original ridge parameters have exactly the original
probability law at the scaled elapsed clock. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, redistribution of ReLU clock scaling. -/
theorem ridgeLaw_balanced_clock (i : Fin d) (p u : EucSpace d) (b κ : ℝ) (hκ : 0 ≤ κ)
    (ρ : Perspective.ProbSphere d) (t : ℝ) :
    ridgeLaw (sqrt κ • p) (sqrt κ • u) (sqrt κ * b) ρ t = ridgeLaw p u b ρ (κ * t) := by
  let μ := ridgeLaw (κ • p) u b ρ
  have hfield (ν : Perspective.ProbSphere d) (s : ℝ) (x : EucSpace d) :
      fullVF d (fun _ => balancedRidgeParams i p u b κ) ν s x =
        fullVF d (fun _ => reluRidgeParams i (κ • p) u b) ν s x := by
    rw [fullVF_balancedRidgeParams i p u b κ hκ, fullVF_reluRidgeParams,
      fullVF_reluRidgeParams, projection_field_smul]
    module
  have hμ : cauchyPB d (fun _ => balancedRidgeParams i p u b κ) μ := by
    intro φ hφ s
    simpa only [hfield] using cauchyPB_ridgeLaw i (κ • p) u b ρ φ hφ s
  have he := cauchyPB_ridge_eq i (sqrt κ • p) (sqrt κ • u) (sqrt κ * b) ρ μ hμ
    (ridgeLaw_zero (κ • p) u b ρ)
  calc
    ridgeLaw (sqrt κ • p) (sqrt κ • u) (sqrt κ * b) ρ t = ridgeLaw (κ • p) u b ρ t :=
      (congrFun he t).symm
    _ = ridgeLaw p u b ρ (κ * t) := ridgeLaw_smul_output κ p u b ρ t

/-- The balanced parameters paired with their actual original ridge
law form the phase used in a complete sequence. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, bounded clock realization. -/
noncomputable def balancedRidgePhase (i : Fin d) (p u : EucSpace d) (b κ : ℝ) : AutonomousPhase d :=
  ridgePhase i (sqrt κ • p) (sqrt κ • u) (sqrt κ * b)

/-- Every balanced phase has a proved actual original weak law,
independently of the sign of the parameter supplied to sqrt.
Source auxiliary: arXiv:2411.04551v3, §1, actual phase validity. -/
theorem isAutonomousPhase_balancedRidge (i : Fin d) (p u : EucSpace d) (b κ : ℝ) :
    IsAutonomousPhase (balancedRidgePhase i p u b κ) :=
  isAutonomousPhase_ridge i _ _ _

/-- Actual balanced ridge phases retain zero scores.
Source auxiliary: arXiv:2411.04551v3, §1 and §5. -/
theorem balancedRidgePhase_B_zero (i : Fin d) (p u : EucSpace d) (b κ : ℝ) :
    (balancedRidgePhase i p u b κ).1.B = 0 := rfl

/-- The complete balanced phase has the exact original scaled clock
on every probability, including zero clock. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, original time scaling. -/
theorem balancedRidgePhase_law (i : Fin d) (p u : EucSpace d) (b κ : ℝ) (hκ : 0 ≤ κ)
    (ρ : Perspective.ProbSphere d) (t : ℝ) :
    (balancedRidgePhase i p u b κ).2 ρ t = ridgeLaw p u b ρ (κ * t) :=
  ridgeLaw_balanced_clock i p u b κ hκ ρ t

/-- At unit intrinsic phase time, the balanced actual phase performs
exactly the requested original clock. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, composition of bounded phases. -/
theorem phaseSequenceLaw_balanced_singleton (i : Fin d) (p u : EucSpace d)
    (b κ : ℝ) (hκ : 0 ≤ κ) (ρ : Perspective.ProbSphere d) :
    phaseSequenceLaw [balancedRidgePhase i p u b κ] ρ = ridgeLaw p u b ρ κ := by
  change (balancedRidgePhase i p u b κ).2 ρ 1 = _
  rw [balancedRidgePhase_law i p u b κ hκ, mul_one]

/-- Any nonnegative intrinsic ridge clock is realized at a positive
physical horizon with its actual square-root parameter cost. Global
original weak solutions exist and every weak endpoint is identified.
Source auxiliary: arXiv:2411.04551v3, §1 and §5, bounded prescribed-time controls. -/
theorem exists_balanced_ridge_controls (i : Fin d) (p : SSphere d) (u : EucSpace d)
    (b τ T : ℝ) (hτ : 0 ≤ τ) (hT : 0 < T) :
    ∃ θ : TimeParams d, PiecewiseConstant d θ T 1 ∧
      (∀ s, (θ s).V = 0 ∧ (θ s).B = 0) ∧
      (∀ s, ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖
        ≤ sqrt (τ / T) * (1 + ‖u‖ + |b|)) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ →
        μ T = ridgeLaw (p : EucSpace d) u b (μ 0) τ := by
  let κ := τ / T
  have hκ : 0 ≤ κ := div_nonneg hτ hT.le
  let θ : TimeParams d := fun _ => balancedRidgeParams i (p : EucSpace d) u b κ
  refine ⟨θ, piecewiseConstant_constant _ T hT.le, (fun _ => ⟨rfl, rfl⟩),
    (fun _ => norm_balancedRidgeParams_unit_output_le i p u b κ), ?_, ?_⟩
  · intro ρ
    exact exists_cauchyPB_reluRidgeParams i (sqrt κ • (p : EucSpace d)) (sqrt κ • u) (sqrt κ * b) ρ
  · intro μ hμ
    have he := cauchyPB_ridge_transport i (sqrt κ • (p : EucSpace d)) (sqrt κ • u) (sqrt κ * b) μ hμ T
    rw [ridgeLaw_balanced_clock i (p : EucSpace d) u b κ hκ] at he
    have hclock : κ * T = τ := div_mul_cancel₀ τ hT.ne'
    rwa [hclock] at he

/-- A genuine exact gated transfer admits a square-root physical
clock cost with constants chosen before the horizon. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, bounded movement of selected atoms. -/
theorem exists_balanced_gated_transfer_controls (i : Fin d) (p q : SSphere d)
    (hpq : p ≠ q) (hanti : q ≠ antipode d p) :
    ∃ τ c : ℝ, 0 < τ ∧ 0 < c ∧ ∀ H : ℝ, 0 < H →
      ∃ θ : TimeParams d, PiecewiseConstant d θ H 1 ∧
        (∀ s, (θ s).V = 0 ∧ (θ s).B = 0) ∧
        (∀ s, ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖
          ≤ sqrt (τ / H) * (1 + ‖(p : EucSpace d) + (q : EucSpace d)‖ + c)) ∧
        (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = ρ ∧ cauchyPB d θ μ) ∧
        ∀ μ : ℝ → Perspective.ProbSphere d,
          μ 0 = Perspective.diracProb d p → cauchyPB d θ μ → μ H = Perspective.diracProb d q := by
  obtain ⟨a, c, τ, hc, hτ, hend, _⟩ := exists_exact_gated_transfer p q hpq hanti
  refine ⟨τ, c, hτ, hc, ?_⟩
  intro H hH
  obtain ⟨θ, hpc, hVB, hnorm, hex, hfinal⟩ := exists_balanced_ridge_controls i a
    (-((p : EucSpace d) + (q : EucSpace d))) (-c) τ H hτ.le hH
  refine ⟨θ, hpc, hVB, ?_, hex, ?_⟩
  · intro s
    simpa only [norm_neg, abs_neg, abs_of_pos hc] using hnorm s
  · intro μ h0 hμ
    rw [hfinal μ hμ, h0, ridgeLaw_diracProb, hend]

/-- A nonzero original output, active transverse input atom, positive
clock and horizon, and an actual original weak law jointly realize
all balancing and prescribed-time hypotheses. -/
example : ∃ (p : SSphere 2) (u : EucSpace 2) (ρ : Perspective.ProbSphere 2)
    (μ : ℝ → Perspective.ProbSphere 2),
    (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ μ 0 = ρ ∧
      cauchyPB 2 (fun _ => balancedRidgeParams 0 p u 2 2) μ := by
  let p := basePoint 1
  let u : EucSpace 2 := PiLp.single 2 (1 : Fin 2) 1
  let x : SSphere 2 := ⟨u, by simp [u]⟩
  let ρ := Perspective.diracProb 2 x
  obtain ⟨μ, h0, hμ⟩ := exists_cauchyPB_reluRidgeParams (0 : Fin 2)
    (sqrt 2 • (p : EucSpace 2)) (sqrt 2 • u) (sqrt 2 * 2) ρ
  exact ⟨p, u, ρ, μ, by norm_num, one_pos, h0, hμ⟩

/-- A genuine transverse unit pair satisfies both the distinctness
and non-antipodal premises of bounded gated transfer. -/
example : ∃ p q : SSphere 2, p ≠ q ∧ q ≠ antipode 2 p := by
  let p := basePoint 1
  let q : SSphere 2 := ⟨PiLp.single 2 (1 : Fin 2) (1 : ℝ), by simp⟩
  refine ⟨p, q, ?_, ?_⟩
  · intro he
    have hh := congrArg (fun z : SSphere 2 => (z : EucSpace 2) (0 : Fin 2)) he
    norm_num [p, q, basePoint] at hh
  · intro he
    have hh := congrArg (fun z : SSphere 2 => (z : EucSpace 2) (1 : Fin 2)) he
    norm_num [p, q, basePoint, antipode] at hh

end Transformer.Interpolation
