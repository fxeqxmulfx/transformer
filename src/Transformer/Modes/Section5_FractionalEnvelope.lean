import Transformer.Modes.Section5_PhaseSummation

/-!
# Fractional decay for every positive bandwidth

The source's inverse square root estimate in §5.5 of arXiv:2412.09080v3
fails for `β > 2`. Its use of a fixed spatial cutoff leaves a tail that
does not decay with frequency. This module gives a different corrected
estimate that applies to every `β > 0`, with a smaller positive exponent.
The already proved `uniform_decay` keeps its original corrected range.

On each unit interval, the absolute integral is bounded by the shifted
Gaussian amplitude envelope `E`. The stationary phase estimate also
bounds it by `E * B * exp (βR²/4) / sqrt |ρ|`, where `B` is a cubic
polynomial envelope. Interpolating these two bounds with a real power
`0 < p ≤ 1` gives the rate `|ρ|^(-p/2)` and the Gaussian loss `βp`.

The interpolation retains the actual integral and the actual Gaussian
weight. The polynomial's fractional power is bounded by `1 + B`;
completing the square leaves the summable envelope
`(1 + |m|)³ exp (-(2 - βp)m²/8)` whenever `βp < 2`.
In particular `p = 1 / (β + 1)` meets both requirements for all `β > 0`.
Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay` and its proof.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- Interpolate a nonoscillatory bound and an inverse square root bound.
The constant's fractional power is absorbed into `1 + B`.
Source: arXiv:2412.09080v3, §5.5, repairing the fixed-cutoff tail. -/
theorem fractional_interpolation_bound {u E B ρ p a : ℝ}
    (hu : 0 ≤ u) (hE : 0 < E) (hB : 0 ≤ B) (hρ : 0 < ρ)
    (hp : 0 ≤ p) (hp1 : p ≤ 1) (huE : u ≤ E)
    (huosc : u ≤ E * (B * Real.exp a / Real.sqrt ρ)) :
    u ≤ E * (1 + B) * Real.exp (a * p) / ρ ^ (p / 2) := by
  have hr0 : 0 ≤ u / E := div_nonneg hu hE.le
  have hr1 : u / E ≤ 1 := (div_le_iff₀ hE).mpr (by simpa using huE)
  have hrF : u / E ≤ B * Real.exp a / Real.sqrt ρ :=
    (div_le_iff₀ hE).mpr (by simpa only [mul_comm] using huosc)
  have hr := (Real.self_le_rpow_of_le_one hr0 hr1 hp1).trans
    (Real.rpow_le_rpow hr0 hrF hp)
  have hBp : B ^ p ≤ 1 + B := by
    by_cases hB1 : B ≤ 1
    · exact (Real.rpow_le_one hB hB1 hp).trans (by linarith)
    · have h := Real.rpow_le_rpow_of_exponent_le (le_of_not_ge hB1) hp1
      rw [Real.rpow_one] at h
      linarith
  have heq : (B * Real.exp a / Real.sqrt ρ) ^ p =
      B ^ p * Real.exp (a * p) / ρ ^ (p / 2) := by
    rw [Real.div_rpow (by positivity) (Real.sqrt_nonneg _) p,
      Real.mul_rpow hB (Real.exp_pos _).le, ← Real.exp_mul,
      ← Real.rpow_div_two_eq_sqrt p hρ.le]
  have h := (div_le_iff₀ hE).mp hr
  calc u ≤ E * (B * Real.exp a / Real.sqrt ρ) ^ p := by simpa only [mul_comm] using h
    _ = E * (B ^ p * Real.exp (a * p) / ρ ^ (p / 2)) := by rw [heq]
    _ ≤ E * ((1 + B) * Real.exp (a * p) / ρ ^ (p / 2)) :=
      mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right
        (mul_le_mul_of_nonneg_right hBp (Real.exp_pos _).le) (by positivity)) hE.le
    _ = _ := by ring

example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 1 * (1 * Real.exp 0 / Real.sqrt 1) := by norm_num

/-- A unit interval's absolute integral is bounded by the genuine shifted
Gaussian envelope, independently of frequency and bandwidth.
Source: arXiv:2412.09080v3, §5.5, the Gaussian tail estimate. -/
theorem gaussian_phase_unit_nonoscillatory_bound
    (β ρ : ℝ) (θ : ℝ × ℝ) (m t : ℝ) :
    ‖∫ u in m..m + 1, gaussianPhaseIntegrand β ρ θ t u‖ ≤
      Real.exp (-(m ^ 2) / 2 + (1 + |t|) * |m|) := by
  have h := intervalIntegral.norm_integral_le_of_norm_le_const
    (a := m) (b := m + 1) (f := gaussianPhaseIntegrand β ρ θ t)
    (C := Real.exp (-(m ^ 2) / 2 + (1 + |t|) * |m|)) (fun u hu => by
      rw [norm_gaussianPhaseIntegrand]
      exact gaussianPhaseAmplitude_unit_bound (by
        simp only [Set.uIoc_of_le (show m ≤ m + 1 by linarith)] at hu
        exact ⟨hu.1.le, hu.2⟩) t)
  simpa using h

/-- Every unit interval has a summable envelope with frequency exponent
`p/2` when `βp < 2`. The source prints exponent `1/2` for all `β > 0`;
this explicit bandwidth-dependent exponent repairs that false claim.
The constant also retains the dependence on fixed `t`.
Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`. -/
theorem gaussian_phase_fractional_local_envelope {β p : ℝ}
    (hβ : 0 < β) (hp : 0 < p) (hp1 : p ≤ 1) (hβp : β * p < 2) (t : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ρ : ℝ, ρ ≠ 0 → ∀ θ : ℝ × ℝ, max |θ.1| |θ.2| = 1 →
      ∀ m : ℝ, ‖∫ u in m..m + 1, gaussianPhaseIntegrand β ρ θ t u‖ ≤
        (C / |ρ| ^ (p / 2)) *
          ((1 + |m|) ^ 3 * Real.exp (-((2 - β * p) / 8) * m ^ 2)) := by
  let D := Real.sqrt (1 + 7 * β + 7 * β ^ 2 + β ^ 3) / Real.sqrt (3 * β)
  let L := β * p / 4 + 2 * (1 + |t| + β * p / 2) ^ 2 / (2 - β * p)
  let C := (1 + 640 * D * (2 + |t|)) * Real.exp L
  have hD : 0 < D := by
    dsimp [D]
    exact div_pos (Real.sqrt_pos.mpr (by positivity)) (Real.sqrt_pos.mpr (by positivity))
  refine ⟨C, by dsimp [C]; positivity, ?_⟩
  intro ρ hρ θ hθ m
  let R := |m| + 1
  let E := Real.exp (-(m ^ 2) / 2 + (1 + |t|) * |m|)
  let B := 160 * D * (2 + |m|) ^ 2 * (2 + |m| + |t|)
  have hR : 0 ≤ R := by dsimp [R]; positivity
  have hE : 0 < E := Real.exp_pos _
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hsharp : ‖∫ u in m..m + 1, gaussianPhaseIntegrand β ρ θ t u‖ ≤
      E * (B * Real.exp (β * R ^ 2 / 4) / Real.sqrt |ρ|) := by
    have h := gaussian_phase_weighted_unit_bound hρ hβ hθ m t
    have hfloor := gaussianPhaseFloor_inv_sqrt_bound hβ hR
    have heq : (160 / Real.sqrt (|ρ| * gaussianPhaseFloor β R)) *
        ((2 + |m| + |t|) * E) =
        (160 / Real.sqrt |ρ|) * ((1 / Real.sqrt (gaussianPhaseFloor β R)) *
          (2 + |m| + |t|) * E) := by
      rw [Real.sqrt_mul (abs_nonneg ρ)]
      ring
    change _ ≤ (160 / Real.sqrt (|ρ| * gaussianPhaseFloor β R)) *
      ((2 + |m| + |t|) * E) at h
    rw [heq] at h
    calc _ ≤ (160 / Real.sqrt |ρ|) * ((1 / Real.sqrt (gaussianPhaseFloor β R)) *
          (2 + |m| + |t|) * E) := h
      _ ≤ (160 / Real.sqrt |ρ|) * ((D * (1 + R) ^ 2 * Real.exp (β * R ^ 2 / 4)) *
          (2 + |m| + |t|) * E) := by gcongr
      _ = _ := by dsimp [B, R]; ring
  have hinterp := fractional_interpolation_bound (norm_nonneg _) hE hB
    (abs_pos.mpr hρ) hp.le hp1 (gaussian_phase_unit_nonoscillatory_bound β ρ θ m t) hsharp
  have hpoly : 1 + B ≤ (1 + 640 * D * (2 + |t|)) * (1 + |m|) ^ 3 := by
    have h := mul_le_mul_of_nonneg_left (gaussian_phase_polynomial_factor_bound m t)
      (show 0 ≤ 160 * D by positivity)
    have hbase : 1 ≤ (1 + |m|) ^ 3 := by nlinarith [abs_nonneg m, sq_nonneg |m|]
    dsimp [B]
    nlinarith
  have hexp : E * Real.exp ((β * R ^ 2 / 4) * p) ≤
      Real.exp L * Real.exp (-((2 - β * p) / 8) * m ^ 2) := by
    rw [← Real.exp_add, ← Real.exp_add]
    apply Real.exp_le_exp.mpr
    dsimp [E, R, L]
    convert gaussian_phase_exponent_bound hβp m t using 1 <;> ring
  calc _ ≤ E * (1 + B) * Real.exp ((β * R ^ 2 / 4) * p) / |ρ| ^ (p / 2) := hinterp
    _ = (1 + B) * (E * Real.exp ((β * R ^ 2 / 4) * p)) / |ρ| ^ (p / 2) := by ring
    _ ≤ ((1 + 640 * D * (2 + |t|)) * (1 + |m|) ^ 3) *
        (Real.exp L * Real.exp (-((2 - β * p) / 8) * m ^ 2)) / |ρ| ^ (p / 2) :=
      div_le_div_of_nonneg_right (mul_le_mul hpoly hexp (by positivity) (by positivity))
        (by positivity)
    _ = _ := by dsimp [C]; ring

example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 ∧
    (3 : ℝ) * (1 / 4) < 2 ∧ (1 : ℝ) ≠ 0 ∧ max |(1 : ℝ)| |0| = 1 := by norm_num

end Transformer.Modes
