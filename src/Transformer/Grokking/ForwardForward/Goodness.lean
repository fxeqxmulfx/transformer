module

public import Transformer.Grokking.GradientEvidence.ForecastRMS
public import Transformer.Grokking.Geometry.ProbabilityTransfer
public import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# Directional goodness for normalized complex blocks

Source: `review/flat_spectrum.md`, *Forward–Forward reformulation*, LF1–LF2.
Hinton, arXiv:2212.13345 §§2–3.3 measures squared activities before
normalization. Our variant uses corrected label distances after normalization.
The real inner product covers complex vectors through their underlying real
space; no complex multiplication or source architecture is changed.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.ForwardForward

open Finset Geometry Landscape.Neural GradientEvidence

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- LF1: corrected distance goodness; subtracting the class-norm bias
preserves the original dot-product classifier. -/
def goodness (h e : V) : ℝ := (‖e‖ ^ 2 - ‖h - e‖ ^ 2) / 2

/-- LF1: the only additional term is common to every class. -/
theorem goodness_eq_inner (h e : V) :
    goodness h e = inner ℝ h e - ‖h‖ ^ 2 / 2 := by
  unfold goodness
  rw [norm_sub_sq_real]
  ring

/-- LF1: positive/negative goodness differences are actual source margins. -/
theorem goodness_difference (h e f : V) :
    goodness h e - goodness h f = inner ℝ h e - inner ℝ h f := by
  rw [goodness_eq_inner, goodness_eq_inner]
  ring

/-- LF1: corrected distances do not alter any strict class ordering. -/
theorem goodness_order (h e f : V) :
    goodness h e < goodness h f ↔ inner ℝ h e < inner ℝ h f := by
  rw [goodness_eq_inner, goodness_eq_inner]
  exact sub_lt_sub_iff_right _

/-- LF1: putting goodness into ordinary softmax merely recovers CE.
It is not, by itself, a different learning algorithm. -/
theorem goodness_crossEntropy {C : Type*} [Fintype C] [DecidableEq C]
    (h : V) (e : C → V) (t : C) :
    crossEntropy (fun c => goodness h (e c)) t =
      crossEntropy (fun c => inner ℝ h (e c)) t := by
  rw [crossEntropy_eq_log_odds, crossEntropy_eq_log_odds]
  simp_rw [oddsEnvelope, goodness_difference]

/-- LF2: finite-epsilon normalization leaves only a saturating radial
statistic in squared activity. RMS adds the fixed width multiplier. -/
theorem normalized_energy (eps : ℝ) (heps : 0 ≤ eps) (x : V) :
    ‖softNormalize eps x‖ ^ 2 = ‖x‖ ^ 2 / (‖x‖ ^ 2 + eps) := by
  have hs : softNorm eps x ^ 2 = ‖x‖ ^ 2 + eps := Real.sq_sqrt (by positivity)
  rw [softNormalize, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, inv_pow, hs]
  ring

example := normalized_energy 1 (by norm_num) (2 : ℝ)

/-- LF2: with zero epsilon every nonzero normalized vector has identical
energy, so this post-normalization statistic carries no class direction. -/
theorem normalized_energy_zero_eps (x : V) (hx : x ≠ 0) :
    ‖softNormalize 0 x‖ ^ 2 = 1 := by
  rw [normalized_energy 0 (by norm_num), add_zero]
  exact div_self (pow_ne_zero _ (norm_ne_zero_iff.mpr hx))

example := normalized_energy_zero_eps (2 : ℝ) (by norm_num)

/-- LF2: the residual energy sensitivity is epsilon divided by a squared
denominator. Its small value is not a new angular learning signal. -/
theorem normalized_energy_derivative (eps q : ℝ) (h : q + eps ≠ 0) :
    HasDerivAt (fun r : ℝ => r / (r + eps)) (eps / (q + eps) ^ 2) q := by
  have hd := (hasDerivAt_id q).fun_div ((hasDerivAt_id q).add_const eps) h
  simpa only [id_eq, one_mul, mul_one, add_sub_cancel_left] using hd

example := normalized_energy_derivative 1 4 (by norm_num)

/-- LF1–LF2: equal normalized activity energies can still have opposite
directional preferences. This excludes a constant-score reformulation. -/
theorem equal_energy_opposite_preferences :
    ‖(1 : ℝ)‖ ^ 2 = ‖(-1 : ℝ)‖ ^ 2 ∧
    goodness (1 : ℝ) 1 - goodness (1 : ℝ) (-1) = 2 ∧
    goodness (-1 : ℝ) 1 - goodness (-1 : ℝ) (-1) = -2 := by
  norm_num [goodness, Real.norm_eq_abs]

end Transformer.Grokking.ForwardForward
