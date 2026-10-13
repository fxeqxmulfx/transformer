import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Tactic

/-!
# Nonvanishing from initial data for a bounded growth coefficient

Source: Liu et al., arXiv:2205.10343v2, section 3.2, the spectral
evolution claim. This is an auxiliary strengthening for the corrected
quotient-flow analysis: a ground component solves `f' = c(t) f`, with
a nonnegative, uniformly bounded coefficient derived from the loss.

The growth equation is an explicit hypothesis. Nonvanishing is a conclusion
derived from initial data, rather than an assumed property of every time.
Two actual monotone quantities, squared amplitude and an integrating-factor
weighted squared amplitude, control the forward and backward directions.
This module alone does not identify a neural trajectory with that equation.
-/

namespace Transformer.Grokking.EffectiveTheory

/-- Squared amplitude with a fixed integrating factor. Source: the
bounded-coefficient correction to arXiv:2205.10343v2, section 3.2. -/
noncomputable def weightedEnergy (f : ℝ → ℝ) (K t : ℝ) : ℝ :=
  f t ^ 2 * Real.exp (-2 * K * t)

/-- Differentiate the actual squared amplitude under the scalar growth
equation. Source correction: arXiv:2205.10343v2, section 3.2. -/
theorem growth_energy_deriv (f c : ℝ → ℝ) (t : ℝ)
    (hf : HasDerivAt f (c t * f t) t) :
    HasDerivAt (fun s => f s ^ 2) (2 * c t * f t ^ 2) t := by
  convert hf.pow 2 using 1
  norm_num
  ring

example : HasDerivAt Real.exp ((1 : ℝ) * Real.exp 0) 0 := by
  simpa using Real.hasDerivAt_exp 0

/-- Squared amplitude cannot decrease when the coefficient is nonnegative.
Source correction: arXiv:2205.10343v2, section 3.2, ground-mode dynamics. -/
theorem growth_energy_monotone (f c : ℝ → ℝ)
    (hf : ∀ t, HasDerivAt f (c t * f t) t) (hc : ∀ t, 0 ≤ c t) :
    Monotone (fun t => f t ^ 2) := by
  have hd := fun t => growth_energy_deriv f c t (hf t)
  apply monotone_of_deriv_nonneg (fun t => (hd t).differentiableAt)
  intro t
  rw [(hd t).deriv]
  have hp := hc t
  positivity

example : ∃ f c : ℝ → ℝ,
    (∀ t, HasDerivAt f (c t * f t) t) ∧ (∀ t, 0 ≤ c t) := by
  refine ⟨Real.exp, fun _ => 1, ?_, ?_⟩
  · intro t
    simpa using Real.hasDerivAt_exp t
  · intro t
    norm_num

/-- Actual integrating-factor derivative; the coefficient bound is not
required for this equality. Source correction: arXiv:2205.10343v2,
section 3.2, the normalized-loss ground-mode equation. -/
theorem weightedEnergy_deriv (f c : ℝ → ℝ) (K t : ℝ)
    (hf : HasDerivAt f (c t * f t) t) :
    HasDerivAt (weightedEnergy f K)
      (2 * (c t - K) * (f t ^ 2 * Real.exp (-2 * K * t))) t := by
  have he := (Real.hasDerivAt_exp (-2 * K * t)).comp t
    ((hasDerivAt_id t).const_mul (-2 * K))
  convert (growth_energy_deriv f c t hf).mul he using 1
  · rfl
  · dsimp
    ring

example : HasDerivAt Real.exp ((1 : ℝ) * Real.exp 2) 2 := by
  simpa using Real.hasDerivAt_exp 2

/-- An upper coefficient bound makes weighted squared amplitude decrease.
Source correction: arXiv:2205.10343v2, section 3.2. This prevents a
nonzero initial ground component from arising out of a finite-time zero. -/
theorem weightedEnergy_antitone (f c : ℝ → ℝ) (K : ℝ)
    (hf : ∀ t, HasDerivAt f (c t * f t) t) (hc : ∀ t, c t ≤ K) :
    Antitone (weightedEnergy f K) := by
  have hd := fun t => weightedEnergy_deriv f c K t (hf t)
  apply antitone_of_deriv_nonpos (fun t => (hd t).differentiableAt)
  intro t
  rw [(hd t).deriv]
  apply mul_nonpos_of_nonpos_of_nonneg
  · have hp := hc t
    linarith
  · positivity

example : ∃ (f c : ℝ → ℝ) (K : ℝ),
    (∀ t, HasDerivAt f (c t * f t) t) ∧ (∀ t, c t ≤ K) := by
  refine ⟨Real.exp, fun _ => 1, 1, ?_, ?_⟩
  · intro t
    simpa using Real.hasDerivAt_exp t
  · intro t
    norm_num

/-- Quantitative finite-interval bound under the actual growth equation.
Source correction: arXiv:2205.10343v2, section 3.2. The coefficient may
vary with time; a uniform upper bound controls the squared amplitude. -/
theorem growth_interval_square_bound (f c : ℝ → ℝ) (K s t : ℝ)
    (hf : ∀ u, HasDerivAt f (c u * f u) u) (hc : ∀ u, c u ≤ K) (hst : s ≤ t) :
    f t ^ 2 ≤ f s ^ 2 * Real.exp (2 * K * (t - s)) := by
  have h := weightedEnergy_antitone f c K hf hc hst
  have hm := mul_le_mul_of_nonneg_right h (Real.exp_pos (2 * K * t)).le
  have hl : weightedEnergy f K t * Real.exp (2 * K * t) = f t ^ 2 := by
    unfold weightedEnergy
    rw [mul_assoc, ← Real.exp_add]
    have he : -2 * K * t + 2 * K * t = 0 := by ring
    rw [he, Real.exp_zero]
    ring
  have hr : weightedEnergy f K s * Real.exp (2 * K * t) =
      f s ^ 2 * Real.exp (2 * K * (t - s)) := by
    unfold weightedEnergy
    rw [mul_assoc, ← Real.exp_add]
    have he : -2 * K * s + 2 * K * t = 2 * K * (t - s) := by ring
    rw [he]
  rw [hl, hr] at hm
  exact hm

example : (0 : ℝ) ≤ 1 ∧ ∃ (f c : ℝ → ℝ) (K : ℝ),
    (∀ t, HasDerivAt f (c t * f t) t) ∧ (∀ t, c t ≤ K) := by
  refine ⟨by norm_num, Real.exp, fun _ => 1, 1, ?_, ?_⟩
  · intro t
    simpa using Real.hasDerivAt_exp t
  · intro t
    norm_num

/-- Nonzero initial amplitude stays nonzero at every finite time. Source:
arXiv:2205.10343v2, section 3.2, strengthening the relative-mode proof
with a proved bounded-coefficient premise instead of assumed nonvanishing. -/
theorem growth_nonvanishing_from_initial (f c : ℝ → ℝ) (K : ℝ)
    (hf : ∀ t, HasDerivAt f (c t * f t) t)
    (hc : ∀ t, 0 ≤ c t ∧ c t ≤ K) (h0 : f 0 ≠ 0) :
    ∀ t, f t ≠ 0 := by
  have hm := growth_energy_monotone f c hf (fun t => (hc t).1)
  have ha := weightedEnergy_antitone f c K hf (fun t => (hc t).2)
  have hp := sq_pos_of_ne_zero h0
  intro t ht
  by_cases htime : 0 ≤ t
  · have h := hm htime
    dsimp at h
    rw [ht] at h
    nlinarith
  · have htime' : t ≤ 0 := by linarith
    have h := ha htime'
    unfold weightedEnergy at h
    norm_num [ht] at h
    nlinarith

example : ∃ (f c : ℝ → ℝ) (K : ℝ),
    (∀ t, HasDerivAt f (c t * f t) t) ∧
    (∀ t, 0 ≤ c t ∧ c t ≤ K) ∧ f 0 ≠ 0 := by
  refine ⟨Real.exp, fun _ => 1, 1, ?_, ?_, ?_⟩
  · intro t
    simpa using Real.hasDerivAt_exp t
  · intro t
    norm_num
  · norm_num

end Transformer.Grokking.EffectiveTheory
