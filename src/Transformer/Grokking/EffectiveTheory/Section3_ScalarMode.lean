import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Tactic

/-!
# Exact scalar decay, its time scale and its limits

Source: Liu et al., arXiv:2205.10343v2, section 3.2, paragraph "Time
towards the linear structure", the exponential eigenmode formula and
the inverse-rate characteristic time. The raw normalized embedding flow
is not assumed linear: its counterexample is in `Section3_Conservation`.

Here the scalar linear ODE is explicit. Its exponential solution is proved
unique, not prescribed as the behavior of a trained transformer. The next
module derives this ODE for relative modes of the actual quotient flow.
A threshold crossing time still depends on amplitude and threshold.
-/

namespace Transformer.Grokking.EffectiveTheory

/-- Candidate scalar mode with initial amplitude `a` and coefficient `rate`. Source:
arXiv:2205.10343v2, section 3.2, exponential mode formula. -/
noncomputable def scalarMode (a rate t : ℝ) : ℝ := a * Real.exp (-rate * t)

/-- The candidate actually solves the stated scalar negative-rate ODE.
Source: arXiv:2205.10343v2, section 3.2, eigenmode evolution. -/
theorem scalarMode_deriv (a rate t : ℝ) :
    HasDerivAt (scalarMode a rate) (-rate * scalarMode a rate t) t := by
  have h := ((Real.hasDerivAt_exp (-rate * t)).comp t
    ((hasDerivAt_id t).const_mul (-rate))).const_mul a
  convert h using 1
  · rfl
  · unfold scalarMode
    ring

/-- Initial amplitude is a, independently of the rate. Source:
arXiv:2205.10343v2, section 3.2, eigenmode initial condition. -/
theorem scalarMode_initial (a rate : ℝ) : scalarMode a rate 0 = a := by
  unfold scalarMode
  norm_num

/-- Every differentiable solution of the actual scalar ODE is exponential.
Source: arXiv:2205.10343v2, section 3.2, the mode formula; the ODE premise
is required and is not inferred from a normalized loss being a quotient. -/
theorem scalar_ode_solution (f : ℝ → ℝ) (rate : ℝ)
    (hf : ∀ t, HasDerivAt f (-rate * f t) t) (t : ℝ) :
    f t = f 0 * Real.exp (-rate * t) := by
  have hd : ∀ s, HasDerivAt (fun u => f u * Real.exp (rate * u)) 0 s := by
    intro s
    have he := (Real.hasDerivAt_exp (rate * s)).comp s
      ((hasDerivAt_id s).const_mul rate)
    convert (hf s).mul he using 1
    · rfl
    · dsimp
      ring
  have hc : f t * Real.exp (rate * t) = f 0 := by
    have h := is_const_of_deriv_eq_zero (fun s => (hd s).differentiableAt)
      (fun s => (hd s).deriv) t 0
    simpa using h
  calc
    f t = (f t * Real.exp (rate * t)) * Real.exp (-rate * t) := by
      rw [mul_assoc, ← Real.exp_add]
      have he : rate * t + -rate * t = 0 := by ring
      rw [he, Real.exp_zero]
      ring
    _ = f 0 * Real.exp (-rate * t) := by rw [hc]

example : ∀ t : ℝ, HasDerivAt (scalarMode 1 1) (-(1 : ℝ) * scalarMode 1 1 t) t := by
  intro t
  exact scalarMode_deriv 1 1 t

/-- A positive mode with a positive rate decreases strictly. Source:
arXiv:2205.10343v2, section 3.2, interpretation of a positive eigenvalue
as a decay rate; neither positivity condition may be silently omitted. -/
theorem scalarMode_strictAnti (a rate : ℝ) (ha : 0 < a) (hrate : 0 < rate) :
    StrictAnti (scalarMode a rate) := by
  intro s t hst
  unfold scalarMode
  apply mul_lt_mul_of_pos_left _ ha
  apply Real.exp_lt_exp.mpr
  have hm := mul_lt_mul_of_pos_left hst hrate
  linarith

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 := by norm_num

/-- Threshold crossing is determined by the actual scalar mode. Source:
arXiv:2205.10343v2, section 3.2, inverse-rate time scale. The threshold
here is explicitly the value at T, not a universal accuracy criterion. -/
theorem scalarMode_threshold_iff (a rate T t : ℝ) (ha : 0 < a) (hrate : 0 < rate) :
    scalarMode a rate t < scalarMode a rate T ↔ T < t := by
  constructor
  · intro h
    have he := lt_of_mul_lt_mul_left h ha.le
    have hm := Real.exp_lt_exp.mp he
    have hp : rate * T < rate * t := by linarith
    exact lt_of_mul_lt_mul_left hp hrate.le
  · intro h
    exact scalarMode_strictAnti a rate ha hrate h

example : (0 : ℝ) < 2 ∧ (0 : ℝ) < 3 := by norm_num

/-- The inverse rate gives a factor exp(-1), not a complete grokking
time without amplitude and threshold information. Source:
arXiv:2205.10343v2, section 3.2, the characteristic time `1 / rate`.
Its algebraic identity needs rate ≠ 0; decay interpretation needs rate > 0. -/
theorem scalarMode_characteristic_time (a rate : ℝ) (hrate : rate ≠ 0) :
    scalarMode a rate (1 / rate) = a * Real.exp (-1) := by
  unfold scalarMode
  have ht : -rate * (1 / rate) = -1 := by field_simp
  rw [ht]

example : (2 : ℝ) ≠ 0 := by norm_num

/-- A zero rate has no temporal progress. Source: arXiv:2205.10343v2,
section 3.2, zero eigenmodes. It cannot support a positive-rate estimate. -/
theorem scalarMode_zero_rate (a t : ℝ) : scalarMode a 0 t = a := by
  unfold scalarMode
  norm_num

/-- Squared mode energy decays with twice the scalar rate. Source:
arXiv:2205.10343v2, section 3.2, its quadratic loss and exponential modes.
This identity does not identify test accuracy with mode energy. -/
theorem scalarMode_energy (a rate t : ℝ) :
    scalarMode a rate t ^ 2 = a ^ 2 * Real.exp (-2 * rate * t) := by
  have he : Real.exp (-rate * t) ^ 2 = Real.exp (-2 * rate * t) := by
    calc
      Real.exp (-rate * t) ^ 2 = Real.exp (-rate * t) * Real.exp (-rate * t) := by ring
      _ = Real.exp (-rate * t + -rate * t) := by rw [Real.exp_add]
      _ = Real.exp (-2 * rate * t) := by congr 1; ring
  unfold scalarMode
  rw [← he]
  ring

/-- Positive initial amplitude never vanishes at finite time, for any
real rate. Source: arXiv:2205.10343v2, section 3.2, mode formula. -/
theorem scalarMode_positive (a rate t : ℝ) (ha : 0 < a) :
    0 < scalarMode a rate t := by
  unfold scalarMode
  exact mul_pos ha (Real.exp_pos _)

example : (0 : ℝ) < 1 := by norm_num

/-- Fixed time increments have the same multiplicative decay factor.
Source: arXiv:2205.10343v2, section 3.2, exponential mode dynamics. -/
theorem scalarMode_time_shift (a rate s t : ℝ) :
    scalarMode a rate (s + t) = scalarMode a rate s * Real.exp (-rate * t) := by
  unfold scalarMode
  have h : -rate * (s + t) = -rate * s + -rate * t := by ring
  rw [h, Real.exp_add]
  ring

end Transformer.Grokking.EffectiveTheory
