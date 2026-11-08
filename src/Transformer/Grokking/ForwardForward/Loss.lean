module

public import Transformer.Grokking.ForwardForward.Goodness
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Pairwise local loss and its actual derivatives

Source: `review/flat_spectrum.md`, *Forward–Forward reformulation*, LF3–LF4.
This modifies Hinton's arXiv:2212.13345 §§2–3.3 objective to paired
directional margins. A local derivative is taken with the previous feature
held fixed; a concrete two-layer witness distinguishes it from global credit.
There is no theorem here that local optimization learns the modular task.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.ForwardForward

open Finset

/-- LF3: smooth positive part in each positive/negative margin pair. -/
def softplus (a : ℝ) : ℝ := Real.log (1 + Real.exp a)

/-- LF3: actual derivative of softplus, used as a pair's error weight. -/
def logistic (a : ℝ) : ℝ := Real.exp a / (1 + Real.exp a)

/-- LF3: differentiate the objective rather than postulating a direction. -/
theorem softplus_derivative (a : ℝ) : HasDerivAt softplus (logistic a) a := by
  exact ((Real.hasDerivAt_exp a).const_add 1).log (by positivity)

/-- LF3: every wrong-class contribution is nonnegative. -/
theorem softplus_nonneg (a : ℝ) : 0 ≤ softplus a := by
  unfold softplus
  exact Real.log_nonneg (by have := Real.exp_pos a; linarith)

/-- LF3: larger wrong-minus-right margins incur larger penalties. -/
theorem softplus_strictMono : StrictMono softplus := by
  intro a b hab
  exact Real.log_lt_log (by positivity) (by linarith [Real.exp_lt_exp.mpr hab])

variable {C : Type*} [Fintype C] [DecidableEq C]

/-- LF3: sum over every negative label. The native implementation divides
by C-1; keeping the sum exposes the exact per-query correctness threshold. -/
def pairLoss (margin : ℝ) (z : C → ℝ) (t : C) : ℝ :=
  ∑ c ∈ univ.erase t, softplus (margin + z c - z t)

/-- LF3: adding a common goodness offset changes neither loss nor scores. -/
theorem pairLoss_shift (margin : ℝ) (z : C → ℝ) (t : C) (a : ℝ) :
    pairLoss margin (fun c => z c + a) t = pairLoss margin z t := by
  unfold pairLoss
  apply sum_congr rfl
  intro c hc
  congr 1
  ring

/-- LF1/LF3: corrected distance and dot-product objectives agree exactly. -/
theorem pairLoss_goodness {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (margin : ℝ) (h : V) (e : C → V) (t : C) :
    pairLoss margin (fun c => goodness h (e c)) t =
      pairLoss margin (fun c => inner ℝ h (e c)) t := by
  simp_rw [goodness_eq_inner, sub_eq_add_neg]
  exact pairLoss_shift margin _ t _

/-- LF3: the true directional derivative includes all competitors and
their distinct logistic weights; no direction is supplied by backprop. -/
theorem pairLoss_directional (margin : ℝ) (z v : C → ℝ) (t : C) :
    HasDerivAt (fun a => pairLoss margin (fun c => z c + a * v c) t)
      (∑ c ∈ univ.erase t, logistic (margin + z c - z t) * (v c - v t)) 0 := by
  have hd (c : C) : HasDerivAt
      (fun a : ℝ => margin + (z c + a * v c) - (z t + a * v t)) (v c - v t) 0 := by
    simpa only [one_mul, id_eq] using
      ((((hasDerivAt_id (0 : ℝ)).mul_const (v c)).const_add (z c)).const_add margin).fun_sub
        (((hasDerivAt_id (0 : ℝ)).mul_const (v t)).const_add (z t))
  have hs := HasDerivAt.sum (u := univ.erase t) (fun c _ =>
    (softplus_derivative (margin + (z c + 0 * v c) - (z t + 0 * v t))).comp (0 : ℝ) (hd c))
  simp only [zero_mul, add_zero] at hs
  convert hs using 1
  funext a
  simp only [pairLoss, Finset.sum_apply, Function.comp_apply]

/-- LF3: each wrong class is controlled by the sum, with no averaging
of margins and no reference to the unknown labels of other inputs. -/
theorem pair_term_le (margin : ℝ) (z : C → ℝ) (t c : C) (hc : c ≠ t) :
    softplus (margin + z c - z t) ≤ pairLoss margin z t := by
  unfold pairLoss
  exact single_le_sum (fun j _ => softplus_nonneg (margin + z j - z t))
    (mem_erase.mpr ⟨hc, mem_univ c⟩)

example := pair_term_le 1 (fun c : Bool => if c then 2 else 0) true false (by decide)

/-- LF3: a strict sum-loss threshold guarantees every margin is positive.
For the native mean loss, the threshold is softplus(margin)/(C-1), not 90%. -/
theorem winner_of_pairLoss_lt (margin : ℝ) (z : C → ℝ) (t : C)
    (hloss : pairLoss margin z t < softplus margin) :
    ∀ c, c ≠ t → z c < z t := by
  intro c hc
  have h := lt_of_le_of_lt (pair_term_le margin z t c hc) hloss
  have harg := softplus_strictMono.lt_iff_lt.mp h
  linarith

example : ∀ c : Bool, c ≠ true → (if c then (2 : ℝ) else 0) < (2 : ℝ) := by
  apply winner_of_pairLoss_lt 1 (fun c : Bool => if c then 2 else 0) true
  norm_num [pairLoss]
  exact softplus_strictMono (by norm_num)

omit [Fintype C] [DecidableEq C] in
/-- LF4: local linear-block learning differentiates its own weight with
the saved incoming feature fixed. This still requires local arithmetic. -/
theorem local_linear_derivative (margin input weight : ℝ) :
    HasDerivAt (fun w => softplus (margin - w * input))
      (-input * logistic (margin - weight * input)) weight := by
  have h := (softplus_derivative (margin - weight * input)).comp weight
    (((hasDerivAt_id weight).mul_const input).const_sub margin)
  convert h using 1
  · funext w; rfl
  · ring

omit [Fintype C] [DecidableEq C] in
/-- LF4: the terminal loss of two layers carries the downstream weight.
It is not the first layer's local objective. -/
theorem terminal_first_derivative (margin input first second : ℝ) :
    HasDerivAt (fun w => softplus (margin - second * (w * input)))
      (-(second * input) * logistic (margin - second * (first * input))) first := by
  have h := (softplus_derivative (margin - second * (first * input))).comp first
    ((((hasDerivAt_id first).mul_const input).const_mul second).const_sub margin)
  convert h using 1
  · funext w; rfl
  · ring

/-- LF4: at the same nontrivial two-layer state the local and global
derivatives differ. Calling a loss Forward–Forward does not change this. -/
theorem local_global_derivative_witness :
    deriv (fun w : ℝ => softplus (-w)) 0 = -1 / 2 ∧
    deriv (fun w : ℝ => softplus (-(2 * w))) 0 = -1 := by
  have hl := (local_linear_derivative 0 1 0).deriv
  have hg := (terminal_first_derivative 0 1 0 2).deriv
  norm_num [logistic] at hl hg
  constructor
  · linarith [hl]
  · exact hg

end Transformer.Grokking.ForwardForward
