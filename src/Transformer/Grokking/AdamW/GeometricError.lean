import Transformer.Grokking.AdamW.RetainedInputDissipation
import Mathlib.Topology.Order.MonotoneConvergence

/-!
# Absorbing geometric clock errors in a numerical observer

Source: native AdamW at lab commit 79f4fb0, completed-clock first
correction powers in adam.py lines 529--547; retained input bounds at
9d87822. These are auxiliary real-sequence estimates for a weighted
physical parameter/first-moment observer, not an optimizer replacement.

If a true one-step observer law admits an error bounded by budget
times beta^n, add the explicit remaining geometric budget to the
observer. The corrected sequence is antitone. Legal beta and
nonnegative budget give a finite global observer ceiling, and a
nonnegative original observer has a finite nonnegative limit.

The one-step error bound is an external numerical recurrence
hypothesis here. A native CE application must prove it from its own
retained moments and correction clocks; it is not smuggled into the
definition of an observer or supplied as future parameter convergence.
No native or transformer convergence is inferred merely from clipping.

The limit need not be zero: a constant positive sequence with a zero
budget satisfies the law. A separate actual input-generated decrement
must exclude such a limit or force physical mass collapse. These
statements neither determine a grokking time nor select Gen over Mem.
All formulas use exact reals; frozen experiments, optimizers and
checkpoints remain unchanged. Learned stochastic/numerical GPTMini
and thermodynamic system-size transfer are separate questions.
-/

namespace Transformer.Grokking.AdamW

open Filter

/-- The remaining geometric correction pays exactly one clock-error
term. Source: the completed-clock beta powers in native AdamW at
79f4fb0; this is a real arithmetic identity for any beta different
from one, not a claim of parameter or buffer convergence. -/
theorem geometric_error_correction_step (beta budget : ℝ) (n : ℕ) (hb : beta ≠ 1) :
    budget / (1 - beta) * beta ^ (n + 1) + budget * beta ^ n =
      budget / (1 - beta) * beta ^ n := by
  have hden : 1 - beta ≠ 0 := by
    intro hz
    apply hb
    linarith only [hz]
  rw [pow_succ]
  field_simp [hden]
  ring

example : (9 / 10 : ℝ) ≠ 1 := by norm_num

/-- Adding the remaining geometric budget makes a numerical observer
antitone when its true next increment obeys the supplied error law.
Source: native clock powers at 79f4fb0; the application must establish
that one-step law independently of any future convergence. -/
theorem geometric_corrected_observer_antitone (beta budget : ℝ) (observer : ℕ → ℝ)
    (hb : beta ≠ 1)
    (hstep : ∀ n, observer (n + 1) ≤ observer n + budget * beta ^ n) :
    Antitone (fun n => observer n + budget / (1 - beta) * beta ^ n) := by
  apply antitone_nat_of_succ_le
  intro n
  have hpaid := geometric_error_correction_step beta budget n hb
  nlinarith only [hstep n, hpaid]

example : (9 / 10 : ℝ) ≠ 1 ∧
    ∀ n : ℕ, (9 / 10 : ℝ) ^ (n + 1) ≤ (9 / 10 : ℝ) ^ n + 1 * (9 / 10 : ℝ) ^ n := by
  refine ⟨by norm_num, ?_⟩
  intro n
  rw [pow_succ]
  have hn := pow_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10) n
  nlinarith only [hn]

/-- Legal geometric error data generate a finite global observer
ceiling from its initialization. Source: native first-clock powers
at 79f4fb0 and the corrected observer law above; no future box or
observer convergence is an independent premise. -/
theorem observer_ceiling_of_geometric_error (beta budget : ℝ) (observer : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hbudget : 0 ≤ budget)
    (hstep : ∀ n, observer (n + 1) ≤ observer n + budget * beta ^ n) :
    ∀ n, observer n ≤ observer 0 + budget / (1 - beta) := by
  have hne : beta ≠ 1 := by intro h; linarith only [h1, h]
  have hanti := geometric_corrected_observer_antitone beta budget observer hne hstep
  have hcoeff : 0 ≤ budget / (1 - beta) := div_nonneg hbudget (by linarith only [h1])
  intro n
  have hcorrection := mul_nonneg hcoeff (pow_nonneg hb n)
  have hupper := hanti (Nat.zero_le n)
  simp only [pow_zero, mul_one] at hupper
  nlinarith only [hcorrection, hupper]

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    ∀ n : ℕ, (9 / 10 : ℝ) ^ (n + 1) ≤ (9 / 10 : ℝ) ^ n + 1 * (9 / 10 : ℝ) ^ n := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro n
  rw [pow_succ]
  have hn := pow_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10) n
  nlinarith only [hn]

/-- A nonnegative observer with a legal geometric one-step error
budget has a finite nonnegative limit. Source: native correction
powers at 79f4fb0; order completeness is applied to the generated
corrected observer, without assuming a future limit of the original. -/
theorem nonnegative_observer_tendsto_of_geometric_error
    (beta budget : ℝ) (observer : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hbudget : 0 ≤ budget)
    (hnonnegative : ∀ n, 0 ≤ observer n)
    (hstep : ∀ n, observer (n + 1) ≤ observer n + budget * beta ^ n) :
    ∃ value : ℝ, 0 ≤ value ∧ Tendsto observer atTop (nhds value) := by
  let corrected := fun n => observer n + budget / (1 - beta) * beta ^ n
  have hne : beta ≠ 1 := by intro h; linarith only [h1, h]
  have hanti := geometric_corrected_observer_antitone beta budget observer hne hstep
  have hcoeff : 0 ≤ budget / (1 - beta) := div_nonneg hbudget (by linarith only [h1])
  have hnonnegativeCorrected : ∀ n, 0 ≤ corrected n := fun n =>
    add_nonneg (hnonnegative n) (mul_nonneg hcoeff (pow_nonneg hb n))
  have hbounded : BddBelow (Set.range corrected) := by
    apply bddBelow_def.mpr
    refine ⟨0, ?_⟩
    rintro value ⟨n, rfl⟩
    exact hnonnegativeCorrected n
  have ht := tendsto_atTop_ciInf hanti hbounded
  have hcorrection : Tendsto (fun n : ℕ => budget / (1 - beta) * beta ^ n) atTop (nhds 0) := by
    simpa only [mul_zero] using (tendsto_pow_atTop_nhds_zero_of_lt_one hb h1).const_mul (budget / (1 - beta))
  have horiginal : Tendsto observer atTop (nhds (⨅ n, corrected n)) := by
    simpa only [corrected, add_sub_cancel_right, sub_zero] using ht.sub hcorrection
  exact ⟨_, ge_of_tendsto horiginal (Eventually.of_forall hnonnegative), horiginal⟩

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ n : ℕ, 0 ≤ (9 / 10 : ℝ) ^ n) ∧
    (∀ n : ℕ, (9 / 10 : ℝ) ^ (n + 1) ≤ (9 / 10 : ℝ) ^ n + 1 * (9 / 10 : ℝ) ^ n) := by
  refine ⟨by norm_num, by norm_num, by norm_num, fun n => pow_nonneg (by norm_num) n, ?_⟩
  intro n
  rw [pow_succ]
  have hn := pow_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 10) n
  nlinarith only [hn]

/-- The geometric error law alone allows a positive limiting
observer. Source: the auxiliary clock-error recurrence above;
this explicit counterexample explains why its CE application still
needs a true input-generated decrement to derive zero mass. -/
theorem geometric_error_positive_constant_counterexample :
    (∀ n : ℕ, (1 : ℝ) ≤ 1 + 0 * (9 / 10 : ℝ) ^ n) ∧
      Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) ∧
      ¬Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 0) := by
  refine ⟨fun _ => by norm_num, tendsto_const_nhds, ?_⟩
  intro hz
  have hone : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) := tendsto_const_nhds
  have hneq := tendsto_nhds_unique hone hz
  norm_num at hneq

end Transformer.Grokking.AdamW
