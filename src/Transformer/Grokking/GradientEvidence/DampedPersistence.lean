module

public import Transformer.Grokking.GradientEvidence.MainAdamMotion
public import Mathlib.Analysis.Normed.Module.Basic

/-!
# Finite-window persistence from realized optimizer forcing

Source: `review/flat_spectrum.md`, *Observed finite-window persistence*, DP1.
Common weight decay stays in the anchor scale. The radius accumulates the
norm of actual forcing, including moment history and rounding; it does not
replace that forcing by a partial cross-entropy gradient. Any window can be
reindexed at zero. No bound on an unknown future gradient is asserted.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- DP1: the exact accumulated common decay, without summing decay as motion. -/
def dampedScale (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | t+1 => a t * dampedScale a t

/-- DP1: accumulate upper norms of forcing, with the same decay as the path. -/
def forcingRadius (a b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t+1 => a t * forcingRadius a b t + b t

/-- DP1: nonnegative step scales give a nonnegative accumulated scale. -/
theorem dampedScale_nonneg (a : ℕ → ℝ) (ha : ∀ t, 0 ≤ a t) (n : ℕ) :
    0 ≤ dampedScale a n := by
  induction n with
  | zero => norm_num [dampedScale]
  | succ t ih => exact mul_nonneg (ha t) ih

/-- DP1: a fixed linear class projection commutes with common decay.
The remaining forcing contains the whole adaptive direction and rounding. -/
theorem projected_adam_forcing (P : E →ₗ[ℝ] F) (p u e : E) (η ρ : ℝ) :
    P ((1-η*ρ) • p - η • u + e) - (1-η*ρ) • P p =
      -(η • P u) + P e := by
  simp only [P.map_add, P.map_sub, P.map_smul]
  abel

/-- DP1: the projected direction norm gives a sufficient forcing budget.
It is the actual buffered direction, not the uniform coordinate bound AD2. -/
theorem projected_adam_forcing_bound (P : E →ₗ[ℝ] F) (p u e : E) (η ρ : ℝ)
    (hη : 0 ≤ η) :
    ‖P ((1-η*ρ) • p - η • u + e) - (1-η*ρ) • P p‖ ≤
      η * ‖P u‖ + ‖P e‖ := by
  rw [projected_adam_forcing]
  have h := norm_add_le (-(η • P u)) (P e)
  simpa only [norm_neg, norm_smul, Real.norm_eq_abs, abs_of_nonneg hη] using h

/-- DP1: every finite prefix stays in a ball around its damped initial state.
The forcing hypothesis concerns each actual update and permits nonzero motion. -/
theorem damped_forcing_tube (x : ℕ → E) (a b : ℕ → ℝ)
    (ha : ∀ t, 0 ≤ a t) (hf : ∀ t, ‖x (t+1) - a t • x t‖ ≤ b t) (n : ℕ) :
    ‖x n - dampedScale a n • x 0‖ ≤ forcingRadius a b n := by
  induction n with
  | zero => simp only [dampedScale, forcingRadius, one_smul, sub_self, norm_zero, le_refl]
  | succ t ih =>
    have he : x (t+1) - (a t * dampedScale a t) • x 0 =
        (x (t+1) - a t • x t) + a t • (x t - dampedScale a t • x 0) := by
      rw [smul_sub, smul_smul]
      abel
    rw [dampedScale, forcingRadius, he]
    have hs : ‖a t • (x t - dampedScale a t • x 0)‖ ≤ a t * forcingRadius a b t := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (ha t)]
      exact mul_le_mul_of_nonneg_left ih (ha t)
    have ht := (norm_add_le (x (t+1)-a t • x t)
      (a t • (x t-dampedScale a t • x 0))).trans (add_le_add (hf t) hs)
    linarith

/-- DP1: the tube gives lower and upper projected norms. An exhausted
radius gives only the zero lower bound; that failure remains explicit. -/
theorem damped_forcing_norm_bounds (x : ℕ → E) (a b : ℕ → ℝ)
    (ha : ∀ t, 0 ≤ a t) (hf : ∀ t, ‖x (t+1) - a t • x t‖ ≤ b t) (n : ℕ) :
    max 0 (dampedScale a n * ‖x 0‖ - forcingRadius a b n) ≤ ‖x n‖ ∧
      ‖x n‖ ≤ dampedScale a n * ‖x 0‖ + forcingRadius a b n := by
  have ht := damped_forcing_tube x a b ha hf n
  have h := (abs_norm_sub_norm_le (x n) (dampedScale a n • x 0)).trans ht
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (dampedScale_nonneg a ha n)] at h
  obtain ⟨hl, hu⟩ := abs_le.mp h
  constructor
  · exact max_le (norm_nonneg _) (by linarith)
  · linarith

/-- DP1: a uniform forcing-to-decay balance gives this comparison at every
horizon. Its bounded-radius interpretation also uses nonnegative B and
scales at most one. The balance does not follow from a small learning rate. -/
theorem forcingRadius_uniform (a b : ℕ → ℝ) (B : ℝ)
    (ha : ∀ t, 0 ≤ a t) (hb : ∀ t, b t ≤ (1-a t)*B) (n : ℕ) :
    forcingRadius a b n ≤ (1-dampedScale a n)*B := by
  induction n with
  | zero => norm_num [forcingRadius, dampedScale]
  | succ t ih =>
    have h := mul_le_mul_of_nonneg_left ih (ha t)
    simp only [forcingRadius, dampedScale]
    nlinarith [hb t]

example (n : ℕ) := dampedScale_nonneg (fun _ => (9/10 : ℝ)) (by intro t; norm_num) n
example := projected_adam_forcing (LinearMap.id : ℝ →ₗ[ℝ] ℝ) 2 (-1) (1/10000) (1/100) (1/10)
example := projected_adam_forcing_bound (LinearMap.id : ℝ →ₗ[ℝ] ℝ) 2 (-1)
  (1/10000) (1/100) (1/10) (by norm_num)
example (n : ℕ) := damped_forcing_tube (fun t : ℕ => (2 : ℝ)+t/100)
  (fun _ => 9/10) (fun t => ‖((2 : ℝ)+(t+1)/100)-(9/10)*((2 : ℝ)+t/100)‖)
  (by intro t; norm_num) (by intro t; norm_num [smul_eq_mul]) n
example (n : ℕ) := damped_forcing_norm_bounds (fun t : ℕ => (2 : ℝ)+t/100)
  (fun _ => 9/10) (fun t => ‖((2 : ℝ)+(t+1)/100)-(9/10)*((2 : ℝ)+t/100)‖)
  (by intro t; norm_num) (by intro t; norm_num [smul_eq_mul]) n
example (n : ℕ) := forcingRadius_uniform (fun _ => 9/10) (fun _ => 1/10) 2
  (by intro t; norm_num) (by intro t; norm_num) n

end Transformer.Grokking.GradientEvidence
