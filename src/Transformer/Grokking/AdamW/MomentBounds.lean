import Transformer.Grokking.AdamW.MomentRecurrence

/-!
# Causal bounds for corrected native moments

Source: PyTorch 2.14.1 AdamW at lab commit 0033b1b, ordinary zero-
initialized exponential buffers and positive-clock bias correction.
Prove comparison against constant inputs, then the actual corrected
first/second-moment bounds from a bounded gradient prefix. An exact-real
coordinate bound follows from ideal norm clipping, but the numerical
clipping kernel and a particular GPTMini gradient are not certified here.

The adaptive direction has the stated epsilon-floor bound. This does
not assert alignment of a stale first moment with the current gradient,
or descent at the prescribed rate. Gradients beyond the current clock
are unconstrained. No long-time generalization conclusion is presumed.
-/

namespace Transformer.Grokking.AdamW

/-- Prefix input order propagates through the native buffer recurrence.
Source: PyTorch 2.14.1 AdamW at 0033b1b; nonnegative averaging factors
are explicit, and the two initial buffers need only be ordered. -/
theorem moment_prefix_order (beta left right : ℝ) (a b : ℕ → ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1) (hi : left ≤ right)
    (hg : ∀ k : ℕ, k < n → a k ≤ b k) :
    momentAt beta a left n ≤ momentAt beta b right n := by
  revert hg
  induction n with
  | zero => intro hg; exact hi
  | succ n ih =>
    intro hg
    have hm := ih (fun k hk => hg k (by omega))
    have hc := hg n (by omega)
    exact add_le_add (mul_le_mul_of_nonneg_left hm hb)
      (mul_le_mul_of_nonneg_left hc (by linarith))

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ k : ℕ, k < 3 → (if k = 0 then (-1 : ℝ) else 0) ≤ 1) := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro k hk
  split_ifs <;> norm_num

/-- The zero-initialized first moment has its actual clock-dependent
bound, not merely the looser bound on an arbitrary initial buffer.
Source: native AdamW at 0033b1b, comparison with constant inputs. -/
theorem first_moment_prefix_abs_bound (beta bound : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1)
    (hg : ∀ k : ℕ, k < n → |gradient k| ≤ bound) :
    |firstMomentAt beta gradient n| ≤ (1 - beta ^ n) * bound := by
  have hl := moment_prefix_order beta 0 0 (fun _ => -bound) gradient n hb h1 le_rfl
    (fun k hk => (abs_le.mp (hg k hk)).1)
  have hu := moment_prefix_order beta 0 0 gradient (fun _ => bound) n hb h1 le_rfl
    (fun k hk => (abs_le.mp (hg k hk)).2)
  rw [moment_constant] at hl hu
  unfold firstMomentAt
  apply abs_le.mpr
  constructor <;> linarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 ∧
    (∀ k : ℕ, k < 3 → |(if k = 0 then (1 : ℝ) else -1)| ≤ 1) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro k hk
  split_ifs <;> norm_num

/-- Bias correction retains the prefix coordinate bound at every
positive clock. Source: native AdamW at 0033b1b; division by the proved
positive correction factor cannot conceal an invalid beta or clock. -/
theorem corrected_first_prefix_abs_bound (beta bound : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hn : 0 < n)
    (hg : ∀ k : ℕ, k < n → |gradient k| ≤ bound) :
    |firstMomentAt beta gradient n / (1 - beta ^ n)| ≤ bound := by
  have hc := bias_correction_positive beta n hb h1 hn
  rw [abs_div, abs_of_nonneg (le_of_lt hc), div_le_iff₀ hc]
  have hm := first_moment_prefix_abs_bound beta bound gradient n hb (le_of_lt h1) hg
  nlinarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ 0 < (3 : ℕ) ∧
    (∀ k : ℕ, k < 3 → |(1 : ℝ)| ≤ 1) := by norm_num

/-- The corrected second moment lies between zero and the squared
prefix bound. Source: PyTorch 2.14.1 non-AMSGrad AdamW at 0033b1b;
negative variance is excluded by the actual squared-input recurrence. -/
theorem corrected_second_prefix_bound (beta bound : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hn : 0 < n)
    (hg : ∀ k : ℕ, k < n → |gradient k| ≤ bound) :
    0 ≤ secondMomentAt beta gradient n / (1 - beta ^ n) ∧
      secondMomentAt beta gradient n / (1 - beta ^ n) ≤ bound ^ 2 := by
  have hc := bias_correction_positive beta n hb h1 hn
  refine ⟨div_nonneg (second_moment_nonneg beta gradient n hb (le_of_lt h1))
    (le_of_lt hc), ?_⟩
  have hu := moment_prefix_order beta 0 0 (fun k => gradient k ^ 2)
    (fun _ => bound ^ 2) n hb (le_of_lt h1) le_rfl (fun k hk =>
      sq_le_sq' (abs_le.mp (hg k hk)).1 (abs_le.mp (hg k hk)).2)
  rw [moment_constant] at hu
  rw [div_le_iff₀ hc]
  unfold secondMomentAt
  nlinarith

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ 0 < (3 : ℕ) ∧
    (∀ k : ℕ, k < 3 → |(-1 : ℝ)| ≤ 1) := by norm_num

/-- The exact native history direction is bounded using the epsilon
floor and causal coordinate bound. Source: native AdamW at 0033b1b;
the estimate can be large at the lab's epsilon=1e-8 and says nothing
about its current-loss sign or the effect of decoupled decay. -/
theorem history_direction_prefix_abs_bound (b1 b2 eps bound : ℝ)
    (gradient : ℕ → ℝ) (n : ℕ) (hb : 0 ≤ b1) (h1 : b1 < 1)
    (he : 0 < eps) (hn : 0 < n)
    (hg : ∀ k : ℕ, k < n → |gradient k| ≤ bound) :
    |historyDirection b1 b2 eps gradient n| ≤ bound / eps := by
  have hm := corrected_first_prefix_abs_bound b1 bound gradient n hb h1 hn hg
  have hv := Real.sqrt_nonneg (secondMomentAt b2 gradient n / (1 - b2 ^ n))
  have hd : 0 ≤ Real.sqrt (secondMomentAt b2 gradient n / (1 - b2 ^ n)) + eps := by linarith
  unfold historyDirection
  rw [abs_div, abs_of_nonneg hd]
  calc
    _ ≤ |firstMomentAt b1 gradient n / (1 - b1 ^ n)| / eps :=
      div_le_div_of_nonneg_left (abs_nonneg _) he (by linarith)
    _ ≤ bound / eps := div_le_div_of_nonneg_right hm (le_of_lt he)

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 < (3 : ℕ) ∧ (∀ k : ℕ, k < 3 → |(1 : ℝ)| ≤ 1) := by norm_num

/-- A genuinely constant gradient has exactly corrected moments g
and g squared, at every positive clock. Source: PyTorch 2.14.1 AdamW
at 0033b1b; these values are derived from its recurrence and correction,
not stipulated as independent buffers. -/
theorem constant_gradient_corrected (b1 b2 value : ℝ) (n : ℕ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (hn : 0 < n) :
    firstMomentAt b1 (fun _ => value) n / (1 - b1 ^ n) = value ∧
      secondMomentAt b2 (fun _ => value) n / (1 - b2 ^ n) = value ^ 2 := by
  have hc1 := bias_correction_positive b1 n hb1 h1 hn
  have hc2 := bias_correction_positive b2 n hb2 h2 hn
  unfold firstMomentAt secondMomentAt
  rw [moment_constant, moment_constant]
  constructor <;> field_simp <;> ring

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ 0 < (3 : ℕ) := by norm_num

/-- Every positive-clock direction for a constant gradient agrees
with its corrected first-step formula. Source: native AdamW at
0033b1b; this constant-input case is not imposed on training gradients. -/
theorem constant_gradient_direction (b1 b2 eps value : ℝ) (n : ℕ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1) (hn : 0 < n) :
    historyDirection b1 b2 eps (fun _ => value) n = value / (|value| + eps) := by
  have hm := constant_gradient_corrected b1 b2 value n hb1 h1 hb2 h2 hn
  unfold historyDirection
  rw [hm.1, hm.2, Real.sqrt_sq_eq_abs]

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ 0 < (5 : ℕ) := by norm_num

end Transformer.Grokking.AdamW
