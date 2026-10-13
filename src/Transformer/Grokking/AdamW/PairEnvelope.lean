import Transformer.Grokking.AdamW.ScalarDenominator

/-!
# Retained pair mass under explicit feedback lower bounds

Sources: Varma et al., arXiv:2309.02390v1, section 3 and appendix C's
two-factor feedback; native AdamW denominator at lab commit f9d9eae.
The parameter mass and negative first-moment mass remain separate.
An explicit lower bound on their two recurrences controls a weighted
sum without discarding the retained moment or replacing AdamW by GD.

The coefficient and denominator ceiling here are supplied bounds.
The closed CE application must derive them from its actual partners,
shared clipping and retained variance. The critical inequality is
coefficient >= decay * ceiling. It implies tail nondecrease of the
weighted mass. A positive mass with a vanishing moment then has a
positive finite parameter limit, conditional on convergence.

These algebraic envelope laws neither assume nor prove that an
arbitrary CE path meets that threshold. They are not an attraction,
global-convergence, delayed-crossing or learned GPTMini theorem.
-/

namespace Transformer.Grokking.AdamW

open Filter

/-- Numerical combination of parameter and negative retained-moment
mass. Sources: section 3's two-factor feedback and native AdamW at
f9d9eae; the coefficients retain beta1 and the actual constant rate. -/
def retainedPairWeight (beta ceiling rate parameterMass momentMass : ℝ) : ℝ :=
  (1 - beta) * ceiling * parameterMass + beta * rate * momentMass

/-- Explicit pair recurrence bounds give the exact weighted growth
lower bound. Sources: appendix C partner feedback and native AdamW
at f9d9eae; the parameter bound has its positive denominator cleared. -/
theorem retained_pair_weight_growth (beta ceiling rate decay coefficient u v w z : ℝ)
    (h1 : beta ≤ 1) (heta : 0 ≤ rate)
    (hm : beta * w + (1 - beta) * coefficient * u ≤ z)
    (hp : ceiling * (1 - rate * decay) * u + rate * z ≤ ceiling * v) :
    rate * (1 - beta) * (coefficient - decay * ceiling) * u ≤
      retainedPairWeight beta ceiling rate v z - retainedPairWeight beta ceiling rate u w := by
  have ha := mul_le_mul_of_nonneg_left hp (show 0 ≤ 1 - beta by linarith)
  have hb := mul_le_mul_of_nonneg_left hm heta
  unfold retainedPairWeight
  nlinarith only [ha, hb]

example : (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (9 / 10 : ℝ) * 1 + (1 - 9 / 10) * (1 / 5) * 1 ≤ 1 ∧
    (1 : ℝ) * (1 - (1 / 1000) * (1 / 10)) * 1 + (1 / 1000) * 1 ≤ 1 * 2 := by norm_num

/-- Above the critical feedback threshold, a positive parameter
mass gives strictly positive weighted growth. Sources: appendix C
pair mechanism and native AdamW at f9d9eae; no time-to-success is inferred. -/
theorem retained_pair_weight_strict_growth (beta ceiling rate decay coefficient u v w z : ℝ)
    (h1 : beta < 1) (heta : 0 < rate) (hc : decay * ceiling < coefficient) (hu : 0 < u)
    (hm : beta * w + (1 - beta) * coefficient * u ≤ z)
    (hp : ceiling * (1 - rate * decay) * u + rate * z ≤ ceiling * v) :
    retainedPairWeight beta ceiling rate u w < retainedPairWeight beta ceiling rate v z := by
  have hg := retained_pair_weight_growth beta ceiling rate decay coefficient u v w z
    (le_of_lt h1) (le_of_lt heta) hm hp
  have hpos : 0 < rate * (1 - beta) * (coefficient - decay * ceiling) * u :=
    mul_pos (mul_pos (mul_pos heta (by linarith)) (by linarith)) hu
  linarith

example : (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 1000 ∧ (1 / 10 : ℝ) * 1 < 1 / 5 ∧
    (0 : ℝ) < 1 ∧ (9 / 10 : ℝ) * 1 + (1 - 9 / 10) * (1 / 5) * 1 ≤ 1 ∧
    (1 : ℝ) * (1 - (1 / 1000) * (1 / 10)) * 1 + (1 / 1000) * 1 ≤ 1 * 2 := by norm_num

/-- A positive parameter mass and nonnegative negative-moment mass
give positive weighted mass. Sources: native AdamW at f9d9eae and
section 3's nonzero seeds; beta1=0 is included. -/
theorem retained_pair_weight_pos (beta ceiling rate u w : ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hd : 0 < ceiling) (heta : 0 ≤ rate)
    (hu : 0 < u) (hw : 0 ≤ w) : 0 < retainedPairWeight beta ceiling rate u w := by
  have ha : 0 < (1 - beta) * ceiling * u := mul_pos (mul_pos (by linarith) hd) hu
  have hm : 0 ≤ beta * rate * w := mul_nonneg (mul_nonneg hb heta) hw
  unfold retainedPairWeight
  linarith

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 := by norm_num

/-- Tail feedback and parameter lower bounds prevent weighted mass
from dropping below its value at the start of that tail. Sources:
appendix C feedback and native AdamW at f9d9eae; all earlier history
is retained in the starting mass instead of reset. -/
theorem retained_pair_weight_tail_floor (beta ceiling rate decay coefficient : ℝ)
    (u w : ℕ → ℝ) (start : ℕ) (h1 : beta ≤ 1) (heta : 0 ≤ rate)
    (hc : 0 ≤ coefficient - decay * ceiling) (hu : ∀ n, 0 ≤ u n)
    (hm : ∀ n, start ≤ n → beta * w n + (1 - beta) * coefficient * u n ≤ w (n + 1))
    (hp : ∀ n, start ≤ n → ceiling * (1 - rate * decay) * u n + rate * w (n + 1) ≤ ceiling * u (n + 1)) :
    ∀ k, retainedPairWeight beta ceiling rate (u start) (w start) ≤
      retainedPairWeight beta ceiling rate (u (start + k)) (w (start + k)) := by
  intro k
  induction k with
  | zero => simp only [Nat.add_zero, le_refl]
  | succ k ih =>
    have hg := retained_pair_weight_growth beta ceiling rate decay coefficient
      (u (start + k)) (u (start + k + 1)) (w (start + k)) (w (start + k + 1))
      h1 heta (hm _ (by omega)) (hp _ (by omega))
    have hn := mul_nonneg (mul_nonneg (mul_nonneg heta (by linarith : 0 ≤ 1 - beta)) hc) (hu (start + k))
    change _ ≤ retainedPairWeight beta ceiling rate (u (start + k + 1)) (w (start + k + 1))
    linarith

example : (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 0 - 0 * 1 ∧
    (∀ _ : ℕ, (0 : ℝ) ≤ 1) ∧
    (∀ n : ℕ, 3 ≤ n → (9 / 10 : ℝ) * 0 + (1 - 9 / 10) * 0 * 1 ≤ 0) ∧
    (∀ n : ℕ, 3 ≤ n → (1 : ℝ) * (1 - (1 / 1000) * 0) * 1 + (1 / 1000) * 0 ≤ 1 * 1) := by
  refine ⟨by norm_num, by norm_num, by norm_num, fun _ => by norm_num, fun _ _ => by norm_num,
    fun _ _ => by norm_num⟩

/-- Positive mass cannot have a zero finite parameter limit while
the retained moment vanishes under critical-or-larger feedback.
Sources: appendix C pair mechanism and native AdamW at f9d9eae;
convergence is explicit and the lower bounds need an actual CE bridge. -/
theorem retained_pair_positive_limit (beta ceiling rate decay coefficient value : ℝ)
    (u w : ℕ → ℝ) (start : ℕ) (hb : 0 ≤ beta) (h1 : beta < 1) (hd : 0 < ceiling)
    (heta : 0 ≤ rate) (hc : 0 ≤ coefficient - decay * ceiling)
    (hu : ∀ n, 0 < u n) (hw : ∀ n, 0 ≤ w n)
    (hm : ∀ n, start ≤ n → beta * w n + (1 - beta) * coefficient * u n ≤ w (n + 1))
    (hp : ∀ n, start ≤ n → ceiling * (1 - rate * decay) * u n + rate * w (n + 1) ≤ ceiling * u (n + 1))
    (hl : Tendsto u atTop (nhds value)) (hwl : Tendsto w atTop (nhds 0)) : 0 < value := by
  have hf := retained_pair_weight_tail_floor beta ceiling rate decay coefficient u w start
    (le_of_lt h1) heta hc (fun n => le_of_lt (hu n)) hm hp
  have hpos := retained_pair_weight_pos beta ceiling rate (u start) (w start) hb h1 hd heta (hu start) (hw start)
  have ht : Tendsto (fun n => retainedPairWeight beta ceiling rate (u n) (w n)) atTop
      (nhds ((1 - beta) * ceiling * value)) := by
    simpa only [retainedPairWeight, mul_zero, add_zero] using
      (hl.const_mul ((1 - beta) * ceiling)).add (hwl.const_mul (beta * rate))
  have hevent : ∀ᶠ n in atTop, retainedPairWeight beta ceiling rate (u start) (w start) ≤
      retainedPairWeight beta ceiling rate (u n) (w n) := by
    apply eventually_atTop.mpr
    refine ⟨start, fun n hn => ?_⟩
    have heq : start + (n - start) = n := by omega
    simpa only [heq] using hf (n - start)
  have hlim := ge_of_tendsto ht hevent
  have hfactor : 0 < (1 - beta) * ceiling := mul_pos (by linarith) hd
  exact (mul_pos_iff_of_pos_left hfactor).mp (lt_of_lt_of_le hpos hlim)

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 1000 ∧
    (0 : ℝ) ≤ 0 - 0 * 1 ∧ (∀ _ : ℕ, (0 : ℝ) < 1) ∧ (∀ _ : ℕ, (0 : ℝ) ≤ 0) ∧
    (∀ n : ℕ, 3 ≤ n → (9 / 10 : ℝ) * 0 + (1 - 9 / 10) * 0 * 1 ≤ 0) ∧
    (∀ n : ℕ, 3 ≤ n → (1 : ℝ) * (1 - (1 / 1000) * 0) * 1 + (1 / 1000) * 0 ≤ 1 * 1) ∧
    Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) ∧ Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (nhds 0) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, fun _ => by norm_num,
    fun _ => by norm_num, fun _ _ => by norm_num, fun _ _ => by norm_num,
    tendsto_const_nhds, tendsto_const_nhds⟩

end Transformer.Grokking.AdamW
