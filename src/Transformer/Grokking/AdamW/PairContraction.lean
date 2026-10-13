import Transformer.Grokking.AdamW.PairEnvelope

/-!
# Retained feedback contraction at the strict decay-denominator threshold

Sources: Varma et al., arXiv:2309.02390v1, section 3's CE/decay
competition and appendix C's product partials; native retained mass
envelopes at f9d9eae/ada36f3 and actual cold ceiling at 696630b.

The two numerical upper envelopes are
mNext <= beta*m + (1-beta)*coefficient*p and
pNext <= (1-rate*decay)*p + rate*mNext/floor.
For positive coefficient, floor and rate, valid beta and nonnegative
remaining decay, coefficient < decay*floor generates two positive
measurement weights and a strictly smaller-than-one common factor.

The construction retains beta and both coordinates. Choose parameter
weight (1-beta)*floor and add rate*(decay*floor-coefficient)/(2*coefficient)
to the usual beta*rate moment weight. Both normalized row sums become
strictly below one. Their maximum gives the common contraction factor.
No instantaneous moment/gradient equality or zero-beta replacement is
used. A subsequent actual-path application must generate both envelopes
and the complete corrected denominator floor.

Current nonnegative masses and those explicit bounds contract the
numerical weighted sum, and induction gives tail power ceilings.
These results neither define a new optimizer nor assert that every
trained transformer meets the bounds. Exact real fixed-table/native
decay differs from the source's coupled assigned norm-cost GD.
The abstract coefficient is not a temperature or a system-size limit.
-/

namespace Transformer.Grokking.AdamW

/-- Strict decay feedback gap generates positive weights with both
native retained row coefficients below one common contraction factor.
Sources: section 3 competition, appendix C pair feedback and native
mass bounds at f9d9eae/ada36f3; no trajectory is supplied as a premise. -/
theorem retained_feedback_contraction_weights (beta floor rate decay coefficient : ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hd : 0 < floor) (heta : 0 < rate)
    (hc : 0 < coefficient) (hkeep : 0 ≤ 1 - rate * decay) (hgap : coefficient < decay * floor) :
    ∃ parameterWeight momentWeight factor : ℝ, 0 < parameterWeight ∧ 0 < momentWeight ∧
      0 ≤ factor ∧ factor < 1 ∧
      parameterWeight * (1 - rate * decay) +
        (parameterWeight * rate / floor + momentWeight) * (1 - beta) * coefficient ≤ factor * parameterWeight ∧
      (parameterWeight * rate / floor + momentWeight) * beta ≤ factor * momentWeight := by
  let gap := decay * floor - coefficient
  let a := (1 - beta) * floor
  let b := rate * beta + rate * gap / (2 * coefficient)
  let k := rate + rate * gap / (2 * coefficient)
  let pRow := a * (1 - rate * decay) + k * (1 - beta) * coefficient
  let mRow := k * beta
  have hg : 0 < gap := by dsimp only [gap]; linarith only [hgap]
  have hbeta : 0 < 1 - beta := by linarith only [h1]
  have ha : 0 < a := mul_pos hbeta hd
  have ht : 0 < rate * gap / (2 * coefficient) := by positivity
  have hbWeight : 0 < b := by
    have hr := mul_nonneg (le_of_lt heta) hb
    dsimp only [b]
    linarith only [hr, ht]
  have hk : 0 < k := by dsimp only [k]; linarith only [heta, ht]
  have hkIdentity : a * rate / floor + b = k := by
    dsimp only [a, b, k]
    field_simp [ne_of_gt hd]
    ring
  have hparameterGap : 2 * (a - pRow) = rate * (1 - beta) * gap := by
    dsimp only [a, pRow, k, gap]
    field_simp [ne_of_gt hc]
    ring
  have hmomentGap : 2 * coefficient * (b - mRow) = rate * (1 - beta) * gap := by
    dsimp only [b, mRow, k]
    field_simp [ne_of_gt hc]
    ring
  have hpositiveGap := mul_pos (mul_pos heta hbeta) hg
  have hpLt : pRow < a := by nlinarith only [hparameterGap, hpositiveGap]
  have hmLt : mRow < b := by nlinarith only [hmomentGap, hpositiveGap, hc]
  have hpNonneg : 0 ≤ pRow := add_nonneg (mul_nonneg (le_of_lt ha) hkeep)
    (mul_nonneg (mul_nonneg (le_of_lt hk) (le_of_lt hbeta)) (le_of_lt hc))
  have hmNonneg : 0 ≤ mRow := mul_nonneg (le_of_lt hk) hb
  let factor := max (pRow / a) (mRow / b)
  have hfactor : 0 ≤ factor := le_trans (div_nonneg hpNonneg (le_of_lt ha)) (le_max_left _ _)
  have hfactorLt : factor < 1 := max_lt
    ((div_lt_iff₀ ha).mpr (by simpa only [one_mul] using hpLt))
    ((div_lt_iff₀ hbWeight).mpr (by simpa only [one_mul] using hmLt))
  have hpRow : pRow ≤ factor * a := (div_le_iff₀ ha).mp (le_max_left _ _)
  have hmRow : mRow ≤ factor * b := (div_le_iff₀ hbWeight).mp (le_max_right _ _)
  refine ⟨a, b, factor, ha, hbWeight, hfactor, hfactorLt, ?_, ?_⟩
  · rw [hkIdentity]
    exact hpRow
  · rw [hkIdentity]
    exact hmRow

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 3 / 2 ∧
    (0 : ℝ) ≤ 1 - (1 / 1000) * 100 ∧ (3 / 2 : ℝ) < 100 * 1 := by norm_num

/-- Two actual-form retained upper envelopes and numerical row bounds
contract their weighted current mass. Sources: native mass at ada36f3
and appendix C feedback; no future parameter or moment limit is assumed. -/
theorem retained_feedback_weight_step
    (beta keep rate floor coefficient parameterWeight momentWeight factor p m pNext mNext : ℝ)
    (heta : 0 ≤ rate) (hd : 0 < floor) (ha : 0 ≤ parameterWeight) (hb : 0 ≤ momentWeight)
    (hp : 0 ≤ p) (hm : 0 ≤ m)
    (hpRow : parameterWeight * keep + (parameterWeight * rate / floor + momentWeight) *
      (1 - beta) * coefficient ≤ factor * parameterWeight)
    (hmRow : (parameterWeight * rate / floor + momentWeight) * beta ≤ factor * momentWeight)
    (hmNext : mNext ≤ beta * m + (1 - beta) * coefficient * p)
    (hpNext : pNext ≤ keep * p + rate * (mNext / floor)) :
    parameterWeight * pNext + momentWeight * mNext ≤
      factor * (parameterWeight * p + momentWeight * m) := by
  have hden := div_nonneg (mul_nonneg ha heta) (le_of_lt hd)
  have hpScaled := mul_le_mul_of_nonneg_left hpNext ha
  have hpLinear : parameterWeight * pNext ≤ parameterWeight * keep * p +
      (parameterWeight * rate / floor) * mNext := by
    convert hpScaled using 1
    ring
  have hmScaled := mul_le_mul_of_nonneg_left hmNext (add_nonneg hden hb)
  have hpRowScaled := mul_le_mul_of_nonneg_right hpRow hp
  have hmRowScaled := mul_le_mul_of_nonneg_right hmRow hm
  nlinarith only [hpLinear, hmScaled, hpRowScaled, hmRowScaled]

example : (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (1 : ℝ) * (9 / 10) + (1 * (1 / 1000) / 1 + 1) * (1 - 9 / 10) * (1 / 10) ≤ (99 / 100) * 1 ∧
    (1 * (1 / 1000) / 1 + 1 : ℝ) * (9 / 10) ≤ (99 / 100) * 1 ∧
    (9 / 10 : ℝ) ≤ (9 / 10) * 1 + (1 - 9 / 10) * (1 / 10) * 1 ∧
    (9 / 10 : ℝ) ≤ (9 / 10) * 1 + (1 / 1000) * ((9 / 10) / 1) := by norm_num

/-- Tail envelope bounds give an initialized geometric ceiling for
the retained weighted measurement. Sources: appendix C feedback and
native mass at ada36f3; the tail's complete current state is retained. -/
theorem retained_feedback_weight_tail_power
    (beta keep rate floor coefficient parameterWeight momentWeight factor : ℝ)
    (p m : ℕ → ℝ) (start : ℕ)
    (heta : 0 ≤ rate) (hd : 0 < floor) (ha : 0 ≤ parameterWeight) (hb : 0 ≤ momentWeight)
    (hf : 0 ≤ factor) (hp : ∀ n, 0 ≤ p n) (hm : ∀ n, 0 ≤ m n)
    (hpRow : parameterWeight * keep + (parameterWeight * rate / floor + momentWeight) *
      (1 - beta) * coefficient ≤ factor * parameterWeight)
    (hmRow : (parameterWeight * rate / floor + momentWeight) * beta ≤ factor * momentWeight)
    (hmNext : ∀ n, start ≤ n → m (n + 1) ≤ beta * m n + (1 - beta) * coefficient * p n)
    (hpNext : ∀ n, start ≤ n → p (n + 1) ≤ keep * p n + rate * (m (n + 1) / floor)) :
    ∀ k, parameterWeight * p (start + k) + momentWeight * m (start + k) ≤
      factor ^ k * (parameterWeight * p start + momentWeight * m start) := by
  intro k
  induction k with
  | zero => simp only [Nat.add_zero, pow_zero, one_mul, le_refl]
  | succ k ih =>
    have hs := retained_feedback_weight_step beta keep rate floor coefficient parameterWeight momentWeight factor
      (p (start + k)) (m (start + k)) (p (start + k + 1)) (m (start + k + 1))
      heta hd ha hb (hp _) (hm _) hpRow hmRow (hmNext _ (by omega)) (hpNext _ (by omega))
    have hmul := mul_le_mul_of_nonneg_left ih hf
    calc
      _ ≤ factor * (parameterWeight * p (start + k) + momentWeight * m (start + k)) := hs
      _ ≤ factor * (factor ^ k * (parameterWeight * p start + momentWeight * m start)) := hmul
      _ = _ := by rw [pow_succ]; ring

example : (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 99 / 100 ∧ (∀ _ : ℕ, (0 : ℝ) ≤ 0) ∧ (∀ _ : ℕ, (0 : ℝ) ≤ 0) ∧
    (1 : ℝ) * (9 / 10) + (1 * (1 / 1000) / 1 + 1) * (1 - 9 / 10) * (1 / 10) ≤ (99 / 100) * 1 ∧
    (1 * (1 / 1000) / 1 + 1 : ℝ) * (9 / 10) ≤ (99 / 100) * 1 ∧
    (∀ n : ℕ, 7 ≤ n → (0 : ℝ) ≤ (9 / 10) * 0 + (1 - 9 / 10) * (1 / 10) * 0) ∧
    (∀ n : ℕ, 7 ≤ n → (0 : ℝ) ≤ (9 / 10) * 0 + (1 / 1000) * (0 / 1)) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun _ => le_rfl, fun _ => le_rfl, by norm_num, by norm_num, fun _ _ => by norm_num, fun _ _ => by norm_num⟩

end Transformer.Grokking.AdamW
