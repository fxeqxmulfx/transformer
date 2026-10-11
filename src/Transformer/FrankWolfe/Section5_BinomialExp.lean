/-
# Attention's forward pass and Frank-Wolfe — the exponential binomial bound

The binomial union estimate `choose(N,M) p^M` is at most `(e N p/M)^M`
for `M > 0`. The factorial bound follows from the term of degree `M`
in the exponential series at `M`; no asymptotic form of Stirling is used.
At `p = n² exp(-β c₀/2)` and `N ≤ t`, this is the exponential expression
in the corrected metastability theorem.

The printed Chernoff expression `(N p/M)^M` omits the factor `e^M`.
It is retained here as the additive `1` in the logarithmic exponent.
Source: arXiv:2508.09628v1, §5, proof of `lem: metastab.1`,
`eq: binomial.2` and `eq: metastability.bound`.
-/

import Mathlib.Data.Nat.Choose.Bounds
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Transformer.FrankWolfe.Section5_RareEvents

open scoped ENNReal
open MeasureTheory

namespace Transformer.FrankWolfe

/-- The exponential expression of the binomial estimate, with its factor
`e^M`. Source: arXiv:2508.09628v1, §5, corrected Chernoff step in the proof
of `lem: metastab.1`. -/
theorem choose_mul_pow_le_exp (N M : ℕ) (hM : 0 < M) {p : ℝ} (hp : 0 ≤ p) :
    (N.choose M : ℝ) * p ^ M ≤ Real.exp (M : ℝ) * ((N : ℝ) * p / M) ^ M := by
  have hm : 0 < (M : ℝ) := by exact_mod_cast hM
  have hf : 0 < (M.factorial : ℝ) := by exact_mod_cast Nat.factorial_pos M
  have hfactorial := Real.pow_div_factorial_le_exp (M : ℝ) hm.le M
  calc (N.choose M : ℝ) * p ^ M
      ≤ ((N : ℝ) ^ M / M.factorial) * p ^ M :=
        mul_le_mul_of_nonneg_right (Nat.choose_le_pow_div M N) (by positivity)
    _ = ((M : ℝ) ^ M / M.factorial) * ((N : ℝ) * p / M) ^ M := by
      rw [div_pow, mul_pow]
      field_simp [hm.ne', hf.ne']
    _ ≤ Real.exp (M : ℝ) * ((N : ℝ) * p / M) ^ M :=
      mul_le_mul_of_nonneg_right hfactorial (by positivity)

example : 0 < (1 : ℕ) ∧ 0 ≤ (1 / 2 : ℝ) := by norm_num

/-- Express the logarithmic tail as the product of the exponential-series
factor and the attention factor. Source: arXiv:2508.09628v1, §5,
auxiliary to the last exponent in the proof of `lem: metastab.1`. -/
theorem exp_log_tail_eq {q : ℝ} (hq : 0 < q) (a : ℝ) (M : ℕ) :
    Real.exp (M * (1 + Real.log q - a)) =
      Real.exp (M : ℝ) * (q * Real.exp (-a)) ^ M := by
  have he : Real.exp (Real.log q - a) = q * Real.exp (-a) := by
    rw [sub_eq_add_neg, Real.exp_add, Real.exp_log hq]
  rw [← he, ← Real.exp_nat_mul, ← Real.exp_add]
  congr 1
  ring

example : 0 < (1 : ℝ) := one_pos

/-- The logarithmic tail estimate for a positive number of particles.
Source: arXiv:2508.09628v1, §5, `lem: metastab.1`; the coefficient of `log n`
is the union-bound factor `n²` from actual bad attention steps. -/
theorem choose_attention_tail_le_exp_pos (N M n : ℕ) (hM : 0 < M) (hn : 0 < n)
    {t : ℝ} (ht : 0 < t) (hN : (N : ℝ) ≤ t) (a : ℝ) :
    (N.choose M : ℝ) * (n ^ 2 * Real.exp (-a)) ^ M ≤
      Real.exp (M * (1 + Real.log (t * n ^ 2 / M) - a)) := by
  have hm : 0 < (M : ℝ) := by exact_mod_cast hM
  have hn' : 0 < (n : ℝ) := by exact_mod_cast hn
  have hp : 0 ≤ (n : ℝ) ^ 2 * Real.exp (-a) := by positivity
  have hq : 0 < t * (n : ℝ) ^ 2 / M := by positivity
  rw [exp_log_tail_eq hq]
  refine (choose_mul_pow_le_exp N M hM hp).trans ?_
  apply mul_le_mul_of_nonneg_left _ (Real.exp_pos _).le
  apply pow_le_pow_left₀ (by positivity)
  calc (N : ℝ) * (n ^ 2 * Real.exp (-a)) / M
      ≤ t * (n ^ 2 * Real.exp (-a)) / M :=
        div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hN hp) hm.le
    _ = t * n ^ 2 / M * Real.exp (-a) := by ring

/-- The positive-particle hypotheses have a concrete interior realization:
one candidate time, threshold one, one particle, and horizon two. -/
example :
    0 < (1 : ℕ) ∧ 0 < (1 : ℕ) ∧ 0 < (2 : ℝ) ∧ (1 : ℝ) ≤ 2 := by norm_num

/-- The exponential attention-tail estimate includes `n = 0`: the count
coefficient then vanishes at a positive threshold. Source:
arXiv:2508.09628v1, §5, auxiliary to `eq: metastability.bound`. -/
theorem choose_attention_tail_le_exp (N M n : ℕ) (hM : 0 < M)
    {t : ℝ} (ht : 0 < t) (hN : (N : ℝ) ≤ t) (a : ℝ) :
    (N.choose M : ℝ) * (n ^ 2 * Real.exp (-a)) ^ M ≤
      Real.exp (M * (1 + Real.log (t * n ^ 2 / M) - a)) := by
  by_cases hn : n = 0
  · subst n
    simpa only [Nat.cast_zero, zero_pow (by omega : 2 ≠ 0), zero_mul,
      mul_zero, zero_pow hM.ne', zero_div, Real.log_zero, add_zero] using
      (Real.exp_pos ((M : ℝ) * (1 - a))).le
  · exact choose_attention_tail_le_exp_pos N M n hM (Nat.pos_of_ne_zero hn) ht hN a

example : 0 < (1 : ℕ) ∧ 0 < (2 : ℝ) ∧ (1 : ℝ) ≤ 2 := by norm_num

/-- The conditional rare-step estimate as a real probability, before its
exponential simplification. Source: arXiv:2508.09628v1, §5, auxiliary to
`eq: binomial.2`; an upper bound for dependent steps suffices. -/
theorem count_rare_steps_real_le {d n : ℕ}
    (P : Measure (ℕ → Idx n → EucSpace d)) [IsProbabilityMeasure P]
    (E : ℕ → Set (ℕ → Idx n → EucSpace d)) {p : ℝ} (hp : 0 ≤ p)
    (hE : ∀ t, PastDet (t + 1) (E t))
    (hstep : ∀ t S, PastDet t S → P (S ∩ E t) ≤ ENNReal.ofReal p * P S) (N M : ℕ) :
    (P {x | M ≤ (rareStepTimes E N x).card}).toReal ≤ (N.choose M : ℝ) * p ^ M := by
  have hb : (N.choose M : ℝ≥0∞) * ENNReal.ofReal p ^ M ≠ ⊤ := by finiteness
  have h := ENNReal.toReal_mono hb (measure_count_rare_steps_le P E _ hE hstep N M)
  simpa only [ENNReal.toReal_mul, ENNReal.toReal_pow, ENNReal.toReal_natCast,
    ENNReal.toReal_ofReal hp] using h

/-- The full exponential estimate for a count of conditionally rare steps,
without independence. Source: arXiv:2508.09628v1, §5, corrected Chernoff
step in the proof of `lem: metastab.1`. -/
theorem count_rare_steps_real_exp_le {d n : ℕ}
    (P : Measure (ℕ → Idx n → EucSpace d)) [IsProbabilityMeasure P]
    (E : ℕ → Set (ℕ → Idx n → EucSpace d)) {p : ℝ} (hp : 0 ≤ p)
    (hE : ∀ t, PastDet (t + 1) (E t))
    (hstep : ∀ t S, PastDet t S → P (S ∩ E t) ≤ ENNReal.ofReal p * P S)
    (N M : ℕ) (hM : 0 < M) : (P {x | M ≤ (rareStepTimes E N x).card}).toReal ≤
      Real.exp (M : ℝ) * ((N : ℝ) * p / M) ^ M :=
  (count_rare_steps_real_le P E hp hE hstep N M).trans (choose_mul_pow_le_exp N M hM hp)

/-- The attention-sized conditional bound gives the logarithmic count tail.
Source: arXiv:2508.09628v1, §5, corrected estimate in `lem: metastab.1`. -/
theorem count_attention_steps_real_exp_le {d n : ℕ}
    (P : Measure (ℕ → Idx n → EucSpace d)) [IsProbabilityMeasure P]
    (E : ℕ → Set (ℕ → Idx n → EucSpace d)) (a : ℝ)
    (hE : ∀ t, PastDet (t + 1) (E t))
    (hstep : ∀ t S, PastDet t S →
      P (S ∩ E t) ≤ ENNReal.ofReal (n ^ 2 * Real.exp (-a)) * P S)
    (N M : ℕ) (hM : 0 < M) {t : ℝ} (ht : 0 < t) (hN : (N : ℝ) ≤ t) :
    (P {x | M ≤ (rareStepTimes E N x).card}).toReal ≤
      Real.exp (M * (1 + Real.log (t * n ^ 2 / M) - a)) :=
  (count_rare_steps_real_le P E (by positivity) hE hstep N M).trans
    (choose_attention_tail_le_exp N M n hM ht hN a)

/-- These probability hypotheses hold at the Dirac law of the stationary
origin path, with empty observed events and the finite bound `p = 1`. -/
example :
    IsProbabilityMeasure (Measure.dirac (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1))) ∧
      (∀ t, PastDet (t + 1) (∅ : Set (ℕ → Idx 1 → EucSpace 1))) ∧
      (∀ t S, PastDet t S →
        (Measure.dirac (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1))) (S ∩ ∅) ≤
          ENNReal.ofReal 1 * (Measure.dirac (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1))) S) ∧
      0 ≤ (1 : ℝ) ∧ 0 < (1 : ℕ) ∧ 0 < (2 : ℝ) ∧ (1 : ℝ) ≤ 2 := by
  refine ⟨inferInstance, fun _ => ⟨MeasurableSet.empty, fun _ _ _ => Iff.rfl⟩,
    ?_, by norm_num, by norm_num, by norm_num, by norm_num⟩
  intro t S _
  simp

end Transformer.FrankWolfe
