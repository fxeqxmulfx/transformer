import Transformer.GPTMini.Convex.Reparameterization.Interpolation

/-!
# A likelihood midpoint has a strictly positive log-odds minor

New finite counterexample derived from arXiv:2211.11052v1, §3's row-softmax
query/key product and Boyd and Vandenberghe (2004), §3.5's log-concavity
interpolation condition. Every normalized positive interpolation table
between the two actual scalar heads has a positive log-odds minor.

The proof uses rational probability intervals, not numerical logarithm
approximations: a peak has mass at least 2/5; every other category lies
between 7/25 and 8/25. Diagonal contrasts are at least log(5/4), whereas
off-diagonal contrasts have magnitude at most log(8/7). Since 5/4 > 8/7,
the two-by-two determinant is strictly positive.
The arithmetic mean of the physical endpoints satisfies all hypotheses.
-/

noncomputable section

namespace Transformer.GPTMini.Convex.Reparameterization

open scoped BigOperators

/-- A peak's log-odds against a bounded nonpeak category has a positive lower bound.
Source: the new rational certificate derived from §3.5's interpolation condition. -/
theorem peak_log_odds (u v : ℝ) (hu : 2 / 5 ≤ u) (hvpos : 0 < v) (hv : v ≤ 8 / 25) :
    Real.log (5 / 4) ≤ Real.log u - Real.log v := by
  have hupos : 0 < u := by linarith
  have hr : (5 / 4 : ℝ) ≤ u / v := (le_div_iff₀ hvpos).mpr (by linarith)
  have h := Real.log_le_log (by norm_num : (0 : ℝ) < 5 / 4) hr
  rwa [Real.log_div hupos.ne' hvpos.ne'] at h

example : (2 / 5 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 8 / 25 := by
  norm_num

/-- Two nonpeak probabilities in the stated interval have uniformly bounded log-odds.
Source: the new rational interpolation certificate and monotonicity of the real logarithm. -/
theorem small_log_odds (u v : ℝ) (hul : 7 / 25 ≤ u) (huu : u ≤ 8 / 25)
    (hvl : 7 / 25 ≤ v) (hvu : v ≤ 8 / 25) :
    -Real.log (8 / 7) ≤ Real.log u - Real.log v ∧
      Real.log u - Real.log v ≤ Real.log (8 / 7) := by
  have hupos : 0 < u := by linarith
  have hvpos : 0 < v := by linarith
  have h0 := Real.log_le_log (by norm_num : (0 : ℝ) < 7 / 25) hul
  have h1 := Real.log_le_log hupos huu
  have h2 := Real.log_le_log (by norm_num : (0 : ℝ) < 7 / 25) hvl
  have h3 := Real.log_le_log hvpos hvu
  have he : Real.log (8 / 25) - Real.log (7 / 25) = Real.log (8 / 7) := by
    rw [← Real.log_div (by norm_num : (8 / 25 : ℝ) ≠ 0)
      (by norm_num : (7 / 25 : ℝ) ≠ 0)]
    norm_num
  constructor <;> linarith

example :
    (7 / 25 : ℝ) ≤ 3 / 10 ∧ (3 / 10 : ℝ) ≤ 8 / 25 ∧
    (7 / 25 : ℝ) ≤ 3 / 10 ∧ (3 / 10 : ℝ) ≤ 8 / 25 := by
  norm_num

/-- The diagonal log-odds threshold strictly exceeds the positive off-diagonal threshold.
Source: the new exact rational logarithm comparison; no floating-point certificate is used. -/
theorem log_odds_thresholds :
    0 < Real.log (8 / 7) ∧ Real.log (8 / 7) < Real.log (5 / 4) := by
  exact ⟨Real.log_pos (by norm_num), Real.log_lt_log (by norm_num) (by norm_num)⟩

/-- The stated diagonal and off-diagonal contrast intervals force a positive determinant.
Source: the new two-by-two rank obstruction for the scalar-head likelihood midpoint. -/
theorem bounded_minor_pos (d₀ d₁ o₀ o₁ : ℝ)
    (hd₀ : Real.log (5 / 4) ≤ d₀) (hd₁ : Real.log (5 / 4) ≤ d₁)
    (hol₀ : -Real.log (8 / 7) ≤ o₀) (hou₀ : o₀ ≤ Real.log (8 / 7))
    (hol₁ : -Real.log (8 / 7) ≤ o₁) (hou₁ : o₁ ≤ Real.log (8 / 7)) :
    0 < d₀ * d₁ - o₀ * o₁ := by
  let a : ℝ := Real.log (5 / 4)
  let b : ℝ := Real.log (8 / 7)
  have hb : 0 < b := log_odds_thresholds.1
  have hab : b < a := log_odds_thresholds.2
  have ha : 0 ≤ a := by linarith
  have hd : a * a ≤ d₀ * d₁ := mul_le_mul hd₀ hd₁ ha (by linarith)
  have hc₀ : 0 ≤ (b - o₀) * (b + o₁) := mul_nonneg (by linarith) (by linarith)
  have hc₁ : 0 ≤ (b + o₀) * (b - o₁) := mul_nonneg (by linarith) (by linarith)
  have hs : 0 < (a - b) * (a + b) := mul_pos (by linarith) (by linarith)
  nlinarith

example :
    Real.log (5 / 4) ≤ Real.log (5 / 4) ∧ Real.log (5 / 4) ≤ Real.log (5 / 4) ∧
    -Real.log (8 / 7) ≤ 0 ∧ (0 : ℝ) ≤ Real.log (8 / 7) ∧
    -Real.log (8 / 7) ≤ 0 ∧ (0 : ℝ) ≤ Real.log (8 / 7) := by
  have h := log_odds_thresholds.1
  refine ⟨le_refl _, le_refl _, ?_, ?_, ?_, ?_⟩ <;> linarith

/-- Every consistent positive normalized likelihood bridge violates the scalar-head zero minor.
Source: the new actual-softmax counterexample derived from §3 and §3.5.
The hypotheses admit the endpoint arithmetic mean; the conclusion excludes the head class. -/
theorem bridge_minor_pos (p : SharedProbability) (hpos : ∀ r c, 0 < p r c)
    (hsum : ∀ r, ∑ c, p r c = 1)
    (hproduct : ∀ r c, leftProbability r c * rightProbability r c ≤ p r c ^ 2) :
    0 < logOddsMinor p := by
  have hl := bridge_lower p hpos hproduct
  have hu := bridge_small_upper p hsum hl
  have h02 : p 0 2 ≤ 8 / 25 := hu 0 2 (by norm_num [peakCategory])
  have h01 : p 0 1 ≤ 8 / 25 := hu 0 1 (by norm_num [peakCategory])
  have h12 : p 1 2 ≤ 8 / 25 := hu 1 2 (by norm_num [peakCategory])
  have h10 : p 1 0 ≤ 8 / 25 := hu 1 0 (by norm_num [peakCategory])
  have hd₀ := peak_log_odds (p 0 0) (p 0 2)
    (by simpa [peakCategory] using hl 0 0) (hpos 0 2) h02
  have hd₁ := peak_log_odds (p 1 1) (p 1 2)
    (by simpa [peakCategory] using hl 1 1) (hpos 1 2) h12
  have ho₀ := small_log_odds (p 0 1) (p 0 2)
    (by simpa [peakCategory] using hl 0 1) h01
    (by simpa [peakCategory] using hl 0 2) h02
  have ho₁ := small_log_odds (p 1 0) (p 1 2)
    (by simpa [peakCategory] using hl 1 0) h10
    (by simpa [peakCategory] using hl 1 2) h12
  exact bounded_minor_pos _ _ _ _ hd₀ hd₁ ho₀.1 ho₀.2 ho₁.1 ho₁.2

example :
    (∀ r c, 0 < meanProbability r c) ∧
    (∀ r, ∑ c, meanProbability r c = 1) ∧
    (∀ r c, leftProbability r c * rightProbability r c ≤ meanProbability r c ^ 2) := by
  exact ⟨meanProbability_pos, meanProbability_sum, meanProbability_product⟩

/-- The explicit feasible midpoint table has a positive log-odds minor.
Source: the new actual endpoint arithmetic mean and the interpolation certificate. -/
theorem meanProbability_minor_pos : 0 < logOddsMinor meanProbability := by
  exact bridge_minor_pos meanProbability meanProbability_pos
    meanProbability_sum meanProbability_product

/-- The explicit feasible midpoint cannot be represented by any original scalar query/key factors.
Source: the new shared-memory prediction-space counterexample to class closure. -/
theorem meanProbability_not_scalarHead : ¬ScalarHeadClass meanProbability := by
  intro h
  have he := scalarHeadClass_minor h
  have hp := meanProbability_minor_pos
  linarith

/-- Any normalized positive table meeting all midpoint likelihood bounds leaves the scalar class.
Source: the new §3.5 interpolation obstruction, not just the arithmetic mean witness. -/
theorem bridge_not_scalarHead (p : SharedProbability) (hpos : ∀ r c, 0 < p r c)
    (hsum : ∀ r, ∑ c, p r c = 1)
    (hproduct : ∀ r c, leftProbability r c * rightProbability r c ≤ p r c ^ 2) :
    ¬ScalarHeadClass p := by
  intro h
  have he := scalarHeadClass_minor h
  have hp := bridge_minor_pos p hpos hsum hproduct
  linarith

example :
    (∀ r c, 0 < meanProbability r c) ∧
    (∀ r, ∑ c, meanProbability r c = 1) ∧
    (∀ r c, leftProbability r c * rightProbability r c ≤ meanProbability r c ^ 2) := by
  exact ⟨meanProbability_pos, meanProbability_sum, meanProbability_product⟩

end Transformer.GPTMini.Convex.Reparameterization
