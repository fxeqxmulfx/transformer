import Transformer.GPTMini.Sparsemax.ActiveDirection
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Inv

/-!
# A bounded QKNorm gain that needs no routing labels

The QKNorm score implementation at commit `73f8a0b` normalizes queries
and keys and multiplies their inner product by a learned exponential gain.
The existing norm theorem bounds raw scores between minus and plus that
gain, including epsilon clipping.

We derive an input restriction from arXiv:1602.02068v2, §2.2,
Proposition 1: a gain strictly below one half excludes singleton rows
whenever two positions are visible. A new sigmoid parameterization keeps
the gain below a prescribed cap and has a positive scalar derivative at
every finite parameter. This is a derived modification, not the paper's
score rule or a proven experiment. Outer losses and query/key parameter
derivatives can still vanish; visible exact zeros remain possible.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

/-- New bounded replacement for the learned gain in `QKNormScores`
at commit `73f8a0b`: sigmoid times a cap rather than `exp alpha`.
Source context: the unit support gap in arXiv:1602.02068v2, §2.2. -/
def boundedHeadGain (cap parameter : ℝ) : ℝ :=
  cap / (1 + Real.exp (-parameter))

/-- Positive caps give a strictly positive gain strictly below the cap.
Source context: `QKNormScores` at `73f8a0b`; a derived sigmoid restriction
for Proposition 1 of arXiv:1602.02068v2, §2.2. -/
theorem boundedHeadGain_bounds (cap parameter : ℝ) (hc : 0 < cap) :
    0 < boundedHeadGain cap parameter ∧ boundedHeadGain cap parameter < cap := by
  have he : 0 < Real.exp (-parameter) := Real.exp_pos _
  have hd : 0 < 1 + Real.exp (-parameter) := by positivity
  constructor
  · exact div_pos hc hd
  · unfold boundedHeadGain
    apply (div_lt_iff₀ hd).mpr
    nlinarith [mul_pos hc he]

/-- The bounded-gain premise is realized below the half-unit cap.
Source context: arXiv:1602.02068v2, §2.2, derived score restriction. -/
example : (0 : ℝ) < 12 / 25 := by norm_num

/-- The sigmoid gain's derivative is an actual calculus result, not a
custom backward rule. Source context: the new parameterization of
`QKNormScores` at `73f8a0b`, motivated by §2.2's support criterion. -/
theorem boundedHeadGain_hasDerivAt (cap parameter : ℝ) :
    HasDerivAt (boundedHeadGain cap)
      (cap * Real.exp (-parameter) / (1 + Real.exp (-parameter)) ^ 2) parameter := by
  have he : HasDerivAt (fun x : ℝ => Real.exp (-x)) (-Real.exp (-parameter))
      parameter := by
    simpa using (hasDerivAt_id parameter).neg.exp
  have hd := he.const_add 1
  have hn : (1 + Real.exp (-parameter)) ≠ 0 := ne_of_gt (by positivity)
  have h := (hasDerivAt_const parameter cap).div hd hn
  unfold boundedHeadGain
  simpa [Pi.div_def] using h

/-- The gain parameterization itself has no flat finite-parameter region
when its cap is positive. Source context: the derived sigmoid replacement
for `QKNormScores` at `73f8a0b`; this does not assert a nonzero loss gradient. -/
theorem boundedHeadGain_derivative_pos (cap parameter : ℝ) (hc : 0 < cap) :
    0 < cap * Real.exp (-parameter) / (1 + Real.exp (-parameter)) ^ 2 := by
  positivity

/-- Positive-derivative hypotheses are inhabited by the same cap.
Source context: the derived gain restriction for arXiv:1602.02068v2, §2.2. -/
example : (0 : ℝ) < 12 / 25 := by norm_num

/-- Actual QKNorm scores with the new bounded gain, expressed through
the existing exponential-score function using the logarithm of the gain.
Source: `QKNormScores` at `73f8a0b`; only the gain law is modified. -/
def boundedHeadScores {T d : ℕ} (cap parameter eps : ℝ) (q : EucSpace d)
    (keys : Fin T → EucSpace d) : Fin T → ℝ :=
  fun j => score (Real.log (boundedHeadGain cap parameter)) eps q (keys j)

/-- The epsilon-clipped raw scores are strictly bounded by the cap.
Source: `GPTMini.score_bounded` for `QKNormScores` at `73f8a0b`, with
the derived bounded gain; the norm and clipping definitions are unchanged. -/
theorem boundedHeadScores_abs_lt_cap {T d : ℕ} (cap parameter eps : ℝ)
    (q : EucSpace d) (keys : Fin T → EucSpace d) (hc : 0 < cap) (heps : 0 ≤ eps) :
    ∀ j, |boundedHeadScores cap parameter eps q keys j| < cap := by
  obtain ⟨hg, hcap⟩ := boundedHeadGain_bounds cap parameter hc
  intro j
  calc
    _ ≤ Real.exp (Real.log (boundedHeadGain cap parameter)) :=
      score_bounded _ eps heps q (keys j)
    _ = boundedHeadGain cap parameter := Real.exp_log hg
    _ < cap := hcap

/-- Score-bound premises admit epsilon clipping and zero vectors.
Source context: `QKNormScores` at `73f8a0b`, derived bounded-gain rule. -/
example : (0 : ℝ) < 12 / 25 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- The original QKNorm score map also excludes singleton saturation
under a persistent gain bound, not merely a smaller initialization.
Source: arXiv:1602.02068v2, §2.2, Proposition 1, combined with
`GPTMini.score_bounded` at `73f8a0b`. Two visible slots are required. -/
theorem qknorm_has_two_positive_of_gain_bound {T d : ℕ} (parameter eps : ℝ)
    (q : EucSpace d) (keys : Fin T → EucSpace d) (i a b : Fin T)
    (heps : 0 ≤ eps) (hg : 2 * Real.exp parameter < 1)
    (ha : a ≤ i) (hb : b ≤ i) (hne : a ≠ b) :
    ∃ j k, j ≠ k ∧ 0 < sparseWeights (fun l => score parameter eps q (keys l)) i j ∧
      0 < sparseWeights (fun l => score parameter eps q (keys l)) i k := by
  apply sparseWeights_has_two_positive_of_pairwise_gap _ i a b ha hb hne
  intro j k
  have hj := (abs_le.mp (score_bounded parameter eps heps q (keys j))).2
  have hk := (abs_le.mp (score_bounded parameter eps heps q (keys k))).1
  linarith

/-- Original-score gain-bound premises have a finite logarithmic gain.
Source context: arXiv:1602.02068v2, §2.2, derived QKNorm restriction. -/
example : (0 : ℝ) ≤ 1 ∧ 2 * Real.exp (Real.log (2 / 5)) < 1 ∧
    (0 : Fin 2) ≤ 1 ∧ (1 : Fin 2) ≤ 1 ∧ (0 : Fin 2) ≠ 1 := by
  rw [Real.exp_log (by norm_num : (0 : ℝ) < 2 / 5)]
  norm_num

/-- A cap below one half guarantees two active visible positions at every
finite gain parameter, without any target route. Source: the derived
QKNorm restriction from arXiv:1602.02068v2, §2.2, Proposition 1.
The first causal row, with only one visible slot, is deliberately excluded. -/
theorem boundedHeadScores_has_two_positive {T d : ℕ} (cap parameter eps : ℝ)
    (q : EucSpace d) (keys : Fin T → EucSpace d) (i a b : Fin T)
    (hc : 0 < cap) (hcap : 2 * cap < 1) (heps : 0 ≤ eps)
    (ha : a ≤ i) (hb : b ≤ i) (hne : a ≠ b) :
    ∃ j k, j ≠ k ∧ 0 < sparseWeights (boundedHeadScores cap parameter eps q keys) i j ∧
      0 < sparseWeights (boundedHeadScores cap parameter eps q keys) i k := by
  apply sparseWeights_has_two_positive_of_pairwise_gap _ i a b ha hb hne
  intro j k
  have hj := (abs_lt.mp (boundedHeadScores_abs_lt_cap cap parameter eps q keys hc heps j)).2
  have hk := (abs_lt.mp (boundedHeadScores_abs_lt_cap cap parameter eps q keys hc heps k)).1
  linarith

/-- Bounded-score premises are inhabited with two visible positions.
Source context: arXiv:1602.02068v2, §2.2, derived bounded-gain rule. -/
example : (0 : ℝ) < 12 / 25 ∧ 2 * (12 / 25 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : Fin 2) ≤ 1 ∧ (1 : Fin 2) ≤ 1 ∧ (0 : Fin 2) ≠ 1 := by norm_num

/-- The gain of the bounded sparse score example is attainable at finite
parameter and cap below one half. Source context: §2.2's derived
restriction, with the sigmoid rule above rather than a hard clamp. -/
theorem boundedHeadGain_sparse_example :
    boundedHeadGain (12 / 25) (Real.log 5) = (2 / 5 : ℝ) := by
  unfold boundedHeadGain
  rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 5)]
  norm_num

/-- The bounded upstream gain rules out a zero derivative of the actual
projection with respect to its raw score vector. Source: §2.2 and §2.5
of arXiv:1602.02068v2, combined with `QKNormScores` at `73f8a0b`.
This is not the derivative with respect to queries, keys or the gain. -/
theorem boundedHeadScores_no_zero_score_derivative {T d : ℕ} (cap parameter eps : ℝ)
    (q : EucSpace d) (keys : Fin T → EucSpace d) (i a b : Fin T)
    (hc : 0 < cap) (hcap : 2 * cap < 1) (heps : 0 ≤ eps)
    (ha : a ≤ i) (hb : b ≤ i) (hne : a ≠ b) :
    ¬ HasFDerivAt (𝕜 := ℝ) (fun z : Fin T → ℝ => sparseWeights z i) 0
      (boundedHeadScores cap parameter eps q keys) := by
  obtain ⟨j, k, hjk, hj, hk⟩ :=
    boundedHeadScores_has_two_positive cap parameter eps q keys i a b hc hcap heps ha hb hne
  exact sparseWeights_not_hasFDerivAt_zero_of_two_active _ i j k hjk hj hk

/-- No-zero-score-derivative premises also allow epsilon-clipped zeros.
Source context: the bounded gain derived from arXiv:1602.02068v2, §2.2–§2.5. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (fun z : Fin 2 → ℝ => sparseWeights z 1) 0
    (boundedHeadScores (12 / 25) 0 1 (0 : EucSpace 1) (fun _ => 0)) :=
  boundedHeadScores_no_zero_score_derivative _ _ _ _ _ 1 0 1
    (by norm_num) (by norm_num) (by norm_num) (by decide) (by decide) (by decide)

end Transformer.GPTMini.Sparsemax
