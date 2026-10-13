import Transformer.Grokking.Composition.Basic

/-!
# Negative removal contrasts do not identify product logits

Source comparison: Nanda et al., arXiv:2301.05217v1, appendix Further
speculations on grokking, Hypothesis: Phase Transitions are inherent to
composition. The source proposes useful interacting circuit parts; it
does not claim that a four-corner CE contrast identifies such a circuit.

Explicit diagnostic counterexample: two examples have additive logits
`2*a-b` and `2*b-a` for the same correct binary class. Use the mean of
their actual finite-class CE, as an observation does. At `(1,1)` both
individual removals increase that mean, and the four-corner contrast is
negative. Both logits nevertheless have zero mixed coordinate derivative.
Thus even adding individual-usefulness checks does not identify a product
in the logits. Aggregation and nonlinear loss can supply that pattern.

No learned transformer algorithm or actual numerical-kernel bridge is
claimed. This explicitly fixed two-example task is not division or EOS.
-/

namespace Transformer.Grokking.Composition

open Transformer.Grokking.GradientEvidence

/-- Actual binary CE in its scalar-logit form. Source: the ordinary
finite-class objective used in arXiv:2301.05217v1, appendix composition
hypothesis; the equality reuses the verified two-class finite sum. -/
theorem binaryCE_true_eq_log (s : ℝ) :
    binaryCE true s = Real.log (1 + Real.exp (-s)) := by
  simpa [coupledLoss] using coupledLoss_eq_exp s 1

/-- Reversing the correct-class score adds the original score to CE.
Source: ordinary binary softmax CE, arXiv:2301.05217v1 appendix
composition hypothesis; this identity keeps the logit dependence. -/
theorem binaryCE_true_neg (s : ℝ) : binaryCE true (-s) = binaryCE true s + s := by
  have he := Real.exp_pos s
  have hm : 1 + Real.exp s = Real.exp s * (1 + Real.exp (-s)) := by
    rw [Real.exp_neg]
    field_simp
    ring
  rw [binaryCE_true_eq_log, binaryCE_true_eq_log]
  rw [neg_neg, hm, Real.log_mul (ne_of_gt he) (by positivity), Real.log_exp]
  ring

/-- Every finite two-class CE here is positive. Source: the ordinary
softmax loss in arXiv:2301.05217v1, appendix competition explanation;
perfect decisions do not make this loss zero at a finite logit. -/
theorem binaryCE_true_pos (s : ℝ) : 0 < binaryCE true s := by
  rw [binaryCE_true_eq_log]
  apply Real.log_pos
  have h := Real.exp_pos (-s)
  linarith

/-- A fixed bound used in the exact diagnostic counterexample. Source:
the binary CE specialization of arXiv:2301.05217v1's appendix; no
floating-point approximation to log two enters the comparison. -/
theorem binary_axis_loss_below_one : Real.log 2 < 1 := by
  have he : 2 < Real.exp (1 : ℝ) := by
    have h := Real.add_one_lt_exp (x := (1 : ℝ)) (by norm_num)
    norm_num at h
    exact h
  have h := Real.log_lt_log (by norm_num : (0 : ℝ) < 2) he
  simpa using h

/-- Mean actual CE of two examples with explicitly additive scores.
Source comparison: arXiv:2301.05217v1 appendix compositional hypothesis;
deviation: fixed two-example binary task, with no learned product logit. -/
noncomputable def additivePairLoss (a b : ℝ) : ℝ :=
  (binaryCE true (2 * a - b) + binaryCE true (2 * b - a)) / 2

/-- Four actual losses when either additive component is set to zero.
Source: the explicit diagnostic counterexample to identifying circuits
from an arXiv:2301.05217v1-inspired removal measurement. -/
theorem additivePairLoss_corners :
    additivePairLoss 1 1 = binaryCE true 1 ∧
      additivePairLoss 0 1 = (binaryCE true 2 + binaryCE true (-1)) / 2 ∧
      additivePairLoss 1 0 = (binaryCE true 2 + binaryCE true (-1)) / 2 ∧
      additivePairLoss 0 0 = Real.log 2 := by
  unfold additivePairLoss
  norm_num only [mul_one, mul_zero, sub_zero, zero_sub, sub_self, zero_mul]
  have hz : binaryCE true 0 = Real.log 2 := by
    rw [binaryCE_true_eq_log]
    norm_num
  rw [hz]
  constructor
  · ring
  constructor
  · ring
  constructor
  · trivial
  · ring

/-- Both individual removals hurt despite the additive logits. Source:
arXiv:2301.05217v1 appendix composition hypothesis motivates removal;
the stronger usefulness check alone still does not identify a product. -/
theorem additivePair_individual_removals_hurt :
    additivePairLoss 1 1 < additivePairLoss 0 1 ∧
      additivePairLoss 1 1 < additivePairLoss 1 0 := by
  obtain ⟨hb, ha, hc, _⟩ := additivePairLoss_corners
  rw [hb, ha, hc, binaryCE_true_neg]
  have hp := binaryCE_true_pos 2
  have hs : binaryCE true 1 < 1 := by
    have h := coupledLoss_below_axes 1 1 (by norm_num)
    have hh : binaryCE true 1 < Real.log 2 := by simpa [coupledLoss] using h
    linarith [binary_axis_loss_below_one]
  constructor <;> linarith

/-- Same raw four-corner contrast as the head-removal measurement.
Source: arXiv:2301.05217v1-inspired diagnostic; both component arguments
enter actual mean CE, and a desired sign is not built into the definition. -/
noncomputable def additivePairInteraction (a b : ℝ) : ℝ :=
  additivePairLoss a b - additivePairLoss 0 b - additivePairLoss a 0 + additivePairLoss 0 0

/-- The actual contrast is negative although the scores are additive.
Source comparison: arXiv:2301.05217v1 appendix compositional hypothesis;
this is a measurement-identification counterexample, not a source claim. -/
theorem additivePairInteraction_negative : additivePairInteraction 1 1 < 0 := by
  unfold additivePairInteraction
  obtain ⟨hb, ha, hc, hz⟩ := additivePairLoss_corners
  rw [hb, ha, hc, hz, binaryCE_true_neg]
  have hp := binaryCE_true_pos 2
  linarith [binary_axis_loss_below_one]

/-- Both actual score functions have zero mixed partial derivatives.
Source: the explicitly additive diagnostic specialization above, compared
with arXiv:2301.05217v1's product/circuit interpretation. The nonlinearity
of CE and aggregation remain, while no score product is learned here. -/
theorem additive_scores_zero_cross_derivative (a b : ℝ) :
    deriv (fun u => deriv (fun v => 2 * u - v) b) a = 0 ∧
      deriv (fun u => deriv (fun v => 2 * v - u) b) a = 0 := by
  have hfirst (u : ℝ) : deriv (fun v => 2 * u - v) b = -1 := by
    convert ((hasDerivAt_id b).const_sub (2 * u)).deriv using 1
    norm_num
  have hsecond (u : ℝ) : deriv (fun v => 2 * v - u) b = 2 := by
    convert (((hasDerivAt_id b).const_mul 2).sub_const u).deriv using 1
    · norm_num
    · norm_num
  simp only [hfirst, hsecond]
  exact ⟨(hasDerivAt_const a (-1 : ℝ)).deriv, (hasDerivAt_const a (2 : ℝ)).deriv⟩

/-- Four-corner contrast before applying CE or averaging examples.
Source comparison: arXiv:2301.05217v1 appendix compositional hypothesis;
this separates output-score interaction from the nonlinear loss effect. -/
def scoreInteraction (score : ℝ → ℝ → ℝ) (a b : ℝ) : ℝ :=
  score a b - score 0 b - score a 0 + score 0 0

/-- Both additive score contrasts vanish for every component state.
Source: the explicit diagnostic counterexample above, compared with
arXiv:2301.05217v1's circuit hypothesis; CE contrast can still be negative. -/
theorem additive_score_contrasts_zero (a b : ℝ) :
    scoreInteraction (fun x y => 2 * x - y) a b = 0 ∧
      scoreInteraction (fun x y => 2 * y - x) a b = 0 := by
  unfold scoreInteraction
  constructor <;> ring

/-- The bilinear specialization has a genuine score interaction.
Source: arXiv:2301.05217v1 appendix composition hypothesis, explicit
scalar-product specialization. A nonzero contrast does not identify the
algorithm of a general transformer with other nonlinear components. -/
theorem bilinear_score_contrast_eq (a b : ℝ) :
    scoreInteraction (fun x y => x * y) a b = a * b := by
  unfold scoreInteraction
  ring

/-- Negative contrast, both useful removals, and zero mixed score
derivatives coexist in one actual supervised objective. Source comparison:
arXiv:2301.05217v1 appendix composition hypothesis; these measurements
therefore do not by themselves identify multiplicative learned logits. -/
theorem useful_negative_contrast_with_additive_scores :
    additivePairInteraction 1 1 < 0 ∧
      additivePairLoss 1 1 < additivePairLoss 0 1 ∧
      additivePairLoss 1 1 < additivePairLoss 1 0 ∧
      deriv (fun u : ℝ => deriv (fun v => 2 * u - v) 1) 1 = 0 ∧
      deriv (fun u : ℝ => deriv (fun v => 2 * v - u) 1) 1 = 0 := by
  exact ⟨additivePairInteraction_negative, additivePair_individual_removals_hurt.1,
    additivePair_individual_removals_hurt.2, additive_scores_zero_cross_derivative 1 1⟩

end Transformer.Grokking.Composition
