module

public import Transformer.RoPE.Scores
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Learnable RoPE frequencies

Karypis et al., *LeRoPE*, arXiv:2607.10134v1, §3.1, Eq. (5), and §3.2.
The learned frequencies are positive and can reach any positive target.
The derivative is the direct local score contribution with query/key
coefficients held fixed, as required in the source's backpropagation formula.
-/

@[expose] public section

namespace Transformer.RoPE.LeRoPE

/-- LeRoPE Eq. (5), with a learned logarithmic scale for each frequency. -/
noncomputable def learnedFrequency (omega alpha : ℝ) : ℝ := Real.exp alpha * omega

/-- §3.1: positivity is preserved for every finite value of the learned scale. -/
theorem learned_frequency_pos {omega : ℝ} (homega : 0 < omega) (alpha : ℝ) :
    0 < learnedFrequency omega alpha := by
  exact mul_pos (Real.exp_pos alpha) homega

example : 0 < learnedFrequency 1 (-100) := learned_frequency_pos (by norm_num) (-100)

/-- §3.1: zero initialization recovers the original frequency grid. -/
theorem learned_frequency_init (omega : ℝ) : learnedFrequency omega 0 = omega := by
  simp [learnedFrequency]

/-- Eq. (5): logarithmic parametrization can represent any positive target
frequency. It does not require knowing the target dependency width in advance. -/
theorem learned_frequency_surjective {omega target : ℝ}
    (homega : 0 < omega) (htarget : 0 < target) :
    ∃ alpha, learnedFrequency omega alpha = target := by
  refine ⟨Real.log (target / omega), ?_⟩
  unfold learnedFrequency
  rw [Real.exp_log (div_pos htarget homega), div_mul_cancel₀ _ homega.ne']

example : ∃ alpha, learnedFrequency 1 (alpha : ℝ) = Real.pi / 4095 :=
  learned_frequency_surjective (by norm_num) (by positivity)

/-- §3.1: equal parameter increments produce equal multiplicative frequency
changes, independently of the current scale. -/
theorem learned_frequency_increment (omega alpha step : ℝ) :
    learnedFrequency omega (alpha + step) = Real.exp step * learnedFrequency omega alpha := by
  simp only [learnedFrequency, Real.exp_add]
  ring

/-- §3.2: derivative of the log-parametrized frequency. -/
theorem learned_frequency_hasDerivAt (omega alpha : ℝ) :
    HasDerivAt (learnedFrequency omega) (learnedFrequency omega alpha) alpha := by
  exact (Real.hasDerivAt_exp alpha).mul_const omega

/-- §3.2: the local rotary-score gradient carries both the current frequency
and the relative-distance factor. No downstream model derivative is hidden. -/
theorem score_log_frequency_hasDerivAt (a b omega distance alpha : ℝ) :
    HasDerivAt (fun value => score a b (learnedFrequency omega value) distance)
      (learnedFrequency omega alpha * distance *
        (-a * Real.sin (learnedFrequency omega alpha * distance) +
          b * Real.cos (learnedFrequency omega alpha * distance))) alpha := by
  have hphase := (learned_frequency_hasDerivAt omega alpha).mul_const distance
  convert (hphase.cos.const_mul a).add (hphase.sin.const_mul b) using 1 <;>
    first | (funext value; simp only [Pi.add_apply, score, mul_comm]) | ring

end Transformer.RoPE.LeRoPE
