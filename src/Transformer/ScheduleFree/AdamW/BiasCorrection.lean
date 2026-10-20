module

public import Transformer.ScheduleFree.AdamW.Invariants

/-!
# What moving AdamW's bias correction changes

Source: facebookresearch/schedule_free, commit `59ed9b662d5ec5799aded7390b2482758423653c`,
the changes to `lr` and `denom` in both branches of `adamw_schedulefree.py`.
Compare arXiv:2405.15682v4, §4.4, Algorithm 1, and the preserved implementation
`adamw_schedulefree_paper.py`. Writing `b=1-beta2^(t+1)` and `eta=lr*warmup`,
the revised gradient step is `eta*g/(sqrt(v/b)+eps)`. In the old rate convention
this is `eta*sqrt(b)*g/(sqrt(v)+eps*sqrt(b))`. Thus moving bias correction
also rescales epsilon; equality with the paper's gradient step requires eps=0
or another degeneracy. Meanwhile the decay coefficient changes from
`eta*sqrt(b)*lambda` to `eta*lambda`. These algebraic facts do not prove a
general convergence or floating-point stability claim.

The comparison fixes a common variance, query, and gradient. Subsequent
trajectories can differ because their averaging weights also change.
For the default powers and no warmup, the weights already differ on
the second update. `PaperComparison` checks base and averaging differences
against the imported manuscript recurrence.
All variance and bias assumptions are explicit in the lemmas below;
positive epsilon covers zero-gradient and zero-variance inputs.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AdamW

variable {ι : Type*}

/-- Algorithm 1 multiplies the revised warmup-only rate by `sqrt(bias_correction2)`. -/
theorem paper_rate (p : Parameters) (t : ℕ) :
    adamLearningRate p.adam t = rate p t * Real.sqrt (correction p.adam.betaTwo t) := by
  dsimp [adamLearningRate, rate, correction]
  ring

/-- The revised normalization equals the old-rate form with a rescaled epsilon. -/
theorem corrected_gradient (η g v ε b : ℝ) (hv : 0 ≤ v) (hε : 0 < ε) (hb : 0 < b) :
    η * g / (Real.sqrt (v / b) + ε) =
      η * Real.sqrt b * g / (Real.sqrt v + ε * Real.sqrt b) := by
  have hs : 0 < Real.sqrt b := Real.sqrt_pos.mpr hb
  have hleft : Real.sqrt v / Real.sqrt b + ε ≠ 0 :=
    ne_of_gt (add_pos_of_nonneg_of_pos (div_nonneg (Real.sqrt_nonneg v) hs.le) hε)
  have hright : Real.sqrt v + ε * Real.sqrt b ≠ 0 :=
    ne_of_gt (add_pos_of_nonneg_of_pos (Real.sqrt_nonneg v) (mul_pos hε hs))
  rw [Real.sqrt_div hv]
  field_simp

example :
    (1 : ℝ) * 2 / (Real.sqrt (4 / (1/4)) + 1) =
      1 * Real.sqrt (1/4) * 2 / (Real.sqrt 4 + 1 * Real.sqrt (1/4)) :=
  corrected_gradient _ _ _ _ _ (by norm_num) zero_lt_one (by norm_num)

/-- With epsilon zero and positive variance, the two gradient steps coincide. -/
theorem gradient_without_epsilon (η g v b : ℝ) (hv : 0 < v) (hb : 0 < b) :
    η * g / Real.sqrt (v / b) = η * Real.sqrt b * g / Real.sqrt v := by
  have hsv : Real.sqrt v ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hv)
  have hsb : Real.sqrt b ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hb)
  rw [Real.sqrt_div hv.le]
  field_simp

example : (1 : ℝ) * 2 / Real.sqrt (4 / (1/4)) =
    1 * Real.sqrt (1/4) * 2 / Real.sqrt 4 :=
  gradient_without_epsilon _ _ _ _ (by norm_num) (by norm_num)

/-- For a nonnegative numerator and positive epsilon, the revised gradient step
is at least the paper step, because its equivalent epsilon is at most epsilon.
This rules out interpreting the change as uniformly shrinking gradient updates. -/
theorem gradient_comparison (η g v ε b : ℝ) (hη : 0 ≤ η) (hg : 0 ≤ g)
    (hv : 0 ≤ v) (hε : 0 < ε) (hb : 0 < b) (hb1 : b ≤ 1) :
    η * Real.sqrt b * g / (Real.sqrt v + ε) ≤
      η * g / (Real.sqrt (v / b) + ε) := by
  rw [corrected_gradient η g v ε b hv hε hb]
  apply div_le_div_of_nonneg_left
  · exact mul_nonneg (mul_nonneg hη (Real.sqrt_nonneg b)) hg
  · exact add_pos_of_nonneg_of_pos (Real.sqrt_nonneg v)
      (mul_pos hε (Real.sqrt_pos.mpr hb))
  · have hs : Real.sqrt b ≤ 1 := Real.sqrt_le_one.mpr hb1
    have he := mul_le_mul_of_nonneg_left hs hε.le
    linarith

example : (1 : ℝ) * Real.sqrt (1/4) * 2 / (Real.sqrt 4 + 1) ≤
    1 * 2 / (Real.sqrt (4 / (1/4)) + 1) :=
  gradient_comparison _ _ _ _ _ zero_le_one (by norm_num) (by norm_num)
    zero_lt_one (by norm_num) (by norm_num)

/-- On the first update the corrected denominator is `abs(g)+eps`, independent
of beta2. This follows from the actual initialized raw-variance recurrence. -/
theorem first_denominator (p : Parameters) (point g : ι → ℝ)
    (hβ : 0 ≤ p.adam.betaTwo) (hβ1 : p.adam.betaTwo < 1) (i : ι) :
    denominator p 0 (initial point).history g i = |g i| + p.adam.epsilon := by
  have hb := (correction_bounds p.adam.betaTwo 0 hβ hβ1).1.ne'
  have hv : nextSecond p (initial point).history g i / correction p.adam.betaTwo 0 =
      g i ^ 2 := by
    dsimp [nextSecond, initial, correction] at *
    field_simp
    ring
  unfold denominator
  rw [hv, Real.sqrt_sq_eq_abs]

example : denominator ⟨⟨1, 0, 0, 9/10, 999/1000, 1⟩, 0, 2⟩ 0
    (initial (fun _ : Fin 1 => 2)).history (fun _ => -3) 0 = 4 := by
  rw [first_denominator _ _ _ (by norm_num) (by norm_num)]
  norm_num

/-- The initialized revised base step uses `g/(abs(g)+eps)` and decay at the
initial point. In particular no square root of the first bias correction
attenuates the decay term, even before warmup has finished. -/
theorem first_base (p : Parameters) (point g : ι → ℝ)
    (hβ : 0 ≤ p.adam.betaTwo) (hβ1 : p.adam.betaTwo < 1) (i : ι) :
    (step p 0 (initial point) g).base i =
      point i - rate p 0 * (g i / (|g i| + p.adam.epsilon) + p.adam.decay * point i) := by
  change point i - rate p 0 *
    (g i / denominator p 0 (initial point).history g i +
      p.adam.decay * query p (initial point) i) = _
  rw [first_denominator p point g hβ hβ1, query_initial]

example : (step ⟨⟨1, 1/2, 2, 9/10, 999/1000, 1⟩, 0, 2⟩ 0
    (initial (fun _ : Fin 1 => 2)) (fun _ => 3)).base 0 = 9/8 := by
  rw [first_base _ _ _ (by norm_num) (by norm_num)]
  norm_num [rate, warmupFactor]

/-- The manuscript attenuates decay by `sqrt(b)`; the revised code applies the
full warmup-only AdamW decay coefficient. The bounds concern coefficients,
not a contraction of the full coupled `x,y,z` dynamics. -/
theorem paper_decay_le_revised (p : Parameters) (t : ℕ)
    (hlr : 0 ≤ p.adam.learningRate) (hdecay : 0 ≤ p.adam.decay)
    (hβ : 0 ≤ p.adam.betaTwo) (hβ1 : p.adam.betaTwo < 1) :
    0 ≤ adamLearningRate p.adam t * p.adam.decay ∧
      adamLearningRate p.adam t * p.adam.decay ≤ rate p t * p.adam.decay := by
  rw [paper_rate]
  constructor
  · exact mul_nonneg (mul_nonneg (rate_nonneg p t hlr) (Real.sqrt_nonneg _)) hdecay
  · have hs := Real.sqrt_le_one.mpr (correction_bounds _ t hβ hβ1).2
    have hrate := mul_le_mul_of_nonneg_left hs (rate_nonneg p t hlr)
    simpa only [mul_one] using mul_le_mul_of_nonneg_right hrate hdecay

example (t : ℕ) :
    0 ≤ adamLearningRate ⟨1, 1/2, 10, 9/10, 999/1000, 1⟩ t * (1/2) ∧
      adamLearningRate ⟨1, 1/2, 10, 9/10, 999/1000, 1⟩ t * (1/2) ≤
        rate ⟨⟨1, 1/2, 10, 9/10, 999/1000, 1⟩, 0, 2⟩ t * (1/2) :=
  paper_decay_le_revised ⟨⟨1, 1/2, 10, 9/10, 999/1000, 1⟩, 0, 2⟩ t
    zero_le_one (by norm_num) (by norm_num) (by norm_num)

end Transformer.ScheduleFree.AdamW
