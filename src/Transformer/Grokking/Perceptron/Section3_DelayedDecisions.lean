import Transformer.Grokking.Perceptron.Section3_SymmetricMSE

/-!
# Arbitrarily delayed classification under a convex gradient flow

Source: Žunkovič and Ilievski, arXiv:2210.15435v1, section 3,
perceptron grokking. This two-point specialization uses the actual sample
MSE and its verified gradient flow from `Section3_SymmetricMSE`. It has
no transformer layers, no stochastic optimizer and no thermodynamic limit.

The model starts at weight 1 and bias 1/2. Train inputs are ±1; held-out
inputs are ±δ with labels matching their signs. Positive scores, including
ties, predict the positive class. Choosing δ = exp(-T)/2 gives a derived
decision crossing at any prescribed T > 0. The task margin varies with T;
this is not arbitrary delay on one fixed task. Smooth scores and convex
loss therefore do not rule out delayed, abrupt finite-test accuracy.
-/

namespace Transformer.Grokking.Perceptron

/-- Actual accuracy of the two signed examples at ±a, using a positive
tie rule. Source specialization: arXiv:2210.15435v1, section 3. -/
noncomputable def pairAccuracy (w b a : ℝ) : ℝ :=
  ((if 0 ≤ score w b a then 1 else 0) +
    (if score w b (-a) < 0 then 1 else 0)) / 2

/-- A held-out margin determining the test task, not the training flow.
Source: explicit two-point variant of arXiv:2210.15435v1, section 3. -/
noncomputable def testMargin (T : ℝ) : ℝ := Real.exp (-T) / 2

/-- Held-out accuracy computed from the optimized perceptron's scores.
Source: the explicit variant of arXiv:2210.15435v1, section 3. -/
noncomputable def testAccuracy (T t : ℝ) : ℝ :=
  pairAccuracy (flowWeight 1 t) (flowBias (1 / 2) t) (testMargin T)

/-- Correct scores imply the actual two-point accuracy is one. Source:
arXiv:2210.15435v1, section 3, affine-sign classification. -/
theorem pairAccuracy_eq_one (w b a : ℝ)
    (hp : 0 ≤ score w b a) (hn : score w b (-a) < 0) :
    pairAccuracy w b a = 1 := by
  unfold pairAccuracy
  rw [ite_eq_left hp, ite_eq_left hn]
  norm_num

example : 0 ≤ score 1 0 1 ∧ score 1 0 (-1) < 0 := by
  unfold score
  norm_num

/-- One correct positive example and one incorrectly nonnegative negative
score give one half accuracy. Source: arXiv:2210.15435v1, section 3. -/
theorem pairAccuracy_eq_half (w b a : ℝ)
    (hp : 0 ≤ score w b a) (hn : 0 ≤ score w b (-a)) :
    pairAccuracy w b a = 1 / 2 := by
  unfold pairAccuracy
  rw [ite_eq_left hp, ite_eq_right (not_lt.mpr hn)]
  norm_num

example : 0 ≤ score 1 2 1 ∧ 0 ≤ score 1 2 (-1) := by
  unfold score
  norm_num

/-- For T > 0 the test margin is positive and smaller than the initial
bias. Source: the explicit variant of arXiv:2210.15435v1, section 3. -/
theorem testMargin_bounds (T : ℝ) (hT : 0 < T) :
    0 < testMargin T ∧ testMargin T < 1 / 2 := by
  have he : Real.exp (-T) < 1 := by
    have h := Real.exp_lt_exp.mpr (show -T < 0 by linarith)
    simpa only [Real.exp_zero] using h
  unfold testMargin
  constructor <;> nlinarith [Real.exp_pos (-T)]

example : (0 : ℝ) < 1 := by norm_num

/-- Training classification is already perfect throughout nonnegative
time, despite subsequent motion of the learned bias. Source:
arXiv:2210.15435v1, section 3, explicit two-point specialization. -/
theorem trainAccuracy_along_flow (t : ℝ) (ht : 0 ≤ t) :
    pairAccuracy (flowWeight 1 t) (flowBias (1 / 2) t) 1 = 1 := by
  have he : Real.exp (-t) ≤ 1 := by
    have h := Real.exp_le_exp.mpr (show -t ≤ 0 by linarith)
    simpa only [Real.exp_zero] using h
  apply pairAccuracy_eq_one
  · unfold score flowWeight flowBias
    nlinarith [Real.exp_pos (-t)]
  · unfold score flowWeight flowBias
    nlinarith

example : (0 : ℝ) ≤ 0 := by norm_num

/-- Before the boundary crossing, including the tie at T, the actual
held-out accuracy stays at one half. Source specialization:
arXiv:2210.15435v1, section 3; no prescribed step function is used. -/
theorem testAccuracy_before (T t : ℝ) (ht : t ≤ T) :
    testAccuracy T t = 1 / 2 := by
  have he : Real.exp (-T) ≤ Real.exp (-t) :=
    Real.exp_le_exp.mpr (by linarith)
  unfold testAccuracy
  apply pairAccuracy_eq_half
  · unfold score flowWeight flowBias testMargin
    nlinarith [Real.exp_pos (-T), Real.exp_pos (-t)]
  · unfold score flowWeight flowBias testMargin
    nlinarith

example : (0 : ℝ) ≤ 1 := by norm_num

/-- After the boundary crossing both held-out signs are correct. Source:
arXiv:2210.15435v1, section 3, the explicit variant stated above. -/
theorem testAccuracy_after (T t : ℝ) (ht : T < t) :
    testAccuracy T t = 1 := by
  have he : Real.exp (-t) < Real.exp (-T) :=
    Real.exp_lt_exp.mpr (by linarith)
  unfold testAccuracy
  apply pairAccuracy_eq_one
  · unfold score flowWeight flowBias testMargin
    nlinarith [Real.exp_pos (-T), Real.exp_pos (-t)]
  · unfold score flowWeight flowBias testMargin
    nlinarith

example : (1 : ℝ) < 2 := by norm_num

/-- Every positive delay is realized by a task margin on the same convex
training flow. Source: explicit variant of arXiv:2210.15435v1, section 3.
The quantified task family is essential; the claim is not arbitrary delay
for one fixed held-out set. -/
theorem arbitrarily_delayed_decisions (T : ℝ) (hT : 0 < T) :
    0 < testMargin T ∧ testMargin T < 1 / 2 ∧
      ∀ t, 0 ≤ t →
        pairAccuracy (flowWeight 1 t) (flowBias (1 / 2) t) 1 = 1 ∧
        (t ≤ T → testAccuracy T t = 1 / 2) ∧
        (T < t → testAccuracy T t = 1) := by
  obtain ⟨hp, hb⟩ := testMargin_bounds T hT
  refine ⟨hp, hb, ?_⟩
  intro t ht
  exact ⟨trainAccuracy_along_flow t ht, testAccuracy_before T t,
    testAccuracy_after T t⟩

example : (0 : ℝ) < 10 := by norm_num

/-- The negative test score is differentiable even at the accuracy jump.
Source specialization: arXiv:2210.15435v1, section 3. Abrupt finite-test
accuracy alone does not establish a nonsmooth learned score or a
thermodynamic singularity. -/
theorem heldoutScore_deriv (T t : ℝ) :
    HasDerivAt (fun s => score (flowWeight 1 s) (flowBias (1 / 2) s)
      (-testMargin T)) (-flowBias (1 / 2) t) t := by
  convert (flowBias_deriv (1 / 2) t).const_add (-testMargin T) using 1
  funext s
  unfold score flowWeight
  ring

/-- Counterexample to reading section 3's "by construction" grokking
wording as delayed generalization for every initialization on separable
data: zero initial bias already classifies every positive margin correctly.
Source: arXiv:2210.15435v1, section 3. The paper's distribution-specific
grokking probability and time results are not contradicted by this case. -/
theorem zero_bias_no_delay (a t : ℝ) (ha : 0 < a) :
    pairAccuracy (flowWeight 1 t) (flowBias 0 t) a = 1 := by
  apply pairAccuracy_eq_one
  · unfold score flowWeight flowBias
    nlinarith
  · unfold score flowWeight flowBias
    nlinarith

example : (0 : ℝ) < 1 / 4 := by norm_num

end Transformer.Grokking.Perceptron
