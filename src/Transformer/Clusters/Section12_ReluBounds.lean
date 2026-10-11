import Transformer.Clusters.Section12_ReluScalar

/-!
# An integrable drift bound and three frozen ReLU tokens

For the four-token configuration `(y, -1, 0, 1)` in §12,
`eq: mlp.transfo` of arXiv:2305.05465v6, assume `y ≤ -6`.
The moving token is the minimum, so its unactivated drift is
nonnegative. Its attention to every other token is suppressed by
its own larger score. At inverse temperature `b > 0`, the bound
`4(1-y) ≤ y²+y` and `exp(b(y²+y)) ≥ b(y²+y)` give drift at most
`1/b`. For the actual scores, `b = exp(2t)`, this is `exp(-2t)`.

The other three unactivated drifts are nonpositive. For the query
`-1`, the far negative token contributes at most `-5`, whereas
the contributions from `0` and `1` sum to at most `3`. The zero
query sees a uniform average, and the largest query has only
nonpositive differences. ReLU therefore freezes all three.

All estimates concern the literal exponential weights; no
asymptotic replacement of the attention matrix is used.
-/

open scoped BigOperators
open Real
namespace Transformer.Clusters

/-- The moving coordinate is the minimum in the four-token test of
§12, `eq: mlp.transfo`, when it lies below minus six. -/
theorem self_le_reluFourCoords {y : ℝ} (hy : y ≤ -6) (j : Idx 4) :
    y ≤ reluFourCoords y 1 j := by
  fin_cases j
  · exact le_rfl
  · change y ≤ -1
    linarith
  · change y ≤ 0
    linarith
  · change y ≤ 1
    linarith

example : (-8 : ℝ) ≤ -6 := by norm_num

/-- Every coordinate is at most the fixed largest token.
Source: arXiv:2305.05465v6, §12, the four-token test of `eq: mlp.transfo`. -/
theorem reluFourCoords_le_one {y : ℝ} (hy : y ≤ -6) (j : Idx 4) :
    reluFourCoords y 1 j ≤ 1 := by
  fin_cases j
  · change y ≤ 1
    linarith
  · norm_num [reluFourCoords]
  · norm_num [reluFourCoords]
  · exact le_rfl

example : (-8 : ℝ) ≤ -6 := by norm_num

/-- The actual moving token has nonnegative unactivated drift.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`, positivity
of attention and the minimum-token property. -/
theorem reluFourDrift_zero_nonneg (b : ℝ) {y : ℝ} (hy : y ≤ -6) :
    0 ≤ reluFourDrift b y 1 0 := by
  apply Finset.sum_nonneg
  intro j hj
  exact mul_nonneg (Perspective.softmaxWeight_nonneg _ j)
    (sub_nonneg.mpr (self_le_reluFourCoords hy j))

example : (-8 : ℝ) ≤ -6 := by norm_num

/-- The minimum token's attention drift is bounded by inverse temperature.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`;
the self score dominates the four-token test configuration. -/
theorem reluFourDrift_zero_le_inv {b y : ℝ} (hb : 0 < b) (hy : y ≤ -6) :
    reluFourDrift b y 1 0 ≤ b⁻¹ := by
  let L := y ^ 2 + y
  have hL : 4 * (1 - y) ≤ L := by dsimp [L]; nlinarith [sq_nonneg (y + 6)]
  have hLpos : 0 < L := by linarith
  have hterm (j : Idx 4) :
      Perspective.softmaxWeight (fun k => b * y * reluFourCoords y 1 k) j *
          (reluFourCoords y 1 j - y) ≤ Real.exp (-(b * L)) * (1 - y) := by
    by_cases hj : j = 0
    · subst j
      simp only [reluFourCoords_zero, sub_self, mul_zero]
      exact mul_nonneg (Real.exp_pos _).le (by linarith)
    · have hv : -1 ≤ reluFourCoords y 1 j := by
        fin_cases j <;> simp_all [reluFourCoords]
      have hscore : b * y * (reluFourCoords y 1 j - y) ≤ -(b * L) := by
        have hmul := mul_le_mul_of_nonpos_left hv
          (mul_nonpos_of_nonneg_of_nonpos hb.le (by linarith : y ≤ 0))
        dsimp [L]
        nlinarith
      have hw : Perspective.softmaxWeight (fun k => b * y * reluFourCoords y 1 k) j ≤
          Real.exp (-(b * L)) := by
        calc
          _ ≤ Real.exp (b * y * reluFourCoords y 1 j - b * y * y) :=
            softmaxWeight_le_exp_sub _ 0 j
          _ ≤ _ := Real.exp_le_exp.mpr (by nlinarith [hscore])
      exact mul_le_mul hw (by linarith [reluFourCoords_le_one hy j])
        (sub_nonneg.mpr (self_le_reluFourCoords hy j)) (Real.exp_pos _).le
  have hsum : reluFourDrift b y 1 0 ≤ 4 * (1 - y) * Real.exp (-(b * L)) := by
    calc
      _ ≤ ∑ j : Idx 4, Real.exp (-(b * L)) * (1 - y) :=
        Finset.sum_le_sum (fun j hj => hterm j)
      _ = _ := by simp; ring
  have he := Real.add_one_le_exp (b * L)
  have hexp : Real.exp (b * L) * Real.exp (-(b * L)) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hprod : b * L * Real.exp (-(b * L)) ≤ 1 := by
    calc
      _ ≤ Real.exp (b * L) * Real.exp (-(b * L)) :=
        mul_le_mul_of_nonneg_right (by linarith) (Real.exp_pos _).le
      _ = 1 := hexp
  have hinv : L * Real.exp (-(b * L)) ≤ b⁻¹ := by
    rw [inv_eq_one_div, le_div_iff₀ hb]
    nlinarith [hprod]
  exact hsum.trans ((mul_le_mul_of_nonneg_right hL (Real.exp_pos _).le).trans hinv)

example : (0 : ℝ) < 1 ∧ (-8 : ℝ) ≤ -6 := by norm_num

/-- With the actual time-dependent scores, the moving drift has an
integrable exponential bound. Source: arXiv:2305.05465v6, §12,
`eq: mlp.transfo`, specialized to identity matrices in dimension one. -/
theorem reluFourDrift_zero_le_exp (t : ℝ) {y : ℝ} (hy : y ≤ -6) :
    reluFourDrift (Real.exp (2 * t)) y 1 0 ≤ Real.exp (-(2 * t)) := by
  rw [Real.exp_neg]
  exact reluFourDrift_zero_le_inv (Real.exp_pos _) hy

example : (-8 : ℝ) ≤ -6 := by norm_num

/-- The query at minus one has nonpositive drift, so ReLU freezes it.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`,
the dominant contribution of the far negative token. -/
theorem reluFourDrift_one_nonpos {b y : ℝ} (hb : 0 < b) (hy : y ≤ -6) :
    reluFourDrift b y 1 1 ≤ 0 := by
  have hx : 1 ≤ Real.exp (-(b * y)) := Real.one_le_exp_iff.mpr
    (by nlinarith [mul_nonpos_of_nonneg_of_nonpos hb.le (by linarith : y ≤ 0)])
  have hz : Real.exp (-b) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hneg : Real.exp (-(b * y)) * (y + 1) ≤ -5 := by
    calc
      _ ≤ Real.exp (-(b * y)) * (-5) :=
        mul_le_mul_of_nonneg_left (by linarith) (Real.exp_pos _).le
      _ ≤ _ := by nlinarith
  unfold reluFourDrift Perspective.softmaxWeight
  simp only [div_mul_eq_mul_div, ← Finset.sum_div]
  apply div_nonpos_of_nonpos_of_nonneg
  · simp [Fin.sum_univ_four, reluFourCoords]
    nlinarith
  · exact Finset.sum_nonneg (fun j hj => (Real.exp_pos _).le)

example : (0 : ℝ) < 1 ∧ (-8 : ℝ) ≤ -6 := by norm_num

/-- The zero query sees the negative uniform average and is frozen by ReLU.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`. -/
theorem reluFourDrift_two_nonpos (b : ℝ) {y : ℝ} (hy : y ≤ -6) :
    reluFourDrift b y 1 2 ≤ 0 := by
  simp [reluFourDrift, Perspective.softmaxWeight, reluFourCoords, Fin.sum_univ_four]
  linarith

example : (-8 : ℝ) ≤ -6 := by norm_num

/-- The largest token has nonpositive differences and is frozen by ReLU.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`. -/
theorem reluFourDrift_three_nonpos (b : ℝ) {y : ℝ} (hy : y ≤ -6) :
    reluFourDrift b y 1 3 ≤ 0 := by
  apply Finset.sum_nonpos
  intro j hj
  exact mul_nonpos_of_nonneg_of_nonpos (Perspective.softmaxWeight_nonneg _ j)
    (sub_nonpos.mpr (reluFourCoords_le_one hy j))

example : (-8 : ℝ) ≤ -6 := by norm_num

end Transformer.Clusters
