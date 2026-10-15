/-
# Exact scalar averages for the original multi-head iteration

ArXiv:2305.05465v6, §12, `sec:conclusion`, the multi-head display,
and §7, `l:logsumexp` and `l:distnondec`.
In dimension one each original head has score coefficient
`c_h=⟨Q_h e,K_h e⟩` and value coefficient `v_h=(V_h e)_0`.
Its scalar average is computed from the original softmax weights.

The actual head average is monotone in the query when `c_h≥0`.
The proof uses the already proved softmax monotonicity, retaining
each head's own coefficient. Nonnegative value coefficients preserve
this monotonicity under the original finite head sum. Each average
also lies in the original token interval, and the summed drift lies
in that interval multiplied by the sum of the value coefficients.
These are estimates for the exact multi-head field; no common head
temperature, replacement dynamics, or limiting weights are assumed.
-/

import Transformer.Clusters.Section7_DistNonDec
import Transformer.Clusters.Section7_BooleanScaling

open scoped BigOperators
open Real Set

namespace Transformer.Clusters

variable {n H : ℕ}

/-- One original head's scalar average, with its own query-key coefficient.
Source: arXiv:2305.05465v6, §12, `sec:conclusion`, the inner head sum. -/
noncomputable def scalarHeadAverage (c : ℝ) (y : Idx n → ℝ) (r : ℝ) : ℝ :=
  ∑ j, Perspective.softmaxWeight (fun l => c * r * y l) j * y j

/-- The actual one-dimensional attention row gives precisely this scalar average.
Source: arXiv:2305.05465v6, §12, the multi-head display, read through §7's scalar reduction. -/
theorem attention_scalarHeadAverage (Q K : ParamMatrix 1) (Y : Idx n → EucSpace 1) (i : Idx n) :
    (∑ j, attentionMatrix Q K Y i j * Y j 0) =
      scalarHeadAverage (inner (𝕜 := ℝ) (Q unit1) (K unit1)) (fun j => Y j 0) (Y i 0) := by
  simp only [scalarHeadAverage, attentionMatrix_eq_one_dim, mul_assoc]

/-- Increasing the original scalar query cannot decrease its head average.
Source: arXiv:2305.05465v6, §7, `l:logsumexp` and `l:distnondec`,
with the original head coefficient kept explicit as in §12. -/
theorem scalarHeadAverage_monotone (c : ℝ) (hc : 0 ≤ c) (y : Idx n → ℝ) :
    Monotone (scalarHeadAverage c y) := by
  intro r s hrs
  rcases eq_or_lt_of_le hc with hc0 | hcpos
  · subst c
    simp only [scalarHeadAverage, zero_mul]
    exact le_rfl
  rcases eq_or_lt_of_le hrs with rfl | hlt
  · exact le_rfl
  have hs := softmax_monotone (fun j => c * s * y j) (fun j => c * r * y j)
  have he : (∑ j, (Perspective.softmaxWeight (fun l => c * s * y l) j -
      Perspective.softmaxWeight (fun l => c * r * y l) j) * (c * s * y j - c * r * y j)) =
      (c * (s - r)) * (scalarHeadAverage c y s - scalarHeadAverage c y r) := by
    simp only [scalarHeadAverage, ← Finset.sum_sub_distrib, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j hj
    ring
  rw [he] at hs
  exact sub_nonneg.mp ((mul_nonneg_iff_of_pos_left (mul_pos hcpos (sub_pos.mpr hlt))).mp hs)

/-- Every actual head output lies in the original scalar token interval.
Source: arXiv:2305.05465v6, §7, the average bounds used by `l:exactasymptotic`,
for the separate original heads of §12. -/
theorem scalarHeadAverage_mem_Icc (hn : 0 < n) (c : ℝ) (y : Idx n → ℝ) (r L U : ℝ)
    (hy : ∀ j, y j ∈ Icc L U) : scalarHeadAverage c y r ∈ Icc L U := by
  have hs := Perspective.sum_softmaxWeight hn (fun j => c * r * y j)
  constructor
  · calc
      L = ∑ j, Perspective.softmaxWeight (fun l => c * r * y l) j * L := by
        rw [← Finset.sum_mul, hs, one_mul]
      _ ≤ _ := Finset.sum_le_sum fun j hj =>
        mul_le_mul_of_nonneg_left (hy j).1 (Perspective.softmaxWeight_nonneg _ _)
  · calc
      _ ≤ ∑ j, Perspective.softmaxWeight (fun l => c * r * y l) j * U :=
        Finset.sum_le_sum fun j hj =>
          mul_le_mul_of_nonneg_left (hy j).2 (Perspective.softmaxWeight_nonneg _ _)
      _ = U := by rw [← Finset.sum_mul, hs, one_mul]

/-- A single original coordinate value is preserved by every normalized head average.
Source: arXiv:2305.05465v6, §12, the normalized head weights. -/
theorem scalarHeadAverage_const (hn : 0 < n) (c a r : ℝ) :
    scalarHeadAverage c (fun _ : Idx n => a) r = a := by
  have h := scalarHeadAverage_mem_Icc hn c (fun _ => a) r a a (fun _ => ⟨le_rfl, le_rfl⟩)
  exact le_antisymm h.2 h.1

/-- A zero original query reads the unweighted empirical mean.
Source: arXiv:2305.05465v6, §12, the original scalar head formula at zero query. -/
theorem scalarHeadAverage_zero (c : ℝ) (y : Idx n → ℝ) :
    scalarHeadAverage c y 0 = (n : ℝ)⁻¹ * ∑ j, y j := by
  simp only [scalarHeadAverage, mul_zero, zero_mul, Perspective.softmaxWeight, Real.exp_zero,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one, one_div]
  rw [← Finset.mul_sum]

/-- Simultaneously reversing the original tokens and query reverses the head output.
Source: arXiv:2305.05465v6, §7, `l:distnondec`, and §12, the separate head scores. -/
theorem scalarHeadAverage_neg (c : ℝ) (y : Idx n → ℝ) (r : ℝ) :
    scalarHeadAverage c (fun j => -y j) (-r) = -scalarHeadAverage c y r := by
  simp only [scalarHeadAverage, mul_neg, neg_mul, neg_neg, Finset.sum_neg_distrib]

/-- The exact finite sum of the original scalar head drifts.
Source: arXiv:2305.05465v6, §12, `sec:conclusion`, the sum over heads. -/
noncomputable def multiHeadScalarDrift (c v : Idx H → ℝ) (y : Idx n → ℝ) (r : ℝ) : ℝ :=
  ∑ h, v h * scalarHeadAverage (c h) y r

/-- Positive query-key and value coefficients make the original summed drift monotone.
Source: arXiv:2305.05465v6, §12, the multi-head display under §2's positive scalar hypotheses. -/
theorem multiHeadScalarDrift_monotone (c v : Idx H → ℝ) (hc : ∀ h, 0 ≤ c h)
    (hv : ∀ h, 0 ≤ v h) (y : Idx n → ℝ) : Monotone (multiHeadScalarDrift c v y) := by
  intro r s hrs
  exact Finset.sum_le_sum fun h hh =>
    mul_le_mul_of_nonneg_left (scalarHeadAverage_monotone (c h) (hc h) y hrs) (hv h)

/-- The actual summed drift retains the original token interval times the total value coefficient.
Source: arXiv:2305.05465v6, §12, the original multi-head drift and §7's average bounds. -/
theorem multiHeadScalarDrift_mem_Icc (hn : 0 < n) (c v : Idx H → ℝ) (hv : ∀ h, 0 ≤ v h)
    (y : Idx n → ℝ) (r L U : ℝ) (hy : ∀ j, y j ∈ Icc L U) :
    multiHeadScalarDrift c v y r ∈ Icc ((∑ h, v h) * L) ((∑ h, v h) * U) := by
  constructor
  · rw [Finset.sum_mul]
    exact Finset.sum_le_sum fun h hh => mul_le_mul_of_nonneg_left
      (scalarHeadAverage_mem_Icc hn (c h) y r L U hy).1 (hv h)
  · rw [Finset.sum_mul]
    exact Finset.sum_le_sum fun h hh => mul_le_mul_of_nonneg_left
      (scalarHeadAverage_mem_Icc hn (c h) y r L U hy).2 (hv h)

/-- The source's genuine matrix hypotheses imply positive scalar head coefficients.
Source: arXiv:2305.05465v6, §2, `t:boolean`, applied separately to the heads of §12. -/
theorem scalar_head_parameters_pos (Q K V : ParamMatrix 1) (hV : IsPosDefOp V)
    (hQK : IsPosDefQK Q K) :
    0 < inner (𝕜 := ℝ) (Q unit1) (K unit1) ∧ 0 < (V unit1) 0 := by
  have hu : unit1 ≠ 0 := fun h => by simpa using congrArg norm h
  refine ⟨hQK.2 unit1 hu, ?_⟩
  have hv := hV unit1 hu
  rw [eq_smul_unit1 (V unit1), real_inner_smul_left, real_inner_self_eq_norm_sq, norm_unit1] at hv
  simpa using hv

/-- Two nonzero, distinct original scalar tokens and two different positive
head coefficients jointly realize every scalar monotonicity and interval premise. -/
example : ∃ (y : Idx 2 → ℝ) (c v : Idx 2 → ℝ),
    (0 : ℕ) < 2 ∧ (∀ j, y j ∈ Icc (-1 : ℝ) 1) ∧ y 0 < y 1 ∧
    (∀ h, 0 ≤ c h) ∧ (∀ h, 0 ≤ v h) ∧ c 0 ≠ c 1 ∧ v 0 ≠ v 1 := by
  refine ⟨![-1, 1], ![1, 2], ![1, 3], by decide, ?_, by norm_num, ?_, ?_, by norm_num, by norm_num⟩
  · intro j
    fin_cases j <;> norm_num
  · intro h
    fin_cases h <;> norm_num
  · intro h
    fin_cases h <;> norm_num

/-- Actual identity head matrices realize the original strict matrix hypotheses. -/
example : IsPosDefOp (1 : ParamMatrix 1) ∧ IsPosDefQK (1 : ParamMatrix 1) 1 :=
  ⟨isPosDefOp_id 1, isPosDefQK_of_isAttentionRoot (isAttentionRoot_id 1)⟩

end Transformer.Clusters
