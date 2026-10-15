/-
# Quantitative pull of the original scalar heads

ArXiv:2305.05465v6, §7, `e:lowerboundxi` and the proof of
`l:exactasymptotic`, extended to the separate original heads of §12.
For query r>0 and any reference token y_M, each original head average
is at least y_M-n/(c r). The reference need not be the largest token:
terms above it contribute nonnegatively, and every term below it is
bounded with z exp(c r z) ≥ -1/(c r).

Summing with the actual nonnegative value coefficients yields the
original drift bound with total value coefficient Σv and error
n Σ(v/c)/r. At the largest token this also bounds the entire drift
error. Simultaneous sign reversal supplies the corresponding negative
query bound. No escape or limiting configuration is assumed here.
-/

import Transformer.Clusters.Section12_MultiHeadScalar
import Transformer.Clusters.Section7_UnboundedParticles

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {m H : ℕ}

/-- Every original head average is at most the largest original token.
Source: arXiv:2305.05465v6, §7, the average bound in the proof of `l:exactasymptotic`. -/
theorem scalarHeadAverage_le_max (c : ℝ) (y : Idx (m + 1) → ℝ) (r : ℝ)
    (M : Idx (m + 1)) (hM : ∀ j, y j ≤ y M) : scalarHeadAverage c y r ≤ y M := by
  calc
    _ ≤ ∑ j, Perspective.softmaxWeight (fun l => c * r * y l) j * y M :=
      Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hM j) (Perspective.softmaxWeight_nonneg _ _)
    _ = _ := by rw [← Finset.sum_mul, Perspective.sum_softmaxWeight (Nat.succ_pos m), one_mul]

/-- A positive original query pulls toward any chosen original reference token.
Source: arXiv:2305.05465v6, §7, `e:lowerboundxi` and proof of `l:exactasymptotic`;
the retained head coefficient changes the error to n/(c r), as prescribed by §12. -/
theorem scalarHeadAverage_ge_reference (c : ℝ) (hc : 0 < c)
    (y : Idx (m + 1) → ℝ) (r : ℝ) (hr : 0 < r) (M : Idx (m + 1)) :
    y M - (m + 1) / (c * r) ≤ scalarHeadAverage c y r := by
  have hs := Perspective.sum_softmaxWeight (Nat.succ_pos m) (fun l => c * r * y l)
  have he : scalarHeadAverage c y r = y M +
      ∑ j, Perspective.softmaxWeight (fun l => c * r * y l) j * (y j - y M) := by
    simp only [scalarHeadAverage, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hs, one_mul]
    ring
  have hj : ∀ j, -(1 / (c * r)) ≤
      Perspective.softmaxWeight (fun l => c * r * y l) j * (y j - y M) := by
    intro j
    rcases le_or_gt (y j) (y M) with hle | hgt
    · have hw := softmaxWeight_le_exp_sub (fun l => c * r * y l) M j
      have hex : c * r * y j - c * r * y M = (c * r) * (y j - y M) := by ring
      rw [hex] at hw
      exact (neg_inv_le_exp_mul_mul (mul_pos hc hr) (y j - y M)).trans
        (mul_le_mul_of_nonpos_right hw (sub_nonpos.mpr hle))
    · exact (neg_nonpos.mpr (by positivity)).trans
        (mul_nonneg (Perspective.softmaxWeight_nonneg _ _) (sub_nonneg.mpr hgt.le))
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hj j
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
  rw [he]
  push_cast at hsum ⊢
  have hneg : ((m : ℝ) + 1) * -(1 / (c * r)) = -((m + 1) / (c * r)) := by ring
  rw [hneg] at hsum
  linarith

/-- The error of an actual positive-query head relative to the largest token is nonnegative and small.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, for an original head of §12. -/
theorem scalarHeadAverage_max_error (c : ℝ) (hc : 0 < c)
    (y : Idx (m + 1) → ℝ) (r : ℝ) (hr : 0 < r)
    (M : Idx (m + 1)) (hM : ∀ j, y j ≤ y M) :
    0 ≤ y M - scalarHeadAverage c y r ∧ y M - scalarHeadAverage c y r ≤ (m + 1) / (c * r) := by
  have hle := scalarHeadAverage_le_max c y r M hM
  have hge := scalarHeadAverage_ge_reference c hc y r hr M
  constructor <;> linarith

/-- The exact original head sum retains the reference-token pull with its full coefficient-dependent error.
Source: arXiv:2305.05465v6, §12's displayed head sum and §7, `e:lowerboundxi`. -/
theorem multiHeadScalarDrift_ge_reference (c v : Idx H → ℝ) (hc : ∀ h, 0 < c h)
    (hv : ∀ h, 0 ≤ v h) (y : Idx (m + 1) → ℝ) (r : ℝ) (hr : 0 < r) (M : Idx (m + 1)) :
    (∑ h, v h) * y M - ((m + 1) * ∑ h, v h / c h) / r ≤ multiHeadScalarDrift c v y r := by
  have hsum := Finset.sum_le_sum fun h (_ : h ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (scalarHeadAverage_ge_reference (c h) (hc h) y r hr M) (hv h)
  have he : ∀ h, v h * (y M - (m + 1) / (c h * r)) =
      v h * y M - ((m + 1) / r) * (v h / c h) := by
    intro h
    field_simp [(hc h).ne', hr.ne']
  simp_rw [he] at hsum
  rw [Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.mul_sum] at hsum
  have hratio : ((m + 1) / r) * (∑ h, v h / c h) = ((m + 1) * ∑ h, v h / c h) / r := by ring
  rwa [hratio] at hsum

/-- The actual summed head drift reverses with simultaneous token and query reversal.
Source: arXiv:2305.05465v6, §7's sign symmetry, for the original separate heads of §12. -/
theorem multiHeadScalarDrift_neg (c v : Idx H → ℝ) (y : Idx (m + 1) → ℝ) (r : ℝ) :
    multiHeadScalarDrift c v (fun j => -y j) (-r) = -multiHeadScalarDrift c v y r := by
  simp only [multiHeadScalarDrift, scalarHeadAverage_neg, mul_neg, Finset.sum_neg_distrib]

/-- A negative original query has the reflected reference-token bound.
Source: arXiv:2305.05465v6, §7, the symmetric bound following `e:lowerboundxi`, for §12's head sum. -/
theorem multiHeadScalarDrift_le_reference_neg (c v : Idx H → ℝ) (hc : ∀ h, 0 < c h)
    (hv : ∀ h, 0 ≤ v h) (y : Idx (m + 1) → ℝ) (r : ℝ) (hr : r < 0) (M : Idx (m + 1)) :
    multiHeadScalarDrift c v y r ≤ (∑ h, v h) * y M - ((m + 1) * ∑ h, v h / c h) / r := by
  have hg := multiHeadScalarDrift_ge_reference c v hc hv (fun j => -y j) (-r) (by linarith) M
  rw [multiHeadScalarDrift_neg, mul_neg, div_neg] at hg
  linarith

/-- The whole original drift has a controlled error relative to the largest token.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, for §12's original head sum. -/
theorem multiHeadScalarDrift_max_error (c v : Idx H → ℝ) (hc : ∀ h, 0 < c h)
    (hv : ∀ h, 0 ≤ v h) (y : Idx (m + 1) → ℝ) (r : ℝ) (hr : 0 < r)
    (M : Idx (m + 1)) (hM : ∀ j, y j ≤ y M) :
    0 ≤ (∑ h, v h) * y M - multiHeadScalarDrift c v y r ∧
      (∑ h, v h) * y M - multiHeadScalarDrift c v y r ≤ ((m + 1) * ∑ h, v h / c h) / r := by
  have hle : multiHeadScalarDrift c v y r ≤ (∑ h, v h) * y M := by
    rw [multiHeadScalarDrift, Finset.sum_mul]
    exact Finset.sum_le_sum fun h _ =>
      mul_le_mul_of_nonneg_left (scalarHeadAverage_le_max (c h) y r M hM) (hv h)
  have hge := multiHeadScalarDrift_ge_reference c v hc hv y r hr M
  constructor <;> linarith

/-- The reflected whole-head error is controlled relative to the smallest token.
Source: arXiv:2305.05465v6, §7, the negative counterpart of `l:exactasymptotic`, for §12. -/
theorem multiHeadScalarDrift_min_error (c v : Idx H → ℝ) (hc : ∀ h, 0 < c h)
    (hv : ∀ h, 0 ≤ v h) (y : Idx (m + 1) → ℝ) (r : ℝ) (hr : r < 0)
    (L : Idx (m + 1)) (hL : ∀ j, y L ≤ y j) :
    0 ≤ multiHeadScalarDrift c v y r - (∑ h, v h) * y L ∧
      multiHeadScalarDrift c v y r - (∑ h, v h) * y L ≤ ((m + 1) * ∑ h, v h / c h) / (-r) := by
  have hm := multiHeadScalarDrift_max_error c v hc hv (fun j => -y j) (-r) (by linarith)
    L (fun j => by linarith [hL j])
  rw [multiHeadScalarDrift_neg, mul_neg] at hm
  constructor <;> linarith [hm.1, hm.2]

/-- Distinct original tokens and two different positive heads jointly realize
the positive-query, maximal-reference, and all original coefficient premises. -/
example : ∃ (c v : Idx 2 → ℝ) (y : Idx 2 → ℝ) (r : ℝ) (M : Idx 2),
    (∀ h, 0 < c h) ∧ (∀ h, 0 ≤ v h) ∧ 0 < r ∧ (∀ j, y j ≤ y M) ∧ y 0 < y 1 := by
  refine ⟨![1, 2], ![1, 3], ![-1, 1], 1, 1, ?_, ?_, one_pos, ?_, by norm_num⟩
  · intro h; fin_cases h <;> norm_num
  · intro h; fin_cases h <;> norm_num
  · intro j; fin_cases j <;> norm_num

/-- The same nonzero tokens and positive heads also realize the negative-query premise. -/
example : ∃ (c v : Idx 2 → ℝ) (y : Idx 2 → ℝ) (r : ℝ) (M : Idx 2),
    (∀ h, 0 < c h) ∧ (∀ h, 0 ≤ v h) ∧ r < 0 ∧ y M < 0 ∧ (∀ j, y M ≤ y j) := by
  refine ⟨![1, 2], ![1, 3], ![-1, 1], -1, 0, ?_, ?_, by norm_num, by norm_num, ?_⟩
  · intro h; fin_cases h <;> norm_num
  · intro h; fin_cases h <;> norm_num
  · intro j; fin_cases j <;> norm_num

end Transformer.Clusters
