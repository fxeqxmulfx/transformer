module

public import Transformer.Modes.Section2_ZeroCharts
public import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# A genuine regular zero-set point with a degenerate KDE query

ArXiv:2412.09080v3, §2.1, the assertion about modes following
`eq:Fn`. Almost-sure query nondegeneracy cannot be replaced by a
pointwise claim even when every original sample derivative is
regular. We construct five distinct samples, one at zero and two
symmetric pairs. Varying the inner pair gives a zero of the
query derivative while the central sample derivative is nonzero.

This also supplies nonempty critical-chart premises for the
projection argument. The point uses the actual Gaussian score,
not a prescribed derivative or an auxiliary field.
-/

@[expose] public section

open Real Set

namespace Transformer.Modes

/-- Five original one-dimensional samples with independently positioned symmetric pairs.
Source: arXiv:2412.09080v3, §2.1, a concrete configuration for the original `eq:Fn`. -/
def symmetricFiveSample (a b : ℝ) : Idx 5 → ℝ := ![0, -a, a, -b, b]

/-- Symmetry gives an actual zero of the original five-sample field at the central sample.
Source: arXiv:2412.09080v3, §2.1, the original field `eq:Fn`. -/
theorem fieldF_symmetricFive_zero (β a b : ℝ) : fieldF β (symmetricFiveSample a b) 0 = 0 := by
  simp [fieldF, symmetricFiveSample, Fin.sum_univ_succ]

/-- The original query derivative at this symmetric zero retains both pair contributions.
Source: arXiv:2412.09080v3, §2.1, differentiation of `eq:Fn`, at original bandwidth one. -/
theorem fieldF_symmetricFive_deriv (a b : ℝ) :
    deriv (fieldF 1 (symmetricFiveSample a b)) 0 =
      (1 / Real.sqrt 5) * (1 + 2 * ((1 - a ^ 2) * Real.exp (-(a ^ 2) / 2)) +
        2 * ((1 - b ^ 2) * Real.exp (-(b ^ 2) / 2))) := by
  rw [(hasDerivAt_fieldF_query 1 _ 0).deriv]
  simp [symmetricFiveSample, Fin.sum_univ_succ]
  ring_nf

/-- The actual exterior score at displacement sqrt(3) has exponential scale strictly between four and eight.
Source: arXiv:2412.09080v3, §2.1, the original exponential in `eq:Fn`.
The elementary estimates use Mathlib's proved bounds on exp(1). -/
theorem exp_three_halves_bounds : (4 : ℝ) < Real.exp (3 / 2) ∧ Real.exp (3 / 2) < 8 := by
  have hl := Real.exp_one_gt_d9
  have hu := Real.exp_one_lt_three
  have he := Real.exp_pos (3 / 2 : ℝ)
  have hsq : Real.exp (3 / 2 : ℝ) ^ 2 = Real.exp 1 ^ 3 := by
    rw [sq, ← Real.exp_add, ← Real.exp_nat_mul]
    norm_num
  constructor <;> nlinarith [sq_nonneg (Real.exp 1 - 2), sq_nonneg (Real.exp 1 - 3)]

/-- Varying the actual inner pair produces a degenerate query strictly between displacements one and sqrt(3).
Source: arXiv:2412.09080v3, §2.1, the derivative of the original `eq:Fn`.
The sign change is in the original Gaussian score, and the intermediate value theorem supplies the pair. -/
theorem exists_symmetricFive_query_degenerate : ∃ a : ℝ, 1 < a ∧ a < Real.sqrt 3 ∧
    deriv (fieldF 1 (symmetricFiveSample a (Real.sqrt 3))) 0 = 0 := by
  let g : ℝ → ℝ := fun a => 1 + 2 * ((1 - a ^ 2) * Real.exp (-(a ^ 2) / 2)) +
    2 * ((1 - (Real.sqrt 3) ^ 2) * Real.exp (-((Real.sqrt 3) ^ 2) / 2))
  have hrt : (1 : ℝ) < Real.sqrt 3 := by
    have hp := Real.sqrt_nonneg (3 : ℝ)
    have hs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)
    nlinarith
  have hr2 := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)
  have hpos := Real.exp_pos (3 / 2 : ℝ)
  have hb := exp_three_halves_bounds
  have hneg : Real.exp (-(3 / 2 : ℝ)) = (Real.exp (3 / 2 : ℝ))⁻¹ := Real.exp_neg _
  have hg1 : 0 < g 1 := by
    dsimp [g]
    norm_num only [one_pow, sub_self, zero_mul, mul_zero, add_zero, hr2]
    rw [hneg]
    have hm := (inv_pos.mpr hpos)
    have hi := mul_inv_cancel₀ hpos.ne'
    nlinarith [hb.1]
  have hgr : g (Real.sqrt 3) < 0 := by
    dsimp [g]
    rw [hr2]
    norm_num only [show -(3 : ℝ) / 2 = -(3 / 2 : ℝ) by ring, hneg]
    have hm := (inv_pos.mpr hpos)
    have hi := mul_inv_cancel₀ hpos.ne'
    nlinarith [hb.2]
  have hcont : Continuous g := by dsimp [g]; fun_prop
  obtain ⟨a, ha, hz⟩ := intermediate_value_Icc' hrt.le hcont.continuousOn
    (show (0 : ℝ) ∈ Icc (g (Real.sqrt 3)) (g 1) from ⟨hgr.le, hg1.le⟩)
  have ha1 : 1 < a := by
    by_contra h
    have he : a = 1 := by linarith [ha.1]
    subst a
    linarith
  have har : a < Real.sqrt 3 := by
    by_contra h
    have he : a = Real.sqrt 3 := by linarith [ha.2]
    subst a
    linarith
  refine ⟨a, ha1, har, ?_⟩
  rw [fieldF_symmetricFive_deriv]
  change (1 / Real.sqrt 5) * g a = 0
  rw [hz, mul_zero]

/-- These five actual samples are distinct and avoid all exceptional score-pair distances.
Source: arXiv:2412.09080v3, §2.1, the original `eq:Fn` sample geometry. -/
theorem symmetricFiveSample_pair_gaps (a : ℝ) (ha : 1 < a) (har : a < Real.sqrt 3) :
    ∀ i j : Idx 5, i ≠ j → symmetricFiveSample a (Real.sqrt 3) i ≠
      symmetricFiveSample a (Real.sqrt 3) j ∧
      (1 : ℝ) * (symmetricFiveSample a (Real.sqrt 3) i -
        symmetricFiveSample a (Real.sqrt 3) j) ^ 2 ≠ 4 := by
  have hr0 := Real.sqrt_nonneg (3 : ℝ)
  have hrs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)
  have hr1 : (1 : ℝ) < Real.sqrt 3 := by nlinarith
  have hr2 : Real.sqrt 3 < (2 : ℝ) := by nlinarith
  have hab : a < 2 := har.trans hr2
  have hp : ∀ i j : Idx 5, i ≠ j →
      symmetricFiveSample a (Real.sqrt 3) i ≠ symmetricFiveSample a (Real.sqrt 3) j ∧
      symmetricFiveSample a (Real.sqrt 3) i - symmetricFiveSample a (Real.sqrt 3) j ≠ 2 ∧
      symmetricFiveSample a (Real.sqrt 3) i - symmetricFiveSample a (Real.sqrt 3) j ≠ -2 := by
    intro i j hij
    fin_cases i <;> fin_cases j <;> try exact (hij rfl).elim
    all_goals norm_num [symmetricFiveSample]
    all_goals constructor
    all_goals first
      | (constructor <;> (intro he; linarith))
      | (intro he; linarith)
  intro i j hij
  refine ⟨(hp i j hij).1, ?_⟩
  intro he
  have hs : (symmetricFiveSample a (Real.sqrt 3) i -
      symmetricFiveSample a (Real.sqrt 3) j) ^ 2 = (2 : ℝ) ^ 2 := by norm_num at he ⊢; exact he
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp hs with h | h
  · exact (hp i j hij).2.1 h
  · exact (hp i j hij).2.2 h

/-- An actual regular zero-set point has zero original query derivative.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`.
Thus the critical chart set is nonempty even for distinct data avoiding every pair exception. -/
theorem exists_regular_degenerate_fieldF : ∃ X : Idx 5 → ℝ,
    (∀ i j, i ≠ j → X i ≠ X j ∧ (1 : ℝ) * (X i - X j) ^ 2 ≠ 4) ∧
    fieldF 1 X 0 = 0 ∧ deriv (fieldF 1 X) 0 = 0 ∧
    deriv (kdeScore 1) (0 - X 0) ≠ 0 := by
  obtain ⟨a, ha, har, hd⟩ := exists_symmetricFive_query_degenerate
  refine ⟨symmetricFiveSample a (Real.sqrt 3), symmetricFiveSample_pair_gaps a ha har,
    fieldF_symmetricFive_zero 1 a _, hd, ?_⟩
  norm_num [symmetricFiveSample, (hasDerivAt_kdeScore 1 0).deriv]

/-- The variable-pair theorem's strict interval hypotheses have an explicit simultaneous instance. -/
example : (1 : ℝ) < 3 / 2 ∧ (3 / 2 : ℝ) < Real.sqrt 3 := by
  have hs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)
  have hp := Real.sqrt_nonneg (3 : ℝ)
  constructor <;> nlinarith

end Transformer.Modes
