/-
# A nonempty stopping set violating the cap-exit time bound

ArXiv:2410.06833v1, §5, `claim: de sortie de cap`, in Step 2 of
`thm: metastability MF`. The source asserts a bound on the infimum of
`η(t) V(t) exp(-(1-η(t)) β) ≤ 2 exp(-c β)` within the escape window.

Use the stationary Dirac solution and its actual global flow on the
circle, with one cap, `β = 1000000`, `ε = 1/10000`,
`c = 801/1000000` and a cap-boundary minimizing trajectory.
Time `801` belongs to the stopping set; no time in `[0,1]` does.
Consequently `1 ≤ T_* ≤ 801`, whereas the claimed bound is below
`0.0012`. This is a strict counterexample with a nonempty stopping set.

The whole transported cap stays in the escape window at every
nonnegative time. The parameter satisfies `8ε < c < γ`, and the
existing `variance_counter_gamma_lower_bound` verifies `γ = Ω(1)`.
The one-cap convention is `alphaDist_one`, with `α = 0`.
The existential theorem in `MeanFieldCapExit` records every original
hypothesis, continuity of the measure curve and its push-forward law.
-/

import Transformer.Metastability.Section5_CapExitNumerics
import Transformer.Metastability.Section5_VarianceCounterexample

open Real MeasureTheory
namespace Transformer.Metastability
open Perspective

/-- The source's stopping set for the explicit Dirac cap and numerical parameters.
This is the original set evaluated at concrete data, without changing its threshold.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
noncomputable def capExitCounterSet : Set ℝ :=
  capExitSet 2 1000000 (801 / 1000000) (1 / 10000) 1
    (fun _ => varianceCounterCentre) (diracProb 2 varianceCounterCentre)
    (diracFlow 2 varianceCounterCentre) 0
    (fun t => diracFlow 2 varianceCounterCentre t varianceCounterBoundary)

/-- Every nonnegative time lies in the original geometric escape window.
There is no finite escape boundary cutting off the stopping set.

Source: arXiv:2410.06833v1, §5, Step 1, definition of `T_esc`. -/
theorem capExitCounter_escapeWindow_eq :
    escapeWindow 2 1 (fun _ => varianceCounterCentre)
      (diracFlow 2 varianceCounterCentre) (1 / 10000) = Set.Ici (0 : ℝ) := by
  ext t
  constructor
  · exact fun ht => ht.1
  · intro ht
    exact mem_escapeWindow_diracFlow 2 1 _ _ t (by norm_num) ht

/-- An explicit finite time satisfies the stopping inequality within the escape window.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounter_late_membership : (801 : ℝ) ∈ capExitCounterSet := by
  refine ⟨mem_escapeWindow_diracFlow 2 1 _ _ 801 (by norm_num) (by norm_num), ?_⟩
  rw [capMin_diracFlow 2 _ _ _ 801 varianceCounterBoundary_coordinate,
    capVariance_diracFlow 2 _ _ _ 801 (by norm_num) varianceCounterBoundary_coordinate]
  have hf := diracAlong_forward_bounds 801 (1 - 1 / 10000) (by norm_num) (by norm_num)
  have hV : 0 ≤ 1 - diracAlong 801 (1 - 1 / 10000) := by linarith [hf.2]
  have hdecay := one_sub_diracAlong_upper_exp 801 (1 - 1 / 10000) (by norm_num)
  have hexp : Real.exp (-(1 - diracAlong 801 (1 - 1 / 10000)) * 1000000) ≤ 1 :=
    Real.exp_le_one_iff.mpr (by linarith)
  have hηV : diracAlong 801 (1 - 1 / 10000) *
      (1 - diracAlong 801 (1 - 1 / 10000)) ≤ 1 - diracAlong 801 (1 - 1 / 10000) := by
    nlinarith [mul_nonneg hV hV]
  have hηV0 : 0 ≤ diracAlong 801 (1 - 1 / 10000) *
      (1 - diracAlong 801 (1 - 1 / 10000)) := mul_nonneg (by linarith [hf.1]) hV
  have hmono : Real.exp (-2 * (801 : ℝ)) ≤ Real.exp (-(801 / 1000000 : ℝ) * 1000000) :=
    Real.exp_le_exp.mpr (by norm_num)
  calc
    _ ≤ diracAlong 801 (1 - 1 / 10000) *
        (1 - diracAlong 801 (1 - 1 / 10000)) * 1 :=
      mul_le_mul_of_nonneg_left hexp hηV0
    _ ≤ 1 - diracAlong 801 (1 - 1 / 10000) := by simpa using hηV
    _ ≤ 2 * (1 - (1 - 1 / 10000)) * Real.exp (-2 * 801) := hdecay
    _ ≤ 2 * Real.exp (-2 * 801) :=
      mul_le_mul_of_nonneg_right (by norm_num) (Real.exp_pos _).le
    _ ≤ 2 * Real.exp (-(801 / 1000000 : ℝ) * 1000000) :=
      mul_le_mul_of_nonneg_left hmono (by norm_num)

/-- The entire first time interval is excluded from the source's stopping set.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounter_not_mem_before_one (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    t ∉ capExitCounterSet := by
  intro h
  have hstop := h.2
  rw [capMin_diracFlow 2 _ _ _ t varianceCounterBoundary_coordinate,
    capVariance_diracFlow 2 _ _ _ t (by norm_num) varianceCounterBoundary_coordinate] at hstop
  exact not_le_of_gt (capExitCounter_threshold_before_one t ht) hstop

/-- Nonemptiness is proved from the explicit flow, independently of the refuted claim.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounterSet_nonempty : capExitCounterSet.Nonempty :=
  ⟨801, capExitCounter_late_membership⟩

/-- Every member occurs after time one, so the genuine infimum is at least one.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounter_time_lower_bound : (1 : ℝ) ≤ sInf capExitCounterSet := by
  apply le_csInf capExitCounterSet_nonempty
  intro t ht
  by_contra h
  exact capExitCounter_not_mem_before_one t ⟨ht.1.1, le_of_not_ge h⟩ ht

/-- The stopping infimum also has an explicit finite upper bound from the genuine member.
The set is bounded below by time zero, as required for the real infimum.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounter_time_upper_bound : sInf capExitCounterSet ≤ (801 : ℝ) := by
  have hb : BddBelow capExitCounterSet := ⟨0, fun _ ht => ht.1.1⟩
  exact csInf_le hb capExitCounter_late_membership

/-- The claimed strict upper bound on `T_*` fails with a strict gap.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounter_strict_failure :
    4 * (1 / 10000 : ℝ) / 1 *
      Real.exp (((801 / 1000000) - 8 * (1 / 10000)) * 1000000) < sInf capExitCounterSet :=
  capExitCounter_claimed_bound_lt_one.trans_le capExitCounter_time_lower_bound

/-- The parameter lies inside the narrower interval used subsequently in the source.
Thus adding `8ε < c < γ` would not repair the claimed time bound.

Source: arXiv:2410.06833v1, §5, Step 2, following `claim: de sortie de cap`. -/
theorem capExitCounter_parameter_mem (d : ℕ) (w : SSphere d) :
    8 * (1 / 10000 : ℝ) < 801 / 1000000 ∧
      801 / 1000000 < γβ 1 1000000 (αDist d 1 (fun _ => w) (1 / 10000)) (1 / 10000) := by
  have hγ := variance_counter_gamma_lower_bound d w 1000000 (by norm_num)
  constructor <;> linarith

/-- The exclusion lemma's interval hypothesis holds at an interior positive time. -/
example : (1 / 2 : ℝ) ∈ Set.Icc (0 : ℝ) 1 ∧ (1 / 2 : ℝ) ∉ capExitCounterSet :=
  ⟨by norm_num, capExitCounter_not_mem_before_one (1 / 2) (by norm_num)⟩

/-- The counterexample has a genuine nonempty stopping set and a finite positive infimum. -/
example : capExitCounterSet.Nonempty ∧
    (1 : ℝ) ≤ sInf capExitCounterSet ∧ sInf capExitCounterSet ≤ (801 : ℝ) :=
  ⟨capExitCounterSet_nonempty, capExitCounter_time_lower_bound, capExitCounter_time_upper_bound⟩

/-- Both strict parameter inequalities hold for the explicit circle center. -/
example : 8 * (1 / 10000 : ℝ) < 801 / 1000000 ∧
    801 / 1000000 < γβ 1 1000000
      (αDist 2 1 (fun _ => varianceCounterCentre) (1 / 10000)) (1 / 10000) :=
  capExitCounter_parameter_mem 2 varianceCounterCentre

end Transformer.Metastability
