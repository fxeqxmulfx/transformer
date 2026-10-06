import Transformer.GPTMini.Sparsemax.AtomicMatchingOptimality

/-!
# Selecting new genuine matching heads changes the learned response

New witnesses for the atomic architecture following arXiv:2211.11052v1,
Appendix A.4, with original sparsemax Eq. (1) of arXiv:1602.02068v2.
The two witness heads have identical independent values. Their different
Q/K alone change which input occurrence receives attention. Mixing them
gives a strict ordinary-output improvement without a route target.

The convex path adds mass to a different head. It is not interpolation
of the raw query/key coordinates of one fixed head: the midpoint of
those coordinates has a different physical prediction. An arbitrary
finite number of copies of the uniform head cannot achieve the fitting
prediction of the nonuniform head. Thus optimizing the weights of an
unchanging atom bank cannot certify the full matching problem.

These are physical forward witnesses; no inverse decoder enters the path.
The oracle certificate itself remains conditional on a price bound for all heads.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex
open scoped BigOperators

/-- A learned mixture path changes genuine Q/K matching while all original value tables agree.
Source: the new two-head physical witness after Appendix A.4. -/
def matchingScalarMixture (a : ℝ) : MatchingMixture 2 1 1 :=
  (1 - a) • Finsupp.single (matchingScalarHead 0) 1 + a • Finsupp.single (matchingScalarHead 1) 1

/-- Both physical atoms use the identical independent original value table. -/
example : (matchingScalarHead 0).2 = (matchingScalarHead 1).2 := rfl

/-- The complete genuine atom domain contains the support-changing matching path.
Source: the unit-mass convex domain following Appendix A.4. -/
theorem matchingScalarMixture_mem (a : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) :
    matchingScalarMixture a ∈ matchingMixtureDomain 2 1 1 1 := by
  apply matchingMixtureDomain_convex 2 1 1 1
    (matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num)))
    (matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num)))
  · linarith [ha.2]
  · exact ha.1
  · ring

/-- An interior support-changing matching state inhabits the complete mixture domain. -/
example : matchingScalarMixture (1 / 2) ∈ matchingMixtureDomain 2 1 1 1 :=
  matchingScalarMixture_mem _ (by norm_num)

/-- The new Q/K head changes physical predictions along the convex mixture path.
Source: genuine two-token sparsemax §2.2 inside the new Appendix A.4 architecture. -/
theorem matchingScalarMixture_output (a : ℝ) :
    matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
      (matchingScalarMixture a) 0 = (1 + a) / 2 := by
  rw [matchingScalarMixture, map_add, map_smul, map_smul,
    matchingMixture_single_output, matchingMixture_single_output]
  change (1 - a) * matchingHeadOutput (matchingScalarHead 0) id 1 0 +
    a * matchingHeadOutput (matchingScalarHead 1) id 1 0 = _
  rw [matchingScalarHead_zero_output, matchingScalarHead_one_output]
  ring

/-- Ordinary output error reacts to the learned matching, without any supplied attention label.
Source: the derived actual forward on the new Appendix A.4 mixture path. -/
theorem matchingScalarMixture_error (a : ℝ) :
    (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
      (matchingScalarMixture a) 0 - 1) ^ 2 = (1 - a) ^ 2 / 4 := by
  rw [matchingScalarMixture_output]
  ring

/-- Every positive feasible addition of the new matching head strictly improves the uniform answer.
Source: the new genuine-head descent witness after Appendix A.4, not a raw-matrix gradient claim. -/
theorem matchingScalarMixture_improves (a : ℝ) (ha : 0 < a ∧ a ≤ 1) :
    matchingScalarMixture a ∈ matchingMixtureDomain 2 1 1 1 ∧
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (matchingScalarMixture a) 0 - 1) ^ 2 <
        (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
          (matchingScalarMixture 0) 0 - 1) ^ 2 := by
  refine ⟨matchingScalarMixture_mem a ⟨ha.1.le, ha.2⟩, ?_⟩
  rw [matchingScalarMixture_error, matchingScalarMixture_error]
  have hm : 0 < a * (2 - a) := mul_pos ha.1 (by linarith [ha.2])
  nlinarith

/-- Full addition of a genuinely learned nonuniform head attains the answer from the same values. -/
example : matchingScalarMixture 1 ∈ matchingMixtureDomain 2 1 1 1 ∧
    (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
      (matchingScalarMixture 1) 0 - 1) ^ 2 <
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (matchingScalarMixture 0) 0 - 1) ^ 2 :=
  matchingScalarMixture_improves _ (by norm_num)

/-- The physical query/key midpoint is computed by the original sparsemax projection.
Source: sparsemax Eq. (1) and §2.2, threshold -3/8 for scores zero and one quarter. -/
theorem matchingScalarHead_half_output :
    matchingHeadOutput (matchingScalarHead (1 / 2)) id 1 0 = 5 / 8 := by
  have hw : sparseWeights (matchingHeadScores (matchingScalarHead (1 / 2)) id 1) 1 =
      thresholdWeights (matchingHeadScores (matchingScalarHead (1 / 2)) id 1) 1 (-3 / 8) := by
    symm
    apply thresholdWeights_eq_sparseWeights
    norm_num [thresholdWeights, matchingHeadScores, matchingScalarHead, Fin.sum_univ_two]
  unfold matchingHeadOutput
  rw [hw]
  norm_num [thresholdWeights, matchingHeadScores, matchingScalarHead, Fin.sum_univ_two]

/-- Raw one-head Q/K interpolation differs from the proved convex matching-mixture interpolation.
Source: the new architecture's boundary, consistent with the nonconvex matching in §3.1. -/
theorem matchingScalar_midpoints_differ :
    matchingHeadOutput (matchingScalarHead (1 / 2)) id 1 0 ≠
      matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (matchingScalarMixture (1 / 2)) 0 := by
  rw [matchingScalarHead_half_output, matchingScalarMixture_output]
  norm_num

/-- Genuine atom pricing is already nonconvex along an affine raw-Q/K path with values fixed.
Source: new computational boundary of Appendix A.4's atomic idea, using sparsemax Eq. (1). -/
theorem matchingScalar_price_not_convex :
    ¬ ConvexOn ℝ (Set.Icc 0 1) (fun t : ℝ =>
      matchingHeadPrice (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (fun _ => -1) (matchingScalarHead t)) := by
  intro hc
  have hj := hc.2
  have he := hj (x := 0) (y := 1) (by norm_num) (by norm_num)
    (a := (1 / 2 : ℝ)) (b := (1 / 2 : ℝ)) (by norm_num) (by norm_num) (by norm_num)
  norm_num [smul_eq_mul] at he
  norm_num [matchingHeadPrice, matchingOutputPairing, matchingHeadSample,
    matchingScalarHead_zero_output, matchingScalarHead_one_output,
    matchingScalarHead_half_output] at he

/-- Repeating an existing uniform head, with any normalized weights, leaves its answer unchanged.
Source: exact finite reconstruction after Appendix A.4, illustrating the fixed-bank restriction. -/
theorem matchingUniformFamily_output {ι : Type*} [Fintype ι] (w : ι → ℝ) (hs : ∑ i, w i = 1) :
    matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
      (matchingMixtureOfFamily (fun _ : ι => matchingScalarHead 0) w) 0 = 1 / 2 := by
  rw [matchingMixtureOfFamily_sample, ← Finset.sum_smul, hs, one_smul]
  exact matchingScalarHead_zero_output

/-- Distinct positive weights on an actual repeated head inhabit every fixed-bank premise. -/
example : matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
    (matchingMixtureOfFamily (fun _ : Fin 2 => matchingScalarHead 0)
      (fun i => if i = 0 then (1 / 3 : ℝ) else 2 / 3)) 0 = 1 / 2 :=
  matchingUniformFamily_output _ (by norm_num [Fin.sum_univ_two])

/-- A new freely chosen Q/K head outperforms every normalized family from the old uniform bank.
Source: the new genuine-head witness after Appendix A.4; selecting atoms is essential. -/
theorem matchingUniformFamily_suboptimal {ι : Type*} [Fintype ι] (w : ι → ℝ)
    (hs : ∑ i, w i = 1) :
    (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
      (Finsupp.single (matchingScalarHead 1) 1) 0 - 1) ^ 2 <
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (matchingMixtureOfFamily (fun _ : ι => matchingScalarHead 0) w) 0 - 1) ^ 2 := by
  rw [matchingUniformFamily_output w hs, matchingMixture_single_output]
  change (matchingHeadOutput (matchingScalarHead 1) id 1 0 - 1) ^ 2 < _
  rw [matchingScalarHead_one_output]
  norm_num

/-- Even a normalized two-copy bank misses a fitting head available in the full matching domain. -/
example : (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
    (Finsupp.single (matchingScalarHead 1) 1) 0 - 1) ^ 2 <
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (matchingMixtureOfFamily (fun _ : Fin 2 => matchingScalarHead 0) (fun _ => (1 / 2 : ℝ))) 0 - 1) ^ 2 :=
  matchingUniformFamily_suboptimal _ (by norm_num [Fin.sum_univ_two])

end Transformer.GPTMini.Sparsemax
