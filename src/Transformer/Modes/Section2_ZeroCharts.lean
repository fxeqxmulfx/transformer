module

public import Transformer.Modes.Section2_ScoreInverses
public import Mathlib.Analysis.Calculus.FDeriv.Pi

/-!
# Actual sample-coordinate charts of the Gaussian KDE zero set

ArXiv:2412.09080v3, §2.1, the original field `eq:Fn`.
One coordinate of the chart parameter is the original query;
all others retain the original sample coordinates. Solving
the original score sum for the omitted sample uses one of
the three genuine inverse branches. The chart and its
derivative below are explicit in those original quantities.

The inverse-image condition is retained in every derivative
statement. The chart identities use the original zero equation,
not an assumed implicit function. Query-critical points will
give a nontrivial kernel of the chart derivative in the next step.
-/

@[expose] public section

open Real Set

namespace Transformer.Modes

/-- The original score sum with one sample omitted, with sign chosen to solve `eq:Fn`.
Source: arXiv:2412.09080v3, §2.1. The chart coordinate `u i` is the query. -/
noncomputable def kdeScoreRemainder {n : ℕ} (β : ℝ) (i : Idx n) (u : Idx n → ℝ) : ℝ :=
  -∑ j ∈ Finset.univ.erase i, kdeScore β (u i - u j)

/-- The actual derivative of the omitted original score sum.
Source: arXiv:2412.09080v3, §2.1, differentiating the summands of `eq:Fn`. -/
noncomputable def kdeScoreRemainderDeriv {n : ℕ} (β : ℝ) (i : Idx n) (u : Idx n → ℝ) :
    (Idx n → ℝ) →L[ℝ] ℝ :=
  -∑ j ∈ Finset.univ.erase i, deriv (kdeScore β) (u i - u j) •
    ((ContinuousLinearMap.proj i : (Idx n → ℝ) →L[ℝ] ℝ) - ContinuousLinearMap.proj j)

/-- Solving one original sample coordinate through an actual score inverse branch.
Source: arXiv:2412.09080v3, §2.1, the zero equation `eq:Fn`.
The query occupies parameter coordinate `i`; the other parameters are the original samples. -/
noncomputable def kdeZeroChart {n : ℕ} (β : ℝ) (i : Idx n) (b : Fin 3)
    (u : Idx n → ℝ) : Idx n → ℝ :=
  Function.update u i (u i - kdeScoreInverse β b (kdeScoreRemainder β i u))

/-- The explicit derivative of the original sample-coordinate chart.
Source: arXiv:2412.09080v3, §2.1, reciprocal differentiation of the original `eq:Fn` score. -/
noncomputable def kdeZeroChartDeriv {n : ℕ} (β : ℝ) (i : Idx n) (b : Fin 3)
    (u : Idx n → ℝ) : (Idx n → ℝ) →L[ℝ] (Idx n → ℝ) :=
  ContinuousLinearMap.pi fun j => if j = i then
    ContinuousLinearMap.proj i -
      (deriv (kdeScore β) (kdeScoreInverse β b (kdeScoreRemainder β i u)))⁻¹ •
        kdeScoreRemainderDeriv β i u
    else ContinuousLinearMap.proj j

/-- Differentiating the genuine omitted score sum in all original parameter directions.
Source: arXiv:2412.09080v3, §2.1, the original summands in `eq:Fn`. -/
theorem hasFDerivAt_kdeScoreRemainder {n : ℕ} (β : ℝ) (i : Idx n) (u : Idx n → ℝ) :
    HasFDerivAt (kdeScoreRemainder β i) (kdeScoreRemainderDeriv β i u) u := by
  unfold kdeScoreRemainder kdeScoreRemainderDeriv
  apply HasFDerivAt.neg
  have h : ∀ j ∈ Finset.univ.erase i,
      HasFDerivAt (fun v : Idx n → ℝ => kdeScore β (v i - v j))
        (deriv (kdeScore β) (u i - u j) •
          ((ContinuousLinearMap.proj i : (Idx n → ℝ) →L[ℝ] ℝ) - ContinuousLinearMap.proj j)) u := by
    intro j hj
    convert! (((contDiff_kdeScore (k := 1) β).differentiable (by norm_num) _).hasDerivAt).comp_hasFDerivAt u
      ((hasFDerivAt_apply (𝕜 := ℝ) i u).sub (hasFDerivAt_apply j u)) using 1
  convert! HasFDerivAt.sum h using 1
  ext v
  simp only [Finset.sum_apply]

/-- The original chart is differentiable wherever its inverse input belongs to the actual branch image.
Source: arXiv:2412.09080v3, §2.1, the regular sample-coordinate solution of `eq:Fn`. -/
theorem hasFDerivAt_kdeZeroChart {β : ℝ} (hβ : 0 < β) {n : ℕ} (i : Idx n) (b : Fin 3)
    (u : Idx n → ℝ) (hu : kdeScoreRemainder β i u ∈ kdeScore β '' kdeScoreBranch β b) :
    HasFDerivAt (kdeZeroChart β i b) (kdeZeroChartDeriv β i b u) u := by
  change HasFDerivAt (fun v j => Function.update v i
    (v i - kdeScoreInverse β b (kdeScoreRemainder β i v)) j) _ u
  apply hasFDerivAt_pi.mpr
  intro j
  by_cases hji : j = i
  · subst j
    convert! (hasFDerivAt_apply (𝕜 := ℝ) i u).sub
        ((hasDerivAt_kdeScoreInverse hβ b _ hu).comp_hasFDerivAt u
          (hasFDerivAt_kdeScoreRemainder β i u)) using 1
    · ext v
      simp only [Function.update_self, Pi.sub_apply, Function.comp_apply]
    · simp only [ite_eq_left]
  · simpa only [Function.update_of_ne hji, kdeZeroChartDeriv, ite_eq_right hji] using
      hasFDerivAt_apply (𝕜 := ℝ) j u

/-- Substituting the original query into one parameter coordinate leaves precisely the omitted score sum.
Source: arXiv:2412.09080v3, §2.1, the original field equation `eq:Fn`. -/
theorem kdeScoreRemainder_update {n : ℕ} (β : ℝ) (i : Idx n) (X : Idx n → ℝ) (t : ℝ) :
    kdeScoreRemainder β i (Function.update X i t) =
      -∑ j ∈ Finset.univ.erase i, kdeScore β (t - X j) := by
  unfold kdeScoreRemainder
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  rw [Function.update_self, Function.update_of_ne (Finset.mem_erase.mp hj).1]

/-- The original zero equation identifies the inverse input with the score of the omitted displacement.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`, including its original nonzero normalization. -/
theorem kdeScoreRemainder_eq_score_of_fieldF_zero {n : ℕ} (hn : 0 < n)
    (β : ℝ) (X : Idx n → ℝ) (t : ℝ) (ht : fieldF β X t = 0) (i : Idx n) :
    kdeScoreRemainder β i (Function.update X i t) = kdeScore β (t - X i) := by
  have hnorm : (1 / Real.sqrt (n : ℝ)) ≠ 0 :=
    one_div_ne_zero (Real.sqrt_ne_zero'.mpr (by exact_mod_cast hn))
  have hsum : ∑ j, kdeScore β (t - X j) = 0 :=
    (mul_eq_zero.mp ht).resolve_left hnorm
  rw [kdeScoreRemainder_update]
  have h := Finset.add_sum_erase Finset.univ (fun j => kdeScore β (t - X j)) (Finset.mem_univ i)
  linarith

/-- Every original zero with its omitted displacement in a branch is represented by the actual chart.
Source: arXiv:2412.09080v3, §2.1, solving the original zero equation `eq:Fn`. -/
theorem kdeZeroChart_update_of_fieldF_zero {β : ℝ} (hβ : 0 < β) {n : ℕ} (hn : 0 < n)
    (X : Idx n → ℝ) (t : ℝ) (ht : fieldF β X t = 0) (i : Idx n) (b : Fin 3)
    (hb : t - X i ∈ kdeScoreBranch β b) :
    kdeZeroChart β i b (Function.update X i t) = X := by
  unfold kdeZeroChart
  rw [kdeScoreRemainder_eq_score_of_fieldF_zero hn β X t ht i,
    kdeScoreInverse_kdeScore hβ b _ hb, Function.update_self]
  ext j
  by_cases hji : j = i
  · subst j
    simp
  · simp only [Function.update_of_ne hji]

/-- Every actual branch-image input gives a zero of the original field at the chart query.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`.
This checks the original score sum after inserting the solved sample coordinate. -/
theorem fieldF_kdeZeroChart {n : ℕ} (β : ℝ) (i : Idx n) (b : Fin 3) (u : Idx n → ℝ)
    (hu : kdeScoreRemainder β i u ∈ kdeScore β '' kdeScoreBranch β b) :
    fieldF β (kdeZeroChart β i b u) (u i) = 0 := by
  have hi : u i - kdeZeroChart β i b u i =
      kdeScoreInverse β b (kdeScoreRemainder β i u) := by
    simp [kdeZeroChart]
  have hothers : ∑ j ∈ Finset.univ.erase i, kdeScore β (u i - kdeZeroChart β i b u j) =
      ∑ j ∈ Finset.univ.erase i, kdeScore β (u i - u j) := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [kdeZeroChart, Function.update_of_ne (Finset.mem_erase.mp hj).1]
  have hsum := Finset.add_sum_erase Finset.univ
    (fun j => kdeScore β (u i - kdeZeroChart β i b u j)) (Finset.mem_univ i)
  rw [hi, kdeScore_kdeScoreInverse β b _ hu, hothers] at hsum
  have hz : ∑ j, kdeScore β (u i - kdeZeroChart β i b u j) = 0 := by
    unfold kdeScoreRemainder at hsum
    linarith
  change (1 / Real.sqrt (n : ℝ)) * (∑ j, kdeScore β (u i - kdeZeroChart β i b u j)) = 0
  rw [hz, mul_zero]

/-- An actual singleton zero realizes the positive bandwidth, original zero equation,
branch displacement, inverse-image condition, and differentiable chart premises together. -/
example : (0 : ℝ) < 1 ∧ (0 : ℕ) < 1 ∧
    fieldF 1 (fun _ : Idx 1 => 0) 0 = 0 ∧
    (0 : ℝ) ∈ kdeScoreBranch 1 1 ∧
    kdeScoreRemainder 1 (0 : Idx 1) (fun _ => 0) ∈ kdeScore 1 '' kdeScoreBranch 1 1 := by
  have hb : (0 : ℝ) ∈ kdeScoreBranch 1 1 := by norm_num [kdeScoreBranch, kdeScoreRadius]
  refine ⟨by norm_num, by norm_num, by simp [fieldF], hb, 0, hb, ?_⟩
  simp [kdeScore, kdeScoreRemainder]

end Transformer.Modes
