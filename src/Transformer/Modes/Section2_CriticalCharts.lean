module

public import Transformer.Modes.Section2_RegularDegenerate
public import Mathlib.LinearAlgebra.Determinant

/-!
# Singular projections at degenerate original KDE queries

ArXiv:2412.09080v3, §2.1, the original `eq:Fn`.
The score-chart derivative sends the query coordinate vector
to zero whenever the original query derivative vanishes.
Its reciprocal factor is legitimate on the actual branch image.
The query coordinate vector is nonzero, so the derivative
has a nontrivial kernel and determinant zero.

All quantities are the derivatives of the original field and
the previously constructed sample-coordinate chart.
The critical domain is the explicit branch-image condition
and the original query-critical equation; it assumes no
measure or nondegeneracy conclusion.
-/

@[expose] public section

open Real Set

namespace Transformer.Modes

/-- The original query derivative is exactly the normalized sum of the original score derivatives.
Source: arXiv:2412.09080v3, §2.1, differentiating `eq:Fn`. -/
theorem fieldF_deriv_eq_sum_kdeScore_deriv {n : ℕ} (β : ℝ) (X : Idx n → ℝ) (t : ℝ) :
    deriv (fieldF β X) t = (1 / Real.sqrt (n : ℝ)) * ∑ j, deriv (kdeScore β) (t - X j) := by
  rw [(hasDerivAt_fieldF_query β X t).deriv]
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  rw [(hasDerivAt_kdeScore β (t - X j)).deriv]

/-- The omitted-score derivative in the query direction is the negative sum of the remaining slopes.
Source: arXiv:2412.09080v3, §2.1, the original derivative of `eq:Fn`. -/
theorem kdeScoreRemainderDeriv_single {n : ℕ} (β : ℝ) (i : Idx n) (u : Idx n → ℝ) :
    kdeScoreRemainderDeriv β i u (Pi.single i 1) =
      -∑ j ∈ Finset.univ.erase i, deriv (kdeScore β) (u i - u j) := by
  rw [kdeScoreRemainderDeriv, neg_apply, sum_apply]
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  simp [sub_apply, (Finset.mem_erase.mp hj).1]

/-- The genuine query-critical equation identifies the omitted-score slope with the solved-coordinate slope.
Source: arXiv:2412.09080v3, §2.1, the actual zero derivative of `eq:Fn`. -/
theorem kdeScoreRemainderDeriv_single_of_critical {n : ℕ} (β : ℝ) (i : Idx n) (b : Fin 3)
    (u : Idx n → ℝ) (hd : deriv (fieldF β (kdeZeroChart β i b u)) (u i) = 0) :
    kdeScoreRemainderDeriv β i u (Pi.single i 1) =
      deriv (kdeScore β) (kdeScoreInverse β b (kdeScoreRemainder β i u)) := by
  have hn : 0 < n := by have hi := i.isLt; omega
  have hnorm : (1 / Real.sqrt (n : ℝ)) ≠ 0 :=
    one_div_ne_zero (Real.sqrt_ne_zero'.mpr (by exact_mod_cast hn))
  rw [fieldF_deriv_eq_sum_kdeScore_deriv] at hd
  have hsum := (mul_eq_zero.mp hd).resolve_left hnorm
  have hi : u i - kdeZeroChart β i b u i =
      kdeScoreInverse β b (kdeScoreRemainder β i u) := by simp [kdeZeroChart]
  have hothers : ∑ j ∈ Finset.univ.erase i, deriv (kdeScore β) (u i - kdeZeroChart β i b u j) =
      ∑ j ∈ Finset.univ.erase i, deriv (kdeScore β) (u i - u j) := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [kdeZeroChart, Function.update_of_ne (Finset.mem_erase.mp hj).1]
  have he := Finset.add_sum_erase Finset.univ
    (fun j => deriv (kdeScore β) (u i - kdeZeroChart β i b u j)) (Finset.mem_univ i)
  rw [hi, hothers, hsum] at he
  rw [kdeScoreRemainderDeriv_single]
  linarith

/-- At an original degenerate query the actual chart derivative has a nonzero query-direction kernel vector.
Source: arXiv:2412.09080v3, §2.1, critical projection of the original `eq:Fn` zero set. -/
theorem kdeZeroChartDeriv_single_eq_zero {β : ℝ} (hβ : 0 < β) {n : ℕ} (i : Idx n) (b : Fin 3)
    (u : Idx n → ℝ) (hu : kdeScoreRemainder β i u ∈ kdeScore β '' kdeScoreBranch β b)
    (hd : deriv (fieldF β (kdeZeroChart β i b u)) (u i) = 0) :
    kdeZeroChartDeriv β i b u (Pi.single i 1) = 0 := by
  have hr := kdeScoreRemainderDeriv_single_of_critical β i b u hd
  have hne := kdeScoreBranch_deriv_ne_zero hβ b _ (kdeScoreInverse_mem β b _ hu)
  ext j
  simp only [kdeZeroChartDeriv, ContinuousLinearMap.pi_apply]
  by_cases hji : j = i
  · subst j
    simp [hr, hne]
  · simp [hji]

/-- The actual original chart projection is singular at each query-critical point in its domain.
Source: arXiv:2412.09080v3, §2.1, the critical-value argument for `eq:Fn`. -/
theorem kdeZeroChartDeriv_det_eq_zero {β : ℝ} (hβ : 0 < β) {n : ℕ} (i : Idx n) (b : Fin 3)
    (u : Idx n → ℝ) (hu : kdeScoreRemainder β i u ∈ kdeScore β '' kdeScoreBranch β b)
    (hd : deriv (fieldF β (kdeZeroChart β i b u)) (u i) = 0) :
    (kdeZeroChartDeriv β i b u).toLinearMap.det = 0 := by
  apply LinearMap.det_eq_zero_iff_ker_ne_bot.mpr
  apply (Submodule.ne_bot_iff _).mpr
  refine ⟨Pi.single i 1, LinearMap.mem_ker.mpr
    (kdeZeroChartDeriv_single_eq_zero hβ i b u hu hd), ?_⟩
  intro he
  have hv := congrFun he i
  norm_num at hv

/-- The actual branch-image and original query-critical conditions for a chart parameter.
Source: arXiv:2412.09080v3, §2.1, critical projection of the zero set in `eq:Fn`. -/
noncomputable def kdeCriticalChartDomain {n : ℕ} (β : ℝ) (i : Idx n) (b : Fin 3) :
    Set (Idx n → ℝ) := {u | kdeScoreRemainder β i u ∈ kdeScore β '' kdeScoreBranch β b ∧
      deriv (fieldF β (kdeZeroChart β i b u)) (u i) = 0}

/-- On the genuine critical domain the original chart has the previously computed derivative.
Source: arXiv:2412.09080v3, §2.1, the critical projection chart for `eq:Fn`. -/
theorem hasFDerivWithinAt_kdeCriticalChart {β : ℝ} (hβ : 0 < β) {n : ℕ} (i : Idx n) (b : Fin 3)
    (u : Idx n → ℝ) (hu : u ∈ kdeCriticalChartDomain β i b) :
    HasFDerivWithinAt (kdeZeroChart β i b) (kdeZeroChartDeriv β i b u)
      (kdeCriticalChartDomain β i b) u :=
  (hasFDerivAt_kdeZeroChart hβ i b u hu.1).hasFDerivWithinAt

/-- Every derivative determinant on the genuine critical domain vanishes.
Source: arXiv:2412.09080v3, §2.1, the singular projection in the original `eq:Fn` sample dimension. -/
theorem kdeCriticalChart_det_zero {β : ℝ} (hβ : 0 < β) {n : ℕ} (i : Idx n) (b : Fin 3)
    (u : Idx n → ℝ) (hu : u ∈ kdeCriticalChartDomain β i b) :
    (kdeZeroChartDeriv β i b u).toLinearMap.det = 0 :=
  kdeZeroChartDeriv_det_eq_zero hβ i b u hu.1 hu.2

/-- A genuine five-sample zero realizes the positive bandwidth, inverse-image membership,
query-critical equation, nonzero sample slope, and nonzero query direction together.
Source: arXiv:2412.09080v3, §2.1, the original `eq:Fn` field. -/
example : ∃ u : Idx 5 → ℝ, (0 : ℝ) < 1 ∧
    u ∈ kdeCriticalChartDomain 1 (0 : Idx 5) 1 ∧
    kdeScoreRemainder 1 (0 : Idx 5) u ∈ kdeScore 1 '' kdeScoreBranch 1 1 ∧
    deriv (fieldF 1 (kdeZeroChart 1 (0 : Idx 5) 1 u)) (u 0) = 0 ∧
    fieldF 1 (kdeZeroChart 1 (0 : Idx 5) 1 u) (u 0) = 0 ∧
    deriv (kdeScore 1) (kdeScoreInverse 1 1 (kdeScoreRemainder 1 (0 : Idx 5) u)) ≠ 0 ∧
    (Pi.single (0 : Idx 5) 1 : Idx 5 → ℝ) ≠ 0 := by
  obtain ⟨a, ha, har, hd⟩ := exists_symmetricFive_query_degenerate
  let X := symmetricFiveSample a (Real.sqrt 3)
  let u := Function.update X (0 : Idx 5) 0
  have hβ : (0 : ℝ) < 1 := by norm_num
  have hn : (0 : ℕ) < 5 := by norm_num
  have hf : fieldF 1 X 0 = 0 := fieldF_symmetricFive_zero 1 a _
  have hb : 0 - X 0 ∈ kdeScoreBranch 1 1 := by
    norm_num [X, symmetricFiveSample, kdeScoreBranch, kdeScoreRadius]
  have hrem := kdeScoreRemainder_eq_score_of_fieldF_zero hn 1 X 0 hf (0 : Idx 5)
  have hu : kdeScoreRemainder 1 (0 : Idx 5) u ∈ kdeScore 1 '' kdeScoreBranch 1 1 := by
    rw [hrem]
    exact ⟨0 - X 0, hb, rfl⟩
  have he : kdeZeroChart 1 (0 : Idx 5) 1 u = X :=
    kdeZeroChart_update_of_fieldF_zero hβ hn X 0 hf (0 : Idx 5) 1 hb
  have hui : u 0 = 0 := Function.update_self _ _ _
  have hcrit : deriv (fieldF 1 (kdeZeroChart 1 (0 : Idx 5) 1 u)) (u 0) = 0 := by
    rw [he, hui]
    exact hd
  refine ⟨u, hβ, ⟨hu, hcrit⟩, hu, hcrit, ?_, ?_, ?_⟩
  · rw [he, hui]
    exact hf
  · exact kdeScoreBranch_deriv_ne_zero hβ 1 _ (kdeScoreInverse_mem 1 1 _ hu)
  · intro hz
    have hv := congrFun hz (0 : Idx 5)
    norm_num at hv

end Transformer.Modes
