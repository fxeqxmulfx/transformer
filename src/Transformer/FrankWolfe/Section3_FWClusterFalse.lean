/-
# Attention's forward pass and Frank-Wolfe — `thm: fw.cluster` is false

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §3.

`thm: fw.cluster` claims `J^t(x_i^{t+1}) ≤ (2/(t+1)) λ_max(B_*^0) 𝖽(𝒦^0)²`
under `0 ∈ 𝒦^0`.  Its proof (`sec: bach.proof`) uses `0 ∈ 𝒦^t` for every
`t`, but only `0 ∈ 𝒦^0` is assumed, and the hull shrinks away from the origin.
On the line with `B_* = 1` and particles `3, -1`:

* `t = 0`, `γ = 1`: each particle jumps to the other end, `(-1, 3)`;
* `t = 1`, `γ = 2/3`: `(5/3, 1/3)`, and `0 ∉ 𝒦^2 = [1/3, 5/3]`;
* `t ≥ 2`: the particle at `1/3` is the left end and stays there, the other
  one approaches it, `1/3 + 8/(t(t+1))`.

So `J^t(x_2^{t+1}) = 1/18` for every `t ≥ 1`, while the claimed bound is
`32/(t+1)`.  No `argmin` in the trajectory is a tie.
-/

import Transformer.FrankWolfe.Section3_NegativeDefinite

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- A minimizer of `⟨a, ·⟩` over the configuration's points minimizes it over
their hull. -/
theorem isMinimizerOn_configHull {a : EucSpace d} {X : Idx n → EucSpace d} (k : Idx n)
    (h : ∀ j, inner (𝕜 := ℝ) a (X k) ≤ inner (𝕜 := ℝ) a (X j)) :
    IsMinimizerOn a (configHull X) (X k) := by
  have hconv : Convex ℝ {w : EucSpace d | inner (𝕜 := ℝ) a (X k) ≤ inner (𝕜 := ℝ) a w} :=
    convex_halfSpace_ge (innerₛₗ ℝ a).isLinear _
  have hsub : configHull X ⊆ {w | inner (𝕜 := ℝ) a (X k) ≤ inner (𝕜 := ℝ) a w} :=
    convexHull_min (Set.range_subset_iff.2 h) hconv
  exact ⟨subset_convexHull ℝ _ ⟨k, rfl⟩, fun _ hz => hsub hz⟩

/-- The point `a e₀` of `ℝ^1`. -/
noncomputable def lineVec (a : ℝ) : EucSpace 1 := a • EuclideanSpace.single 0 1

theorem inner_lineVec (a b : ℝ) :
    inner (𝕜 := ℝ) (lineVec a) (lineVec b) = a * b := by
  simp [lineVec, real_inner_smul_left, real_inner_smul_right, mul_comm]

theorem norm_lineVec (a : ℝ) : ‖lineVec a‖ = |a| := by
  simp [lineVec, norm_smul]

theorem lineVec_step (a b c : ℝ) :
    lineVec a + c • (lineVec b - lineVec a) = lineVec (a + c * (b - a)) := by
  simp only [lineVec, ← sub_smul, smul_smul, ← add_smul]

/-- The positions of the counterexample to `thm: fw.cluster`. -/
noncomputable def cexPos : ℕ → Idx 2 → ℝ
  | 0 => ![3, -1]
  | 1 => ![-1, 3]
  | t + 2 => ![1 / 3 + 8 / (((t : ℝ) + 2) * ((t : ℝ) + 3)), 1 / 3]

/-- The trajectory of the counterexample is a Frank-Wolfe trajectory with
`B_* = 1` and `γ^t = 2/(t+2)`. -/
theorem cex_isFrankWolfeStep (t : ℕ) :
    IsFrankWolfeStep (ContinuousLinearMap.id ℝ (EucSpace 1)) (2 / ((t : ℝ) + 2))
      (fun i => lineVec (cexPos t i)) (fun i => lineVec (cexPos (t + 1) i)) := by
  have key : ∀ (k : Idx 2) (i : Idx 2),
      (∀ j, cexPos t i * cexPos t k ≤ cexPos t i * cexPos t j) →
      cexPos (t + 1) i = cexPos t i + 2 / ((t : ℝ) + 2) * (cexPos t k - cexPos t i) →
      ∃ y, IsMinimizerOn ((ContinuousLinearMap.id ℝ (EucSpace 1)) (lineVec (cexPos t i)))
        (configHull fun i => lineVec (cexPos t i)) y ∧
        lineVec (cexPos (t + 1) i) =
          lineVec (cexPos t i) + (2 / ((t : ℝ) + 2)) • (y - lineVec (cexPos t i)) := by
    intro k i hmin heq
    refine ⟨_, isMinimizerOn_configHull k fun j => ?_, by rw [lineVec_step, heq]⟩
    simpa [inner_lineVec] using hmin j
  intro i
  match t with
  | 0 =>
    fin_cases i
    · exact key 1 0 (by norm_num [Fin.forall_fin_two, cexPos]) (by norm_num [cexPos])
    · exact key 0 1 (by norm_num [Fin.forall_fin_two, cexPos]) (by norm_num [cexPos])
  | 1 =>
    fin_cases i
    · exact key 1 0 (by norm_num [Fin.forall_fin_two, cexPos]) (by norm_num [cexPos])
    · exact key 0 1 (by norm_num [Fin.forall_fin_two, cexPos]) (by norm_num [cexPos])
  | t + 2 =>
    have hp : 0 < 8 / (((t : ℝ) + 2) * ((t : ℝ) + 3)) := by positivity
    fin_cases i
    · refine key 1 0 ?_ ?_
      · simp only [Fin.forall_fin_two, cexPos]
        simp
        nlinarith
      · simp [cexPos]; field_simp; ring
    · refine key 1 1 ?_ ?_
      · simp only [Fin.forall_fin_two, cexPos]
        simp
        linarith
      · simp [cexPos]

/-- **`thm: fw.cluster` is false.**  With `d = 1`, `n = 2`, `B_*^t ≡ 1`,
`λ = 1`, `γ^t = 2/(t+2)` and initial particles `3, -1` (so `0 ∈ 𝒦^0`), the
second particle sits at `1/3` from `t = 2` on: `J^t(x_2^{t+1}) = 1/18`, while
`(2/(t+1)) λ 𝖽(𝒦^0)² ≤ 32/(t+1)`, which is smaller at `t = 1000`.

The source's proof uses `0 ∈ 𝒦^t` for every `t` (to get `J^t ≥ 0` on `𝒦^t`
from convexity), which does not follow from `0 ∈ 𝒦^0`: `𝒦^2 = [1/3, 5/3]`.

Source: arXiv:2508.09628v1, §3, `thm: fw.cluster`, and its proof in
`sec: bach.proof`. -/
theorem not_fw_cluster : ¬ ∀ (d n : ℕ) (Bs : ℕ → ParamMatrix d) (γ : ℕ → ℝ)
    (x : ℕ → Idx n → EucSpace d) (lam : ℝ),
    (∀ t : ℕ, ContinuousLinearMap.IsPositive (Bs t)) →
    (∀ t : ℕ, ContinuousLinearMap.IsPositive (Bs t - Bs (t + 1))) →
    (∀ t : ℕ, γ t = 2 / ((t : ℝ) + 2)) →
    (0 : EucSpace d) ∈ configHull (x 0) →
    (∀ y : EucSpace d, inner (𝕜 := ℝ) (Bs 0 y) y ≤ lam * ‖y‖ ^ 2) →
    (∀ t : ℕ, IsFrankWolfeStep (Bs t) (γ t) (x t) (x (t + 1))) →
    ∀ (t : ℕ) (i : Idx n),
      quadForm (Bs t) (x (t + 1) i)
        ≤ 2 / ((t : ℝ) + 1) * lam * Metric.diam (configHull (x 0)) ^ 2 := by
  intro H
  set x : ℕ → Idx 2 → EucSpace 1 := fun t i => lineVec (cexPos t i)
  have hmem : ∀ i : Idx 2, x 0 i ∈ configHull (x 0) := fun i => subset_convexHull ℝ _ ⟨i, rfl⟩
  have h0 : (0 : EucSpace 1) ∈ configHull (x 0) := by
    have := (convex_convexHull ℝ _) (hmem 0) (hmem 1) (by norm_num : (0 : ℝ) ≤ 1 / 4)
      (by norm_num : (0 : ℝ) ≤ 3 / 4) (by norm_num)
    show (0 : EucSpace 1) ∈ convexHull ℝ (Set.range (x 0))
    convert this using 1
    simp [x, cexPos, lineVec, smul_smul]
    norm_num
  have hsub : configHull (x 0) ⊆ Metric.closedBall (lineVec 1) 2 := by
    refine convexHull_min (Set.range_subset_iff.2 fun i => ?_) (convex_closedBall _ _)
    rw [Metric.mem_closedBall, dist_eq_norm, show x 0 i - lineVec 1 = lineVec (cexPos 0 i - 1) by
      simp only [x, lineVec, sub_smul], norm_lineVec]
    fin_cases i <;> simp [cexPos] <;> norm_num
  have hD : Metric.diam (configHull (x 0)) ≤ 4 :=
    (Metric.diam_mono hsub Metric.isBounded_closedBall).trans
      ((Metric.diam_closedBall (by norm_num)).trans (by norm_num))
  have hbound := H 1 2 (fun _ => ContinuousLinearMap.id ℝ _) (fun t => 2 / ((t : ℝ) + 2)) x 1
    (fun _ => ContinuousLinearMap.isPositive_id) (fun _ => by simp [ContinuousLinearMap.isPositive_zero]) (fun _ => rfl) h0
    (fun y => by simp) cex_isFrankWolfeStep 1000 1
  have hJ : quadForm (ContinuousLinearMap.id ℝ (EucSpace 1)) (x 1001 1) = 1 / 18 := by
    simp only [quadForm, ContinuousLinearMap.coe_id', id, x, inner_lineVec, cexPos]
    norm_num
  rw [hJ] at hbound
  have hD2 : Metric.diam (configHull (x 0)) ^ 2 ≤ 16 := by
    nlinarith [Metric.diam_nonneg (s := configHull (x 0))]
  norm_num at hbound
  nlinarith

end FrankWolfe
end Transformer
