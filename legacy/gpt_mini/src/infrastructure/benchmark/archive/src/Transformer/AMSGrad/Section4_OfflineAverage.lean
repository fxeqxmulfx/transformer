/-
# AMSGrad — an averaged parameter with vanishing objective gap

Offline specialization of arXiv:1904.03590v4, §2 and Corollary 4.5.
For a fixed convex objective, Jensen's inequality converts the regret
guarantee of the original projected algorithm into an objective guarantee
for the actual average of its parameters.
-/

import Transformer.AMSGrad.Section4_OfflineConvergence
import Mathlib.Analysis.Convex.Jensen

open scoped Topology
open Filter Finset

noncomputable section

namespace Transformer.AMSGrad

variable {d : ℕ}

/-- The mean of the first `T` actual parameters, using the source's
one-based indexing. Source: arXiv:1904.03590v4, §2 and Corollary 4.5,
offline specialization. -/
def offlineAverage (S : Setup d) (T : ℕ) : Vec d :=
  ∑ t ∈ Icc 1 T, (1 / (T : ℝ)) • S.x amsgradRule t

/-- A positive-length average is feasible and its objective gap is
between zero and average regret. The convexity, actual projected run,
fixed loss and minimizer are all explicit. Source: arXiv:1904.03590v4,
§2 and Corollary 4.5, offline specialization via Jensen's inequality. -/
theorem offline_average_gap_bound {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (f : Vec d → ℝ) (hf : ∀ t, S.f t = f)
    (star : Vec d) (hmin : ∀ x ∈ F, f star ≤ f x) (T : ℕ) (hT : 0 < T) :
    offlineAverage S T ∈ F ∧ 0 ≤ f (offlineAverage S T) - f star ∧
      f (offlineAverage S T) - f star ≤ S.regret amsgradRule star T / T := by
  have hTne : (T : ℝ) ≠ 0 := (Nat.cast_pos.mpr hT).ne'
  have hw0 : ∀ t ∈ Icc 1 T, (0 : ℝ) ≤ 1 / (T : ℝ) :=
    fun _ _ => div_nonneg zero_le_one (Nat.cast_nonneg T)
  have hw1 : (∑ t ∈ Icc 1 T, (1 / (T : ℝ))) = 1 := by
    simp [Nat.card_Icc, hTne]
  have hmem := hS.convex.sum_mem hw0 hw1
    (fun t _ => x_mem hS amsgradRule t)
  have hconvex : ConvexOn ℝ Set.univ f := by
    rw [← hf 1]
    exact hS.convexOn 1
  have hJ := hconvex.map_sum_le (p := fun t => S.x amsgradRule t)
    hw0 hw1 (fun _ _ => Set.mem_univ _)
  have hregret : (∑ t ∈ Icc 1 T, (1 / (T : ℝ)) * f (S.x amsgradRule t)) - f star =
      S.regret amsgradRule star T / T := by
    simp only [Setup.regret, hf, Finset.sum_sub_distrib, Finset.sum_const,
      nsmul_eq_mul, Nat.card_Icc, Nat.add_sub_cancel]
    rw [← Finset.mul_sum]
    field_simp
  refine ⟨hmem, sub_nonneg.mpr (hmin _ hmem), ?_⟩
  change f (∑ t ∈ Icc 1 T, (1 / (T : ℝ)) • S.x amsgradRule t) - f star ≤ _
  rw [← hregret]
  simpa only [smul_eq_mul] using sub_le_sub_right hJ (f star)

/-- The average-gap hypotheses are realized by a nonconstant convex
loss and a genuine feasible minimizer. Source: arXiv:1904.03590v4,
§2 and Corollary 4.5, offline specialization. -/
example :
    let S := offlineLinearSetup (fun t => 1 / Real.sqrt t) (fun _ => 0) (1 / 4)
    offlineAverage S 1 ∈ Set.Icc (fun _ => -1) (fun _ => 1) ∧
      0 ≤ (fun x : Vec 1 => 1 * x 0) (offlineAverage S 1) - 1 * (-1) ∧
      (fun x : Vec 1 => 1 * x 0) (offlineAverage S 1) - 1 * (-1) ≤
        S.regret amsgradRule (fun _ => -1) 1 / 1 := by
  dsimp only
  simpa only [Nat.cast_one] using offline_average_gap_bound
    (offlineLinearSetup_online (fun t => 1 / Real.sqrt t) (fun _ => 0) (1 / 4))
    (fun x : Vec 1 => 1 * x 0) (fun _ => rfl)
    (fun _ => -1) offlineLinearSetup_minimum.2 1 (by norm_num)

/-- Vanishing average regret implies that the actual averaged parameter
attains the fixed objective's minimum in the limit. Source:
arXiv:1904.03590v4, §2 and Corollary 4.5, offline specialization.
The regret limit is explicit here and is proved for the source schedules
in `offline_average_gap_lambda` and `offline_average_gap_inv`. -/
theorem offline_average_loss_of_regret_limit {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (f : Vec d → ℝ) (hf : ∀ t, S.f t = f)
    (star : Vec d) (hmin : ∀ x ∈ F, f star ≤ f x)
    (hregret : Tendsto (fun T : ℕ => S.regret amsgradRule star T / T) atTop (𝓝 0)) :
    Tendsto (fun T : ℕ => f (offlineAverage S T)) atTop (𝓝 (f star)) := by
  have hgap : Tendsto (fun T : ℕ => f (offlineAverage S T) - f star) atTop (𝓝 0) := by
    apply squeeze_zero' (g := fun T : ℕ => S.regret amsgradRule star T / T)
    · filter_upwards [eventually_gt_atTop (0 : ℕ)] with T hT
      exact (offline_average_gap_bound hS f hf star hmin T hT).2.1
    · filter_upwards [eventually_gt_atTop (0 : ℕ)] with T hT
      exact (offline_average_gap_bound hS f hf star hmin T hT).2.2
    · exact hregret
  simpa only [sub_add_cancel, zero_add] using hgap.add_const (f star)

/-- The helper's convergence premise is established by the original
AMSGrad theorem, not assumed in its concrete witness. Source:
arXiv:1904.03590v4, Corollary 4.5, offline specialization. -/
example :
    let S := offlineLinearSetup (fun t => 1 / Real.sqrt t) (fun _ => 0) (1 / 4)
    Tendsto (fun T : ℕ => (fun x : Vec 1 => 1 * x 0) (offlineAverage S T))
      atTop (𝓝 ((fun x : Vec 1 => 1 * x 0) (fun _ => -1))) := by
  dsimp only
  apply offline_average_loss_of_regret_limit (offlineLinearSetup_online _ _ _)
    (fun x : Vec 1 => 1 * x 0) (fun _ => rfl) (fun _ => -1) offlineLinearSetup_minimum.2
  apply offline_average_gap_inv (offlineLinearSetup_online _ _ _)
    (fun x : Vec 1 => 1 * x 0) (fun _ => rfl) (fun _ => -1)
    offlineLinearSetup_minimum.1 offlineLinearSetup_minimum.2
    (α := 1) (β₁ := 0) (by norm_num) rfl (by norm_num) (by norm_num)
  · intro t
    simp [offlineLinearSetup]
  · norm_num [offlineLinearSetup]
  · norm_num [offlineLinearSetup]
  · norm_num [offlineLinearSetup]

end Transformer.AMSGrad
