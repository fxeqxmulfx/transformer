module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Loss

/-!
# One fixed logit scale for probability-valued experts

Source: `review/flat_spectrum.md`, CM8. Scaling all experts by the same fixed nonnegative
number preserves ordinary CE convexity and the one-hot lower bound. The scale is not a
trained parameter. This permits causal and spectral predictions to share one objective.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments

open Finset ConvexRecall
open Landscape.Neural (crossEntropy sum_exp_pos)

variable {m n C S : Type*} [Fintype C] [Fintype S]

/-- Ordinary observed CE at a fixed logit scale (CM8). -/
noncomputable def scaledLoss (scale : ℝ) (cell : S → m × n) (labels : S → C)
    (P : Table m n C) : ℝ := ∑ s, crossEntropy (scale • P (cell s).1 (cell s).2) (labels s)

/-- A fixed scale leaves the complete CE objective convex (CM8). -/
theorem scaled_loss_convex (scale : ℝ) (cell : S → m × n) (labels : S → C) :
    ConvexOn ℝ Set.univ (scaledLoss scale cell labels) := by
  refine ⟨convex_univ, ?_⟩
  intro P _ Q _ a b ha hb hab
  simp only [scaledLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  have he : scale • (a • P + b • Q) (cell s).1 (cell s).2 =
      a • (scale • P (cell s).1 (cell s).2) + b • (scale • Q (cell s).1 (cell s).2) := by
    funext c
    simp only [Pi.smul_apply, Pi.add_apply, smul_eq_mul]
    ring
  rw [he]
  exact (convex_crossEntropy (labels s)).2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

variable [DecidableEq C]

/-- The lower bound uses the same fixed scale and the actual observed labels (CM8). -/
noncomputable def scaledIdealLoss (scale : ℝ) (labels : S → C) : ℝ :=
  ∑ s, crossEntropy (scale • targetLogits (labels s)) (labels s)

/-- Nonnegative bounded target mass cannot improve on the scaled one-hot CE (CM8). -/
theorem scaled_target_le {scale : ℝ} (hs : 0 ≤ scale) {z : C → ℝ} (t : C)
    (hz : ∀ c, 0 ≤ z c) (ht : z t ≤ 1) :
    crossEntropy (scale • targetLogits t) t ≤ crossEntropy (scale • z) t := by
  rw [crossEntropy_relative, crossEntropy_relative]
  apply Real.log_le_log (sum_exp_pos _ t)
  apply Finset.sum_le_sum
  intro c _
  apply Real.exp_le_exp.mpr
  have h := mul_le_mul_of_nonneg_left (target_relative_le t hz ht c) hs
  simpa only [Pi.smul_apply, smul_eq_mul, mul_sub] using h

/-- The common simplex lower bound survives every nonnegative fixed scale (CM8). -/
theorem scaled_ideal_le {scale : ℝ} (hs : 0 ≤ scale) (cell : S → m × n)
    (labels : S → C) {P : Table m n C} (hP : Rows P) :
    scaledIdealLoss scale labels ≤ scaledLoss scale cell labels P := by
  apply Finset.sum_le_sum
  intro s _
  exact scaled_target_le hs _ (hP.1 _ _) (coordinate_le_one hP _ _ _)

/-- An exact observed fit attains the scaled lower bound (CM8). -/
theorem scaled_loss_eq_ideal (scale : ℝ) (cell : S → m × n) (labels : S → C)
    {P : Table m n C} (hfit : ∀ s, P (cell s).1 (cell s).2 = targetLogits (labels s)) :
    scaledLoss scale cell labels P = scaledIdealLoss scale labels := by
  apply Finset.sum_congr rfl
  intro s _
  rw [hfit s]

example := scaled_target_le (by norm_num : (0 : ℝ) ≤ 10) (true : Bool)
  (z := targetLogits true) (by intro c; simp [targetLogits]; positivity)
  (by simp [targetLogits])

example := scaled_ideal_le (by norm_num : (0 : ℝ) ≤ 10)
  (fun s : Bool => (s, ())) id (rows_pointMass (fun (x : Bool) (_ : Unit) => x))

example := scaled_loss_eq_ideal (10 : ℝ) (fun s : Bool => (s, ())) id
  (P := pointMass (fun (x : Bool) (_ : Unit) => x))
  (by intro s; funext c; simp [pointMass, targetLogits])

end Transformer.Grokking.FlatSpectrum.CoupledMoments
