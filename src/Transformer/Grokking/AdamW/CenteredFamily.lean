import Transformer.Grokking.AdamW.LossDescent

/-!
# Exact first update of a centered effective-loss family

Source: Liu et al., arXiv:2205.10343v2, section 3.2, eq:l_eff;
PyTorch 2.14.1 AdamW at lab commit 43d4d66, with zero moment buffers.
The one-constraint quotient is specialized to `(t, -t/2, -t/2)`.
This family is centered, has nonzero gradient for positive `t`, and
has the same initial normalized loss for every positive scale.

Derive the true gradient, bias-corrected adaptive direction, finite
updated coordinates, norm and loss difference. Centering is preserved
by this particular family; it is not imposed on the update. This permits
an exact finite-step threshold without identifying AdamW with the
source's assumed Euclidean flow or with actual GPTMini training.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.EffectiveTheory

/-- Adaptive direction magnitude derived below for the centered family.
Source: native first-step normalization at 43d4d66, applied to the actual
quotient gradient of arXiv:2205.10343v2, eq:l_eff. -/
noncomputable def familyDirection (eps t : ℝ) : ℝ := 3 / (3 + eps * t)

/-- The family's actual centroid, norm and numerator. Source:
arXiv:2205.10343v2, appendix conservation quantities and eq:l_eff,
specialized to one parallelogram residual. -/
theorem centered_family_statistics (t : ℝ) :
    t + (-t / 2) + (-t / 2) = 0 ∧
    squaredNorm t (-t / 2) (-t / 2) = 3 * t ^ 2 / 2 ∧
    numerator t (-t / 2) (-t / 2) = 9 * t ^ 2 / 4 := by
  unfold squaredNorm numerator EffectiveTheory.residual
  constructor
  · ring
  · constructor <;> ring

/-- Every positive scale gives the same nonzero normalized loss.
Source: arXiv:2205.10343v2, eq:l_eff; the nonzero norm is checked
rather than hidden in a division convention. -/
theorem centered_family_loss (t : ℝ) (ht : 0 < t) :
    squaredNorm t (-t / 2) (-t / 2) ≠ 0 ∧
      normalizedLoss t (-t / 2) (-t / 2) = 3 / 2 := by
  have hn := (centered_family_statistics t).2.1
  have hN := (centered_family_statistics t).2.2
  have hZ : squaredNorm t (-t / 2) (-t / 2) ≠ 0 := by rw [hn]; positivity
  refine ⟨hZ, ?_⟩
  unfold normalizedLoss
  rw [hn, hN]
  field_simp
  ring

example : 0 < (1 / 1000000 : ℝ) := by norm_num

/-- The family has a genuine nonstationary quotient gradient.
Source: arXiv:2205.10343v2, appendix quotient differentiation;
denominator partial derivatives are retained. -/
theorem centered_family_gradient (t : ℝ) (ht : 0 < t) :
    quotientGradient t (-t / 2) (-t / 2) = (0, -3 / t, 3 / t) := by
  rw [quotientGradient_eq _ _ _ (centered_family_loss t ht).1]
  unfold numerator EffectiveTheory.residual squaredNorm
  have htn : t ≠ 0 := by positivity
  apply Prod.ext
  · dsimp
    field_simp
    ring
  · apply Prod.ext <;> dsimp <;> field_simp <;> ring

example : 0 < (1 : ℝ) := by norm_num

/-- The finite update's adaptive magnitude is positive. Source:
native first-step AdamW normalization at 43d4d66; epsilon and scale
are positive as in the finite-step comparison. -/
theorem family_direction_pos (eps t : ℝ) (he : 0 < eps) (ht : 0 < t) :
    0 < familyDirection eps t := by
  unfold familyDirection
  positivity

example : 0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000000 : ℝ) := by norm_num

/-- Actual native directions on the centered family, not an assumed
common preconditioner. Source: PyTorch 2.14.1 first-step equations
at 43d4d66 and arXiv:2205.10343v2, eq:l_eff. -/
theorem centered_family_directions (b1 b2 eps t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (ht : 0 < t) :
    (firstDirection b1 b2 eps (quotientGradient t (-t / 2) (-t / 2)).1,
      firstDirection b1 b2 eps (quotientGradient t (-t / 2) (-t / 2)).2.1,
      firstDirection b1 b2 eps (quotientGradient t (-t / 2) (-t / 2)).2.2) =
      (0, -familyDirection eps t, familyDirection eps t) := by
  rw [centered_family_gradient t ht]
  have hd (g : ℝ) := firstDirection_eq b1 b2 eps g h1 h2
  simp only [hd]
  have hp : 0 < (3 : ℝ) / t := by positivity
  have hn : (-3 : ℝ) / t < 0 := div_neg_iff.mpr (Or.inr ⟨by norm_num, ht⟩)
  rw [abs_of_neg hn, abs_of_pos hp]
  norm_num
  unfold familyDirection
  have htn : t ≠ 0 := by positivity
  have hden : 3 + eps * t ≠ 0 := by positivity
  constructor <;> field_simp

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 : ℝ) := by norm_num

/-- The updated coordinates follow from the true family gradient and
native decoupled update. Source: AdamW at 43d4d66; no centering step
or optimizer modification is inserted. -/
theorem centered_family_update (b1 b2 eps decay eta t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (ht : 0 < t) :
    (firstUpdate b1 b2 eps decay eta t (quotientGradient t (-t / 2) (-t / 2)).1,
      firstUpdate b1 b2 eps decay eta (-t / 2) (quotientGradient t (-t / 2) (-t / 2)).2.1,
      firstUpdate b1 b2 eps decay eta (-t / 2) (quotientGradient t (-t / 2) (-t / 2)).2.2) =
      ((1 - eta * decay) * t,
        -((1 - eta * decay) * t) / 2 + eta * familyDirection eps t,
        -((1 - eta * decay) * t) / 2 - eta * familyDirection eps t) := by
  have h := centered_family_directions b1 b2 eps t h1 h2 he ht
  have h0 := congrArg Prod.fst h
  have h1' := congrArg (fun p : ℝ × ℝ × ℝ => p.2.1) h
  have h2' := congrArg (fun p : ℝ × ℝ × ℝ => p.2.2) h
  unfold firstUpdate
  dsimp at h0 h1' h2'
  rw [h0, h1', h2']
  apply Prod.ext
  · ring
  · apply Prod.ext <;> dsimp <;> ring

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 : ℝ) ∧ 0 < (2 : ℝ) := by norm_num

/-- Actual loss at the three updated coordinates. Source:
arXiv:2205.10343v2, eq:l_eff, evaluated after native AdamW at
43d4d66. The norm is computed, not assumed conserved. -/
theorem centered_family_updated_loss (b1 b2 eps decay eta t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (ht : 0 < t) :
    lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) =
      normalizedLoss ((1 - eta * decay) * t)
        (-((1 - eta * decay) * t) / 2 + eta * familyDirection eps t)
        (-((1 - eta * decay) * t) / 2 - eta * familyDirection eps t) := by
  have h := centered_family_update b1 b2 eps decay eta t h1 h2 he ht
  have h0 := congrArg Prod.fst h
  have h1' := congrArg (fun p : ℝ × ℝ × ℝ => p.2.1) h
  have h2' := congrArg (fun p : ℝ × ℝ × ℝ => p.2.2) h
  unfold lossAfterFirstStep
  dsimp at h0 h1' h2'
  rw [h0, h1', h2']

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000000 : ℝ) := by norm_num

end Transformer.Grokking.AdamW
