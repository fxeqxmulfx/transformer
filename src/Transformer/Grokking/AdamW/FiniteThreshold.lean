import Transformer.Grokking.AdamW.CenteredFamily

/-!
# A state-dependent finite-step threshold for the effective loss

Source: Liu et al., arXiv:2205.10343v2, section 3.2, eq:l_eff;
native PyTorch 2.14.1 AdamW first step at lab commit 43d4d66.
The source assumes Euclidean gradient flow. These are separately
derived statements for its one-constraint quotient under the actual
zero-moment finite update, not a transfer of that flow's dynamics.

On the centered family, let `A = 3 / (3 + epsilon * t)` and let
`L = (1 - rate * decay) * t`. The updated coordinates are
`(L, -L/2 + rate*A, -L/2 - rate*A)`. Their norm remains nonzero at
every positive rate because `rate*A` is positive. Comparing their
actual quotient loss gives a strict threshold depending on the scale
`t`. Centering is derived, and norm conservation is not assumed.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.EffectiveTheory

/-- Actual centroid, norm and numerator at the updated family shape.
Source: arXiv:2205.10343v2, appendix conservation quantities and
eq:l_eff, evaluated at the native first-update coordinates. -/
theorem centered_updated_statistics (L b : ℝ) :
    L + (-L / 2 + b) + (-L / 2 - b) = 0 ∧
    squaredNorm L (-L / 2 + b) (-L / 2 - b) = 3 * L ^ 2 / 2 + 2 * b ^ 2 ∧
    numerator L (-L / 2 + b) (-L / 2 - b) = (3 * L / 2 - 3 * b) ^ 2 := by
  unfold squaredNorm numerator EffectiveTheory.residual
  constructor
  · ring
  · constructor <;> ring

/-- The actual finite loss difference, with a strictly positive updated
norm. Source: arXiv:2205.10343v2, eq:l_eff, and native AdamW at
43d4d66. The initial loss is the derived value 3/2, not a chosen target. -/
theorem centered_family_loss_difference (b1 b2 eps decay eta t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (ht : 0 < t) (heta : 0 < eta) :
    lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) -
        normalizedLoss t (-t / 2) (-t / 2) =
      3 * (eta * familyDirection eps t) *
        (2 * (eta * familyDirection eps t) - 3 * ((1 - eta * decay) * t)) /
        (3 * ((1 - eta * decay) * t) ^ 2 / 2 + 2 * (eta * familyDirection eps t) ^ 2) := by
  rw [centered_family_updated_loss b1 b2 eps decay eta t h1 h2 he ht,
    (centered_family_loss t ht).2]
  let L := (1 - eta * decay) * t
  let b := eta * familyDirection eps t
  have hb : 0 < b := by dsimp [b]; exact mul_pos heta (family_direction_pos eps t he ht)
  have hden : 3 * L ^ 2 / 2 + 2 * b ^ 2 ≠ 0 := by positivity
  change normalizedLoss L (-L / 2 + b) (-L / 2 - b) - 3 / 2 =
    3 * b * (2 * b - 3 * L) / (3 * L ^ 2 / 2 + 2 * b ^ 2)
  unfold normalizedLoss
  rw [(centered_updated_statistics L b).2.1, (centered_updated_statistics L b).2.2]
  field_simp
  ring

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 : ℝ) ∧ 0 < (1 / 1000 : ℝ) := by norm_num

/-- Exact finite-step improvement criterion without a sign assumption
on decay. Source: native first-step equations at 43d4d66 and
arXiv:2205.10343v2, eq:l_eff; positivity of the norm is derived. -/
theorem centered_family_finite_descent_iff (b1 b2 eps decay eta t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (ht : 0 < t) (heta : 0 < eta) :
    (lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) <
      normalizedLoss t (-t / 2) (-t / 2)) ↔
      2 * (eta * familyDirection eps t) < 3 * ((1 - eta * decay) * t) := by
  let L := (1 - eta * decay) * t
  let b := eta * familyDirection eps t
  have hb : 0 < b := by dsimp [b]; exact mul_pos heta (family_direction_pos eps t he ht)
  have hp : 0 < 3 * b := by positivity
  have hden : 0 < 3 * L ^ 2 / 2 + 2 * b ^ 2 := by positivity
  have heq := centered_family_loss_difference b1 b2 eps decay eta t h1 h2 he ht heta
  have hsub : (lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) <
      normalizedLoss t (-t / 2) (-t / 2)) ↔
      lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) -
        normalizedLoss t (-t / 2) (-t / 2) < 0 := by constructor <;> intro h <;> linarith
  rw [hsub, heq, div_lt_iff₀ hden]
  rw [MulZeroClass.zero_mul]
  change 3 * b * (2 * b - 3 * L) < 0 ↔ 2 * b < 3 * L
  rw [mul_neg_iff]
  constructor
  · rintro (⟨_, h⟩ | ⟨h, _⟩) <;> linarith
  · intro h
    exact Or.inl ⟨hp, by linarith⟩

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 : ℝ) ∧ 0 < (2 : ℝ) ∧ 0 < (1 / 1000 : ℝ) := by norm_num

/-- Explicit positive threshold for nonnegative decoupled decay.
Source: arXiv:2205.10343v2, eq:l_eff, under native AdamW at
43d4d66. A fixed positive learning rate need not be below this
state-dependent threshold as the initial embedding scale changes. -/
theorem centered_family_rate_threshold (b1 b2 eps decay eta t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hd : 0 ≤ decay)
    (ht : 0 < t) (heta : 0 < eta) :
    (lossAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) <
      normalizedLoss t (-t / 2) (-t / 2)) ↔
      eta < 3 * t / (2 * familyDirection eps t + 3 * decay * t) := by
  have hA := family_direction_pos eps t he ht
  have hden : 0 < 2 * familyDirection eps t + 3 * decay * t := by positivity
  rw [centered_family_finite_descent_iff b1 b2 eps decay eta t h1 h2 he ht heta,
    lt_div_iff₀ hden]
  constructor <;> intro h <;> nlinarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 ≤ (1 / 10 : ℝ) ∧
    0 < (1 / 1000000 : ℝ) ∧ 0 < (1 / 1000 : ℝ) := by norm_num

/-- The threshold itself is a positive finite-state quantity. Source:
the native finite-update comparison above, using arXiv:2205.10343v2,
eq:l_eff. Positivity supplies no uniform lower bound over all scales. -/
theorem centered_family_threshold_pos (eps decay t : ℝ)
    (he : 0 < eps) (hd : 0 ≤ decay) (ht : 0 < t) :
    0 < 3 * t / (2 * familyDirection eps t + 3 * decay * t) := by
  have hA := family_direction_pos eps t he ht
  positivity

example : 0 < (1 : ℝ) ∧ 0 ≤ (1 / 10 : ℝ) ∧ 0 < (1 : ℝ) := by norm_num

/-- Every positive first step of this family stays nonzero and centered.
Source: native AdamW at 43d4d66 and arXiv:2205.10343v2, appendix
norm/centroid quantities. This is a special-family domain guarantee;
neither quantity is assumed globally conserved by AdamW. -/
theorem centered_family_updated_domain (b1 b2 eps decay eta t : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (ht : 0 < t) (heta : 0 < eta) :
    0 < squaredNormAfterFirstStep b1 b2 eps decay eta t (-t / 2) (-t / 2) ∧
      firstUpdate b1 b2 eps decay eta t (quotientGradient t (-t / 2) (-t / 2)).1 +
        firstUpdate b1 b2 eps decay eta (-t / 2) (quotientGradient t (-t / 2) (-t / 2)).2.1 +
        firstUpdate b1 b2 eps decay eta (-t / 2) (quotientGradient t (-t / 2) (-t / 2)).2.2 = 0 := by
  have h := centered_family_update b1 b2 eps decay eta t h1 h2 he ht
  have h0 := congrArg Prod.fst h
  have h1' := congrArg (fun p : ℝ × ℝ × ℝ => p.2.1) h
  have h2' := congrArg (fun p : ℝ × ℝ × ℝ => p.2.2) h
  dsimp at h0 h1' h2'
  have hs := centered_updated_statistics ((1 - eta * decay) * t) (eta * familyDirection eps t)
  constructor
  · unfold squaredNormAfterFirstStep
    rw [h0, h1', h2', hs.2.1]
    have hb : 0 < eta * familyDirection eps t := mul_pos heta (family_direction_pos eps t he ht)
    positivity
  · rw [h0, h1', h2']
    exact hs.1

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (1 / 1000000 : ℝ) ∧
    0 < (1 / 1000 : ℝ) := by norm_num

end Transformer.Grokking.AdamW
