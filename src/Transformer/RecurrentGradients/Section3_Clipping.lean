/-
# Scaling down the gradients

arXiv:1211.5063, §3.2, Algorithm 1: `ĝ ← ∂L/∂θ`, and if `‖ĝ‖ ≥ threshold`,
`ĝ ← (threshold/‖ĝ‖) ĝ` (`clip`).  The clipped gradient has norm
`min ‖g‖ threshold` (`norm_clip`) and is `g` times `min 1 (threshold/‖g‖)`,
"adapting the learning rate based on the norm of the gradient"
(`clip_eq_smul`).  For a positive threshold it is a positive multiple of a
nonzero gradient (`inner_clip_pos`), so a small enough step against it lowers
the loss (`eventually_lt_clip`): "ensuring that we always move in a descent
direction with respect to the current mini-batch".

Mikolov's proposal, "clipping the gradient's temporal components element-wise
(clipping an entry when it exceeds in absolute value a fixed threshold)"
(§3.1), from which the paper diverged for that guarantee, does not give it:
the temporal components `10`, `-19/2`, `-2/5` of the derivative `1/10` clip at
`1` to a sum of `-2/5`, so a small step against the clipped sum raises every
loss with that derivative (`eventually_gt_clampEntry`).
-/

import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.InnerProductSpace.Basic

open Filter Topology
open scoped RealInnerProductSpace

namespace Transformer.RecurrentGradients

/-- A function with a negative derivative at `0` is below its value at `0` just
right of `0`. -/
theorem eventually_lt_of_hasDerivAt {φ : ℝ → ℝ} {φ' : ℝ} (hφ : HasDerivAt φ φ' 0)
    (h : φ' < 0) : ∀ᶠ η in 𝓝[>] 0, φ η < φ 0 := by
  have hs := (hasDerivAt_iff_tendsto_slope.1 hφ).mono_left
    (nhdsWithin_mono _ fun x (hx : 0 < x) => hx.ne')
  filter_upwards [hs.eventually (gt_mem_nhds h), self_mem_nhdsWithin] with η hη (hη0 : 0 < η)
  rw [slope_def_field, sub_zero, div_lt_iff₀ hη0, zero_mul] at hη
  linarith

/-- The hypotheses of `eventually_lt_of_hasDerivAt` are satisfiable: `φ = -id`. -/
example : ∀ᶠ η in 𝓝[>] 0, -η < -(0 : ℝ) :=
  eventually_lt_of_hasDerivAt (hasDerivAt_id (0 : ℝ)).neg (by norm_num)

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Algorithm 1**: `g` rescaled to norm `τ` when `‖g‖ ≥ τ`. -/
noncomputable def clip (τ : ℝ) (g : E) : E := if τ ≤ ‖g‖ then (τ / ‖g‖) • g else g

/-- The clipped gradient has norm `min ‖g‖ τ`. -/
theorem norm_clip {τ : ℝ} (hτ : 0 ≤ τ) (g : E) : ‖clip τ g‖ = min ‖g‖ τ := by
  unfold clip
  split_ifs with h
  · rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (div_nonneg hτ (norm_nonneg g)),
      min_eq_right h]
    rcases (norm_nonneg g).eq_or_lt with h0 | h0
    · rw [← h0, div_zero, zero_mul]
      exact (le_antisymm (h0 ▸ h) hτ).symm
    · exact div_mul_cancel₀ τ h0.ne'
  · exact (min_eq_left (not_le.1 h).le).symm

/-- The hypothesis of `norm_clip` is satisfiable: `τ = 1`. -/
example (g : E) : ‖clip 1 g‖ = min ‖g‖ 1 := norm_clip zero_le_one g

/-- **§3.2: clipping adapts the learning rate**: the clipped gradient is `g`
times `min 1 (τ/‖g‖)`. -/
theorem clip_eq_smul {τ : ℝ} (hτ : 0 < τ) (g : E) : clip τ g = min 1 (τ / ‖g‖) • g := by
  unfold clip
  split_ifs with h
  · rw [min_eq_right ((div_le_one (hτ.trans_le h)).2 h)]
  · rcases eq_or_ne g 0 with rfl | hg
    · simp
    · rw [min_eq_left ((one_le_div (norm_pos_iff.2 hg)).2 (not_le.1 h).le), one_smul]

/-- The hypothesis of `clip_eq_smul` is satisfiable: `τ = 1`. -/
example (g : E) : clip 1 g = min 1 (1 / ‖g‖) • g := clip_eq_smul one_pos g

/-- For `τ > 0` the clipped gradient is a positive multiple of a nonzero `g`. -/
theorem inner_clip_pos {τ : ℝ} (hτ : 0 < τ) {g : E} (hg : g ≠ 0) : 0 < ⟪g, clip τ g⟫ := by
  rw [clip_eq_smul hτ, real_inner_smul_right, real_inner_self_eq_norm_sq]
  exact mul_pos (lt_min one_pos (div_pos hτ (norm_pos_iff.2 hg))) (by positivity)

/-- The hypotheses of `inner_clip_pos` are satisfiable: `g = 1` in `ℝ`. -/
example : 0 < ⟪(1 : ℝ), clip 1 (1 : ℝ)⟫ := inner_clip_pos one_pos one_ne_zero

variable [CompleteSpace E]

/-- A step against `d` lowers `L` for all small enough rates when `⟪∇L, d⟫ > 0`:
`-d` is a descent direction. -/
theorem eventually_lt_of_inner_pos {L : E → ℝ} {g θ d : E} (hL : HasGradientAt L g θ)
    (h : 0 < ⟪g, d⟫) : ∀ᶠ η in 𝓝[>] (0 : ℝ), L (θ - η • d) < L θ := by
  have hline : HasDerivAt (fun η : ℝ => θ - η • d) (-d) 0 := by
    simpa using HasDerivAt.const_sub θ (HasDerivAt.smul_const (hasDerivAt_id (0 : ℝ)) d)
  have hφ := HasFDerivAt.comp_hasDerivAt_of_eq 0 (HasGradientAt.hasFDerivAt hL) hline (by simp)
  simp only [InnerProductSpace.toDual_apply_apply, inner_neg_right] at hφ
  simpa using eventually_lt_of_hasDerivAt hφ (neg_neg_of_pos h)

/-- The hypotheses of `eventually_lt_of_inner_pos` are satisfiable: the identity
on `ℝ`, whose gradient is `1`. -/
example : ∀ᶠ η in 𝓝[>] (0 : ℝ), (0 : ℝ) - η • (1 : ℝ) < 0 :=
  eventually_lt_of_inner_pos (L := id) (hasDerivAt_id (0 : ℝ)).hasGradientAt' (by simp)

/-- **§3.2: norm clipping moves in a descent direction.**  If `L` has the
gradient `g ≠ 0` at `θ` and `τ > 0`, a small enough step against the clipped
gradient lowers `L`. -/
theorem eventually_lt_clip {L : E → ℝ} {g θ : E} (hL : HasGradientAt L g θ) (hg : g ≠ 0)
    {τ : ℝ} (hτ : 0 < τ) : ∀ᶠ η in 𝓝[>] (0 : ℝ), L (θ - η • clip τ g) < L θ :=
  eventually_lt_of_inner_pos hL (inner_clip_pos hτ hg)

/-- The hypotheses of `eventually_lt_clip` are satisfiable: the identity on
`ℝ`, whose gradient is `1`. -/
example : ∀ᶠ η in 𝓝[>] (0 : ℝ), (0 : ℝ) - η • clip 1 (1 : ℝ) < 0 :=
  eventually_lt_clip (L := id) (hasDerivAt_id (0 : ℝ)).hasGradientAt' one_ne_zero one_pos

/-- Mikolov's clipping of an entry at `τ`: clamped to `[-τ, τ]`. -/
def clampEntry (τ x : ℝ) : ℝ := max (-τ) (min τ x)

/-- Three temporal components `10`, `-19/2`, `-2/5`, of sum `1/10`. -/
noncomputable def temporal : Fin 3 → ℝ := ![10, -19 / 2, -2 / 5]

theorem sum_temporal : ∑ k, temporal k = 1 / 10 := by
  norm_num [temporal, Fin.sum_univ_three]

theorem sum_clampEntry_temporal : ∑ k, clampEntry 1 (temporal k) = -2 / 5 := by
  norm_num [temporal, clampEntry, Fin.sum_univ_three]

/-- **Clipping the temporal components entry by entry can move uphill.**  Every
loss whose derivative is the sum `1/10` of the temporal components `temporal`
rises under a small enough step against the sum `-2/5` of their clipped
entries, where clipping the sum by norm descends (`eventually_lt_clip`). -/
theorem eventually_gt_clampEntry {L : ℝ → ℝ} {θ : ℝ} (hL : HasDerivAt L (∑ k, temporal k) θ) :
    ∀ᶠ η in 𝓝[>] (0 : ℝ), L θ < L (θ - η * ∑ k, clampEntry 1 (temporal k)) := by
  rw [sum_temporal] at hL
  have h := eventually_lt_of_inner_pos (L := fun x => -L x) (g := -(1 / 10))
    (d := ∑ k, clampEntry 1 (temporal k)) (HasDerivAt.neg hL).hasGradientAt'
    (by rw [sum_clampEntry_temporal]; norm_num)
  simpa using h

/-- The hypothesis of `eventually_gt_clampEntry` is satisfiable: the loss
`θ/10`. -/
example : ∀ᶠ η in 𝓝[>] (0 : ℝ), (fun x : ℝ => x / 10) 0 <
    (fun x : ℝ => x / 10) (0 - η * ∑ k, clampEntry 1 (temporal k)) :=
  eventually_gt_clampEntry (by
    rw [sum_temporal]; simpa using HasDerivAt.div_const (hasDerivAt_id (0 : ℝ)) 10)

end Transformer.RecurrentGradients
