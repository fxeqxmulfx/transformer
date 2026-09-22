/-
# The stream before the norm, seen through it

**Not a statement of any paper.**  In a pre-norm block of parameter-golf
(github.com/openai/parameter-golf,
`records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`,
`Block.forward`) the residual stream `x_i ∈ ℝ^d` is not normalised between the
embedding and the final norm.  Attention reads the direction of
`x_in = mix[0] ⊙ x + mix[1] ⊙ x0`, the feed-forward term that of the stream
after attention (the RMS norm, up to the constant factors `√d` and
`ln_scale_factor`).  With `mix[0] = 1 + c` in every channel the stream is

  `ẋ_i = c x_i + g_i(t)`,

where `g_i` is everything the block adds to token `i` — attention with any
values, the feed-forward term, the injection `mix[1] ⊙ x0_i`; for the drive of
a block, `‖g_i‖ ≤ H A N + B + Z` (`norm_blockDrive_le`).

The directions of a stream `ẋ_i = v_i` obey the sphere model with the drive
`v_i / ‖x_i‖` (`hasDerivAt_normalize`, `isDrivenFlowOn_direction`), and `c x_i`
drops out of it: the sphere of the survey is the stream seen through the norm,
with a clock that runs at the speed `1 / ‖x_i‖` of each token.
`Perspective.RawGrowth` shows what that clock does when the stream grows.
-/

import Transformer.Perspective.DrivenFlow
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.SpecialFunctions.Sqrt

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Directions of nearby vectors are close, relative to the length:**
`‖u/‖u‖ - v/‖v‖‖ ≤ 2 ‖u - v‖ / ‖u‖` for `u ≠ 0` and any `v` (with `0/‖0‖ = 0`). -/
theorem norm_normalize_sub_le_div {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {u v : E} (hu : u ≠ 0) : ‖‖u‖⁻¹ • u - ‖v‖⁻¹ • v‖ ≤ 2 * ‖u - v‖ / ‖u‖ := by
  have hu0 : 0 < ‖u‖ := norm_pos_iff.2 hu
  rcases eq_or_ne v 0 with rfl | hv
  · rw [norm_zero, inv_zero, zero_smul, sub_zero, sub_zero, norm_smul, norm_inv, norm_norm,
      inv_mul_cancel₀ hu0.ne', mul_div_assoc, div_self hu0.ne']
    norm_num
  have hv0 : 0 < ‖v‖ := norm_pos_iff.2 hv
  have hsplit : ‖u‖⁻¹ • u - ‖v‖⁻¹ • v = ‖u‖⁻¹ • (u - v) + (‖u‖⁻¹ - ‖v‖⁻¹) • v := by
    rw [smul_sub, sub_smul]; abel
  have h1 : ‖‖u‖⁻¹ • (u - v)‖ = ‖u - v‖ / ‖u‖ := by
    rw [norm_smul, norm_inv, norm_norm, inv_mul_eq_div]
  have h2 : ‖(‖u‖⁻¹ - ‖v‖⁻¹) • v‖ ≤ ‖u - v‖ / ‖u‖ := by
    rw [norm_smul, Real.norm_eq_abs, inv_sub_inv hu0.ne' hv0.ne', abs_div, abs_mul, abs_of_pos hu0,
      abs_of_pos hv0, div_mul_eq_mul_div, mul_div_mul_right _ _ hv0.ne']
    exact div_le_div_of_nonneg_right ((abs_norm_sub_norm_le v u).trans_eq (norm_sub_rev v u))
      hu0.le
  rw [hsplit]
  calc ‖‖u‖⁻¹ • (u - v) + (‖u‖⁻¹ - ‖v‖⁻¹) • v‖
      ≤ ‖‖u‖⁻¹ • (u - v)‖ + ‖(‖u‖⁻¹ - ‖v‖⁻¹) • v‖ := norm_add_le _ _
    _ ≤ ‖u - v‖ / ‖u‖ + ‖u - v‖ / ‖u‖ := add_le_add h1.le h2
    _ = 2 * ‖u - v‖ / ‖u‖ := by ring

/-- The hypothesis of `norm_normalize_sub_le_div` is satisfiable: `u = 1` in `ℝ`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **The direction of a moving vector moves on the sphere:** if `ẋ = v` and
`x ≠ 0`, then `y = x / ‖x‖` has `ẏ = Proj_y(v / ‖x‖)`.

Source: none — posed here; the tangential part of `v`, as in `eq: albert`
(arXiv:2312.10794v5, §2.3), divided by the length. -/
theorem hasDerivAt_normalize {f : ℝ → EucSpace d} {f' : EucSpace d} {t : ℝ}
    (hf : HasDerivAt f f' t) (h0 : f t ≠ 0) :
    HasDerivAt (fun s => ‖f s‖⁻¹ • f s) (proj d (‖f t‖⁻¹ • f t) (‖f t‖⁻¹ • f')) t := by
  have hn : 0 < ‖f t‖ := norm_pos_iff.2 h0
  have hN : HasDerivAt (fun s => ‖f s‖) (2 * inner (𝕜 := ℝ) (f t) f' / (2 * ‖f t‖)) t := by
    have h := hf.norm_sq.sqrt (by positivity)
    simp only [Real.sqrt_sq (norm_nonneg _)] at h
    exact h
  refine ((hN.fun_inv hn.ne').fun_smul hf).congr_deriv ?_
  rw [proj, real_inner_smul_left, real_inner_smul_right, smul_smul, sub_eq_add_neg, ← neg_smul]
  congr 2
  field_simp

/-- The hypotheses of `hasDerivAt_normalize` are satisfiable: a vector at rest. -/
example : HasDerivAt (fun _ : ℝ => (basePoint 0 : EucSpace 1)) 0 (0 : ℝ) ∧
    (basePoint 0 : EucSpace 1) ≠ 0 :=
  ⟨hasDerivAt_const _ _, by simp [basePoint]⟩

/-- **The directions of a stream are a driven flow on the sphere.**  If every
`x_i(t)` is nonzero and `ẋ_i = v_i`, the directions `y_i = x_i / ‖x_i‖` solve
`ẏ_i = Proj_{y_i}(v_i / ‖x_i‖)` on every window: the sphere model, with the
drive of token `i` divided by its length.

Source: none — posed here; `IsDrivenFlowOn`, for the stream behind the RMS
norm of a pre-norm block. -/
theorem isDrivenFlowOn_direction {x v : ℝ → Idx n → EucSpace d} (h0 : ∀ t i, x t i ≠ 0)
    (hx : ∀ t i, HasDerivAt (fun s => x s i) (v t i) t) (t₀ t₁ : ℝ) :
    IsDrivenFlowOn (fun t i => ⟨‖x t i‖⁻¹ • x t i,
        mem_sphere_zero_iff_norm.2 (norm_smul_inv_norm (h0 t i))⟩)
      (fun t i => ‖x t i‖⁻¹ • v t i) t₀ t₁ :=
  ⟨fun i s _ => (hasDerivAt_normalize (hx s i) (h0 s i)).continuousAt.continuousWithinAt,
    fun t _ i => hasDerivAt_normalize (hx t i) (h0 t i)⟩

/-- The hypotheses of `isDrivenFlowOn_direction` are satisfiable: one token at
rest. -/
example : (∀ (t : ℝ) (i : Idx 1), (fun _ _ => (basePoint 0 : EucSpace 1)) t i ≠ 0) ∧
    ∀ (t : ℝ) (i : Idx 1),
      HasDerivAt (fun s => (fun _ _ => (basePoint 0 : EucSpace 1)) s i) 0 t :=
  ⟨fun _ _ => by simp [basePoint], fun t _ => hasDerivAt_const t _⟩

end Perspective
end Transformer
