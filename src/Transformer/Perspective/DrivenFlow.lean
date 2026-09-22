/-
# Flows with an arbitrary drive, on a window of time

**Not a statement of any paper.**  The flow

  `ẋ_i = Proj_{x_i}( v_i(t) )`   on a window `t₀ < t < t₁`   (`IsDrivenFlowOn`),

where the drive `v_i(t)` is any vector: attention with any values `V`, several
heads, a feed-forward term, an input injection — whatever a residual block adds
to token `i`.  Only the window is required to carry the flow, so a stack of
different blocks is one flow per block, each on its own window.

This file holds the estimate behind `IsDrivenFlowOn.spread`
(`Perspective.DrivenSpread`).  The injected flow of `Perspective.InjectedFlow`
needed the attention term to be killed by `Proj_{x_i}` near consensus, which
is what `V = I_d` gives.  For a general drive nothing is killed, but whatever
the tokens share cancels in a *difference* `v_k - v_l`, so the Lyapunov
function is `ψ = ⟨x_k - x_l, w₁⟩ + ⟨x_m - x_p, w₂⟩`.  Near consensus
`ψ̇ ≈ ‖Proj_x w₁‖² + ‖Proj_x w₂‖²` at one point `x`, where `w₁ ≈ v_k - v_l` and
`w₂ ≈ v_m - v_p`, and no unit `x` makes that small when `w₁`, `w₂` are
independent (`gram_le_tangent`).
-/

import Transformer.Perspective.InjectedFlow

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **A driven flow on a window:** `ẋ_i = Proj_{x_i}(v_i(t))` for
`t₀ < t < t₁`, the particles continuous on `[t₀, t₁]`.  The drive `v` is any
function of time; a drive that depends on the particles is `v t = F (X t)`.

Source: none — posed here; the right-hand side of `eq: albert`
(arXiv:2312.10794v5, §2.3), `Proj_{x_i}` of a vector, with that vector left
arbitrary and the equation asked of one window only. -/
def IsDrivenFlowOn (X : ℝ → SphereTuple d n) (v : ℝ → Idx n → EucSpace d) (t₀ t₁ : ℝ) :
    Prop :=
  (∀ i, ContinuousOn (fun s => (X s i : EucSpace d)) (Set.Icc t₀ t₁)) ∧
    ∀ t ∈ Set.Ioo t₀ t₁, ∀ i,
      HasDerivAt (fun s => (X s i : EucSpace d)) (proj d (X t i : EucSpace d) (v t i)) t

/-- **Projections at nearby points differ little:**
`‖Proj_x v - Proj_y v‖ ≤ 2 ‖x - y‖ ‖v‖` for unit `x`, `y`, since
`Proj_x v - Proj_y v = ⟨y, v⟩ (y - x) + ⟨y - x, v⟩ x`. -/
theorem norm_proj_sub_proj_le_mul {x y : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    (v : EucSpace d) : ‖proj d x v - proj d y v‖ ≤ 2 * ‖x - y‖ * ‖v‖ := by
  have hsplit : proj d x v - proj d y v =
      (inner (𝕜 := ℝ) y v) • (y - x) + (inner (𝕜 := ℝ) (y - x) v) • x := by
    simp only [proj, inner_sub_left, sub_smul, smul_sub]
    abel
  have h1 : |inner (𝕜 := ℝ) y v| ≤ ‖v‖ := by
    simpa [hy] using abs_real_inner_le_norm y v
  have h2 := abs_real_inner_le_norm (y - x) v
  rw [norm_sub_rev y x] at h2
  rw [hsplit]
  calc ‖(inner (𝕜 := ℝ) y v) • (y - x) + (inner (𝕜 := ℝ) (y - x) v) • x‖
      ≤ ‖(inner (𝕜 := ℝ) y v) • (y - x)‖ + ‖(inner (𝕜 := ℝ) (y - x) v) • x‖ :=
        norm_add_le _ _
    _ = |inner (𝕜 := ℝ) y v| * ‖x - y‖ + |inner (𝕜 := ℝ) (y - x) v| := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, hx, mul_one,
          norm_sub_rev y x]
    _ ≤ ‖v‖ * ‖x - y‖ + ‖x - y‖ * ‖v‖ :=
        add_le_add (mul_le_mul_of_nonneg_right h1 (norm_nonneg _)) h2
    _ = 2 * ‖x - y‖ * ‖v‖ := by ring

/-- The hypotheses of `norm_proj_sub_proj_le_mul` are satisfiable: `x = y` the
first standard basis vector of `ℝ¹`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp

/-- **One pair of tokens, driven apart.**  For unit `x`, `y` and any drives
`u`, `v`, the velocity difference `Proj_x u - Proj_y v` has component along `w`
at least

  `‖w‖² - ⟨x, w⟩² - (‖u - v - w‖ + 2 ‖x - y‖ ‖v‖) ‖w‖`,

the tangential energy `‖Proj_x w‖²` of `w`, less what the drives miss `w` by
and what the two base points differ by. -/
theorem le_inner_proj_sub_proj {x y : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1)
    (u v w : EucSpace d) :
    ‖w‖ ^ 2 - (inner (𝕜 := ℝ) x w) ^ 2 - (‖u - v - w‖ + 2 * ‖x - y‖ * ‖v‖) * ‖w‖ ≤
      inner (𝕜 := ℝ) (proj d x u - proj d y v) w := by
  have hsplit : proj d x u - proj d y v =
      proj d x (u - v - w) + proj d x w + (proj d x v - proj d y v) := by
    simp only [proj, inner_sub_right, sub_smul]
    abel
  have h1 : -(‖u - v - w‖ * ‖w‖) ≤ inner (𝕜 := ℝ) (proj d x (u - v - w)) w := by
    have h := (abs_real_inner_le_norm (proj d x (u - v - w)) w).trans
      (mul_le_mul_of_nonneg_right (norm_proj_le hx _) (norm_nonneg _))
    linarith [neg_abs_le (inner (𝕜 := ℝ) (proj d x (u - v - w)) w)]
  have h2 : inner (𝕜 := ℝ) (proj d x w) w = ‖w‖ ^ 2 - (inner (𝕜 := ℝ) x w) ^ 2 := by
    simp only [proj, inner_sub_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
    ring
  have h3 : -(2 * ‖x - y‖ * ‖v‖ * ‖w‖) ≤ inner (𝕜 := ℝ) (proj d x v - proj d y v) w := by
    have h := (abs_real_inner_le_norm (proj d x v - proj d y v) w).trans
      (mul_le_mul_of_nonneg_right (norm_proj_sub_proj_le_mul hx hy v) (norm_nonneg _))
    linarith [neg_abs_le (inner (𝕜 := ℝ) (proj d x v - proj d y v) w)]
  rw [hsplit, inner_add_left, inner_add_left, h2]
  linarith

/-- The hypotheses of `le_inner_proj_sub_proj` are satisfiable: `x = y` the
first standard basis vector of `ℝ¹`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp

/-- **Two pairs of tokens, driven apart.**  If the points `X_i` are pairwise
within `δ`, the drives are bounded by `M`, and the drive differences
`v_k - v_l`, `v_m - v_p` are within `L δ` of `w₁`, `w₂`, then

  `⟨Proj_{x_k} v_k - Proj_{x_l} v_l, w₁⟩ + ⟨Proj_{x_m} v_m - Proj_{x_p} v_p, w₂⟩ ≥ c - K δ`

with `c = (‖w₁‖²‖w₂‖² - ⟨w₁, w₂⟩²) / (2(‖w₁‖² + ‖w₂‖²))` and
`K = (L + 2M)(‖w₁‖ + ‖w₂‖) + 2 ‖w₂‖²`.  Along a driven flow the left side is the
derivative of `⟨x_k - x_l, w₁⟩ + ⟨x_m - x_p, w₂⟩`. -/
theorem le_inner_proj_sub_proj_add (X : SphereTuple d n) (v : Idx n → EucSpace d)
    (k l m p : Idx n) (w₁ w₂ : EucSpace d) {L M δ : ℝ} (hδ : 0 ≤ δ)
    (hX : ∀ i j, ‖(X i : EucSpace d) - X j‖ ≤ δ) (hv : ∀ i, ‖v i‖ ≤ M)
    (h₁ : ‖v k - v l - w₁‖ ≤ L * δ) (h₂ : ‖v m - v p - w₂‖ ≤ L * δ) :
    (‖w₁‖ ^ 2 * ‖w₂‖ ^ 2 - (inner (𝕜 := ℝ) w₁ w₂) ^ 2) / (2 * (‖w₁‖ ^ 2 + ‖w₂‖ ^ 2))
        - ((L + 2 * M) * (‖w₁‖ + ‖w₂‖) + 2 * ‖w₂‖ ^ 2) * δ ≤
      inner (𝕜 := ℝ) (proj d (X k) (v k) - proj d (X l) (v l)) w₁
        + inner (𝕜 := ℝ) (proj d (X m) (v m) - proj d (X p) (v p)) w₂ := by
  have hxk := norm_coe_tuple X k
  have hxm := norm_coe_tuple X m
  have e₁ : ‖v k - v l - w₁‖ + 2 * ‖(X k : EucSpace d) - X l‖ * ‖v l‖ ≤ L * δ + 2 * δ * M := by
    have := mul_le_mul (hX k l) (hv l) (norm_nonneg _) hδ
    linarith
  have e₂ : ‖v m - v p - w₂‖ + 2 * ‖(X m : EucSpace d) - X p‖ * ‖v p‖ ≤ L * δ + 2 * δ * M := by
    have := mul_le_mul (hX m p) (hv p) (norm_nonneg _) hδ
    linarith
  have f₁ := mul_le_mul_of_nonneg_right e₁ (norm_nonneg w₁)
  have f₂ := mul_le_mul_of_nonneg_right e₂ (norm_nonneg w₂)
  have g₁ := le_inner_proj_sub_proj hxk (norm_coe_tuple X l) (v k) (v l) w₁
  have g₂ := le_inner_proj_sub_proj hxm (norm_coe_tuple X p) (v m) (v p) w₂
  have h3 : (‖w₁‖ ^ 2 * ‖w₂‖ ^ 2 - (inner (𝕜 := ℝ) w₁ w₂) ^ 2) / (2 * (‖w₁‖ ^ 2 + ‖w₂‖ ^ 2)) ≤
      (‖w₁‖ ^ 2 - (inner (𝕜 := ℝ) (X k : EucSpace d) w₁) ^ 2)
        + (‖w₂‖ ^ 2 - (inner (𝕜 := ℝ) (X k : EucSpace d) w₂) ^ 2) :=
    div_le_of_le_mul₀ (by positivity)
      (by linarith [sq_inner_le_norm_sq hxk w₁, sq_inner_le_norm_sq hxk w₂])
      (by linarith [gram_le_tangent hxk w₁ w₂])
  have h4 := sq_inner_le_sq_inner_add hxk hxm (hX m k) w₂
  linarith

/-- The hypotheses of `le_inner_proj_sub_proj_add` are satisfiable: one
particle, no drive, `δ = L = M = 0` and `w₁ = w₂ = 0`. -/
example : (0 : ℝ) ≤ 0 ∧
    (∀ i j : Idx 1, ‖(((fun _ => basePoint 0 : SphereTuple 1 1) i : EucSpace 1))
      - (fun _ => basePoint 0 : SphereTuple 1 1) j‖ ≤ 0) ∧
    (∀ i : Idx 1, ‖(fun _ => (0 : EucSpace 1)) i‖ ≤ 0) ∧
    ‖(fun _ => (0 : EucSpace 1)) 0 - (fun _ => (0 : EucSpace 1)) 0 - 0‖ ≤ 0 * 0 := by
  simp

end Perspective
end Transformer
