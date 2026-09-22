/-
# The feed-forward term of parameter-golf is bounded and Lipschitz on the sphere

**Not a statement of any paper.**  The feed-forward term of a block of
parameter-golf (github.com/openai/parameter-golf,
`records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`,
`MLP.forward`) is

  `G(x) = W₂ σ_α(W₁ x)²`, the square taken entry by entry,   (`sqMLP`)

with the leaky ReLU `σ_α(a) = a` for `a > 0` and `α a` otherwise, at `α = 1/2`
(`F.leaky_relu(·, negative_slope=0.5).square()`; the fused kernel
`linear_leaky_relu_square_kernel` computes the same), and no bias.  The block
multiplies the output by `mlp_scale` and the input, the RMS norm of the stream,
by `ln_scale_factor`; on the sphere the RMS norm is `√d x`, so all three
factors are linear and go into `W₁` and `W₂`.

On the unit ball, `‖G(x)‖ ≤ ‖W₂‖ ‖W₁‖²` (`norm_sqMLP_le`) and `G` is
`2 ‖W₂‖ ‖W₁‖²`-Lipschitz (`norm_sqMLP_sub_le`).  With the residual weight
beside it as the linear term `D = mix[0] - 1`, the feed-forward part of the
block is bounded by `‖D‖ + ‖W₂‖ ‖W₁‖²` and `(‖D‖ + 2 ‖W₂‖ ‖W₁‖²)`-Lipschitz on
the sphere (`norm_add_sqMLP_le`, `norm_add_sqMLP_sub_le`): the bounds that
`IsBoundedBlockOn` asks of a feed-forward term, with constants read off the
weights.  `Perspective.BlockMLPSpread` draws the consequence.

What stays out: in the block the feed-forward term reads the stream after
attention, here the token `x_i`; the tokens live on the sphere.
-/

import Transformer.Perspective.BlockSpread

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d m : ℕ}

/-- **Leaky ReLU:** `σ_α(a) = a` for `a > 0`, `α a` otherwise.

Source: `F.leaky_relu` with `negative_slope = α`, as `MLP.forward` of
parameter-golf calls it, at `α = 1/2`. -/
noncomputable def leakyRelu (α a : ℝ) : ℝ := if 0 < a then a else α * a

/-- **Leaky ReLU does not lengthen:** `|σ_α(a)| ≤ |a|` for `|α| ≤ 1`. -/
theorem abs_leakyRelu_le {α : ℝ} (hα : |α| ≤ 1) (a : ℝ) : |leakyRelu α a| ≤ |a| := by
  unfold leakyRelu
  split_ifs
  · exact le_rfl
  · rw [abs_mul]
    exact mul_le_of_le_one_left (abs_nonneg a) hα

/-- **Leaky ReLU is `1`-Lipschitz:** `|σ_α(a) - σ_α(b)| ≤ |a - b|` for `|α| ≤ 1`. -/
theorem abs_leakyRelu_sub_le {α : ℝ} (hα : |α| ≤ 1) (a b : ℝ) :
    |leakyRelu α a - leakyRelu α b| ≤ |a - b| := by
  obtain ⟨h1, h2⟩ := abs_le.mp hα
  unfold leakyRelu
  split_ifs with ha hb hb
  · exact le_rfl
  · push Not at hb
    rw [abs_of_pos (by linarith : (0 : ℝ) < a - b), abs_le]
    constructor <;> nlinarith
  · push Not at ha
    rw [abs_of_neg (by linarith : a - b < 0), abs_le]
    constructor <;> nlinarith
  · rw [← mul_sub, abs_mul]
    exact mul_le_of_le_one_left (abs_nonneg _) hα

/-- **The activation of the feed-forward term:** `σ_α(u)²`, entry by entry.

Source: `F.leaky_relu(·, negative_slope=0.5).square()` in `MLP.forward` of
parameter-golf, at `α = 1/2`. -/
noncomputable def sqAct (α : ℝ) (u : EucSpace m) : EucSpace m :=
  WithLp.toLp 2 fun k => leakyRelu α (u k) ^ 2

/-- **The feed-forward term of parameter-golf:** `G(x) = W₂ σ_α(W₁ x)²`.

Source: `MLP.forward` of parameter-golf,
`F.linear(F.leaky_relu(F.linear(x, up_w), negative_slope=0.5).square(), down_w)`,
with `W₁ = up_w`, `W₂ = down_w` and `α = 1/2`. -/
noncomputable def sqMLP (α : ℝ) (W₁ : EucSpace d →L[ℝ] EucSpace m)
    (W₂ : EucSpace m →L[ℝ] EucSpace d) (x : EucSpace d) : EucSpace d :=
  W₂ (sqAct α (W₁ x))

/-- **The activation is at most quadratic:** `‖σ_α(u)²‖ ≤ ‖u‖²` for `|α| ≤ 1`. -/
theorem norm_sqAct_le {α : ℝ} (hα : |α| ≤ 1) (u : EucSpace m) : ‖sqAct α u‖ ≤ ‖u‖ ^ 2 := by
  have h : ‖sqAct α u‖ ^ 2 ≤ (‖u‖ ^ 2) ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
    calc ∑ k, (sqAct α u k) ^ 2 = ∑ k, (leakyRelu α (u k) ^ 2) ^ 2 := by simp [sqAct]
      _ ≤ ∑ k, (u k ^ 2) ^ 2 := Finset.sum_le_sum fun k _ =>
          pow_le_pow_left₀ (sq_nonneg _) (sq_le_sq.mpr (abs_leakyRelu_le hα _)) 2
      _ ≤ (∑ k, u k ^ 2) ^ 2 := Finset.sum_sq_le_sq_sum_of_nonneg fun k _ => sq_nonneg _
  exact (sq_le_sq₀ (norm_nonneg _) (sq_nonneg _)).mp h

/-- **The activation is Lipschitz on bounded sets:**
`‖σ_α(u)² - σ_α(v)²‖ ≤ (‖u‖ + ‖v‖) ‖u - v‖` for `|α| ≤ 1`. -/
theorem norm_sqAct_sub_le {α : ℝ} (hα : |α| ≤ 1) (u v : EucSpace m) :
    ‖sqAct α u - sqAct α v‖ ≤ (‖u‖ + ‖v‖) * ‖u - v‖ := by
  have hk : ∀ k, |leakyRelu α (u k) ^ 2 - leakyRelu α (v k) ^ 2| ≤
      (‖u‖ + ‖v‖) * |u k - v k| := fun k => by
    have hu := PiLp.norm_apply_le u k
    have hv := PiLp.norm_apply_le v k
    rw [Real.norm_eq_abs] at hu hv
    rw [sq_sub_sq, abs_mul]
    refine mul_le_mul ?_ (abs_leakyRelu_sub_le hα _ _) (abs_nonneg _) (by positivity)
    exact (abs_add_le _ _).trans
      (add_le_add ((abs_leakyRelu_le hα _).trans hu) ((abs_leakyRelu_le hα _).trans hv))
  have h : ‖sqAct α u - sqAct α v‖ ^ 2 ≤ ((‖u‖ + ‖v‖) * ‖u - v‖) ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, mul_pow, EuclideanSpace.real_norm_sq_eq, Finset.mul_sum]
    refine Finset.sum_le_sum fun k _ => ?_
    have := pow_le_pow_left₀ (abs_nonneg _) (hk k) 2
    rw [sq_abs, mul_pow, sq_abs] at this
    simpa [sqAct] using this
  exact (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp h

/-- **The feed-forward term is bounded on the unit ball:**
`‖W₂ σ_α(W₁ x)²‖ ≤ ‖W₂‖ ‖W₁‖²` for `‖x‖ ≤ 1` and `|α| ≤ 1`. -/
theorem norm_sqMLP_le {α : ℝ} (hα : |α| ≤ 1) (W₁ : EucSpace d →L[ℝ] EucSpace m)
    (W₂ : EucSpace m →L[ℝ] EucSpace d) {x : EucSpace d} (hx : ‖x‖ ≤ 1) :
    ‖sqMLP α W₁ W₂ x‖ ≤ ‖W₂‖ * ‖W₁‖ ^ 2 := by
  have h1 : ‖W₁ x‖ ≤ ‖W₁‖ := (W₁.le_opNorm x).trans (mul_le_of_le_one_right (norm_nonneg _) hx)
  exact (W₂.le_opNorm _).trans (mul_le_mul_of_nonneg_left
    ((norm_sqAct_le hα _).trans (pow_le_pow_left₀ (norm_nonneg _) h1 2)) (norm_nonneg _))

/-- **The feed-forward term is Lipschitz on the unit ball:**
`‖G(x) - G(y)‖ ≤ 2 ‖W₂‖ ‖W₁‖² ‖x - y‖` for `‖x‖, ‖y‖ ≤ 1` and `|α| ≤ 1`. -/
theorem norm_sqMLP_sub_le {α : ℝ} (hα : |α| ≤ 1) (W₁ : EucSpace d →L[ℝ] EucSpace m)
    (W₂ : EucSpace m →L[ℝ] EucSpace d) {x y : EucSpace d} (hx : ‖x‖ ≤ 1) (hy : ‖y‖ ≤ 1) :
    ‖sqMLP α W₁ W₂ x - sqMLP α W₁ W₂ y‖ ≤ 2 * ‖W₂‖ * ‖W₁‖ ^ 2 * ‖x - y‖ := by
  have h1 : ∀ z : EucSpace d, ‖z‖ ≤ 1 → ‖W₁ z‖ ≤ ‖W₁‖ := fun z hz =>
    (W₁.le_opNorm z).trans (mul_le_of_le_one_right (norm_nonneg _) hz)
  have h2 : ‖W₁ x - W₁ y‖ ≤ ‖W₁‖ * ‖x - y‖ := by rw [← map_sub]; exact W₁.le_opNorm _
  calc ‖sqMLP α W₁ W₂ x - sqMLP α W₁ W₂ y‖
      = ‖W₂ (sqAct α (W₁ x) - sqAct α (W₁ y))‖ := by simp only [sqMLP, map_sub]
    _ ≤ ‖W₂‖ * ‖sqAct α (W₁ x) - sqAct α (W₁ y)‖ := W₂.le_opNorm _
    _ ≤ ‖W₂‖ * ((‖W₁‖ + ‖W₁‖) * (‖W₁‖ * ‖x - y‖)) := by
      refine mul_le_mul_of_nonneg_left ((norm_sqAct_sub_le hα _ _).trans ?_) (norm_nonneg _)
      exact mul_le_mul (add_le_add (h1 x hx) (h1 y hy)) h2 (norm_nonneg _) (by positivity)
    _ = 2 * ‖W₂‖ * ‖W₁‖ ^ 2 * ‖x - y‖ := by ring

/-- **Beside the residual weight, the feed-forward term is bounded on the
sphere:** `‖D x + W₂ σ_α(W₁ x)²‖ ≤ L + K₂ K₁²` for `‖D‖ ≤ L`, `‖W₁‖ ≤ K₁`,
`‖W₂‖ ≤ K₂` and `|α| ≤ 1`.  `D` stands for `mix[0] - 1`. -/
theorem norm_add_sqMLP_le {α L K₁ K₂ : ℝ} (hα : |α| ≤ 1) {D : EucSpace d →L[ℝ] EucSpace d}
    {W₁ : EucSpace d →L[ℝ] EucSpace m} {W₂ : EucSpace m →L[ℝ] EucSpace d} (hD : ‖D‖ ≤ L)
    (hW₁ : ‖W₁‖ ≤ K₁) (hW₂ : ‖W₂‖ ≤ K₂) (x : SSphere d) :
    ‖D x + sqMLP α W₁ W₂ x‖ ≤ L + K₂ * K₁ ^ 2 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hK₂ : 0 ≤ K₂ := (norm_nonneg _).trans hW₂
  refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
  · exact (D.le_opNorm _).trans (by rw [hx, mul_one]; exact hD)
  · exact (norm_sqMLP_le hα W₁ W₂ hx.le).trans
      (mul_le_mul hW₂ (pow_le_pow_left₀ (norm_nonneg _) hW₁ 2) (by positivity) hK₂)

/-- **Beside the residual weight, the feed-forward term is Lipschitz on the
sphere:** `x ↦ D x + W₂ σ_α(W₁ x)²` is `(L + 2 K₂ K₁²)`-Lipschitz there for
`‖D‖ ≤ L`, `‖W₁‖ ≤ K₁`, `‖W₂‖ ≤ K₂` and `|α| ≤ 1`. -/
theorem norm_add_sqMLP_sub_le {α L K₁ K₂ : ℝ} (hα : |α| ≤ 1) {D : EucSpace d →L[ℝ] EucSpace d}
    {W₁ : EucSpace d →L[ℝ] EucSpace m} {W₂ : EucSpace m →L[ℝ] EucSpace d} (hD : ‖D‖ ≤ L)
    (hW₁ : ‖W₁‖ ≤ K₁) (hW₂ : ‖W₂‖ ≤ K₂) (x y : SSphere d) :
    ‖D x + sqMLP α W₁ W₂ x - (D y + sqMLP α W₁ W₂ y)‖ ≤
      (L + 2 * K₂ * K₁ ^ 2) * ‖(x : EucSpace d) - y‖ := by
  have hx : ‖(x : EucSpace d)‖ ≤ 1 := (mem_sphere_zero_iff_norm.mp x.2).le
  have hy : ‖(y : EucSpace d)‖ ≤ 1 := (mem_sphere_zero_iff_norm.mp y.2).le
  have hK₂ : 0 ≤ K₂ := (norm_nonneg _).trans hW₂
  have hsplit : D x + sqMLP α W₁ W₂ x - (D y + sqMLP α W₁ W₂ y) =
      D ((x : EucSpace d) - y) + (sqMLP α W₁ W₂ x - sqMLP α W₁ W₂ y) := by
    rw [map_sub]; abel
  have h1 : ‖D ((x : EucSpace d) - y)‖ ≤ L * ‖(x : EucSpace d) - y‖ :=
    (D.le_opNorm _).trans (mul_le_mul_of_nonneg_right hD (norm_nonneg _))
  have h2 : 2 * ‖W₂‖ * ‖W₁‖ ^ 2 ≤ 2 * K₂ * K₁ ^ 2 :=
    mul_le_mul (mul_le_mul_of_nonneg_left hW₂ two_pos.le)
      (pow_le_pow_left₀ (norm_nonneg _) hW₁ 2) (by positivity) (by positivity)
  have h3 := (norm_sqMLP_sub_le hα W₁ W₂ hx hy).trans
    (mul_le_mul_of_nonneg_right h2 (norm_nonneg _))
  rw [hsplit]
  calc _ ≤ L * ‖(x : EucSpace d) - y‖ + 2 * K₂ * K₁ ^ 2 * ‖(x : EucSpace d) - y‖ :=
        (norm_add_le _ _).trans (add_le_add h1 h3)
    _ = (L + 2 * K₂ * K₁ ^ 2) * ‖(x : EucSpace d) - y‖ := by ring

/-- The hypotheses of every theorem above are satisfiable: the slope `α = 1/2`
of parameter-golf, the origin, and `D = 0`, `W₁ = W₂ = 0` with
`L = K₁ = K₂ = 0`. -/
example : |(1 / 2 : ℝ)| ≤ 1 ∧ ‖(0 : EucSpace 1)‖ ≤ 1 ∧
    ‖(0 : EucSpace 1 →L[ℝ] EucSpace 1)‖ ≤ 0 := by
  rw [abs_of_pos (by norm_num)]
  norm_num

end Perspective
end Transformer
