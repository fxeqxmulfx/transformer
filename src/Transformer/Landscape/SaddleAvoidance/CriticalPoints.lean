module

public import Transformer.Landscape.SaddleAvoidance.Convergence
public import Mathlib.Analysis.Normed.Module.FiniteDimension

/-!
# Saddle avoidance — critical points

Let `∇f` be continuous and `f` continuous at `z`. If `z` is a limit point, up to
translations in a closed set leaving `f` invariant, of an orbit of gradient descent
whose steps lower `f` by at least `η κ ‖∇f‖²`, `κ > 0`, then `z` is a critical point
(`gradient_eq_zero_of_frequently_of_le_sub`): were `∇f(z) ≠ 0`, the Łojasiewicz
inequality would hold at `z`, so the orbit would converge, to a critical point
differing from `z` by a translation (`sub_mem_of_tendsto_of_frequently`). An
`L`-Lipschitz gradient with `η L < 1` gives the decrease with `κ = 1 - η L`
(`gradient_eq_zero_of_frequently`).

At a critical point where the Hessian `D∇f` is injective, the Łojasiewicz
inequality holds with exponent `1/2` (`isLojasiewiczAt_of_injective`): near
it the distance to the point is bounded by the gradient, and the gap of
values by the square of the distance.
-/

@[expose] public section

open Filter Metric
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {η κ : ℝ}

/-- **A limit point of an orbit up to translations is a critical point**, along an
orbit whose steps lower `f` by at least `η κ ‖∇f‖²`, `κ > 0`. Let `∇f` be
continuous, `0 < η`, `f` continuous at `z`, and let the closed set `Λ` leave `f`
invariant. If the orbit of `x` comes arbitrarily close to `z` up to translations in
`Λ`, then `∇f(z) = 0`. -/
theorem gradient_eq_zero_of_frequently_of_le_sub (hη : 0 < η) (hκ : 0 < κ)
    (hc : Continuous (gradient f)) {Λ : Set E} (hΛc : IsClosed Λ)
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {x z : E} (hfc : ContinuousAt f z)
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hz : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    gradient f z = 0 := by
  by_contra h
  obtain ⟨w, hw⟩ := exists_tendsto_of_isLojasiewiczAt_of_le_sub hη hκ hΛ le_rfl one_pos
    (isLojasiewiczAt_of_gradient_ne_zero hc.continuousAt h le_rfl) hfc hdec hz
  have h1 := gradient_add_eq (hΛ _ (sub_mem_of_tendsto_of_frequently hΛc hw hz)) w
  rw [add_sub_cancel, gradient_eq_zero_of_tendsto hc hη.ne' hw] at h1
  exact h h1

/-- **A limit point of an orbit up to translations is a critical point**, for an
`L`-Lipschitz gradient and `η L < 1`: `gradient_eq_zero_of_frequently_of_le_sub` with
`κ = 1 - η L`. -/
theorem gradient_eq_zero_of_frequently (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hΛc : IsClosed Λ) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {x z : E}
    (hz : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    gradient f z = 0 :=
  gradient_eq_zero_of_frequently_of_le_sub hη (by linarith) hL.continuous hΛc hΛ
    hf.continuous.continuousAt (iterate_succ_le_sub hf hL η x) hz

/-- **The Łojasiewicz inequality at a nondegenerate critical point**, with
exponent `1/2`. If `∇f(z) = 0` and the derivative `A` of the gradient at `z`
is injective, then near `z` the distance to `z` is at most `2 C ‖∇f‖` and
`|f - f z|` at most `B` times its square. The argument is that of
`Landscape.Lojasiewicz.analytic_gradient_inequality_of_nondegenerate`, with a
differentiable `f` in place of an analytic one and the antilipschitz bound
of an injective map in place of an inverse. -/
theorem isLojasiewiczAt_of_injective [FiniteDimensional ℝ E] (hf : Differentiable ℝ f) {z : E}
    {A : E →L[ℝ] E} (hA : HasFDerivAt (gradient f) A z) (hz : gradient f z = 0)
    (hinj : Function.Injective A) : IsLojasiewiczAt f z (1 / 2) := by
  obtain ⟨K, -, hK⟩ := (A : E →ₗ[ℝ] E).exists_antilipschitzWith (LinearMap.ker_eq_bot.2 hinj)
  set C : ℝ := K + 1
  have hC : 0 < C := by positivity
  have hlow : ∀ h, ‖h‖ ≤ C * ‖A h‖ := fun h =>
    (ContinuousLinearMap.bound_of_antilipschitz A hK h).trans
      (mul_le_mul_of_nonneg_right (by simp [C]) (norm_nonneg _))
  set B : ℝ := ‖A‖ + 1 / (2 * C)
  have hB : 0 < B := by positivity
  obtain ⟨r, hr, herr⟩ := Metric.eventually_nhds_iff.1
    (hA.isLittleO.def (show (0 : ℝ) < 1 / (2 * C) by positivity))
  have hcontrol : ∀ y, dist y z < r →
      ‖gradient f y‖ ≤ B * ‖y - z‖ ∧ ‖y - z‖ ≤ 2 * C * ‖gradient f y‖ := by
    intro y hy
    have h := herr hy
    rw [hz, sub_zero] at h
    have h1 := norm_sub_norm_le (gradient f y) (A (y - z))
    have h2 := norm_sub_norm_le (A (y - z)) (gradient f y)
    rw [norm_sub_rev] at h2
    have h3 := A.le_opNorm (y - z)
    have h4 : C * ‖A (y - z)‖ ≤ C * ‖gradient f y‖ + ‖y - z‖ / 2 := by
      have h5 : C * (1 / (2 * C) * ‖y - z‖) = ‖y - z‖ / 2 := by
        field_simp
      nlinarith [mul_le_mul_of_nonneg_left (show ‖A (y - z)‖ ≤
        ‖gradient f y‖ + 1 / (2 * C) * ‖y - z‖ by linarith) hC.le]
    constructor
    · simp only [B, add_mul]
      linarith
    · linarith [hlow (y - z)]
  refine ⟨1 / (Real.sqrt B * (2 * C)), by positivity, 1, one_pos, ?_⟩
  filter_upwards [ball_mem_nhds z hr] with y hy _ _
  have hyz : ‖y - z‖ < r := by rwa [mem_ball, dist_eq_norm] at hy
  have hbound : ∀ w ∈ closedBall z ‖y - z‖, ‖fderiv ℝ f w‖ ≤ B * ‖y - z‖ := by
    intro w hw
    rw [mem_closedBall, dist_eq_norm] at hw
    rw [(hf w).hasGradientAt.hasFDerivAt.fderiv, (InnerProductSpace.toDual ℝ E).norm_map]
    exact (hcontrol w (by rw [dist_eq_norm]; linarith)).1.trans
      (mul_le_mul_of_nonneg_left hw hB.le)
  have hgap := (convex_closedBall z ‖y - z‖).norm_image_sub_le_of_norm_fderiv_le
    (fun w _ => hf w) hbound (mem_closedBall_self (norm_nonneg _))
    (by rw [mem_closedBall, dist_eq_norm])
  rw [Real.norm_eq_abs] at hgap
  have hsq : |f y - f z| ^ (1 / 2 : ℝ) ≤ Real.sqrt B * ‖y - z‖ := by
    rw [← Real.sqrt_eq_rpow, ← Real.sqrt_sq (norm_nonneg (y - z)), ← Real.sqrt_mul hB.le]
    exact Real.sqrt_le_sqrt (by nlinarith)
  have h2 := (hcontrol y hy).2
  calc 1 / (Real.sqrt B * (2 * C)) * |f y - f z| ^ (1 / 2 : ℝ)
      ≤ 1 / (Real.sqrt B * (2 * C)) * (Real.sqrt B * (2 * C * ‖gradient f y‖)) := by
        gcongr
        exact hsq.trans (mul_le_mul_of_nonneg_left h2 (Real.sqrt_nonneg B))
    _ = ‖gradient f y‖ := by
        field_simp

/-- `‖x‖² / 2` on the line, `L = 1`, `η = 1/2`, the closed set `{0}` of translations,
and the orbit `2⁻ᵏ` of `1`, which converges to the critical point `0`, where the
Hessian is the identity. -/
example := gradient_eq_zero_of_frequently (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1)
  ((contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero)
  (by rw [gradient_half_mul_norm_sq]; exact LipschitzWith.id) (η := 1 / 2) (by norm_num)
  (by norm_num) isClosed_singleton (Λ := {0}) (by simp) (x := 1) (z := 0) fun ε hε =>
    ((tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2) (by norm_num)
      1).eventually (ball_mem_nhds 0 hε)).frequently.mono fun _ hk =>
        ⟨0, Set.mem_singleton 0, by simpa using hk⟩

example := isLojasiewiczAt_of_injective (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  ((contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero)
  (A := ContinuousLinearMap.id ℝ ℝ) (z := 0)
  (by rw [gradient_half_mul_norm_sq]; exact hasFDerivAt_id 0)
  (by rw [gradient_half_mul_norm_sq]; rfl) Function.injective_id

/-- With `η = 3/2`, beyond `η L < 1` for every Lipschitz constant `L ≥ 1` of the
gradient, the orbit `(-1/2)ᵏ` of `1` lowers `‖x‖² / 2` by `η κ ‖∇f‖²` at every step,
`κ = 1 - η / 2 = 1/4`, and comes close to the critical point `0`. -/
example := gradient_eq_zero_of_frequently_of_le_sub (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (η := 3 / 2) (κ := 1 - 3 / 2 / 2) (by norm_num) (by norm_num)
  (by rw [gradient_half_mul_norm_sq]; exact continuous_id) isClosed_singleton (Λ := {0})
  (by simp) (x := 1) (z := 0) (by fun_prop)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  fun ε hε => ((tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 3 / 2)
    (by norm_num) 1).eventually (ball_mem_nhds 0 hε)).frequently.mono fun _ hk =>
      ⟨0, Set.mem_singleton 0, by simpa using hk⟩

end Transformer.Landscape.SaddleAvoidance
