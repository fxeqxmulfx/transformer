/-
# Saddle avoidance — critical points

If a sequence converges to `w` and comes arbitrarily close to `z` up to
translations in a closed set `Λ`, then `z - w ∈ Λ`
(`sub_mem_of_tendsto_of_frequently`).  A bound on a function along the
sequence holds at `z` if the translations leave the function invariant and it
is continuous at `z` (`le_of_frequently_norm_add_sub_lt`).  A limit point `z` of an orbit of
gradient descent up to translations leaving `f` invariant is a critical
point (`gradient_eq_zero_of_frequently`): were `∇f(z) ≠ 0`, the Łojasiewicz
inequality would hold at `z`, so the orbit would converge, to a critical
point differing from `z` by a translation.

At a critical point where the Hessian `D∇f` is injective, the Łojasiewicz
inequality holds with exponent `1/2` (`isLojasiewiczAt_of_injective`): near
it the distance to the point is bounded by the gradient, and the gap of
values by the square of the distance.
-/

import Transformer.SaddleAvoidance.Convergence
import Mathlib.Analysis.Normed.Module.FiniteDimension

open Filter Metric
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

section Translate

variable {E : Type*} [NormedAddCommGroup E]

/-- **A limit point up to translations differs from the limit by a
translation**: if `u` converges to `w` and comes arbitrarily close to `z` up
to translations in a closed set `Λ`, then `z - w ∈ Λ`. -/
theorem sub_mem_of_tendsto_of_frequently {Λ : Set E} (hΛ : IsClosed Λ) {u : ℕ → E}
    {w z : E} (hw : Tendsto u atTop (𝓝 w))
    (hz : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖u k + v - z‖ < ε) : z - w ∈ Λ := by
  rw [← hΛ.closure_eq, Metric.mem_closure_iff]
  intro ε hε
  obtain ⟨k, ⟨v, hv, hk⟩, hk'⟩ := ((hz (ε / 2) (half_pos hε)).and_eventually
    (Metric.tendsto_nhds.1 hw (ε / 2) (half_pos hε))).exists
  refine ⟨v, hv, ?_⟩
  rw [dist_eq_norm] at hk' ⊢
  calc ‖z - w - v‖ = ‖(u k - w) - (u k + v - z)‖ := by
        congr 1
        abel
    _ ≤ ‖u k - w‖ + ‖u k + v - z‖ := norm_sub_le _ _
    _ < ε / 2 + ε / 2 := add_lt_add hk' hk
    _ = ε := add_halves ε

/-- **A bound on `f` along a sequence holds at its limit points up to
translations** leaving `f` invariant, where `f` is continuous: if
`f (u i) ≤ b` eventually along `l` and, for every `ε > 0`, frequently some
`u i + v` with `v ∈ Λ` lies within `ε` of `z`, then `f z ≤ b`. -/
theorem le_of_frequently_norm_add_sub_lt {f : E → ℝ} {ι : Type*} {l : Filter ι} {u : ι → E}
    {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {z : E} (hc : ContinuousAt f z)
    (hz : ∀ ε > 0, ∃ᶠ i in l, ∃ v ∈ Λ, ‖u i + v - z‖ < ε) {b : ℝ}
    (hb : ∀ᶠ i in l, f (u i) ≤ b) : f z ≤ b := by
  refine le_of_forall_pos_lt_add fun δ hδ => ?_
  obtain ⟨ε, hε, hball⟩ := Metric.continuousAt_iff.1 hc δ hδ
  obtain ⟨i, ⟨v, hv, hi⟩, hbi⟩ := ((hz ε hε).and_eventually hb).exists
  have h := hball (show dist (u i + v) z < ε by rwa [dist_eq_norm])
  rw [Real.dist_eq, hΛ v hv] at h
  linarith [(abs_lt.1 h).1]

end Translate

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {η : ℝ}

/-- **A limit point of an orbit up to translations is a critical point.**
Let `f` have an `L`-Lipschitz gradient, `0 < η`, `η L < 1`, and let the
closed set `Λ` leave `f` invariant.  If the orbit of `x` comes arbitrarily
close to `z` up to translations in `Λ`, then `∇f(z) = 0`. -/
theorem gradient_eq_zero_of_frequently (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hΛc : IsClosed Λ) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {x z : E}
    (hz : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    gradient f z = 0 := by
  by_contra h
  obtain ⟨w, hw⟩ := exists_tendsto_of_isLojasiewiczAt hf hL hη hηL hΛ le_rfl one_pos
    (isLojasiewiczAt_of_gradient_ne_zero hL.continuous.continuousAt h le_rfl) hz
  have h1 := gradient_add_eq (hΛ _ (sub_mem_of_tendsto_of_frequently hΛc hw hz)) w
  rw [add_sub_cancel, gradient_eq_zero_of_tendsto hL.continuous hη.ne' hw] at h1
  exact h h1

/-- **The Łojasiewicz inequality at a nondegenerate critical point**, with
exponent `1/2`.  If `∇f(z) = 0` and the derivative `A` of the gradient at `z`
is injective, then near `z` the distance to `z` is at most `2 C ‖∇f‖` and
`|f - f z|` at most `B` times its square.  The argument is that of
`Normalization.analytic_gradient_inequality_of_nondegenerate`, with a
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

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line, `L = 1`, `η = 1/2`, the closed set `{0}` of translations, and the orbit
`2⁻ᵏ` of `1`, which converges to the critical point `0`, where the Hessian is
the identity; `f` is continuous, and along the orbit at most `f 1`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    let u := fun k => (gradientStep f (1 / 2))^[k] 1
    IsClosed ({0} : Set ℝ) ∧ Tendsto u atTop (𝓝 0) ∧
      (∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ ({0} : Set ℝ), ‖u k + v - 0‖ < ε) ∧
      Differentiable ℝ f ∧ LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧
      (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧ (∀ v ∈ ({0} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      HasFDerivAt (gradient f) (ContinuousLinearMap.id ℝ ℝ) 0 ∧ gradient f 0 = 0 ∧
      Function.Injective (ContinuousLinearMap.id ℝ ℝ) ∧ ContinuousAt f 0 ∧
      ∀ᶠ k in atTop, f (u k) ≤ f 1 := by
  intro f u
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hu : Tendsto u atTop (𝓝 0) := by
    simp only [u, f, iterate_gradientStep_half_mul_norm_sq, smul_eq_mul, mul_one]
    exact tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  have hd : Differentiable ℝ f :=
    (contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero
  have hL : LipschitzWith 1 (gradient f) := by rw [hg]; exact LipschitzWith.id
  refine ⟨isClosed_singleton, hu, fun ε hε => ?_, hd, hL, by norm_num, by norm_num, by simp,
    by rw [hg]; exact hasFDerivAt_id 0, by rw [hg]; rfl, Function.injective_id,
    hd.continuous.continuousAt, Eventually.of_forall fun k =>
      antitone_iterate hd hL (by norm_num) (by norm_num) 1 (Nat.zero_le k)⟩
  exact (Metric.tendsto_nhds.1 hu ε hε).frequently.mono fun k hk =>
    ⟨0, Set.mem_singleton 0, by rwa [add_zero, ← dist_eq_norm]⟩

end Transformer.SaddleAvoidance
