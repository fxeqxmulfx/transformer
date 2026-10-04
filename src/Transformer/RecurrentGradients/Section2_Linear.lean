/-
# The linear model

arXiv:1211.5063, §2.1 and the supplementary "Analytical analysis of the
exploding and vanishing gradients problem".  With `σ` the identity, eq. (2) is
linear and `∂x_{k+l}/∂x_k = W_rec^l` (`factor_id`); the supplementary's
eq. (`prod_wk`) writes `(W_recᵀ)^l`, the transpose of §1.1's eq. (5).  §2.1:
"It is sufficient for the largest eigenvalue `λ₁` of the recurrent weight matrix
to be smaller than 1 for long term components to vanish (as `t → ∞`) and
necessary for it to be larger than 1 for gradients to explode."  Both hold,
`λ₁` the spectral radius of the complexified `W_rec`, by Gelfand's formula,
which bounds `‖W_rec^l‖` by `r^l` eventually for each `r > λ₁`
(`eventually_norm_pow_le`, with `norm_le_norm_map_ofReal`): with `λ₁ < 1` the
factors go to `0` (`tendsto_norm_factor_id`), and factors of at least `C α^l`
for infinitely many `l` force `λ₁ ≥ α` (`le_spectralRadius_of_frequently`),
so `λ₁ > 1` when `α > 1` (`one_lt_spectralRadius`), the supplementary's
"necessary condition for gradients to grow".
-/

import Transformer.RecurrentGradients.Section2_Counterexample
import Mathlib.Analysis.Normed.Algebra.GelfandFormula

open scoped Matrix Matrix.Norms.L2Operator
open Filter Topology

namespace Transformer.RecurrentGradients

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] {W : Matrix ι ι ℝ}
  {Win : Matrix ι κ ℝ} {b : EuclideanSpace ℝ ι}

/-- The 2-norm of a real matrix is at most that of its complexification. -/
theorem norm_le_norm_map_ofReal (A : Matrix ι ι ℝ) : ‖A‖ ≤ ‖A.map Complex.ofReal‖ := by
  rw [← Matrix.l2_opNorm_toEuclideanCLM A]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) fun x => ?_
  have h := (Matrix.toEuclideanCLM (𝕜 := ℂ) (A.map Complex.ofReal)).le_opNorm
    (WithLp.toLp 2 fun i => (x i : ℂ))
  rw [Matrix.l2_opNorm_toEuclideanCLM, Matrix.toEuclideanCLM_toLp] at h
  convert h using 2
  · rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [Matrix.mulVec, dotProduct, Matrix.map_apply, Matrix.ofLp_toEuclideanCLM]
    rw [show ∑ j, (A i j : ℂ) * (x.ofLp j : ℂ) = ((∑ j, A i j * x.ofLp j : ℝ) : ℂ) by
      push_cast; rfl, Complex.norm_real, Real.norm_eq_abs, sq_abs]
  · simp [EuclideanSpace.norm_eq]

/-- `(W^l)` complexified is the `l`-th power of `W` complexified. -/
theorem map_ofReal_pow (A : Matrix ι ι ℝ) (l : ℕ) :
    (A ^ l).map Complex.ofReal = A.map Complex.ofReal ^ l :=
  Matrix.map_pow A Complex.ofRealHom l

/-- **Gelfand's formula as a bound**: for `r > λ₁`, eventually `‖a^l‖ ≤ r^l`. -/
theorem eventually_norm_pow_le (a : Matrix ι ι ℂ) {r : ℝ}
    (hr : spectralRadius ℂ a < ENNReal.ofReal r) : ∀ᶠ l in atTop, ‖a ^ l‖ ≤ r ^ l := by
  filter_upwards [(spectrum.pow_norm_pow_one_div_tendsto_nhds_spectralRadius a).eventually
    (gt_mem_nhds hr), eventually_ne_atTop 0] with l hl hl0
  have hr0 : 0 < r := by
    by_contra h0
    rw [ENNReal.ofReal_of_nonpos (not_lt.1 h0)] at hl
    exact ENNReal.not_lt_zero hl
  rw [ENNReal.ofReal_lt_ofReal_iff hr0] at hl
  calc ‖a ^ l‖ = (‖a ^ l‖ ^ (1 / l : ℝ)) ^ l := by
        rw [one_div, Real.rpow_inv_natCast_pow (norm_nonneg _) hl0]
    _ ≤ r ^ l := pow_le_pow_left₀ (by positivity) hl.le l

/-- The hypothesis of `eventually_norm_pow_le` is satisfiable: `a = 0`, `r = 1`. -/
example : ∀ᶠ l in atTop, ‖(0 : Matrix ι ι ℂ) ^ l‖ ≤ 1 ^ l :=
  eventually_norm_pow_le 0 (by simp)

/-- **The supplementary's eq. (`prod_wk`), corrected**: for the linear model
`∂x_{k+l}/∂x_k = W_rec^l`, where the paper writes `(W_recᵀ)^l`. -/
theorem factor_id (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    factor id W Win b u x₀ k l = Matrix.toEuclideanCLM (𝕜 := ℝ) (W ^ l) := by
  rw [factor, (hasFDerivAt_states_step (σ' := fun _ => 1) hasDerivAt_id u x₀ k l).fderiv]
  simp only [Matrix.diagonal_one, mul_one, transport_const]

theorem norm_factor_id_le (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    ‖factor id W Win b u x₀ k l‖ ≤ ‖W.map Complex.ofReal ^ l‖ := by
  rw [factor_id, Matrix.l2_opNorm_toEuclideanCLM, ← map_ofReal_pow]
  exact norm_le_norm_map_ofReal _

/-- **§2.1, the linear model: `λ₁ < 1` is sufficient for vanishing.**  The
factors `∂x_{k+l}/∂x_k = W_rec^l` go to `0` as `l → ∞`. -/
theorem tendsto_norm_factor_id (hW : spectralRadius ℂ (W.map Complex.ofReal) < 1)
    (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k : ℕ) :
    Tendsto (fun l => ‖factor id W Win b u x₀ k l‖) atTop (𝓝 0) := by
  obtain ⟨r, hr0, hρr, hr1⟩ := ENNReal.lt_iff_exists_real_btwn.1 hW
  refine squeeze_zero' (Eventually.of_forall fun _ => norm_nonneg _) ?_
    (tendsto_pow_atTop_nhds_zero_of_lt_one hr0 (ENNReal.ofReal_lt_one.1 hr1))
  filter_upwards [eventually_norm_pow_le _ hρr] with l hl
  exact (norm_factor_id_le u x₀ k l).trans hl

/-- The hypothesis of `tendsto_norm_factor_id` is satisfiable: `W_rec = 0`. -/
example (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k : ℕ) :
    Tendsto (fun l => ‖factor id (0 : Matrix ι ι ℝ) Win b u x₀ k l‖) atTop (𝓝 0) :=
  tendsto_norm_factor_id (by simp) u x₀ k

/-- **§2.1, the linear model, quantitatively**: factors of norm at least `C α^l`
for infinitely many `l` force `λ₁ ≥ α`. -/
theorem le_spectralRadius_of_frequently {C α : ℝ} (hC : 0 < C) (hα : 0 < α)
    (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k : ℕ)
    (h : ∃ᶠ l in atTop, C * α ^ l ≤ ‖factor id W Win b u x₀ k l‖) :
    ENNReal.ofReal α ≤ spectralRadius ℂ (W.map Complex.ofReal) := by
  by_contra hlt
  obtain ⟨r, hr0, hρr, hrα⟩ := ENNReal.lt_iff_exists_real_btwn.1 (not_le.1 hlt)
  have hr : r / α < 1 := (div_lt_one hα).2 ((ENNReal.ofReal_lt_ofReal_iff hα).1 hrα)
  refine Filter.not_frequently.2 ?_ h
  filter_upwards [eventually_norm_pow_le _ hρr,
    (tendsto_pow_atTop_nhds_zero_of_lt_one (by positivity) hr).eventually (gt_mem_nhds hC)]
    with l hl hl1
  refine not_le.2 ((norm_factor_id_le u x₀ k l).trans_lt (hl.trans_lt ?_))
  calc r ^ l = (r / α) ^ l * α ^ l := by rw [div_pow, div_mul_cancel₀ _ (pow_ne_zero _ hα.ne')]
    _ < C * α ^ l := mul_lt_mul_of_pos_right hl1 (pow_pos hα l)

/-- One unit with `W_rec = 2`: the factor `∂x_{k+l}/∂x_k` of the linear model
has norm `2^l`. -/
theorem norm_factor_id_two (Win : Matrix (Fin 1) κ ℝ) (b : EuclideanSpace ℝ (Fin 1))
    (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ (Fin 1)) (k l : ℕ) :
    ‖factor id (Matrix.diagonal fun _ : Fin 1 => (2 : ℝ)) Win b u x₀ k l‖ = 2 ^ l := by
  rw [factor_id, Matrix.l2_opNorm_toEuclideanCLM, Matrix.diagonal_pow,
    Matrix.l2_opNorm_diagonal, Pi.pow_def, pi_norm_const, norm_pow, Real.norm_two]

/-- The hypotheses of `le_spectralRadius_of_frequently` are satisfiable: one
unit with `W_rec = 2`, `C = 1`, `α = 2`. -/
example (Win : Matrix (Fin 1) κ ℝ) (b : EuclideanSpace ℝ (Fin 1)) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ (Fin 1)) (k : ℕ) :
    ENNReal.ofReal 2 ≤
      spectralRadius ℂ ((Matrix.diagonal fun _ : Fin 1 => (2 : ℝ)).map Complex.ofReal) :=
  le_spectralRadius_of_frequently one_pos two_pos u x₀ k
    (Frequently.of_forall fun l => by rw [norm_factor_id_two Win b, one_mul])

/-- **§2.1 and the supplementary, the linear model: `λ₁ > 1` is necessary for
exploding.**  Factors of norm at least `C α^l`, `α > 1`, for infinitely many
`l` force `λ₁ > 1`. -/
theorem one_lt_spectralRadius {C α : ℝ} (hC : 0 < C) (hα : 1 < α) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ ι) (k : ℕ)
    (h : ∃ᶠ l in atTop, C * α ^ l ≤ ‖factor id W Win b u x₀ k l‖) :
    1 < spectralRadius ℂ (W.map Complex.ofReal) :=
  (ENNReal.one_lt_ofReal.2 hα).trans_le
    (le_spectralRadius_of_frequently hC (by linarith) u x₀ k h)

/-- The hypotheses of `one_lt_spectralRadius` are satisfiable: one unit with
`W_rec = 2`, `C = 1`, `α = 2`. -/
example (Win : Matrix (Fin 1) κ ℝ) (b : EuclideanSpace ℝ (Fin 1)) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ (Fin 1)) (k : ℕ) :
    1 < spectralRadius ℂ ((Matrix.diagonal fun _ : Fin 1 => (2 : ℝ)).map Complex.ofReal) :=
  one_lt_spectralRadius one_pos one_lt_two u x₀ k
    (Frequently.of_forall fun l => by rw [norm_factor_id_two Win b, one_mul])

end Transformer.RecurrentGradients
