/-
# The regularizer

arXiv:1211.5063, §3.3.  The regularizer of eq. (`reg_term`) is `Ω = Σ_k Ω_k`,
`Ω_k = (‖∂L/∂x_{k+1} ∂x_{k+1}/∂x_k‖ / ‖∂L/∂x_{k+1}‖ - 1)²` (`omegaTerm`,
`omegaSum`), the error `∂L/∂x_{k+1}` a covector.  `Ω_k` vanishes exactly when
the step preserves the norm of a nonzero error (`omegaTerm_eq_zero_iff`), and
`Ω` when every `Ω_k` does (`omegaSum_eq_zero_iff`); `Section3_ErrorSignal`
draws the consequences for the error signal.

With the states `x_k` and the errors held fixed, `Ω` is, as a function of
`W_rec` for eq. (2), `Σ_k (‖∂L/∂x_{k+1} W_rec diag(σ'(x_k))‖ / ‖∂L/∂x_{k+1}‖ - 1)²`
(`omegaSum_step`), where eq. (`dir_deriv`) writes `W_recᵀ` for `W_rec`, the
transpose of eq. (5) again.  "Our regularization term only forces the Jacobian
matrices `∂x_{k+1}/∂x_k` to preserve norm in the relevant direction of the
error `∂L/∂x_{k+1}`, not for any direction (i.e. we do not enforce that all
eigenvalues are close to 1)": `Ω_k = 0` for `J = diag(1, 0)` and the error
`(1, 0)`, while `J` sends `(0, 1)` to `0` (`not_forall_norm_eq_of_omegaTerm_eq_zero`).
-/

import Transformer.RecurrentGradients.Section1_Gradients

open scoped Matrix

namespace Transformer.RecurrentGradients

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- `Ω_k` of eq. (`reg_term`): `(‖δ J‖ / ‖δ‖ - 1)²` for the error `δ = ∂L/∂x_{k+1}`
and the Jacobian `J = ∂x_{k+1}/∂x_k`. -/
noncomputable def omegaTerm (δ : E →L[ℝ] ℝ) (J : E →L[ℝ] E) : ℝ :=
  (‖δ ∘L J‖ / ‖δ‖ - 1) ^ 2

/-- `Ω = Σ_k Ω_k` of eq. (`reg_term`), over `0 ≤ k < T`, for the errors
`δ k = ∂L/∂x_k` and the Jacobians `A k = ∂x_{k+1}/∂x_k`. -/
noncomputable def omegaSum (δ : ℕ → E →L[ℝ] ℝ) (A : ℕ → E →L[ℝ] E) (T : ℕ) : ℝ :=
  ∑ k ∈ Finset.range T, omegaTerm (δ (k + 1)) (A k)

theorem omegaTerm_nonneg (δ : E →L[ℝ] ℝ) (J : E →L[ℝ] E) : 0 ≤ omegaTerm δ J :=
  sq_nonneg _

/-- `Ω_k = 0` exactly when the step preserves the norm of the nonzero error. -/
theorem omegaTerm_eq_zero_iff {δ : E →L[ℝ] ℝ} (hδ : δ ≠ 0) (J : E →L[ℝ] E) :
    omegaTerm δ J = 0 ↔ ‖δ ∘L J‖ = ‖δ‖ := by
  rw [omegaTerm, pow_eq_zero_iff two_ne_zero, sub_eq_zero,
    div_eq_one_iff_eq (norm_ne_zero_iff.2 hδ)]

theorem id_ne_zero : ContinuousLinearMap.id ℝ ℝ ≠ 0 := fun h =>
  one_ne_zero (congrArg (· 1) h : (1 : ℝ) = 0)

/-- The hypothesis of `omegaTerm_eq_zero_iff` is satisfiable: the identity of `ℝ`. -/
example (J : ℝ →L[ℝ] ℝ) :
    omegaTerm (ContinuousLinearMap.id ℝ ℝ) J = 0 ↔
      ‖ContinuousLinearMap.id ℝ ℝ ∘L J‖ = ‖ContinuousLinearMap.id ℝ ℝ‖ :=
  omegaTerm_eq_zero_iff id_ne_zero J

/-- `Ω = 0` exactly when every `Ω_k = 0`. -/
theorem omegaSum_eq_zero_iff (δ : ℕ → E →L[ℝ] ℝ) (A : ℕ → E →L[ℝ] E) (T : ℕ) :
    omegaSum δ A T = 0 ↔ ∀ k < T, omegaTerm (δ (k + 1)) (A k) = 0 := by
  rw [omegaSum, Finset.sum_eq_zero_iff_of_nonneg fun k _ => omegaTerm_nonneg _ _]
  simp only [Finset.mem_range]

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] {σ σ' : ℝ → ℝ}
  {Win : Matrix ι κ ℝ} {b : EuclideanSpace ℝ ι}

/-- **Eq. (`dir_deriv`), corrected.**  For eq. (2), with the states `x_k` and the
errors `∂L/∂x_{k+1}` held fixed, `Ω` is the function
`W_rec ↦ Σ_k (‖∂L/∂x_{k+1} W_rec diag(σ'(x_k))‖ / ‖∂L/∂x_{k+1}‖ - 1)²`, whose
"immediate" derivative the paper takes; the paper writes `W_recᵀ` for `W_rec`. -/
theorem omegaSum_step (hσ : ∀ y, HasDerivAt σ (σ' y) y) (δ : ℕ → EuclideanSpace ℝ ι →L[ℝ] ℝ)
    (x : ℕ → EuclideanSpace ℝ ι) (u : ℕ → κ → ℝ) (T : ℕ) :
    (fun W => omegaSum δ (fun k => fderiv ℝ (fun y => step σ W Win b y (u (k + 1))) (x k)) T) =
      fun W : Matrix ι ι ℝ => ∑ k ∈ Finset.range T, (‖δ (k + 1) ∘L Matrix.toEuclideanCLM
        (𝕜 := ℝ) (W * Matrix.diagonal fun i => σ' (x k i))‖ / ‖δ (k + 1)‖ - 1) ^ 2 :=
  funext fun _ => Finset.sum_congr rfl fun _ _ => by
    dsimp only
    rw [(hasFDerivAt_step fun _ => hσ _).fderiv, omegaTerm]

/-- The hypothesis of `omegaSum_step` is satisfiable: the identity. -/
example (δ : ℕ → EuclideanSpace ℝ ι →L[ℝ] ℝ) (x : ℕ → EuclideanSpace ℝ ι) (u : ℕ → κ → ℝ)
    (T : ℕ) :
    (fun W => omegaSum δ (fun k => fderiv ℝ (fun y => step id W Win b y (u (k + 1))) (x k)) T) =
      fun W : Matrix ι ι ℝ => ∑ k ∈ Finset.range T, (‖δ (k + 1) ∘L Matrix.toEuclideanCLM
        (𝕜 := ℝ) (W * Matrix.diagonal fun _ : ι => (1 : ℝ))‖ / ‖δ (k + 1)‖ - 1) ^ 2 :=
  omegaSum_step (σ' := fun _ => 1) hasDerivAt_id δ x u T

/-- **§3.3: the regularizer preserves norm "in the relevant direction of the
error, not for any direction"**: `Ω_k = 0` does not make the Jacobian preserve
every norm.  With the error `(1, 0)` and `J = diag(1, 0)`, `Ω_k = 0` while `J`
sends `(0, 1)` to `0`. -/
theorem not_forall_norm_eq_of_omegaTerm_eq_zero :
    ¬ ∀ (δ : EuclideanSpace ℝ (Fin 2) →L[ℝ] ℝ) (J : Matrix (Fin 2) (Fin 2) ℝ), δ ≠ 0 →
      omegaTerm δ (Matrix.toEuclideanCLM (𝕜 := ℝ) J) = 0 →
        ∀ v, ‖Matrix.toEuclideanCLM (𝕜 := ℝ) J v‖ = ‖v‖ := by
  intro h
  have hδ : EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2) ≠ 0 := fun h0 => by
    simpa using congrArg (· (WithLp.toLp 2 ![1, 0])) h0
  have hJ : EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2) ∘L
      Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal ![1, 0]) = EuclideanSpace.proj 0 := by
    ext v
    simp [Matrix.mulVec_diagonal]
  have h0 : Matrix.diagonal ![(1 : ℝ), 0] *ᵥ ![0, 1] = 0 := by
    ext i
    fin_cases i <;> simp [Matrix.mulVec_diagonal]
  have := h _ (Matrix.diagonal ![1, 0]) hδ ((omegaTerm_eq_zero_iff hδ _).2 (by rw [hJ]))
    (WithLp.toLp 2 ![0, 1])
  rw [Matrix.toEuclideanCLM_toLp, h0, WithLp.toLp_zero, norm_zero] at this
  have h1 := congrArg (fun w : EuclideanSpace ℝ (Fin 2) => w 1) (norm_eq_zero.1 this.symm)
  simp at h1

end Transformer.RecurrentGradients
