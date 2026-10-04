/-
# The mechanics of vanishing and exploding gradients

arXiv:1211.5063, §2.1 ("The mechanics").  The factor `∂x_t/∂x_k` of a temporal
contribution (`factor`, with `t = k + l`) is a product of `l` Jacobians
(eq. (5)).  For eq. (2) with `|σ'| ≤ γ` the paper "proves" that `λ₁ < 1/γ`,
`λ₁` the largest absolute value of an eigenvalue of `W_rec`, suffices for the
long term contributions to vanish, and, "by inverting this proof", that
`λ₁ > 1/γ` is necessary for them to explode.  Both are false
(`Section2_Counterexample`): eq. (6) bounds the 2-norm `‖W_recᵀ‖` by `1/γ`,
which `λ₁ < 1/γ` does not give.  The proof proves the statements with the
2-norm `‖W_rec‖` (`= ‖W_recᵀ‖`, `Matrix.l2_opNorm_conjTranspose`) in place of
`λ₁`:

* eq. (6): `‖∂x_{k+1}/∂x_k‖ ≤ ‖W_rec‖ γ` (`norm_jacobian_le`);
* eq. (7): `‖∂L_t/∂x_t · ∂x_t/∂x_k‖ ≤ (‖W_rec‖ γ)^{t-k} ‖∂L_t/∂x_t‖`
  (`norm_comp_transport_le`, `norm_comp_factor_le`), so that with
  `‖W_rec‖ < 1/γ` the factors go to `0` exponentially fast
  (`tendsto_norm_factor`);
* inverted: factors unbounded in `t - k` force `‖W_rec‖ > 1/γ`
  (`one_lt_of_not_bddAbove`).

`γ = 1` for tanh and `γ = 1/4` for the sigmoid, each the least bound on `|σ'|`
(`isGreatest_abs_deriv_tanh`, `isGreatest_abs_deriv_sigmoid`).
-/

import Transformer.RecurrentGradients.Section1_Gradients
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecificLimits.Basic

open scoped Matrix Matrix.Norms.L2Operator
open Filter Topology

namespace Transformer.RecurrentGradients

section Transport

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F]
  [NormedSpace ℝ F]

/-- A product of `l` factors of norm at most `η` has norm at most `η^l`. -/
theorem norm_transport_le {A : ℕ → E →L[ℝ] E} {η : ℝ} (hA : ∀ i, ‖A i‖ ≤ η) (k l : ℕ) :
    ‖transport A k l‖ ≤ η ^ l := by
  induction l with
  | zero => exact ContinuousLinearMap.norm_id_le
  | succ l ih =>
    rw [transport, pow_succ']
    exact (norm_mul_le _ _).trans
      (mul_le_mul (hA _) ih (norm_nonneg _) ((norm_nonneg _).trans (hA 0)))

/-- The hypothesis of `norm_transport_le` is satisfiable: the zero factors. -/
example (k l : ℕ) : ‖transport (fun _ => (0 : E →L[ℝ] E)) k l‖ ≤ 0 ^ l :=
  norm_transport_le (fun _ => norm_zero.le) k l

/-- **Eq. (7)**: if `‖∂x_{i+1}/∂x_i‖ ≤ η` for every `i`, then
`‖∂L_t/∂x_t ∏_{i=k}^{t-1} ∂x_{i+1}/∂x_i‖ ≤ η^{t-k} ‖∂L_t/∂x_t‖`. -/
theorem norm_comp_transport_le {A : ℕ → E →L[ℝ] E} {η : ℝ} (hA : ∀ i, ‖A i‖ ≤ η)
    (v : E →L[ℝ] F) (k l : ℕ) : ‖v ∘L transport A k l‖ ≤ η ^ l * ‖v‖ := by
  rw [mul_comm]
  exact (v.opNorm_comp_le _).trans
    (mul_le_mul_of_nonneg_left (norm_transport_le hA k l) (norm_nonneg v))

/-- The hypothesis of `norm_comp_transport_le` is satisfiable: the zero factors. -/
example (v : E →L[ℝ] F) (k l : ℕ) :
    ‖v ∘L transport (fun _ => (0 : E →L[ℝ] E)) k l‖ ≤ 0 ^ l * ‖v‖ :=
  norm_comp_transport_le (fun _ => norm_zero.le) v k l

end Transport

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] {σ σ' : ℝ → ℝ} {γ : ℝ}
  {W : Matrix ι ι ℝ} {Win : Matrix ι κ ℝ} {b : EuclideanSpace ℝ ι}

/-- The factor `∂x_{k+l}/∂x_k` of eq. (2): the derivative of the state `l` steps
after `x_k` in `x_k`. -/
noncomputable def factor (σ : ℝ → ℝ) (W : Matrix ι ι ℝ) (Win : Matrix ι κ ℝ)
    (b : EuclideanSpace ℝ ι) (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    EuclideanSpace ℝ ι →L[ℝ] EuclideanSpace ℝ ι :=
  fderiv ℝ (fun x => states (step σ W Win b) (fun i => u (k + i)) x l)
    (states (step σ W Win b) u x₀ k)

/-- **Eq. (6)**: `‖∂x_{k+1}/∂x_k‖ ≤ ‖W_rec‖ ‖diag(σ'(x_k))‖ ≤ ‖W_rec‖ γ` in the
2-norm, where `|σ'| ≤ γ`. -/
theorem norm_jacobian_le (hγ : ∀ y, |σ' y| ≤ γ) (x : EuclideanSpace ℝ ι) :
    ‖W * Matrix.diagonal (fun i => σ' (x i))‖ ≤ ‖W‖ * γ := by
  refine (Matrix.l2_opNorm_mul _ _).trans (mul_le_mul_of_nonneg_left ?_ (norm_nonneg W))
  rw [Matrix.l2_opNorm_diagonal]
  exact (pi_norm_le_iff_of_nonneg ((abs_nonneg _).trans (hγ 0))).2
    fun i => (Real.norm_eq_abs _).trans_le (hγ _)

/-- The hypothesis of `norm_jacobian_le` is satisfiable: `σ' = 1`, `γ = 1`. -/
example (x : EuclideanSpace ℝ ι) : ‖W * Matrix.diagonal (fun _ : ι => (1 : ℝ))‖ ≤ ‖W‖ * 1 :=
  norm_jacobian_le (σ' := fun _ => 1) (fun _ => by simp) x

/-- The factors of eq. (2) shrink by `‖W_rec‖ γ` a step:
`‖∂x_{k+l}/∂x_k‖ ≤ (‖W_rec‖ γ)^l`. -/
theorem norm_factor_le (hσ : ∀ y, HasDerivAt σ (σ' y) y) (hγ : ∀ y, |σ' y| ≤ γ)
    (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    ‖factor σ W Win b u x₀ k l‖ ≤ (‖W‖ * γ) ^ l := by
  rw [factor, (hasFDerivAt_states_step hσ u x₀ k l).fderiv, map_transport]
  refine norm_transport_le (fun i => ?_) k l
  rw [Matrix.l2_opNorm_toEuclideanCLM]
  exact norm_jacobian_le hγ _

/-- The hypotheses of `norm_factor_le` are satisfiable: the linear network. -/
example (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    ‖factor id W Win b u x₀ k l‖ ≤ (‖W‖ * 1) ^ l :=
  norm_factor_le (σ' := fun _ => 1) hasDerivAt_id (fun _ => by simp) u x₀ k l

/-- **Eq. (7) for eq. (2)**:
`‖∂L_t/∂x_t · ∂x_t/∂x_k‖ ≤ (‖W_rec‖ γ)^{t-k} ‖∂L_t/∂x_t‖`. -/
theorem norm_comp_factor_le (hσ : ∀ y, HasDerivAt σ (σ' y) y) (hγ : ∀ y, |σ' y| ≤ γ)
    (v : EuclideanSpace ℝ ι →L[ℝ] ℝ) (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    ‖v ∘L factor σ W Win b u x₀ k l‖ ≤ (‖W‖ * γ) ^ l * ‖v‖ := by
  rw [mul_comm]
  exact (v.opNorm_comp_le _).trans
    (mul_le_mul_of_nonneg_left (norm_factor_le hσ hγ u x₀ k l) (norm_nonneg v))

/-- The hypotheses of `norm_comp_factor_le` are satisfiable: the linear network. -/
example (v : EuclideanSpace ℝ ι →L[ℝ] ℝ) (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    ‖v ∘L factor id W Win b u x₀ k l‖ ≤ (‖W‖ * 1) ^ l * ‖v‖ :=
  norm_comp_factor_le (σ' := fun _ => 1) hasDerivAt_id (fun _ => by simp) v u x₀ k l

/-- **§2.1, the sufficient condition for vanishing, as proved.**  The paper
states it with `λ₁ < 1/γ` (false, `Section2_Counterexample`); its proof gives
it with `‖W_rec‖ < 1/γ`, written `‖W_rec‖ γ < 1`: the factors `∂x_{k+l}/∂x_k`
go to `0` as `l → ∞`, exponentially fast by `norm_factor_le`. -/
theorem tendsto_norm_factor (hσ : ∀ y, HasDerivAt σ (σ' y) y) (hγ : ∀ y, |σ' y| ≤ γ)
    (hW : ‖W‖ * γ < 1) (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k : ℕ) :
    Tendsto (fun l => ‖factor σ W Win b u x₀ k l‖) atTop (𝓝 0) :=
  squeeze_zero (fun _ => norm_nonneg _) (norm_factor_le hσ hγ u x₀ k)
    (tendsto_pow_atTop_nhds_zero_of_lt_one
      (mul_nonneg (norm_nonneg _) ((abs_nonneg _).trans (hγ 0))) hW)

/-- The hypotheses of `tendsto_norm_factor` are satisfiable: `W_rec = 0`. -/
example (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k : ℕ) :
    Tendsto (fun l => ‖factor id (0 : Matrix ι ι ℝ) Win b u x₀ k l‖) atTop (𝓝 0) :=
  tendsto_norm_factor (σ' := fun _ => 1) (γ := 1) hasDerivAt_id (fun _ => by simp) (by simp) u x₀ k

/-- **§2.1, the necessary condition for exploding, as proved.**  "By inverting
this proof we get the necessary condition for exploding gradients, namely that
the largest eigenvalue `λ₁` is larger than `1/γ`": false with `λ₁`
(`Section2_Counterexample`); inverted, the proof gives `‖W_rec‖ > 1/γ`, written
`1 < ‖W_rec‖ γ`, as soon as the factors `∂x_{k+l}/∂x_k` are unbounded in `l`. -/
theorem one_lt_of_not_bddAbove (hσ : ∀ y, HasDerivAt σ (σ' y) y) (hγ : ∀ y, |σ' y| ≤ γ)
    (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k : ℕ)
    (h : ¬ BddAbove (Set.range fun l => ‖factor σ W Win b u x₀ k l‖)) : 1 < ‖W‖ * γ := by
  by_contra hW
  refine h ⟨1, ?_⟩
  rintro _ ⟨l, rfl⟩
  exact (norm_factor_le hσ hγ u x₀ k l).trans
    (pow_le_one₀ (mul_nonneg (norm_nonneg _) ((abs_nonneg _).trans (hγ 0))) (not_lt.1 hW))

/-- The hypotheses of `one_lt_of_not_bddAbove` are satisfiable: the linear
network of one unit with `W_rec = 2`, whose factors are `2^l`. -/
example (u : ℕ → Fin 1 → ℝ) (Win : Matrix (Fin 1) (Fin 1) ℝ) (b x₀ : EuclideanSpace ℝ (Fin 1))
    (k : ℕ) : 1 < ‖Matrix.diagonal fun _ : Fin 1 => (2 : ℝ)‖ * 1 := by
  refine one_lt_of_not_bddAbove (σ' := fun _ => 1) (Win := Win) (b := b) hasDerivAt_id
    (fun _ => by simp) u x₀ k ?_
  rw [not_bddAbove_iff]
  intro r
  obtain ⟨l, hl⟩ := pow_unbounded_of_one_lt r (by norm_num : (1 : ℝ) < 2)
  refine ⟨_, ⟨l, rfl⟩, ?_⟩
  dsimp only
  rw [factor, (hasFDerivAt_states_step (σ' := fun _ => 1) hasDerivAt_id u x₀ k l).fderiv,
    Matrix.l2_opNorm_toEuclideanCLM, transport_const, Matrix.diagonal_mul_diagonal,
    Matrix.diagonal_pow, Matrix.l2_opNorm_diagonal]
  convert hl using 1
  rw [show ((fun _ : Fin 1 => (2 : ℝ) * 1) ^ l) = fun _ => (2 : ℝ) ^ l from
    funext fun _ => by simp, pi_norm_const]
  simp

/-- `tanh' = 1 - tanh²`. -/
theorem hasDerivAt_tanh (y : ℝ) : HasDerivAt Real.tanh (1 - Real.tanh y ^ 2) y := by
  have h := (Real.hasDerivAt_sinh y).div (Real.hasDerivAt_cosh y) (Real.cosh_pos y).ne'
  have e : Real.sinh / Real.cosh = Real.tanh :=
    funext fun z => (Real.tanh_eq_sinh_div_cosh z).symm
  rw [e] at h
  convert h using 1
  have hc := (Real.cosh_pos y).ne'
  rw [Real.tanh_eq_sinh_div_cosh]
  field_simp

/-- **§2.1: `γ = 1` for tanh**: `|tanh'| ≤ 1`, with equality at `0`. -/
theorem isGreatest_abs_deriv_tanh : IsGreatest (Set.range fun y => |deriv Real.tanh y|) 1 := by
  refine ⟨⟨0, by simp [(hasDerivAt_tanh 0).deriv]⟩, ?_⟩
  rintro _ ⟨y, rfl⟩
  have h := Real.tanh_sq_lt_one y
  simp only [(hasDerivAt_tanh y).deriv]
  rw [abs_of_nonneg (by linarith)]
  nlinarith [sq_nonneg (Real.tanh y)]

/-- **§2.1: `γ = 1/4` for the sigmoid**: `|sigmoid'| ≤ 1/4`, with equality at
`0`. -/
theorem isGreatest_abs_deriv_sigmoid :
    IsGreatest (Set.range fun y => |deriv Real.sigmoid y|) (1 / 4) := by
  refine ⟨⟨0, by norm_num [Real.deriv_sigmoid]⟩, ?_⟩
  rintro _ ⟨y, rfl⟩
  have h₀ := Real.sigmoid_nonneg y
  have h₁ := Real.sigmoid_le_one y
  simp only [Real.deriv_sigmoid]
  rw [abs_of_nonneg (mul_nonneg h₀ (by linarith))]
  nlinarith [sq_nonneg (Real.sigmoid y - 1 / 2)]

end Transformer.RecurrentGradients
