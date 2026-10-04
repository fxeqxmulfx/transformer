/-
# The derivative in an input

arXiv:1211.5063, §3.1 and §3.3.  The information an input inserts in the model
(§3.1) reaches the later states through the derivative of the state in that
input.  The derivative of `x_{k+1+l}` in `u_{k+1}` is `∂x_{k+1+l}/∂x_{k+1}`
applied to the immediate derivative of `x_{k+1}` in `u_{k+1}`
(`hasDerivAt_states_input`), as the states up to `k` do not see `u_{k+1}`
(`states_congr`): "`∂x_t/∂x_k` is a factor in `∂L_t/∂u_k`" (§3.3), whose other
factor is `∂L_t/∂x_t`.  For eq. (2) the immediate derivative is
`W_in` (`hasDerivAt_step_input`).  For the form
`y_t = σ(W_rec y_{t-1} + W_in u_t + b)` of footnote 1 it is
`diag(σ'(W_rec y + W_in u + b)) W_in` (`hasDerivAt_stepAfter_input`), and the
Jacobian is `diag(σ'(W_rec y + W_in u + b)) W_rec` (`hasFDerivAt_stepAfter`).
-/

import Transformer.RecurrentGradients.Section1_Gradients
import Mathlib.Analysis.Calculus.Deriv.Comp

open scoped Matrix

namespace Transformer.RecurrentGradients

/-- The states up to `t` depend on the inputs up to `t` only. -/
theorem states_congr {E U : Type*} {F : E → U → E} {u v : ℕ → U} {x₀ : E} {t : ℕ}
    (h : ∀ i < t, u (i + 1) = v (i + 1)) : states F u x₀ t = states F v x₀ t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change F (states F u x₀ t) (u (t + 1)) = F (states F v x₀ t) (v (t + 1))
    rw [ih fun i hi => h i (by omega), h t (by omega)]

/-- The hypothesis of `states_congr` is satisfiable: inputs that differ at `3` only. -/
example (x₀ : ℝ) : states (fun x v => x + v) (fun i => if i = 3 then (1 : ℝ) else 0) x₀ 2 =
    states (fun x v => x + v) (fun _ => (0 : ℝ)) x₀ 2 :=
  states_congr fun i hi => ite_eq_right (by omega)

/-- **The derivative of `x_{k+1+l}` in the input `u_{k+1}`**, along `e`, is
`∂x_{k+1+l}/∂x_{k+1}` applied to that of `x_{k+1}`. -/
theorem hasDerivAt_states_input {E U : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [AddCommGroup U] [Module ℝ U] {F : E → U → E} {u : ℕ → U} {x₀ : E} {A : ℕ → E →L[ℝ] E}
    (hF : ∀ i, HasFDerivAt (fun x => F x (u (i + 1))) (A i) (states F u x₀ i)) {k : ℕ} {e : U}
    {v : E} (hv : HasDerivAt (fun s : ℝ => F (states F u x₀ k) (u (k + 1) + s • e)) v 0)
    (l : ℕ) :
    HasDerivAt (fun s : ℝ =>
      states F (Function.update u (k + 1) (u (k + 1) + s • e)) x₀ (k + 1 + l))
      (transport A (k + 1) l v) 0 := by
  have e₁ : ∀ s : ℝ,
      states F (Function.update u (k + 1) (u (k + 1) + s • e)) x₀ (k + 1 + l) =
        states F (fun i => u (k + 1 + i)) (F (states F u x₀ k) (u (k + 1) + s • e)) l := by
    intro s
    have hk : states F (Function.update u (k + 1) (u (k + 1) + s • e)) x₀ k = states F u x₀ k :=
      states_congr fun i hi => Function.update_of_ne (by omega) _ _
    rw [states_add, states_congr
      (u := fun i => Function.update u (k + 1) (u (k + 1) + s • e) (k + 1 + i))
      (v := fun i => u (k + 1 + i)) fun i _ => Function.update_of_ne (by omega) _ _]
    change states F _ (F (states F _ x₀ k) (Function.update u (k + 1) _ (k + 1))) l = _
    rw [Function.update_self, hk]
  rw [show (fun s : ℝ =>
      states F (Function.update u (k + 1) (u (k + 1) + s • e)) x₀ (k + 1 + l)) =
      (fun x => states F (fun i => u (k + 1 + i)) x l) ∘
        fun s : ℝ => F (states F u x₀ k) (u (k + 1) + s • e) from funext e₁]
  exact HasFDerivAt.comp_hasDerivAt_of_eq 0 (hasFDerivAt_states hF (k + 1) l) hv
    (show states F u x₀ (k + 1) = F (states F u x₀ k) (u (k + 1) + (0 : ℝ) • e) by
      rw [zero_smul, add_zero]; rfl)

/-- The hypotheses of `hasDerivAt_states_input` are satisfiable:
`x_t = x_{t-1} + u_t` on `ℝ`, whose Jacobians are `1`. -/
example (u : ℕ → ℝ) (x₀ e : ℝ) (k l : ℕ) :
    HasDerivAt (fun s : ℝ =>
      states (fun x v => x + v) (Function.update u (k + 1) (u (k + 1) + s • e)) x₀ (k + 1 + l))
      (transport (fun _ => ContinuousLinearMap.id ℝ ℝ) (k + 1) l e) 0 :=
  hasDerivAt_states_input (F := fun x v => x + v) (u := u) (x₀ := x₀) (k := k) (e := e)
    (fun _ => (hasFDerivAt_id _).add_const _)
    (by simpa using (hasDerivAt_id (0 : ℝ)).smul_const e) l

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] {σ σ' : ℝ → ℝ}
  {W : Matrix ι ι ℝ} {Win : Matrix ι κ ℝ} {b : EuclideanSpace ℝ ι}

/-- The immediate derivative of eq. (2) in the input, along `e`, is `W_in e`. -/
theorem hasDerivAt_step_input (x : EuclideanSpace ℝ ι) (v e : κ → ℝ) :
    HasDerivAt (fun s : ℝ => step σ W Win b x (v + s • e)) (WithLp.toLp 2 (Win *ᵥ e)) 0 := by
  have h := HasDerivAt.const_add (step σ W Win b x v) (HasDerivAt.smul_const
    (hasDerivAt_id (0 : ℝ)) (WithLp.toLp 2 (Win *ᵥ e) : EuclideanSpace ℝ ι))
  rw [one_smul] at h
  convert h using 1
  funext s
  simp only [step, Matrix.mulVec_add, Matrix.mulVec_smul, WithLp.toLp_add, WithLp.toLp_smul, id]
  abel

/-- The Jacobian of `y ↦ σ(W_rec y + W_in u + b)` is
`diag(σ'(W_rec y + W_in u + b)) W_rec`. -/
theorem hasFDerivAt_stepAfter (hσ : ∀ y, HasDerivAt σ (σ' y) y) (u : κ → ℝ)
    (y : EuclideanSpace ℝ ι) :
    HasFDerivAt (fun y => stepAfter σ W Win b y u) (Matrix.toEuclideanCLM (𝕜 := ℝ)
      (Matrix.diagonal (fun i => σ' ((Matrix.toEuclideanCLM (𝕜 := ℝ) W y +
        WithLp.toLp 2 (Win *ᵥ u) + b) i)) * W)) y := by
  have h := (hasFDerivAt_act (σ' := σ') fun _ => hσ _).comp y
    (((Matrix.toEuclideanCLM (𝕜 := ℝ) W).hasFDerivAt).add_const (WithLp.toLp 2 (Win *ᵥ u) + b))
  have e : (fun y => stepAfter σ W Win b y u) =
      act σ ∘ fun y => Matrix.toEuclideanCLM (𝕜 := ℝ) W y + (WithLp.toLp 2 (Win *ᵥ u) + b) := by
    funext y
    simp only [stepAfter, Function.comp_apply, add_assoc]
  rw [e, map_mul, add_assoc]
  exact h

/-- The hypothesis of `hasFDerivAt_stepAfter` is satisfiable: the identity. -/
example (u : κ → ℝ) (y : EuclideanSpace ℝ ι) :
    HasFDerivAt (fun y => stepAfter id W Win b y u)
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal (fun _ : ι => (1 : ℝ)) * W)) y :=
  hasFDerivAt_stepAfter (σ' := fun _ => 1) hasDerivAt_id u y

/-- The immediate derivative of `y ↦ σ(W_rec y + W_in u + b)` in the input, along
`e`, is `diag(σ'(W_rec y + W_in u + b)) W_in e`. -/
theorem hasDerivAt_stepAfter_input (hσ : ∀ y, HasDerivAt σ (σ' y) y) (y : EuclideanSpace ℝ ι)
    (v e : κ → ℝ) :
    HasDerivAt (fun s : ℝ => stepAfter σ W Win b y (v + s • e))
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal fun i => σ' ((Matrix.toEuclideanCLM
        (𝕜 := ℝ) W y + WithLp.toLp 2 (Win *ᵥ v) + b) i)) (WithLp.toLp 2 (Win *ᵥ e))) 0 := by
  have hline := HasDerivAt.const_add
    (Matrix.toEuclideanCLM (𝕜 := ℝ) W y + WithLp.toLp 2 (Win *ᵥ v) + b)
    (HasDerivAt.smul_const (hasDerivAt_id (0 : ℝ)) (WithLp.toLp 2 (Win *ᵥ e) : EuclideanSpace ℝ ι))
  rw [one_smul] at hline
  have h := HasFDerivAt.comp_hasDerivAt_of_eq 0 (hasFDerivAt_act (σ' := σ')
    (x := Matrix.toEuclideanCLM (𝕜 := ℝ) W y + WithLp.toLp 2 (Win *ᵥ v) + b) fun _ => hσ _)
    hline (by simp)
  convert h using 1
  funext s
  simp only [stepAfter, Function.comp_apply, Matrix.mulVec_add, Matrix.mulVec_smul,
    WithLp.toLp_add, WithLp.toLp_smul, id]
  congr 1
  abel

/-- The hypothesis of `hasDerivAt_stepAfter_input` is satisfiable: the identity. -/
example (y : EuclideanSpace ℝ ι) (v e : κ → ℝ) :
    HasDerivAt (fun s : ℝ => stepAfter id W Win b y (v + s • e))
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal fun _ : ι => (1 : ℝ))
        (WithLp.toLp 2 (Win *ᵥ e))) 0 :=
  hasDerivAt_stepAfter_input (σ' := fun _ => 1) hasDerivAt_id y v e

end Transformer.RecurrentGradients
