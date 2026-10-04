/-
# Recurrent networks and their Jacobians

arXiv:1211.5063, §1.1 ("Training recurrent networks").  The generic network of
eq. (1), `x_t = F(x_{t-1}, u_t, θ)`, has the states `states F u x₀`, with `θ`
inside `F`.  Its parametrization of eq. (2),
`x_t = W_rec σ(x_{t-1}) + W_in u_t + b`, is `step σ W_rec W_in b`, on the
Euclidean space, whose norm is the 2-norm of §2.1.  Footnote 1 calls eq. (2)
equivalent to `x_t = σ(W_rec x_{t-1} + W_in u_t + b)` (`stepAfter`): `σ` maps
the states of eq. (2) onto those of the other form (`act_states`), and from the
first step on every state of the other form is such an image
(`states_stepAfter`).

The Jacobian of eq. (2) is `W_rec diag(σ'(x))` (`hasFDerivAt_step`), where
eq. (5) writes `W_recᵀ diag(σ'(x))` (`Section1_Gradients`).  The "immediate"
derivative of eq. (2) in the `(i, j)` entry of `W_rec` is `σ(x)_j` in unit `i`
and `0` in the others (`hasDerivAt_step_single`): row `i` of `∂⁺x_k/∂W_rec` is
`σ(x_{k-1})`, as §1.1 says.
-/

import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Analysis.Calculus.FDeriv.WithLp
import Mathlib.Data.Matrix.Basis

open scoped Matrix

namespace Transformer.RecurrentGradients

section Generic

variable {E U : Type*}

/-- The states of the generic recurrent network of eq. (1),
`x_t = F(x_{t-1}, u_t, θ)`, with `θ` fixed inside `F`, from the state `x₀`. -/
def states (F : E → U → E) (u : ℕ → U) (x₀ : E) : ℕ → E
  | 0 => x₀
  | t + 1 => F (states F u x₀ t) (u (t + 1))

/-- Running `k + l` steps is running `k`, then `l` more on the later inputs. -/
theorem states_add (F : E → U → E) (u : ℕ → U) (x₀ : E) (k l : ℕ) :
    states F u x₀ (k + l) = states F (fun i => u (k + i)) (states F u x₀ k) l := by
  induction l with
  | zero => rfl
  | succ l ih =>
    change F (states F u x₀ (k + l)) (u (k + l + 1)) = _
    rw [ih]
    rfl

end Generic

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ]
  {σ σ' : ℝ → ℝ} {W : Matrix ι ι ℝ} {Win : Matrix ι κ ℝ} {b x : EuclideanSpace ℝ ι}
  {u : κ → ℝ}

/-- `σ` applied to each unit, as in eq. (2). -/
noncomputable def act (σ : ℝ → ℝ) (x : EuclideanSpace ℝ ι) : EuclideanSpace ℝ ι :=
  WithLp.toLp 2 fun i => σ (x i)

/-- The map of eq. (2) at the input `u`: `x ↦ W_rec σ(x) + W_in u + b`. -/
noncomputable def step (σ : ℝ → ℝ) (W : Matrix ι ι ℝ) (Win : Matrix ι κ ℝ)
    (b : EuclideanSpace ℝ ι) (x : EuclideanSpace ℝ ι) (u : κ → ℝ) : EuclideanSpace ℝ ι :=
  Matrix.toEuclideanCLM (𝕜 := ℝ) W (act σ x) + WithLp.toLp 2 (Win *ᵥ u) + b

/-- The form that footnote 1 calls "more widely known":
`x ↦ σ(W_rec x + W_in u + b)`. -/
noncomputable def stepAfter (σ : ℝ → ℝ) (W : Matrix ι ι ℝ) (Win : Matrix ι κ ℝ)
    (b : EuclideanSpace ℝ ι) (x : EuclideanSpace ℝ ι) (u : κ → ℝ) : EuclideanSpace ℝ ι :=
  act σ (Matrix.toEuclideanCLM (𝕜 := ℝ) W x + WithLp.toLp 2 (Win *ᵥ u) + b)

/-- **Footnote 1, one way.**  `σ` maps the states of eq. (2) from `x₀` onto the
states of `x_t = σ(W_rec x_{t-1} + W_in u_t + b)` from `σ(x₀)`. -/
theorem act_states (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (t : ℕ) :
    act σ (states (step σ W Win b) u x₀ t) = states (stepAfter σ W Win b) u (act σ x₀) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change act σ (step σ W Win b _ _) = stepAfter σ W Win b _ _
    rw [← ih]
    rfl

/-- **Footnote 1, the other way.**  From the first step on, the states of
`x_t = σ(W_rec x_{t-1} + W_in u_t + b)` from any `y₀`, which need not lie in
the range of `σ`, are `σ` of the states of eq. (2) from the pre-activation
`W_rec y₀ + W_in u_1 + b`, on the inputs from `u_2` on. -/
theorem states_stepAfter (u : ℕ → κ → ℝ) (y₀ : EuclideanSpace ℝ ι) (t : ℕ) :
    states (stepAfter σ W Win b) u y₀ (t + 1) =
      act σ (states (step σ W Win b) (fun i => u (i + 1))
        (Matrix.toEuclideanCLM (𝕜 := ℝ) W y₀ + WithLp.toLp 2 (Win *ᵥ u 1) + b) t) := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change stepAfter σ W Win b (states (stepAfter σ W Win b) u y₀ (t + 1)) _ = _
    rw [ih]
    rfl

/-- The Jacobian of `σ` applied to each unit is `diag(σ'(x))`. -/
theorem hasFDerivAt_act (hσ : ∀ i, HasDerivAt σ (σ' (x i)) (x i)) :
    HasFDerivAt (act σ) (Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal fun i => σ' (x i))) x := by
  have h : HasFDerivAt (fun y : EuclideanSpace ℝ ι => fun i => σ (y i))
      (ContinuousLinearMap.pi fun i => (ContinuousLinearMap.toSpanSingleton ℝ (σ' (x i))).comp
        (PiLp.proj 2 (fun _ : ι => ℝ) i)) x :=
    hasFDerivAt_pi.2 fun i => (hσ i).hasFDerivAt.comp x (PiLp.hasFDerivAt_apply 2 x i)
  refine ((PiLp.hasFDerivAt_toLp 2 _).comp x h).congr_fderiv ?_
  ext v i
  simp [Matrix.mulVec_diagonal, mul_comm]

/-- The hypothesis of `hasFDerivAt_act` is satisfiable: the identity. -/
example (x : EuclideanSpace ℝ ι) :
    HasFDerivAt (act id) (Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.diagonal fun _ : ι => (1 : ℝ))) x :=
  hasFDerivAt_act (σ' := fun _ => 1) fun i => hasDerivAt_id (x i)

/-- **The Jacobian of eq. (2)** at `x` is `W_rec diag(σ'(x))`: its `(a, b)`
entry is `W_{ab} σ'(x_b)`. -/
theorem hasFDerivAt_step (hσ : ∀ i, HasDerivAt σ (σ' (x i)) (x i)) :
    HasFDerivAt (fun y => step σ W Win b y u)
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (W * Matrix.diagonal fun i => σ' (x i))) x := by
  have h := ((Matrix.toEuclideanCLM (𝕜 := ℝ) W).hasFDerivAt.comp x (hasFDerivAt_act hσ)).add_const
    (WithLp.toLp 2 (Win *ᵥ u) + b)
  have e : (fun y => step σ W Win b y u) =
      fun y => (Matrix.toEuclideanCLM (𝕜 := ℝ) W ∘ act σ) y + (WithLp.toLp 2 (Win *ᵥ u) + b) := by
    funext y
    simp only [step, Function.comp_apply, add_assoc]
  rw [e, map_mul]
  exact h

/-- The hypothesis of `hasFDerivAt_step` is satisfiable: the linear network. -/
example (x : EuclideanSpace ℝ ι) :
    HasFDerivAt (fun y => step id W Win b y u)
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (W * Matrix.diagonal fun _ : ι => (1 : ℝ))) x :=
  hasFDerivAt_step (σ' := fun _ => 1) fun i => hasDerivAt_id (x i)

/-- **The immediate derivative in `W_rec`** (§1.1): moving the `(i, j)` entry of
`W_rec` moves unit `i` at the rate `σ(x)_j` and no other unit, so that row `i`
of `∂⁺x_k/∂W_rec` is `σ(x_{k-1})`. -/
theorem hasDerivAt_step_single (i j : ι) :
    HasDerivAt (fun s : ℝ => step σ (W + s • Matrix.single i j 1) Win b x u)
      (WithLp.toLp 2 (Function.update (0 : ι → ℝ) i (σ (x j)))) 0 := by
  have h := HasDerivAt.const_add (step σ W Win b x u) (HasDerivAt.smul_const (hasDerivAt_id' (0 : ℝ))
    (WithLp.toLp 2 (Function.update (0 : ι → ℝ) i (σ (x j))) : EuclideanSpace ℝ ι))
  rw [one_smul] at h
  have e : (fun s : ℝ => step σ (W + s • Matrix.single i j 1) Win b x u) =
      fun s => step σ W Win b x u + s • (WithLp.toLp 2 (Function.update (0 : ι → ℝ) i (σ (x j))) :
        EuclideanSpace ℝ ι) := by
    funext s
    have hs : Matrix.toEuclideanCLM (𝕜 := ℝ) (Matrix.single i j 1) (act σ x) =
        WithLp.toLp 2 (Function.update (0 : ι → ℝ) i (σ (x j))) := by
      ext1
      simp [Matrix.single_mulVec, act]
    simp only [step, map_add, map_smul, add_apply, smul_apply, hs]
    abel
  rw [e]
  exact h

end Transformer.RecurrentGradients
