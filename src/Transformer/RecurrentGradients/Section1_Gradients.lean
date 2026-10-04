/-
# The gradient as a sum of products

arXiv:1211.5063, §1.1, eqs. (3)–(5).  Let `A_i = ∂x_{i+1}/∂x_i` and
`B_i = ∂⁺x_{i+1}/∂θ` be the partial derivatives of eq. (1) at `x_i`, the latter
"where `x_i` is taken as a constant with respect to `θ`".  The chain rule gives
`∂x_{k+l}/∂x_k = A_{k+l-1} ⋯ A_k` (`transport`; eq. (5), `hasFDerivAt_states`),
the derivative of `x_t` in `θ` as the sum over `1 ≤ k ≤ t` of the temporal
contributions `∂x_t/∂x_k ∂⁺x_k/∂θ` (`hasFDerivAt_states_param`), that of
`L_t = 𝓛(x_t)` with `∂L_t/∂x_t` in front (eq. (4), `hasFDerivAt_loss`), and that
of `L = Σ_{1 ≤ t ≤ T} L_t` as the sum of those (eq. (3), `hasFDerivAt_sum_loss`).

For eq. (2) the paper writes eq. (5) as `∏_{t ≥ i > k} W_recᵀ diag(σ'(x_{i-1}))`.
The Jacobian is `W_rec diag(σ'(x_{i-1}))` (`hasFDerivAt_step`), so
`hasFDerivAt_states_step` proves the product with `W_rec` in place of
`W_recᵀ`; the bounds of §2.1 hold for both, as `‖W_recᵀ‖ = ‖W_rec‖`.
-/

import Transformer.RecurrentGradients.Section1_Recurrence
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Analysis.Calculus.FDeriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Comp

open scoped Matrix

namespace Transformer.RecurrentGradients

section Monoid

variable {M : Type*} [Monoid M]

/-- `A (k + l - 1) ⋯ A (k + 1) A k`, the product of eq. (5), which carries `x_k`
to `x_{k+l}` when `A i` is the Jacobian `∂x_{i+1}/∂x_i`. -/
def transport (A : ℕ → M) (k : ℕ) : ℕ → M
  | 0 => 1
  | l + 1 => A (k + l) * transport A k l

theorem map_transport {N F : Type*} [Monoid N] [FunLike F M N] [MonoidHomClass F M N] (f : F)
    (A : ℕ → M) (k l : ℕ) : f (transport A k l) = transport (fun i => f (A i)) k l := by
  induction l with
  | zero => exact map_one f
  | succ l ih => rw [transport, transport, map_mul, ih]

/-- Along a constant Jacobian the product is a power. -/
theorem transport_const (a : M) (k l : ℕ) : transport (fun _ => a) k l = a ^ l := by
  induction l with
  | zero => exact (pow_zero a).symm
  | succ l ih => rw [transport, ih, pow_succ']

/-- A sum over `1 ≤ k ≤ t` as a sum over `range t`. -/
theorem sum_Icc_one {N : Type*} [AddCommMonoid N] (f : ℕ → N) (t : ℕ) :
    ∑ k ∈ Finset.Icc 1 t, f k = ∑ i ∈ Finset.range t, f (i + 1) := by
  rw [← Finset.Ico_add_one_right_eq_Icc, Finset.sum_Ico_eq_sum_range, Nat.add_sub_cancel]
  simp only [add_comm 1]

end Monoid

variable {E U Θ : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup Θ]
  [NormedSpace ℝ Θ]

/-- **Eq. (5)**, `∂x_{k+l}/∂x_k = ∏ ∂x_{i+1}/∂x_i`: if the step of eq. (1) from
`x_i` has the Jacobian `A i`, the state `l` steps after `x_k` has the Jacobian
`A (k + l - 1) ⋯ A k` in `x_k`. -/
theorem hasFDerivAt_states {F : E → U → E} {u : ℕ → U} {x₀ : E} {A : ℕ → E →L[ℝ] E}
    (hF : ∀ i, HasFDerivAt (fun x => F x (u (i + 1))) (A i) (states F u x₀ i)) (k l : ℕ) :
    HasFDerivAt (fun x => states F (fun i => u (k + i)) x l) (transport A k l)
      (states F u x₀ k) := by
  induction l with
  | zero => exact hasFDerivAt_id _
  | succ l ih =>
    have h := hF (k + l)
    rw [states_add] at h
    exact h.comp (states F u x₀ k) ih

/-- The hypothesis of `hasFDerivAt_states` is satisfiable: the identity steps. -/
example (x₀ : E) (k l : ℕ) :
    HasFDerivAt (fun x => states (fun x (_ : Unit) => x) (fun _ => ()) x l)
      (transport (fun _ => (1 : E →L[ℝ] E)) k l) (states (fun x (_ : Unit) => x) (fun _ => ()) x₀ k) :=
  hasFDerivAt_states (F := fun x (_ : Unit) => x) (u := fun _ => ()) (A := fun _ => 1)
    (fun _ => hasFDerivAt_id _) k l

/-- **The derivative of `x_t` in `θ`**, the factor of eq. (4) after `∂L_t/∂x_t`:
the sum over `1 ≤ k ≤ t` of the temporal contributions
`∂x_t/∂x_k ∂⁺x_k/∂θ`, where `x_0` does not depend on `θ`. -/
theorem hasFDerivAt_states_param {F : Θ → E → U → E} {u : ℕ → U} {x₀ : E} {θ : Θ}
    {A : ℕ → E →L[ℝ] E} {B : ℕ → Θ →L[ℝ] E}
    (hF : ∀ i, HasFDerivAt (fun p : E × Θ => F p.2 p.1 (u (i + 1))) ((A i).coprod (B i))
      (states (F θ) u x₀ i, θ)) (t : ℕ) :
    HasFDerivAt (fun θ' => states (F θ') u x₀ t)
      (∑ k ∈ Finset.Icc 1 t, transport A k (t - k) ∘L B (k - 1)) θ := by
  rw [sum_Icc_one]
  simp only [Nat.add_sub_cancel]
  induction t with
  | zero =>
    simp only [Finset.range_zero, Finset.sum_empty]
    exact hasFDerivAt_const x₀ θ
  | succ t ih =>
    have hp : HasFDerivAt (fun θ' => (states (F θ') u x₀ t, θ'))
        ((∑ i ∈ Finset.range t, transport A (i + 1) (t - (i + 1)) ∘L B i).prod
          (ContinuousLinearMap.id ℝ Θ)) θ := ih.prodMk (hasFDerivAt_id θ)
    refine (HasFDerivAt.comp (f := fun θ' => (states (F θ') u x₀ t, θ')) θ (hF t) hp).congr_fderiv
      ?_
    have e : ((A t).coprod (B t)).comp
        ((∑ i ∈ Finset.range t, transport A (i + 1) (t - (i + 1)) ∘L B i).prod
          (ContinuousLinearMap.id ℝ Θ)) =
        (A t).comp (∑ i ∈ Finset.range t, transport A (i + 1) (t - (i + 1)) ∘L B i) + B t :=
      ContinuousLinearMap.ext fun v => by simp
    rw [e, ContinuousLinearMap.comp_finsetSum, Finset.sum_range_succ, Nat.sub_self]
    congr 1
    refine Finset.sum_congr rfl fun i hi => ?_
    have hi : i < t := Finset.mem_range.mp hi
    rw [show t + 1 - (i + 1) = t - (i + 1) + 1 by omega, transport,
      show i + 1 + (t - (i + 1)) = t by omega]
    rfl

/-- The hypothesis of `hasFDerivAt_states_param` is satisfiable: `θ` added at
every step. -/
example (x₀ θ : E) (t : ℕ) :
    HasFDerivAt (fun θ' => states (fun x (_ : Unit) => x + θ') (fun _ => ()) x₀ t)
      (∑ k ∈ Finset.Icc 1 t, transport (fun _ => (1 : E →L[ℝ] E)) k (t - k) ∘L
        ContinuousLinearMap.id ℝ E) θ :=
  hasFDerivAt_states_param (A := fun _ => 1) (B := fun _ => ContinuousLinearMap.id ℝ E)
    (F := fun θ' x (_ : Unit) => x + θ')
    (fun _ => (hasFDerivAt_fst.add hasFDerivAt_snd).congr_fderiv
      (ContinuousLinearMap.ext fun p => by simp)) t

/-- **Eq. (4)**: `∂L_t/∂θ = Σ_{1 ≤ k ≤ t} ∂L_t/∂x_t ∂x_t/∂x_k ∂⁺x_k/∂θ`. -/
theorem hasFDerivAt_loss {F : Θ → E → U → E} {u : ℕ → U} {x₀ : E} {θ : Θ}
    {A : ℕ → E →L[ℝ] E} {B : ℕ → Θ →L[ℝ] E} {ℓ : E → ℝ} {ℓ' : E →L[ℝ] ℝ} {t : ℕ}
    (hF : ∀ i, HasFDerivAt (fun p : E × Θ => F p.2 p.1 (u (i + 1))) ((A i).coprod (B i))
      (states (F θ) u x₀ i, θ)) (hℓ : HasFDerivAt ℓ ℓ' (states (F θ) u x₀ t)) :
    HasFDerivAt (fun θ' => ℓ (states (F θ') u x₀ t))
      (∑ k ∈ Finset.Icc 1 t, ℓ' ∘L transport A k (t - k) ∘L B (k - 1)) θ := by
  have h := hℓ.comp θ (hasFDerivAt_states_param hF t)
  rwa [ContinuousLinearMap.comp_finsetSum] at h

/-- **Eq. (3)**: `∂L/∂θ = Σ_{1 ≤ t ≤ T} ∂L_t/∂θ` for `L = Σ_{1 ≤ t ≤ T} L_t`, each
term expanded by eq. (4). -/
theorem hasFDerivAt_sum_loss {F : Θ → E → U → E} {u : ℕ → U} {x₀ : E} {θ : Θ}
    {A : ℕ → E →L[ℝ] E} {B : ℕ → Θ →L[ℝ] E} {ℓ : ℕ → E → ℝ} {ℓ' : ℕ → E →L[ℝ] ℝ}
    (hF : ∀ i, HasFDerivAt (fun p : E × Θ => F p.2 p.1 (u (i + 1))) ((A i).coprod (B i))
      (states (F θ) u x₀ i, θ)) (hℓ : ∀ t, HasFDerivAt (ℓ t) (ℓ' t) (states (F θ) u x₀ t))
    (T : ℕ) :
    HasFDerivAt (fun θ' => ∑ t ∈ Finset.Icc 1 T, ℓ t (states (F θ') u x₀ t))
      (∑ t ∈ Finset.Icc 1 T, ∑ k ∈ Finset.Icc 1 t, ℓ' t ∘L transport A k (t - k) ∘L B (k - 1))
      θ :=
  HasFDerivAt.fun_sum fun t _ => hasFDerivAt_loss hF (hℓ t)

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] {σ σ' : ℝ → ℝ}
  {W : Matrix ι ι ℝ} {Win : Matrix ι κ ℝ} {b : EuclideanSpace ℝ ι}

/-- **Eq. (5) for eq. (2), corrected.**  `∂x_{k+l}/∂x_k` is the product of the
matrices `W_rec diag(σ'(x_i))` for `k ≤ i < k + l`, the latest on the left.  The
paper writes `∏_{t ≥ i > k} W_recᵀ diag(σ'(x_{i-1}))`; the transpose is an
error, since the `(a, b)` entry of `∂x_i/∂x_{i-1}` is `W_{ab} σ'(x_{i-1,b})`
(`hasFDerivAt_step`). -/
theorem hasFDerivAt_states_step (hσ : ∀ y, HasDerivAt σ (σ' y) y) (u : ℕ → κ → ℝ)
    (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    HasFDerivAt (fun x => states (step σ W Win b) (fun i => u (k + i)) x l)
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (transport
        (fun i => W * Matrix.diagonal fun j => σ' (states (step σ W Win b) u x₀ i j)) k l))
      (states (step σ W Win b) u x₀ k) := by
  rw [map_transport]
  exact hasFDerivAt_states (fun _ => hasFDerivAt_step fun _ => hσ _) k l

/-- The hypothesis of `hasFDerivAt_states_step` is satisfiable: the identity. -/
example (u : ℕ → κ → ℝ) (x₀ : EuclideanSpace ℝ ι) (k l : ℕ) :
    HasFDerivAt (fun x => states (step id W Win b) (fun i => u (k + i)) x l)
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (transport
        (fun _ => W * Matrix.diagonal fun _ => (1 : ℝ)) k l)) (states (step id W Win b) u x₀ k) :=
  hasFDerivAt_states_step (σ' := fun _ => 1) hasDerivAt_id u x₀ k l

end Transformer.RecurrentGradients
