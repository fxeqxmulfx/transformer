module

public import Transformer.Landscape.SaddleAvoidance.SecondOrderCritical
public import Transformer.Grokking.FlatSpectrum.Training.Convergent

/-!
# Dependent columns: the Hessian along a free column

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD10.
Full row rank of the start needs width `k ≥ m + n`. For any width, let the columns of
`W = [U; V]` be dependent: `U z = 0` and `V z = 0` for some `z ≠ 0`. Moving the factors along
`(a zᴴ, b zᴴ)` changes neither the product nor the weight decay to first order, and with
`S = [a; b]` the gradient of the weight-decayed loss `F` satisfies
`⟨S zᴴ, ∇F(W + t S zᴴ)⟩ = t ‖z‖² ⟨S, M_t S⟩` exactly. So the Hessian of `F` at `W` in the
direction `S zᴴ` is `‖z‖² ⟨S, M S⟩` (`inner_fderiv_gradient_flatLoss`): a negative direction
of `M` is a negative direction of the Hessian. A critical point on the factor space is
stationary, `M W = 0` (`lift_mul_stack_eq_zero_of_gradient_eq_zero`). `Training.Width` draws
the consequences.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k p ι : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype p] [Fintype ι]
  [DecidableEq ι] [Unique ι]

/-- For a one-column `Z`, `Zᴴ Z` is the `1 × 1` matrix `‖Z‖²`. -/
lemma conjTranspose_mul_self_unit (Z : Matrix k ι ℂ) :
    Zᴴ * Z = frobeniusSq Z • (1 : Matrix ι ι ℂ) := by
  ext i j
  obtain rfl := Unique.eq_default i
  obtain rfl := Unique.eq_default j
  simp only [Matrix.mul_apply, conjTranspose_apply, Matrix.smul_apply, Matrix.one_apply_eq,
    frobeniusSq, Finset.univ_unique, Finset.sum_singleton, Complex.real_smul, mul_one]
  push_cast
  exact Finset.sum_congr rfl fun i _ => by rw [Complex.star_def, Complex.conj_mul']

/-- `⟨P Zᴴ, Q Zᴴ⟩ = ‖Z‖² ⟨P, Q⟩` for a one-column `Z`. -/
lemma pairing_mul_conjTranspose_unit (P Q : Matrix p ι ℂ) (Z : Matrix k ι ℂ) :
    pairing (P * Zᴴ) (Q * Zᴴ) = frobeniusSq Z * pairing P Q := by
  rw [pairing_mul_right, conjTranspose_conjTranspose, Matrix.mul_assoc,
    conjTranspose_mul_self_unit, Matrix.mul_smul, Matrix.mul_one, pairing_smul_right]

variable {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ} [DecidableEq m]
  [DecidableEq n]

/-- **The Hessian along a free column** (`review/flat_spectrum.md`, item WD10): let `L` be
real-analytic in the entries with the first-order inequality and a continuous `G`, and let the
one-column `Z` have `U Z = 0` and `V Z = 0`. For one-column `A`, `B` the Hessian of the
weight-decayed loss on the factor space in the direction `(A Zᴴ, B Zᴴ)` is
`‖Z‖² ⟨S, M S⟩` with `S = [A; B]` and `M = [[λ, G], [Gᴴ, λ]]` at `G = G (U Vᴴ)`. -/
theorem inner_fderiv_gradient_flatLoss (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y)
    (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (lam : ℝ) (x : EuclideanSpace ℂ ((m ⊕ n) × k)) {Z : Matrix k Unit ℂ}
    (hU : (factors x).1 * Z = 0) (hV : (factors x).2 * Z = 0) (A : Matrix m Unit ℂ)
    (B : Matrix n Unit ℂ) :
    inner ℝ (flatten (A * Zᴴ, B * Zᴴ))
        (fderiv ℝ (gradient (flatLoss L lam)) x (flatten (A * Zᴴ, B * Zᴴ))) =
      frobeniusSq Z * pairing (stack A B)
        (lift lam (G ((factors x).1 * (factors x).2ᴴ)) * stack A B) := by
  set h : EuclideanSpace ℂ ((m ⊕ n) × k) := flatten (A * Zᴴ, B * Zᴴ) with hh
  have hf : AnalyticOnNhd ℝ (flatLoss (k := k) L lam) Set.univ :=
    fun y _ => analyticAt_flatLoss hL lam y
  have hgd : DifferentiableAt ℝ (gradient (flatLoss (k := k) L lam)) x :=
    (Landscape.SaddleAvoidance.contDiff_gradient hf.contDiff).differentiable one_ne_zero x
  have hline : HasDerivAt (fun t : ℝ => x + t • h) h 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const h).const_add x
  have h1 := (innerSL ℝ h).hasFDerivAt.comp_hasDerivAt (0 : ℝ)
    (hgd.hasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) hline (by simp))
  have hWZ : stack (factors x).1 (factors x).2 * Z = 0 := by
    rw [stack, fromRows_mul, hU, hV]
    exact stack_zero
  have key : ∀ t : ℝ, innerSL ℝ h (gradient (flatLoss L lam) (x + t • h)) =
      t * (frobeniusSq Z * pairing (stack A B)
        (lift lam (G (((factors x).1 + t • (A * Zᴴ)) * ((factors x).2 + t • (B * Zᴴ))ᴴ)) *
          stack A B)) := by
    intro t
    have hfac : factors (x + t • h) =
        ((factors x).1 + t • (A * Zᴴ), (factors x).2 + t • (B * Zᴴ)) := by
      rw [factors_add, factors_smul, hh, factors_flatten]
    have e1 : stack (A * Zᴴ) (B * Zᴴ) = stack A B * Zᴴ := (fromRows_mul A B Zᴴ).symm
    have e2 : stack ((factors x).1 + t • (A * Zᴴ)) ((factors x).2 + t • (B * Zᴴ)) =
        stack (factors x).1 (factors x).2 + t • (stack A B * Zᴴ) := by
      rw [← e1]
      ext (i | i) j <;> simp [stack]
    rw [innerSL_apply_apply, gradient_flatLoss hconv hG hL, real_inner_eq_pairing_factors,
      factors_flatten, hfac, factors_flatten]
    dsimp only
    rw [← pairing_stack, ← lift_mul_stack, e1, e2, Matrix.mul_add, Matrix.mul_smul,
      pairing_add_right, pairing_smul_right, pairing_mul_right, conjTranspose_conjTranspose,
      Matrix.mul_assoc, hWZ, Matrix.mul_zero, pairing_zero_right, zero_add, ← Matrix.mul_assoc,
      pairing_mul_conjTranspose_unit]
  have hcont : ContinuousAt (fun t : ℝ => frobeniusSq Z * pairing (stack A B)
      (lift lam (G (((factors x).1 + t • (A * Zᴴ)) * ((factors x).2 + t • (B * Zᴴ))ᴴ)) *
        stack A B)) 0 := by
    have hc1 : Continuous fun t : ℝ =>
        ((factors x).1 + t • (A * Zᴴ), (factors x).2 + t • (B * Zᴴ)) :=
      (continuous_const.fun_add (continuous_id.fun_smul continuous_const)).prodMk
        (continuous_const.fun_add (continuous_id.fun_smul continuous_const))
    have hc2 : Continuous fun t : ℝ => lift lam
        (G (((factors x).1 + t • (A * Zᴴ)) * ((factors x).2 + t • (B * Zᴴ))ᴴ)) :=
      (((continuous_lift lam).comp hG).comp continuous_mul_conjTranspose).comp hc1
    exact ((continuous_pairing continuous_const (hc2.matrix_mul continuous_const)).const_mul
      (frobeniusSq Z)).continuousAt
  have h2 : HasDerivAt (fun t : ℝ => innerSL ℝ h (gradient (flatLoss L lam) (x + t • h)))
      (frobeniusSq Z * pairing (stack A B)
        (lift lam (G ((factors x).1 * (factors x).2ᴴ)) * stack A B)) 0 := by
    rw [hasDerivAt_iff_tendsto_slope_zero]
    have hlim := hcont.tendsto.mono_left (nhdsWithin_le_nhds (s := {(0 : ℝ)}ᶜ))
    simp only [zero_smul, add_zero] at hlim
    refine hlim.congr' ?_
    filter_upwards [self_mem_nhdsWithin] with t ht
    rw [zero_add, key t, key 0, zero_mul, sub_zero, smul_eq_mul, ← mul_assoc,
      inv_mul_cancel₀ ht, one_mul]
  have h3 := h1.unique h2
  rw [innerSL_apply_apply] at h3
  exact h3

/-- The squared loss of the `1 × 1` kernel `1` at the origin of the factor space of width one,
where every column is free: `Z = 1`, `A = B = 1`. -/
example := inner_fderiv_gradient_flatLoss (k := Unit)
  (fitLoss_first_order (Set.univ : Set (Unit × Unit)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) 1 0 (Z := 1) (by ext i j; simp [factors])
  (by ext i j; simp [factors]) 1 1

/-- A critical point of the weight-decayed loss on the factor space is stationary:
`M [U; V] = 0`. -/
lemma lift_mul_stack_eq_zero_of_gradient_eq_zero
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    {lam : ℝ} {x : EuclideanSpace ℂ ((m ⊕ n) × k)} (hgrad : gradient (flatLoss L lam) x = 0) :
    lift lam (G ((factors x).1 * (factors x).2ᴴ)) * stack (factors x).1 (factors x).2 = 0 := by
  rw [gradient_flatLoss hconv hG hL] at hgrad
  rw [lift_mul_stack]
  ext i j
  have h := congrArg (fun y : EuclideanSpace ℂ ((m ⊕ n) × k) => y (i, j)) hgrad
  simpa [flatten] using h

/-- The origin of the factor space is a critical point: the squared loss of the `1 × 1` kernel
`1` at width one. -/
example := lift_mul_stack_eq_zero_of_gradient_eq_zero (k := Fin 1) (lam := 1) (x := 0)
  (fitLoss_first_order (Set.univ : Set (Fin 1 × Fin 1)) 1) (continuous_mask_sub Set.univ 1)
  (analyticAt_fitLoss Set.univ 1) (by
    rw [gradient_flatLoss (fitLoss_first_order Set.univ 1) (continuous_mask_sub Set.univ 1)
      (analyticAt_fitLoss Set.univ 1)]
    ext ⟨i | i, j⟩ <;> simp [flatten, stack, factors, Matrix.mul_apply])

end Transformer.Grokking.FlatSpectrum.Training
