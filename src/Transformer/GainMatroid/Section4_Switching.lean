module

public import Mathlib.Tactic.LinearCombination
public import Transformer.GainMatroid.Section2_Gains

/-!
# Switching and projective equivalence

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4.1. Proposition 4.1 (Zaslavsky): switching
equivalent `Fˣ`-gain functions and switching-and-scaling equivalent `F⁺`-gain functions have
projectively equivalent canonical matrices.

The source's proof of the lift case has a slip that does not affect the statement. With `s φ^η = ψ`
the first row of `T` must be `(s, -sη(v₁), …, -sη(vₙ))` and `S` must be the identity. The source
prints `+sη(vᵢ)` and `s₁₁ = 1/s`, and `T A_L(φ) S` is then not `A_L(ψ)`.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {V E F : Type*} [Fintype V] [DecidableEq V] [Fintype E] [DecidableEq E] [Field F]
  (G : OrientedGraph V E)

/-- The witnesses of Proposition 4.1 in the frame case: `T = diag(η)` and `S_e = η(tail e)⁻¹`. -/
theorem diagonal_mul_frameMatrix_mul_diagonal (η : V → Fˣ) (φ : E → Fˣ) :
    diagonal (fun w => (η w : F)) * frameMatrix G φ *
        diagonal (fun e => (((η (G.tail e))⁻¹ : Fˣ) : F)) = frameMatrix G (switch G η φ) := by
  ext w e
  rw [mul_diagonal, diagonal_mul, frameMatrix_apply, frameMatrix_apply]
  simp only [switch_apply, Units.val_mul, Units.val_inv_eq_inv_val]
  have ht₀ : (η (G.tail e) : F) ≠ 0 := (η _).ne_zero
  by_cases ht : w = G.tail e
  · subst ht
    by_cases hh : G.tail e = G.head e
    · rw [ite_eq_left rfl, ite_eq_left hh, ← hh]
      field_simp
    · rw [ite_eq_left rfl, ite_eq_right hh]
      field_simp
      simp
  · by_cases hh : w = G.head e
    · subst hh
      rw [ite_eq_right ht, ite_eq_left rfl]
      field_simp
      simp
    · simp [ht, hh]

/-- **Proposition 4.1** (Zaslavsky), frame case: switching equivalent `Fˣ`-gain functions have
projectively equivalent canonical frame matrices. -/
theorem projEquiv_frameMatrix_of_switchEquiv {φ ψ : E → Fˣ} (h : SwitchEquiv G φ ψ) :
    ProjEquiv (frameMatrix G φ) (frameMatrix G ψ) := by
  obtain ⟨η, rfl⟩ := h
  refine (projEquiv_iff_exists_mul_mul_diagonal _ _).2
    ⟨diagonal fun w => (η w : F), fun e => (η (G.tail e))⁻¹, ?_,
      diagonal_mul_frameMatrix_mul_diagonal G η φ⟩
  rw [Matrix.isUnit_iff_isUnit_det, det_diagonal, isUnit_iff_ne_zero]
  exact Finset.prod_ne_zero_iff.2 fun w _ => (η w).ne_zero

/-- The matrix `T` of the lift case: first row `(a, τ)`, first column `(a, 0, …, 0)` and the
identity on the remaining rows and columns. -/
def liftRowOp (a : F) (τ : V → F) : Matrix (Option V) (Option V) F :=
  Matrix.of fun r c =>
    match r, c with
    | none, none => a
    | none, some w => τ w
    | some _, none => 0
    | some v, some w => if v = w then 1 else 0

omit [Fintype E] [DecidableEq E] in
theorem liftRowOp_mul_none {X : Type*} (a : F) (τ : V → F) (M : Matrix (Option V) X F) (x : X) :
    (liftRowOp a τ * M) none x = a * M none x + ∑ w, τ w * M (some w) x := by
  simp [Matrix.mul_apply, Fintype.sum_option, liftRowOp]

omit [Fintype E] [DecidableEq E] in
theorem liftRowOp_mul_some {X : Type*} (a : F) (τ : V → F) (M : Matrix (Option V) X F) (v : V)
    (x : X) : (liftRowOp a τ * M) (some v) x = M (some v) x := by
  simp [Matrix.mul_apply, Fintype.sum_option, liftRowOp]

omit [Fintype E] [DecidableEq E] in
/-- `T` is nonsingular when its corner is nonzero. -/
theorem isUnit_liftRowOp {a : F} (ha : a ≠ 0) (τ : V → F) : IsUnit (liftRowOp a τ) := by
  refine IsUnit.of_mul_eq_one (liftRowOp a⁻¹ fun w => -(a⁻¹ * τ w)) ?_
  ext r c
  cases r with
  | none =>
    rw [liftRowOp_mul_none]
    cases c with
    | none => simp [liftRowOp, ha]
    | some c => simp [liftRowOp, ha]
  | some v =>
    rw [liftRowOp_mul_some]
    cases c with
    | none => simp [liftRowOp]
    | some c => simp [liftRowOp, Matrix.one_apply]

omit [Fintype E] [DecidableEq E] in
theorem sum_mul_liftMatrix_some (τ : V → F) (ψ : E → F) (e : E) :
    ∑ w, τ w * liftMatrix G ψ (some w) e = τ (G.tail e) - τ (G.head e) := by
  simp [liftMatrix_some, mul_sub, Finset.sum_sub_distrib]

omit [Fintype E] [DecidableEq E] in
/-- The witness of Proposition 4.1 in the lift case, with the sign the source misprints. -/
theorem liftRowOp_mul_liftMatrix (a : Fˣ) (η : V → F) (φ : E → F) :
    liftRowOp (a : F) (fun v => -((a : F) * η v)) * liftMatrix G φ =
      liftMatrix G ((a : F) • addSwitch G η φ) := by
  ext r e
  cases r with
  | none =>
    rw [liftRowOp_mul_none, sum_mul_liftMatrix_some]
    simp only [liftMatrix_none, Pi.smul_apply, addSwitch_apply, smul_eq_mul]
    ring
  | some v =>
    rw [liftRowOp_mul_some]
    rfl

/-- **Proposition 4.1** (Zaslavsky), lift case: switching-and-scaling equivalent `F⁺`-gain functions
have projectively equivalent canonical lift matrices. -/
theorem projEquiv_liftMatrix_of_switchScaleEquiv {φ ψ : E → F} (h : SwitchScaleEquiv G φ ψ) :
    ProjEquiv (liftMatrix G φ) (liftMatrix G ψ) := by
  obtain ⟨η, a, rfl⟩ := h
  refine (projEquiv_iff_exists_mul_mul_diagonal _ _).2
    ⟨liftRowOp (a : F) (fun v => -((a : F) * η v)), 1, isUnit_liftRowOp a.ne_zero _, ?_⟩
  rw [liftRowOp_mul_liftMatrix]
  simp

example (g : ℚˣ) : ProjEquiv (frameMatrix linkGraph fun _ => g) (frameMatrix linkGraph fun _ => g) :=
  projEquiv_frameMatrix_of_switchEquiv _ ((switchEquiv_equivalence _).refl _)

example (g : ℚ) : ProjEquiv (liftMatrix linkGraph fun _ => g) (liftMatrix linkGraph fun _ => g) :=
  projEquiv_liftMatrix_of_switchScaleEquiv _ ((switchScaleEquiv_equivalence _).refl _)

example : IsUnit (liftRowOp (2 : ℚ) fun _ : Fin 3 => 5) :=
  isUnit_liftRowOp two_ne_zero _

end Transformer.GainMatroid
