module

public import Transformer.GainMatroid.Section4_Switching

/-!
# Switching as projective equivalence through special matrices

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4.1, Lemma 4.2: on a loopless graph, switching
equivalence is projective equivalence through diagonal `T` and `S` on canonical frame matrices, and
switching-and-scaling equivalence is projective equivalence through a `T` that is the identity
outside the row `v₀` on canonical lift matrices.

The source assumes that the two matrices are projectively equivalent. The existence of `T` and `S`
already says so, and the statements below do not assume it. In the proof of part (ii) the switching
function must be `η(vᵢ) = -T_{v₀vᵢ} / T_{v₀v₀}`; the source prints `T_{v₀vᵢ}`.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {V E F : Type*} [Fintype V] [DecidableEq V] [Fintype E] [DecidableEq E] [Field F]
  (G : OrientedGraph V E)

/-- **Lemma 4.2 (i).** On a loopless graph, `Fˣ`-gain functions are switching equivalent exactly
when `T A_F(φ) S = A_F(ψ)` for nonsingular diagonal `T` and `S`. -/
theorem switchEquiv_iff_exists_diagonal (hG : G.Loopless) (φ ψ : E → Fˣ) :
    SwitchEquiv G φ ψ ↔ ∃ (T : Matrix V V F) (S : Matrix E E F),
      T.IsDiag ∧ IsUnit T ∧ S.IsDiag ∧ IsUnit S ∧ T * frameMatrix G φ * S = frameMatrix G ψ := by
  constructor
  · rintro ⟨η, rfl⟩
    refine ⟨diagonal fun w => (η w : F), diagonal fun e => (((η (G.tail e))⁻¹ : Fˣ) : F),
      isDiag_diagonal _, ?_, isDiag_diagonal _, ?_, diagonal_mul_frameMatrix_mul_diagonal G η φ⟩
    · rw [Matrix.isUnit_iff_isUnit_det, det_diagonal, isUnit_iff_ne_zero]
      exact Finset.prod_ne_zero_iff.2 fun w _ => (η w).ne_zero
    · rw [Matrix.isUnit_iff_isUnit_det, det_diagonal, isUnit_iff_ne_zero]
      exact Finset.prod_ne_zero_iff.2 fun e _ => Units.ne_zero _
  · rintro ⟨T, S, hTd, hTu, hSd, hSu, hTS⟩
    obtain ⟨τ, rfl⟩ := exists_units_of_isDiag_of_isUnit hTd hTu
    obtain ⟨s, rfl⟩ := exists_units_of_isDiag_of_isUnit hSd hSu
    refine ⟨τ, ?_⟩
    funext e
    have hentry : ∀ w, (τ w : F) * frameMatrix G φ w e * (s e : F) = frameMatrix G ψ w e := by
      intro w
      have := congrFun (congrFun hTS w) e
      rwa [mul_diagonal, diagonal_mul] at this
    have ht := hentry (G.tail e)
    have hh := hentry (G.head e)
    rw [frameMatrix_tail G φ (hG e), frameMatrix_tail G ψ (hG e), mul_one] at ht
    rw [frameMatrix_head G φ (hG e), frameMatrix_head G ψ (hG e)] at hh
    have hs : (s e : F) = (τ (G.tail e) : F)⁻¹ := eq_inv_of_mul_eq_one_right ht
    rw [hs] at hh
    apply Units.ext
    simp only [switch_apply, Units.val_mul, Units.val_inv_eq_inv_val]
    have hg : (τ (G.head e) : F) * (φ e : F) * (τ (G.tail e) : F)⁻¹ = (ψ e : F) := by
      linear_combination -hh
    rw [← hg]
    ring

/-- **Lemma 4.2 (ii).** On a loopless graph, `F⁺`-gain functions are switching-and-scaling equivalent
exactly when `T A_L(φ) S = A_L(ψ)` for a nonsingular `T` and a nonsingular diagonal `S`, where
removing the row and column `v₀` of `T` leaves an identity matrix and all but the first entry of
column `v₀` vanish. -/
theorem switchScaleEquiv_iff_exists_block (hG : G.Loopless) (φ ψ : E → F) :
    SwitchScaleEquiv G φ ψ ↔ ∃ (T : Matrix (Option V) (Option V) F) (S : Matrix E E F),
      IsUnit T ∧ S.IsDiag ∧ IsUnit S ∧ (∀ v w : V, T (some v) (some w) = if v = w then 1 else 0) ∧
        (∀ v : V, T (some v) none = 0) ∧ T * liftMatrix G φ * S = liftMatrix G ψ := by
  constructor
  · rintro ⟨η, a, rfl⟩
    refine ⟨liftRowOp (a : F) (fun v => -((a : F) * η v)), 1, isUnit_liftRowOp a.ne_zero _,
      isDiag_one, isUnit_one, fun v w => rfl, fun v => rfl, ?_⟩
    rw [liftRowOp_mul_liftMatrix, Matrix.mul_one]
  · rintro ⟨T, S, hTu, hSd, hSu, hT₁, hT₂, hTS⟩
    obtain ⟨s, rfl⟩ := exists_units_of_isDiag_of_isUnit hSd hSu
    have hTeq : T = liftRowOp (T none none) (fun w => T none (some w)) := by
      ext r c
      cases r <;> cases c <;> simp [liftRowOp, hT₁, hT₂]
    have ha : T none none ≠ 0 := by
      intro h0
      have hdet : T.det = 0 := det_eq_zero_of_column_eq_zero none fun i => by
        cases i with
        | none => exact h0
        | some v => exact hT₂ v
      exact ((Matrix.isUnit_iff_isUnit_det T).1 hTu).ne_zero hdet
    set a := T none none with ha_def
    set τ : V → F := fun w => T none (some w) with hτ_def
    have hentry : ∀ r e, (T * liftMatrix G φ) r e * (s e : F) = liftMatrix G ψ r e := by
      intro r e
      have := congrFun (congrFun hTS r) e
      rwa [mul_diagonal] at this
    have hs : ∀ e, (s e : F) = 1 := by
      intro e
      have h := hentry (some (G.tail e)) e
      rw [hTeq, liftRowOp_mul_some, liftMatrix_tail G φ (hG e), liftMatrix_tail G ψ (hG e),
        one_mul] at h
      exact h
    refine ⟨fun w => -(a⁻¹ * τ w), Units.mk0 a ha, ?_⟩
    funext e
    have h := hentry none e
    rw [hs e, mul_one, hTeq, liftRowOp_mul_none, sum_mul_liftMatrix_some, liftMatrix_none,
      liftMatrix_none] at h
    simp only [Pi.smul_apply, addSwitch_apply, smul_eq_mul, Units.val_mk0]
    rw [← h]
    field_simp
    ring

example (g : ℚˣ) : ∃ (T : Matrix (Fin 3) (Fin 3) ℚ) (S : Matrix Unit Unit ℚ), T.IsDiag ∧ IsUnit T ∧
    S.IsDiag ∧ IsUnit S ∧
      T * frameMatrix linkGraph (fun _ => g) * S = frameMatrix linkGraph fun _ => g :=
  (switchEquiv_iff_exists_diagonal linkGraph (fun _ => linkGraph_isLink) _ _).1
    ((switchEquiv_equivalence _).refl _)

example (g : ℚ) : ∃ (T : Matrix (Option (Fin 3)) (Option (Fin 3)) ℚ) (S : Matrix Unit Unit ℚ),
    IsUnit T ∧ S.IsDiag ∧ IsUnit S ∧ (∀ v w : Fin 3, T (some v) (some w) = if v = w then 1 else 0) ∧
      (∀ v : Fin 3, T (some v) none = 0) ∧
        T * liftMatrix linkGraph (fun _ => g) * S = liftMatrix linkGraph fun _ => g :=
  (switchScaleEquiv_iff_exists_block linkGraph (fun _ => linkGraph_isLink) _ _).1
    ((switchScaleEquiv_equivalence _).refl _)

end Transformer.GainMatroid
