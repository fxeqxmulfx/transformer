module

public import Transformer.GainMatroid.Section4_TwoVertices

/-!
# Lemmas 4.11 and 4.12: the biased graphs `𝖴₂` and `𝖴₃`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4.1. `𝖴₂` and `𝖴₃` are `U₂` and `U₃` with every
cycle unbalanced. Two `Fˣ`-realizations of `𝖴₂` have projectively equivalent frame matrices exactly
when they give the 2-cycle `e₃e₄` the same gain (Lemma 4.11). Two `F⁺`-realizations of `𝖴₃` have
projectively equivalent lift matrices exactly when their restrictions to the theta `{e₂, e₃, e₄}`
are switching-and-scaling equivalent (Lemma 4.12). Edges are indexed from `0`, so `e₁` is `0`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {F : Type*} [Field F]

private theorem eq_zero_of_loop_entry {t g s : F} (hg : (1 : F) - g ≠ 0) (hs : s ≠ 0)
    (h : (t - t * g) * s = 0) : t = 0 := by
  have h' : t * ((1 - g) * s) = 0 := by linear_combination h
  exact (mul_eq_zero.1 h').resolve_right (mul_ne_zero hg hs)

private theorem one_sub_val_ne_zero {g : Fˣ} (h : g ≠ 1) : (1 : F) - g ≠ 0 :=
  sub_ne_zero.2 fun hc => h (Units.ext hc.symm)

/-- **Lemma 4.11.** Two `Fˣ`-realizations of `𝖴₂` have projectively equivalent frame matrices if and
only if they agree on the closed walk `e₃e₄`. -/
theorem U2_projEquiv_iff {φ ψ : Fin 4 → Fˣ} (hφ : U2.contrabalanced.Realizes φ)
    (hψ : U2.contrabalanced.Realizes ψ) :
    ProjEquiv (frameMatrix U2 φ) (frameMatrix U2 ψ) ↔ φ 2 * φ 3 = ψ 2 * ψ 3 := by
  constructor
  · intro h
    obtain ⟨T, s, -, hT⟩ := (projEquiv_iff_exists_mul_mul_diagonal _ _).1 h
    have hentry : ∀ r e, (T r (U2.tail e) - T r (U2.head e) * φ e) * s e =
        frameMatrix U2 ψ r e := by
      intro r e
      have := congrFun (congrFun hT r) e
      rwa [Matrix.mul_diagonal, mul_frameMatrix_apply] at this
    have h10 : T 1 0 = 0 := eq_zero_of_loop_entry
      (one_sub_val_ne_zero (contrabalanced_realizes_loop_ne hφ (e := 0) rfl)) (s 0).ne_zero
      ((hentry 1 0).trans (by simp [frameMatrix_apply, U2]))
    have h01 : T 0 1 = 0 := eq_zero_of_loop_entry
      (one_sub_val_ne_zero (contrabalanced_realizes_loop_ne hφ (e := 1) rfl)) (s 1).ne_zero
      ((hentry 0 1).trans (by simp [frameMatrix_apply, U2]))
    have e02 : (T 0 0 - T 0 1 * φ 2) * s 2 = 1 :=
      (hentry 0 2).trans (by simp [frameMatrix_apply, U2])
    have e12 : (T 1 0 - T 1 1 * φ 2) * s 2 = -ψ 2 :=
      (hentry 1 2).trans (by simp [frameMatrix_apply, U2])
    have e03 : (T 0 1 - T 0 0 * φ 3) * s 3 = -ψ 3 :=
      (hentry 0 3).trans (by simp [frameMatrix_apply, U2])
    have e13 : (T 1 1 - T 1 0 * φ 3) * s 3 = 1 :=
      (hentry 1 3).trans (by simp [frameMatrix_apply, U2])
    rw [h01] at e02 e03
    rw [h10] at e12 e13
    apply Units.ext
    rw [Units.val_mul, Units.val_mul]
    linear_combination (-(φ 2 : F) * φ 3 * T 1 1 * s 3) * e02 - (ψ 3 : F) * e12 -
      (T 1 1 * φ 2 * s 2) * e03 - (φ 2 : F) * φ 3 * e13
  · intro h
    refine (projEquiv_frameMatrix_of_switchEquiv U2 ⟨![1, (φ 2)⁻¹ * ψ 2], rfl⟩).trans
      (frame_projEquiv_of_eq_off_loops fun e => ?_)
    have hl₀ := contrabalanced_realizes_loop_ne hφ (e := 0) rfl
    have hl₁ := contrabalanced_realizes_loop_ne hφ (e := 1) rfl
    have hr₀ := contrabalanced_realizes_loop_ne hψ (e := 0) rfl
    have hr₁ := contrabalanced_realizes_loop_ne hψ (e := 1) rfl
    fin_cases e
    · exact Or.inl ⟨rfl, by simpa [switch_apply, U2] using hl₀, hr₀⟩
    · refine Or.inl ⟨rfl, ?_, hr₁⟩
      have hs : switch U2 ![1, (φ 2)⁻¹ * ψ 2] φ 1 = φ 1 := by
        change ((φ 2)⁻¹ * ψ 2)⁻¹ * φ 1 * ((φ 2)⁻¹ * ψ 2) = φ 1
        rw [mul_comm, mul_inv_cancel_left]
      exact hs ▸ hl₁
    · refine Or.inr ?_
      simp [switch_apply, U2]
    · refine Or.inr ?_
      change ((φ 2)⁻¹ * ψ 2)⁻¹ * φ 3 * 1 = ψ 3
      rw [mul_one, mul_inv, inv_inv]
      calc (φ 2 * (ψ 2)⁻¹) * φ 3 = (ψ 2)⁻¹ * (φ 2 * φ 3) := by ac_rfl
        _ = ψ 3 := by rw [h, inv_mul_cancel_left]

/-- **Lemma 4.12.** Two `F⁺`-realizations of `𝖴₃` have projectively equivalent lift matrices if and
only if their restrictions to `{e₂, e₃, e₄}` are switching-and-scaling equivalent. -/
theorem U3_projEquiv_iff {φ ψ : Fin 4 → F} (hφ : U3.contrabalanced.AddRealizes φ)
    (hψ : U3.contrabalanced.AddRealizes ψ) :
    ProjEquiv (liftMatrix U3 φ) (liftMatrix U3 ψ) ↔
      SwitchScaleEquiv (U3.delete {0}) (fun e => φ e) (fun e => ψ e) := by
  have hφ0 : φ 0 ≠ 0 := contrabalanced_addRealizes_loop_ne hφ (e := 0) rfl
  have hψ0 : ψ 0 ≠ 0 := contrabalanced_addRealizes_loop_ne hψ (e := 0) rfl
  constructor
  · intro h
    obtain ⟨T, s, -, hT⟩ := (projEquiv_iff_exists_mul_mul_diagonal _ _).1 h
    have hentry : ∀ r e, (T r none * φ e + T r (some (U3.tail e)) - T r (some (U3.head e))) * s e =
        liftMatrix U3 ψ r e := by
      intro r e
      have := congrFun (congrFun hT r) e
      rwa [Matrix.mul_diagonal, mul_liftMatrix_apply] at this
    have hq : T (some 0) none = 0 := by
      have e : (T (some 0) none * φ 0 + T (some 0) (some 0) - T (some 0) (some 0)) * s 0 = 0 :=
        (hentry (some 0) 0).trans (liftMatrix_loop U3 ψ rfl 0)
      have e' : T (some 0) none * (φ 0 * s 0) = 0 := by linear_combination e
      exact (mul_eq_zero.1 e').resolve_right (mul_ne_zero hφ0 (s 0).ne_zero)
    have hp : T none none ≠ 0 := fun h0 => hψ0 (by
      have e : (T none none * φ 0 + T none (some 0) - T none (some 0)) * s 0 = ψ 0 := hentry none 0
      rw [← e, h0]
      ring)
    have A1 : (T (some 0) none * φ 1 + T (some 0) (some 0) - T (some 0) (some 1)) * s 1 = 1 :=
      (hentry (some 0) 1).trans (by simp [liftMatrix_some, U3])
    have A2 : (T (some 0) none * φ 2 + T (some 0) (some 0) - T (some 0) (some 1)) * s 2 = 1 :=
      (hentry (some 0) 2).trans (by simp [liftMatrix_some, U3])
    have A3 : (T (some 0) none * φ 3 + T (some 0) (some 1) - T (some 0) (some 0)) * s 3 = -1 :=
      (hentry (some 0) 3).trans (by simp [liftMatrix_some, U3])
    rw [hq] at A1 A2 A3
    have B1 : (T none none * φ 1 + T none (some 0) - T none (some 1)) * s 1 = ψ 1 := hentry none 1
    have B2 : (T none none * φ 2 + T none (some 0) - T none (some 1)) * s 2 = ψ 2 := hentry none 2
    have B3 : (T none none * φ 3 + T none (some 1) - T none (some 0)) * s 3 = ψ 3 := hentry none 3
    have h12 : (s 1 : F) = s 2 := by linear_combination (s 2 : F) * A1 - (s 1 : F) * A2
    have h13 : (s 1 : F) = s 3 := by linear_combination (s 3 : F) * A1 + (s 1 : F) * A3
    rw [← h12] at B2
    rw [← h13] at B3
    refine ⟨![0, (T none (some 0) - T none (some 1)) / T none none],
      s 1 * Units.mk0 _ hp, ?_⟩
    have key : ∀ e : Fin 4, e ≠ 0 → (s 1 : F) * T none none *
        (-(![0, (T none (some 0) - T none (some 1)) / T none none] : Fin 2 → F) (U3.tail e) + φ e +
          (![0, (T none (some 0) - T none (some 1)) / T none none] : Fin 2 → F) (U3.head e)) =
        ψ e := by
      intro e he
      fin_cases e
      · exact absurd rfl he
      · change (s 1 : F) * T none none *
          (-0 + φ 1 + (T none (some 0) - T none (some 1)) / T none none) = ψ 1
        rw [← B1]
        field_simp
        ring
      · change (s 1 : F) * T none none *
          (-0 + φ 2 + (T none (some 0) - T none (some 1)) / T none none) = ψ 2
        rw [← B2]
        field_simp
        ring
      · change (s 1 : F) * T none none *
          (-((T none (some 0) - T none (some 1)) / T none none) + φ 3 + 0) = ψ 3
        rw [← B3]
        field_simp
        ring
    funext ⟨e, he⟩
    exact key e he
  · rintro ⟨η, a, h⟩
    refine (projEquiv_liftMatrix_of_switchScaleEquiv U3 ⟨η, a, rfl⟩).trans
      (lift_projEquiv_of_eq_off_loops fun e => ?_)
    by_cases he : e = 0
    · subst he
      refine Or.inl ⟨rfl, ?_, hψ0⟩
      change (a : F) * (-η 0 + φ 0 + η 0) ≠ 0
      rw [show (a : F) * (-η 0 + φ 0 + η 0) = a * φ 0 by ring]
      exact mul_ne_zero a.ne_zero hφ0
    · exact Or.inr (congrFun h ⟨e, he⟩)

example : ProjEquiv (frameMatrix U2 U2Gain₁) (frameMatrix U2 U2Gain₂) :=
  (U2_projEquiv_iff U2Gain₁_realizes U2Gain₂_realizes).2 (by decide)

example : ProjEquiv (liftMatrix U3 U3Gain₁) (liftMatrix U3 U3Gain₂) ↔
    SwitchScaleEquiv (U3.delete {0}) (fun e => U3Gain₁ e) (fun e => U3Gain₂ e) :=
  U3_projEquiv_iff U3Gain₁_realizes U3Gain₂_realizes

end Transformer.GainMatroid
