module

public import Transformer.GainMatroid.Section2_K4Forms

/-!
# Every representation of `M(K₄)` is projectively equivalent to `I(K₄)`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4: "Over any field, a matrix representing the
matroid `M(K₄)` is projectively equivalent to the matrix `I(K₄)` shown below." The elements of
`k4Matroid` are `a, b, c = inl 0, inl 1, inl 2` on the triangle and
`a′, b′, c′ = inr 0, inr 1, inr 2` on the triad, and `k4Label` sends them to the printed columns of
`I(K₄)`.

Scaling the columns of a representation gives `a = b′ − c′`, `b = a′ − c′` and `c = a′ − b′`
(`IsK4Frame`). The independent columns `a′`, `b′`, `c′` then determine every vector of the row
space, so all matrices of this form have the same row space (`IsK4Frame.rowSpace_le`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {R R' F : Type*} [Field F]

/-- The columns on the triangle are the differences of the independent columns on the triad. -/
structure IsK4Frame (N : Matrix R (Fin 3 ⊕ Fin 3) F) : Prop where
  rel₀ : ∀ r, N r (.inl 0) = N r (.inr 1) - N r (.inr 2)
  rel₁ : ∀ r, N r (.inl 1) = N r (.inr 0) - N r (.inr 2)
  rel₂ : ∀ r, N r (.inl 2) = N r (.inr 0) - N r (.inr 1)
  indep : LinearIndepOn F (fun e r => N r e)
    (({.inr 0, .inr 1, .inr 2} : Finset (Fin 3 ⊕ Fin 3)) : Set (Fin 3 ⊕ Fin 3))

/-- Two matrices of this form have comparable row spaces, hence equal ones. -/
theorem IsK4Frame.rowSpace_le {N : Matrix R (Fin 3 ⊕ Fin 3) F} {N' : Matrix R' (Fin 3 ⊕ Fin 3) F}
    (h : IsK4Frame N) (h' : IsK4Frame N') : rowSpace N ≤ rowSpace N' := by
  refine rowSpace_le_of_forall_mem fun r => ?_
  obtain ⟨w, hw, hwv⟩ := exists_mem_rowSpace_eq_on h'.indep (N r)
  obtain ⟨ℓ, hℓ⟩ := exists_linearMap_col_eq hw
  have h₀ : w (.inr 0) = N r (.inr 0) := hwv _ (by simp)
  have h₁ : w (.inr 1) = N r (.inr 1) := hwv _ (by simp)
  have h₂ : w (.inr 2) = N r (.inr 2) := hwv _ (by simp)
  have hsub {e f g : Fin 3 ⊕ Fin 3} (hrel : ∀ r, N' r e = N' r f - N' r g) :
      w e = w f - w g := by
    rw [← hℓ e, show (fun r => N' r e) = (fun r => N' r f) - fun r => N' r g from funext hrel,
      map_sub, hℓ, hℓ]
  have h3 : ∀ i : Fin 3, i = 0 ∨ i = 1 ∨ i = 2 := by decide
  have heq : w = N r := by
    funext e
    rcases e with i | i
    · rcases h3 i with rfl | rfl | rfl
      · rw [hsub h'.rel₀, h₁, h₂, h.rel₀]
      · rw [hsub h'.rel₁, h₀, h₂, h.rel₁]
      · rw [hsub h'.rel₂, h₀, h₁, h.rel₂]
    · rcases h3 i with rfl | rfl | rfl <;> assumption
  exact heq ▸ hw

/-- The labelled `I(K₄)` has this form. -/
theorem iK4_isK4Frame : IsK4Frame ((iK4 : Matrix (Fin 4) (Fin 6) F).submatrix id k4Label) := by
  refine ⟨fun r => ?_, fun r => ?_, fun r => ?_, ?_⟩
  · fin_cases r <;> simp [iK4, k4Label]
  · fin_cases r <;> simp [iK4, k4Label]
  · fin_cases r <;> simp [iK4, k4Label]
  · have hc : IsTriadColumns ((iK4 : Matrix (Fin 4) (Fin 6) F).submatrix id k4Label)
        (.inr 0) (.inr 1) (.inr 2) :=
      ⟨Set.univ, Subtype.val, by simp, Subtype.val_injective, fun _ => ⟨rfl, rfl, rfl⟩,
        fun r hr => (hr ⟨⟨r, trivial⟩, rfl⟩).elim⟩
    simpa using hc.linearIndepOn (by decide) (by decide) (by decide)

private theorem k4_isCircuit {C : Finset (Fin 3 ⊕ Fin 3)}
    (h : ¬k4Indep C ∧ ∀ e ∈ C, k4Indep (C.erase e)) :
    k4Matroid.IsCircuit (C : Set (Fin 3 ⊕ Fin 3)) :=
  (matroidOfFinset_isCircuit_iff C).2 h

private theorem k4_triangle₁ : k4Matroid.IsCircuit {.inr 1, .inl 0, .inr 2} := by
  simpa using k4_isCircuit (C := {.inr 1, .inl 0, .inr 2}) (by decide +kernel)

private theorem k4_triangle₂ : k4Matroid.IsCircuit {.inr 0, .inl 1, .inr 2} := by
  simpa using k4_isCircuit (C := {.inr 0, .inl 1, .inr 2}) (by decide +kernel)

private theorem k4_triangle₃ : k4Matroid.IsCircuit {.inr 0, .inl 2, .inr 1} := by
  simpa using k4_isCircuit (C := {.inr 0, .inl 2, .inr 1}) (by decide +kernel)

private theorem k4_triangle₄ : k4Matroid.IsCircuit {.inl 0, .inl 2, .inl 1} := by
  simpa using k4_isCircuit (C := {.inl 0, .inl 2, .inl 1}) (by decide +kernel)

private theorem coeffs_zero {col : Fin 3 ⊕ Fin 3 → R → F}
    (h : LinearIndepOn F col
      (({.inr 0, .inr 1, .inr 2} : Finset (Fin 3 ⊕ Fin 3)) : Set (Fin 3 ⊕ Fin 3)))
    {α β γ : F} (hz : α • col (.inr 0) + β • col (.inr 1) + γ • col (.inr 2) = 0) :
    α = 0 ∧ β = 0 ∧ γ = 0 := by
  have hs := linearIndepOn_finset_iff.1 h (Sum.elim 0 ![α, β, γ]) (by
    rw [Finset.sum_insert (by decide), Finset.sum_pair (by decide)]
    simpa [add_assoc] using hz)
  exact ⟨hs (.inr 0) (by simp), hs (.inr 1) (by simp), hs (.inr 2) (by simp)⟩

/-- **§2.4.** "Over any field, a matrix representing the matroid `M(K₄)` is projectively equivalent
to the matrix `I(K₄)`." -/
theorem projEquiv_iK4 {A : Matrix R (Fin 3 ⊕ Fin 3) F} (hA : IsMatrixRepresentation k4Matroid A) :
    ProjEquiv A ((iK4 : Matrix (Fin 4) (Fin 6) F).submatrix id k4Label) := by
  obtain ⟨-, p₁, q₁, hp₁, hq₁, h₁⟩ :=
    exists_col_eq_of_isCircuit hA (by decide) (by decide) (by decide) k4_triangle₁
  obtain ⟨-, p₂, q₂, hp₂, hq₂, h₂⟩ :=
    exists_col_eq_of_isCircuit hA (by decide) (by decide) (by decide) k4_triangle₂
  obtain ⟨-, p₃, q₃, hp₃, -, h₃⟩ :=
    exists_col_eq_of_isCircuit hA (by decide) (by decide) (by decide) k4_triangle₃
  obtain ⟨-, p₄, q₄, -, -, h₄⟩ :=
    exists_col_eq_of_isCircuit hA (by decide) (by decide) (by decide) k4_triangle₄
  have e₁ (r : R) : A r (.inl 0) = p₁ * A r (.inr 1) + q₁ * A r (.inr 2) := by
    simpa using congrFun h₁ r
  have e₂ (r : R) : A r (.inl 1) = p₂ * A r (.inr 0) + q₂ * A r (.inr 2) := by
    simpa using congrFun h₂ r
  have e₃ (r : R) : A r (.inl 2) = p₃ * A r (.inr 0) + q₃ * A r (.inr 1) := by
    simpa using congrFun h₃ r
  have e₄ (r : R) : A r (.inl 2) = p₄ * A r (.inl 0) + q₄ * A r (.inl 1) := by
    simpa using congrFun h₄ r
  have hbase : LinearIndepOn F (fun e r => A r e)
      (({.inr 0, .inr 1, .inr 2} : Finset (Fin 3 ⊕ Fin 3)) : Set (Fin 3 ⊕ Fin 3)) :=
    (hA.2 _).1 ((matroidOfFinset_indep _).2 (by decide +kernel))
  obtain ⟨H₁, H₂, H₃⟩ := coeffs_zero hbase (α := p₃ - q₄ * p₂) (β := q₃ - p₄ * p₁)
    (γ := -(p₄ * q₁) - q₄ * q₂) (by
      funext r
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply]
      linear_combination -e₃ r + e₄ r + p₄ * e₁ r + q₄ * e₂ r)
  have hrel : q₃ * q₁ * p₂ + p₁ * q₂ * p₃ = 0 := by
    linear_combination (q₁ * p₂) * H₂ + (p₁ * q₂) * H₁ - (p₁ * p₂) * H₃
  have ht₁ : q₁ * p₂ / (q₂ * p₃) * p₃ = q₁ * p₂ / q₂ := by field_simp
  have ht₂ : q₁ * p₂ / (q₂ * p₃) * q₃ = -p₁ := by
    rw [div_mul_eq_mul_div, div_eq_iff (mul_ne_zero hq₂ hp₃)]
    linear_combination hrel
  have ht₃ : q₁ / q₂ * q₂ = q₁ := div_mul_cancel₀ q₁ hq₂
  let s : Fin 3 ⊕ Fin 3 → Fˣ := Sum.elim
    ![1, Units.mk0 (q₁ / q₂) (div_ne_zero hq₁ hq₂),
      Units.mk0 (q₁ * p₂ / (q₂ * p₃)) (div_ne_zero (mul_ne_zero hq₁ hp₂) (mul_ne_zero hq₂ hp₃))]
    ![Units.mk0 (q₁ * p₂ / q₂) (div_ne_zero (mul_ne_zero hq₁ hp₂) hq₂), Units.mk0 p₁ hp₁,
      Units.mk0 (-q₁) (neg_ne_zero.2 hq₁)]
  have hB : IsK4Frame (scaleCols s A) := by
    refine ⟨fun r => ?_, fun r => ?_, fun r => ?_,
      (ProjEquiv.linearIndepOn_iff ⟨s, rfl⟩ _).1 hbase⟩
    · change A r (.inl 0) * ((1 : Fˣ) : F) = A r (.inr 1) * p₁ - A r (.inr 2) * -q₁
      rw [e₁ r, Units.val_one]
      ring
    · change A r (.inl 1) * (q₁ / q₂) = A r (.inr 0) * (q₁ * p₂ / q₂) - A r (.inr 2) * -q₁
      rw [e₂ r]
      linear_combination A r (.inr 2) * ht₃
    · change A r (.inl 2) * (q₁ * p₂ / (q₂ * p₃)) = A r (.inr 0) * (q₁ * p₂ / q₂) -
        A r (.inr 1) * p₁
      rw [e₃ r]
      linear_combination A r (.inr 0) * ht₁ + A r (.inr 1) * ht₂
  exact ⟨s, le_antisymm (hB.rowSpace_le iK4_isK4Frame) (iK4_isK4Frame.rowSpace_le hB)⟩

set_option maxRecDepth 100000 in
/-- The labelled `I(K₄)` represents `M(K₄)` over `𝔽₂`. -/
theorem iK4_represents_binary : IsMatrixRepresentation k4Matroid
    ((iK4 : Matrix (Fin 4) (Fin 6) (ZMod 2)).submatrix id k4Label) := by
  refine ⟨rfl, fun I => ?_⟩
  obtain ⟨I, rfl⟩ := I.toFinite.exists_finset_coe
  refine (matroidOfFinset_indep I).trans (Iff.trans ?_ linearIndepOn_finset_iff.symm)
  revert I
  decide +kernel

example := (iK4_isK4Frame (F := ℚ)).rowSpace_le iK4_isK4Frame

example := projEquiv_iK4 iK4_represents_binary

end Transformer.GainMatroid
