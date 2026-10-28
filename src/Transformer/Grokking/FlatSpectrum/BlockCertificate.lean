module

public import Transformer.Grokking.FlatSpectrum.BlockWitnessCases
public import Transformer.Grokking.FlatSpectrum.CrossCertificate

/-!
# The certificate of a block, and the least norm off a block

Source: `review/flat_spectrum.md`, *The failure margin* (added). For `R ⊆ m` and `C ⊆ n` with
`k = #R` and `l = #C`, the matrix `Z` equal to `c₁` on `R × Cᶜ`, to `c₂` on `Rᶜ × C` and zero
elsewhere (`blockCert`) maps `v` to `c₁ ∑_{j ∉ C} v j` in the rows of `R` and to
`c₂ ∑_{j ∈ C} v j` in the others (`blockCert_mulVec`). By Cauchy–Schwarz it is a contraction
once `c₁² k (n − l) ≤ 1` and `c₂² l (m − k) ≤ 1` (`isContraction_blockCert`). With
`c₁ = 1/√(k (n − l))` and `c₂ = 1/√(l (m − k))` it pairs with every `X` equal to `1` on
`R × Cᶜ` and on `Rᶜ × C` to `√(k (n − l)) + √(l (m − k))`, so every such `X` has
`‖X‖_S1 ≥ √(k (n − l)) + √(l (m − k))` (`sqrt_add_sqrt_le_nuclearNorm_of_block`). For nonempty
`R`, `C` with `m n ≤ k n + l m`, the witness `J − t 1_R 1_Cᵀ` at its best `t` reaches this
bound, so it is the least nuclear norm of the matrices equal to `J` off `R × C`
(`isLeast_nuclearNorm_compl_block`). For `R = {i₀}ᶜ` and `C = {j₀}ᶜ`, `Z` is `crossCert`
(`blockCert_compl_singleton`) and the least is item 9's `2√(N − 1)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- The certificate of the block `R × C`: `c₁` on `R × Cᶜ`, `c₂` on `Rᶜ × C`, zero
elsewhere. -/
noncomputable def blockCert (R : Finset m) (C : Finset n) (c₁ c₂ : ℝ) : Matrix m n ℂ :=
  of fun i j => if i ∈ R then (if j ∈ C then 0 else (c₁ : ℂ)) else (if j ∈ C then (c₂ : ℂ) else 0)

omit [Fintype m] in
/-- Row `i` of `Z v` is `c₁ ∑_{j ∉ C} v j` for `i ∈ R` and `c₂ ∑_{j ∈ C} v j` otherwise. -/
lemma blockCert_mulVec (R : Finset m) (C : Finset n) (c₁ c₂ : ℝ) (v : n → ℂ) (i : m) :
    (blockCert R C c₁ c₂ *ᵥ v) i =
      if i ∈ R then (c₁ : ℂ) * ∑ j ∈ Cᶜ, v j else (c₂ : ℂ) * ∑ j ∈ C, v j := by
  rw [mulVec, dotProduct, ← sum_add_sum_compl C, mul_sum, mul_sum]
  by_cases hi : i ∈ R
  · rw [ite_eq_left hi, sum_eq_zero fun j hj => by simp [blockCert, hi, hj], zero_add]
    exact sum_congr rfl fun j hj => by simp [blockCert, hi, mem_compl.mp hj]
  · rw [ite_eq_right hi, sum_eq_zero (s := Cᶜ) fun j hj => by
      simp [blockCert, hi, mem_compl.mp hj], add_zero]
    exact sum_congr rfl fun j hj => by simp [blockCert, hi, hj]

example := blockCert_mulVec (m := Fin 2) (n := Fin 2) {0} {0} 1 1 0 0

/-- `#Cᶜ = n − #C` in any additive group with one. -/
lemma card_compl_cast {K : Type*} [AddGroupWithOne K] (C : Finset n) :
    ((#Cᶜ : ℕ) : K) = Fintype.card n - #C := by
  rw [card_compl, Nat.cast_sub (card_le_univ C)]

example := card_compl_cast (K := ℝ) (∅ : Finset (Fin 2))

/-- For `R = {i₀}ᶜ`, `C = {j₀}ᶜ` and `c₁ = c₂ = c` the block certificate is item 9's cross
certificate. -/
lemma blockCert_compl_singleton (i₀ j₀ : n) (c : ℝ) :
    blockCert ({i₀}ᶜ : Finset n) ({j₀}ᶜ : Finset n) c c = crossCert i₀ j₀ c := by
  ext i j
  by_cases hi : i = i₀ <;> by_cases hj : j = j₀ <;> simp [blockCert, crossCert, hi, hj]

example := blockCert_compl_singleton (n := Fin 2) 0 1 1

/-- **The block certificate is a contraction** once `c₁² k (n − l) ≤ 1` and
`c₂² l (m − k) ≤ 1`, for `k = #R` and `l = #C`. -/
lemma isContraction_blockCert (R : Finset m) (C : Finset n) {c₁ c₂ : ℝ}
    (h₁ : c₁ ^ 2 * (#R * (Fintype.card n - #C)) ≤ 1)
    (h₂ : c₂ ^ 2 * (#C * (Fintype.card m - #R)) ≤ 1) :
    IsContraction (blockCert R C c₁ c₂) := by
  refine isContraction_of_forall_sum_norm_sq_le fun v => ?_
  have hcs (s : Finset n) : ‖∑ j ∈ s, v j‖ ^ 2 ≤ #s * ∑ j ∈ s, ‖v j‖ ^ 2 :=
    (pow_le_pow_left₀ (norm_nonneg _) (norm_sum_le _ _) 2).trans sq_sum_le_card_mul_sum_sq
  have hrow (i : m) : ‖(blockCert R C c₁ c₂ *ᵥ v) i‖ ^ 2 = if i ∈ R
      then c₁ ^ 2 * ‖∑ j ∈ Cᶜ, v j‖ ^ 2 else c₂ ^ 2 * ‖∑ j ∈ C, v j‖ ^ 2 := by
    rw [blockCert_mulVec]
    split_ifs <;> rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, mul_pow, sq_abs]
  rw [sum_congr rfl fun i _ => hrow i, ← sum_add_sum_compl R,
    sum_congr rfl fun i hi => ite_eq_left hi,
    sum_congr rfl fun i hi => ite_eq_right (mem_compl.mp hi), sum_const, sum_const, nsmul_eq_mul,
    nsmul_eq_mul, ← sum_add_sum_compl C (fun j => ‖v j‖ ^ 2), card_compl_cast R]
  have hA := hcs Cᶜ
  have hB := hcs C
  rw [card_compl_cast C] at hA
  have hSA := sum_nonneg fun j (_ : j ∈ Cᶜ) => sq_nonneg ‖v j‖
  have hSB := sum_nonneg fun j (_ : j ∈ C) => sq_nonneg ‖v j‖
  have hk : (0 : ℝ) ≤ #R := Nat.cast_nonneg _
  have hmk : (0 : ℝ) ≤ Fintype.card m - #R := sub_nonneg.mpr (by exact_mod_cast card_le_univ R)
  nlinarith [mul_le_mul_of_nonneg_left hA (mul_nonneg hk (sq_nonneg c₁)),
    mul_le_mul_of_nonneg_left hB (mul_nonneg hmk (sq_nonneg c₂)),
    mul_le_mul_of_nonneg_right h₁ hSA, mul_le_mul_of_nonneg_right h₂ hSB]

example := isContraction_blockCert (m := Fin 2) (n := Fin 2) {0} {0} (c₁ := 1) (c₂ := 1)
  (by norm_num) (by norm_num)

/-- `(1/√x)² x ≤ 1` for `x ≥ 0`. -/
lemma inv_sqrt_sq_mul_le_one {x : ℝ} (hx : 0 ≤ x) : (√x)⁻¹ ^ 2 * x ≤ 1 := by
  rw [inv_pow, Real.sq_sqrt hx]
  rcases hx.eq_or_lt with h | h
  · simp [← h]
  · rw [inv_mul_cancel₀ h.ne']

example := inv_sqrt_sq_mul_le_one (x := 1) zero_le_one

/-- The pairing of the block certificate with every `X` equal to `1` on `R × Cᶜ` and on
`Rᶜ × C`: `tr (Zᴴ X) = c₁ k (n − l) + c₂ l (m − k)`. -/
lemma trace_blockCert_mul {R : Finset m} {C : Finset n} {X : Matrix m n ℂ}
    (h₁ : ∀ i ∈ R, ∀ j ∉ C, X i j = 1) (h₂ : ∀ i ∉ R, ∀ j ∈ C, X i j = 1) (c₁ c₂ : ℝ) :
    ((blockCert R C c₁ c₂)ᴴ * X).trace =
      ((c₁ * (#R * (Fintype.card n - #C)) + c₂ * (#C * (Fintype.card m - #R)) : ℝ) : ℂ) := by
  have hrow (i : m) : ∑ j, star (blockCert R C c₁ c₂ i j) * X i j =
      if i ∈ R then (c₁ : ℂ) * #Cᶜ else (c₂ : ℂ) * #C := by
    rw [← sum_add_sum_compl C]
    by_cases hi : i ∈ R
    · rw [ite_eq_left hi, sum_eq_zero fun j hj => by simp [blockCert, hi, hj], zero_add,
        sum_congr rfl (g := fun _ => (c₁ : ℂ)) fun j hj => by
          simp [blockCert, hi, mem_compl.mp hj, h₁ i hi j (mem_compl.mp hj)],
        sum_const, nsmul_eq_mul, mul_comm]
    · rw [ite_eq_right hi, sum_eq_zero (s := Cᶜ) fun j hj => by
          simp [blockCert, hi, mem_compl.mp hj], add_zero,
        sum_congr rfl (g := fun _ => (c₂ : ℂ)) fun j hj => by
          simp [blockCert, hi, hj, h₂ i hi j hj], sum_const, nsmul_eq_mul, mul_comm]
  rw [trace_conjTranspose_mul, sum_congr rfl fun i _ => hrow i, ← sum_add_sum_compl R,
    sum_congr rfl fun i hi => ite_eq_left hi,
    sum_congr rfl fun i hi => ite_eq_right (mem_compl.mp hi), sum_const, sum_const,
    nsmul_eq_mul, nsmul_eq_mul, card_compl_cast, card_compl_cast]
  push_cast
  ring

example := trace_blockCert_mul (m := Fin 2) (n := Fin 2) (R := {0}) (C := {0})
  (X := onesMatrix (Fin 2)) (fun _ _ _ _ => rfl) (fun _ _ _ _ => rfl) 1 1

/-- **The block bound**, `review/flat_spectrum.md`, *The failure margin*: every `X` equal to `1`
on `R × Cᶜ` and on `Rᶜ × C` has `‖X‖_S1 ≥ √(k (n − l)) + √(l (m − k))`, for `k = #R` and
`l = #C`. -/
theorem sqrt_add_sqrt_le_nuclearNorm_of_block {R : Finset m} {C : Finset n} {X : Matrix m n ℂ}
    (h₁ : ∀ i ∈ R, ∀ j ∉ C, X i j = 1) (h₂ : ∀ i ∉ R, ∀ j ∈ C, X i j = 1) :
    √(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card m - #R)) ≤ nuclearNorm X := by
  have ha : (0 : ℝ) ≤ #R * (Fintype.card n - #C) := mul_nonneg (Nat.cast_nonneg _)
    (sub_nonneg.mpr (by exact_mod_cast card_le_univ C))
  have hb : (0 : ℝ) ≤ #C * (Fintype.card m - #R) := mul_nonneg (Nat.cast_nonneg _)
    (sub_nonneg.mpr (by exact_mod_cast card_le_univ R))
  have h := norm_trace_le_nuclearNorm (isContraction_blockCert R C
    (inv_sqrt_sq_mul_le_one ha) (inv_sqrt_sq_mul_le_one hb)) X
  rwa [trace_blockCert_mul h₁ h₂, inv_mul_eq_div, inv_mul_eq_div, Real.div_sqrt,
    Real.div_sqrt, Complex.norm_real, Real.norm_of_nonneg (by positivity)] at h

example := sqrt_add_sqrt_le_nuclearNorm_of_block (m := Fin 2) (n := Fin 2) (R := {0}) (C := {0})
  (X := onesMatrix (Fin 2)) (fun _ _ _ _ => rfl) fun _ _ _ _ => rfl

/-- **The least norm off a block**, `review/flat_spectrum.md`, *The failure margin*: for
nonempty `R`, `C` with `m n ≤ k n + l m`, the least nuclear norm of the matrices equal to `J`
off `R × C` is `√(k (n − l)) + √(l (m − k))`, reached by `J − t 1_R 1_Cᵀ` at
`t = 1 − g/(k l)`, `g = √(k l (m − k)(n − l))`. -/
theorem isLeast_nuclearNorm_compl_block {R : Finset m} {C : Finset n} (hR : R.Nonempty)
    (hC : C.Nonempty)
    (h : (Fintype.card m * Fintype.card n : ℝ) ≤ #R * Fintype.card n + #C * Fintype.card m) :
    IsLeast (nuclearNorm '' {X | ∀ i j, ¬(i ∈ R ∧ j ∈ C) → X i j = 1})
      (√(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card m - #R))) := by
  refine ⟨⟨onesSubBlock R C
    (1 - √(#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C)) / (#R * #C)),
    fun i j hij => by simp only [onesSubBlock, of_apply, ite_eq_right hij],
    nuclearNorm_onesSubBlock_best hR hC h⟩, ?_⟩
  rintro _ ⟨X, hX, rfl⟩
  exact sqrt_add_sqrt_le_nuclearNorm_of_block (fun i _ j hj => hX i j fun h => hj h.2)
    fun i hi j _ => hX i j fun h => hi h.1

example := isLeast_nuclearNorm_compl_block (R := ({1, 2} : Finset (Fin 3)))
  (C := ({1, 2} : Finset (Fin 3))) ⟨1, by decide⟩ ⟨1, by decide⟩ (by simp; norm_num)

end Transformer.Grokking.FlatSpectrum
