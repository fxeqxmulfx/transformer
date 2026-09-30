/-
# DASH — block sizes, remainders, and tensor dimensions

arXiv:2602.02016v2, §4 and Appendix D. The numerical block
example and the vision-layer reshape are exact dimension counts.
-/

import Transformer.DASH.Section4_Blocking

namespace Transformer.DASH

/-- Euclidean division gives full blocks and one smaller remainder,
arXiv:2602.02016v2, §4, the `32000×2048` example. -/
theorem block_dimension_decomposition (m B : ℕ) (hB : 0 < B) :
    m = (m / B) * B + m % B ∧ m % B < B := by
  exact ⟨by simpa only [Nat.mul_comm] using (Nat.div_add_mod m B).symm,
    Nat.mod_lt m hB⟩

/-- Positive block sizes exist, arXiv:2602.02016v2, §4. -/
example : 0 < (1024 : ℕ) := by norm_num

/-- The embedding example contains 62 full blocks and two `256×1024`
blocks. Their areas add up to all original gradient entries.
Source: arXiv:2602.02016v2, §4, `V=32000,E=2048,B=1024`. -/
theorem embedding_block_dimensions :
    32000 / 1024 = (31 : ℕ) ∧ 2048 / 1024 = (2 : ℕ) ∧
      32000 % 1024 = (256 : ℕ) ∧ 31 * 2 = (62 : ℕ) ∧
      62 * 1024 * 1024 + 2 * 256 * 1024 = (32000 * 2048 : ℕ) := by norm_num

/-- Left and right Gram dimensions for a rectangular residual block,
arXiv:2602.02016v2, §4, `L_rest∈(2,256,256), R_rest∈(2,1024,1024)`. -/
theorem residual_gram_dimensions :
    Fintype.card (Fin 2 × Fin 256 × Fin 256) = (2 * 256 * 256 : ℕ) ∧
      Fintype.card (Fin 2 × Fin 1024 × Fin 1024) = (2 * 1024 * 1024 : ℕ) := by
  simp

/-- `L_full`, `R_full`, and `R_rest` supply 126 matrices of shape `1024²`.
Source: arXiv:2602.02016v2, §4, `S_full=stack(L_full,R_full,R_rest)`. -/
theorem embedding_stacked_count :
    Fintype.card (Fin 62 ⊕ (Fin 62 ⊕ Fin 2)) = 126 := by simp

/-- Splitting `E=2B` gives two gradient blocks per normalization layer,
arXiv:2602.02016v2, §4, “Normalization Layers”. -/
theorem normalization_block_dimensions (N B : ℕ) :
    Fintype.card (Fin N × Fin 2 × Fin B × Fin 1) = 2 * N * B ∧
      Fintype.card (Fin (2 * N) × Fin B × Fin B) = 2 * N * B * B := by
  simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

/-- The second dimension in a vision patch is `3·16·16=768`,
arXiv:2602.02016v2, Appendix D, the `(E,3,16,16)→(E,768)` reshape. -/
def visionPatchEquiv : (Fin 3 × (Fin 16 × Fin 16)) ≃ Fin 768 :=
  (Equiv.prodCongr (Equiv.refl (Fin 3)) finProdFinEquiv).trans finProdFinEquiv

/-- The source's vision reshape preserves every weight and all four input
axes; the prose calls this a “3D embedding” despite listing four axes.
Source: arXiv:2602.02016v2, Appendix D. -/
def flattenVisionLayer {E : ℕ}
    (A : Fin E → Fin 3 × (Fin 16 × Fin 16) → ℝ) : Matrix (Fin E) (Fin 768) ℝ :=
  fun i j => A i (visionPatchEquiv.symm j)

/-- The inverse of the source's vision reshape,
arXiv:2602.02016v2, Appendix D. -/
def unflattenVisionLayer {E : ℕ} (A : Matrix (Fin E) (Fin 768) ℝ) :
    Fin E → Fin 3 × (Fin 16 × Fin 16) → ℝ := fun i j => A i (visionPatchEquiv j)

/-- Flattening followed by its inverse preserves each vision weight,
arXiv:2602.02016v2, Appendix D, merging the remaining patch dimensions. -/
theorem vision_reshape_roundtrip {E : ℕ}
    (A : Fin E → Fin 3 × (Fin 16 × Fin 16) → ℝ) :
    unflattenVisionLayer (flattenVisionLayer A) = A := by
  funext i j
  simp only [unflattenVisionLayer, flattenVisionLayer, Equiv.symm_apply_apply]

/-- The shape change preserves the parameter count,
arXiv:2602.02016v2, Appendix D. -/
theorem vision_parameter_count (E : ℕ) :
    Fintype.card (Fin E × (Fin 3 × (Fin 16 × Fin 16))) = E * 768 := by simp

end Transformer.DASH
