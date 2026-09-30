/-
# DASH — lossless blocking with arbitrary rectangular remainders

arXiv:2602.02016v2, §4. The full and residual block indices are
disjoint and cover every gradient entry, even when both axes have a
remainder. The source's embedding example has a remainder on one axis.
-/

import Transformer.DASH.Section4_Dimensions

namespace Transformer.DASH

/-- Split an axis into full-block/local-coordinate pairs and residual entries.
Source: arXiv:2602.02016v2, §4, `G_full` and `G_rest`. -/
def residualAxisEquiv (m B : ℕ) :
    Fin m ≃ ((Fin (m / B) × Fin B) ⊕ Fin (m % B)) :=
  (finCongr (by simpa only [Nat.mul_comm] using (Nat.div_add_mod m B).symm)).trans
    (finSumFinEquiv.symm.trans
      (Equiv.sumCongr finProdFinEquiv.symm (Equiv.refl (Fin (m % B)))))

/-- Actual reindexing of the gradient into full and residual axis coordinates.
Source: arXiv:2602.02016v2, §4, rectangular residual blocks. -/
def blockResidualGradient {m n : ℕ} (B : ℕ) (G : Matrix (Fin m) (Fin n) ℝ) :
    Matrix ((Fin (m / B) × Fin B) ⊕ Fin (m % B))
      ((Fin (n / B) × Fin B) ⊕ Fin (n % B)) ℝ :=
  fun i j => G ((residualAxisEquiv m B).symm i) ((residualAxisEquiv n B).symm j)

/-- Inverse reindexing after the block-wise preconditioned update.
Source: arXiv:2602.02016v2, §4, recovering the original gradient shape. -/
def unblockResidualGradient {m n : ℕ} (B : ℕ)
    (G : Matrix ((Fin (m / B) × Fin B) ⊕ Fin (m % B))
      ((Fin (n / B) × Fin B) ⊕ Fin (n % B)) ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => G (residualAxisEquiv m B i) (residualAxisEquiv n B j)

/-- Both directions of the residual block layout preserve every entry.
Source: arXiv:2602.02016v2, §4, full and smaller blocks. -/
theorem residual_layout_roundtrip {m n : ℕ} (B : ℕ) (G : Matrix (Fin m) (Fin n) ℝ) :
    unblockResidualGradient B (blockResidualGradient B G) = G := by
  ext i j
  simp only [unblockResidualGradient, blockResidualGradient, Equiv.symm_apply_apply]

/-- Reblocking a reconstructed residual layout preserves every block.
Source: arXiv:2602.02016v2, §4, inverse of the residual layout. -/
theorem residual_layout_inverse {m n : ℕ} (B : ℕ)
    (G : Matrix ((Fin (m / B) × Fin B) ⊕ Fin (m % B))
      ((Fin (n / B) × Fin B) ⊕ Fin (n % B)) ℝ) :
    blockResidualGradient B (unblockResidualGradient B G) = G := by
  ext i j
  simp only [blockResidualGradient, unblockResidualGradient, Equiv.apply_symm_apply]

/-- The full, right, bottom, and corner block areas sum to the original area.
This supplies the general count behind the source's `62` full and `2` residual
blocks. Source: arXiv:2602.02016v2, §4, embedding-layer example. -/
theorem residual_block_parameter_count (m n B : ℕ) :
    (m / B) * (n / B) * B * B + (m / B) * B * (n % B) +
      (n / B) * (m % B) * B + (m % B) * (n % B) = m * n := by
  calc
    _ = ((m / B) * B + m % B) * ((n / B) * B + n % B) := by ring
    _ = m * n := by
      rw [Nat.mul_comm (m / B) B, Nat.div_add_mod,
        Nat.mul_comm (n / B) B, Nat.div_add_mod]

/-- Updating in the residual layout is exactly the reconstructed parameter
update in the original shape. Source: arXiv:2602.02016v2, §4. -/
theorem residual_layout_update {m n : ℕ} (B : ℕ) (η : ℝ)
    (θ : Matrix (Fin m) (Fin n) ℝ)
    (U : Matrix ((Fin (m / B) × Fin B) ⊕ Fin (m % B))
      ((Fin (n / B) × Fin B) ⊕ Fin (n % B)) ℝ) :
    unblockResidualGradient B (blockResidualGradient B θ - η • U) =
      θ - η • unblockResidualGradient B U := by
  ext i j
  simp only [unblockResidualGradient, blockResidualGradient, Matrix.sub_apply,
    Matrix.smul_apply, Equiv.symm_apply_apply]

/-- Reindex the actual normalization-layer gradient into `2N` blocks of shape
`B×1`, arXiv:2602.02016v2, §4, “Normalization Layers”, `E=2B`. -/
def blockNormalization {N B : ℕ} (G : Matrix (Fin N) (Fin (2 * B)) ℝ) :
    (Fin N × Fin 2) → Matrix (Fin B) (Fin 1) ℝ :=
  fun i j _ => G i.1 (finProdFinEquiv (i.2, j))

/-- Recover the original `N×E` normalization-layer array,
arXiv:2602.02016v2, §4, “Normalization Layers”. -/
def unblockNormalization {N B : ℕ}
    (G : (Fin N × Fin 2) → Matrix (Fin B) (Fin 1) ℝ) : Matrix (Fin N) (Fin (2 * B)) ℝ :=
  fun i j => G (i, (finProdFinEquiv.symm j).1) (finProdFinEquiv.symm j).2 0

/-- The normalization-layer layout preserves each gradient coordinate.
Source: arXiv:2602.02016v2, §4, “Normalization Layers”. -/
theorem normalization_layout_roundtrip {N B : ℕ} (G : Matrix (Fin N) (Fin (2 * B)) ℝ) :
    unblockNormalization (blockNormalization G) = G := by
  ext i j
  simp only [unblockNormalization, blockNormalization, Prod.mk.eta, Equiv.apply_symm_apply]

end Transformer.DASH
