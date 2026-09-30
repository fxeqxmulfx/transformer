/-
# DASH — lossless block layout and stacked batches

arXiv:2602.02016v2, §4. The equivalence is with the same
blockwise Shampoo computation, not with unblocked Shampoo.
-/

import Transformer.DASH.Section2_Models
import Mathlib.Logic.Equiv.Fin.Basic

namespace Transformer.DASH

variable {r c B : ℕ}

/-- Split a divisible matrix into the source's `N_m N_n` square blocks,
arXiv:2602.02016v2, §4, the displayed block matrix. -/
def blockGradient (G : Matrix (Fin (r * B)) (Fin (c * B)) ℝ) :
    (Fin r × Fin c) → Matrix (Fin B) (Fin B) ℝ :=
  fun ij k l => G (finProdFinEquiv (ij.1, k)) (finProdFinEquiv (ij.2, l))

/-- Recover the original matrix entries from their block layout,
arXiv:2602.02016v2, §4, conversion of preconditioned updates back to layers. -/
def unblockGradient (G : (Fin r × Fin c) → Matrix (Fin B) (Fin B) ℝ) :
    Matrix (Fin (r * B)) (Fin (c * B)) ℝ :=
  fun k l => G ((finProdFinEquiv.symm k).1, (finProdFinEquiv.symm l).1)
    (finProdFinEquiv.symm k).2 (finProdFinEquiv.symm l).2

/-- Splitting and reassembling preserves every original entry,
arXiv:2602.02016v2, §4, gradient block conversion. -/
theorem unblock_blockGradient (G : Matrix (Fin (r * B)) (Fin (c * B)) ℝ) :
    unblockGradient (blockGradient G) = G := by
  ext i j
  simp only [unblockGradient, blockGradient, Prod.mk.eta, Equiv.apply_symm_apply]

/-- Reassembling and splitting preserves every block,
arXiv:2602.02016v2, §4, lossless stacked block layout. -/
theorem block_unblockGradient (G : (Fin r × Fin c) → Matrix (Fin B) (Fin B) ℝ) :
    blockGradient (unblockGradient G) = G := by
  funext ij k l
  simp [unblockGradient, blockGradient]

/-- Parameter subtraction and scalar updates commute with reassembly,
arXiv:2602.02016v2, §4, applying the stacked update to the model. -/
theorem unblockGradient_update (θ : Matrix (Fin (r * B)) (Fin (c * B)) ℝ)
    (U : (Fin r × Fin c) → Matrix (Fin B) (Fin B) ℝ) (η : ℝ) :
    unblockGradient (fun j => blockGradient θ j - η • U j) = θ - η • unblockGradient U := by
  ext i j
  simp only [unblockGradient, blockGradient, Matrix.sub_apply, Matrix.smul_apply,
    Prod.mk.eta, Equiv.apply_symm_apply]

/-- The batch has exactly the stated number of full gradient blocks,
arXiv:2602.02016v2, §4, `N=N_m N_n`. -/
theorem blockGradient_count (r c : ℕ) : Fintype.card (Fin r × Fin c) = r * c := by simp

/-- Stack three shape-compatible tensors using disjoint batch indices,
arXiv:2602.02016v2, §4, `stack(X,Y,Z)`. -/
def stackThree {α β γ V : Type*} (X : α → V) (Y : β → V) (Z : γ → V) :
    (α ⊕ (β ⊕ γ)) → V := Sum.elim X (Sum.elim Y Z)

/-- Unstacking each bucket recovers it exactly,
arXiv:2602.02016v2, §4, the combined inverse-root buffer. -/
theorem stackThree_recover {α β γ V : Type*} (X : α → V) (Y : β → V) (Z : γ → V) :
    (stackThree X Y Z ∘ Sum.inl = X) ∧
      (stackThree X Y Z ∘ Sum.inr ∘ Sum.inl = Y) ∧
      (stackThree X Y Z ∘ Sum.inr ∘ Sum.inr = Z) := by
  exact ⟨rfl, rfl, rfl⟩

/-- Any independent per-matrix inverse-root solver commutes with stacking.
The argument is the actual solver function; the result preserves the data
and each input/output pair, without asserting a hardware speedup.
Source: arXiv:2602.02016v2, §4, “exactly the same inverse root procedure”. -/
theorem stackThree_map {α β γ V W : Type*} (f : V → W)
    (X : α → V) (Y : β → V) (Z : γ → V) :
    f ∘ stackThree X Y Z = stackThree (f ∘ X) (f ∘ Y) (f ∘ Z) := by
  funext i
  rcases i with i | i
  · rfl
  · rcases i with i | i <;> rfl

/-- Batched multiplication is matrix multiplication separately on the batch
axis, arXiv:2602.02016v2, §4, `bmm` and swapping only the last two dimensions. -/
def batchMul {ι : Type*} {m n p : ℕ}
    (A : ι → Matrix (Fin m) (Fin n) ℝ) (C : ι → Matrix (Fin n) (Fin p) ℝ) :
    ι → Matrix (Fin m) (Fin p) ℝ := fun j => A j * C j

/-- A batched matrix product commutes with stacking compatible buckets,
arXiv:2602.02016v2, §4, batched Gram and preconditioned-gradient products. -/
theorem stackThree_batchMul {α β γ : Type*} {m n p : ℕ}
    (A₁ : α → Matrix (Fin m) (Fin n) ℝ) (A₂ : β → Matrix (Fin m) (Fin n) ℝ)
    (A₃ : γ → Matrix (Fin m) (Fin n) ℝ) (C₁ : α → Matrix (Fin n) (Fin p) ℝ)
    (C₂ : β → Matrix (Fin n) (Fin p) ℝ) (C₃ : γ → Matrix (Fin n) (Fin p) ℝ) :
    batchMul (stackThree A₁ A₂ A₃) (stackThree C₁ C₂ C₃) =
      stackThree (batchMul A₁ C₁) (batchMul A₂ C₂) (batchMul A₃ C₃) := by
  funext i
  rcases i with i | i
  · rfl
  · rcases i with i | i <;> rfl

/-- Stacking has the source's summed batch dimension,
arXiv:2602.02016v2, §4, `N_X+N_Y+N_Z`. -/
theorem stackThree_count (x y z : ℕ) :
    Fintype.card (Fin x ⊕ (Fin y ⊕ Fin z)) = x + y + z := by simp [Nat.add_assoc]

end Transformer.DASH
