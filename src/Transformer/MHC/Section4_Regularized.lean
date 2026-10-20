module

public import Transformer.MHC.Section4_Parameterization

/-!
# The epsilon-regularized normalization used in released models

Sources: DeepSeek V4.1's `hc_split_sinkhorn_kernel` in
`papers/DeepSeek-V4.1-Flash/kernel.py`, and Transformers'
`Glm5NextTextHyperConnection.forward` in
`papers/GLM-5.3-Flash/modeling_glm5_next.py`, at the revisions recorded
in `review/residual_connections.md`. The shared algorithm starts with
row softmax plus epsilon, normalizes columns once, then performs nineteen
row/column cycles. This differs from the ideal iteration of
arXiv:2512.24880v2, §4.2. Arithmetic is exact real arithmetic here.
-/

@[expose] public section

open scoped BigOperators
open Transformer.Residual

namespace Transformer.MHC

variable {ι : Type*} [Fintype ι] [Nonempty ι]
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Released kernels add epsilon to every row denominator. -/
noncomputable def regularizedRow (ε : ℝ) (M : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  Matrix.of (fun i j => M i j / ((∑ k, M i k) + ε))

/-- Released kernels add epsilon to every column denominator. -/
noncomputable def regularizedColumn (ε : ℝ) (M : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  Matrix.of (fun i j => M i j / ((∑ k, M k j) + ε))

/-- One initial column normalization, followed by `t` row/column cycles.
Both released models use `t=19`, giving twenty column normalizations. -/
noncomputable def regularizedIterate (ε : ℝ) (M : Matrix ι ι ℝ) : ℕ → Matrix ι ι ℝ
  | 0 => regularizedColumn ε M
  | t + 1 => regularizedColumn ε (regularizedRow ε (regularizedIterate ε M t))

/-- Initialization is row softmax plus epsilon, not elementwise exp alone.
Sources: both pinned implementations, before their Sinkhorn loops. -/
noncomputable def implementationSeed (ε : ℝ) (S : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  Matrix.of (fun i j => Real.exp (S i j) / (∑ k, Real.exp (S i k)) + ε)

/-- The combination matrix produced by the released twenty-iteration code. -/
noncomputable def implementationComb (ε : ℝ) (S : Matrix ι ι ℝ) : Matrix ι ι ℝ :=
  regularizedIterate ε (implementationSeed ε S) 19

/-- Positive row normalization remains positive with nonnegative epsilon. -/
theorem regularizedRow_pos (ε : ℝ) (hε : 0 ≤ ε) (M : Matrix ι ι ℝ)
    (hM : ∀ i j, 0 < M i j) : ∀ i j, 0 < regularizedRow ε M i j := by
  intro i j
  exact div_pos (hM i j) (add_pos_of_pos_of_nonneg (row_sum_pos M hM i) hε)

/-- The regularized positivity hypotheses are jointly realizable. -/
example : (0 : ℝ) ≤ 1 / 1000000 ∧
    ∀ i j : Fin 4, (0 : ℝ) < (Matrix.of (fun _ _ : Fin 4 => (1 : ℝ))) i j := by
  norm_num

/-- Positive column normalization remains positive with nonnegative epsilon. -/
theorem regularizedColumn_pos (ε : ℝ) (hε : 0 ≤ ε) (M : Matrix ι ι ℝ)
    (hM : ∀ i j, 0 < M i j) : ∀ i j, 0 < regularizedColumn ε M i j := by
  intro i j
  exact div_pos (hM i j) (add_pos_of_pos_of_nonneg
    (Finset.sum_pos (fun k _ => hM k j) Finset.univ_nonempty) hε)

/-- The column positivity hypotheses hold for the same concrete input. -/
example : (0 : ℝ) ≤ 1 / 1000000 ∧
    ∀ i j : Fin 4, (0 : ℝ) < (Matrix.of (fun _ _ : Fin 4 => (1 : ℝ))) i j := by
  norm_num

/-- Every finite normalization step remains strictly positive. -/
theorem regularizedIterate_pos (ε : ℝ) (hε : 0 ≤ ε) (M : Matrix ι ι ℝ)
    (hM : ∀ i j, 0 < M i j) (t : ℕ) : ∀ i j, 0 < regularizedIterate ε M t i j := by
  induction t with
  | zero => exact regularizedColumn_pos ε hε M hM
  | succ t ih => exact regularizedColumn_pos ε hε _ (regularizedRow_pos ε hε _ ih)

/-- Positive epsilon and a positive seed meet all iteration hypotheses. -/
example : (0 : ℝ) ≤ 1 / 1000000 ∧
    ∀ i j : Fin 4, (0 : ℝ) < (Matrix.of (fun _ _ : Fin 4 => (1 : ℝ))) i j := by
  norm_num

/-- Adding epsilon to the denominator makes a normalized column sum
strictly below one. Thus this exact real implementation is not exactly
doubly stochastic even before accounting for floating-point rounding. -/
theorem regularizedColumn_sum_lt_one (ε : ℝ) (hε : 0 < ε)
    (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 < M i j) (j : ι) :
    ∑ i, regularizedColumn ε M i j < 1 := by
  have hs : 0 < ∑ i, M i j :=
    Finset.sum_pos (fun i _ => hM i j) Finset.univ_nonempty
  change (∑ i, M i j / ((∑ k, M k j) + ε)) < 1
  rw [← Finset.sum_div, div_lt_one (by positivity)]
  linarith

/-- The strict column-defect hypotheses hold for the shipped epsilon. -/
example : (0 : ℝ) < 1 / 1000000 ∧
    ∀ i j : Fin 4, (0 : ℝ) < (Matrix.of (fun _ _ : Fin 4 => (1 : ℝ))) i j := by
  norm_num

/-- The strict column defect persists at every finite iteration count. -/
theorem regularizedIterate_column_lt_one (ε : ℝ) (hε : 0 < ε)
    (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 < M i j) (t : ℕ) (j : ι) :
    ∑ i, regularizedIterate ε M t i j < 1 := by
  cases t with
  | zero => exact regularizedColumn_sum_lt_one ε hε M hM j
  | succ t =>
    apply regularizedColumn_sum_lt_one ε hε
    exact regularizedRow_pos ε hε.le _ (regularizedIterate_pos ε hε.le M hM t)

/-- The finite defect theorem has a strictly positive four-stream witness. -/
example : (0 : ℝ) < 1 / 1000000 ∧
    ∀ i j : Fin 4, (0 : ℝ) < (Matrix.of (fun _ _ : Fin 4 => (1 : ℝ))) i j := by
  norm_num

/-- The released softmax-plus-epsilon seed is strictly positive. -/
theorem implementationSeed_pos (ε : ℝ) (hε : 0 ≤ ε) (S : Matrix ι ι ℝ) :
    ∀ i j, 0 < implementationSeed ε S i j := by
  intro i j
  apply add_pos_of_pos_of_nonneg _ hε
  exact div_pos (Real.exp_pos _) (Finset.sum_pos
    (fun k _ => Real.exp_pos (S i k)) Finset.univ_nonempty)

/-- Nonnegative seed regularization includes the actual configured epsilon. -/
example : (0 : ℝ) ≤ 1 / 1000000 := by norm_num

/-- The actual twenty-iteration matrix cannot be assumed to satisfy the
ideal Birkhoff constraint when epsilon is positive. This corrects only an
exact reading of the implementation comments, not their numerical intent. -/
theorem implementationComb_not_doubly_stochastic [DecidableEq ι]
    (ε : ℝ) (hε : 0 < ε) (S : Matrix ι ι ℝ) :
    implementationComb ε S ∉ doublyStochastic ℝ ι := by
  intro h
  obtain ⟨j⟩ := ‹Nonempty ι›
  have hlt := regularizedIterate_column_lt_one ε hε _ (implementationSeed_pos ε hε.le S) 19 j
  have heq := sum_col_of_mem_doublyStochastic h j
  change (∑ i, regularizedIterate ε (implementationSeed ε S) 19 i j) = 1 at heq
  linarith

/-- The exact-defect hypothesis holds for `hc_eps=1e-6` in both configs. -/
example : (0 : ℝ) < 1 / 1000000 := by norm_num

/-- GLM-5.3-Flash uses current read gates `sigmoid(a)+epsilon` and the
transpose of its combination matrix. Sources: pinned config and
`Glm5NextTextHyperConnection`, `Glm5NextTextDecoderLayer.forward`.
Logits are supplied after the learned normalization/projection. -/
noncomputable def glm53FlashStep (ε : ℝ) (a c : Fin 4 → ℝ)
    (S : Matrix (Fin 4) (Fin 4) ℝ) (F : E → E) (x : Fin 4 → E) : Fin 4 → E :=
  hyperStep (fun i => gate (a i) + ε) (fun i => writeGate (c i))
    (implementationComb ε S).transpose F x

/-- GLM's final stream collapse is an unweighted mean, not a learned read
gate. Source: pinned Transformers `Glm5NextTextHyperHead.forward`. -/
noncomputable def glm53FlashHead (x : Fin 4 → E) : E :=
  weightedSum (fun _ => 1 / 4) x

/-- The released unweighted mean preserves identical output streams.
Source: `Glm5NextTextHyperHead.forward` in the pinned Transformers source. -/
theorem glm53FlashHead_const (v : E) : glm53FlashHead (fun _ => v) = v := by
  unfold glm53FlashHead
  rw [weightedSum_const]
  norm_num

end Transformer.MHC
