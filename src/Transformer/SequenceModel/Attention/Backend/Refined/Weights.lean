module

public import Transformer.SequenceModel.Attention.Backend.Refined.NormInputs
public import Transformer.SequenceModel.Attention.Adaptive.Attention

/-!
# Exact stored matrices and their separate Q/K/V bounds

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M3.
Selectors preserve the source fused layout. The norm inequalities are
checked from the stored dyadic entries, without a supplied operator norm.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution Complete Adaptive
open scoped BigOperators
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

/-- The Q block of the stored fused matrix, in source head order. -/
def qMatrix : Matrix (Fin 4) (Fin 4) ℝ := fun r c => matrix_qkv ⟨0 + r.val, by omega⟩ c

/-- Exact linear map for the Q slice. -/
def qMap : EucSpace 4 →L[ℝ] EucSpace 4 := matrixEquiv 4 4 qMatrix

/-- M3: slicing the actual fused source map equals the retained component map. -/
theorem qMap_apply (x : EucSpace 4) : qMap x =
    headSlice cfg (qkvSlice cfg (qkvQ cfg) (blockWeights.qkv x)) 0 := by
  ext c
  fin_cases c <;> rfl

/-- M3: coordinates of the stored Q component. -/
theorem qMap_coordinates (x : EucSpace 4) : qMap x =
    WithLp.toLp 2 (fun r => ∑ c, qMatrix r c * x c) := rfl

/-- M3: Q has its own certified Frobenius bound. -/
theorem qMap_norm : ‖qMap‖ ≤ 1 / 8000 := by
  apply matrix_checked qMap qMatrix _ (by norm_num) (by intros; rfl)
  norm_num [qMatrix, matrix_qkv, Fin.sum_univ_succ]

/-- M3: the stored Q slice has its own rounding allowance. -/
theorem local_qpart (i : Fin 2) : ‖qPart i - qMap (raw_pre i)‖ ≤ 4 / 1000000000000 := by
  rw [qMap_coordinates]
  apply vector_local (2 / 1000000000000) _ (by norm_num) (by norm_num)
  intro c
  fin_cases i <;> fin_cases c <;> norm_num [qPart, qMatrix, matrix_qkv, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- The K block of the stored fused matrix, in source head order. -/
def kMatrix : Matrix (Fin 4) (Fin 4) ℝ := fun r c => matrix_qkv ⟨4 + r.val, by omega⟩ c

/-- Exact linear map for the K slice. -/
def kMap : EucSpace 4 →L[ℝ] EucSpace 4 := matrixEquiv 4 4 kMatrix

/-- M3: slicing the actual fused source map equals the retained component map. -/
theorem kMap_apply (x : EucSpace 4) : kMap x =
    headSlice cfg (qkvSlice cfg (qkvK cfg) (blockWeights.qkv x)) 0 := by
  ext c
  fin_cases c <;> rfl

/-- M3: coordinates of the stored K component. -/
theorem kMap_coordinates (x : EucSpace 4) : kMap x =
    WithLp.toLp 2 (fun r => ∑ c, kMatrix r c * x c) := rfl

/-- M3: K has its own certified Frobenius bound. -/
theorem kMap_norm : ‖kMap‖ ≤ 1 / 8000 := by
  apply matrix_checked kMap kMatrix _ (by norm_num) (by intros; rfl)
  norm_num [kMatrix, matrix_qkv, Fin.sum_univ_succ]

/-- M3: the stored K slice has its own rounding allowance. -/
theorem local_kpart (i : Fin 2) : ‖kPart i - kMap (raw_pre i)‖ ≤ 4 / 1000000000000 := by
  rw [kMap_coordinates]
  apply vector_local (2 / 1000000000000) _ (by norm_num) (by norm_num)
  intro c
  fin_cases i <;> fin_cases c <;> norm_num [kPart, kMatrix, matrix_qkv, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- The V block of the stored fused matrix, in source head order. -/
def vMatrix : Matrix (Fin 4) (Fin 4) ℝ := fun r c => matrix_qkv ⟨8 + r.val, by omega⟩ c

/-- Exact linear map for the V slice. -/
def vMap : EucSpace 4 →L[ℝ] EucSpace 4 := matrixEquiv 4 4 vMatrix

/-- M3: slicing the actual fused source map equals the retained component map. -/
theorem vMap_apply (x : EucSpace 4) : vMap x =
    headSlice cfg (qkvSlice cfg (qkvV cfg) (blockWeights.qkv x)) 0 := by
  ext c
  fin_cases c <;> rfl

/-- M3: coordinates of the stored V component. -/
theorem vMap_coordinates (x : EucSpace 4) : vMap x =
    WithLp.toLp 2 (fun r => ∑ c, vMatrix r c * x c) := rfl

/-- M3: V has its own certified Frobenius bound. -/
theorem vMap_norm : ‖vMap‖ ≤ 101 / 100 := by
  apply matrix_checked vMap vMatrix _ (by norm_num) (by intros; rfl)
  norm_num [vMatrix, matrix_qkv, Fin.sum_univ_succ]

/-- M3: the stored V slice has its own rounding allowance. -/
theorem local_vpart (i : Fin 2) : ‖vPart i - vMap (raw_pre i)‖ ≤ 4 / 100000000 := by
  rw [vMap_coordinates]
  apply vector_local (2 / 100000000) _ (by norm_num) (by norm_num)
  intro c
  fin_cases i <;> fin_cases c <;> norm_num [vPart, vMatrix, matrix_qkv, raw_fused, raw_pre, headSlice, qkvSlice, headSplit, qkvQ, qkvK, qkvV, qkvSplit, finSumFinEquiv, finProdFinEquiv, cfg, Config.head_dim, Fin.sum_univ_succ]

/-- M3: the actual stored projection matrix has this certified norm bound. -/
theorem projection_norm : ‖blockWeights.projection‖ ≤ 13 / 50 := by
  apply matrix_checked _ matrix_projection _ (by norm_num) (by intros; rfl)
  norm_num [matrix_projection, Fin.sum_univ_succ]

/-- M3: the actual stored gate matrix has this certified norm bound. -/
theorem gate_norm : ‖blockWeights.gate‖ ≤ 21 / 50 := by
  apply matrix_checked _ matrix_gate _ (by norm_num) (by intros; rfl)
  norm_num [matrix_gate, Fin.sum_univ_succ]

/-- M3: the actual stored value matrix has this certified norm bound. -/
theorem value_norm : ‖blockWeights.value‖ ≤ 21 / 50 := by
  apply matrix_checked _ matrix_value _ (by norm_num) (by intros; rfl)
  norm_num [matrix_value, Fin.sum_univ_succ]

/-- M3: the actual stored output matrix has this certified norm bound. -/
theorem output_norm : ‖blockWeights.output‖ ≤ 21 / 50 := by
  apply matrix_checked _ matrix_output _ (by norm_num) (by intros; rfl)
  norm_num [matrix_output, Fin.sum_univ_succ]

end Transformer.SequenceModel.Attention.Backend.Refined
