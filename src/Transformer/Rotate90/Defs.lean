module

public import Transformer.Landscape.Neural.Products

/-!
# The current Rotate90 forward pass

Source: the untracked working file `grokking_abmod/scripts/grok-simple.py`,
`Rotate90.forward`, inspected 2026-10-31; the accompanying
`docs/rotate90-swiglu-model.md`, Model. This is the one-block real-arithmetic
model, with no biases or learned normalization gains. It retains the input
RMSNorm, branch-input RMSNorm, branch-output RMSNorm and final RMSNorm.
The inspected script has SHA-256
`67c8c2f15d559180f8731e3a9efbc2759a51811c07e96932907334f1b0a38800`.
`Landscape.Neural.gatedState` describes the older unnormalized residual branch;
we reuse its parameter coordinates and scalar operators, not that forward pass.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Rotate90

open Landscape.Neural

section Model

variable {V K H : Type*} [Fintype K] [Fintype H]

/-- Current source, `x = norm(E[a] + J E[b])`. -/
noncomputable def input (eps : ℝ) (w : GatedWeights V K H) (a b : V) : Channels K → ℝ :=
  rmsNorm eps (tiedInput (gatedEmbedding w) a b)

/-- Current source, the actual three-matrix SwiGLU before its output normalization. -/
noncomputable def branch (w : GatedWeights V K H) (r : Channels K → ℝ) : Channels K → ℝ :=
  dense (outputMatrix w)
    (fun j => swiglu (dense (gateMatrix w) r j) (dense (valueMatrix w) r j))

/-- Current source, `x + norm(SwiGLU(norm(x)))`. -/
noncomputable def state (eps : ℝ) (w : GatedWeights V K H) (a b : V) : Channels K → ℝ :=
  fun i => input eps w a b i + rmsNorm eps (branch w (rmsNorm eps (input eps w a b))) i

/-- Current source, `norm(x) @ E.T`, with the same embedding as both input lookups. -/
noncomputable def logit (eps : ℝ) (w : GatedWeights V K H) (a b q : V) : ℝ :=
  tiedReadout (gatedEmbedding w) (rmsNorm eps (state eps w a b)) q

/-- Final RMSNorm divides all tied logits by its actual common denominator. -/
theorem readout_rmsNorm {C : Type*} [Fintype C] (eps : ℝ)
    (E : V → C → ℝ) (v : C → ℝ) (q : V) :
    tiedReadout E (rmsNorm eps v) q =
      tiedReadout E v q / Real.sqrt (eps + meanSquare v) := by
  simp only [tiedReadout, rmsNorm, div_mul_eq_mul_div, ← Finset.sum_div]

/-- A common real scale factors through the tied linear readout. -/
theorem readout_mul {C : Type*} [Fintype C] (E : V → C → ℝ)
    (v : C → ℝ) (s : ℝ) (q : V) :
    tiedReadout E (fun i => s * v i) q = s * tiedReadout E v q := by
  simp only [tiedReadout, mul_assoc, ← Finset.mul_sum]

end Model

variable {K : Type*} [Fintype K]

/-- Four active adjacent pairs, followed by arbitrary unused channel pairs. -/
abbrev PairIndex (K : Type*) := Fin 4 ⊕ K

/-- Place four pairs in the residual stream, leaving all extra pairs zero. -/
def pack (v : Fin 4 → Fin 2 → ℝ) : Channels (PairIndex K) → ℝ
  | (.inl k, j) => v k j
  | (.inr _, _) => 0

/-- The width includes every padding coordinate in RMSNorm's denominator. -/
def width (K : Type*) [Fintype K] : ℝ := Fintype.card (Channels (PairIndex K))

/-- The actual first RMSNorm scale when the raw input has squared length four. -/
noncomputable def inputScale (eps : ℝ) : ℝ := (Real.sqrt (eps + 4 / width K))⁻¹

/-- The product of the two consecutive input RMSNorm scales. -/
noncomputable def branchScale (eps : ℝ) : ℝ :=
  inputScale (K := K) eps *
    (Real.sqrt (eps + (inputScale (K := K) eps) ^ 2 * (4 / width K)))⁻¹

/-- Source RMSNorm averages over the actual positive, even width. -/
theorem width_pos : 0 < width K := by
  simp only [width, Channels, PairIndex, Fintype.card_prod, Fintype.card_sum,
    Fintype.card_fin, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat]
  positivity

/-- Positive epsilon makes the first normalization multiplier positive. -/
theorem inputScale_pos {eps : ℝ} (heps : 0 < eps) : 0 < inputScale (K := K) eps := by
  unfold inputScale
  have hw := width_pos (K := K)
  positivity

/-- The second input normalization also has a positive, finite multiplier. -/
theorem branchScale_pos {eps : ℝ} (heps : 0 < eps) : 0 < branchScale (K := K) eps := by
  unfold branchScale
  have hw := width_pos (K := K)
  have ha := inputScale_pos (K := K) heps
  positivity

/-- The mean square scales quadratically; epsilon is not removed. -/
theorem meanSquare_mul {I : Type*} [Fintype I] (s : ℝ) (v : I → ℝ) :
    meanSquare (fun i => s * v i) = s ^ 2 * meanSquare v := by
  simp only [meanSquare, mul_pow, ← Finset.mul_sum]
  ring

/-- The exact first normalization on every vector of squared length four. -/
theorem rmsNorm_of_sum_sq (eps : ℝ) (v : Channels (PairIndex K) → ℝ)
    (hv : ∑ i, v i ^ 2 = 4) :
    rmsNorm eps v = fun i => inputScale (K := K) eps * v i := by
  funext i
  simp only [rmsNorm, meanSquare, hv, inputScale, width, div_eq_mul_inv, mul_comm]

/-- Both input normalizations evaluated exactly on such a vector. -/
theorem rmsNorm_twice_of_sum_sq (eps : ℝ) (v : Channels (PairIndex K) → ℝ)
    (hv : ∑ i, v i ^ 2 = 4) :
    rmsNorm eps (rmsNorm eps v) = fun i => branchScale (K := K) eps * v i := by
  rw [rmsNorm_of_sum_sq eps v hv]
  funext i
  rw [rmsNorm, meanSquare_mul]
  simp only [meanSquare, hv, branchScale, width, div_eq_mul_inv]
  ring

/-- Padding contributes zero to the squared length. -/
theorem sum_sq_pack (v : Fin 4 → Fin 2 → ℝ) :
    ∑ i, (pack (K := K) v i) ^ 2 = ∑ k, ∑ j, v k j ^ 2 := by
  simp only [Fintype.sum_prod_type, Fintype.sum_sum_type, pack, zero_pow (by decide : 2 ≠ 0),
    Finset.sum_const_zero, add_zero]

/-- A vector constant on its eight active coordinates witnesses the normalization premise. -/
example : ∑ i, (pack (K := Fin 0) ![![1, 1], ![1, 1], ![0, 0], ![0, 0]] i) ^ 2 = 4 := by
  rw [sum_sq_pack]
  norm_num [Fin.sum_univ_succ]

example : 0 < inputScale (K := Fin 4) (1 / 8388608) := inputScale_pos (by norm_num)
example : 0 < branchScale (K := Fin 0) (1 / 8388608) := branchScale_pos (by norm_num)

example := rmsNorm_of_sum_sq (K := Fin 0) (1 / 8388608)
  (pack ![![1, 1], ![1, 1], ![0, 0], ![0, 0]])
  (by rw [sum_sq_pack]; norm_num [Fin.sum_univ_succ])

example := rmsNorm_twice_of_sum_sq (K := Fin 4) (1 / 8388608)
  (pack ![![1, 1], ![1, 1], ![0, 0], ![0, 0]])
  (by rw [sum_sq_pack]; norm_num [Fin.sum_univ_succ])

example := readout_rmsNorm (1 / 8388608) (fun q i : Fin 2 => (q.val + i.val : ℝ))
  (fun i : Fin 2 => (i.val + 1 : ℝ)) 0

example := readout_mul (fun q i : Fin 2 => (q.val + i.val : ℝ))
  (fun i : Fin 2 => (i.val + 1 : ℝ)) 2 0

end Transformer.Rotate90
