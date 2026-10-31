module

public import Transformer.Rotate90.Basic

/-!
# Actual Rotate90 coordinates for every powered-division task

Source: `review/flat_spectrum.md`, *Actual network expressivity for every exponent*,
construction and normalization. The source forward pass is
`grokking_abmod/docs/rotate90-swiglu-model.md`, Model, and `Rotate90.Defs`.

Seven active pairs store a constant, the three-coordinate faithful answer code C(a),
and C(a^k). All lookup values belong to the same embedding used for output. Input and
branch vectors have squared length six; optional padding is retained in every RMSNorm.
The branch vector here specifies the construction's target. `Power.Products` derives it
from ten explicit products and actual SwiGLU matrices, without putting a/b in the model.
-/

@[expose] public section

namespace Transformer.Rotate90.Power

open Landscape.Neural
open scoped BigOperators

variable {K : Type*} [Fintype K]

/-- Seven active pairs and optional unused pairs, source construction. -/
abbrev PairIndex (K : Type*) := Fin 7 ⊕ K

/-- Pad a fourteen-coordinate vector with zeros in the actual residual space. -/
def pack (v : Fin 7 → Fin 2 → ℝ) : Channels (PairIndex K) → ℝ
  | (.inl i, j) => v i j
  | (.inr _, _) => 0

/-- All channels, including padding, enter the normalization denominator.
This is the full model width, not the dimension of the active subspace. -/
def width (K : Type*) [Fintype K] : ℝ := Fintype.card (Channels (PairIndex K))

/-- First input scale for raw squared length six; epsilon is retained. -/
noncomputable def inputScale (eps : ℝ) : ℝ := (Real.sqrt (eps + 6 / width K))⁻¹

/-- Scale after both input normalizations, source normalization calculation. -/
noncomputable def branchScale (eps : ℝ) : ℝ :=
  inputScale (K := K) eps *
    (Real.sqrt (eps + (inputScale (K := K) eps) ^ 2 * (6 / width K)))⁻¹

/-- The actual residual width is positive for every padding size. -/
theorem width_pos : 0 < width K := by
  simp only [width, Channels, PairIndex, Fintype.card_prod, Fintype.card_sum,
    Fintype.card_fin, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat]
  positivity

/-- The construction's width is fourteen active coordinates plus the finite padding. -/
theorem width_eq : width K = 14 + 2 * (Fintype.card K : ℝ) := by
  simp only [width, Channels, PairIndex, Fintype.card_prod, Fintype.card_sum,
    Fintype.card_fin, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat]
  ring

/-- Arbitrarily many further pairs increase the actual RMSNorm width by twice their count. -/
theorem width_add_pairs (n : ℕ) : width (K ⊕ Fin n) = width K + 2 * n := by
  rw [width_eq, width_eq]
  simp only [Fintype.card_sum, Fintype.card_fin, Nat.cast_add]
  ring

/-- Positive epsilon gives a positive first normalization scale. -/
theorem inputScale_pos {eps : ℝ} (heps : 0 < eps) : 0 < inputScale (K := K) eps := by
  unfold inputScale
  have hw := width_pos (K := K)
  positivity

/-- Both input normalization scales have positive product. -/
theorem branchScale_pos {eps : ℝ} (heps : 0 < eps) : 0 < branchScale (K := K) eps := by
  unfold branchScale
  have hw := width_pos (K := K)
  have hs := inputScale_pos (K := K) heps
  positivity

/-- Padding contributes no squared norm, but still contributes to `width`. -/
theorem sum_sq_pack (v : Fin 7 → Fin 2 → ℝ) :
    ∑ i, (pack (K := K) v i) ^ 2 = ∑ i, ∑ j, v i j ^ 2 := by
  simp only [Fintype.sum_prod_type, Fintype.sum_sum_type, pack,
    zero_pow (by decide : 2 ≠ 0), Finset.sum_const_zero, add_zero]

/-- Exact evaluation of one source RMSNorm at squared length six. -/
theorem rmsNorm_of_sum_sq (eps : ℝ) (v : Channels (PairIndex K) → ℝ)
    (hv : ∑ i, v i ^ 2 = 6) :
    rmsNorm eps v = fun i => inputScale (K := K) eps * v i := by
  funext i
  simp only [rmsNorm, meanSquare, hv, inputScale, width, div_eq_mul_inv, mul_comm]

/-- Exact evaluation of the two consecutive input normalizations. -/
theorem rmsNorm_twice_of_sum_sq (eps : ℝ) (v : Channels (PairIndex K) → ℝ)
    (hv : ∑ i, v i ^ 2 = 6) :
    rmsNorm eps (rmsNorm eps v) = fun i => branchScale (K := K) eps * v i := by
  rw [rmsNorm_of_sum_sq eps v hv]
  funext i
  rw [rmsNorm, Rotate90.meanSquare_mul]
  simp only [meanSquare, hv, branchScale, width, div_eq_mul_inv]
  ring

variable {p : ℕ} [Fact p.Prime]

/-- The single tied embedding stores C(a) and C(a^k) alongside a constant, source
construction. Even slots contain features; all odd slots are zero. -/
noncomputable def embedding (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a : ZMod p) :
    Channels (PairIndex K) → ℝ :=
  pack ![![1, 0], ![(χ a).re, 0], ![(χ a).im, 0], ![zeroFlag a, 0],
    ![(χ (a ^ k)).re, 0], ![(χ (a ^ k)).im, 0], ![zeroFlag (a ^ k), 0]]

/-- The unnormalized rotated sum of the two embedding rows. -/
noncomputable def raw (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a b : ZMod p) :
    Channels (PairIndex K) → ℝ :=
  pack ![![1, 1], ![(χ a).re, (χ b).re], ![(χ a).im, (χ b).im], ![zeroFlag a, zeroFlag b],
    ![(χ (a ^ k)).re, (χ (b ^ k)).re], ![(χ (a ^ k)).im, (χ (b ^ k)).im],
    ![zeroFlag (a ^ k), zeroFlag (b ^ k)]]

/-- The branch cancels the input answer code and the powered-code output channels.
Its odd coordinates balance the norm, while the tied readout sees only even coordinates.
`Power.branch_weights` proves this is produced by actual fixed matrices. -/
noncomputable def branchVector (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a b : ZMod p) :
    Channels (PairIndex K) → ℝ :=
  pack ![![0, 0],
    ![(χ ((a / b) ^ k)).re - (χ a).re, (χ ((a / b) ^ k)).re + (χ a).re],
    ![(χ ((a / b) ^ k)).im - (χ a).im, (χ ((a / b) ^ k)).im + (χ a).im],
    ![zeroFlag ((a / b) ^ k) - zeroFlag a, zeroFlag ((a / b) ^ k) + zeroFlag a],
    ![-(χ (a ^ k)).re, (χ (a ^ k)).re], ![-(χ (a ^ k)).im, (χ (a ^ k)).im],
    ![-zeroFlag (a ^ k), zeroFlag (a ^ k)]]

example := width_pos (K := Fin 1)
example := width_eq (K := Fin 57)
example := width_add_pairs (K := Fin 1) 1000
example := inputScale_pos (K := Fin 1) (eps := 1 / 8388608) (by norm_num)
example := branchScale_pos (K := Fin 1) (eps := 1 / 8388608) (by norm_num)
example := sum_sq_pack (K := Fin 1)
  ![![1, 1], ![1, 1], ![1, 1], ![0, 0], ![0, 0], ![0, 0], ![0, 0]]

example := rmsNorm_of_sum_sq (K := Fin 1) (1 / 8388608)
  (pack ![![1, 1], ![1, 1], ![1, 1], ![0, 0], ![0, 0], ![0, 0], ![0, 0]])
  (by rw [sum_sq_pack]; norm_num [Fin.sum_univ_succ])

example := rmsNorm_twice_of_sum_sq (K := Fin 1) (1 / 8388608)
  (pack ![![1, 1], ![1, 1], ![1, 1], ![0, 0], ![0, 0], ![0, 0], ![0, 0]])
  (by rw [sum_sq_pack]; norm_num [Fin.sum_univ_succ])

/-- The smallest active space has fourteen channels, and one extra pair gives width 16. -/
example : width (Fin 0) = 14 ∧ width (Fin 1) = 16 := by
  norm_num [width, Channels, PairIndex]

end Transformer.Rotate90.Power
