module

public import Transformer.Grokking.GradientEvidence.AdaptiveResidual

/-!
# Adaptive forcing includes noncommutation and moment discrepancy

Source: `review/flat_spectrum.md`, *Adaptive interaction control*, AI1–AI2.
Stored-moment equations follow rotate90_complex's native engine/update and
product/moment at source hash 14cff3b5 (betas .9/.98, epsilon 1e-8).
These are exact-real equations, not a floating-point execution theorem.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset FlatSpectrum.FeatureCodes

variable {K : Type*} [Fintype K]

/-- Bias-corrected first moment from an already updated buffer, AI1.
Native steps start at one; a saved buffer is not a current full gradient. -/
def correctedMoment (b : ℝ) (t : ℕ) (m : ℝ) : ℝ := m / (1 - b ^ t)

/-- Native diagonal inverse scale, retaining its positive epsilon, AI1. -/
def adamDiagonal (b eps : ℝ) (t : ℕ) (v : ℝ) : ℝ :=
  1 / (Real.sqrt v / Real.sqrt (1 - b ^ t) + eps)

/-- One coordinate of the native update from the updated moment buffers,
AI1. Source: engine.rs::update and product.rs::moment, source 14cff3b5. -/
def bufferedAdamUpdate (b1 b2 eps η ρ : ℝ) (t : ℕ) (p m v : ℝ) : ℝ :=
  (1 - η * ρ) * p -
    (η / (1 - b1 ^ t)) * m / (Real.sqrt v / Real.sqrt (1 - b2 ^ t) + eps)

/-- Factor the actual buffered direction without identifying its first
moment with any chosen reference gradient, AI1. -/
theorem bufferedAdamUpdate_factor (b1 b2 eps η ρ : ℝ) (t : ℕ) (p m v : ℝ) :
    bufferedAdamUpdate b1 b2 eps η ρ t p m v =
      (1 - η * ρ) * p - η * adamDiagonal b2 eps t v * correctedMoment b1 t m := by
  unfold bufferedAdamUpdate adamDiagonal correctedMoment
  ring

/-- Applying a diagonal scale before versus after the residual map, AI2. -/
def diagonalCommutator (Q : K → K → ℝ) (d g : K → ℝ) (k : K) : ℝ :=
  matrixApply Q (fun l => d l * g l) k - d k * matrixApply Q g k

/-- Coordinate scale differences cause mixing of a rotated subspace, AI2. -/
theorem diagonalCommutator_eq_sum (Q : K → K → ℝ) (d g : K → ℝ) (k : K) :
    diagonalCommutator Q d g k = ∑ l, Q k l * (d l - d k) * g l := by
  simp only [diagonalCommutator, matrixApply, linearScore, mul_sum, ← sum_sub_distrib]
  apply sum_congr rfl
  intro l hl
  ring

/-- A constant diagonal scale commutes with every residual map, AI2. -/
theorem diagonalCommutator_constant (Q : K → K → ℝ) (g : K → ℝ) (a : ℝ) (k : K) :
    diagonalCommutator Q (fun _ => a) g k = 0 := by
  rw [diagonalCommutator_eq_sum]
  simp only [sub_self, mul_zero, zero_mul, sum_const_zero]

/-- **Adaptive forcing has three separate sources**, AI2: the scaled
projected reference gradient, noncommutation, and the moment discrepancy. -/
theorem adaptive_forcing_decomposition (Q : K → K → ℝ) (d m g : K → ℝ) (k : K) :
    matrixApply Q (fun l => d l * m l) k =
      d k * matrixApply Q g k + diagonalCommutator Q d g k +
        matrixApply Q (fun l => d l * (m l - g l)) k := by
  simp only [diagonalCommutator, matrixApply, linearScore, mul_sub, sum_sub_distrib]
  ring

/-- Project the actual buffered Adam coordinate update, AI1–AI2.
The stored moments are free inputs, not replaced by the partial CE gradient. -/
theorem projected_bufferedAdamUpdate (b1 b2 eps η ρ : ℝ) (t : ℕ)
    (Q : K → K → ℝ) (w m v : K → ℝ) (k : K) :
    matrixApply Q (fun l => bufferedAdamUpdate b1 b2 eps η ρ t (w l) (m l) (v l)) k =
      (1 - η * ρ) * matrixApply Q w k -
        η * matrixApply Q (fun l => adamDiagonal b2 eps t (v l) * correctedMoment b1 t (m l)) k := by
  simp_rw [bufferedAdamUpdate_factor]
  simpa only [mul_assoc] using matrixApply_sub_smul Q w
    (fun l => adamDiagonal b2 eps t (v l) * correctedMoment b1 t (m l)) (1 - η * ρ) η k

/-- A nonzero commutator creates residual forcing although Q annihilates
the actual RD5 CE gradient. Its energy is exactly 1/180, AI2. -/
theorem diagonal_mixing_witness :
    let Q : Bool → Bool → ℝ := fun k l =>
      if k then (if l then 1 / 5 else -2 / 5) else (if l then -2 / 5 else 4 / 5)
    let g : Bool → ℝ := fun k => if k then -1 else -1 / 2
    let d : Bool → ℝ := fun k => if k then 1 / 2 else 2 / 3
    (∀ k, matrixApply Q g k = 0) ∧ ∑ k, diagonalCommutator Q d g k ^ 2 = 1 / 180 := by
  dsimp
  constructor
  · intro k; cases k <;> norm_num [matrixApply, linearScore]
  · norm_num [diagonalCommutator, matrixApply, linearScore]

end Transformer.Grokking.GradientEvidence
