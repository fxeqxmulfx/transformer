module

public import Transformer.Grokking.GradientEvidence.AdaptiveForcing

/-!
# Loss controls one part of adaptive forcing

Source: `review/flat_spectrum.md`, *Adaptive interaction control*, AI3.
Combine the actual CE-gradient theorem with an explicit diagonal-scale bound.
Noncommutation and the discrepancy from the saved first moment remain visible.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset

variable {I C K : Type*} [Fintype I] [Fintype C] [Fintype K]

omit [Fintype I] in
/-- A bounded diagonal map scales squared energy by at most its squared
coordinate bound, AI3. No sign or constancy of the scale is presumed. -/
theorem diagonal_energy_le (D A : C → K → ℝ) (M : ℝ)
    (hD : ∀ c k, D c k ^ 2 ≤ M ^ 2) :
    matrixEnergy (fun c k => D c k * A c k) ≤ M ^ 2 * matrixEnergy A := by
  simp only [matrixEnergy, mul_sum, mul_pow]
  exact sum_le_sum fun c _ => sum_le_sum fun k _ =>
    mul_le_mul_of_nonneg_right (hD c k) (sq_nonneg _)

omit [Fintype I] in
/-- Three explicit forcing terms have this conservative energy bound,
AI3. Their signed cross terms can cancel; equality is never assumed. -/
theorem three_term_energy_le (A B F : C → K → ℝ) :
    matrixEnergy (fun c k => A c k + B c k + F c k) ≤
      3 * (matrixEnergy A + matrixEnergy B + matrixEnergy F) := by
  calc
    _ ≤ ∑ c, ∑ k, 3 * (A c k ^ 2 + B c k ^ 2 + F c k ^ 2) := by
      apply sum_le_sum
      intro c hc
      apply sum_le_sum
      intro k hk
      nlinarith [sq_nonneg (A c k - B c k), sq_nonneg (A c k - F c k),
        sq_nonneg (B c k - F c k)]
    _ = _ := by simp only [matrixEnergy, mul_add, sum_add_distrib, mul_sum]

variable [DecidableEq C] [DecidableEq K]

/-- The scaled projected *actual* readout gradient is loss-controlled,
AI3. This is one term of the adaptive forcing, not its entirety. -/
theorem scaled_readoutGradient_energy_le (T : I → K → ℝ) (y : I → C)
    (E D : C → K → ℝ) (Q : K → K → ℝ) (M : ℝ)
    (hD : ∀ c k, D c k ^ 2 ≤ M ^ 2) :
    matrixEnergy (fun c k => D c k * matrixApply Q (readoutGradient T y E c) k) ≤
      M ^ 2 * (2 * readoutLoss T y E *
        (matrixEnergy (fun i => matrixApply Q (T i)) / Fintype.card I)) := by
  exact (diagonal_energy_le D _ M hD).trans
    (mul_le_mul_of_nonneg_left (projected_readoutGradient_energy_le T y E Q) (sq_nonneg M))

/-- **Objective-controlled adaptive forcing plus its two missing terms**,
AI3. The commutator and saved-moment discrepancy are not declared small. -/
theorem adaptive_forcing_energy_le (T : I → K → ℝ) (y : I → C)
    (E D m : C → K → ℝ) (Q : K → K → ℝ) (M : ℝ)
    (hD : ∀ c k, D c k ^ 2 ≤ M ^ 2) :
    matrixEnergy (fun c => matrixApply Q (fun k => D c k * m c k)) ≤
      3 * (M ^ 2 * (2 * readoutLoss T y E *
          (matrixEnergy (fun i => matrixApply Q (T i)) / Fintype.card I)) +
        matrixEnergy (fun c => diagonalCommutator Q (D c) (readoutGradient T y E c)) +
        matrixEnergy (fun c => matrixApply Q
          (fun k => D c k * (m c k - readoutGradient T y E c k)))) := by
  calc
    _ = matrixEnergy (fun c k => D c k * matrixApply Q (readoutGradient T y E c) k +
        diagonalCommutator Q (D c) (readoutGradient T y E c) k +
        matrixApply Q (fun l => D c l * (m c l - readoutGradient T y E c l)) k) := by
      unfold matrixEnergy
      apply sum_congr rfl
      intro c hc
      apply sum_congr rfl
      intro k hk
      congr 1
      exact adaptive_forcing_decomposition Q (D c) (m c) (readoutGradient T y E c) k
    _ ≤ _ := by
      have ht := three_term_energy_le
        (fun c k => D c k * matrixApply Q (readoutGradient T y E c) k)
        (fun c => diagonalCommutator Q (D c) (readoutGradient T y E c))
        (fun c => matrixApply Q (fun k => D c k * (m c k - readoutGradient T y E c k)))
      have hg := scaled_readoutGradient_energy_le T y E D Q M hD
      linarith

example := diagonal_energy_le (fun (c k : Bool) => if c = k then (2 : ℝ) else 1)
  (fun (c k : Bool) => if c = k then (3 : ℝ) else -1) 2
  (by intro c k; split <;> norm_num)
example := scaled_readoutGradient_energy_le
  (fun (i : Bool) (_ : Fin 1) => if i then (2 : ℝ) else 1) id
  (fun (c : Bool) (_ : Fin 1) => if c then (1 : ℝ) else -1)
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 1)
  (fun (_ : Fin 1) (_ : Fin 1) => 1) 2 (by intro c k; cases c <;> norm_num)
example := adaptive_forcing_energy_le
  (fun (i : Bool) (_ : Fin 1) => if i then (2 : ℝ) else 1) id
  (fun (c : Bool) (_ : Fin 1) => if c then (1 : ℝ) else -1)
  (fun (c : Bool) (_ : Fin 1) => if c then (2 : ℝ) else 1)
  (fun (c : Bool) (_ : Fin 1) => if c then (1 / 3 : ℝ) else -1 / 3)
  (fun (_ : Fin 1) (_ : Fin 1) => 1) 2 (by intro c k; cases c <;> norm_num)

end Transformer.Grokking.GradientEvidence
