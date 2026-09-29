/-
# A cube of independent FP4 stochastic-rounding coins

arXiv:2601.22813v2, §3.1: one SR coin per FP4 element, with its
dequantized expectation equal to the original element when fixed scales are
positive and the normalized element lies in `[-6, 6]`.
-/

import Transformer.Quartet.Section3_EdenCoins
import Mathlib.MeasureTheory.Integral.Pi

open MeasureTheory

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- One independent uniform SR coin per FP4 entry (§3.1). -/
def srCoinCube (k : ℕ) : Set ((Fin (2 ^ k) × Fin 16) → ℝ) :=
  Set.univ.pi (fun _ => Set.Icc (0 : ℝ) 1)

/-- A function of one coin integrates over the whole cube as it does over a
single uniform coin (§3.1, the expectation over `ω`). -/
theorem integral_srCoinCube_eval {α : Type*} [Fintype α] (g : ℝ → ℝ) (a : α) :
    ∫ u in Set.univ.pi (fun _ : α => Set.Icc (0 : ℝ) 1), g (u a) =
      ∫ t in (0 : ℝ)..1, g t := by
  classical
  rw [intervalIntegral.integral_of_le zero_le_one, ← integral_Icc_eq_integral_Ioc]
  have hvol : (volume : Measure (α → ℝ)) = Measure.pi (fun _ => volume) := rfl
  rw [hvol, Measure.restrict_pi_pi]
  have : ∀ _ : α, IsProbabilityMeasure (volume.restrict (Set.Icc (0 : ℝ) 1)) :=
    fun _ => ⟨by simp⟩
  have h := integral_fintype_prod_eq_prod (𝕜 := ℝ)
    (μ := fun _ : α => volume.restrict (Set.Icc (0 : ℝ) 1))
    (fun b : α => if b = a then g else fun _ => 1)
  have hL : ∀ u : α → ℝ,
      ∏ b : α, (if b = a then g else fun _ => (1 : ℝ)) (u b) = g (u a) := by
    intro u
    rw [Finset.prod_eq_single a (fun b _ hb => by simp [hb]) (by simp)]
    simp
  have hR : ∏ b : α, ∫ t, (if b = a then g else fun _ => (1 : ℝ)) t
      ∂(volume.restrict (Set.Icc (0 : ℝ) 1)) =
      ∫ t, g t ∂(volume.restrict (Set.Icc (0 : ℝ) 1)) := by
    rw [Finset.prod_eq_single a (fun b _ hb => by simp [hb]) (by simp)]
    simp
  simpa only [hL, hR] using h

/-- Integrability of a one-coin observable on the full cube. -/
theorem integrable_srCoinCube_eval {α : Type*} [Fintype α] {g : ℝ → ℝ}
    (hg : Integrable g (volume.restrict (Set.Icc (0 : ℝ) 1))) (a : α) :
    Integrable (fun u : α → ℝ => g (u a))
      (volume.restrict (Set.univ.pi (fun _ : α => Set.Icc (0 : ℝ) 1))) := by
  classical
  have hvol : (volume : Measure (α → ℝ)) = Measure.pi (fun _ => volume) := rfl
  rw [hvol, Measure.restrict_pi_pi]
  have hf : ∀ b : α, Integrable (if b = a then g else fun _ => (1 : ℝ))
      (volume.restrict (Set.Icc (0 : ℝ) 1)) := by
    intro b
    by_cases hb : b = a
    · simpa [hb] using hg
    · simp [hb]
  have h := Integrable.fintype_prod
    (μ := fun _ : α => volume.restrict (Set.Icc (0 : ℝ) 1)) hf
  have hL : (fun u : α → ℝ =>
      ∏ b : α, (if b = a then g else fun _ => (1 : ℝ)) (u b)) =
      fun u => g (u a) := by
    funext u
    rw [Finset.prod_eq_single a (fun b _ hb => by simp [hb]) (by simp)]
    simp
  rw [← hL]
  exact h

/-- Dequantize a stochastic FP4 entry using fixed group and tensor scales.
The scales are allowed to depend on the input and fixed RHT seed, but not on
the entry's SR coin (§3.1). -/
noncomputable def qSRScaledAt (g : Fin (2 ^ k) → ℝ) (t : ℝ)
    (y : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) (u : ℝ) : ℝ :=
  sr fp4 (y i j / (g i * t)) u * g i * t

/-- A positive, non-clipping choice of scales makes one FP4 entry unbiased.
This is §3.1's SR identity with its scale assumptions exposed. -/
theorem integral_qSRScaledAt (g : Fin (2 ^ k) → ℝ) (t : ℝ)
    (y : Fin (2 ^ k) → Fin 16 → ℝ) (ht : 0 < t)
    (hg : ∀ i, 0 < g i)
    (hclip : ∀ i j, |y i j / (g i * t)| ≤ 6)
    (i : Fin (2 ^ k)) (j : Fin 16) :
    ∫ u in (0 : ℝ)..1, qSRScaledAt g t y i j u = y i j := by
  obtain ⟨hlo, hhi⟩ := abs_le.mp (hclip i j)
  have hsr := integral_sr (G := fp4) (x := y i j / (g i * t))
    (by unfold fp4; exact Set.toFinite _) ⟨-6, by norm_num [fp4], hlo⟩
    ⟨6, by norm_num [fp4], hhi⟩
  unfold qSRScaledAt
  rw [intervalIntegral.integral_mul_const, intervalIntegral.integral_mul_const, hsr,
    mul_assoc, div_mul_cancel₀ _ (mul_pos (hg i) ht).ne']

/-- The dequantized value of one FP4 entry is integrable over its coin. -/
theorem integrable_qSRScaledAt (g : Fin (2 ^ k) → ℝ) (t : ℝ)
    (y : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) :
    Integrable (qSRScaledAt g t y i j)
      (volume.restrict (Set.Icc (0 : ℝ) 1)) := by
  unfold qSRScaledAt
  exact ((integrable_sr fp4 (y i j / (g i * t))).mul_const (g i)).mul_const t
end Quartet
end Transformer
