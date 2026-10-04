/-
# Estimating the noise batch and the peak learning rate

arXiv:2405.14578, §3.2, and Appendix G.  For nonzero `S, E, S_min, E_min`, eq. (20) is
equivalent to each line of eq. (46), the last `1/S = -(E_min/S_min)(1/E) + 1/S_min`
(`tradeoff_tfae`).  So a run at a constant batch `B > 0` has
`1/S = -B_crit(1/E) + 1/S_min`, eq. (22) (`inv_totalSteps_eq`), with the critical batch
`B_crit = E_min/S_min` of eq. (21) for the paper's `B_noise ≈ B_crit`: the mean of `B_noise`
over the run, `B_noise` itself at constant `μ, σ, H` (`critBatch_const`).  The points
`(1/E, 1/S)` of any two batches lie on a line of slope `-B_crit`, which a linear fit recovers
(`inv_totalSteps_slope`).

Inverting eq. (13), a pair `(B, ε_opt(B))` and `B_noise` determine
`ε_max = (ε_opt(B)/2)(√(B_noise/B) + √(B/B_noise))` (`lrPeak_eq_mul`), so its mean over a grid of
batches, eq. (23), is `ε_max` (`expect_lrPeak`).  Inverting the form
`ε_opt(B) = ε_max/(1 + B_noise/B)^α` of eq. (A.3) of arXiv:1812.06162 gives eq. (24) likewise
(`lrCentral_mul_rpow`, `expect_lrCentral`), exactly for SGD at `α = 1` (`stepOpt_mul`); its
`α = 0.5` for Adam is `NoiseScale.adam_lrCentral_half`.
-/

import Mathlib.Algebra.BigOperators.Expect
import Transformer.NoiseScale.SectionE_Optimization
import Transformer.Surge.Section2_Tradeoff

open MeasureTheory Real
open scoped BigOperators

namespace Transformer.Surge

open Transformer.NoiseScale

/-- **Eq. (46)**: for nonzero `S, E, S_min, E_min`, eq. (20) is equivalent to each line of
eq. (46), the last `1/S = -(E_min/S_min)(1/E) + 1/S_min`. -/
theorem tradeoff_tfae {S E Smin Emin : ℝ} (hS : S ≠ 0) (hE : E ≠ 0) (hSm : Smin ≠ 0)
    (hEm : Emin ≠ 0) :
    [(S / Smin - 1) * (E / Emin - 1) = 1,
      S * E - Smin * E - S * Emin + Smin * Emin = Smin * Emin,
      Smin * E + S * Emin = S * E,
      Smin / S + Emin / E = 1,
      1 / S = -(Emin / Smin) * (1 / E) + 1 / Smin].TFAE := by
  tfae_have 1 ↔ 2 := by
    rw [div_sub_one hSm, div_sub_one hEm, div_mul_div_comm,
      div_eq_one_iff_eq (mul_ne_zero hSm hEm)]
    constructor <;> intro h <;> linear_combination h
  tfae_have 2 ↔ 3 := by constructor <;> intro h <;> linear_combination -h
  tfae_have 3 ↔ 4 := by rw [div_add_div _ _ hS hE, div_eq_one_iff_eq (mul_ne_zero hS hE)]
  tfae_have 4 ↔ 5 := by
    have h : 1 / S - (-(Emin / Smin) * (1 / E) + 1 / Smin) =
        (Smin / S + Emin / E - 1) / Smin := by
      field_simp
      ring
    rw [← sub_eq_zero, ← sub_eq_zero (a := 1 / S), h, div_eq_zero_iff, or_iff_left hSm]
  tfae_finish

/-- The hypotheses of `tradeoff_tfae` are satisfiable: `S = E = S_min = E_min = 1`. -/
example := tradeoff_tfae one_ne_zero one_ne_zero one_ne_zero one_ne_zero

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {T : Type*} [MeasurableSpace T]
  {ν : Measure T} [IsFiniteMeasure ν]

/-- **Eq. (22)**: a run with `S_min = ∫ds > 0` and `E_min = ∫B_noise ds ≥ 0`, at a constant
batch `B > 0`, takes `S` steps and `E` examples with `1/S = -B_crit(1/E) + 1/S_min`. -/
theorem inv_totalSteps_eq {μ σ : T → ι → ℝ} {H : T → Matrix ι ι ℝ} (hν : 0 < ν.real Set.univ)
    (hi : Integrable (runNoiseBatch μ σ H) ν) (hE : 0 ≤ ∫ s, runNoiseBatch μ σ H s ∂ν) {B : ℝ}
    (hB : 0 < B) :
    1 / totalSteps ν (runNoiseBatch μ σ H) (fun _ => B) = -critBatch ν (runNoiseBatch μ σ H) *
      (1 / totalExamples ν (runNoiseBatch μ σ H) (fun _ => B)) + 1 / ν.real Set.univ := by
  rw [totalSteps_const_eq hi, totalExamples_eq hi (integrable_const B), integral_const,
    smul_eq_mul, critBatch]
  generalize ∫ s, runNoiseBatch μ σ H s ∂ν = Em at hE ⊢
  generalize ν.real Set.univ = Sm at hν ⊢
  have : 0 < Em + Sm * B := by positivity
  field_simp
  ring

/-- The hypotheses of `inv_totalSteps_eq` are satisfiable: one step at `μ = σ = 1`, all
`H_ij = 1`, `B = 1`. -/
example := inv_totalSteps_eq (ν := Measure.dirac ()) (μ := fun _ (_ : Fin 2) => 1)
  (σ := fun _ _ => 1) (H := fun _ => Matrix.of fun _ _ => 1) (by simp) .of_finite
  (by simp [runNoiseBatch, noiseBatch, offSum, Fin.sum_univ_two]; positivity) one_pos

/-- §3.2: the points `(1/E, 1/S)` of eq. (22) at two constant batches `B₁ ≠ B₂` determine the
slope `-B_crit`, which a linear fit therefore recovers. -/
theorem inv_totalSteps_slope {μ σ : T → ι → ℝ} {H : T → Matrix ι ι ℝ}
    (hν : 0 < ν.real Set.univ) (hi : Integrable (runNoiseBatch μ σ H) ν)
    (hE : 0 ≤ ∫ s, runNoiseBatch μ σ H s ∂ν) {B₁ B₂ : ℝ} (h₁ : 0 < B₁) (h₂ : 0 < B₂)
    (h : B₁ ≠ B₂) :
    (1 / totalSteps ν (runNoiseBatch μ σ H) (fun _ => B₁) -
      1 / totalSteps ν (runNoiseBatch μ σ H) (fun _ => B₂)) /
      (1 / totalExamples ν (runNoiseBatch μ σ H) (fun _ => B₁) -
        1 / totalExamples ν (runNoiseBatch μ σ H) (fun _ => B₂)) =
      -critBatch ν (runNoiseBatch μ σ H) := by
  have hne : 1 / totalExamples ν (runNoiseBatch μ σ H) (fun _ => B₁) -
      1 / totalExamples ν (runNoiseBatch μ σ H) (fun _ => B₂) ≠ 0 := by
    rw [totalExamples_eq hi (integrable_const _), totalExamples_eq hi (integrable_const _),
      integral_const, integral_const, smul_eq_mul, smul_eq_mul, sub_ne_zero, one_div, one_div,
      ne_eq, inv_inj, add_right_inj]
    exact fun h' => h (mul_left_cancel₀ hν.ne' h')
  rw [inv_totalSteps_eq hν hi hE h₁, inv_totalSteps_eq hν hi hE h₂, div_eq_iff hne]
  ring

/-- The hypotheses of `inv_totalSteps_slope` are satisfiable: one step at `μ = σ = 1`, all
`H_ij = 1`, `B₁ = 1`, `B₂ = 2`. -/
example := inv_totalSteps_slope (ν := Measure.dirac ()) (μ := fun _ (_ : Fin 2) => 1)
  (σ := fun _ _ => 1) (H := fun _ => Matrix.of fun _ _ => 1) (by simp) .of_finite
  (by simp [runNoiseBatch, noiseBatch, offSum, Fin.sum_univ_two]; positivity) one_pos two_pos
  (by norm_num)

variable {μ σ : ι → ℝ} {H : Matrix ι ι ℝ} {B : ℝ}

/-- §3.2: "we only need a simple search for a pair of (batch size, optimal learning rate) to
determine" `ε_max`: inverting eq. (13), `ε_max = (ε_opt(B)/2)(√(B_noise/B) + √(B/B_noise))`, when
`ΣH_ii > 0`, `S > 0`, `B > 0`. -/
theorem lrPeak_eq_mul (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) (hB : 0 < B) :
    lrPeak μ σ H = lrSign (signLin μ σ B) μ H / 2 *
      (√(noiseBatch μ σ H / B) + √(B / noiseBatch μ σ H)) := by
  have h : 0 < √(noiseBatch μ σ H / B) + √(B / noiseBatch μ σ H) := by
    have := one_le_half_sqrt_add (noiseBatch_pos hD hS) hB
    linarith
  have h0 := h.ne'
  rw [lrSign_signLin_eq_peak hD hS hB]
  field_simp

/-- The hypotheses of `lrPeak_eq_mul` are satisfiable: `μ = σ = 1`, all `H_ij = 1`, `B = 1`. -/
example := lrPeak_eq_mul (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) one_pos

/-- **Eq. (23)**: over a nonempty grid of batches `B > 0`, the mean of
`(ε_opt(B)/2)(√(B_noise/B) + √(B/B_noise))` is `ε_max`, when `ΣH_ii > 0` and `S > 0`. -/
theorem expect_lrPeak (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) {s : Finset ℝ}
    (hs : s.Nonempty) (hpos : ∀ B ∈ s, 0 < B) :
    𝔼 B ∈ s, lrSign (signLin μ σ B) μ H / 2 *
      (√(noiseBatch μ σ H / B) + √(B / noiseBatch μ σ H)) = lrPeak μ σ H := by
  rw [Finset.expect_congr rfl fun B hB => (lrPeak_eq_mul hD hS (hpos B hB)).symm,
    Finset.expect_const hs]

/-- The hypotheses of `expect_lrPeak` are satisfiable: `μ = σ = 1`, all `H_ij = 1`, the grid
`{1}`. -/
example := expect_lrPeak (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two])
  (Finset.singleton_nonempty 1) (by simp)

/-- §3.2, after eq. (A.3) of arXiv:1812.06162: a pair `(B, ε_opt(B))` with
`ε_opt(B) = ε_max/(1 + B_noise/B)^α` determines `ε_max = ε_opt(B)(1 + B_noise/B)^α`, when
`B_noise ≥ 0` and `B > 0`. -/
theorem lrCentral_mul_rpow (εs : ℝ) {Bs : ℝ} (hBs : 0 ≤ Bs) (α : ℝ) (hB : 0 < B) :
    lrCentral εs Bs α B * (1 + Bs / B) ^ α = εs :=
  div_mul_cancel₀ εs (Real.rpow_pos_of_pos (by positivity) α).ne'

/-- The hypotheses of `lrCentral_mul_rpow` are satisfiable: `B_noise = 0`, `B = 1`. -/
example (εs α : ℝ) := lrCentral_mul_rpow εs le_rfl α one_pos

/-- **Eq. (24)**: over a nonempty grid of batches `B > 0` at which
`ε_opt(B) = ε_max/(1 + B_noise/B)^α`, the mean of `ε_opt(B)(1 + B_noise/B)^α` is `ε_max`, when
`B_noise ≥ 0`. -/
theorem expect_lrCentral (εs : ℝ) {Bs : ℝ} (hBs : 0 ≤ Bs) (α : ℝ) {s : Finset ℝ}
    (hs : s.Nonempty) (hpos : ∀ B ∈ s, 0 < B) :
    𝔼 B ∈ s, lrCentral εs Bs α B * (1 + Bs / B) ^ α = εs := by
  rw [Finset.expect_congr rfl fun B hB => lrCentral_mul_rpow εs hBs α (hpos B hB),
    Finset.expect_const hs]

/-- The hypotheses of `expect_lrCentral` are satisfiable: `B_noise = 0`, the grid `{1}`. -/
example (εs α : ℝ) := expect_lrCentral εs le_rfl α (Finset.singleton_nonempty 1) (by simp)

omit [DecidableEq ι] in
/-- Eq. (24) at `α = 1` is exact for SGD: the best step `ε_opt(B)` of eq. (2.6) of
arXiv:1812.06162 has `ε_opt(B)(1 + B_noise/B) = ε_max`, when `B_noise ≥ 0` and `B > 0`. -/
theorem stepOpt_mul {G : ι → ℝ} {S : Matrix ι ι ℝ} (hN : 0 ≤ noiseScale G H S) (hB : 0 < B) :
    stepOpt G H S B * (1 + noiseScale G H S / B) = stepMax G H := by
  rw [← lrCentral_one, ← Real.rpow_one (1 + noiseScale G H S / B)]
  exact lrCentral_mul_rpow _ hN 1 hB

/-- The hypotheses of `stepOpt_mul` are satisfiable: `G = H = Σ = 1`, `B = 1`. -/
example := stepOpt_mul (G := fun _ : Unit => 1) (H := 1) (S := 1) (by simp [noiseScale]) one_pos

end Transformer.Surge
