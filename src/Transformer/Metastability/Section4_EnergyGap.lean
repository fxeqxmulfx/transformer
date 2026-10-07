/-
# Energy deficits and a common spherical cap

For the energy printed in arXiv:2410.06833v1, §1,
`eq: interaction.energy`, every pair contributes a nonnegative deficit
from the maximum `exp β`. The deficit of one pair is bounded by their sum.
Consequently, energy sufficiently close to `1/(2β)` puts every token in
a cap centred at any chosen token. This supplies the geometric step for the
nonempty energy window asked for in §4, `sec: energy.levels`.
-/

import Transformer.Metastability.EnergyScale
import Transformer.Metastability.AttnFlow
import Transformer.Metastability.InitialUniform
import Mathlib.MeasureTheory.Function.LocallyIntegrable

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Metastability
variable {d n : ℕ}

/-- Each unit-vector interaction is at most `exp β` for `β ≥ 0`.
Source: arXiv:2410.06833v1, §1, `eq: interaction.energy`. -/
theorem exp_pair_le_exp (β : ℝ) (hβ : 0 ≤ β) (x y : SSphere d) :
    Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ≤ Real.exp β := by
  apply Real.exp_le_exp.mpr
  have h := real_inner_le_norm (x : EucSpace d) (y : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2] at h
  nlinarith

/-- A single pair's deficit is bounded by the sum of all pair deficits.
The ordered double sum has exactly `n²` terms, including the diagonal.
Source: arXiv:2410.06833v1, §1, `eq: interaction.energy`; §4,
`sec: energy.levels` (auxiliary estimate). -/
theorem pair_deficit_le_total (β : ℝ) (hβ : 0 ≤ β) (X : SphereTuple d n)
    (i j : Idx n) :
    Real.exp β - Real.exp (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)) ≤
      (n : ℝ) ^ 2 * Real.exp β - ∑ k : Idx n, ∑ l : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (X k : EucSpace d) (X l : EucSpace d)) := by
  let D : Idx n → Idx n → ℝ := fun k l => Real.exp β - Real.exp
    (β * inner (𝕜 := ℝ) (X k : EucSpace d) (X l : EucSpace d))
  have hnD (k l : Idx n) : 0 ≤ D k l := sub_nonneg.mpr (exp_pair_le_exp β hβ _ _)
  have h : D i j ≤ ∑ k : Idx n, ∑ l : Idx n, D k l :=
    (Finset.single_le_sum (fun l _ => hnD i l) (Finset.mem_univ j)).trans
      (Finset.single_le_sum (fun k _ => Finset.sum_nonneg (fun l _ => hnD k l))
        (Finset.mem_univ i))
  dsimp [D] at h
  simp_rw [Finset.sum_sub_distrib] at h
  simpa [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, pow_two, mul_assoc] using h

/-- Energy plus the normalized deficit of any one pair is at most
`1/(2β)`. No dynamics or separation hypothesis is used.
Source: arXiv:2410.06833v1, §1, `eq: interaction.energy`; §4,
`sec: energy.levels` (auxiliary estimate). -/
theorem energy_add_pair_deficit_le (β : ℝ) (hβ : 0 < β) (hn : 1 ≤ n)
    (X : SphereTuple d n) (i j : Idx n) :
    Eβ d n β X + (Real.exp β - Real.exp
      (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d))) /
      (2 * β * Real.exp β * (n : ℝ) ^ 2) ≤ 1 / (2 * β) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hD : 0 < 2 * β * Real.exp β * (n : ℝ) ^ 2 := by positivity
  have he : Eβ d n β X = (∑ k : Idx n, ∑ l : Idx n, Real.exp
      (β * inner (𝕜 := ℝ) (X k : EucSpace d) (X l : EucSpace d))) /
      (2 * β * Real.exp β * (n : ℝ) ^ 2) := by unfold Eβ; ring
  rw [he, ← add_div, div_le_iff₀ hD]
  have hc : 1 / (2 * β) * (2 * β * Real.exp β * (n : ℝ) ^ 2) =
      (n : ℝ) ^ 2 * Real.exp β := by field_simp
  rw [hc]
  linarith [pair_deficit_le_total β hβ.le X i j]

/-- A sufficiently small energy deficit forces every token into the
height-`ε` cap centred at token `j`. If one inner product were below
`1 - ε`, that pair alone would exceed the available total deficit.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels`, with the normalization
of §1, `eq: interaction.energy`, and caps of `eq: cones`. -/
theorem high_energy_mem_cap (β ε : ℝ) (hβ : 0 < β) (hn : 1 ≤ n)
    (X : SphereTuple d n) (j : Idx n)
    (hE : 1 / (2 * β) - (Real.exp β - Real.exp (β * (1 - ε))) /
        (2 * β * Real.exp β * (n : ℝ) ^ 2) < Eβ d n β X) :
    ∀ i : Idx n, X i ∈ sphericalCap d (X j) ε := by
  intro i
  by_contra hi
  have hc : inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d) < 1 - ε :=
    lt_of_not_ge hi
  have he : Real.exp (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)) ≤
      Real.exp (β * (1 - ε)) := Real.exp_le_exp.mpr (by nlinarith)
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hD : 0 < 2 * β * Real.exp β * (n : ℝ) ^ 2 := by positivity
  have h := energy_add_pair_deficit_le β hβ hn X i j
  have hle := div_le_div_of_nonneg_right (show Real.exp β - Real.exp (β * (1 - ε)) ≤
    Real.exp β - Real.exp (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)) by linarith)
    hD.le
  linarith

/-- Consensus attains the printed maximum `1/(2β)`.
Source: arXiv:2410.06833v1, §1, `eq: interaction.energy` and the following
paragraph identifying global maxima as clusters. -/
theorem energy_consensus (hn : 1 ≤ n) (β : ℝ) (hβ : 0 < β) (w : SSphere d) :
    Eβ d n β (fun _ => w) = 1 / (2 * β) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  unfold Eβ
  simp only [inner_sphere_self, mul_one, Finset.sum_const, Finset.card_univ,
    Idx, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- The cap-forcing energy tolerance is positive for a positive cap height.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels` (auxiliary estimate). -/
theorem cap_energy_tolerance_pos (β ε : ℝ) (hβ : 0 < β) (hε : 0 < ε) (hn : 1 ≤ n) :
    0 < (Real.exp β - Real.exp (β * (1 - ε))) /
      (2 * β * Real.exp β * (n : ℝ) ^ 2) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  apply div_pos _ (by positivity)
  exact sub_pos.mpr (Real.exp_lt_exp.mpr (by nlinarith))

/-- The finite interaction energy is continuous in its configuration.
Source: arXiv:2410.06833v1, §1, `eq: interaction.energy`. -/
theorem continuous_energy (β : ℝ) : Continuous (Eβ d n β) := by
  unfold Eβ
  fun_prop

/-- Compactness of the configuration space makes the energy integrable
against every finite law. This justifies its expectation in §4.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels`. -/
theorem integrable_energy (β : ℝ) (P : Measure (SphereTuple d n)) [IsFiniteMeasure P] :
    Integrable (Eβ d n β) P :=
  (continuous_energy β).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)

/-- Positive temperature, two actual unit vectors, and two actual indices
witness the hypotheses of the pair-deficit estimates and consensus identity. -/
example : (0 : ℝ) < 1 ∧ 1 ≤ 2 ∧
    Eβ 2 2 1 (fun _ => basePoint 1) = 1 / 2 ∧
    Eβ 2 2 1 (fun _ => basePoint 1) + (Real.exp 1 - Real.exp
      (1 * inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2))) /
      (2 * 1 * Real.exp 1 * ((2 : ℕ) : ℝ) ^ 2) ≤ 1 / 2 := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  · simpa only [mul_one] using
      energy_consensus (n := 2) (by norm_num) 1 (by norm_num) (basePoint 1)
  · simpa only [mul_one] using energy_add_pair_deficit_le (n := 2) 1 (by norm_num)
      (by norm_num) (fun _ => basePoint 1) 0 1

/-- Consensus satisfies the strict high-energy hypothesis with positive
cap height; the geometric implication is therefore nonvacuous. -/
example : ∃ X : SphereTuple 2 2,
    1 / (2 * (1 : ℝ)) - (Real.exp 1 - Real.exp (1 * (1 - (1 / 100)))) /
      (2 * 1 * Real.exp 1 * ((2 : ℕ) : ℝ) ^ 2) < Eβ 2 2 1 X := by
  refine ⟨fun _ => basePoint 1, ?_⟩
  rw [energy_consensus (by norm_num) 1 (by norm_num)]
  have := cap_energy_tolerance_pos (n := 2) 1 (1 / 100)
    (by norm_num) (by norm_num) (by norm_num)
  linarith

/-- A Dirac probability is a finite law, as required by `integrable_energy`. -/
example : IsFiniteMeasure (Measure.dirac (fun _ : Idx 2 => basePoint 1)) := inferInstance

end Transformer.Metastability
