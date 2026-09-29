/-
# Integrating one coin out of the cube of `MS-EDEN`

The expectation of Algorithm 1 (`Transformer.Quartet.mean`) integrates over the sign seeds, and
over a cube of coins, one per group.  A function of a single coin `u_g` integrates against the
cube as against the unit interval — the other coordinates have total mass `1` — so a sum over
the groups of a stochastic rounding of a fixed argument integrates to `2^k` times that argument
(`integral_sum_sr`), by the unbiasedness of `SR` (`integral_sr`).  And the mean of a function
whose integral does not depend on the seed is that integral (`mean_eq_of_forall`).

Source: arXiv:2601.22813v2, §3.1 (`SR` is unbiased), Algorithm 1 (one seed `ω_SR` per group).
-/

import Transformer.Quartet.Section3_Eden
import Transformer.Quartet.Section3_Unbiased
import Transformer.Quartet.Fp8Grid
import Mathlib.MeasureTheory.Integral.Pi

open MeasureTheory

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- A function of one coordinate of the cube of coins is integrable as soon as it is
integrable on the unit interval. -/
theorem integrable_pi_eval {g : ℝ → ℝ} (hg : Integrable g (volume.restrict (Set.Icc (0 : ℝ) 1)))
    (i : Fin (2 ^ k)) :
    Integrable (fun u : Fin (2 ^ k) → ℝ => g (u i))
      (volume.restrict (Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1))) := by
  have hvol : (volume : Measure (Fin (2 ^ k) → ℝ)) = Measure.pi (fun _ => volume) := rfl
  rw [hvol, Measure.restrict_pi_pi]
  have hf : ∀ j : Fin (2 ^ k), Integrable (if j = i then g else fun _ => (1 : ℝ))
      (volume.restrict (Set.Icc (0 : ℝ) 1)) := by
    intro j
    by_cases hj : j = i
    · simpa [hj] using hg
    · simp [hj]
  have h := Integrable.fintype_prod (μ := fun _ : Fin (2 ^ k) => volume.restrict (Set.Icc (0 : ℝ) 1))
    hf
  have hL : (fun u : Fin (2 ^ k) → ℝ =>
      ∏ j : Fin (2 ^ k), (if j = i then g else fun _ => (1 : ℝ)) (u j)) = fun u => g (u i) := by
    funext u
    rw [Finset.prod_eq_single i (fun j _ hj => by simp [hj]) (by simp)]
    simp
  rw [← hL]
  exact h

/-- **A function of one coin integrates against the cube as against the unit interval.** -/
theorem integral_pi_eval (g : ℝ → ℝ) (i : Fin (2 ^ k)) :
    ∫ u in Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1), g (u i) =
      ∫ t in (0 : ℝ)..1, g t := by
  rw [intervalIntegral.integral_of_le zero_le_one, ← integral_Icc_eq_integral_Ioc]
  have hvol : (volume : Measure (Fin (2 ^ k) → ℝ)) = Measure.pi (fun _ => volume) := rfl
  rw [hvol, Measure.restrict_pi_pi]
  have : ∀ _ : Fin (2 ^ k), IsProbabilityMeasure (volume.restrict (Set.Icc (0 : ℝ) 1)) :=
    fun _ => ⟨by simp⟩
  have h := integral_fintype_prod_eq_prod (𝕜 := ℝ)
    (μ := fun _ : Fin (2 ^ k) => volume.restrict (Set.Icc (0 : ℝ) 1))
    (fun j : Fin (2 ^ k) => if j = i then g else fun _ => 1)
  have hL : ∀ u : Fin (2 ^ k) → ℝ,
      ∏ j : Fin (2 ^ k), (if j = i then g else fun _ => (1 : ℝ)) (u j) = g (u i) := by
    intro u
    rw [Finset.prod_eq_single i (fun j _ hj => by simp [hj]) (by simp)]
    simp
  have hR : ∏ j : Fin (2 ^ k), ∫ x, (if j = i then g else fun _ => (1 : ℝ)) x
      ∂(volume.restrict (Set.Icc (0 : ℝ) 1)) = ∫ x, g x ∂(volume.restrict (Set.Icc (0 : ℝ) 1)) := by
    rw [Finset.prod_eq_single i (fun j _ hj => by simp [hj]) (by simp)]
    simp
  simpa only [hL, hR] using h

/-- The stochastic rounding of a fixed argument is a measurable function of the coin. -/
theorem measurable_sr (G : Set ℝ) (x : ℝ) : Measurable (sr G x) := by
  unfold sr
  exact Measurable.ite (measurableSet_lt (measurable_id.mul_const _) measurable_const)
    measurable_const measurable_const

/-- And a bounded one, so integrable over the unit interval. -/
theorem integrable_sr (G : Set ℝ) (x : ℝ) :
    Integrable (sr G x) (volume.restrict (Set.Icc (0 : ℝ) 1)) := by
  refine Integrable.of_bound (measurable_sr G x).aestronglyMeasurable
    (|ceilOn G x| + |floorOn G x|) (Filter.Eventually.of_forall fun u => ?_)
  rw [Real.norm_eq_abs]
  unfold sr
  split
  · exact le_add_of_nonneg_right (abs_nonneg _)
  · exact le_add_of_nonneg_left (abs_nonneg _)

/-- **A sum over the groups of the stochastic rounding onto E4M3 integrates to `2^k` times its
argument**, for an argument in `[0, 448]`: each group has its own coin, and `SR` is unbiased
(`integral_sr`). -/
theorem integral_sum_sr {z : ℝ} (hz₀ : 0 ≤ z) (hz : z ≤ 448) :
    ∫ u in Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1),
        ∑ i' : Fin (2 ^ k), sr fp8 z (u i') = 2 ^ k * z := by
  have h0 : (0 : ℝ) ∈ fp8 := ⟨by norm_num, 0, 0, by norm_num⟩
  have h448 : (448 : ℝ) ∈ fp8 := ⟨by norm_num, 14, 5, by norm_num⟩
  rw [integral_finsetSum _ fun i' _ => integrable_pi_eval (integrable_sr fp8 z) i']
  simp only [integral_pi_eval (sr fp8 z), integral_sr fp8_finite ⟨0, h0, hz₀⟩ ⟨448, h448, hz⟩]
  simp

/-- The hypotheses of `integral_sum_sr` are satisfiable, at `z = 448`, the top of the grid. -/
example : ∫ u in Set.univ.pi (fun _ : Fin (2 ^ 1) => Set.Icc (0 : ℝ) 1),
    ∑ i' : Fin (2 ^ 1), sr fp8 448 (u i') = 2 ^ 1 * 448 :=
  integral_sum_sr (by norm_num) le_rfl

/-- **The mean of a function whose integral over the coins is the same for every seed is that
integral.** -/
theorem mean_eq_of_forall {f : (Fin (2 ^ k) → Fin 16 → Bool) → (Fin (2 ^ k) → ℝ) → ℝ} {a : ℝ}
    (h : ∀ ε, ∫ u in Set.univ.pi (fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1), f ε u = a) :
    mean k f = a := by
  unfold mean
  rw [Finset.sum_congr rfl fun ε _ => h ε, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hc : (0 : ℝ) < (Fintype.card (Fin (2 ^ k) → Fin 16 → Bool) : ℝ) := by
    exact_mod_cast Fintype.card_pos
  field_simp

/-- The hypothesis of `mean_eq_of_forall` is satisfiable: a constant function. -/
example : mean 0 (fun _ _ => (3 : ℝ)) = 3 :=
  mean_eq_of_forall fun _ => by
    rw [setIntegral_const, measureReal_def, Set.pi_univ_Icc, Real.volume_Icc_pi]
    simp

end Quartet
end Transformer
