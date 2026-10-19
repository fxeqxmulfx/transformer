module

public import Transformer.Homogenized.Section7_SimplexGronwall

/-!
# Absorbing the actual simplex bias within the source's radius

ArXiv:2604.01978v1, §7, eq:Gronwall_E_selfcontained,
eq:martingale_sup_bound_clean and thm:clustering_random_init.
The actual finite dimension drift leaves the term 3T/d. The proved
noise energy bound 3T/d improves the source's 4T/d, leaving room
to use seven eighths of its displayed radius for the noise.

These scalar estimates prove that the remaining bias is absorbed
by exp(2T). They also verify that the exponent from the improved
noise energy still pays for the original 2n²/δ union factor.
No martingale tail bound is asserted or assumed here: the last
estimate is the numerical exponential expression used by that bound.
-/

@[expose] public section

open Set

namespace Transformer.Homogenized

/-- The exact radius displayed in the source's clustering conclusion.
Source: arXiv:2604.01978v1, §7, thm:clustering_random_init. -/
noncomputable def simplexNoiseRadius (d n : ℕ) (T δ : ℝ) : ℝ :=
  Real.sqrt (8 * T / (d : ℝ) * Real.log (2 * (n : ℝ) ^ 2 / δ))

/-- The source's logarithmic factor is at least one throughout its
parameter range. Source: arXiv:2604.01978v1, §7, thm:clustering_random_init. -/
theorem one_le_simplex_noise_log (n : ℕ) (hn : 2 ≤ n) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    1 ≤ Real.log (2 * (n : ℝ) ^ 2 / δ) := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have harg : (4 : ℝ) ≤ 2 * (n : ℝ) ^ 2 / δ :=
    (le_div_iff₀ hδ.1).2 (by nlinarith [hδ.2])
  have hlog2 := Real.one_sub_inv_le_log_of_pos (by norm_num : (0 : ℝ) < 2)
  have hlog4 : (1 : ℝ) ≤ Real.log 4 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]
    norm_num at hlog2 ⊢
    linarith
  exact hlog4.trans (Real.log_le_log (by norm_num) harg)

/-- The actual source radius has its displayed squared value.
Source: arXiv:2604.01978v1, §7, thm:clustering_random_init. -/
theorem simplexNoiseRadius_sq (d n : ℕ) (hn : 2 ≤ n) (T : ℝ) (hT : 0 ≤ T)
    (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    simplexNoiseRadius d n T δ ^ 2 = 8 * T / (d : ℝ) * Real.log (2 * (n : ℝ) ^ 2 / δ) := by
  apply Real.sq_sqrt
  have hlog := one_le_simplex_noise_log n hn δ hδ
  exact mul_nonneg (div_nonneg (by positivity) (Nat.cast_nonneg d)) (by linarith)

/-- Seven eighths of the radius plus the actual bias fits inside exp(2T)R.
Source: arXiv:2604.01978v1, §7, eq:Gronwall_E_selfcontained,
with its nonzero finite dimension bias retained. Here S represents T/d. -/
theorem overlap_bias_absorption {T S R : ℝ} (hT : 0 ≤ T) (hS : 0 ≤ S)
    (hST : 2 * S ≤ T) (hR : 0 ≤ R) (hR2 : 8 * S ≤ R ^ 2) :
    (7 / 8 : ℝ) * R + 3 * S ≤ Real.exp (2 * T) * R := by
  let B := (1 / 8 : ℝ) + 2 * T
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hB2 : 9 * S ≤ 8 * B ^ 2 := by
    dsimp [B]
    nlinarith [sq_nonneg (T - 1 / 128)]
  have hmul := mul_le_mul_of_nonneg_left hB2 hS
  have hmul2 := mul_le_mul_of_nonneg_right hR2 (sq_nonneg B)
  have hsq : (3 * S) ^ 2 ≤ (B * R) ^ 2 := by nlinarith only [hmul, hmul2]
  have hlin := le_of_sq_le_sq hsq (mul_nonneg hB hR)
  calc
    _ ≤ (1 + 2 * T) * R := by dsimp [B] at hlin; nlinarith
    _ ≤ Real.exp (2 * T) * R :=
      mul_le_mul_of_nonneg_right (by simpa [add_comm] using Real.add_one_le_exp (2 * T)) hR

/-- The actual 3T/d bias is absorbed with the source's original radius.
Source: arXiv:2604.01978v1, §7, thm:clustering_random_init.
No lower bound on T and no change to the displayed radius are needed. -/
theorem simplex_noise_bias_absorption (d n : ℕ) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (T : ℝ) (hT : 0 ≤ T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    (7 / 8 : ℝ) * simplexNoiseRadius d n T δ + 3 * T / (d : ℝ) ≤
      Real.exp (2 * T) * simplexNoiseRadius d n T δ := by
  have hdR : (2 : ℝ) ≤ d := by exact_mod_cast hd
  have hdpos : (0 : ℝ) < d := by linarith
  have hS : 0 ≤ T / (d : ℝ) := div_nonneg hT hdpos.le
  have hST : 2 * (T / (d : ℝ)) ≤ T := by
    calc
      _ = (2 * T) / (d : ℝ) := by ring
      _ ≤ T := (div_le_iff₀ hdpos).2 (by nlinarith)
  have hR2 : 8 * (T / (d : ℝ)) ≤ simplexNoiseRadius d n T δ ^ 2 := by
    rw [simplexNoiseRadius_sq d n hn T hT δ hδ]
    calc
      _ = (8 * T / (d : ℝ)) * 1 := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left (one_le_simplex_noise_log n hn δ hδ) (by positivity)
  have hR : 0 ≤ simplexNoiseRadius d n T δ := Real.sqrt_nonneg _
  have h := overlap_bias_absorption hT hS hST hR hR2
  convert h using 1
  ring

/-- The actual drift comparison coefficient 80 and the retained bias fit
the original exponential form with universal coefficient 82.
Source: arXiv:2604.01978v1, §7, thm:clustering_random_init.
Here D is the original nonnegative dσA⁴β²n factor. -/
theorem simplex_comparison_envelope (d n : ℕ) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (T : ℝ) (hT : 0 ≤ T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1) (D : ℝ) (hD : 0 ≤ D) :
    ((7 / 8 : ℝ) * simplexNoiseRadius d n T δ + 3 * T / (d : ℝ)) *
        Real.exp (80 * (1 + D) * T) ≤
      Real.exp (82 * (1 + D) * T) * simplexNoiseRadius d n T δ := by
  have hR : 0 ≤ simplexNoiseRadius d n T δ := Real.sqrt_nonneg _
  calc
    _ ≤ (Real.exp (2 * T) * simplexNoiseRadius d n T δ) * Real.exp (80 * (1 + D) * T) :=
      mul_le_mul_of_nonneg_right (simplex_noise_bias_absorption d n hd hn T hT δ hδ) (Real.exp_nonneg _)
    _ = Real.exp ((80 * (1 + D) + 2) * T) * simplexNoiseRadius d n T δ := by
      rw [show (80 * (1 + D) + 2) * T = 2 * T + 80 * (1 + D) * T by ring, Real.exp_add]
      ring
    _ ≤ _ := mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr (by nlinarith)) hR

/-- The numerical tail expression with energy 3T/d and radius 7R/8
still pays for the original union factor 2n²/δ.
Source: arXiv:2604.01978v1, §7, eq:martingale_sup_bound_clean.
This is a scalar estimate; a stochastic maximal inequality remains separate. -/
theorem simplex_noise_exponential_reserve (d n : ℕ) (hd : 2 ≤ d) (hn : 2 ≤ n)
    (T : ℝ) (hT : 0 < T) (δ : ℝ) (hδ : δ ∈ Ioo (0 : ℝ) 1) :
    (2 * (n : ℝ) ^ 2) * Real.exp (-(d : ℝ) *
      ((7 / 8 : ℝ) * simplexNoiseRadius d n T δ) ^ 2 / (6 * T)) ≤ δ := by
  have hdpos : (0 : ℝ) < d := by exact_mod_cast (show 0 < d by omega)
  have hnpos : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hδpos := hδ.1
  have harg : 0 < 2 * (n : ℝ) ^ 2 / δ := by positivity
  have hlog := one_le_simplex_noise_log n hn δ hδ
  have hexponent : -(d : ℝ) * ((7 / 8 : ℝ) * simplexNoiseRadius d n T δ) ^ 2 / (6 * T) =
      -(49 / 48 : ℝ) * Real.log (2 * (n : ℝ) ^ 2 / δ) := by
    rw [mul_pow, simplexNoiseRadius_sq d n hn T hT.le δ hδ]
    field_simp
    ring
  rw [hexponent]
  calc
    _ ≤ (2 * (n : ℝ) ^ 2) * Real.exp (-Real.log (2 * (n : ℝ) ^ 2 / δ)) := by
      gcongr
      linarith
    _ = δ := by rw [Real.exp_neg, Real.exp_log harg]; field_simp

/-- Positive times, a nonzero radius and nonzero bias realize all scalar
premises at the source's smallest dimension and token count.
The squared-radius premise follows from the actual logarithmic factor. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧
    0 ≤ (1 / 2 : ℝ) ∧ 2 * (1 / 2 : ℝ) ≤ 1 ∧
    0 ≤ simplexNoiseRadius 2 2 1 (1 / 2) ∧
    8 * (1 / 2 : ℝ) ≤ simplexNoiseRadius 2 2 1 (1 / 2) ^ 2 := by
  refine ⟨le_rfl, by norm_num, by norm_num, by norm_num, by norm_num, Real.sqrt_nonneg _, ?_⟩
  rw [simplexNoiseRadius_sq 2 2 le_rfl 1 (by norm_num) (1 / 2) (by norm_num)]
  have h := one_le_simplex_noise_log 2 le_rfl (1 / 2) (by norm_num)
  norm_num at h ⊢
  linarith

end Transformer.Homogenized
