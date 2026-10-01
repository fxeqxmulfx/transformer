import Transformer.DoubleDescent.Section5_Curves
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Data.Fintype.Fin
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Uniformly incorrect label noise

arXiv:1912.02292v1, Section 4, paragraph "Label Noise". With probability
`p` the label is replaced uniformly among the *incorrect* labels. This
differs from replacement uniformly among all labels. We derive the affine
rescaling from the finite label distribution, rather than assuming it.
-/

namespace Transformer.DoubleDescent

open Finset

/-- Section 4: zero-one classification loss for a finite label alphabet. -/
def classificationError {C : ℕ} (prediction label : Fin C) : ℝ :=
  if prediction = label then 0 else 1

/-- Section 4: expected error conditional on the clean label and prediction,
with noise drawn uniformly from the other `C-1` labels. -/
noncomputable def conditionalNoisyRisk {C : ℕ} (p : ℝ) (prediction truth : Fin C) : ℝ :=
  (1 - p) * classificationError prediction truth +
    p / (C - 1 : ℝ) * ∑ label ∈ univ.erase truth, classificationError prediction label

/-- Section 4: the affine noise map, valid for clean classification error. -/
noncomputable def affineNoise (C : ℕ) (p error : ℝ) : ℝ :=
  p + (1 - p - p / (C - 1 : ℝ)) * error

/-- Section 4: a fixed prediction is incorrect for exactly `C-1` labels. -/
theorem sum_classificationError {C : ℕ} (prediction : Fin C) :
    ∑ label, classificationError prediction label = (C : ℝ) - 1 := by
  have hi (label : Fin C) : classificationError prediction label =
      1 - (if prediction = label then (1 : ℝ) else 0) := by
    by_cases h : prediction = label <;> simp [classificationError, h]
  simp_rw [hi]
  rw [sum_sub_distrib]
  simp

/-- Section 4: summing over the incorrect-label distribution gives the
precise affine formula `p + (1 - p*C/(C-1))*error`. -/
theorem conditionalNoisyRisk_eq_affine {C : ℕ} (hC : 2 ≤ C)
    (p : ℝ) (prediction truth : Fin C) :
    conditionalNoisyRisk p prediction truth =
      affineNoise C p (classificationError prediction truth) := by
  have hc : (2 : ℝ) ≤ C := by exact_mod_cast hC
  have hd : (C : ℝ) - 1 ≠ 0 := by linarith
  have hs := sum_erase_add univ (classificationError prediction) (mem_univ truth)
  rw [sum_classificationError] at hs
  have he : ∑ label ∈ univ.erase truth, classificationError prediction label =
      (C : ℝ) - 1 - classificationError prediction truth := by linarith
  unfold conditionalNoisyRisk affineNoise
  rw [he]
  field_simp [hd]
  ring

/-- Section 4: a nontrivial incorrect-label distribution exists with two labels. -/
example : 2 ≤ (2 : ℕ) := le_rfl

/-- Section 4: population clean error for an explicitly weighted finite dataset. -/
noncomputable def weightedCleanRisk {N C : ℕ} (weight : Fin N → ℝ)
    (prediction truth : Fin N → Fin C) : ℝ :=
  ∑ i, weight i * classificationError (prediction i) (truth i)

/-- Section 4: population noisy error for the same weighted examples. -/
noncomputable def weightedNoisyRisk {N C : ℕ} (weight : Fin N → ℝ)
    (prediction truth : Fin N → Fin C) (p : ℝ) : ℝ :=
  ∑ i, weight i * conditionalNoisyRisk p (prediction i) (truth i)

/-- Section 4: the conditional formula averages to the same affine map.
Nonnegative weights summing to one and `0 ≤ p ≤ 1` give the probabilistic
interpretation; the algebraic identity itself only needs normalization. -/
theorem weightedNoisyRisk_eq_affine {N C : ℕ} (hC : 2 ≤ C)
    (weight : Fin N → ℝ) (hw : ∑ i, weight i = 1)
    (prediction truth : Fin N → Fin C) (p : ℝ) :
    weightedNoisyRisk weight prediction truth p =
      affineNoise C p (weightedCleanRisk weight prediction truth) := by
  unfold weightedNoisyRisk weightedCleanRisk affineNoise
  simp_rw [conditionalNoisyRisk_eq_affine hC, affineNoise]
  calc
    _ = ∑ i, (weight i * p +
        (1 - p - p / (C - 1 : ℝ)) *
          (weight i * classificationError (prediction i) (truth i))) := by
      apply sum_congr rfl
      intro i hi
      ring
    _ = _ := by
      rw [sum_add_distrib, ← sum_mul, ← mul_sum, hw, one_mul]

/-- Section 4: normalized nonnegative weights and two labels are satisfiable. -/
example : 2 ≤ (2 : ℕ) ∧ ∑ i : Fin 1, (fun _ => (1 : ℝ)) i = 1 := by
  simp

/-- Section 4 and Figure 1: for CIFAR-10 with 15 percent noise, the slope
is `5/6`, so clean and noisy error have exactly the same ordering. -/
theorem cifar10_fifteen_percent_rescaling (error : ℝ) :
    affineNoise 10 (3 / 20) error = 3 / 20 + (5 / 6) * error := by
  norm_num [affineNoise]

/-- Section 4 and Figure 1: the noise transformation used in the introduction
preserves double descent. -/
theorem cifar10_noise_preserves_doubleDescent {I : Type*} [Preorder I]
    (error : I → ℝ) :
    HasDoubleDescent (fun i => affineNoise 10 (3 / 20) (error i)) ↔
      HasDoubleDescent error := by
  simp_rw [cifar10_fifteen_percent_rescaling]
  exact doubleDescent_affine (by norm_num : (0 : ℝ) < 5 / 6)

/-- Section 4: the linear-rescaling claim alone does not guarantee ordering
preservation at arbitrary noise rates. With two labels and certain flipping,
a correct prediction has noisy error one and an incorrect prediction has zero. -/
theorem full_binary_noise_reverses_error :
    affineNoise 2 1 0 = 1 ∧ affineNoise 2 1 1 = 0 := by
  norm_num [affineNoise]

/-- Section 4, "Label Noise", and the authors' `intro_ocean_dynamics.ipynb`,
function `plot_dynamics`: its rescaling uses `+ error*p/9` where the noise
model requires a minus sign. At clean error one and 15 percent noise it
even exceeds one; the correct noisy classification error is `59/60`.
Both slopes are positive at this noise rate, so the sign error does not
change the ordering or presence of double descent in those plots. -/
theorem plotting_noise_rescaling_counterexample :
    1 < (1 - (1 - 3 / 20) * (1 - 1) + 1 * (3 / 20) / 9 : ℝ) ∧
      affineNoise 10 (3 / 20) 1 = 59 / 60 := by
  norm_num [affineNoise]

end Transformer.DoubleDescent
