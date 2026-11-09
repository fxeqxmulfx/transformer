module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.TrainingTransport

/-!
# Geometry visible to the class readout

Source: `review/flat_spectrum.md`, *Observable spectral geometry*, OG1–OG3.
Center readout rows across classes. Their Gram form measures only feature
directions visible in score differences, rather than every hidden direction.
The finite contrast bound retains the complete observable error energy.
No assertion about optimizer convergence or a flat learned spectrum is made.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset

variable {C K : Type*} [Fintype C] [Fintype K]

/-- Mean readout row across all classes, OG1. -/
def readoutMean (E : C → K → ℝ) (k : K) : ℝ :=
  (∑ c, E c k) / Fintype.card C

/-- Class-centered readout: a common score offset is removed, OG1. -/
def centeredReadout (E : C → K → ℝ) (c : C) (k : K) : ℝ :=
  E c k - readoutMean E k

/-- Gram form of the class-centered readout, OG2. Its kernel can contain
nonzero hidden-state errors without changing any class comparison. -/
def observableGram (E : C → K → ℝ) (k l : K) : ℝ :=
  ∑ c, centeredReadout E c k * centeredReadout E c l

/-- Sum of squared centered score errors, OG2. -/
def observableEnergy (E : C → K → ℝ) (e : K → ℝ) : ℝ :=
  ∑ c, (linearScore (centeredReadout E c) e) ^ 2

/-- Centering leaves each score difference unchanged, OG1. -/
theorem centeredReadout_contrast (E : C → K → ℝ) (e : K → ℝ) (c t : C) :
    linearScore (centeredReadout E c) e - linearScore (centeredReadout E t) e =
      linearScore (E c) e - linearScore (E t) e := by
  simp only [linearScore, centeredReadout, sub_mul, sum_sub_distrib]
  ring

/-- The observable energy is exactly the centered Gram quadratic form, OG2.
This is an anisotropic seminorm; no lower eigenvalue is assumed. -/
theorem observableEnergy_eq_gram (E : C → K → ℝ) (e : K → ℝ) :
    observableEnergy E e = ∑ k, ∑ l, e k * observableGram E k l * e l := by
  calc
    _ = ∑ c, ∑ k, ∑ l,
        e k * (centeredReadout E c k * centeredReadout E c l) * e l := by
      unfold observableEnergy linearScore
      apply sum_congr rfl
      intro c hc
      rw [pow_two, sum_mul]
      apply sum_congr rfl
      intro k hk
      rw [mul_sum]
      apply sum_congr rfl
      intro l hl
      ring
    _ = ∑ k, ∑ l, ∑ c,
        e k * (centeredReadout E c k * centeredReadout E c l) * e l := by
      rw [sum_comm]
      apply sum_congr rfl
      intro k hk
      rw [sum_comm]
    _ = _ := by simp only [observableGram, mul_sum, sum_mul]

/-- Positivity does not require a nondegenerate readout, OG2. -/
theorem observableEnergy_nonneg (E : C → K → ℝ) (e : K → ℝ) :
    0 ≤ observableEnergy E e := by
  exact sum_nonneg fun c hc => sq_nonneg _

/-- Every pairwise error is bounded by twice the total centered score energy,
OG3. The factor two comes from two distinct class coordinates. -/
theorem observable_contrast_sq_le (E : C → K → ℝ) (e : K → ℝ) (c t : C) :
    (linearScore (E c) e - linearScore (E t) e) ^ 2 ≤ 2 * observableEnergy E e := by
  classical
  by_cases h : c = t
  · subst t
    simpa only [sub_self, zero_pow (by decide : 2 ≠ 0)] using
      mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) (observableEnergy_nonneg E e)
  · have hsum := sum_le_sum_of_subset_of_nonneg (subset_univ {c, t})
      (fun i hi hnot => sq_nonneg (linearScore (centeredReadout E i) e))
    rw [sum_pair h] at hsum
    change _ ≤ observableEnergy E e at hsum
    rw [← centeredReadout_contrast E e c t]
    nlinarith [sq_nonneg
      (linearScore (centeredReadout E c) e + linearScore (centeredReadout E t) e)]

/-- A finite contrast bound in the observable seminorm, OG3. -/
theorem observable_contrast_abs_le (E : C → K → ℝ) (e : K → ℝ) (c t : C) :
    |linearScore (E c) e - linearScore (E t) e| ≤
      Real.sqrt (2 * observableEnergy E e) := by
  exact Real.abs_le_sqrt (observable_contrast_sq_le E e c t)

/-- Zero observable energy means that every centered score error vanishes,
OG2. The hidden error itself need not vanish. -/
theorem observableEnergy_eq_zero_iff_centered (E : C → K → ℝ) (e : K → ℝ) :
    observableEnergy E e = 0 ↔ ∀ c, linearScore (centeredReadout E c) e = 0 := by
  unfold observableEnergy
  rw [sum_sq_eq_zero_iff]
  simp only [mem_univ, forall_true_left]

example := centeredReadout_contrast (C := Bool) (K := Bool)
  (fun c k => if k then 3 else if c then 1 else -1) (fun k => if k then 7 else 0)
  true false

example := observableEnergy_eq_gram (C := Bool) (K := Bool)
  (fun c k => if k then 3 else if c then 1 else -1) (fun k => if k then 7 else 0)

example := observable_contrast_abs_le (C := Bool) (K := Bool)
  (fun c k => if k then 3 else if c then 1 else -1) (fun k => if k then 7 else 0)
  true false

end Transformer.Grokking.FlatSpectrum.FeatureCodes
