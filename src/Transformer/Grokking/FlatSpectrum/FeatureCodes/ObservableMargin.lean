module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ObservableGeometry

/-!
# Invisible feature errors and observable margin budgets

Source: `review/flat_spectrum.md`, *Observable spectral geometry*, OG1–OG4.
Zero centered energy is equivalent to a common score shift. Positive margin
budgets therefore need to control the observable error, rather than recover
every feature coordinate. The hypotheses are finite and remain explicit.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Finset
variable {C K : Type*} [Fintype C] [Fintype K]

/-- Averaging readout rows commutes with linear scoring, OG1. -/
theorem linearScore_readoutMean (E : C → K → ℝ) (e : K → ℝ) :
    linearScore (readoutMean E) e = (∑ c, linearScore (E c) e) / Fintype.card C := by
  simp only [linearScore, readoutMean, div_mul_eq_mul_div, sum_mul, sum_div]
  rw [sum_comm]

/-- The kernel of the observable Gram form consists exactly of feature
errors that add a common score shift, OG2. Empty class sets are allowed. -/
theorem observableEnergy_eq_zero_iff_contrasts (E : C → K → ℝ) (e : K → ℝ) :
    observableEnergy E e = 0 ↔ ∀ c t, linearScore (E c) e = linearScore (E t) e := by
  rw [observableEnergy_eq_zero_iff_centered]
  constructor
  · intro h c t
    have hc := centeredReadout_contrast E e c t
    rw [h c, h t] at hc
    linarith
  · intro h c
    let : Nonempty C := ⟨c⟩
    have hm : linearScore (readoutMean E) e = linearScore (E c) e := by
      rw [linearScore_readoutMean]
      have hs : (∑ i, linearScore (E i) e) =
          (Fintype.card C : ℝ) * linearScore (E c) e := by
        simp_rw [h _ c]
        simp
      rw [hs]
      exact mul_div_cancel_left₀ _ (by exact_mod_cast Fintype.card_ne_zero)
    simp only [linearScore, centeredReadout, sub_mul, sum_sub_distrib]
    change linearScore (E c) e - linearScore (readoutMean E) e = 0
    rw [hm, sub_self]

omit [Fintype C] in
/-- A common added readout row cannot change a score contrast, OG1. -/
theorem common_readout_shift_preserves_contrast
    (E : C → K → ℝ) (b x : K → ℝ) (c t : C) :
    linearScore (fun k => E c k + b k) x - linearScore (fun k => E t k + b k) x =
      linearScore (E c) x - linearScore (E t) x := by
  simp only [linearScore, add_mul, sum_add_distrib]
  ring

/-- A nonzero feature error can preserve every class margin exactly, OG2. -/
theorem invisible_error_preserves_contrast (E : C → K → ℝ) (x e : K → ℝ)
    (h : observableEnergy E e = 0) (c t : C) :
    linearScore (E c) (fun k => x k + e k) -
        linearScore (E t) (fun k => x k + e k) =
      linearScore (E c) x - linearScore (E t) x := by
  have he := (observableEnergy_eq_zero_iff_contrasts E e).mp h c t
  simp only [linearScore, mul_add, sum_add_distrib] at *
  linarith

/-- The finite margin loss is at most the observable error budget, OG3. -/
theorem observable_margin_lower (E : C → K → ℝ) (x e : K → ℝ) (t c : C) :
    linearScore (E t) x - linearScore (E c) x - Real.sqrt (2 * observableEnergy E e) ≤
      linearScore (E t) (fun k => x k + e k) -
        linearScore (E c) (fun k => x k + e k) := by
  have he := (abs_le.mp (observable_contrast_abs_le E e c t)).2
  simp only [linearScore, mul_add, sum_add_distrib] at *
  linarith

/-- A strict winner survives when each old margin exceeds the observable
error budget, OG3. This is conditional, not a training-convergence claim. -/
theorem strict_winner_of_observable_error (E : C → K → ℝ) (x e : K → ℝ) (t : C)
    (h : ∀ c, c ≠ t → Real.sqrt (2 * observableEnergy E e) <
      linearScore (E t) x - linearScore (E c) x) :
    ∀ c, c ≠ t → linearScore (E c) (fun k => x k + e k) <
      linearScore (E t) (fun k => x k + e k) := by
  intro c hc
  have hb := observable_margin_lower E x e t c
  have hm := h c hc
  linarith

/-- OG4: a hidden error of size seven shifts both scores by 21, yet its
observable energy vanishes. Neither hidden recovery nor score recovery holds. -/
theorem nonzero_invisible_error_witness :
    let E : Bool → Bool → ℝ := fun c k => if k then 3 else if c then 1 else -1
    let e : Bool → ℝ := fun k => if k then 7 else 0
    e true ≠ 0 ∧ observableEnergy E e = 0 ∧ ∀ c, linearScore (E c) e = 21 := by
  norm_num [observableEnergy, centeredReadout, readoutMean, linearScore]

/-- OG3: the factor two in the pairwise bound is attained. -/
theorem observable_contrast_sharp_witness :
    let E : Bool → Fin 1 → ℝ := fun c _ => if c then 1 else -1
    let e : Fin 1 → ℝ := fun _ => 1
    0 < observableEnergy E e ∧
      (linearScore (E true) e - linearScore (E false) e) ^ 2 =
        2 * observableEnergy E e := by
  norm_num [observableEnergy, centeredReadout, readoutMean, linearScore]

example := invisible_error_preserves_contrast (C := Bool) (K := Bool)
  (fun c k => if k then 3 else if c then 1 else -1)
  (fun k => if k then 0 else 1) (fun k => if k then 7 else 0)
  (by norm_num [observableEnergy, centeredReadout, readoutMean, linearScore]) true false

example :
    let E : Bool → Bool → ℝ := fun c k => if k then 3 else if c then 1 else -1
    let x : Bool → ℝ := fun k => if k then 0 else 1
    let e : Bool → ℝ := fun k => if k then 7 else 0
    ∀ c, c ≠ true → linearScore (E c) (fun k => x k + e k) <
      linearScore (E true) (fun k => x k + e k) := by
  apply strict_winner_of_observable_error
  intro c hc
  cases c <;> norm_num [observableEnergy, centeredReadout, readoutMean, linearScore] at *

end Transformer.Grokking.FlatSpectrum.FeatureCodes
