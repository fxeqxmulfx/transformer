module

public import Transformer.SequenceModel.Attention.Backend.Activation

/-!
# Exponential and reduction certificates for attention averages

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K4.
The reference masses are the exact masked exponentials; approximate masses
can be rational polynomial witnesses. Their normalized rational average is
checked against the captured SDPA output. This permits fused kernels without
asserting that their hidden reduction order matches a sequential dot product.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend

open scoped BigOperators

/-- K4: normalizing approximate masses costs at most twice their total
absolute error, divided by the checked approximate partition function. -/
theorem normalized_weights_error {T : ℕ} (a b : Fin T → ℝ)
    (hA : 0 < ∑ i, a i) (hB : 0 < ∑ i, b i) (hb : ∀ i, 0 ≤ b i)
    (E : ℝ) (he : ∑ i, |a i - b i| ≤ E) :
    ∑ i, |a i / (∑ j, a j) - b i / (∑ j, b j)| ≤ 2 * E / (∑ j, a j) := by
  let A := ∑ i, a i
  let B := ∑ i, b i
  have hA' : 0 < A := hA
  have hB' : 0 < B := hB
  have hAB : |B - A| ≤ E := by
    rw [show B - A = ∑ i, (b i - a i) by rw [Finset.sum_sub_distrib]]
    exact (Finset.abs_sum_le_sum_abs _ _).trans (by simpa only [abs_sub_comm] using he)
  have hp (i) : |a i / A - b i / B| ≤
      (|a i - b i| + (b i / B) * E) / A := by
    have hi : a i / A - b i / B = (a i - b i) / A + (b i / B) * ((B - A) / A) := by
      field_simp [hA'.ne', hB'.ne']
      ring
    rw [hi]
    calc
      _ ≤ |(a i - b i) / A| + |(b i / B) * ((B - A) / A)| := abs_add_le _ _
      _ = |a i - b i| / A + (b i / B) * (|B - A| / A) := by
        simp only [abs_mul, abs_div, abs_of_pos hA', abs_of_pos hB', abs_of_nonneg (hb i)]
      _ ≤ |a i - b i| / A + (b i / B) * (E / A) := by
        gcongr
        exact div_nonneg (hb i) hB'.le
      _ = _ := by ring
  have hs : ∑ i, b i / B = 1 := by rw [← Finset.sum_div, div_self hB.ne']
  calc
    _ ≤ ∑ i, (|a i - b i| + (b i / B) * E) / A := Finset.sum_le_sum (fun i _ => hp i)
    _ = ((∑ i, |a i - b i|) + E) / A := by
      rw [← Finset.sum_div, Finset.sum_add_distrib, ← Finset.sum_mul, hs, one_mul]
    _ ≤ (E + E) / A := div_le_div_of_nonneg_right (add_le_add he le_rfl) hA.le
    _ = _ := by ring

/-- K4: a rational normalized-average residual and certified mass errors
give the actual vector SDPA contract. Width enters the residual certificate;
the count of visible tokens enters the sum of exponential errors. -/
theorem average_residual_error {T d : ℕ} (a b : Fin T → ℝ) (v : Fin T → EucSpace d)
    (z : EucSpace d) (hA : 0 < ∑ i, a i) (hB : 0 < ∑ i, b i) (hb : ∀ i, 0 ≤ b i)
    (E M β : ℝ) (hM : 0 ≤ M) (he : ∑ i, |a i - b i| ≤ E) (hv : ∀ i, ‖v i‖ ≤ M)
    (hz : ‖z - ∑ i, (a i / (∑ j, a j)) • v i‖ ≤ β) :
    ‖z - ∑ i, (b i / (∑ j, b j)) • v i‖ ≤ β + (2 * E / (∑ j, a j)) * M := by
  have hweights := normalized_weights_error a b hA hB hb E he
  have hvec : ‖(∑ i, (a i / (∑ j, a j)) • v i) - ∑ i, (b i / (∑ j, b j)) • v i‖ ≤
      (2 * E / (∑ j, a j)) * M := by
    rw [← Finset.sum_sub_distrib]
    calc
      _ ≤ ∑ i, ‖(a i / (∑ j, a j)) • v i - (b i / (∑ j, b j)) • v i‖ := norm_sum_le _ _
      _ ≤ ∑ i, |a i / (∑ j, a j) - b i / (∑ j, b j)| * M := by
        apply Finset.sum_le_sum
        intro i hi
        rw [← sub_smul, norm_smul, Real.norm_eq_abs]
        exact mul_le_mul_of_nonneg_left (hv i) (abs_nonneg _)
      _ = (∑ i, |a i / (∑ j, a j) - b i / (∑ j, b j)|) * M := (Finset.sum_mul _ _ _).symm
      _ ≤ _ := mul_le_mul_of_nonneg_right hweights hM
  exact (norm_sub_le_norm_sub_add_norm_sub z _ _).trans (add_le_add hz hvec)

example := normalized_weights_error ![(1 : ℝ), 2] ![1, 1]
  (by norm_num [Fin.sum_univ_succ]) (by norm_num [Fin.sum_univ_succ])
  (by intro i; fin_cases i <;> norm_num) 1 (by norm_num [Fin.sum_univ_succ])
example := average_residual_error ![(1 : ℝ), 2] ![1, 1]
  (fun i : Fin 2 => EuclideanSpace.single i 1)
  ((1 / 3 : ℝ) • EuclideanSpace.single 0 1 + (2 / 3 : ℝ) • EuclideanSpace.single 1 1 : EucSpace 2)
  (by norm_num [Fin.sum_univ_succ]) (by norm_num [Fin.sum_univ_succ])
  (by intro i; fin_cases i <;> norm_num) 1 1 0 (by norm_num)
  (by norm_num [Fin.sum_univ_succ]) (by intro i; simp)
  (by norm_num [Fin.sum_univ_succ])

end Transformer.SequenceModel.Attention.Backend
