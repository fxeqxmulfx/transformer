module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6Fill
public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorDual
public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorFactors

/-!
# PT6: a completion of `⌊a b / 97⌋` with a smaller norm

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Keep the table on
the seen cells and put `L = P Qᴴ` on the others. This completion is `L`, plus the distance `R` from
`L` to the rounded table on the seen cells, plus the rounding `E` of the table there. So its norm
is at most `(‖P‖²_F + ‖Q‖²_F) / 2 + √97 ‖R‖_F + √97 ‖E‖_F`, and the kernel has added up all three
(`TablePart.E6FloorFactors`): the sum is at most 152.8 (`nuclearNorm_floorFill_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- The left factor. -/
noncomputable def pMat : Matrix (Fin 1 → ZMod 97) (Fin 4) ℂ :=
  Matrix.of fun u t => ⟨(pRe (u 0).val t : ℝ) / 2 ^ 10, (pIm (u 0).val t : ℝ) / 2 ^ 10⟩

/-- The right factor. -/
noncomputable def qMat : Matrix (Fin 1 → ZMod 97) (Fin 4) ℂ :=
  Matrix.of fun v t => ⟨(qRe (v 0).val t : ℝ) / 2 ^ 10, (qIm (v 0).val t : ℝ) / 2 ^ 10⟩

/-- The low-rank part of the completion. -/
noncomputable def lowMat : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ := pMat * qMatᴴ

lemma mul_star_pair (a b c d : ℝ) :
    (⟨a / 2 ^ 10, b / 2 ^ 10⟩ : ℂ) * star (⟨c / 2 ^ 10, d / 2 ^ 10⟩ : ℂ) =
      ⟨(a * c + b * d) / 2 ^ 20, (b * c - a * d) / 2 ^ 20⟩ := by
  apply Complex.ext <;> simp only [Complex.mul_re, Complex.mul_im, Complex.star_def,
    Complex.conj_re, Complex.conj_im] <;> ring

/-- The entries of `L`. -/
lemma low_apply (u v : Fin 1 → ZMod 97) : lowMat u v =
    ⟨(lowRe (u 0).val (v 0).val : ℝ) / 2 ^ 20, (lowIm (u 0).val (v 0).val : ℝ) / 2 ^ 20⟩ := by
  rw [lowMat, Matrix.mul_apply]
  simp only [conjTranspose_apply, pMat, qMat, of_apply, mul_star_pair]
  rw [Fin.sum_univ_eq_sum_range (fun t => (⟨((pRe (u 0).val t : ℝ) * qRe (v 0).val t +
    pIm (u 0).val t * qIm (v 0).val t) / 2 ^ 20, ((pIm (u 0).val t : ℝ) * qRe (v 0).val t -
    pRe (u 0).val t * qIm (v 0).val t) / 2 ^ 20⟩ : ℂ)) 4]
  apply Complex.ext
  · rw [Complex.re_sum, lowRe, sumFrom_eq, ← Finset.sum_div]
    simp only [zero_add]
    push_cast
    rfl
  · rw [Complex.im_sum, lowIm, sumFrom_eq, ← Finset.sum_div]
    simp only [zero_add]
    push_cast
    rfl

lemma norm_pair_sq (a b : ℝ) :
    ‖(⟨a / 2 ^ 10, b / 2 ^ 10⟩ : ℂ)‖ ^ 2 = (a * a + b * b) / 2 ^ 20 := by
  rw [Complex.sq_norm, Complex.normSq_apply]
  ring

/-- The size of the two factors is the sum the kernel bounds. -/
lemma frobeniusSq_factors : frobeniusSq pMat + frobeniusSq qMat =
    (∑ i ∈ range 97, ∑ t ∈ range 4, (factorTerm i t : ℝ)) / 2 ^ 20 := by
  unfold frobeniusSq
  simp only [pMat, qMat, of_apply, norm_pair_sq]
  rw [← Finset.sum_add_distrib, sum_line_val fun i =>
    ∑ t : Fin 4, ((pRe i t : ℝ) * pRe i t + pIm i t * pIm i t) / 2 ^ 20 +
      ∑ t : Fin 4, ((qRe i t : ℝ) * qRe i t + qIm i t * qIm i t) / 2 ^ 20, Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Fin.sum_univ_eq_sum_range (fun t => ((pRe i t : ℝ) * pRe i t + pIm i t * pIm i t) / 2 ^ 20) 4,
    Fin.sum_univ_eq_sum_range (fun t => ((qRe i t : ℝ) * qRe i t + qIm i t * qIm i t) / 2 ^ 20) 4,
    ← Finset.sum_add_distrib, Finset.sum_div]
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [factorTerm, sqZ]
  push_cast
  ring

/-- The completion: the table on the seen cells, `L` on the others. -/
noncomputable def floorFill : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  Matrix.of fun u v => if halfMask.testBit (97 * (u 0).val + (v 0).val) then
    tableKernel floorTable u v else lowMat u v

/-- The distance from `L` to the rounded table, on the seen cells. -/
noncomputable def residMat : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  Matrix.of fun u v => if halfMask.testBit (97 * (u 0).val + (v 0).val) then
    phaseHat ((u 0).val * (v 0).val / 97) - lowMat u v else 0

/-- The rounding of the table, on the seen cells. -/
noncomputable def roundMat : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  Matrix.of fun u v => if halfMask.testBit (97 * (u 0).val + (v 0).val) then
    tableKernel floorTable u v - phaseHat ((u 0).val * (v 0).val / 97) else 0

lemma floorFill_eq : floorFill = lowMat + residMat + roundMat := by
  ext u v
  simp only [floorFill, residMat, roundMat, Matrix.add_apply, of_apply]
  by_cases h : halfMask.testBit (97 * (u 0).val + (v 0).val) = true
  · simp only [h, ↓reduceIte]
    ring
  · simp [h]

lemma frobeniusSq_resid : frobeniusSq residMat =
    (∑ i ∈ range 97, ∑ j ∈ range 97, (residTerm i j : ℝ)) / 2 ^ 40 := by
  have hcell (u v : Fin 1 → ZMod 97) :
      ‖residMat u v‖ ^ 2 = (residTerm (u 0).val (v 0).val : ℝ) / 2 ^ 40 := by
    simp only [residMat, of_apply, residTerm]
    by_cases h : halfMask.testBit (97 * (u 0).val + (v 0).val) = true
    · simp only [h, ↓reduceIte]
      rw [low_apply, Complex.sq_norm, Complex.normSq_apply]
      simp only [Complex.sub_re, Complex.sub_im, phaseHat, sqZ]
      push_cast
      ring
    · simp [h]
  unfold frobeniusSq
  simp only [hcell, ← Finset.sum_div]
  rw [sum_lines_val fun i j => (residTerm i j : ℝ)]

lemma frobeniusSq_round_le : frobeniusSq roundMat ≤
    (∑ i ∈ range 97, ∑ j ∈ range 97, (roundTerm i j : ℝ)) / 2 ^ 40 := by
  have hcell (u v : Fin 1 → ZMod 97) :
      ‖roundMat u v‖ ^ 2 ≤ (roundTerm (u 0).val (v 0).val : ℝ) / 2 ^ 40 := by
    simp only [roundMat, of_apply, roundTerm]
    by_cases h : halfMask.testBit (97 * (u 0).val + (v 0).val) = true
    · simp only [h, ↓reduceIte]
      rw [kernel_floor]
      have h2 := pow_le_pow_left₀ (norm_nonneg _)
        (norm_pow_sub_phaseHat_le (floor_exponent_lt u v)) 2
      refine h2.trans (le_of_eq ?_)
      generalize (u 0).val * (v 0).val / 97 = r
      simp only [sqZ]
      push_cast
      ring
    · simp [h]
  unfold frobeniusSq
  refine (Finset.sum_le_sum fun u _ => Finset.sum_le_sum fun v _ => hcell u v).trans (le_of_eq ?_)
  simp only [← Finset.sum_div]
  rw [sum_lines_val fun i j => (roundTerm i j : ℝ)]

/-- A matrix on the lines with a small Frobenius norm has a small nuclear norm. -/
lemma nuclearNorm_le_of_frobeniusSq_le {R : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ}
    {A B : ℝ} (hB : 0 ≤ B) (h : frobeniusSq R ≤ A) (hAB : 97 * A ≤ B ^ 2) :
    nuclearNorm R ≤ B := by
  have hrank : (R.rank : ℝ) ≤ 97 := by
    have h1 := Matrix.rank_le_card_width R
    simp only [Fintype.card_fun, Fintype.card_fin, ZMod.card, pow_one] at h1
    exact_mod_cast h1
  have hFnn : 0 ≤ frobeniusSq R := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => by
    positivity
  have hsq : nuclearNorm R ^ 2 ≤ B ^ 2 := by
    refine (nuclearNorm_sq_le_rank_mul_frobeniusSq R).trans ?_
    calc (R.rank : ℝ) * frobeniusSq R ≤ 97 * A := mul_le_mul hrank h hFnn (by norm_num)
      _ ≤ B ^ 2 := hAB
  exact (abs_le_of_sq_le_sq' hsq hB).2

example : nuclearNorm (0 : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ) ≤ 0 :=
  nuclearNorm_le_of_frobeniusSq_le (A := 0) le_rfl (by simp [frobeniusSq]) (by norm_num)

/-- **The completion has nuclear norm at most 152.8.** -/
theorem nuclearNorm_floorFill_le : nuclearNorm floorFill ≤ 152.8 := by
  have hf : (∑ i ∈ range 97, ∑ t ∈ range 4, (factorTerm i t : ℝ)) ≤ 294649856 := by
    have h := factorSum_le
    simp only [sumFrom_eq, zero_add] at h
    exact_mod_cast h
  have hr : (∑ i ∈ range 97, ∑ j ∈ range 97, (residTerm i j : ℝ)) ≤ 1687000000000 := by
    have h := residSum_le
    simp only [sumFrom_eq, zero_add] at h
    exact_mod_cast h
  have he : (∑ i ∈ range 97, ∑ j ∈ range 97, (roundTerm i j : ℝ)) ≤ 90000000 := by
    have h := roundSum_le
    simp only [sumFrom_eq, zero_add] at h
    exact_mod_cast h
  have hL : nuclearNorm lowMat ≤ 140.5 := by
    refine (nuclearNorm_le_of_eq_mul pMat qMat rfl).trans ?_
    rw [frobeniusSq_factors]
    have h2 : (∑ i ∈ range 97, ∑ t ∈ range 4, (factorTerm i t : ℝ)) / 2 ^ 20 ≤ 281 := by
      rw [div_le_iff₀ (by positivity)]
      refine hf.trans ?_
      norm_num
    linarith
  have hR : nuclearNorm residMat ≤ 12.2 := nuclearNorm_le_of_frobeniusSq_le (by norm_num)
    (frobeniusSq_resid.le.trans (div_le_div_of_nonneg_right hr (by positivity))) (by norm_num)
  have hE : nuclearNorm roundMat ≤ 0.09 := nuclearNorm_le_of_frobeniusSq_le (by norm_num)
    (frobeniusSq_round_le.trans (div_le_div_of_nonneg_right he (by positivity))) (by norm_num)
  rw [floorFill_eq]
  have h1 := nuclearNorm_add_le (lowMat + residMat) roundMat
  have h2 := nuclearNorm_add_le lowMat residMat
  linarith

end Transformer.Grokking.FlatSpectrum.TablePart.E6
