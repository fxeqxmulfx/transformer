module

public import Transformer.Grokking.FlatSpectrum.ThreeCells

/-!
# Three cells of a `2 × 2` table, by rank

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 3. `ThreeCells`: from the cells
other than `(0, 1)`, M recovers `K` exactly when `k₀₁ = conj(k₁₀) k₀₀ k₁₁ / max(|k₁₀|², |k₀₀ k₁₁|)`,
read as `0` when `k₁₀ = k₀₀ k₁₁ = 0`.

* Rank at most one, `k₀₀ k₁₁ = k₀₁ k₁₀`: exactly when `|k₀₁| ≤ |k₁₀|`
  (`forall_isMinOn_threeCells_iff_of_det_eq_zero`); for `|k₀₁| > |k₁₀|` M's fill is
  `|k₁₀| k₀₁ / |k₀₁|` (`threeCellFill_mul_of_lt`).
* Rank two: exactly when `k₀₁ = conj(k₁₀) det K / |det K|`
  (`forall_isMinOn_threeCells_iff_of_det_ne_zero`).
  If it holds, `g = det K / |det K|` is a subgradient as in `sq_norm_add_le_of_subgradient`, so
  `k₀₁` and M's fill are both least points with a quadratic margin, hence equal. Conversely, at
  M's fill `det K = c (1 − |b|² / max(|b|², |c|))`, which is nonzero only when the max is `|c|`.
* Entries of modulus one: M's fill is `k₀₀ k₁₁ / k₁₀`, the rank-one completion
  (`threeCellFill_of_norm_eq_one`), so M recovers `K` exactly when `det K = 0`
  (`forall_isMinOn_threeCells_iff_det_eq_zero`).

So no criterion in `Ω`, the support and the moduli decides recovery at rank two: `!![2, 1; 1, 1]`
is recovered and `!![2, 1; 1, −1]`, with the same moduli, is not.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

/-- At rank one M's fill is `a |b|² / max(|b|², |a| |b|)`. -/
lemma threeCellFill_mul (a b : ℂ) :
    threeCellFill b (a * b) = a * ((‖b‖ ^ 2 / max (‖b‖ ^ 2) (‖a‖ * ‖b‖) : ℝ) : ℂ) := by
  rw [threeCellFill, norm_mul, mul_comm a b, ← mul_assoc, Complex.conj_mul']
  push_cast
  ring

/-- At rank one M's fill is right exactly when `|a| ≤ |b|`. -/
lemma eq_threeCellFill_mul_iff (a b : ℂ) : a = threeCellFill b (a * b) ↔ ‖a‖ ≤ ‖b‖ := by
  rcases eq_or_ne b 0 with rfl | hb
  · simp [threeCellFill]
  have hb0 : 0 < ‖b‖ := norm_pos_iff.mpr hb
  rw [threeCellFill_mul]
  rcases le_or_gt ‖a‖ ‖b‖ with h | h
  · have hm : max (‖b‖ ^ 2) (‖a‖ * ‖b‖) = ‖b‖ ^ 2 := max_eq_left (by nlinarith)
    rw [hm, div_self (pow_pos hb0 2).ne', Complex.ofReal_one, mul_one]
    exact iff_of_true rfl h
  · have hm : max (‖b‖ ^ 2) (‖a‖ * ‖b‖) = ‖a‖ * ‖b‖ := max_eq_right (by nlinarith)
    have ha0 : 0 < ‖a‖ := hb0.trans h
    refine iff_of_false (fun e => ?_) (not_le.mpr h)
    have hn := congrArg norm e
    rw [hm, norm_mul, Complex.norm_of_nonneg (by positivity), sq,
      mul_div_mul_right _ _ hb0.ne', ha0.ne'.isUnit.mul_div_cancel] at hn
    linarith

/-- At rank one with `|a| > |b|`, M's fill is `|b| a / |a|`: `a` turned down to modulus `|b|`. -/
lemma threeCellFill_mul_of_lt {a b : ℂ} (h : ‖b‖ < ‖a‖) :
    threeCellFill b (a * b) = ‖b‖ * a / ‖a‖ := by
  rcases eq_or_ne b 0 with rfl | hb
  · simp [threeCellFill]
  have hb0 : 0 < ‖b‖ := norm_pos_iff.mpr hb
  have hm : max (‖b‖ ^ 2) (‖a‖ * ‖b‖) = ‖a‖ * ‖b‖ := max_eq_right (by nlinarith)
  rw [threeCellFill_mul, hm, sq, mul_div_mul_right _ _ hb0.ne']
  push_cast
  ring

example := threeCellFill_mul_of_lt (a := 2) (b := 1) (by norm_num)

/-- At rank two M's fill reads off the determinant: for `c − a b ≠ 0`, `a` is M's fill exactly
when `a = conj(b) (c − a b) / |c − a b|`. -/
lemma eq_threeCellFill_iff_of_sub_ne_zero {a b c : ℂ} (hD : c - a * b ≠ 0) :
    a = threeCellFill b c ↔ a = starRingEnd ℂ b * (c - a * b) / ‖c - a * b‖ := by
  have hD0 : 0 < ‖c - a * b‖ := norm_pos_iff.mpr hD
  constructor
  · intro ha
    have hl := sub_threeCellFill_mul b c
    rw [← ha] at hl
    obtain ⟨μ, hμ⟩ : ∃ μ : ℝ, max (‖b‖ ^ 2) ‖c‖ = μ := ⟨_, rfl⟩
    have hbμ : ‖b‖ ^ 2 ≤ μ := hμ ▸ le_max_left _ _
    have hcμ : ‖c‖ ≤ μ := hμ ▸ le_max_right _ _
    have hch : μ = ‖b‖ ^ 2 ∨ μ = ‖c‖ := hμ ▸ max_choice _ _
    rw [hμ] at hl
    rw [threeCellFill, hμ] at ha
    have hl0 : 1 - ‖b‖ ^ 2 / μ ≠ 0 := fun h0 => hD (by rw [hl, h0, Complex.ofReal_zero, mul_zero])
    have hlpos : 0 < 1 - ‖b‖ ^ 2 / μ := lt_of_le_of_ne
      (sub_nonneg.mpr (div_le_one_of_le₀ hbμ ((norm_nonneg c).trans hcμ))) (Ne.symm hl0)
    rw [hl, norm_mul, Complex.norm_of_nonneg hlpos.le]
    rcases eq_or_ne b 0 with rfl | hb
    · simp [ha]
    have hμc : μ = ‖c‖ := hch.resolve_left fun h => hl0 (by
      rw [h, div_self (pow_ne_zero 2 (norm_ne_zero_iff.mpr hb)), sub_self])
    subst hμc
    rw [ha, Complex.ofReal_mul, ← mul_assoc,
      mul_div_mul_right _ _ (Complex.ofReal_ne_zero.mpr hl0)]
  · intro ha
    have hg : ‖(c - a * b) / (‖c - a * b‖ : ℂ)‖ ≤ 1 := le_of_eq (by
      rw [norm_div, Complex.norm_real, norm_norm, div_self hD0.ne'])
    have hgz : (starRingEnd ℂ ((c - a * b) / (‖c - a * b‖ : ℂ)) * (c - a * b)).re =
        ‖c - a * b‖ := by
      rw [map_div₀, Complex.conj_ofReal, div_mul_eq_mul_div, Complex.conj_mul', sq,
        mul_div_assoc, div_self (Complex.ofReal_ne_zero.mpr hD0.ne'), mul_one, Complex.ofReal_re]
    have h1 := sq_norm_add_le_of_subgradient hg hgz (ha.trans (by ring)) (threeCellFill b c)
    have h2 := sq_norm_threeCellFill_add_le b c a
    rw [norm_sub_rev (threeCellFill b c) a] at h1
    have hsq : ‖a - threeCellFill b c‖ ^ 2 = 0 := le_antisymm (by linarith) (sq_nonneg _)
    exact sub_eq_zero.mp (norm_eq_zero.mp ((pow_eq_zero_iff two_ne_zero).mp hsq))

example := eq_threeCellFill_iff_of_sub_ne_zero (a := 1) (b := 1) (c := 2) (by norm_num)

/-- With `|b| = |c| = 1` M's fill is `c / b`, the rank-one completion. -/
lemma threeCellFill_of_norm_eq_one {b c : ℂ} (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) :
    threeCellFill b c = c / b := by
  have hb0 : b ≠ 0 := norm_ne_zero_iff.mp (by rw [hb]; exact one_ne_zero)
  rw [threeCellFill, hb, hc, one_pow, max_self, Complex.ofReal_one, div_one, eq_div_iff hb0,
    mul_right_comm, Complex.conj_mul', hb]
  norm_num

example := threeCellFill_of_norm_eq_one (b := 1) (c := 1) (by simp) (by simp)

/-- With `|b| = |c| = 1` M's fill is right exactly when `c = a b`. -/
lemma eq_threeCellFill_iff_of_norm_eq_one {a b c : ℂ} (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) :
    a = threeCellFill b c ↔ c - a * b = 0 := by
  have hb0 : b ≠ 0 := norm_ne_zero_iff.mp (by rw [hb]; exact one_ne_zero)
  rw [threeCellFill_of_norm_eq_one hb hc, eq_div_iff hb0, sub_eq_zero]
  exact eq_comm

example := eq_threeCellFill_iff_of_norm_eq_one (a := 1) (b := 1) (c := 1) (by simp) (by simp)

/-- **Rank one from three cells**: if `det K = 0`, M recovers `K` from the cells other than
`(0, 1)` exactly when `|k₀₁| ≤ |k₁₀|`. -/
theorem forall_isMinOn_threeCells_iff_of_det_eq_zero {K : Matrix (Fin 2) (Fin 2) ℂ}
    (hK : K.det = 0) :
    (∀ X ∈ {X' : Matrix (Fin 2) (Fin 2) ℂ |
        ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm
        {X' | ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2} X ↔
        X = K)) ↔ ‖K 0 1‖ ≤ ‖K 1 0‖ := by
  rw [det_fin_two, sub_eq_zero] at hK
  rw [forall_isMinOn_threeCells_iff, hK]
  exact eq_threeCellFill_mul_iff _ _

/-- M recovers `!![2, 1; 2, 1]` from three cells, and not `!![1, 2; 1, 2]`. -/
example := (forall_isMinOn_threeCells_iff_of_det_eq_zero (K := !![(2 : ℂ), 1; 2, 1])
  (by norm_num)).mpr (by norm_num)

example := (forall_isMinOn_threeCells_iff_of_det_eq_zero (K := !![(1 : ℂ), 2; 1, 2])
  (by norm_num)).not.mpr (by norm_num)

/-- **Rank two from three cells**: if `det K ≠ 0`, M recovers `K` from the cells other than
`(0, 1)` exactly when `k₀₁ = conj(k₁₀) det K / |det K|`. -/
theorem forall_isMinOn_threeCells_iff_of_det_ne_zero {K : Matrix (Fin 2) (Fin 2) ℂ}
    (hK : K.det ≠ 0) :
    (∀ X ∈ {X' : Matrix (Fin 2) (Fin 2) ℂ |
        ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm
        {X' | ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2} X ↔
        X = K)) ↔ K 0 1 = starRingEnd ℂ (K 1 0) * K.det / ‖K.det‖ := by
  rw [det_fin_two] at hK ⊢
  rw [forall_isMinOn_threeCells_iff]
  exact eq_threeCellFill_iff_of_sub_ne_zero hK

/-- M recovers `!![2, 1; 1, 1]` from three cells, and not `!![2, 1; 1, −1]`. -/
example := (forall_isMinOn_threeCells_iff_of_det_ne_zero (K := !![(2 : ℂ), 1; 1, 1])
  (by norm_num)).mpr (by norm_num)

example := (forall_isMinOn_threeCells_iff_of_det_ne_zero (K := !![(2 : ℂ), 1; 1, -1])
  (by norm_num)).not.mpr (by norm_num)

/-- **Moduli one from three cells**: if every entry of `K` has modulus one, M recovers `K` from
the cells other than `(0, 1)` exactly when `det K = 0`, so at rank two never. -/
theorem forall_isMinOn_threeCells_iff_det_eq_zero {K : Matrix (Fin 2) (Fin 2) ℂ}
    (hK : ∀ i j, ‖K i j‖ = 1) :
    (∀ X ∈ {X' : Matrix (Fin 2) (Fin 2) ℂ |
        ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm
        {X' | ∀ ij ∈ {ij : Fin 2 × Fin 2 | ij ≠ (0, 1)}, X' ij.1 ij.2 = K ij.1 ij.2} X ↔
        X = K)) ↔ K.det = 0 := by
  rw [forall_isMinOn_threeCells_iff, det_fin_two]
  exact eq_threeCellFill_iff_of_norm_eq_one (hK 1 0) (by rw [norm_mul, hK, hK, one_mul])

/-- M does not recover the Hadamard matrix `!![1, 1; 1, −1]` from three cells. -/
example := (forall_isMinOn_threeCells_iff_det_eq_zero (K := !![(1 : ℂ), 1; 1, -1])
  (fun i j => by fin_cases i <;> fin_cases j <;> simp)).not.mpr (by norm_num)

end Transformer.Grokking.FlatSpectrum
