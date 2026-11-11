module

public import Transformer.Grokking.FlatSpectrum.Training.CharacterCode
public import Transformer.Grokking.FlatSpectrum.Training.MarginDirection

/-!
# With every cell seen the flat kernel is the only product of largest margin

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE11.
Let the answer code be mirrored at the gap `g > 0`, let every cell be seen, and let the flat
kernel `K` show the labels. Then `μ* = g / ‖K‖_S1` (`maxMargin_codeProbe_eq`), the margin of
`K / ‖K‖_S1` (`margin_inv_smul_codeProbe`). If `g < 2`, no other product of the unit ball has
this margin (`eq_inv_smul_of_margin_codeProbe_eq`). For such a `Z` the bounds of `MirroredCode`
hold with equality at every cell (`mul_re_eq_margin_of_margin_eq`). So `Z_ij` has the readings
`Re (conj K_ij Z_ij) = 1 / ‖K‖_S1` and `Re (conj φ(c) Z_ij) = (1 − g) / ‖K‖_S1` for a
neighbour `φ(c)` of `K_ij`, and two readings along independent codewords fix a complex number
(`eq_mul_of_re_conj_mul`, `apply_cell_eq_of_margin_eq`).

The witness is the kernel of `a³ + a b` for `p = 3` with all nine cells seen and the code `ψ`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open scoped ComplexConjugate

/-- Two readings fix a complex number: for unit `a`, `b` with `|Re (conj b a)| < 1`, if
`Re (conj a x) = r` and `Re (conj b x) = r Re (conj b a)`, then `x = r a`. -/
lemma eq_mul_of_re_conj_mul {a b x : ℂ} {r : ℝ} (ha : ‖a‖ = 1) (hb : ‖b‖ = 1)
    (hab : |(conj b * a).re| < 1) (h1 : (conj a * x).re = r)
    (h2 : (conj b * x).re = r * (conj b * a).re) : x = (r : ℂ) * a := by
  have ha' : a.re * a.re + a.im * a.im = 1 := by
    rw [← Complex.normSq_apply, ← Complex.sq_norm, ha, one_pow]
  have hb' : b.re * b.re + b.im * b.im = 1 := by
    rw [← Complex.normSq_apply, ← Complex.sq_norm, hb, one_pow]
  simp only [Complex.mul_re, Complex.conj_re, Complex.conj_im, neg_mul, sub_neg_eq_add]
    at hab h1 h2
  have hD : a.re * b.im - a.im * b.re ≠ 0 := by
    intro h0
    have h3 : (b.re * a.re + b.im * a.im) ^ 2 = 1 := by
      linear_combination (b.re * b.re + b.im * b.im) * ha' + hb'
        - (a.re * b.im - a.im * b.re) * h0
    have h4 := (sq_lt_one_iff_abs_lt_one _).2 hab
    rw [h3] at h4
    exact lt_irrefl _ h4
  refine Complex.ext ?_ ?_
  · rw [Complex.re_ofReal_mul]
    exact mul_right_cancel₀ hD (by linear_combination b.im * h1 - r * b.im * ha' - a.im * h2)
  · rw [Complex.im_ofReal_mul]
    exact mul_right_cancel₀ hD (by linear_combination a.re * h2 - b.re * h1 + r * b.re * ha')

variable {m n S C : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
  [Fintype S] [Fintype C] [Nonempty S] [Nontrivial C] {φ : C → ℂ} {cell : S → m × n}
  {y : S → C} {K : Matrix m n ℂ} {g c₀ : ℝ} (h : IsMirroredCode φ g) (hg : 0 < g)
  (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hsurj : Function.Surjective cell)
  (hc : 0 < c₀) (hflat : IsFlat K c₀)

include h hg hK hsurj hc hflat

/-- **The largest margin with every cell seen is that of the flat kernel**
(`review/flat_spectrum.md`, item XE11): `μ* = g / ‖K‖_S1`, for a code mirrored at `g > 0` and
a flat `K` that shows the labels. -/
theorem maxMargin_codeProbe_eq : maxMargin (codeProbe φ cell) y = g / nuclearNorm K := by
  have hμ := margin_codeProbe_eq (cell := cell) (y := y) h hK
  have hB : 0 < nuclearNorm K :=
    nuclearNorm_pos_of_margin_pos (A := codeProbe φ cell) (y := y) (hμ.symm ▸ hg)
  refine le_antisymm (csSup_le ⟨_, ⟨0, nuclearNorm_zero.trans_le zero_le_one, rfl⟩⟩ ?_) ?_
  · rintro _ ⟨X, hX, rfl⟩
    rw [le_div_iff₀ hB]
    exact (margin_codeProbe_mul_nuclearNorm_le h hg.le hK hsurj hc hflat X).trans
      (mul_le_of_le_one_right hg.le hX)
  · rw [← hμ]
    exact margin_div_le_maxMargin K

/-- The normalized flat kernel has the largest margin: `μ(K / ‖K‖_S1) = μ*`. -/
theorem margin_inv_smul_codeProbe :
    margin (codeProbe φ cell) y ((nuclearNorm K)⁻¹ • K) = maxMargin (codeProbe φ cell) y := by
  rw [margin_inv_smul, margin_codeProbe_eq h hK, maxMargin_codeProbe_eq h hg hK hsurj hc hflat]

variable {Z : Matrix m n ℂ} (h1 : nuclearNorm Z ≤ 1)
  (h2 : margin (codeProbe φ cell) y Z = g / nuclearNorm K)

include h1 h2

/-- A product of the unit ball with the margin `g / ‖K‖_S1` has
`g Re (conj φ(y s) Z_{cell s}) = μ(Z)` at every sample: the bounds on its margin are
equalities. -/
lemma mul_re_eq_margin_of_margin_eq (s : S) :
    g * (conj (φ (y s)) * Z (cell s).1 (cell s).2).re = margin (codeProbe φ cell) y Z := by
  have hB : nuclearNorm K ≠ 0 := (nuclearNorm_pos_of_margin_pos (A := codeProbe φ cell)
    (y := y) ((margin_codeProbe_eq h hK).symm ▸ hg)).ne'
  have hnn : ∀ i j, 0 ≤ g * (conj (K i j) * Z i j).re - margin (codeProbe φ cell) y Z :=
    fun i j => sub_nonneg.2 (margin_codeProbe_le_mul_re_apply h hK hsurj Z i j)
  have hsum : ∑ i, ∑ j, (g * (conj (K i j) * Z i j).re - margin (codeProbe φ cell) y Z)
      ≤ 0 := by
    have e : ∑ i, ∑ j, (g * (conj (K i j) * Z i j).re - margin (codeProbe φ cell) y Z)
        = g * pairing K Z
          - Fintype.card m * Fintype.card n * margin (codeProbe φ cell) y Z := by
      rw [pairing_eq_sum_re]
      simp only [Finset.sum_sub_distrib, Finset.mul_sum, Finset.sum_const, Finset.card_univ,
        nsmul_eq_mul]
      ring
    have e2 : √c₀ * nuclearNorm K * (g / nuclearNorm K) = g * √c₀ := by field_simp
    rw [e, card_mul_card_eq_of_isFlat hc hflat (norm_apply_eq_one_of_surjective h.1 hK hsurj),
      h2, e2]
    nlinarith [mul_le_mul_of_nonneg_left (hflat.pairing_le hc Z) hg.le,
      mul_le_mul_of_nonneg_left h1 (mul_nonneg hg.le (Real.sqrt_nonneg c₀))]
  have h0 := (Finset.sum_eq_zero_iff_of_nonneg fun i _ =>
    Finset.sum_nonneg fun j _ => hnn i j).1 (le_antisymm hsum
      (Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => hnn i j))
  have h00 := (Finset.sum_eq_zero_iff_of_nonneg fun j _ => hnn (cell s).1 j).1
    (h0 (cell s).1 (Finset.mem_univ _)) (cell s).2 (Finset.mem_univ _)
  rw [hK s] at h00
  linarith

/-- If `g < 2`, such a product is `K / ‖K‖_S1` at the cell of every sample. -/
lemma apply_cell_eq_of_margin_eq (hg2 : g < 2) (s : S) :
    Z (cell s).1 (cell s).2 = ((nuclearNorm K)⁻¹ : ℝ) * φ (y s) := by
  obtain ⟨c, c', hcy, hcy', hsum⟩ := h.2.2 (y s)
  have hr := mul_re_eq_margin_of_margin_eq h hg hK hsurj hc hflat h1 h2 s
  have hZ1 := margin_le_gap (A := codeProbe φ cell) (y := y) Z hcy
  have hZ2 := margin_le_gap (A := codeProbe φ cell) (y := y) Z hcy'
  have hZ3 := gap_codeProbe_add_gap (cell := cell) Z s hsum
  have hK1 := gap_codeProbe (cell := cell) h.1 hK s c
  have hK2 := gap_codeProbe (cell := cell) h.1 hK s c'
  have hK3 := gap_codeProbe_add_gap (cell := cell) K s hsum
  have hd1 := h.2.1 (y s) c hcy
  have hd2 := h.2.1 (y s) c' hcy'
  have haa : (conj (φ (y s)) * φ (y s)).re = 1 := by
    rw [Complex.conj_mul', h.1, ← Complex.ofReal_pow, Complex.ofReal_re, one_pow]
  rw [hK s, haa] at hK3
  have eK : gap (codeProbe φ cell) y K s c = 1 - (conj (φ c) * φ (y s)).re := by
    simp only [gap, pairing_codeProbe, hK s, haa]
  have eZ : gap (codeProbe φ cell) y Z s c = (conj (φ (y s)) * Z (cell s).1 (cell s).2).re
      - (conj (φ c) * Z (cell s).1 (cell s).2).re := by
    simp only [gap, pairing_codeProbe]
  have hab : (conj (φ c) * φ (y s)).re = 1 - g := by linarith
  have hr' : (conj (φ (y s)) * Z (cell s).1 (cell s).2).re = (nuclearNorm K)⁻¹ :=
    mul_left_cancel₀ hg.ne' (hr.trans (h2.trans (div_eq_mul_inv _ _)))
  refine eq_mul_of_re_conj_mul (b := φ c) (h.1 _) (h.1 _) ?_ hr' ?_
  · rw [hab, abs_lt]
    constructor <;> linarith
  · rw [hr'] at eZ hr hZ3
    rw [hab]
    linarith

omit h2 in
/-- **With every cell seen the normalized flat kernel is the only product of largest margin**
(`review/flat_spectrum.md`, item XE11), for a code mirrored at a gap `0 < g < 2`: a product
with `‖Z‖_S1 ≤ 1` and `μ(Z) = μ*` is `K / ‖K‖_S1`. -/
theorem eq_inv_smul_of_margin_codeProbe_eq (hg2 : g < 2)
    (h3 : margin (codeProbe φ cell) y Z = maxMargin (codeProbe φ cell) y) :
    Z = (nuclearNorm K)⁻¹ • K := by
  rw [maxMargin_codeProbe_eq h hg hK hsurj hc hflat] at h3
  ext i j
  obtain ⟨s, hs⟩ := hsurj (i, j)
  have h4 := apply_cell_eq_of_margin_eq h hg hK hsurj hc hflat h1 h3 hg2 s
  rw [← hK s, hs] at h4
  exact h4

omit h hg hK hsurj hc hflat h1 h2

example := eq_mul_of_re_conj_mul (a := 1) (b := Complex.I) (x := 1) (r := 1) norm_one
  Complex.norm_I (by simp) (by simp) (by simp)

/-- The witness: the kernel of `a³ + a b` for `p = 3`, all nine cells seen, the code `ψ`. Its
largest margin is that of `K / ‖K‖_S1`, to which the statements above apply. -/
example (s : ZMod 3 × ZMod 3) :=
  have h := isMirroredCode_stdAddChar (p := 3) (by norm_num)
  have hg := one_sub_cos_two_pi_div_pos (p := 3) (by norm_num)
  have hK : ∀ s : ZMod 3 × ZMod 3,
      cubicKernel 3 (id s).1 (id s).2 = ZMod.stdAddChar (s.1 ^ 3 + s.1 * s.2) := fun _ => rfl
  have hc : (0 : ℝ) < (3 : ℕ) := by norm_num
  have hf := isFlat_cubicKernel 3
  have hmax := maxMargin_codeProbe_eq h hg hK Function.surjective_id hc hf
  have hK' := margin_inv_smul_codeProbe h hg hK Function.surjective_id hc hf
  have hle := nuclearNorm_inv_smul_le (cubicKernel 3)
  And.intro (mul_re_eq_margin_of_margin_eq h hg hK Function.surjective_id hc hf hle
      (hK'.trans hmax) s)
    (And.intro (apply_cell_eq_of_margin_eq h hg hK Function.surjective_id hc hf hle
        (hK'.trans hmax) (one_sub_cos_two_pi_div_lt_two (by norm_num)) s)
      (eq_inv_smul_of_margin_codeProbe_eq h hg hK Function.surjective_id hc hf hle
        (one_sub_cos_two_pi_div_lt_two (by norm_num)) hK'))

end Transformer.Grokking.FlatSpectrum.Training
