module

public import Transformer.Grokking.FlatSpectrum.OneMissing
public import Transformer.Grokking.FlatSpectrum.Kernels

/-!
# The one-entry fill of a flat kernel is the opposite phase

Source: `review/flat_spectrum.md`, Corollary. Write the fill as `X_ij = K_ij (1 − s)`. By
`nuclearNorm_flat_add_single`, `‖X‖_S1 = √N [(N − 2) + √F(s)]` with
`F(s) = |1 − s/N|² + |s|² (N − 1)/N² + 1 + 2 |1 − s/N|`; for real `s ≤ N` this is the
source's `4 − 4s/N + s²/N`. Since `|1 − s/N| ≥ Re (1 − s/N)`,
`F(s) ≥ 4 − 4/N + |s − 2|²/N` with equality at `s = 2`, so the only minimum-norm fill is
`s = 2`, `X_ij = −K_ij`. The source's real formula needs `s ≤ N`: beyond it
`|1 − s/N| = s/N − 1` and `‖X‖_S1 = √N (N − 2) + s`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- The fill `X_ij = K_ij (1 − s)` of a flat `K`, `review/flat_spectrum.md`, Corollary:
`‖X‖_S1 = √N [(N − 2) + √(|1 − s/N|² + |s|² (N − 1)/N² + 1 + 2 |1 − s/N|)]`. -/
theorem nuclearNorm_flat_fill {K : Matrix n n ℂ}
    (hK : Kᴴ * K = (Fintype.card n : ℂ) • 1) {i j : n} (hij : ‖K i j‖ = 1)
    (hN : 2 ≤ Fintype.card n) (s : ℂ) :
    nuclearNorm (K - (s * K i j) • single i j 1) = √(Fintype.card n : ℝ) *
      ((Fintype.card n - 2) + √(‖1 - s / Fintype.card n‖ ^ 2 +
        ‖s‖ ^ 2 * (Fintype.card n - 1) / Fintype.card n ^ 2 + 1 +
        2 * ‖1 - s / Fintype.card n‖)) := by
  have hKK : K i j * star (K i j) = 1 := by
    rw [Complex.star_def, Complex.mul_conj', hij]
    simp
  have h := nuclearNorm_flat_add_single hK hij hN (-(s * K i j))
  rw [neg_smul, ← sub_eq_add_neg] at h
  rw [h, show 1 + -(s * K i j) * star (K i j) / (Fintype.card n : ℂ) = 1 - s / Fintype.card n by
    rw [neg_mul, mul_assoc, hKK, mul_one, neg_div, ← sub_eq_add_neg], norm_neg, norm_mul, hij,
    mul_one]

example := nuclearNorm_flat_fill (K := !![(1 : ℂ), 1; 1, -1]) (i := 0) (j := 0)
  (by ext a b; fin_cases a <;> fin_cases b <;> simp [mul_apply, Fin.sum_univ_two] <;> norm_num)
  (by simp) le_rfl 2

/-- **Corollary**, the source's formula: for real `s ≤ N`,
`‖X‖_S1 = √N [(N − 2) + √(4 − 4 s/N + s²/N)]`. The source writes it for every real `s`; the
bound `s ≤ N`, where `|1 − s/N| = 1 − s/N`, is added: for `s > N` the value is
`√N (N − 2) + s`. -/
theorem nuclearNorm_flat_fill_real {K : Matrix n n ℂ}
    (hK : Kᴴ * K = (Fintype.card n : ℂ) • 1) {i j : n} (hij : ‖K i j‖ = 1)
    (hN : 2 ≤ Fintype.card n) {s : ℝ} (hs : s ≤ Fintype.card n) :
    nuclearNorm (K - ((s : ℂ) * K i j) • single i j 1) = √(Fintype.card n : ℝ) *
      ((Fintype.card n - 2) + √(4 - 4 * s / Fintype.card n + s ^ 2 / Fintype.card n)) := by
  have hN0 : (0 : ℝ) < Fintype.card n := by exact_mod_cast (by omega : 0 < Fintype.card n)
  have h1 : ‖(1 : ℂ) - (s : ℂ) / Fintype.card n‖ = 1 - s / Fintype.card n := by
    rw [show (1 : ℂ) - (s : ℂ) / Fintype.card n = ((1 - s / Fintype.card n : ℝ) : ℂ) by
      push_cast; ring, Complex.norm_real,
      Real.norm_of_nonneg (sub_nonneg.mpr ((div_le_one hN0).mpr hs))]
  rw [nuclearNorm_flat_fill hK hij hN, h1, Complex.norm_real, Real.norm_eq_abs, sq_abs]
  congr 3
  field_simp
  ring

example := nuclearNorm_flat_fill_real (K := !![(1 : ℂ), 1; 1, -1]) (i := 0) (j := 0)
  (by ext a b; fin_cases a <;> fin_cases b <;> simp [mul_apply, Fin.sum_univ_two] <;> norm_num)
  (by simp) le_rfl (s := 2) (by simp)

/-- `F(s) ≥ 4 − 4/N + |s − 2|²/N`: a complex `s` only makes the fill costlier,
`review/flat_spectrum.md`, Corollary. -/
lemma fill_bound {N : ℝ} (hN : 0 < N) (s : ℂ) :
    4 - 4 / N + ‖s - 2‖ ^ 2 / N ≤
      ‖1 - s / N‖ ^ 2 + ‖s‖ ^ 2 * (N - 1) / N ^ 2 + 1 + 2 * ‖1 - s / N‖ := by
  have h1 := Complex.re_le_norm (1 - s / N)
  have e (w : ℂ) : ‖w‖ ^ 2 = w.re ^ 2 + w.im ^ 2 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    ring
  have hre : (1 - s / N).re = 1 - s.re / N := by simp
  have him : (1 - s / N).im = -(s.im / N) := by simp
  rw [hre] at h1
  rw [e (1 - s / N), e s, e (s - 2), hre, him]
  have key : (1 - s.re / N) ^ 2 + (-(s.im / N)) ^ 2 + (s.re ^ 2 + s.im ^ 2) * (N - 1) / N ^ 2 +
      1 + 2 * (1 - s.re / N) = 4 - 4 / N + ((s - 2).re ^ 2 + (s - 2).im ^ 2) / N := by
    simp only [Complex.sub_re, Complex.sub_im, Complex.re_ofNat, Complex.im_ofNat, sub_zero]
    field_simp
    ring
  linarith

example := fill_bound (N := 2) two_pos 2

omit [Fintype n] in
/-- A matrix agreeing with `K` off `(i, j)` is the fill `X_ij = K_ij (1 − s)`,
`s = 1 − X_ij conj(K_ij)`. -/
lemma eq_fill_of_mem {K Y : Matrix n n ℂ} {i j : n} (hij : ‖K i j‖ = 1)
    (hY : ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), Y kl.1 kl.2 = K kl.1 kl.2) :
    Y = K - ((1 - Y i j * star (K i j)) * K i j) • single i j 1 := by
  have hKK : star (K i j) * K i j = 1 := by
    rw [Complex.star_def, Complex.conj_mul', hij]
    simp
  ext k l
  by_cases h : k = i ∧ l = j
  · obtain ⟨rfl, rfl⟩ := h
    simp only [Matrix.sub_apply, Matrix.smul_apply, single_apply_same, smul_eq_mul, mul_one]
    linear_combination (-(Y k l)) * hKK
  · have hkl := hY (k, l) (by simpa using h)
    have hne : ¬(i = k ∧ j = l) := fun ⟨h1, h2⟩ => h ⟨h1.symm, h2.symm⟩
    simp [hne, hkl]

example := eq_fill_of_mem (K := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Y := 1) (i := 0) (j := 0)
  (by simp) (fun _ _ => rfl)

/-- **Corollary** of `review/flat_spectrum.md`: with one entry missing, the
minimum-nuclear-norm completion of a flat `K` (`K* K = N I`, `|K_ij| = 1`, `N ≥ 2`) fills it
with the opposite phase `X_ij = −K_ij`, and no other fill attains the minimum. -/
theorem isMinOn_flat_fill_iff {K : Matrix n n ℂ} (hK : Kᴴ * K = (Fintype.card n : ℂ) • 1)
    {i j : n} (hij : ‖K i j‖ = 1) (hN : 2 ≤ Fintype.card n) {X : Matrix n n ℂ}
    (hX : ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), X kl.1 kl.2 = K kl.1 kl.2) :
    IsMinOn nuclearNorm {Y | ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), Y kl.1 kl.2 = K kl.1 kl.2} X ↔
      X i j = -K i j := by
  have hN' : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
  have hN0 : (0 : ℝ) < Fintype.card n := by linarith
  have hsq : 0 < √(Fintype.card n : ℝ) := Real.sqrt_pos.mpr hN0
  have hKK : star (K i j) * K i j = 1 := by
    rw [Complex.star_def, Complex.conj_mul', hij]
    simp
  have hlow (Y : Matrix n n ℂ)
      (hY : ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)), Y kl.1 kl.2 = K kl.1 kl.2) :
      √(Fintype.card n : ℝ) * ((Fintype.card n - 2) + √(4 - 4 / Fintype.card n +
        ‖(1 - Y i j * star (K i j)) - 2‖ ^ 2 / Fintype.card n)) ≤ nuclearNorm Y := by
    have h := nuclearNorm_flat_fill hK hij hN (1 - Y i j * star (K i j))
    rw [← eq_fill_of_mem hij hY] at h
    rw [h]
    have hb := fill_bound hN0 (1 - Y i j * star (K i j))
    rw [Complex.ofReal_natCast] at hb
    gcongr
  have hval : nuclearNorm (K - ((2 : ℂ) * K i j) • single i j 1) =
      √(Fintype.card n : ℝ) * ((Fintype.card n - 2) + √(4 - 4 / Fintype.card n)) := by
    rw [nuclearNorm_flat_fill hK hij hN]
    have h2 : ‖(1 : ℂ) - 2 / Fintype.card n‖ = 1 - 2 / Fintype.card n := by
      rw [show (1 : ℂ) - 2 / Fintype.card n = ((1 - 2 / Fintype.card n : ℝ) : ℂ) by
        push_cast; ring, Complex.norm_real,
        Real.norm_of_nonneg (sub_nonneg.mpr ((div_le_one hN0).mpr hN'))]
    rw [h2, show ‖(2 : ℂ)‖ = 2 by simp]
    congr 3
    field_simp
    ring
  have hmem0 : ∀ kl ∈ ({(i, j)}ᶜ : Set (n × n)),
      (K - ((2 : ℂ) * K i j) • single i j 1 : Matrix n n ℂ) kl.1 kl.2 = K kl.1 kl.2 := by
    intro kl hkl
    have hne : ¬(i = kl.1 ∧ j = kl.2) := fun ⟨h1, h2⟩ => hkl (by simp [h1, h2])
    simp [hne]
  have h4 : 0 ≤ 4 - 4 / (Fintype.card n : ℝ) := by
    rw [sub_nonneg, div_le_iff₀ hN0]
    linarith
  rw [isMinOn_iff]
  constructor
  · intro hmin
    have h3 := le_of_add_le_add_left (le_of_mul_le_mul_left
      ((hlow X hX).trans ((hmin _ hmem0).trans hval.le)) hsq)
    have h5 := (add_le_iff_nonpos_right _).mp ((Real.sqrt_le_sqrt_iff h4).mp h3)
    rw [div_le_iff₀ hN0, zero_mul] at h5
    have h7 : (1 - X i j * star (K i j)) - 2 = 0 := by
      simpa using le_antisymm h5 (sq_nonneg _)
    linear_combination (-(X i j)) * hKK + (-(K i j)) * h7
  · intro hXij Y hY
    have hX0 : X = K - ((2 : ℂ) * K i j) • single i j 1 := by
      have h := eq_fill_of_mem hij hX
      rwa [hXij, show 1 - -K i j * star (K i j) = 2 by linear_combination hKK] at h
    rw [hX0, hval]
    refine le_trans ?_ (hlow Y hY)
    gcongr
    exact le_add_of_nonneg_right (div_nonneg (sq_nonneg _) hN0.le)

example := isMinOn_flat_fill_iff (K := !![(1 : ℂ), 1; 1, -1]) (i := 0) (j := 0)
  (by ext a b; fin_cases a <;> fin_cases b <;> simp [mul_apply, Fin.sum_univ_two] <;> norm_num)
  (by simp) le_rfl (X := !![(1 : ℂ), 1; 1, -1]) (fun _ _ => rfl)

end Transformer.Grokking.FlatSpectrum
