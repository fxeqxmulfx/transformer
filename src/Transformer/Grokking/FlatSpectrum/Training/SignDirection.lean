module

public import Transformer.Grokking.FlatSpectrum.Training.SignKernel

/-!
# The code `±1`: the largest margin is attained twice and the direction need not converge

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE11 and XE13.
Statement 4 of item XE11 takes a gap `g < 2`. At `g = 2`, the code `±1`, it fails. Let every
cell be seen once, let the flat kernel `K` show the labels, and let `K' ≠ K` have the real
parts of `K` and at most its nuclear norm. Then

* `‖K'‖_S1 = ‖K‖_S1` (`nuclearNorm_eq_of_re_eq`), by statement 3 of item XE11 at `X = K'`;
* `K' / ‖K‖_S1` is a second product of the unit ball with the largest margin
  (`exists_ne_margin_eq_maxMargin_signCode`);
* the multiples `(j + 1) K` are trained products for the weight decays
  `2 √c / (1 + e^{2 (j + 1)}) → 0⁺` (`tendsto_signDecay`), with the direction of `K`
  (`exists_seq_isNuclearMin_tendsto_signCode`);
* with `(j + 1) K'` at the odd `j` instead, the directions alternate between two matrices and
  do not converge (`exists_seq_isNuclearMin_not_tendsto_signCode`); these are products of
  trained fits (`exists_seq_isDecayedMin_not_tendsto_signCode`).

The witness is `signKernel` with `signTwin` (`exists_ne_margin_eq_maxMargin_signKernel`,
`exists_seq_isDecayedMin_not_tendsto_signKernel`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype S]
  [Nonempty S] {cell : S → m × n} {y : S → Bool} {K K' : Matrix m n ℂ} {c₀ : ℝ}

omit [Fintype S] [Nonempty S] in
/-- The weight decays of the multiples `(j + 1) K` tend to `0`. -/
lemma tendsto_signDecay (c₀ : ℝ) :
    Tendsto (fun j : ℕ => 2 * √c₀ / (1 + Real.exp (2 * ((j : ℝ) + 1)))) atTop (𝓝 0) :=
  tendsto_const_nhds.div_atTop (tendsto_atTop_add_const_left _ _ (Real.tendsto_exp_atTop.comp
    ((tendsto_atTop_add_const_right _ _ tendsto_natCast_atTop_atTop).const_mul_atTop two_pos)))

/-- A matrix with the real parts of the flat kernel and at most its nuclear norm has its
nuclear norm, by statement 3 of item XE11. -/
lemma nuclearNorm_eq_of_re_eq (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hre : ∀ i j, (K' i j).re = (K i j).re) (hle : nuclearNorm K' ≤ nuclearNorm K) :
    nuclearNorm K' = nuclearNorm K := by
  refine le_antisymm hle ?_
  have h1 := margin_codeProbe_mul_nuclearNorm_le (y := y) isMirroredCode_signCode zero_le_two
    hK hcell.surjective hc hflat K'
  rw [margin_signCode_congr hre, margin_codeProbe_eq isMirroredCode_signCode hK] at h1
  linarith

/-- A kernel that shows the labels of the code `±1` is not zero. -/
lemma nuclearNorm_pos_of_signCode (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) :
    0 < nuclearNorm K :=
  nuclearNorm_pos_of_margin_pos (A := codeProbe signCode cell) (y := y)
    ((margin_codeProbe_eq isMirroredCode_signCode hK).symm ▸ two_pos)

/-- **At the gap `2` the product of largest margin need not be unique**
(`review/flat_spectrum.md`, item XE11, statement 4): for the code `±1`, a matrix `K' ≠ K` with
the real parts of the flat kernel and at most its nuclear norm gives a second one. -/
theorem exists_ne_margin_eq_maxMargin_signCode (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hre : ∀ i j, (K' i j).re = (K i j).re) (hle : nuclearNorm K' ≤ nuclearNorm K)
    (hne : K' ≠ K) :
    ∃ Z, nuclearNorm Z ≤ 1 ∧
      margin (codeProbe signCode cell) y Z = maxMargin (codeProbe signCode cell) y ∧
      Z ≠ (nuclearNorm K)⁻¹ • K := by
  have hpos := nuclearNorm_pos_of_signCode hK
  refine ⟨(nuclearNorm K)⁻¹ • K', ?_, ?_, fun h => hne ?_⟩
  · rw [nuclearNorm_real_smul (inv_nonneg.2 hpos.le)]
    exact (mul_le_mul_of_nonneg_left hle (inv_nonneg.2 hpos.le)).trans_eq
      (inv_mul_cancel₀ hpos.ne')
  · rw [← margin_inv_smul_codeProbe isMirroredCode_signCode two_pos hK hcell.surjective hc
      hflat]
    exact margin_signCode_congr fun i j => by
      rw [Matrix.smul_apply, Matrix.smul_apply, Complex.smul_re, Complex.smul_re, hre]
  · exact smul_right_injective _ (inv_ne_zero hpos.ne') h

omit [Nonempty S] in
/-- **At the gap `2` some trained products converge in direction to the flat kernel**
(`review/flat_spectrum.md`, item XE11): the multiples `(j + 1) K`. -/
theorem exists_seq_isNuclearMin_tendsto_signCode (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀) :
    ∃ (lams : ℕ → ℝ) (X : ℕ → Matrix m n ℂ), (∀ j, 0 < lams j) ∧ Tendsto lams atTop (𝓝 0) ∧
      (∀ j, IsNuclearMin (ceLoss (codeProbe signCode cell) y) (lams j) (X j)) ∧
      Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 ((nuclearNorm K)⁻¹ • K)) :=
  ⟨_, fun j => ((j : ℝ) + 1) • K, fun j => by positivity, tendsto_signDecay c₀,
    fun j => isNuclearMin_smul_signCode hcell hK hc hflat (by positivity),
    tendsto_const_nhds.congr fun j => (inv_nuclearNorm_smul_smul (by positivity) K).symm⟩

/-- **At the gap `2` the direction of trained products need not converge**
(`review/flat_spectrum.md`, items XE11 and XE13): the multiples `(j + 1) K` at the even `j`
and `(j + 1) K'` at the odd `j`. -/
theorem exists_seq_isNuclearMin_not_tendsto_signCode (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hre : ∀ i j, (K' i j).re = (K i j).re) (hle : nuclearNorm K' ≤ nuclearNorm K)
    (hne : K' ≠ K) :
    ∃ (lams : ℕ → ℝ) (X : ℕ → Matrix m n ℂ), (∀ j, 0 < lams j) ∧ Tendsto lams atTop (𝓝 0) ∧
      (∀ j, IsNuclearMin (ceLoss (codeProbe signCode cell) y) (lams j) (X j)) ∧
      ∀ Z, ¬ Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 Z) := by
  have hpos := nuclearNorm_pos_of_signCode hK
  have hnorm := nuclearNorm_eq_of_re_eq hcell hK hc hflat hre hle
  refine ⟨_, fun j => ((j : ℝ) + 1) • if Even j then K else K', fun j => by positivity,
    tendsto_signDecay c₀, fun j => ?_, fun Z hZ => hne ?_⟩
  · dsimp only
    split_ifs
    · exact isNuclearMin_smul_signCode hcell hK hc hflat (by positivity)
    · exact isNuclearMin_smul_of_re_eq hcell hK hc hflat hre hle (by positivity)
  · have h0 : Tendsto (fun j : ℕ => 2 * j) atTop atTop :=
      tendsto_atTop_mono (fun j => by change j ≤ _; omega) tendsto_id
    have h1 : Tendsto (fun j : ℕ => 2 * j + 1) atTop atTop :=
      tendsto_atTop_mono (fun j => by change j ≤ _; omega) tendsto_id
    have e0 : Z = (nuclearNorm K)⁻¹ • K :=
      tendsto_nhds_unique (hZ.comp h0) (tendsto_const_nhds.congr fun j => by
        simp only [Function.comp_apply, ite_eq_left (even_two_mul j)]
        exact (inv_nuclearNorm_smul_smul (by positivity) K).symm)
    have e1 : Z = (nuclearNorm K')⁻¹ • K' :=
      tendsto_nhds_unique (hZ.comp h1) (tendsto_const_nhds.congr fun j => by
        simp only [Function.comp_apply, ite_eq_right (Nat.not_even_two_mul_add_one j)]
        exact (inv_nuclearNorm_smul_smul (by positivity) K').symm)
    rw [hnorm] at e1
    exact smul_right_injective _ (inv_ne_zero hpos.ne') (e1.symm.trans e0)

/-- **At the gap `2` the direction of trained fits need not converge**
(`review/flat_spectrum.md`, items XE11 and XE13), for some trained fits of the width of the
columns. -/
theorem exists_seq_isDecayedMin_not_tendsto_signCode (hcell : Function.Bijective cell)
    (hK : ∀ s, K (cell s).1 (cell s).2 = signCode (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hre : ∀ i j, (K' i j).re = (K i j).re) (hle : nuclearNorm K' ≤ nuclearNorm K)
    (hne : K' ≠ K) :
    ∃ (lams : ℕ → ℝ) (Us : ℕ → Matrix m n ℂ) (Vs : ℕ → Matrix n n ℂ), (∀ j, 0 < lams j) ∧
      Tendsto lams atTop (𝓝 0) ∧
      (∀ j, IsDecayedMin (ceLoss (codeProbe signCode cell) y) (lams j) (Us j) (Vs j)) ∧
      ∀ Z, ¬ Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
        (𝓝 Z) := by
  obtain ⟨lams, X, hlam, hlim, hmin, hno⟩ :=
    exists_seq_isNuclearMin_not_tendsto_signCode hcell hK hc hflat hre hle hne
  choose Us Vs hUV hfit using fun j => (hmin j).exists_isDecayedMin (hlam j).le
  exact ⟨lams, Us, Vs, hlam, hlim, hfit, fun Z hZ => hno Z (by simpa only [hUV] using hZ)⟩

/-! The witness: `signKernel` with all four cells seen once and the matrix `signTwin`. It
meets every hypothesis of statement 4 of item XE11 but `g < 2`. -/

example : IsMirroredCode signCode 2 ∧ Function.Surjective (id : Fin 2 × Fin 2 → Fin 2 × Fin 2) ∧
    IsFlat signKernel 2 ∧ ∀ s, signKernel (id s).1 (id s).2 = signCode (signLabel s) :=
  ⟨isMirroredCode_signCode, Function.surjective_id, isFlat_signKernel, signKernel_apply⟩

example := nuclearNorm_eq_of_re_eq Function.bijective_id signKernel_apply two_pos
  isFlat_signKernel signTwin_re nuclearNorm_signTwin_le

example := nuclearNorm_pos_of_signCode signKernel_apply

example := exists_seq_isNuclearMin_tendsto_signCode Function.bijective_id signKernel_apply
  two_pos isFlat_signKernel

example := exists_seq_isNuclearMin_not_tendsto_signCode Function.bijective_id signKernel_apply
  two_pos isFlat_signKernel signTwin_re nuclearNorm_signTwin_le signTwin_ne

/-- For the `2 × 2` kernel of signs the product of largest margin is not unique. -/
theorem exists_ne_margin_eq_maxMargin_signKernel :
    ∃ Z, nuclearNorm Z ≤ 1 ∧
      margin (codeProbe signCode id) signLabel Z = maxMargin (codeProbe signCode id) signLabel ∧
      Z ≠ (nuclearNorm signKernel)⁻¹ • signKernel :=
  exists_ne_margin_eq_maxMargin_signCode Function.bijective_id signKernel_apply two_pos
    isFlat_signKernel signTwin_re nuclearNorm_signTwin_le signTwin_ne

/-- For the `2 × 2` kernel of signs some trained fits do not converge in direction. -/
theorem exists_seq_isDecayedMin_not_tendsto_signKernel :
    ∃ (lams : ℕ → ℝ) (Us Vs : ℕ → Matrix (Fin 2) (Fin 2) ℂ), (∀ j, 0 < lams j) ∧
      Tendsto lams atTop (𝓝 0) ∧
      (∀ j, IsDecayedMin (ceLoss (codeProbe signCode id) signLabel) (lams j) (Us j) (Vs j)) ∧
      ∀ Z, ¬ Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
        (𝓝 Z) :=
  exists_seq_isDecayedMin_not_tendsto_signCode Function.bijective_id signKernel_apply two_pos
    isFlat_signKernel signTwin_re nuclearNorm_signTwin_le signTwin_ne

end Transformer.Grokking.FlatSpectrum.Training
