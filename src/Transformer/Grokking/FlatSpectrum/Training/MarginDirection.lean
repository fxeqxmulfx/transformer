module

public import Transformer.Grokking.FlatSpectrum.Training.UnitBall

/-!
# The direction of the trained fits: limit points have the largest margin

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE10.
Item XE6 gives the limit of the margin of the trained fits for their norm. Their direction is
`X_j / ‖X_j‖_S1`, a point of the unit ball of the nuclear norm, which is compact. Let some
product have a positive margin and `λ_j → 0⁺`.

* A limit `Z` of normalized trained fits has `‖Z‖_S1 = 1` and the largest margin of the unit
  ball, `μ(Z) = μ*` (`margin_eq_maxMargin_of_tendsto_inv_smul`). Every subsequence of trained
  fits is one of trained fits, so every limit point of the direction is such a product, and
  limit points exist (`exists_subseq_tendsto_inv_smul`).
* If one product `Z₀` of the unit ball has the margin `μ*`, then `X_j / ‖X_j‖_S1 → Z₀`
  (`tendsto_inv_smul_of_unique`).

The witness is `signProbe`, where `1` is the only product of the unit ball with the margin
`μ* = 2` (`eq_one_of_margin_signProbe`): its trained fits converge in direction to `1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype S] [Fintype C]
  [DecidableEq n] {A : S → C → Matrix m n ℂ} {y : S → C} {lams : ℕ → ℝ}
  {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ}

/-- **A limit of the normalized trained fits has the largest margin**
(`review/flat_spectrum.md`, item XE10): if some product has a positive margin, `λ_j → 0⁺` and
`X_j / ‖X_j‖_S1 → Z`, then `‖Z‖_S1 = 1` and `μ(Z) = μ*`. -/
theorem margin_eq_maxMargin_of_tendsto_inv_smul [Nonempty S] [Nontrivial C]
    {X₀ : Matrix m n ℂ} (hsep : 0 < margin A y X₀) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) {Z₀ : Matrix m n ℂ}
    (hZ : Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 Z₀)) : nuclearNorm Z₀ = 1 ∧ margin A y Z₀ = maxMargin A y := by
  obtain ⟨h1, h2⟩ := margin_eq_maxMargin_of_tendsto hZ (fun j => nuclearNorm_inv_smul_le _)
    ((tendsto_margin_div_nuclearNorm hsep hpos hlam hmin).congr fun j =>
      (margin_inv_smul (A := A) (y := y) _).symm)
  have hM : 0 < maxMargin A y :=
    (div_pos hsep (nuclearNorm_pos_of_margin_pos hsep)).trans_le (margin_div_le_maxMargin X₀)
  have hZpos : 0 < nuclearNorm Z₀ := nuclearNorm_pos_of_margin_pos (h2.symm ▸ hM)
  have h3 := margin_div_le_maxMargin (A := A) (y := y) Z₀
  rw [h2, div_le_iff₀ hZpos] at h3
  exact ⟨le_antisymm h1 (le_of_mul_le_mul_left (by linarith) hM), h2⟩

variable [DecidableEq m]

/-- **A unique product of largest margin is the direction of the trained fits**
(`review/flat_spectrum.md`, item XE10): if `Z₀` is the only product of the unit ball with the
margin `μ*`, then `X_j / ‖X_j‖_S1 → Z₀` as `λ_j → 0⁺`. -/
theorem tendsto_inv_smul_of_unique [Nonempty S] [Nontrivial C] {X₀ : Matrix m n ℂ}
    (hsep : 0 < margin A y X₀) (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) {Z₀ : Matrix m n ℂ}
    (huniq : ∀ Z : Matrix m n ℂ, nuclearNorm Z ≤ 1 → margin A y Z = maxMargin A y → Z = Z₀) :
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop (𝓝 Z₀) := by
  have : FirstCountableTopology (Matrix m n ℂ) :=
    inferInstanceAs (FirstCountableTopology (m → n → ℂ))
  refine tendsto_of_subseq_tendsto fun ns hns => ?_
  obtain ⟨Z, φ, hφ, hZ⟩ := exists_subseq_tendsto_inv_smul fun j => Us (ns j) * (Vs (ns j))ᴴ
  obtain ⟨h1, h2⟩ := margin_eq_maxMargin_of_tendsto_inv_smul (lams := fun j => lams (ns (φ j)))
    (Us := fun j => Us (ns (φ j))) (Vs := fun j => Vs (ns (φ j))) hsep (fun j => hpos _)
    (hlam.comp (hns.comp hφ.tendsto_atTop)) (fun j => hmin _) hZ
  exact ⟨φ, huniq Z h1.le h2 ▸ hZ⟩

/-! The witness: one sample of a `1 × 1` product read with the code `±1`. In the unit ball
`2 Re Z = 2` forces `Z = 1`. -/

lemma eq_one_of_margin_signProbe {Z : Matrix (Fin 1) (Fin 1) ℂ} (h1 : nuclearNorm Z ≤ 1)
    (h2 : margin signProbe (fun _ => true) Z = maxMargin signProbe fun _ => true) : Z = 1 := by
  have h3 := margin_le_gap (A := signProbe) (y := fun _ => true) Z (s := 0) (c := false)
    (by simp)
  rw [h2, maxMargin_signProbe, gap_signProbe] at h3
  have h4 := (norm_apply_le_nuclearNorm Z 0 0).trans h1
  have h5 := Complex.sq_norm (Z 0 0)
  rw [Complex.normSq_apply] at h5
  have h6 : ‖Z 0 0‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg (Z 0 0)]
  have hre : (Z 0 0).re = 1 := by nlinarith [sq_nonneg (Z 0 0).im, sq_nonneg ((Z 0 0).re - 1)]
  have him : (Z 0 0).im = 0 := by nlinarith [sq_nonneg (Z 0 0).im]
  ext i j
  rw [Subsingleton.elim i 0, Subsingleton.elim j 0]
  exact Complex.ext (by simpa using hre) (by simpa using him)

example := eq_one_of_margin_signProbe nuclearNorm_one_fin_one.le
  (le_antisymm (margin_le_maxMargin nuclearNorm_one_fin_one.le)
    (maxMargin_signProbe.trans_le (le_margin le_gap_signProbe_one)))

/-- The trained fits of the witness converge in direction to `1`. -/
example : ∃ Us Vs : ℕ → Matrix (Fin 1) (Fin 1) ℂ,
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop (𝓝 1) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_signProbe
  exact ⟨Us, Vs, tendsto_inv_smul_of_unique
    (two_pos.trans_le (le_margin le_gap_signProbe_one)) (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h fun Z => eq_one_of_margin_signProbe⟩

/-- So their limit `1` has nuclear norm one and the margin `μ*`. -/
example : nuclearNorm (1 : Matrix (Fin 1) (Fin 1) ℂ) = 1 ∧
    margin signProbe (fun _ => true) (1 : Matrix (Fin 1) (Fin 1) ℂ)
      = maxMargin signProbe fun _ => true := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_signProbe
  exact margin_eq_maxMargin_of_tendsto_inv_smul
    (two_pos.trans_le (le_margin le_gap_signProbe_one)) (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h (tendsto_inv_smul_of_unique
      (two_pos.trans_le (le_margin le_gap_signProbe_one)) (fun j => by positivity)
      tendsto_one_div_add_atTop_nhds_zero_nat h fun Z => eq_one_of_margin_signProbe)

end Transformer.Grokking.FlatSpectrum.Training
