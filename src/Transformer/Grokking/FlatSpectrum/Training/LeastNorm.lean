module

public import Transformer.Grokking.FlatSpectrum.Training.MarginDirection

/-!
# The margin problem in its second form: the least nuclear norm at margin one

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE6 and XE10.
Item XE6 states the margin problem as the largest margin `μ*` of the unit ball of the nuclear
norm. Its second form is the least nuclear norm of a product of margin at least one. Let some
product have a positive margin, so that `μ* > 0` (`maxMargin_pos`).

* A product of margin at least one has `‖X‖_S1 ≥ 1 / μ*` (`inv_maxMargin_le_nuclearNorm`), by
  homogeneity, and the bound is attained (`isLeast_nuclearNorm_of_one_le_margin`).
* The two forms have the same solutions up to the scale `μ*`: `Z` has the margin `μ*` in the
  unit ball exactly when `Z / μ*` has margin at least one and the nuclear norm `1 / μ*`
  (`margin_eq_maxMargin_iff_inv_smul`).
* The trained fits have `‖X_j‖_S1 / μ(X_j) → 1 / μ*` (`tendsto_nuclearNorm_div_margin`). Where
  their direction converges to `Z`, scaled to margin one they converge to `Z / μ*`
  (`tendsto_inv_margin_smul`); so they do when the solution is unique
  (`tendsto_inv_margin_smul_of_unique`). Item XE11 gives the limit with every cell seen.

The witness is `signProbe`, where the least norm at margin one is `1 / 2`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [Fintype S] [Fintype C]
  [DecidableEq n] [Nonempty S] [Nontrivial C] {A : S → C → Matrix m n ℂ} {y : S → C}
  {X X₀ Z : Matrix m n ℂ} {lams : ℕ → ℝ} {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ}

omit [Fintype k] in
/-- The largest margin is positive when some product has a positive margin. -/
lemma maxMargin_pos (h : 0 < margin A y X) : 0 < maxMargin A y :=
  (div_pos h (nuclearNorm_pos_of_margin_pos h)).trans_le (margin_div_le_maxMargin X)

omit [Fintype k] in
/-- **A product of margin at least one has `‖X‖_S1 ≥ 1 / μ*`** (`review/flat_spectrum.md`,
item XE6, the second form of the margin problem). -/
theorem inv_maxMargin_le_nuclearNorm (h : 1 ≤ margin A y X) :
    (maxMargin A y)⁻¹ ≤ nuclearNorm X := by
  have hμ : 0 < margin A y X := one_pos.trans_le h
  have h1 := (div_le_iff₀ (nuclearNorm_pos_of_margin_pos hμ)).mp
    (margin_div_le_maxMargin (A := A) (y := y) X)
  rw [inv_le_iff_one_le_mul₀' (maxMargin_pos hμ)]
  linarith

omit [Fintype k] in
/-- **The two forms of the margin problem have the same solutions up to scale**
(`review/flat_spectrum.md`, item XE6): for `μ* > 0`, a product `Z` of the unit ball has the
margin `μ*` exactly when `Z / μ*` has margin at least one and the nuclear norm `1 / μ*`. -/
theorem margin_eq_maxMargin_iff_inv_smul (hM : 0 < maxMargin A y) (Z : Matrix m n ℂ) :
    nuclearNorm Z ≤ 1 ∧ margin A y Z = maxMargin A y ↔
      1 ≤ margin A y ((maxMargin A y)⁻¹ • Z) ∧
        nuclearNorm ((maxMargin A y)⁻¹ • Z) = (maxMargin A y)⁻¹ := by
  have hi : 0 ≤ (maxMargin A y)⁻¹ := inv_nonneg.mpr hM.le
  constructor
  · rintro ⟨h1, h2⟩
    have h3 : margin A y ((maxMargin A y)⁻¹ • Z) = 1 := by
      rw [margin_smul hi, h2, inv_mul_cancel₀ hM.ne']
    refine ⟨h3.ge, le_antisymm ?_ (inv_maxMargin_le_nuclearNorm h3.ge)⟩
    rw [nuclearNorm_real_smul hi]
    exact mul_le_of_le_one_right hi h1
  · rw [margin_smul hi, nuclearNorm_real_smul hi]
    rintro ⟨h1, h2⟩
    have hZ : nuclearNorm Z = 1 :=
      mul_left_cancel₀ (inv_ne_zero hM.ne') (h2.trans (mul_one _).symm)
    refine ⟨hZ.le, le_antisymm (margin_le_maxMargin hZ.le) ?_⟩
    have h3 := mul_le_mul_of_nonneg_left h1 hM.le
    rwa [mul_one, mul_inv_cancel_left₀ hM.ne'] at h3

/-- **The norm of the trained fits for their margin tends to the least norm at margin one**
(`review/flat_spectrum.md`, item XE6): `‖X_j‖_S1 / μ(X_j) → 1 / μ*` as `λ_j → 0⁺`, if some
product has a positive margin. -/
theorem tendsto_nuclearNorm_div_margin (hsep : 0 < margin A y X₀) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => nuclearNorm (Us j * (Vs j)ᴴ) / margin A y (Us j * (Vs j)ᴴ)) atTop
      (𝓝 (maxMargin A y)⁻¹) :=
  ((tendsto_margin_div_nuclearNorm hsep hpos hlam hmin).inv₀ (maxMargin_pos hsep).ne').congr
    fun _ => inv_div _ _

/-- **Scaled to margin one, the trained fits converge where their direction does**
(`review/flat_spectrum.md`, items XE6 and XE10): if `X_j / ‖X_j‖_S1 → Z`, then
`X_j / μ(X_j) → Z / μ*`. -/
theorem tendsto_inv_margin_smul (hsep : 0 < margin A y X₀) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j))
    (hZ : Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 Z)) :
    Tendsto (fun j => (margin A y (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((maxMargin A y)⁻¹ • Z)) := by
  refine ((tendsto_nuclearNorm_div_margin hsep hpos hlam hmin).smul hZ).congr fun j => ?_
  rcases (nuclearNorm_nonneg (Us j * (Vs j)ᴴ)).eq_or_lt with h0 | h0
  · simp only [eq_zero_of_nuclearNorm_eq_zero h0.symm, smul_zero]
  · rw [smul_smul, div_eq_mul_inv, mul_right_comm, mul_inv_cancel₀ h0.ne', one_mul]

variable [DecidableEq m]

omit [Fintype k] in
/-- **The least nuclear norm at margin one is `1 / μ*`** (`review/flat_spectrum.md`, item
XE6, the second form of the margin problem), when some product has a positive margin. -/
theorem isLeast_nuclearNorm_of_one_le_margin (hsep : 0 < margin A y X₀) :
    IsLeast (nuclearNorm '' {X : Matrix m n ℂ | 1 ≤ margin A y X}) (maxMargin A y)⁻¹ := by
  obtain ⟨Z, hZ⟩ := exists_margin_eq_maxMargin A y
  obtain ⟨h1, h2⟩ := (margin_eq_maxMargin_iff_inv_smul (maxMargin_pos hsep) Z).1 hZ
  refine ⟨⟨_, h1, h2⟩, ?_⟩
  rintro _ ⟨X, hX, rfl⟩
  exact inv_maxMargin_le_nuclearNorm hX

/-- **With a unique solution, the trained fits scaled to margin one converge to it**
(`review/flat_spectrum.md`, items XE6 and XE10): if `Z₀` is the only product of the unit ball
with the margin `μ*`, then `X_j / μ(X_j) → Z₀ / μ*`. -/
theorem tendsto_inv_margin_smul_of_unique (hsep : 0 < margin A y X₀) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) {Z₀ : Matrix m n ℂ}
    (huniq : ∀ Z, nuclearNorm Z ≤ 1 → margin A y Z = maxMargin A y → Z = Z₀) :
    Tendsto (fun j => (margin A y (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((maxMargin A y)⁻¹ • Z₀)) :=
  tendsto_inv_margin_smul hsep hpos hlam hmin
    (tendsto_inv_smul_of_unique hsep hpos hlam hmin huniq)

/-! The witness: `signProbe`, one sample of a `1 × 1` product read with the code `±1`. Its
largest margin is `2`, and the product `1/2` has margin one. -/

lemma margin_signProbe_one_pos :
    0 < margin signProbe (fun _ => true) (1 : Matrix (Fin 1) (Fin 1) ℂ) :=
  two_pos.trans_le (le_margin le_gap_signProbe_one)

example := maxMargin_pos margin_signProbe_one_pos

example := inv_maxMargin_le_nuclearNorm (A := signProbe) (y := fun _ => true) (X := 1)
  (one_le_two.trans (le_margin le_gap_signProbe_one))

example := margin_eq_maxMargin_iff_inv_smul (maxMargin_pos margin_signProbe_one_pos) 1

/-- The least norm of a `1 × 1` product of margin one under the code `±1` is `1/2`. -/
example : IsLeast (nuclearNorm '' {X : Matrix (Fin 1) (Fin 1) ℂ |
    1 ≤ margin signProbe (fun _ => true) X}) 2⁻¹ := by
  have h := isLeast_nuclearNorm_of_one_le_margin margin_signProbe_one_pos
  rwa [maxMargin_signProbe] at h

/-- Trained fits of the witness, scaled to margin one, converge to `1/2`. -/
example : ∃ Us Vs : ℕ → Matrix (Fin 1) (Fin 1) ℂ,
    Tendsto (fun j => nuclearNorm (Us j * (Vs j)ᴴ)
        / margin signProbe (fun _ => true) (Us j * (Vs j)ᴴ)) atTop (𝓝 2⁻¹) ∧
      Tendsto (fun j => (margin signProbe (fun _ => true) (Us j * (Vs j)ᴴ))⁻¹
        • (Us j * (Vs j)ᴴ)) atTop (𝓝 ((2 : ℝ)⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℂ))) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_signProbe
  have h1 := tendsto_nuclearNorm_div_margin margin_signProbe_one_pos (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h
  have h2 := tendsto_inv_margin_smul_of_unique margin_signProbe_one_pos
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h
    fun Z => eq_one_of_margin_signProbe
  rw [maxMargin_signProbe] at h1 h2
  exact ⟨Us, Vs, h1, h2⟩

/-- The same limit from the convergence of the direction. -/
example : ∃ Us Vs : ℕ → Matrix (Fin 1) (Fin 1) ℂ,
    Tendsto (fun j => (margin signProbe (fun _ => true) (Us j * (Vs j)ᴴ))⁻¹
      • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((maxMargin signProbe fun _ => true)⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℂ))) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_signProbe
  exact ⟨Us, Vs, tendsto_inv_margin_smul margin_signProbe_one_pos (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h (tendsto_inv_smul_of_unique
      margin_signProbe_one_pos (fun j => by positivity)
      tendsto_one_div_add_atTop_nhds_zero_nat h fun Z => eq_one_of_margin_signProbe)⟩

end Transformer.Grokking.FlatSpectrum.Training
