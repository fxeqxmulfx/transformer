module

public import Transformer.Grokking.FlatSpectrum.Training.SeenDirection

/-!
# A code that is not mirrored: the direction of the trained fits is not the kernel

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE11 and XE13.
Item XE11 takes a mirrored answer code. Without it the conclusion fails, already for one cell.
Take the code `(1, i)` of two classes (`quarterCode`): it is unimodular and injective, and it
is mirrored at no gap (`not_isMirroredCode_quarterCode`). Let one sample read a `1 × 1` product
`x` with the label of the codeword `1`. The kernel `1` is flat and shows the label, and its
cell is seen. The margin of a product is `Re x − Im x` (`margin_quarterCode`), which is `1` at
the kernel and at most `√2` in the unit disk, with equality only at `(1 − i) / √2`
(`maxMargin_quarterCode`, `eq_quarterLimit_of_margin`). So as the weight decay vanishes

* the trained fits converge in direction to `(1 − i) / √2` (`tendsto_inv_smul_quarterCode`);
* they do not converge in direction to the kernel (`not_tendsto_inv_smul_quarterCode`).

The direction turns away from the other codeword. It is still decoded to the label.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

/-- The code `(1, i)` of the two classes. -/
def quarterCode (c : Bool) : ℂ := if c then 1 else Complex.I

lemma norm_quarterCode (c : Bool) : ‖quarterCode c‖ = 1 := by
  cases c <;> simp [quarterCode]

lemma injective_quarterCode : Function.Injective quarterCode := by
  intro a b h
  cases a <;> cases b <;> simp_all [quarterCode, Complex.ext_iff]

/-- The code `(1, i)` is mirrored at no gap: `2 i` is not a real multiple of `1`. -/
lemma not_isMirroredCode_quarterCode (g : ℝ) : ¬ IsMirroredCode quarterCode g := by
  rintro ⟨-, -, h⟩
  obtain ⟨c, c', hc, hc', e⟩ := h true
  have h0 : c = false := by cases c <;> simp_all
  have h0' : c' = false := by cases c' <;> simp_all
  rw [h0, h0'] at e
  have h2 := congrArg Complex.im e
  simp [quarterCode] at h2

local notation "cell₀" => fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 1))

/-- The gap of the witness is `Re x − Im x` for the entry `x` of the product. -/
lemma gap_quarterCode (X : Matrix (Fin 1) (Fin 1) ℂ) (s : Fin 1) :
    gap (codeProbe quarterCode cell₀) (fun _ => true) X s false
      = (X 0 0).re - (X 0 0).im := by
  simp [gap, pairing_codeProbe, quarterCode]

/-- The margin of the witness is its one gap. -/
lemma margin_quarterCode (X : Matrix (Fin 1) (Fin 1) ℂ) :
    margin (codeProbe quarterCode cell₀) (fun _ => true) X = (X 0 0).re - (X 0 0).im := by
  refine le_antisymm ?_ (le_margin fun s c hc => ?_)
  · rw [← gap_quarterCode X 0]
    exact margin_le_gap X (by simp)
  · cases c with
    | false => exact (gap_quarterCode X s).ge
    | true => exact absurd rfl hc

/-- The product `(1 − i) / √2`, of modulus one. -/
noncomputable def quarterLimit : Matrix (Fin 1) (Fin 1) ℂ :=
  ((√2)⁻¹ : ℝ) • Matrix.of ![![1 - Complex.I]]

lemma quarterLimit_apply : quarterLimit 0 0 = (((√2)⁻¹ : ℝ) : ℂ) * (1 - Complex.I) := by
  simp [quarterLimit]

lemma nuclearNorm_quarterLimit_le : nuclearNorm quarterLimit ≤ 1 := by
  refine (nuclearNorm_le_of_eq_mul quarterLimit 1 (by simp)).trans_eq ?_
  have h1 : frobeniusSq (Matrix.of ![![1 - Complex.I]] : Matrix (Fin 1) (Fin 1) ℂ) = 2 := by
    simp [frobeniusSq, Complex.sq_norm, Complex.normSq_apply]
    norm_num
  have h2 : frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) = 1 := by simp [frobeniusSq]
  rw [quarterLimit, frobeniusSq_smul, h1, h2, inv_pow, Real.sq_sqrt zero_le_two]
  norm_num

lemma margin_quarterLimit :
    margin (codeProbe quarterCode cell₀) (fun _ => true) quarterLimit = √2 := by
  have h := Real.sq_sqrt zero_le_two
  have h0 : (0 : ℝ) < √2 := Real.sqrt_pos.mpr two_pos
  rw [margin_quarterCode, quarterLimit_apply, Complex.re_ofReal_mul, Complex.im_ofReal_mul]
  simp only [Complex.sub_re, Complex.one_re, Complex.I_re, Complex.sub_im, Complex.one_im,
    Complex.I_im]
  field_simp
  nlinarith

/-- In the unit ball the margin of the witness is at most `√2`. -/
lemma margin_quarterCode_le {Z : Matrix (Fin 1) (Fin 1) ℂ} (hZ : nuclearNorm Z ≤ 1) :
    margin (codeProbe quarterCode cell₀) (fun _ => true) Z ≤ √2 := by
  have h4 := (norm_apply_le_nuclearNorm Z 0 0).trans hZ
  have h5 := Complex.sq_norm (Z 0 0)
  rw [Complex.normSq_apply] at h5
  have h6 : ‖Z 0 0‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg (Z 0 0)]
  rw [margin_quarterCode]
  exact (le_abs_self _).trans (Real.abs_le_sqrt (by
    nlinarith [sq_nonneg ((Z 0 0).re + (Z 0 0).im)]))

/-- The largest margin of the witness is `√2`. -/
lemma maxMargin_quarterCode :
    maxMargin (codeProbe quarterCode cell₀) (fun _ => true) = √2 := by
  refine le_antisymm (csSup_le ⟨_, ⟨0, nuclearNorm_zero.trans_le zero_le_one, rfl⟩⟩ ?_)
    (margin_quarterLimit.ge.trans (margin_le_maxMargin nuclearNorm_quarterLimit_le))
  rintro _ ⟨X, hX, rfl⟩
  exact margin_quarterCode_le hX

/-- Only `(1 − i) / √2` has the largest margin in the unit ball. -/
lemma eq_quarterLimit_of_margin {Z : Matrix (Fin 1) (Fin 1) ℂ} (h1 : nuclearNorm Z ≤ 1)
    (h2 : margin (codeProbe quarterCode cell₀) (fun _ => true) Z
      = maxMargin (codeProbe quarterCode cell₀) fun _ => true) : Z = quarterLimit := by
  rw [maxMargin_quarterCode, margin_quarterCode] at h2
  have h := Real.sq_sqrt zero_le_two
  have h0 : (0 : ℝ) < √2 := Real.sqrt_pos.mpr two_pos
  have h4 := (norm_apply_le_nuclearNorm Z 0 0).trans h1
  have h5 := Complex.sq_norm (Z 0 0)
  rw [Complex.normSq_apply] at h5
  have h6 : ‖Z 0 0‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg (Z 0 0)]
  have hsum : (Z 0 0).re + (Z 0 0).im = 0 := by
    nlinarith [sq_nonneg ((Z 0 0).re + (Z 0 0).im)]
  ext i j
  rw [Subsingleton.elim i 0, Subsingleton.elim j 0, quarterLimit_apply]
  refine Complex.ext ?_ ?_
  · rw [Complex.re_ofReal_mul]
    simp only [Complex.sub_re, Complex.one_re, Complex.I_re, sub_zero, mul_one]
    field_simp
    nlinarith
  · rw [Complex.im_ofReal_mul]
    simp only [Complex.sub_im, Complex.one_im, Complex.I_im, zero_sub, mul_neg, mul_one]
    field_simp
    nlinarith

variable {k : Type*} [Fintype k] {lams : ℕ → ℝ} {Us Vs : ℕ → Matrix (Fin 1) k ℂ}

/-- **With the code `(1, i)` the trained fits converge in direction to `(1 − i) / √2`**
(`review/flat_spectrum.md`, items XE10 and XE13). -/
theorem tendsto_inv_smul_quarterCode (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss (codeProbe quarterCode cell₀) fun _ => true) (lams j)
      (Us j) (Vs j)) :
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 quarterLimit) :=
  tendsto_inv_smul_of_unique (X₀ := quarterLimit)
    (margin_quarterLimit.symm ▸ Real.sqrt_pos.mpr two_pos) hpos hlam hmin
    fun _ => eq_quarterLimit_of_margin

/-- **For a code that is not mirrored the trained fits need not converge in direction to the
flat kernel that shows the labels**, with every cell seen (`review/flat_spectrum.md`, items
XE11 and XE13): the `1 × 1` kernel `1` with the code `(1, i)`. -/
theorem not_tendsto_inv_smul_quarterCode (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss (codeProbe quarterCode cell₀) fun _ => true) (lams j)
      (Us j) (Vs j)) :
    ¬ Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm (1 : Matrix (Fin 1) (Fin 1) ℂ))⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℂ))) := by
  intro hlim
  have e := tendsto_nhds_unique hlim (tendsto_inv_smul_quarterCode hpos hlam hmin)
  have h := congrArg (margin (codeProbe quarterCode cell₀) fun _ => true) e
  rw [nuclearNorm_one_fin_one, inv_one, one_smul, margin_quarterLimit, margin_quarterCode,
    Matrix.one_apply_eq, Complex.one_re, Complex.one_im, sub_zero] at h
  have h2 := Real.sq_sqrt zero_le_two
  rw [← h] at h2
  norm_num at h2

/-! The witness meets every other hypothesis of item XE11: the kernel `1` is flat, shows the
label, has no zero entry, and its cell is seen. Trained fits exist along `1 / (j + 1)`. -/

example : IsFlat (1 : Matrix (Fin 1) (Fin 1) ℂ) 1 ∧
    (∀ s : Fin 1, (1 : Matrix (Fin 1) (Fin 1) ℂ) (cell₀ s).1 (cell₀ s).2 = quarterCode true) ∧
      Function.Surjective cell₀ :=
  ⟨isFlat_one, fun _ => by simp [quarterCode], fun _ => ⟨0, Subsingleton.elim _ _⟩⟩

example := margin_quarterCode_le nuclearNorm_quarterLimit_le

example := eq_quarterLimit_of_margin nuclearNorm_quarterLimit_le
  (margin_quarterLimit.trans maxMargin_quarterCode.symm)

/-- Trained fits of the witness converge in direction to `(1 − i) / √2`, not to the kernel. -/
example : ∃ Us Vs : ℕ → Matrix (Fin 1) (Fin 1) ℂ,
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
        (𝓝 quarterLimit) ∧
      ¬ Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
        (𝓝 ((nuclearNorm (1 : Matrix (Fin 1) (Fin 1) ℂ))⁻¹
          • (1 : Matrix (Fin 1) (Fin 1) ℂ))) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_ceLoss (k := Fin 1)
    (codeProbe quarterCode cell₀) (fun _ => true) (lams := fun j => 1 / ((j : ℝ) + 1))
    (fun j => by positivity) (Or.inl le_rfl)
  exact ⟨Us, Vs, tendsto_inv_smul_quarterCode (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h, not_tendsto_inv_smul_quarterCode
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h⟩

end Transformer.Grokking.FlatSpectrum.Training
