module

public import Transformer.Grokking.FlatSpectrum.Training.SeenMargin

/-!
# The largest logit belongs to the nearest codeword

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE8.
The note decodes a cell `x = X_ij` to the class of the nearest codeword. For a unit `a`,
`|x − a|² = |x|² + 1 − 2 Re (conj a x)` (`norm_sub_sq_of_norm_eq_one`), so of two unit numbers
the one with the larger reading `Re (conj a x)` is the nearer (`re_conj_mul_le_iff`,
`re_conj_mul_lt_iff`). For the probes of a unimodular answer code:

* the logit of the class `t` is at least that of `c` exactly when `φ t` is at least as near to
  the cell as `φ c` (`pairing_codeProbe_le_iff`): the softmax of item XE8 and the decoder of
  the note choose the same class;
* a gap is positive exactly when the codeword of the label is nearer than that of the other
  class (`gap_codeProbe_pos_iff`), so a product with a positive margin is decoded to the label
  at every sample (`norm_sub_lt_of_margin_pos`);
* as the weight decay vanishes on samples that some product classifies, the trained fits are
  decoded to the label at every sample in the end (`eventually_norm_sub_lt`).

The witness is the `1 × 2` kernel `(1, 1)` with the code `±1` on its first cell.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology
open scoped ComplexConjugate

/-- `|x − a|² = |x|² + 1 − 2 Re (conj a x)` for a unit `a`. -/
lemma norm_sub_sq_of_norm_eq_one {a : ℂ} (ha : ‖a‖ = 1) (x : ℂ) :
    ‖x - a‖ ^ 2 = ‖x‖ ^ 2 + 1 - 2 * (conj a * x).re := by
  rw [← Complex.normSq_eq_norm_sq, Complex.normSq_sub, Complex.normSq_eq_norm_sq,
    Complex.normSq_eq_norm_sq, ha, mul_comm x]
  ring

/-- Of two unit numbers the one with the larger reading `Re (conj a x)` is the nearer to
`x`. -/
theorem re_conj_mul_le_iff {a b : ℂ} (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (x : ℂ) :
    (conj b * x).re ≤ (conj a * x).re ↔ ‖x - a‖ ≤ ‖x - b‖ := by
  rw [← pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero,
    norm_sub_sq_of_norm_eq_one ha, norm_sub_sq_of_norm_eq_one hb]
  constructor <;> intro h <;> linarith

/-- The strict form of `re_conj_mul_le_iff`. -/
theorem re_conj_mul_lt_iff {a b : ℂ} (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (x : ℂ) :
    (conj b * x).re < (conj a * x).re ↔ ‖x - a‖ < ‖x - b‖ := by
  rw [← pow_lt_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero,
    norm_sub_sq_of_norm_eq_one ha, norm_sub_sq_of_norm_eq_one hb]
  constructor <;> intro h <;> linarith

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m]
  [DecidableEq n] {φ : C → ℂ} {cell : S → m × n} {y : S → C} {X : Matrix m n ℂ}

omit [Fintype k] in
/-- **The class of the largest logit is the class of the nearest codeword**
(`review/flat_spectrum.md`, item XE8): for a unimodular code the logit of `t` is at least that
of `c` exactly when `φ t` is at least as near to the cell as `φ c`. -/
theorem pairing_codeProbe_le_iff (hφ : ∀ c, ‖φ c‖ = 1) (X : Matrix m n ℂ) (s : S) (t c : C) :
    pairing (codeProbe φ cell s c) X ≤ pairing (codeProbe φ cell s t) X ↔
      ‖X (cell s).1 (cell s).2 - φ t‖ ≤ ‖X (cell s).1 (cell s).2 - φ c‖ := by
  rw [pairing_codeProbe, pairing_codeProbe]
  exact re_conj_mul_le_iff (hφ t) (hφ c) _

omit [Fintype k] in
/-- A gap is positive exactly when the codeword of the label is nearer to the cell than that
of the other class. -/
theorem gap_codeProbe_pos_iff (hφ : ∀ c, ‖φ c‖ = 1) (X : Matrix m n ℂ) (s : S) (c : C) :
    0 < gap (codeProbe φ cell) y X s c ↔
      ‖X (cell s).1 (cell s).2 - φ (y s)‖ < ‖X (cell s).1 (cell s).2 - φ c‖ := by
  rw [gap, sub_pos, pairing_codeProbe, pairing_codeProbe]
  exact re_conj_mul_lt_iff (hφ (y s)) (hφ c) _

variable [Fintype S] [Fintype C]

omit [Fintype k] in
/-- **A product with a positive margin is decoded to the label at every sample**
(`review/flat_spectrum.md`, items XE4 and XE8): the codeword of the label is the nearest. -/
theorem norm_sub_lt_of_margin_pos (hφ : ∀ c, ‖φ c‖ = 1)
    (h : 0 < margin (codeProbe φ cell) y X) {s : S} {c : C} (hc : c ≠ y s) :
    ‖X (cell s).1 (cell s).2 - φ (y s)‖ < ‖X (cell s).1 (cell s).2 - φ c‖ :=
  (gap_codeProbe_pos_iff hφ X s c).1 (h.trans_le (margin_le_gap X hc))

/-- **With a vanishing weight decay the trained fits are decoded to the labels**
(`review/flat_spectrum.md`, items XE6 and XE8): if some product has a positive margin and
`λ_j → 0⁺`, in the end the codeword of the label is the nearest to the trained product at
every sample. -/
theorem eventually_norm_sub_lt [Nonempty S] [Nontrivial C] (hφ : ∀ c, ‖φ c‖ = 1)
    {X₀ : Matrix m n ℂ} (hsep : 0 < margin (codeProbe φ cell) y X₀) {lams : ℕ → ℝ}
    {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ} (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss (codeProbe φ cell) y) (lams j) (Us j) (Vs j)) :
    ∀ᶠ j in atTop, ∀ s c, c ≠ y s →
      ‖(Us j * (Vs j)ᴴ) (cell s).1 (cell s).2 - φ (y s)‖
        < ‖(Us j * (Vs j)ᴴ) (cell s).1 (cell s).2 - φ c‖ := by
  have hB := nuclearNorm_pos_of_margin_pos hsep
  have h1 : 0 < margin (codeProbe φ cell) y ((nuclearNorm X₀)⁻¹ • X₀) := by
    rw [margin_smul (inv_nonneg.mpr hB.le)]
    exact mul_pos (inv_pos.mpr hB) hsep
  filter_upwards [eventually_lt_margin_div (X' := (nuclearNorm X₀)⁻¹ • X₀)
    (by rw [nuclearNorm_real_smul (inv_nonneg.mpr hB.le), inv_mul_cancel₀ hB.ne']) le_rfl h1
    hpos hlam hmin] with j hj s c hc
  refine norm_sub_lt_of_margin_pos hφ ?_ hc
  by_contra h0
  exact absurd hj (not_lt.mpr (div_nonpos_of_nonpos_of_nonneg (not_lt.mp h0)
    (nuclearNorm_nonneg _)))

/-! The witness: the `1 × 2` kernel `(1, 1)` with the code `±1` on its first cell, labeled
`true`. -/

local notation "cell₀" => fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 2))

example := norm_sub_sq_of_norm_eq_one (a := 1) norm_one 1

example := re_conj_mul_le_iff (a := 1) (b := -1) norm_one (by simp) 1

example := re_conj_mul_lt_iff (a := 1) (b := -1) norm_one (by simp) 1

example := pairing_codeProbe_le_iff (cell := cell₀) norm_signCode
  (Matrix.of ![![(1 : ℂ), 1]]) 0 true false

example := gap_codeProbe_pos_iff (cell := cell₀) (y := fun _ => true) norm_signCode
  (Matrix.of ![![(1 : ℂ), 1]]) 0 false

/-- The kernel `(1, 1)` is decoded to the label of its seen cell. -/
example := norm_sub_lt_of_margin_pos (cell := cell₀) (y := fun _ => true) norm_signCode
  (margin_codeProbe_pos (K := Matrix.of ![![(1 : ℂ), 1]]) norm_signCode injective_signCode
    fun _ => by simp [signCode]) (s := 0) (c := false) (by simp)

/-- Trained fits of `(1, 1)` from its first cell are decoded to the label in the end. -/
example : ∃ (Us : ℕ → Matrix (Fin 1) (Fin 1) ℂ) (Vs : ℕ → Matrix (Fin 2) (Fin 1) ℂ),
    ∀ᶠ j in atTop, ∀ s c, c ≠ true →
      ‖(Us j * (Vs j)ᴴ) (cell₀ s).1 (cell₀ s).2 - signCode true‖
        < ‖(Us j * (Vs j)ᴴ) (cell₀ s).1 (cell₀ s).2 - signCode c‖ := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_ceLoss (k := Fin 1)
    (codeProbe signCode cell₀) (fun _ => true) (lams := fun j => 1 / ((j : ℝ) + 1))
    (fun j => by positivity) (Or.inl le_rfl)
  exact ⟨Us, Vs, eventually_norm_sub_lt norm_signCode
    (margin_codeProbe_pos (K := Matrix.of ![![(1 : ℂ), 1]]) norm_signCode injective_signCode
      fun _ => by simp [signCode]) (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h⟩

end Transformer.Grokking.FlatSpectrum.Training
