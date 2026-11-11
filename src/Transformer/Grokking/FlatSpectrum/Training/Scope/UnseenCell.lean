module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.TwoByTwo
public import Transformer.Grokking.FlatSpectrum.Training.MaxMarginLimit
public import Transformer.Grokking.FlatSpectrum.Training.SignKernel

/-!
# One unseen cell of the `2 × 2` kernel of signs: the product of largest margin

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE13, the bullet
"The unseen cells": "Which class an unseen cell of a seen row and a seen column decodes to
under the cross-entropy is not determined here", and item XE15, which determines it for the
kernel of signs `K = [[1, 1], [−1, 1]]` (`signKernel`), the kernel `ψ(a³ + a b)` for `p = 2`.

Three cells are seen, all but `(1, 1)` (`threeCells`), with the labels that `K` shows and the
code `±1`. The gaps of a product `X` are `2 Re X₀₀`, `2 Re X₀₁` and `−2 Re X₁₀`
(`gap_threeCells`). By the bound of `TwoByTwo`,

    (|X₀₀| − |X₁₁|)² + (|X₀₁| + |X₁₀|)² ≤ ‖X‖_S2² + 2 |det X| ≤ ‖X‖_S1²

(`sq_sub_add_sq_add_le`), so the margin is at most `|X₀₁| + |X₁₀| ≤ ‖X‖_S1`
(`margin_threeCells_le_nuclearNorm`). The rank-one matrix `Z₀ = ½ [[1, 1], [−1, −1]]`
(`signFill`) has nuclear norm at most one and margin one, so the largest margin of the unit
ball is `1` (`maxMargin_threeCells`), and `Z₀` is the only product of the unit ball that has
it (`eq_signFill_of_margin`): equality forces `|X₀₁| = |X₁₀| = ½`, `|X₀₀| = |X₁₁| = ½` and
`det X = 0`. `UnseenCellLimit` draws the consequence for the trained products.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix

/-- The three seen cells: all cells of a `2 × 2` kernel but `(1, 1)`. -/
def threeCells : Fin 3 → Fin 2 × Fin 2 := ![(0, 0), (0, 1), (1, 0)]

/-- The labels that the kernel of signs shows on the seen cells. -/
def threeLabels (s : Fin 3) : Bool := signLabel (threeCells s)

/-- The product of largest margin: `½ [[1, 1], [−1, −1]]`, of rank one. -/
noncomputable def signFill : Matrix (Fin 2) (Fin 2) ℂ := !![1 / 2, 1 / 2; -(1 / 2), -(1 / 2)]

/-- The gap of a sample against the other class is twice the signed real part of its cell. -/
lemma gap_threeCells (X : Matrix (Fin 2) (Fin 2) ℂ) (s : Fin 3) {c : Bool}
    (hc : c ≠ threeLabels s) :
    gap (codeProbe signCode threeCells) threeLabels X s c =
      2 * (signCode (threeLabels s)).re * (X (threeCells s).1 (threeCells s).2).re := by
  have h : (signCode c).re = -(signCode (threeLabels s)).re := by
    revert hc
    cases c <;> cases threeLabels s <;> simp [signCode]
  rw [gap, pairing_codeProbe_signCode, pairing_codeProbe_signCode, h]
  ring

/-- The margin is at most each of the three gaps. -/
lemma margin_threeCells_le (X : Matrix (Fin 2) (Fin 2) ℂ) :
    margin (codeProbe signCode threeCells) threeLabels X ≤ 2 * (X 0 0).re ∧
      margin (codeProbe signCode threeCells) threeLabels X ≤ 2 * (X 0 1).re ∧
      margin (codeProbe signCode threeCells) threeLabels X ≤ -(2 * (X 1 0).re) := by
  have h0 := margin_le_gap (A := codeProbe signCode threeCells) (y := threeLabels) X (s := 0)
    (c := false) (by decide)
  have h1 := margin_le_gap (A := codeProbe signCode threeCells) (y := threeLabels) X (s := 1)
    (c := false) (by decide)
  have h2 := margin_le_gap (A := codeProbe signCode threeCells) (y := threeLabels) X (s := 2)
    (c := true) (by decide)
  rw [gap_threeCells X _ (by decide)] at h0 h1 h2
  simp [threeLabels, threeCells, signLabel, signCode] at h0 h1 h2
  exact ⟨h0, h1, h2⟩

/-- A number below the three gaps is below the margin. -/
lemma le_margin_threeCells {X : Matrix (Fin 2) (Fin 2) ℂ} {μ : ℝ} (h0 : μ ≤ 2 * (X 0 0).re)
    (h1 : μ ≤ 2 * (X 0 1).re) (h2 : μ ≤ -(2 * (X 1 0).re)) :
    μ ≤ margin (codeProbe signCode threeCells) threeLabels X := by
  refine le_margin fun s c hc => ?_
  rw [gap_threeCells X s hc]
  fin_cases s <;> simp [threeLabels, threeCells, signLabel, signCode] <;> linarith

/-- **The margin is at most the nuclear norm.** -/
theorem margin_threeCells_le_nuclearNorm (X : Matrix (Fin 2) (Fin 2) ℂ) :
    margin (codeProbe signCode threeCells) threeLabels X ≤ nuclearNorm X := by
  obtain ⟨-, h1, h2⟩ := margin_threeCells_le X
  have hb := Complex.re_le_norm (X 0 1)
  have hc := (neg_le_abs (X 1 0).re).trans (Complex.abs_re_le_norm (X 1 0))
  have hsum : ‖X 0 1‖ + ‖X 1 0‖ ≤ nuclearNorm X := by
    have h := sq_sub_add_sq_add_le X
    have hN := nuclearNorm_nonneg X
    by_contra hlt
    nlinarith [not_le.1 hlt, sq_nonneg (‖X 0 0‖ - ‖X 1 1‖), norm_nonneg (X 0 1),
      norm_nonneg (X 1 0)]
  linarith

/-- The fill lies in the unit ball of the nuclear norm: it has rank one and `‖Z₀‖_S2 = 1`. -/
lemma nuclearNorm_signFill_le : nuclearNorm signFill ≤ 1 := by
  have h1 := nuclearNorm_sq_le_rank_mul_frobeniusSq signFill
  have h2 : (signFill.rank : ℝ) ≤ 1 := by
    have e : signFill = vecMulVec ![1, -1] ![1 / 2, 1 / 2] := by
      ext i j
      fin_cases i <;> fin_cases j <;> simp [signFill, vecMulVec_apply]
    rw [e]
    exact_mod_cast rank_vecMulVec_le _ _
  have h3 : frobeniusSq signFill = 1 := by
    rw [signFill, frobeniusSq_fin_two]
    norm_num
  rw [h3] at h1
  nlinarith [nuclearNorm_nonneg signFill]

/-- The fill has margin `1`: its three gaps are `1`. -/
lemma margin_signFill : margin (codeProbe signCode threeCells) threeLabels signFill = 1 := by
  refine le_antisymm ?_ (le_margin_threeCells ?_ ?_ ?_)
  · have h := (margin_threeCells_le signFill).1
    norm_num [signFill] at h
    exact h
  all_goals norm_num [signFill]

/-- **The largest margin of the unit ball is `1`.** -/
theorem maxMargin_threeCells : maxMargin (codeProbe signCode threeCells) threeLabels = 1 := by
  refine le_antisymm (csSup_le ⟨_, signFill, nuclearNorm_signFill_le, rfl⟩ ?_) ?_
  · rintro _ ⟨X, hX, rfl⟩
    exact (margin_threeCells_le_nuclearNorm X).trans hX
  · rw [← margin_signFill]
    exact margin_le_maxMargin nuclearNorm_signFill_le

/-- **Only one product of the unit ball has the largest margin**: `½ [[1, 1], [−1, −1]]`. -/
theorem eq_signFill_of_margin {Z : Matrix (Fin 2) (Fin 2) ℂ} (hZ : nuclearNorm Z ≤ 1)
    (hμ : margin (codeProbe signCode threeCells) threeLabels Z =
      maxMargin (codeProbe signCode threeCells) threeLabels) : Z = signFill := by
  rw [maxMargin_threeCells] at hμ
  obtain ⟨h0, h1, h2⟩ := margin_threeCells_le Z
  rw [hμ] at h0 h1 h2
  have hN := nuclearNorm_nonneg Z
  have hsq : nuclearNorm Z ^ 2 ≤ 1 := by nlinarith
  have h := frobeniusSq_add_two_mul_norm_det_le Z
  have key := sq_sub_add_sq_add_le Z
  have hF : frobeniusSq Z = ‖Z 0 0‖ ^ 2 + ‖Z 0 1‖ ^ 2 + ‖Z 1 0‖ ^ 2 + ‖Z 1 1‖ ^ 2 := by
    conv_lhs => rw [Matrix.eta_fin_two Z]
    exact frobeniusSq_fin_two _ _ _ _
  have hA : 1 / 2 ≤ ‖Z 0 0‖ := by linarith [Complex.re_le_norm (Z 0 0)]
  have hB : 1 / 2 ≤ ‖Z 0 1‖ := by linarith [Complex.re_le_norm (Z 0 1)]
  have hC : 1 / 2 ≤ ‖Z 1 0‖ := by
    linarith [(neg_le_abs (Z 1 0).re).trans (Complex.abs_re_le_norm (Z 1 0))]
  have hBC : ‖Z 0 1‖ + ‖Z 1 0‖ ≤ 1 := by
    by_contra hlt
    nlinarith [not_le.1 hlt, sq_nonneg (‖Z 0 0‖ - ‖Z 1 1‖)]
  have hB' : ‖Z 0 1‖ = 1 / 2 := by linarith
  have hC' : ‖Z 1 0‖ = 1 / 2 := by linarith
  have hAD : ‖Z 0 0‖ = ‖Z 1 1‖ := by
    have h5 : (‖Z 0 0‖ - ‖Z 1 1‖) ^ 2 ≤ 0 := by nlinarith
    nlinarith [sq_nonneg (‖Z 0 0‖ - ‖Z 1 1‖)]
  have hdn := norm_nonneg Z.det
  have hA' : ‖Z 0 0‖ = 1 / 2 := by nlinarith
  have hdet : Z.det = 0 := norm_le_zero_iff.1 (by nlinarith)
  have e00 : Z 0 0 = 1 / 2 := by
    simpa using eq_ofReal_of_norm_le_of_le_re hA'.le (by linarith)
  have e01 : Z 0 1 = 1 / 2 := by
    simpa using eq_ofReal_of_norm_le_of_le_re hB'.le (by linarith)
  have e10 : Z 1 0 = -(1 / 2) := by
    have h6 : -Z 1 0 = ((1 / 2 : ℝ) : ℂ) :=
      eq_ofReal_of_norm_le_of_le_re (by rw [norm_neg]; exact hC'.le)
        (by rw [Complex.neg_re]; linarith)
    rw [← neg_eq_iff_eq_neg, h6]
    simp
  have e11 : Z 1 1 = -(1 / 2) := by
    rw [Matrix.det_fin_two, e00, e01, e10] at hdet
    linear_combination 2 * hdet
  rw [Matrix.eta_fin_two Z, e00, e01, e10, e11, signFill]

/-! The witnesses: the kernel of signs on its seen cells, and the fill. -/

example := gap_threeCells signKernel 0 (c := false) (by decide)

example := le_margin_threeCells (X := signFill) (μ := 1) (by norm_num [signFill])
  (by norm_num [signFill]) (by norm_num [signFill])

/-- The fill meets the hypotheses of `eq_signFill_of_margin`. -/
example := eq_signFill_of_margin nuclearNorm_signFill_le
  (margin_signFill.trans maxMargin_threeCells.symm)

end Transformer.Grokking.FlatSpectrum.Training.Scope
