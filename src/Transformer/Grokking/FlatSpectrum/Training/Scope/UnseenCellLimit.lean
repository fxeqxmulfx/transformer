module

public import Transformer.Grokking.FlatSpectrum.Decoding
public import Transformer.Grokking.FlatSpectrum.Training.NuclearMinDirection
public import Transformer.Grokking.FlatSpectrum.Training.Scope.UnseenCell

/-!
# One unseen cell of the `2 × 2` kernel of signs is decoded to the wrong class

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE13, the bullet
"The unseen cells", and item XE15.

The kernel of signs `K = [[1, 1], [−1, 1]]` is the kernel `ψ(a³ + a b)` for `p = 2`
(`cubicKernel_two`), and it is flat. Its cells `(0, 0)`, `(0, 1)` and `(1, 0)` are seen with
the labels it shows and the code `±1`; the cell `(1, 1)`, where `K` shows the class of `+1`,
is not. By `UnseenCell` one product of the unit ball of the nuclear norm has the largest
margin, `Z₀ = ½ [[1, 1], [−1, −1]]`. So as the weight decay vanishes

    X_j / ‖X_j‖_S1 → ½ [[1, 1], [−1, −1]]

for every sequence of trained products (`tendsto_inv_smul_threeCells`, by XE10). The unseen
cell of the limit is `−½`: in the end `Re X_j(1, 1) < 0` (`eventually_re_unseen_neg`), and
the codeword nearest to the unseen cell is that of the class of `−1`, not of the class that
the kernel shows there (`eventually_unseen_wrong`). The rows of `Z₀` are opposite, as the
seen cells of the second row ask, and a matrix of rank one fills the fourth cell with the
sign `−`; the kernel has the sign `+` there, which is what makes it flat.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

/-- **The kernel of signs is the kernel `ψ(a³ + a b)` for `p = 2`.** -/
theorem cubicKernel_two : (cubicKernel 2 : Matrix (Fin 2) (Fin 2) ℂ) = signKernel := by
  have h0 : ZMod.stdAddChar (0 : ZMod 2) = 1 := AddChar.map_zero_eq_one _
  have h1 : ZMod.stdAddChar (1 : ZMod 2) = -1 := by
    rw [stdAddChar_eq_exp]
    have e : ((2 * Real.pi * ((1 : ZMod 2).val : ℝ) / ((2 : ℕ) : ℝ) : ℝ) : ℂ) * Complex.I =
        Real.pi * Complex.I := by
      rw [ZMod.val_one]
      push_cast
      ring
    rw [e, Complex.exp_pi_mul_I]
  ext i j
  fin_cases i <;> fin_cases j
  · show ZMod.stdAddChar ((0 : ZMod 2) ^ 3 + 0 * 0) = signKernel 0 0
    rw [show (0 : ZMod 2) ^ 3 + 0 * 0 = 0 by decide, h0]
    simp [signKernel]
  · show ZMod.stdAddChar ((0 : ZMod 2) ^ 3 + 0 * 1) = signKernel 0 1
    rw [show (0 : ZMod 2) ^ 3 + 0 * 1 = 0 by decide, h0]
    simp [signKernel]
  · show ZMod.stdAddChar ((1 : ZMod 2) ^ 3 + 1 * 0) = signKernel 1 0
    rw [show (1 : ZMod 2) ^ 3 + 1 * 0 = 1 by decide, h1]
    simp [signKernel]
  · show ZMod.stdAddChar ((1 : ZMod 2) ^ 3 + 1 * 1) = signKernel 1 1
    rw [show (1 : ZMod 2) ^ 3 + 1 * 1 = 0 by decide, h0]
    simp [signKernel]

variable {lams : ℕ → ℝ} {X : ℕ → Matrix (Fin 2) (Fin 2) ℂ}

/-- **The trained products converge in direction to the rank-one fill**
(`review/flat_spectrum.md`, item XE15): `X_j / ‖X_j‖_S1 → ½ [[1, 1], [−1, −1]]` as the weight
decay vanishes, for the three seen cells of the kernel of signs. -/
theorem tendsto_inv_smul_threeCells (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss (codeProbe signCode threeCells) threeLabels) (lams j)
      (X j)) :
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 signFill) :=
  tendsto_inv_smul_of_unique_of_isNuclearMin (X₀ := signFill)
    (by rw [margin_signFill]; exact one_pos) hpos hlam hX
    fun _ hZ hμ => eq_signFill_of_margin hZ hμ

/-- **The unseen cell ends with a negative real part.** -/
theorem eventually_re_unseen_neg (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss (codeProbe signCode threeCells) threeLabels) (lams j)
      (X j)) :
    ∀ᶠ j in atTop, (X j 1 1).re < 0 := by
  have h := tendsto_inv_smul_threeCells hpos hlam hX
  have h11 : Tendsto (fun j => (((nuclearNorm (X j))⁻¹ • X j) 1 1).re) atTop
      (𝓝 (signFill 1 1).re) :=
    (Complex.continuous_re.tendsto _).comp (((continuous_id.matrix_elem 1 1).tendsto _).comp h)
  have hneg : (signFill 1 1).re < 0 := by norm_num [signFill]
  filter_upwards [(tendsto_order.1 h11).2 0 hneg] with j hj
  rw [Matrix.smul_apply, Complex.real_smul, Complex.re_ofReal_mul] at hj
  by_contra hge
  exact absurd hj (not_lt.2 (mul_nonneg (inv_nonneg.2 (nuclearNorm_nonneg _)) (not_lt.1 hge)))

/-- A number with a negative real part is nearer to the codeword `−1` than to `+1`. -/
lemma norm_sub_signCode_lt {z : ℂ} (hz : z.re < 0) :
    ‖z - signCode false‖ < ‖z - signCode true‖ := by
  refine lt_of_pow_lt_pow_left₀ 2 (norm_nonneg _) ?_
  rw [Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply, Complex.normSq_apply]
  simp only [signCode, Bool.false_eq_true, ite_false, ite_true, Complex.sub_re, Complex.sub_im,
    Complex.neg_re, Complex.neg_im, Complex.one_re, Complex.one_im]
  nlinarith

/-- **The unseen cell is decoded to the wrong class** (`review/flat_spectrum.md`, item XE15):
in the end the codeword of the class of `−1` is nearer to the unseen cell `(1, 1)` of a
trained product than the codeword of the class that the kernel shows there. -/
theorem eventually_unseen_wrong (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss (codeProbe signCode threeCells) threeLabels) (lams j)
      (X j)) :
    ∀ᶠ j in atTop, ‖X j 1 1 - signCode false‖ < ‖X j 1 1 - signCode (signLabel (1, 1))‖ := by
  filter_upwards [eventually_re_unseen_neg hpos hlam hX] with j hj
  rw [show signLabel (1, 1) = true by decide]
  exact norm_sub_signCode_lt hj

/-! The witnesses: trained products exist for every weight decay `1 / (j + 1)`, so the
hypotheses are met, and the kernel shows the class of `+1` at the unseen cell. -/

example : signKernel 1 1 = signCode (signLabel (1, 1)) := signKernel_apply (1, 1)

example := norm_sub_signCode_lt (z := -1) (by simp)

example : ∃ X : ℕ → Matrix (Fin 2) (Fin 2) ℂ,
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 signFill) ∧
      (∀ᶠ j in atTop, (X j 1 1).re < 0) ∧
      ∀ᶠ j in atTop, ‖X j 1 1 - signCode false‖ < ‖X j 1 1 - signCode (signLabel (1, 1))‖ := by
  choose X hX using fun j : ℕ => exists_isNuclearMin_ceLoss (codeProbe signCode threeCells)
    threeLabels (lam := 1 / ((j : ℝ) + 1)) (by positivity)
  have hpos : ∀ j : ℕ, (0 : ℝ) < 1 / ((j : ℝ) + 1) := fun j => by positivity
  exact ⟨X, tendsto_inv_smul_threeCells hpos tendsto_one_div_add_atTop_nhds_zero_nat hX,
    eventually_re_unseen_neg hpos tendsto_one_div_add_atTop_nhds_zero_nat hX,
    eventually_unseen_wrong hpos tendsto_one_div_add_atTop_nhds_zero_nat hX⟩

end Transformer.Grokking.FlatSpectrum.Training.Scope
