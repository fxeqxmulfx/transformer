module

public import Transformer.Grokking.FlatSpectrum.Training.SignPath

/-!
# The code `±1`: trained products for one weight decay have the same real parts

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE11.
For the code `±1` the loss of a sample whose label has the sign `σ`, at a cell with the entry
`x`, is `log (1 + e^{−2 σ Re x})` (`crossEntropy_signCode`, `ceLoss_signCode_eq`). The function
`u ↦ log (1 + e^u)` is strictly midpoint convex (`log_one_add_exp_midpoint_lt`): for `a ≠ b`,
`2 e^{(a+b)/2} < e^a + e^b`, so `(1 + e^{(a+b)/2})² < (1 + e^a) (1 + e^b)`. Hence the loss at
the midpoint of two products whose real parts differ at a seen cell is less than the mean of
their losses (`ceLoss_signCode_midpoint_lt`), while the nuclear norm is convex. So

    two trained products for one weight decay λ ≥ 0 have the same real parts at every seen cell

(`IsNuclearMin.re_apply_eq_of_signCode`).

The witness is the `1 × 1` kernel `1` on its cell.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open Landscape.Neural (crossEntropy)

/-- **`u ↦ log (1 + e^u)` is strictly midpoint convex.** -/
lemma log_one_add_exp_midpoint_lt {a b : ℝ} (h : a ≠ b) :
    Real.log (1 + Real.exp ((a + b) / 2))
      < (Real.log (1 + Real.exp a) + Real.log (1 + Real.exp b)) / 2 := by
  have hp := Real.exp_pos (a / 2)
  have hq := Real.exp_pos (b / 2)
  have hpq : Real.exp (a / 2) - Real.exp (b / 2) ≠ 0 :=
    sub_ne_zero.2 fun e => h (by linarith [Real.exp_injective e])
  have ea : Real.exp a = Real.exp (a / 2) * Real.exp (a / 2) := by
    rw [← Real.exp_add]
    ring_nf
  have eb : Real.exp b = Real.exp (b / 2) * Real.exp (b / 2) := by
    rw [← Real.exp_add]
    ring_nf
  have eab : Real.exp ((a + b) / 2) = Real.exp (a / 2) * Real.exp (b / 2) := by
    rw [← Real.exp_add]
    ring_nf
  have key : (1 + Real.exp ((a + b) / 2)) ^ 2 < (1 + Real.exp a) * (1 + Real.exp b) := by
    rw [ea, eb, eab]
    nlinarith [mul_self_pos.2 hpq]
  have h1 := Real.log_lt_log (by positivity) key
  rw [Real.log_pow, Real.log_mul (by positivity) (by positivity)] at h1
  push_cast at h1
  linarith

lemma log_one_add_exp_midpoint_le (a b : ℝ) :
    Real.log (1 + Real.exp ((a + b) / 2))
      ≤ (Real.log (1 + Real.exp a) + Real.log (1 + Real.exp b)) / 2 := by
  rcases eq_or_ne a b with rfl | h
  · rw [show (a + a) / 2 = a by ring]
    linarith
  · exact (log_one_add_exp_midpoint_lt h).le

/-- The cross-entropy of the code `±1` at the logits `±r`: `log (1 + e^{−2 σ r})` for the label
of sign `σ`. -/
lemma crossEntropy_signCode (b : Bool) (r : ℝ) :
    crossEntropy (fun c : Bool => (signCode c).re * r) b
      = Real.log (1 + Real.exp (-(2 * ((signCode b).re * r)))) := by
  have h1 : Real.exp r + Real.exp (-r) = Real.exp r * (1 + Real.exp (-(2 * r))) := by
    rw [mul_add, mul_one, ← Real.exp_add]
    ring_nf
  have h2 : Real.exp r + Real.exp (-r) = Real.exp (-r) * (1 + Real.exp (2 * r)) := by
    rw [mul_add, mul_one, ← Real.exp_add]
    ring_nf
  cases b <;> simp only [crossEntropy, Fintype.sum_bool, signCode, ite_true,
    Bool.false_eq_true, ite_false, Complex.one_re, Complex.neg_re, one_mul, neg_mul, neg_neg,
    mul_neg]
  · rw [h2, Real.log_mul (Real.exp_pos _).ne' (by positivity), Real.log_exp]
    ring
  · rw [h1, Real.log_mul (Real.exp_pos _).ne' (by positivity), Real.log_exp]
    ring

variable {m n S : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype S]
  {cell : S → m × n} {y : S → Bool} {X Y : Matrix m n ℂ} {lam : ℝ}

/-- The cross-entropy of the code `±1` is a sum of `log (1 + e^{−2 σ Re x})` over the samples. -/
lemma ceLoss_signCode_eq (X : Matrix m n ℂ) :
    ceLoss (codeProbe signCode cell) y X = ∑ s, Real.log (1 + Real.exp
      (-(2 * ((signCode (y s)).re * (X (cell s).1 (cell s).2).re)))) := by
  simp only [ceLoss, pairing_codeProbe_signCode, crossEntropy_signCode]

/-- **The loss at the midpoint of two products whose real parts differ at a seen cell is less
than the mean of their losses** (`review/flat_spectrum.md`, item XE11), for the code `±1`. -/
theorem ceLoss_signCode_midpoint_lt {s₀ : S}
    (h : (X (cell s₀).1 (cell s₀).2).re ≠ (Y (cell s₀).1 (cell s₀).2).re) :
    ceLoss (codeProbe signCode cell) y ((1 / 2 : ℝ) • (X + Y))
      < (ceLoss (codeProbe signCode cell) y X + ceLoss (codeProbe signCode cell) y Y) / 2 := by
  have hσ : ∀ b : Bool, (signCode b).re ≠ 0 := fun b => by cases b <;> simp [signCode]
  have hmid : ∀ s, -(2 * ((signCode (y s)).re *
      (((1 / 2 : ℝ) • (X + Y)) (cell s).1 (cell s).2).re))
      = (-(2 * ((signCode (y s)).re * (X (cell s).1 (cell s).2).re))
        + -(2 * ((signCode (y s)).re * (Y (cell s).1 (cell s).2).re))) / 2 := fun s => by
    rw [Matrix.smul_apply, Matrix.add_apply, Complex.smul_re, Complex.add_re, smul_eq_mul]
    ring
  rw [ceLoss_signCode_eq, ceLoss_signCode_eq, ceLoss_signCode_eq, ← Finset.sum_add_distrib,
    Finset.sum_div]
  refine Finset.sum_lt_sum (fun s _ => ?_) ⟨s₀, Finset.mem_univ _, ?_⟩
  · rw [hmid]
    exact log_one_add_exp_midpoint_le _ _
  · rw [hmid]
    exact log_one_add_exp_midpoint_lt fun e => h (mul_left_cancel₀ (hσ (y s₀)) (by linarith))

/-- **Two trained products for one weight decay have the same real parts at every seen cell**
(`review/flat_spectrum.md`, item XE11), for the code `±1` and `λ ≥ 0`. -/
theorem IsNuclearMin.re_apply_eq_of_signCode (hlam : 0 ≤ lam)
    (hX : IsNuclearMin (ceLoss (codeProbe signCode cell) y) lam X)
    (hY : IsNuclearMin (ceLoss (codeProbe signCode cell) y) lam Y) (s : S) :
    (X (cell s).1 (cell s).2).re = (Y (cell s).1 (cell s).2).re := by
  by_contra h
  have h1 := ceLoss_signCode_midpoint_lt (y := y) h
  have h2 : nuclearNorm ((1 / 2 : ℝ) • (X + Y)) ≤ (nuclearNorm X + nuclearNorm Y) / 2 := by
    rw [nuclearNorm_real_smul (by norm_num)]
    linarith [nuclearNorm_add_le X Y]
  have h3 := hX ((1 / 2 : ℝ) • (X + Y))
  have h4 := hY ((1 / 2 : ℝ) • (X + Y))
  nlinarith [mul_le_mul_of_nonneg_left h2 hlam]

/-! The witness: the `1 × 1` kernel `1` on its cell, with the label of the codeword `1`. -/

local notation "cell₀" => fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 1))

example := log_one_add_exp_midpoint_lt zero_ne_one

/-- The products `0` and `1` differ at the cell. -/
example := ceLoss_signCode_midpoint_lt (cell := cell₀) (y := fun _ => true) (s₀ := 0)
  (X := 0) (Y := 1) (by simp)

/-- The trained product `1 • 1` of `SignPath`, compared with itself. -/
example :=
  have h := isNuclearMin_smul_signCode (cell := cell₀) (y := fun _ => true) (K := 1)
    ⟨fun _ _ _ => Subsingleton.elim _ _, fun _ => ⟨0, Subsingleton.elim _ _⟩⟩
    (fun _ => by simp [signCode]) one_pos isFlat_one zero_le_one
  h.re_apply_eq_of_signCode (by positivity) h 0

end Transformer.Grokking.FlatSpectrum.Training
