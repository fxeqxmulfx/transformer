module

public import Transformer.Grokking.FlatSpectrum.StepPhase

/-!
# One angle places the wrap of every cell

Source: `review/flat_spectrum.md`, *The step is the phase of a sum of two phases* (added),
item CH7. The chord of `StepChord` starts at the point `−exp (−iπ / 2p)`, between the phases
of the sums `p − 1` and `p`. Move that point by an angle `2γ`:

`stepChordAt p a b γ = z + exp (−iπ / 2p) · exp (2iγ) = 2 cos (φ − γ) · exp (i (φ − π / 2p + γ))`

(`stepChordAt_eq`). For `|γ| < π / 4p` the cosine keeps its sign at every cell
(`cos_halfAngle_sub_pos`, `cos_halfAngle_sub_neg`): the moved point is still between the two
phases. So the normalized chord from the moved point is the normalized chord times the
constant `exp (iγ)` (`stepChordAt_div_norm`): every answer turns by the same angle and no
answer changes its branch.

The wrap of all cells is placed by one angle, with a tolerance of `π / 2p` for the base
point. A comparison fitted cell by cell (`WrapFill`) is decided at each unseen cell of the
boundary on its own.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Complex Real

/-- The chord to the phase of the sum from the base point of `stepChord` moved by the angle
`2γ`; `review/flat_spectrum.md`, item CH7. -/
noncomputable def stepChordAt (p a b : ℕ) (γ : ℝ) : ℂ :=
  cexp (2 * π * I * ((b : ℂ) - a) / p) * cexp (3 * π * I * (modStep p a b : ℂ) / p) +
    cexp (-(π * I) / (2 * p)) * cexp (((2 * γ : ℝ) : ℂ) * I)

/-- At `γ = 0` it is the chord of `StepChord`. -/
theorem stepChordAt_zero (p a b : ℕ) : stepChordAt p a b 0 = stepChord p a b := by
  simp [stepChordAt, stepChord]

/-- The half angle is `π / 4p` plus `π / 2p` for every unit of the sum. -/
lemma halfAngle_eq_add {p : ℕ} (hp : 0 < p) (a b : ℕ) :
    halfAngle p a b = π / (2 * p) / 2 + π / (2 * p) * ((b + (a + b) % p : ℕ) : ℝ) := by
  have hp' : (p : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hp.ne'
  unfold halfAngle
  field_simp
  ring

example := halfAngle_eq_add (p := 97) (by norm_num) 50 90

/-- **The moved chord in polar form**, `review/flat_spectrum.md`, item CH7. -/
theorem stepChordAt_eq {p a b : ℕ} (ha : a < p) (hb : b < p) (γ : ℝ) :
    stepChordAt p a b γ = ((2 * Real.cos (halfAngle p a b - γ) : ℝ) : ℂ) *
      cexp (((halfAngle p a b - π / (2 * p) + γ : ℝ) : ℂ) * I) := by
  have hp : (p : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hz : cexp (2 * π * I * ((b : ℂ) - a) / p) * cexp (3 * π * I * (modStep p a b : ℂ) / p) =
      cexp (((π * ((b + (a + b) % p : ℕ) : ℝ) / p : ℝ) : ℂ) * I) := by
    rw [← exp_sum_eq_mul ha hb]
    congr 1
    push_cast
    ring
  have hβ : cexp (-(π * I) / (2 * p)) * cexp (((2 * γ : ℝ) : ℂ) * I) =
      cexp (((-(π / (2 * p)) + 2 * γ : ℝ) : ℂ) * I) := by
    rw [← Complex.exp_add]
    congr 1
    push_cast
    ring
  have h1 : (π * ((b + (a + b) % p : ℕ) : ℝ) / p - (-(π / (2 * p)) + 2 * γ)) / 2 =
      halfAngle p a b - γ := by
    unfold halfAngle
    field_simp
    ring
  have h2 : (π * ((b + (a + b) % p : ℕ) : ℝ) / p + (-(π / (2 * p)) + 2 * γ)) / 2 =
      halfAngle p a b - π / (2 * p) + γ := by
    unfold halfAngle
    field_simp
    ring
  rw [stepChordAt, hz, hβ, exp_mul_I_add_exp_mul_I, h1, h2]

example := stepChordAt_eq (p := 97) (a := 50) (b := 90) (by norm_num) (by norm_num) (1 / 1000)

/-- Without a wrap the half angle stays below `π / 2` when moved by less than `π / 4p`. -/
theorem cos_halfAngle_sub_pos {p a b : ℕ} {γ : ℝ} (hγ : |γ| < π / (2 * p) / 2)
    (h : b + (a + b) % p < p) : 0 < Real.cos (halfAngle p a b - γ) := by
  have hp : (0 : ℝ) < p := by exact_mod_cast (by omega : 0 < p)
  have hc : 0 < π / (2 * p) := by positivity
  have hcp : π / (2 * p) * p = π / 2 := by field_simp
  have hn1 : ((b + (a + b) % p : ℕ) : ℝ) ≤ p - 1 := by
    have : ((b + (a + b) % p : ℕ) : ℝ) + 1 ≤ p := by exact_mod_cast h
    linarith
  have hm0 : 0 ≤ π / (2 * p) * ((b + (a + b) % p : ℕ) : ℝ) := by positivity
  have hm1 := mul_le_mul_of_nonneg_left hn1 hc.le
  rw [mul_sub, hcp, mul_one] at hm1
  obtain ⟨hγ1, hγ2⟩ := abs_lt.mp hγ
  rw [halfAngle_eq_add (by omega)]
  exact cos_pos_of_mem_Ioo ⟨by linarith [pi_pos], by linarith⟩

example : 0 < Real.cos (halfAngle 97 10 20 - 1 / 1000) :=
  cos_halfAngle_sub_pos (by
    rw [abs_of_pos (by norm_num), lt_div_iff₀ (by norm_num), lt_div_iff₀ (by norm_num)]
    nlinarith [two_le_pi]) (by norm_num)

/-- With a wrap the half angle stays above `π / 2` when moved by less than `π / 4p`. -/
theorem cos_halfAngle_sub_neg {p a b : ℕ} {γ : ℝ} (hb : b < p) (hγ : |γ| < π / (2 * p) / 2)
    (h : p ≤ b + (a + b) % p) : Real.cos (halfAngle p a b - γ) < 0 := by
  have hp : (0 : ℝ) < p := by exact_mod_cast (by omega : 0 < p)
  have hx : (a + b) % p < p := Nat.mod_lt _ (by omega)
  have hc : 0 < π / (2 * p) := by positivity
  have hcp : π / (2 * p) * p = π / 2 := by field_simp
  have hn0 : (p : ℝ) ≤ ((b + (a + b) % p : ℕ) : ℝ) := by exact_mod_cast h
  have hn1 : ((b + (a + b) % p : ℕ) : ℝ) ≤ 2 * p - 2 := by
    have : ((b + (a + b) % p : ℕ) : ℝ) + 2 ≤ 2 * p := by exact_mod_cast (by omega)
    linarith
  have hm0 := mul_le_mul_of_nonneg_left hn0 hc.le
  have hm1 := mul_le_mul_of_nonneg_left hn1 hc.le
  rw [hcp] at hm0
  rw [mul_sub, ← mul_assoc, mul_comm (π / (2 * p)) 2, mul_assoc, hcp] at hm1
  obtain ⟨hγ1, hγ2⟩ := abs_lt.mp hγ
  rw [halfAngle_eq_add (by omega)]
  exact cos_neg_of_pi_div_two_lt_of_lt (by linarith) (by linarith [pi_pos])

example : Real.cos (halfAngle 97 50 90 - 1 / 1000) < 0 :=
  cos_halfAngle_sub_neg (by norm_num) (by
    rw [abs_of_pos (by norm_num), lt_div_iff₀ (by norm_num), lt_div_iff₀ (by norm_num)]
    nlinarith [two_le_pi]) (by norm_num)

/-- The length of the moved chord. -/
theorem norm_stepChordAt {p a b : ℕ} (ha : a < p) (hb : b < p) (γ : ℝ) :
    ‖stepChordAt p a b γ‖ = 2 * |Real.cos (halfAngle p a b - γ)| := by
  rw [stepChordAt_eq ha hb, norm_mul, Complex.norm_real, Complex.norm_exp_ofReal_mul_I, mul_one,
    Real.norm_eq_abs, abs_mul, abs_two]

example := norm_stepChordAt (p := 97) (a := 50) (b := 90) (by norm_num) (by norm_num) 0

/-- **One angle for all cells**, `review/flat_spectrum.md`, item CH7: for a base point moved
by `2γ` with `|γ| < π / 4p`, the normalized chord of every cell is the normalized chord of
`StepChord` times the same constant `exp (iγ)`. No cell changes its branch. -/
theorem stepChordAt_div_norm {p a b : ℕ} (ha : a < p) (hb : b < p) {γ : ℝ}
    (hγ : |γ| < π / (2 * p) / 2) :
    stepChordAt p a b γ / (‖stepChordAt p a b γ‖ : ℂ) =
      cexp ((γ : ℂ) * I) * (stepChord p a b / (‖stepChord p a b‖ : ℂ)) := by
  have hE : cexp (((halfAngle p a b - π / (2 * p) + γ : ℝ) : ℂ) * I) =
      cexp ((γ : ℂ) * I) * cexp (((halfAngle p a b - π / (2 * p) : ℝ) : ℂ) * I) := by
    rw [← Complex.exp_add]
    congr 1
    push_cast
    ring
  have hneg : ∀ x : ℝ, ((2 * -x : ℝ) : ℂ) = -((2 * x : ℝ) : ℂ) := fun x => by
    push_cast
    ring
  have hne : ∀ x : ℝ, x ≠ 0 → ((2 * x : ℝ) : ℂ) ≠ 0 := fun x hx =>
    Complex.ofReal_ne_zero.mpr (mul_ne_zero two_ne_zero hx)
  rw [norm_stepChordAt ha hb, norm_stepChord ha hb, stepChordAt_eq ha hb, stepChord_eq ha hb]
  rcases lt_or_ge (b + (a + b) % p) p with h | h
  · have h1 := cos_halfAngle_sub_pos hγ h
    have h0 := cos_halfAngle_pos h
    rw [abs_of_pos h1, abs_of_pos h0, mul_div_cancel_left₀ _ (hne _ h1.ne'),
      mul_div_cancel_left₀ _ (hne _ h0.ne'), hE]
  · have h1 := cos_halfAngle_sub_neg hb hγ h
    have h0 := cos_halfAngle_neg hb h
    rw [abs_of_neg h1, abs_of_neg h0, hneg, hneg, div_neg, div_neg,
      mul_div_cancel_left₀ _ (hne _ h1.ne), mul_div_cancel_left₀ _ (hne _ h0.ne), hE, mul_neg]

example : stepChordAt 97 50 90 (1 / 1000) / (‖stepChordAt 97 50 90 (1 / 1000)‖ : ℂ) =
    cexp (((1 / 1000 : ℝ) : ℂ) * I) * (stepChord 97 50 90 / (‖stepChord 97 50 90‖ : ℂ)) :=
  stepChordAt_div_norm (by norm_num) (by norm_num) (by
    rw [abs_of_pos (by norm_num), lt_div_iff₀ (by norm_num), lt_div_iff₀ (by norm_num)]
    nlinarith [two_le_pi])

end Transformer.Grokking.FlatSpectrum
