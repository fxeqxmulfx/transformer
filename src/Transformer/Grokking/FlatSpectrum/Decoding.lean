module

public import Transformer.Grokking.FlatSpectrum.FlatFill

/-!
# Decoding the opposite phase

Source: `review/flat_spectrum.md`, Corollary: the fill `−ψ(f)` "decodes to f + (p±1)/2 and is
always wrong". A readout decodes to the nearest answer, the `c` minimizing `|z − ψ(c)|`.
With `ψ(f) = ψ(f − c) ψ(c)`, `|−ψ(f) − ψ(c)|² = 2 + 2 cos (2π k/p)`, `k = (f − c).val`. So the
truth `k = 0` is the farthest answer, and for odd `p = 2m + 1` the cosine is least exactly at
`k = m, m + 1`: `c = f + (p + 1)/2` and `c = f + (p − 1)/2`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Real

variable {p : ℕ} [NeZero p]

/-- `ψ(x) = exp (2πi x/p)`. -/
lemma stdAddChar_eq_exp (x : ZMod p) :
    (ZMod.stdAddChar x : ℂ) = Complex.exp ((2 * π * x.val / p : ℝ) * Complex.I) := by
  rw [ZMod.stdAddChar_apply, ZMod.toCircle_apply]
  congr 1
  push_cast
  ring

/-- `|−ψ(f) − ψ(c)|² = 2 + 2 cos (2π (f − c).val / p)`. -/
lemma norm_neg_stdAddChar_sub_sq (f c : ZMod p) :
    ‖-ZMod.stdAddChar f - ZMod.stdAddChar c‖ ^ 2 = 2 + 2 * cos (2 * π * (f - c).val / p) := by
  have h : ZMod.stdAddChar f = ZMod.stdAddChar (f - c) * ZMod.stdAddChar c := by
    rw [← AddChar.map_add_eq_mul, sub_add_cancel]
  rw [h, show -(ZMod.stdAddChar (f - c) * ZMod.stdAddChar c) - ZMod.stdAddChar c =
      -((ZMod.stdAddChar (f - c) + 1) * ZMod.stdAddChar c) by ring, norm_neg, norm_mul,
    show ‖(ZMod.stdAddChar c : ℂ)‖ = 1 by simp, mul_one, stdAddChar_eq_exp, Complex.sq_norm,
    Complex.normSq_apply]
  simp only [Complex.add_re, Complex.add_im, Complex.exp_ofReal_mul_I_re,
    Complex.exp_ofReal_mul_I_im, Complex.one_re, Complex.one_im, add_zero]
  linear_combination sin_sq_add_cos_sq (2 * π * (f - c).val / p)

example := norm_neg_stdAddChar_sub_sq (p := 5) 1 2

/-- The truth is the farthest answer from the opposite phase:
`|−ψ(f) − ψ(c)| < |−ψ(f) − ψ(f)|` for `c ≠ f`, `review/flat_spectrum.md`, Corollary. -/
theorem norm_neg_stdAddChar_sub_lt {f c : ZMod p} (hc : c ≠ f) :
    ‖-ZMod.stdAddChar f - ZMod.stdAddChar c‖ < ‖-ZMod.stdAddChar f - ZMod.stdAddChar f‖ := by
  rw [← pow_lt_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero,
    norm_neg_stdAddChar_sub_sq, norm_neg_stdAddChar_sub_sq, sub_self, ZMod.val_zero]
  simp only [Nat.cast_zero, mul_zero, zero_div, cos_zero]
  have hp : (0 : ℝ) < p := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne p)
  have hv : (f - c).val ≠ 0 := (ZMod.val_eq_zero _).not.mpr (sub_ne_zero.mpr hc.symm)
  have hlt : ((f - c).val : ℝ) < p := by exact_mod_cast ZMod.val_lt (f - c)
  have h0 : 0 ≤ 2 * π * (f - c).val / p := by positivity
  have h1 : cos (2 * π * (f - c).val / p) ≠ 1 := by
    rw [Ne, cos_eq_one_iff_of_lt_of_lt (by linarith [pi_pos]) (by
      rw [div_lt_iff₀ hp]
      nlinarith [pi_pos])]
    exact div_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero pi_ne_zero)
      (Nat.cast_ne_zero.mpr hv)) hp.ne'
  linarith [lt_of_le_of_ne (cos_le_one _) h1]

example := norm_neg_stdAddChar_sub_lt (p := 5) (f := 1) (c := 2) (by decide)

/-- The opposite phase never decodes to the truth: for `p ≥ 2`, `f` is not the nearest answer
to `−ψ(f)`, `review/flat_spectrum.md`, Corollary ("always wrong"). -/
theorem not_isMinOn_norm_neg_stdAddChar_sub (hp : 2 ≤ p) (f : ZMod p) :
    ¬IsMinOn (fun c => ‖-ZMod.stdAddChar f - ZMod.stdAddChar c‖) Set.univ f := by
  intro h
  have : Fact (1 < p) := ⟨hp⟩
  have hne : f + 1 ≠ f := fun h' => one_ne_zero (by linear_combination h' : (1 : ZMod p) = 0)
  exact (norm_neg_stdAddChar_sub_lt hne).not_ge (h (Set.mem_univ (f + 1)))

example := not_isMinOn_norm_neg_stdAddChar_sub (p := 5) (by norm_num) 1

/-- On `ℤ/(2m + 1)`, `k ↦ cos (2π k/p)` is least exactly at `k = m` and `k = m + 1`. -/
lemma cos_le_cos_odd {m k : ℕ} (hk : k < 2 * m + 1) :
    cos (2 * π * m / (2 * m + 1 : ℕ)) ≤ cos (2 * π * k / (2 * m + 1 : ℕ)) ∧
      (cos (2 * π * k / (2 * m + 1 : ℕ)) = cos (2 * π * m / (2 * m + 1 : ℕ)) ↔
        k = m ∨ k = m + 1) := by
  have hp : (0 : ℝ) < (2 * m + 1 : ℕ) := by positivity
  have hθ (j : ℕ) (hj : j ≤ m) :
      0 ≤ 2 * π * j / (2 * m + 1 : ℕ) ∧ 2 * π * j / (2 * m + 1 : ℕ) ≤ π := by
    refine ⟨by positivity, ?_⟩
    rw [div_le_iff₀ hp]
    have : (j : ℝ) ≤ m := by exact_mod_cast hj
    push_cast
    nlinarith [pi_pos]
  have key (j : ℕ) (hj : j ≤ m) :
      cos (2 * π * m / (2 * m + 1 : ℕ)) ≤ cos (2 * π * j / (2 * m + 1 : ℕ)) ∧
        (cos (2 * π * j / (2 * m + 1 : ℕ)) = cos (2 * π * m / (2 * m + 1 : ℕ)) ↔ j = m) := by
    obtain ⟨h0, hπ⟩ := hθ j hj
    obtain ⟨h0', hπ'⟩ := hθ m le_rfl
    refine ⟨cos_le_cos_of_nonneg_of_le_pi h0 hπ' (by gcongr), fun h => ?_, fun h => by rw [h]⟩
    have h1 := injOn_cos ⟨h0, hπ⟩ ⟨h0', hπ'⟩ h
    rw [div_left_inj' hp.ne', mul_right_inj' (mul_ne_zero two_ne_zero pi_ne_zero)] at h1
    exact_mod_cast h1
  rcases le_or_gt k m with hkm | hkm
  · obtain ⟨h1, h2⟩ := key k hkm
    exact ⟨h1, h2.trans ⟨Or.inl, fun h => h.elim id fun h => by omega⟩⟩
  · have hcos : cos (2 * π * k / (2 * m + 1 : ℕ)) =
        cos (2 * π * (2 * m + 1 - k : ℕ) / (2 * m + 1 : ℕ)) := by
      rw [Nat.cast_sub hk.le, show 2 * π * ((2 * m + 1 : ℕ) - k : ℝ) / (2 * m + 1 : ℕ) =
        2 * π - 2 * π * k / (2 * m + 1 : ℕ) by field_simp, cos_two_pi_sub]
    obtain ⟨h1, h2⟩ := key (2 * m + 1 - k) (by omega)
    rw [hcos]
    exact ⟨h1, h2.trans ⟨fun h => Or.inr (by omega), fun h => h.elim (fun h => by omega)
      fun h => by omega⟩⟩

example := cos_le_cos_odd (m := 2) (k := 3) (by norm_num)

/-- For odd `p`, the nearest answers to the opposite phase `−ψ(f)` are exactly
`f + (p − 1)/2` and `f + (p + 1)/2`, `review/flat_spectrum.md`, Corollary. -/
theorem isMinOn_norm_neg_stdAddChar_sub_iff (hp : Odd p) (f c : ZMod p) :
    IsMinOn (fun c => ‖-ZMod.stdAddChar f - ZMod.stdAddChar c‖) Set.univ c ↔
      c = f + ((p - 1) / 2 : ℕ) ∨ c = f + ((p + 1) / 2 : ℕ) := by
  obtain ⟨m, rfl⟩ := hp
  rw [show (2 * m + 1 - 1) / 2 = m by omega, show (2 * m + 1 + 1) / 2 = m + 1 by omega,
    isMinOn_univ_iff]
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · have := ZMod.subsingleton_iff.mpr (rfl : 2 * 0 + 1 = 1)
    exact iff_of_true (fun c' => le_of_eq (by rw [Subsingleton.elim c c']))
      (Or.inl (Subsingleton.elim _ _))
  have hle (c' : ZMod (2 * m + 1)) :
      ‖-ZMod.stdAddChar f - ZMod.stdAddChar c‖ ≤ ‖-ZMod.stdAddChar f - ZMod.stdAddChar c'‖ ↔
        cos (2 * π * (f - c).val / (2 * m + 1 : ℕ)) ≤
          cos (2 * π * (f - c').val / (2 * m + 1 : ℕ)) := by
    rw [← pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero,
      norm_neg_stdAddChar_sub_sq, norm_neg_stdAddChar_sub_sq]
    constructor <;> intro h <;> linarith
  have h0 : ((2 * m + 1 : ℕ) : ZMod (2 * m + 1)) = 0 := ZMod.natCast_self _
  push_cast at h0
  have hvm : ((m : ℕ) : ZMod (2 * m + 1)).val = m := ZMod.val_natCast_of_lt (by omega)
  have hvm1 : ((m : ZMod (2 * m + 1)) + 1).val = m + 1 := by
    rw [show (m : ZMod (2 * m + 1)) + 1 = ((m + 1 : ℕ) : ZMod (2 * m + 1)) by push_cast; ring]
    exact ZMod.val_natCast_of_lt (by omega)
  have hk := cos_le_cos_odd (ZMod.val_lt (f - c))
  push_cast
  constructor
  · intro hmin
    have h1 := (hle (f - m)).mp (hmin (f - m))
    rw [sub_sub_cancel, hvm] at h1
    rcases hk.2.mp (le_antisymm h1 hk.1) with h2 | h2
    · have h3 : f - c = (m : ZMod (2 * m + 1)) := by
        rw [← ZMod.natCast_zmod_val (f - c), h2]
      exact Or.inr (by linear_combination -h3 - h0)
    · have h3 : f - c = (m : ZMod (2 * m + 1)) + 1 := by
        rw [← ZMod.natCast_zmod_val (f - c), h2]
        push_cast
        ring
      exact Or.inl (by linear_combination -h3 - h0)
  · intro hc c'
    rw [hle]
    have hv : (f - c).val = m ∨ (f - c).val = m + 1 := by
      rcases hc with rfl | rfl
      · rw [show f - (f + (m : ZMod (2 * m + 1))) = (m : ZMod (2 * m + 1)) + 1 by
          linear_combination -h0]
        exact Or.inr hvm1
      · rw [show f - (f + ((m : ZMod (2 * m + 1)) + 1)) = (m : ZMod (2 * m + 1)) by
          linear_combination -h0]
        exact Or.inl hvm
    rw [hk.2.mpr hv]
    exact (cos_le_cos_odd (ZMod.val_lt (f - c'))).1

example := isMinOn_norm_neg_stdAddChar_sub_iff (p := 97) ⟨48, rfl⟩ 5 54

end Transformer.Grokking.FlatSpectrum
