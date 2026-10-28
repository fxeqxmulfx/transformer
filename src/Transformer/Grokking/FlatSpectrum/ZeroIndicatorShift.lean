module

public import Transformer.Grokking.FlatSpectrum.ZeroIndicator

/-!
# The shifted indicator of answer `0`: its columns

Source: `review/flat_spectrum.md`, Torch check: "Best readout is the indicator of answer 0 | Yes,
after centering" and "31.5 at p=97, 60.0 at p=199 | These are the centered values". The readout
`[x = 0] − μ` of `a³ + a b` gives `K_μ(a, b) = [a³ + a b = 0] − μ`. Row `0` of
`K = [a³ + a b = 0]` is all ones and row `a ≠ 0` has its one at `b = −a²`. So column `b` has
`1 + 2 u_b` ones, where `u_b = 1` if `b = −a²` for some `a ≠ 0` and `u_b = 0` otherwise, and two
columns `b ≠ b'` share only row `0`. Counting the ones of the rows `a ≠ 0` by column,
`∑_b u_b = (p − 1)/2`. Hence
`(K_μᴴ K_μ)_{b b'} = 1 − 2μ + p μ² − 2μ (u_b + u_b') + 2 [b = b'] u_b`, and the entries of `K_μ`
sum to `2p − 1 − p² μ`: the centered kernel takes `μ = (2p − 1)/p²`, the mean of `K` over the
`p²` pairs.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {p : ℕ}

variable (p) in
/-- The readout `[x = 0] − μ` of `a³ + a b`: `K_μ(a, b) = [a³ + a b = 0] − μ`. -/
noncomputable def shiftedZeroIndicatorKernel (μ : ℝ) : Matrix (ZMod p) (ZMod p) ℂ :=
  taskKernel (fun a b => a ^ 3 + a * b) fun x => (if x = 0 then 1 else 0) - μ

variable (p) in
/-- The centered indicator of answer `0`, `review/flat_spectrum.md`, Torch check: the shift
`μ = (2p − 1)/p²` is the mean of `[a³ + a b = 0]` over the `p²` pairs
(`sum_centeredZeroIndicatorKernel`). -/
noncomputable def centeredZeroIndicatorKernel : Matrix (ZMod p) (ZMod p) ℂ :=
  shiftedZeroIndicatorKernel p ((2 * p - 1) / p ^ 2)

/-- `K_μ(a, b) = [a³ + a b = 0] − μ`, a real number. -/
lemma shiftedZeroIndicatorKernel_apply (μ : ℝ) (a b : ZMod p) :
    shiftedZeroIndicatorKernel p μ a b = (((if a ^ 3 + a * b = 0 then 1 else 0) - μ : ℝ) : ℂ) := by
  simp only [shiftedZeroIndicatorKernel, taskKernel, of_apply]
  split_ifs <;> simp

variable [Fact p.Prime]

variable (p) in
/-- `u_b = 1` if `b = −a²` for some `a ≠ 0`, else `0`. -/
noncomputable def negSqIndicator (b : ZMod p) : ℝ :=
  if ∃ a : ZMod p, a ≠ 0 ∧ -a ^ 2 = b then 1 else 0

lemma negSqIndicator_mul_self (b : ZMod p) :
    negSqIndicator p b * negSqIndicator p b = negSqIndicator p b := by
  unfold negSqIndicator
  split_ifs <;> norm_num

/-- `#{a ≠ 0 | −a² = b} = 2 u_b`, as `a² = a₀²` exactly at `a = ±a₀`. -/
lemma sum_neg_sq_eq (hp : p ≠ 2) (b : ZMod p) :
    ∑ a : ZMod p, (if a ≠ 0 ∧ -a ^ 2 = b then (1 : ℝ) else 0) = 2 * negSqIndicator p b := by
  unfold negSqIndicator
  split_ifs with h
  · obtain ⟨a₀, ha₀, rfl⟩ := h
    have hne : a₀ ≠ -a₀ := fun h => ha₀ ((eq_neg_iff_of_ne_two hp).mp h)
    have hfilter : ({a | a ≠ 0 ∧ -a ^ 2 = -a₀ ^ 2} : Finset (ZMod p)) = {a₀, -a₀} := by
      ext a
      simp only [mem_filter, mem_univ, true_and, mem_insert, mem_singleton, neg_inj]
      constructor
      · exact fun h => sq_eq_sq_iff_eq_or_eq_neg.mp h.2
      · rintro (rfl | rfl)
        · exact ⟨ha₀, rfl⟩
        · exact ⟨neg_ne_zero.mpr ha₀, by ring⟩
    rw [← sum_filter, hfilter, sum_pair hne]
    norm_num
  · rw [mul_zero]
    exact sum_eq_zero fun a _ => ite_eq_right fun ha => h ⟨a, ha⟩

example := @sum_neg_sq_eq 5 ⟨Nat.prime_five⟩ (by decide) 1

/-- `∑_b u_b = (p − 1)/2`: each `a ≠ 0` is counted once, at `b = −a²`. -/
theorem sum_negSqIndicator (hp : p ≠ 2) : ∑ b, negSqIndicator p b = ((p : ℝ) - 1) / 2 := by
  have h : ∑ b : ZMod p, 2 * negSqIndicator p b = p - 1 := by
    simp_rw [← sum_neg_sq_eq hp]
    rw [sum_comm]
    have hrow (a : ZMod p) : ∑ b : ZMod p, (if a ≠ 0 ∧ -a ^ 2 = b then (1 : ℝ) else 0) =
        if a ≠ 0 then 1 else 0 := by
      rcases eq_or_ne a 0 with rfl | ha
      · simp
      · simp [ha]
    rw [sum_congr rfl fun a _ => hrow a, sum_boole, filter_ne', card_erase_of_mem (mem_univ _),
      card_univ, ZMod.card, Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]
  rw [← mul_sum] at h
  linarith

example := @sum_negSqIndicator 5 ⟨Nat.prime_five⟩ (by decide)

/-- `[a³ + a b = 0] = [a = 0] + [a ≠ 0 ∧ −a² = b]`. -/
lemma ite_cube_add_mul_eq (a b : ZMod p) :
    (if a ^ 3 + a * b = 0 then (1 : ℝ) else 0) =
      (if a = 0 then 1 else 0) + if a ≠ 0 ∧ -a ^ 2 = b then 1 else 0 := by
  rcases eq_or_ne a 0 with rfl | ha
  · simp
  · have h : a ^ 3 + a * b = 0 ↔ -a ^ 2 = b := by
      rw [cube_add_mul_eq_zero_iff, or_iff_right ha, eq_comm]
    simp [h, ha]

/-- Column `b` of `[a³ + a b = 0]` has `1 + 2 u_b` ones. -/
lemma sum_ite_cube_add_mul (hp : p ≠ 2) (b : ZMod p) :
    ∑ a : ZMod p, (if a ^ 3 + a * b = 0 then (1 : ℝ) else 0) = 1 + 2 * negSqIndicator p b := by
  rw [sum_congr rfl fun a _ => ite_cube_add_mul_eq a b, sum_add_distrib, sum_ite_eq',
    ite_eq_left (mem_univ _), sum_neg_sq_eq hp]

example := @sum_ite_cube_add_mul 5 ⟨Nat.prime_five⟩ (by decide) 1

/-- Two columns `b ≠ b'` of `[a³ + a b = 0]` share only row `0`. -/
lemma sum_ite_cube_add_mul_mul (hp : p ≠ 2) (b b' : ZMod p) :
    ∑ a : ZMod p, (if a ^ 3 + a * b = 0 then (1 : ℝ) else 0) *
        (if a ^ 3 + a * b' = 0 then 1 else 0) =
      1 + if b = b' then 2 * negSqIndicator p b else 0 := by
  rcases eq_or_ne b b' with rfl | hbb
  · rw [ite_eq_left rfl, ← sum_ite_cube_add_mul hp]
    exact sum_congr rfl fun a _ => by split_ifs <;> norm_num
  · rw [ite_eq_right hbb, add_zero]
    have key (a : ZMod p) : (if a ^ 3 + a * b = 0 then (1 : ℝ) else 0) *
        (if a ^ 3 + a * b' = 0 then 1 else 0) = if a = 0 then 1 else 0 := by
      simp only [cube_add_mul_eq_zero_iff]
      rcases eq_or_ne a 0 with rfl | ha
      · simp
      · by_cases h1 : b = -a ^ 2
        · have h2 : ¬ b' = -a ^ 2 := fun h2 => hbb (h1.trans h2.symm)
          simp [ha, h1, h2]
        · simp [ha, h1]
    rw [sum_congr rfl fun a _ => key a, sum_ite_eq', ite_eq_left (mem_univ _)]

example := @sum_ite_cube_add_mul_mul 5 ⟨Nat.prime_five⟩ (by decide) 1 1

/-- `(K_μᴴ K_μ)_{b b'} = 1 − 2μ + p μ² − 2μ (u_b + u_b') + 2 [b = b'] u_b`. -/
theorem conjTranspose_mul_shiftedZeroIndicatorKernel_apply (hp : p ≠ 2) (μ : ℝ) (b b' : ZMod p) :
    ((shiftedZeroIndicatorKernel p μ)ᴴ * shiftedZeroIndicatorKernel p μ) b b' =
      ((1 - 2 * μ + p * μ ^ 2 - 2 * μ * (negSqIndicator p b + negSqIndicator p b') +
        if b = b' then 2 * negSqIndicator p b else 0 : ℝ) : ℂ) := by
  simp only [mul_apply, conjTranspose_apply, shiftedZeroIndicatorKernel_apply, Complex.star_def,
    Complex.conj_ofReal, ← Complex.ofReal_mul, ← Complex.ofReal_sum]
  congr 1
  have h (a : ZMod p) : ((if a ^ 3 + a * b = 0 then (1 : ℝ) else 0) - μ) *
      ((if a ^ 3 + a * b' = 0 then 1 else 0) - μ) =
      (if a ^ 3 + a * b = 0 then (1 : ℝ) else 0) * (if a ^ 3 + a * b' = 0 then 1 else 0) -
        μ * (if a ^ 3 + a * b = 0 then 1 else 0) - μ * (if a ^ 3 + a * b' = 0 then 1 else 0) +
        μ ^ 2 := by ring
  rw [sum_congr rfl fun a _ => h a, sum_add_distrib, sum_sub_distrib, sum_sub_distrib, ← mul_sum,
    ← mul_sum, sum_ite_cube_add_mul_mul hp, sum_ite_cube_add_mul hp, sum_ite_cube_add_mul hp,
    sum_const, card_univ, ZMod.card, nsmul_eq_mul]
  ring

example := @conjTranspose_mul_shiftedZeroIndicatorKernel_apply 5 ⟨Nat.prime_five⟩ (by decide) 0 1 1

/-- The entries of `K_μ` sum to `2p − 1 − p² μ`. -/
theorem sum_shiftedZeroIndicatorKernel (hp : p ≠ 2) (μ : ℝ) :
    ∑ a, ∑ b, shiftedZeroIndicatorKernel p μ a b = ((2 * p - 1 - p ^ 2 * μ : ℝ) : ℂ) := by
  simp only [shiftedZeroIndicatorKernel_apply, ← Complex.ofReal_sum]
  congr 1
  rw [sum_comm]
  simp only [sum_sub_distrib, sum_ite_cube_add_mul hp, sum_add_distrib, ← mul_sum,
    sum_negSqIndicator hp, sum_const, card_univ, ZMod.card, nsmul_eq_mul]
  ring

example := @sum_shiftedZeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide) 0

/-- The centered kernel is centered: its entries sum to `0`. -/
theorem sum_centeredZeroIndicatorKernel (hp : p ≠ 2) :
    ∑ a, ∑ b, centeredZeroIndicatorKernel p a b = 0 := by
  have hp0 : (p : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Fact.out : p.Prime).ne_zero
  rw [centeredZeroIndicatorKernel, sum_shiftedZeroIndicatorKernel hp,
    show 2 * (p : ℝ) - 1 - p ^ 2 * ((2 * p - 1) / p ^ 2) = 0 by field_simp; ring,
    Complex.ofReal_zero]

example := @sum_centeredZeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide)

end Transformer.Grokking.FlatSpectrum
