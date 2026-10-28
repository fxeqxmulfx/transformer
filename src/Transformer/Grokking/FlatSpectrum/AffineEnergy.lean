module

public import Transformer.Grokking.FlatSpectrum.Basic

/-!
# The affine energy of a centered function on `ℤ/p`

Source: `review/flat_spectrum.md`, *Readout-free bound for `P(a) + a b`* (added). For
`h : ℤ/p → ℂ` the affine correlation is `E_h(λ, μ) = ∑_x h(x) conj h(λ x + μ)`. Each value is at
most `‖h‖²` for `λ ≠ 0`, and summed over all `p²` pairs `(λ, μ)` the squares give exactly
`p ‖h‖⁴` when `∑ h = 0`: the average `|E_h(λ, μ)|²` is `‖h‖⁴ / p`. The proof sums over `μ`
first, which leaves the autocorrelation `R(λ (y − x))` with `R(d) = ∑_z conj h(z) h(z + d)`, and
then over `λ`, which leaves `p R(0)` on the diagonal `x = y` and `∑_e R(e) = |∑ h|² = 0` off it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- The affine correlation `E_h(λ, μ) = ∑_x h(x) conj h(λ x + μ)`. -/
noncomputable def affineCorr (h : ZMod p → ℂ) (l μ : ZMod p) : ℂ :=
  ∑ x, h x * star (h (l * x + μ))

/-- `x ↦ l x + μ` permutes `ℤ/p` for `l ≠ 0`. -/
lemma sum_mul_add_eq {M : Type*} [AddCommMonoid M] {l : ZMod p} (hl : l ≠ 0) (μ : ZMod p)
    (F : ZMod p → M) : ∑ x, F (l * x + μ) = ∑ x, F x :=
  (Equiv.sum_comp (Equiv.mulLeft₀ l hl) fun y => F (y + μ)).trans
    (Equiv.sum_comp (Equiv.addRight μ) F)

example := @sum_mul_add_eq 5 ⟨Nat.prime_five⟩ ℂ _ 2 (by decide) 1 fun x => (x.val : ℂ)

/-- `|E_h(λ, μ)| ≤ ‖h‖²` for `λ ≠ 0`, by Cauchy–Schwarz. -/
lemma norm_affineCorr_le (h : ZMod p → ℂ) {l : ZMod p} (hl : l ≠ 0) (μ : ZMod p) :
    ‖affineCorr h l μ‖ ≤ ∑ x, ‖h x‖ ^ 2 := by
  have hsum : ∑ x, ‖h (l * x + μ)‖ ^ 2 = ∑ x, ‖h x‖ ^ 2 :=
    sum_mul_add_eq hl μ fun x => ‖h x‖ ^ 2
  calc ‖affineCorr h l μ‖ = ‖∑ x, star (h (l * x + μ)) * h x‖ := by
        rw [affineCorr]
        exact congrArg _ (sum_congr rfl fun x _ => mul_comm _ _)
    _ ≤ √(∑ x, ‖h (l * x + μ)‖ ^ 2) * √(∑ x, ‖h x‖ ^ 2) := norm_sum_star_mul_le _ _ _
    _ = ∑ x, ‖h x‖ ^ 2 := by
        rw [hsum, Real.mul_self_sqrt (sum_nonneg fun _ _ => sq_nonneg _)]

example (h : ZMod 5 → ℂ) := @norm_affineCorr_le 5 ⟨Nat.prime_five⟩ h 2 (by decide) 1

/-- Summing over `μ` leaves the autocorrelation:
`∑_μ conj h(l x + μ) h(l y + μ) = ∑_z conj h(z) h(z + l (y − x))`. -/
lemma sum_star_mul_shift (h : ZMod p → ℂ) (l x y : ZMod p) :
    ∑ μ, star (h (l * x + μ)) * h (l * y + μ) =
      ∑ z, star (h z) * h (z + l * (y - x)) := by
  rw [← Equiv.sum_comp (Equiv.addLeft (l * x)) fun z => star (h z) * h (z + l * (y - x))]
  refine sum_congr rfl fun μ _ => ?_
  simp only [Equiv.coe_addLeft]
  congr 2
  ring

example (h : ZMod 5 → ℂ) := @sum_star_mul_shift 5 ⟨Nat.prime_five⟩ h 2 1 3

/-- Summing the autocorrelation of a centered `h` over the dilations `l d` gives `p R(0)` at
`d = 0` and `0` otherwise. -/
lemma sum_autocorr_mul {h : ZMod p → ℂ} (hh : ∑ x, h x = 0) (d : ZMod p) :
    ∑ l, ∑ z, star (h z) * h (z + l * d) =
      if d = 0 then (p : ℂ) * ∑ z, star (h z) * h z else 0 := by
  split_ifs with hd
  · subst hd
    simp only [mul_zero, add_zero, sum_const, card_univ, ZMod.card, nsmul_eq_mul]
  · have hR : ∑ l, ∑ z, star (h z) * h (z + l * d) = ∑ e, ∑ z, star (h z) * h (z + e) :=
      Equiv.sum_comp (Equiv.mulRight₀ d hd) fun e => ∑ z, star (h z) * h (z + e)
    rw [hR, sum_comm]
    refine sum_eq_zero fun z _ => ?_
    have hz : ∑ e, h (z + e) = ∑ e, h e := Equiv.sum_comp (Equiv.addLeft z) h
    rw [← mul_sum, hz, hh, mul_zero]

example := @sum_autocorr_mul 5 ⟨Nat.prime_five⟩
  (fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib]) 1

/-- `|E_h(λ, μ)|² = ∑_{x, y} h(x) conj h(y) conj h(λ x + μ) h(λ y + μ)`. -/
lemma affineCorr_mul_star (h : ZMod p → ℂ) (l μ : ZMod p) :
    affineCorr h l μ * star (affineCorr h l μ) =
      ∑ x, ∑ y, h x * star (h y) * (star (h (l * x + μ)) * h (l * y + μ)) := by
  rw [affineCorr, star_sum, sum_mul_sum]
  refine sum_congr rfl fun x _ => sum_congr rfl fun y _ => ?_
  rw [star_mul', star_star]
  ring

example (h : ZMod 5 → ℂ) := @affineCorr_mul_star 5 ⟨Nat.prime_five⟩ h 2 1

/-- The affine energy of a centered `h`, as a complex identity. -/
lemma sum_sum_affineCorr_mul_star {h : ZMod p → ℂ} (hh : ∑ x, h x = 0) :
    ∑ l, ∑ μ, affineCorr h l μ * star (affineCorr h l μ) =
      (p : ℂ) * (∑ x, h x * star (h x)) ^ 2 := by
  simp_rw [affineCorr_mul_star]
  have hswap : ∀ l : ZMod p,
      ∑ μ, ∑ x, ∑ y, h x * star (h y) * (star (h (l * x + μ)) * h (l * y + μ)) =
        ∑ x, ∑ y, h x * star (h y) * ∑ z, star (h z) * h (z + l * (y - x)) := by
    intro l
    rw [sum_comm]
    refine sum_congr rfl fun x _ => ?_
    rw [sum_comm]
    refine sum_congr rfl fun y _ => ?_
    rw [← mul_sum, sum_star_mul_shift]
  simp_rw [hswap]
  rw [sum_comm]
  have hx : ∀ x : ZMod p, ∑ l, ∑ y, h x * star (h y) * ∑ z, star (h z) * h (z + l * (y - x)) =
      h x * star (h x) * ((p : ℂ) * ∑ z, star (h z) * h z) := by
    intro x
    rw [sum_comm]
    simp_rw [← mul_sum, sum_autocorr_mul hh, sub_eq_zero]
    rw [sum_eq_single x (fun y _ hy => by rw [ite_eq_right hy, mul_zero]) (by simp),
      ite_eq_left rfl]
  rw [sum_congr rfl fun x _ => hx x, ← sum_mul, sq]
  have hc : ∑ z, star (h z) * h z = ∑ x, h x * star (h x) :=
    sum_congr rfl fun z _ => mul_comm _ _
  rw [hc]
  ring

/-- **The affine energy** of a centered `h : ℤ/p → ℂ`:
`∑_{λ, μ} |∑_x h(x) conj h(λ x + μ)|² = p ‖h‖⁴`. -/
theorem sum_sum_norm_sq_affineCorr {h : ZMod p → ℂ} (hh : ∑ x, h x = 0) :
    ∑ l, ∑ μ, ‖affineCorr h l μ‖ ^ 2 = p * (∑ x, ‖h x‖ ^ 2) ^ 2 := by
  have hz : ∀ z : ℂ, z * star z = ((‖z‖ ^ 2 : ℝ) : ℂ) := fun z => by
    rw [Complex.star_def, Complex.mul_conj']
    push_cast
    ring
  have h1 := sum_sum_affineCorr_mul_star hh
  simp_rw [hz] at h1
  exact_mod_cast h1

example := @sum_sum_norm_sq_affineCorr 5 ⟨Nat.prime_five⟩
  (fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib])

end Transformer.Grokking.FlatSpectrum
