module

public import Transformer.Grokking.FlatSpectrum.CubicSums

/-!
# `|S_r| ≤ 2 √p`, and `S_r = 0` for `p ≡ 2 mod 3`

Source: `review/flat_spectrum.md`, Theorem B: "For p ≡ 1 mod 3, S_r is a sum of two cubic
Gauss sums, so |S_r| ≤ 2√p. For p ≡ 2 mod 3, S_r = 0." The bound is proved here for every
prime `p` and `r ≠ 0` without characters. Put `w = S_r/2` and `E = ℤ/p ∖ {0}`. As
`∑_{c ∈ E} S_c = 0` and `∑_{c ∈ E} |S_c|² ≤ p (3p − 2) − p² = 2p (p − 1)`,
`∑_{c ∈ E} |S_c + w|² ≤ (p − 1)(2p + |w|²)`. The map `h ↦ r h³` sends `E` into `E` at most
three-to-one and `S_{r h³} = S_r`, so `(p − 1) |3 S_r/2|² ≤ 3 (p − 1)(2p + |S_r|²/4)`, that
is `|S_r|² ≤ 4p`. For `p ≡ 2 mod 3`, `x = (x³)^{(2p−1)/3}`, so cubing is a bijection and
`S_r = ∑_y ψ(r y) = 0`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- `h ↦ r h³` is at most three-to-one on `E = ℤ/p ∖ {0}`:
`∑_{h ∈ E} g(r h³) ≤ 3 ∑_{c ∈ E} g(c)` for `g ≥ 0`. -/
lemma sum_mul_cube_le {r : ZMod p} (hr : r ≠ 0) {g : ZMod p → ℝ} (hg : ∀ c, 0 ≤ g c) :
    ∑ h ∈ univ.erase 0, g (r * h ^ 3) ≤ 3 * ∑ c ∈ univ.erase 0, g c := by
  have hmaps : ∀ h ∈ (univ.erase 0 : Finset (ZMod p)),
      r * h ^ 3 ∈ (univ.erase 0 : Finset (ZMod p)) := fun h hh =>
    mem_erase.mpr ⟨mul_ne_zero hr (pow_ne_zero 3 (ne_of_mem_erase hh)), mem_univ _⟩
  rw [← Finset.sum_fiberwise_of_maps_to hmaps, Finset.mul_sum]
  refine Finset.sum_le_sum fun c _ => ?_
  have hcard : #{h ∈ univ.erase (0 : ZMod p) | r * h ^ 3 = c} ≤ 3 := by
    refine (Finset.card_le_card fun h hh => ?_).trans (card_pow_three_eq_le (r⁻¹ * c))
    refine Finset.mem_filter.mpr ⟨mem_univ _, ?_⟩
    rw [← (Finset.mem_filter.mp hh).2, inv_mul_cancel_left₀ hr]
  calc ∑ h ∈ univ.erase 0 with r * h ^ 3 = c, g (r * h ^ 3)
      = ∑ h ∈ univ.erase 0 with r * h ^ 3 = c, g c :=
        Finset.sum_congr rfl fun h hh => by rw [(Finset.mem_filter.mp hh).2]
    _ = #{h ∈ univ.erase (0 : ZMod p) | r * h ^ 3 = c} * g c := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ 3 * g c := by
        gcongr
        · exact hg c
        · exact_mod_cast hcard

example := @sum_mul_cube_le 7 ⟨Nat.prime_seven⟩ 1 (by decide) (fun _ => 1)
  (fun _ => zero_le_one)

/-- `|S_r| ≤ 2 √p` for `r ≠ 0`, `review/flat_spectrum.md`, Theorem B. -/
theorem norm_cubicSum_le {r : ZMod p} (hr : r ≠ 0) : ‖cubicSum r‖ ≤ 2 * √p := by
  have hp := (Fact.out : p.Prime).one_lt
  have hp1 : (1 : ℝ) < p := by exact_mod_cast hp
  have hE : ((univ.erase (0 : ZMod p)).card : ℝ) = p - 1 := by
    rw [Finset.card_erase_of_mem (mem_univ _), Finset.card_univ, ZMod.card,
      Nat.cast_sub hp.le, Nat.cast_one]
  set w : ℂ := ((1 / 2 : ℝ) : ℂ) * cubicSum r with hwdef
  have hsum : ∑ c ∈ univ.erase (0 : ZMod p), cubicSum c = 0 := by
    rw [Finset.sum_erase_eq_sub (mem_univ _), sum_cubicSum, cubicSum_zero, sub_self]
  have hsq : ∑ c ∈ univ.erase (0 : ZMod p), ‖cubicSum c‖ ^ 2 ≤ 2 * p * (p - 1) := by
    rw [Finset.sum_erase_eq_sub (mem_univ _), sum_norm_cubicSum_sq, cubicSum_zero,
      Complex.norm_natCast]
    have h : ((#{xy : ZMod p × ZMod p | xy.1 ^ 3 = xy.2 ^ 3} : ℕ) : ℝ) ≤ 3 * p - 2 := by
      have h1 : ((3 * p - 2 : ℕ) : ℝ) = 3 * p - 2 := by
        rw [Nat.cast_sub (by omega)]
        push_cast
        ring
      rw [← h1]
      exact_mod_cast card_cube_eq_le
    nlinarith
  have hexp : ∑ c ∈ univ.erase (0 : ZMod p), ‖cubicSum c + w‖ ^ 2 =
      ∑ c ∈ univ.erase (0 : ZMod p), ‖cubicSum c‖ ^ 2 + (p - 1) * ‖w‖ ^ 2 := by
    have hcross : ∑ c ∈ univ.erase (0 : ZMod p), (cubicSum c * (starRingEnd ℂ) w).re = 0 := by
      rw [← Complex.re_sum, ← Finset.sum_mul, hsum, zero_mul, Complex.zero_re]
    simp only [Complex.sq_norm, Complex.normSq_add, Finset.sum_add_distrib, ← Finset.mul_sum,
      hcross, Finset.sum_const, nsmul_eq_mul, hE]
    ring
  have hw : ‖w‖ = ‖cubicSum r‖ / 2 := by
    rw [hwdef, norm_mul, Complex.norm_real, Real.norm_of_nonneg (by norm_num)]
    ring
  have hcube : ∀ h ∈ univ.erase (0 : ZMod p),
      ‖cubicSum (r * h ^ 3) + w‖ ^ 2 = 9 / 4 * ‖cubicSum r‖ ^ 2 := fun h hh => by
    rw [cubicSum_mul_cube r (ne_of_mem_erase hh),
      show cubicSum r + w = ((3 / 2 : ℝ) : ℂ) * cubicSum r by rw [hwdef]; push_cast; ring,
      norm_mul, Complex.norm_real, Real.norm_of_nonneg (by norm_num)]
    ring
  have h3 := sum_mul_cube_le hr (g := fun c => ‖cubicSum c + w‖ ^ 2) fun c => sq_nonneg _
  rw [Finset.sum_congr rfl hcube, Finset.sum_const, nsmul_eq_mul, hE, hexp, hw] at h3
  have h4 : ‖cubicSum r‖ ^ 2 ≤ 4 * p := by nlinarith
  rw [← pow_le_pow_iff_left₀ (norm_nonneg _) (by positivity) two_ne_zero, mul_pow,
    Real.sq_sqrt (by positivity)]
  linarith

example := @norm_cubicSum_le 7 ⟨Nat.prime_seven⟩ 1 (by decide)

/-- For `p ≡ 2 mod 3` and `r ≠ 0`, `S_r = 0`: cubing is a bijection, as
`x = (x³)^{(2p−1)/3}`, `review/flat_spectrum.md`, Theorem B. -/
theorem cubicSum_eq_zero (hp3 : p % 3 = 2) {r : ZMod p} (hr : r ≠ 0) : cubicSum r = 0 := by
  have hp1 := (Fact.out : p.Prime).one_le
  have hroot (x : ZMod p) : (x ^ 3) ^ ((2 * p - 1) / 3) = x := by
    rw [← pow_mul, show 3 * ((2 * p - 1) / 3) = p + (p - 1) by omega, pow_add, ZMod.pow_card,
      ← pow_succ', Nat.sub_add_cancel hp1, ZMod.pow_card]
  have hinj : Function.Injective fun x : ZMod p => x ^ 3 := fun x y h => by
    have h' : x ^ 3 = y ^ 3 := h
    rw [← hroot x, ← hroot y, h']
  rw [cubicSum, Fintype.sum_bijective _ hinj.bijective_of_finite _
    (fun y => ZMod.stdAddChar (y * r)) fun x => by rw [mul_comm], sum_stdAddChar_mul]
  simp [hr]

example := @cubicSum_eq_zero 5 ⟨Nat.prime_five⟩ rfl 1 (by decide)

end Transformer.Grokking.FlatSpectrum
