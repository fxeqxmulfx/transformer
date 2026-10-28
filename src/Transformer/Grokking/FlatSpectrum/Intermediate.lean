module

public import Transformer.Grokking.FlatSpectrum.Conditional
public import Transformer.Grokking.FlatSpectrum.CubicBound

/-!
# Theorem B: the fibers of `a³ + a b`

Source: `review/flat_spectrum.md`, Theorem B: "on the fiber f = c ≠ 0, a runs once over the
nonzero residues and ab = c − a³. The fiber f = 0 also contains the line a = 0." For prime
`p`, `a ≠ 0` has the single partner `b = a⁻¹ (c − a³)`, so the fiber sum of `ψ_r(a b)` is
`ψ_r(c) (S_{−r} − 1) + p [c = 0]` over `p − 1 + p [c = 0]` points. Grouping
`‖E[ψ_r(a b) | f]‖²` by fibers then gives the source's formula, with `S_{−r} = conj S_r`.
With `|S_r| ≤ 2 √p` it is at most `9/(2p) + 8/p^{3/2}`, and Cauchy–Schwarz bounds the
correlation of `ψ_r(a b)` with any readout of the label.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- For `a ≠ 0`, `a³ + a b = c` has the single solution `b = a⁻¹ (c − a³)`. -/
lemma cube_add_mul_eq_iff {a b c : ZMod p} (ha : a ≠ 0) :
    a ^ 3 + a * b = c ↔ b = a⁻¹ * (c - a ^ 3) := by
  constructor
  · rintro rfl
    field_simp
    ring
  · rintro rfl
    field_simp
    ring

example := @cube_add_mul_eq_iff 5 ⟨Nat.prime_five⟩ 1 1 2 (by decide)

/-- The rows `a ≠ 0` of the fiber `a³ + a b = c`: one point each, with `a b = c − a³`. -/
lemma sum_row_ne_zero (r c : ZMod p) {a : ZMod p} (ha : a ≠ 0) :
    ∑ b : ZMod p, (if a ^ 3 + a * b = c then ZMod.stdAddChar (r * (a * b)) else 0) =
      ZMod.stdAddChar (r * c) * ZMod.stdAddChar (-r * a ^ 3) := by
  simp_rw [cube_add_mul_eq_iff ha]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [← AddChar.map_add_eq_mul]
  congr 1
  field_simp
  ring

example := @sum_row_ne_zero 5 ⟨Nat.prime_five⟩ 1 2 1 (by decide)

/-- The fiber sum `∑_{a³ + a b = c} ψ_r(a b) = ψ_r(c) (S_{−r} − 1) + p [c = 0]`. -/
lemma sum_fiber_cube_add_mul (r c : ZMod p) :
    ∑ ab : ZMod p × ZMod p with ab.1 ^ 3 + ab.1 * ab.2 = c, ZMod.stdAddChar (r * (ab.1 * ab.2)) =
      ZMod.stdAddChar (r * c) * (cubicSum (-r) - 1) + if c = 0 then (p : ℂ) else 0 := by
  rw [Finset.sum_filter, Fintype.sum_prod_type, ← Finset.add_sum_erase _ _ (mem_univ 0),
    Finset.sum_congr rfl fun a ha => sum_row_ne_zero r c (ne_of_mem_erase ha), ← Finset.mul_sum,
    Finset.sum_erase_eq_sub (mem_univ 0)]
  simp only [cubicSum, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero,
    add_zero, zero_mul, AddChar.map_zero_eq_one]
  rw [add_comm]
  congr 1
  by_cases h : c = 0
  · subst h
    simp [ZMod.card]
  · simp [h, Ne.symm h]

example := @sum_fiber_cube_add_mul 5 ⟨Nat.prime_five⟩ 1 0

/-- The fiber size `#{a³ + a b = c} = p − 1 + p [c = 0]`. -/
lemma card_fiber_cube_add_mul (c : ZMod p) :
    (#{ab : ZMod p × ZMod p | ab.1 ^ 3 + ab.1 * ab.2 = c} : ℝ) =
      p - 1 + if c = 0 then (p : ℝ) else 0 := by
  have h := sum_fiber_cube_add_mul (p := p) 0 c
  simp only [zero_mul, AddChar.map_zero_eq_one, Finset.sum_const, nsmul_eq_mul, mul_one,
    neg_zero, cubicSum_zero, one_mul] at h
  have h' : ((#{ab : ZMod p × ZMod p | ab.1 ^ 3 + ab.1 * ab.2 = c} : ℝ) : ℂ) =
      (((p : ℝ) - 1 + if c = 0 then (p : ℝ) else 0 : ℝ) : ℂ) := by
    push_cast
    rw [h]
    split_ifs <;> simp
  exact_mod_cast h'

example := @card_fiber_cube_add_mul 5 ⟨Nat.prime_five⟩ 0

/-- **Theorem B** of `review/flat_spectrum.md`: over uniform `(a, b) ∈ (ℤ/p)²` with label
`f = a³ + a b` and `ψ_r(x) = ψ(r x)`,
`‖E[ψ_r(a b) | f]‖² = |S_r − 1|²/p² + |p − 1 + conj S_r|²/(p² (2p − 1))`.
The source states it for `r ≠ 0`; it holds for every `r`. -/
theorem meanSq_condMean_cube_add_mul (r : ZMod p) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => ab.1 ^ 3 + ab.1 * ab.2)
      fun ab => ZMod.stdAddChar (r * (ab.1 * ab.2))) =
      ‖cubicSum r - 1‖ ^ 2 / p ^ 2 +
        ‖((p : ℂ) - 1) + star (cubicSum r)‖ ^ 2 / (p ^ 2 * (2 * p - 1)) := by
  have hp := (Fact.out : p.Prime).one_lt
  have hp1 : (1 : ℝ) < p := by exact_mod_cast hp
  have hS : ‖cubicSum (-r) - 1‖ = ‖cubicSum r - 1‖ := by
    rw [cubicSum_neg, show star (cubicSum r) - 1 = star (cubicSum r - 1) by
      rw [star_sub, star_one], norm_star]
  have hrest : ∀ c ∈ univ.erase (0 : ZMod p),
      ‖ZMod.stdAddChar (r * c) * (cubicSum (-r) - 1) + (if c = 0 then (p : ℂ) else 0)‖ ^ 2 /
        ((p : ℝ) - 1 + if c = 0 then (p : ℝ) else 0) = ‖cubicSum r - 1‖ ^ 2 / (p - 1) := by
    intro c hc
    have hc0 : c ≠ 0 := ne_of_mem_erase hc
    have hψ : ‖(ZMod.stdAddChar (r * c) : ℂ)‖ = 1 := by simp
    simp only [hc0, ite_false, add_zero, norm_mul, hψ, one_mul, hS]
  rw [meanSq, sum_norm_condMean_sq]
  simp_rw [sum_fiber_cube_add_mul, card_fiber_cube_add_mul]
  rw [← Finset.add_sum_erase _ _ (mem_univ (0 : ZMod p)), Finset.sum_congr rfl hrest,
    Finset.sum_const, Finset.card_erase_of_mem (mem_univ _), Finset.card_univ, ZMod.card,
    nsmul_eq_mul, Nat.cast_sub hp.le, Fintype.card_prod, ZMod.card]
  simp only [mul_zero, AddChar.map_zero_eq_one, one_mul, ite_true, cubicSum_neg]
  rw [show star (cubicSum r) - 1 + (p : ℂ) = ((p : ℂ) - 1) + star (cubicSum r) by ring]
  rw [show (p : ℝ) - 1 + p = 2 * p - 1 by ring]
  have h0 : (p : ℝ) ≠ 0 := by positivity
  have h1 : (p : ℝ) - 1 ≠ 0 := by linarith
  have h3 : 2 * (p : ℝ) - 1 ≠ 0 := by linarith
  push_cast
  field_simp
  ring

example := @meanSq_condMean_cube_add_mul 5 ⟨Nat.prime_five⟩ 1

/-- The arithmetic of the bound of Theorem B: for `x ≥ 2`,
`(2√x + 1)²/x² + (x − 1 + 2√x)²/(x² (2x − 1)) ≤ 9/(2x) + 8/x^{3/2}`. With `q = √x` the gap is
`(4q − 9/2)/(x (2x − 1)) ≥ 0`. -/
lemma cube_bound_le {x : ℝ} (hx : 2 ≤ x) :
    (2 * √x + 1) ^ 2 / x ^ 2 + (x - 1 + 2 * √x) ^ 2 / (x ^ 2 * (2 * x - 1)) ≤
      9 / (2 * x) + 8 / (x * √x) := by
  set q := √x
  have hq : x = q ^ 2 := (Real.sq_sqrt (by linarith)).symm
  have hq0 : 0 < q := Real.sqrt_pos.mpr (by linarith)
  have hq98 : 9 / 8 ≤ q := by nlinarith
  have h2q : 0 < 2 * q ^ 2 - 1 := by nlinarith
  rw [hq, ← sub_nonneg]
  have key : 9 / (2 * q ^ 2) + 8 / (q ^ 2 * q) - ((2 * q + 1) ^ 2 / (q ^ 2) ^ 2 +
      (q ^ 2 - 1 + 2 * q) ^ 2 / ((q ^ 2) ^ 2 * (2 * q ^ 2 - 1))) =
      (4 * q - 9 / 2) / (q ^ 2 * (2 * q ^ 2 - 1)) := by
    have hq0' : q ≠ 0 := hq0.ne'
    have h2q' : 2 * q ^ 2 - 1 ≠ 0 := h2q.ne'
    field_simp
    ring
  rw [key]
  exact div_nonneg (by linarith) (by positivity)

example := cube_bound_le (x := 2) le_rfl

/-- The bound of Theorem B, `review/flat_spectrum.md`: "the quantity is at most
4.5/p + O(p^{-3/2})", here with the explicit `‖E[ψ_r(a b) | f]‖² ≤ 9/(2p) + 8/p^{3/2}` for
`r ≠ 0`, from `|S_r| ≤ 2 √p` and `cube_bound_le`. -/
theorem meanSq_condMean_cube_add_mul_le {r : ZMod p} (hr : r ≠ 0) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => ab.1 ^ 3 + ab.1 * ab.2)
      fun ab => ZMod.stdAddChar (r * (ab.1 * ab.2))) ≤ 9 / (2 * p) + 8 / (p * √p) := by
  have hp2 : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
  have hS := norm_cubicSum_le hr
  have hB : ‖cubicSum r - 1‖ ≤ 2 * √p + 1 :=
    (norm_sub_le _ _).trans (by rw [norm_one]; linarith)
  have hA : ‖((p : ℂ) - 1) + star (cubicSum r)‖ ≤ p - 1 + 2 * √p := by
    refine (norm_add_le _ _).trans ?_
    rw [norm_star, show (p : ℂ) - 1 = ((p - 1 : ℝ) : ℂ) by push_cast; ring, Complex.norm_real,
      Real.norm_of_nonneg (by linarith)]
    linarith
  rw [meanSq_condMean_cube_add_mul]
  have h2p : (0 : ℝ) < 2 * p - 1 := by linarith
  calc ‖cubicSum r - 1‖ ^ 2 / p ^ 2 +
        ‖((p : ℂ) - 1) + star (cubicSum r)‖ ^ 2 / (p ^ 2 * (2 * p - 1))
      ≤ (2 * √p + 1) ^ 2 / p ^ 2 + (p - 1 + 2 * √p) ^ 2 / (p ^ 2 * (2 * p - 1)) := by
        gcongr
    _ ≤ 9 / (2 * p) + 8 / (p * √p) := cube_bound_le hp2

example := @meanSq_condMean_cube_add_mul_le 7 ⟨Nat.prime_seven⟩ 1 (by decide)

/-- The corollary of Theorem B, `review/flat_spectrum.md`: "no unit-norm function of the label
correlates with ψ_r(ab) by more than O(p^{-1/2})". For `r ≠ 0` and every readout `φ`,
`|E[conj φ(f) ψ_r(a b)]| ≤ ‖φ(f)‖ √(9/(2p) + 8/p^{3/2})`, by Cauchy–Schwarz. -/
theorem norm_mean_star_mul_cube_add_mul_le {r : ZMod p} (hr : r ≠ 0) (φ : ZMod p → ℂ) :
    ‖mean fun ab : ZMod p × ZMod p =>
      star (φ (ab.1 ^ 3 + ab.1 * ab.2)) * ZMod.stdAddChar (r * (ab.1 * ab.2))‖ ≤
      √(meanSq fun ab : ZMod p × ZMod p => φ (ab.1 ^ 3 + ab.1 * ab.2)) *
        √(9 / (2 * p) + 8 / (p * √p)) :=
  (norm_mean_star_mul_le (fun ab : ZMod p × ZMod p => ab.1 ^ 3 + ab.1 * ab.2) _ φ).trans
    (mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (meanSq_condMean_cube_add_mul_le hr))
      (Real.sqrt_nonneg _))

example := @norm_mean_star_mul_cube_add_mul_le 7 ⟨Nat.prime_seven⟩ 1 (by decide) fun _ => 1

end Transformer.Grokking.FlatSpectrum
