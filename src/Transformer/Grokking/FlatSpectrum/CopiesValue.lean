module

public import Transformer.Grokking.FlatSpectrum.CopiesBestReadout

/-!
# Among `k` copies a pair's own value is hidden as much as its intermediate

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV6, *The best readout*.
`condEnergy_copies` holds for every function `h` of the first pair, and the pair's own value is
one. A function of the label has its variance as its energy (`condEnergy_comp`), so the value
`ψ_r(q)` of one pair `q = P(x) + x y` has the energy `1 − p^{−2}` for every `r ≠ 0`
(`condEnergy_value`), and among `k = 1 + m` pairs it has `p^{−2m}` times that, up to a factor in
`[1 − 1/p, 2]` (`condEnergy_copies_value`), as the intermediate does. Between any two functions
of the first pair the other pairs move the ratio of the energies by at most `2/(1 − 1/p)` either
way (`condEnergy_copies_ratio`): relative to the value, the intermediate is hidden by its energy
given one pair alone, at every `k`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ}

lemma condEnergy_nonneg [NeZero p] {α : Type*} [Fintype α] (f : α → ZMod p) (h : α → ℂ) :
    0 ≤ condEnergy f h :=
  sum_nonneg fun _ _ => div_nonneg (sq_nonneg _) (meanSq_nonneg _)

example := condEnergy_nonneg (p := 5) (fun x : ZMod 5 => x) fun _ => 1

/-- **The energy of a function of the label is its variance**: `E[φ(f) | f] = φ(f)`, so
`∑_t |Cov(1{f = t}, φ(f))|² / P(f = t) = Var φ(f)`. -/
theorem condEnergy_comp [NeZero p] {α : Type*} [Fintype α] (f : α → ZMod p) (φ : ZMod p → ℂ) :
    condEnergy f (fun x => φ (f x)) = var fun x => φ (f x) := by
  have hterm (t : ZMod p) :
      ‖cov (fun x => if f x = t then (1 : ℂ) else 0) (fun x => φ (f x))‖ ^ 2 /
          meanSq (fun x => if f x = t then (1 : ℂ) else 0) =
        meanSq fun x => if f x = t then φ t - mean (fun x => φ (f x)) else 0 := by
    have h1 : (fun x => (if f x = t then (1 : ℂ) else 0) * star (φ (f x))) =
        fun x => (if f x = t then (1 : ℂ) else 0) * star (φ t) := by
      funext x
      split_ifs with hx <;> simp [hx]
    have hcov : cov (fun x => if f x = t then (1 : ℂ) else 0) (fun x => φ (f x)) =
        (meanSq (fun x => if f x = t then (1 : ℂ) else 0) : ℂ) *
          star (φ t - mean fun x => φ (f x)) := by
      simp only [cov]
      rw [h1, mean_mul_const, ofReal_meanSq_ite, star_sub, mul_sub]
    have hsq : (meanSq fun x => if f x = t then φ t - mean (fun x => φ (f x)) else 0) =
        meanSq (fun x => if f x = t then (1 : ℂ) else 0) *
          ‖φ t - mean fun x => φ (f x)‖ ^ 2 := by
      simp only [meanSq, div_mul_eq_mul_div, sum_mul]
      congr 1
      refine sum_congr rfl fun x _ => ?_
      split_ifs <;> simp
    rw [hcov, hsq, norm_mul, Complex.norm_real, norm_star, mul_pow, Real.norm_eq_abs, sq_abs]
    rcases eq_or_ne (meanSq fun x => if f x = t then (1 : ℂ) else 0) 0 with hP | hP
    · simp [hP]
    · field_simp
  unfold condEnergy
  rw [sum_congr rfl fun t _ => hterm t]
  simp only [var, meanSq, ← sum_div]
  congr 1
  rw [sum_comm]
  refine sum_congr rfl fun x _ => ?_
  rw [sum_eq_single (f x) (fun t _ ht => by simp [Ne.symm ht]) (by simp)]
  simp

example := condEnergy_comp (p := 5) (fun x : ZMod 5 => x ^ 2) fun s => ZMod.stdAddChar s

variable [Fact p.Prime]

/-- **The value of one pair has the energy `1 − p^{−2}`**: for `q = P(x) + x y` with `P(0) = 0`
and `r ≠ 0`, the energy of `ψ_r(q)` given `q` is `Var ψ_r(q) = 1 − |E ψ_r(q)|²`, and
`E ψ_r(q) = 1/p`. -/
theorem condEnergy_value (P : ZMod p → ZMod p) (hP : P 0 = 0) {r : ZMod p} (hr : r ≠ 0) :
    condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2)
        (fun ab => ZMod.stdAddChar (r * (P ab.1 + ab.1 * ab.2))) = 1 - (p : ℝ)⁻¹ ^ 2 := by
  have hc := condEnergy_comp (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2)
    fun s => ZMod.stdAddChar (r * s)
  beta_reduce at hc
  have h := meanSq_sub_eq_var_add
    (fun ab : ZMod p × ZMod p => ZMod.stdAddChar (r * (P ab.1 + ab.1 * ab.2))) 0
  have h1 : (meanSq fun ab : ZMod p × ZMod p =>
      ZMod.stdAddChar (r * (P ab.1 + ab.1 * ab.2)) - 0) = 1 := by
    simp only [meanSq, sub_zero, AddChar.norm_apply, one_pow, sum_const, card_univ, nsmul_eq_mul,
      mul_one]
    exact div_self (by exact_mod_cast Fintype.card_ne_zero)
  rw [mean_stdAddChar_poly_add_mul P hP hr, sub_zero, norm_inv, Complex.norm_natCast, h1] at h
  rw [hc]
  linarith

example := @condEnergy_value 7 ⟨Nat.prime_seven⟩ (fun x => x ^ 3) (by simp) 1 (by decide)

variable {τ : Type*} [Fintype τ] [DecidableEq τ]

/-- **The value of the first pair among `k = 1 + m` pairs**: its energy is `p^{−2m} (1 − p^{−2})`
up to a factor in `[1 − 1/p, 2]`, the scaling of every function of the first pair. -/
theorem condEnergy_copies_value (P : ZMod p → ZMod p) (hP : P 0 = 0) {r : ZMod p} (hr : r ≠ 0) :
    (1 - (p : ℝ)⁻¹) * ((p : ℝ)⁻¹ ^ Fintype.card τ) ^ 2 * (1 - (p : ℝ)⁻¹ ^ 2) ≤
        condEnergy (copiesLabel (τ := τ) P)
          (fun z => ZMod.stdAddChar (r * (P z.1.1 + z.1.1 * z.1.2))) ∧
      condEnergy (copiesLabel (τ := τ) P)
          (fun z => ZMod.stdAddChar (r * (P z.1.1 + z.1.1 * z.1.2))) ≤
        2 * ((p : ℝ)⁻¹ ^ Fintype.card τ) ^ 2 * (1 - (p : ℝ)⁻¹ ^ 2) := by
  have h := condEnergy_copies (τ := τ) P hP
    fun ab => ZMod.stdAddChar (r * (P ab.1 + ab.1 * ab.2))
  rw [condEnergy_value P hP hr] at h
  exact h

example := @condEnergy_copies_value 7 ⟨Nat.prime_seven⟩ (Fin 2) _ _ (fun x => x ^ 3) (by simp) 1
  (by decide)

/-- **The other pairs move a ratio of energies by at most `2/(1 − 1/p)`**: for any functions `h`,
`h'` of the first pair, `(1 − 1/p) E_k(h) E₁(h') ≤ 2 E₁(h) E_k(h')`, where `E₁` is the energy
given the first pair's label and `E_k` the energy given the label of all `k` pairs. -/
theorem condEnergy_copies_ratio (P : ZMod p → ZMod p) (hP : P 0 = 0)
    (h h' : ZMod p × ZMod p → ℂ) :
    (1 - (p : ℝ)⁻¹) * condEnergy (copiesLabel (τ := τ) P) (fun z => h z.1) *
        condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h' ≤
      2 * condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h *
        condEnergy (copiesLabel (τ := τ) P) (fun z => h' z.1) := by
  obtain ⟨-, hb⟩ := condEnergy_copies (τ := τ) P hP h
  obtain ⟨ha', -⟩ := condEnergy_copies (τ := τ) P hP h'
  have hp : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
  have hi : 0 ≤ 1 - (p : ℝ)⁻¹ := by
    rw [sub_nonneg]
    exact inv_le_one_of_one_le₀ (by linarith)
  have hE := condEnergy_nonneg (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h
  have hE' := condEnergy_nonneg (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h'
  calc (1 - (p : ℝ)⁻¹) * condEnergy (copiesLabel (τ := τ) P) (fun z => h z.1) *
        condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h'
      ≤ (1 - (p : ℝ)⁻¹) * (2 * ((p : ℝ)⁻¹ ^ Fintype.card τ) ^ 2 *
          condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h) *
        condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h' :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hb hi) hE'
    _ = 2 * condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h *
        ((1 - (p : ℝ)⁻¹) * ((p : ℝ)⁻¹ ^ Fintype.card τ) ^ 2 *
          condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h') := by ring
    _ ≤ 2 * condEnergy (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2) h *
        condEnergy (copiesLabel (τ := τ) P) (fun z => h' z.1) :=
        mul_le_mul_of_nonneg_left ha' (by linarith)

/-- The intermediate `ψ(x₁ y₁)` against the value `ψ(x₁³ + x₁ y₁)` among three pairs of
`x³ + x y` over `ℤ/7`. -/
example := @condEnergy_copies_ratio 7 ⟨Nat.prime_seven⟩ (Fin 2) _ _ (fun x => x ^ 3) (by simp)
  (fun ab => ZMod.stdAddChar (ab.1 * ab.2)) fun ab => ZMod.stdAddChar (ab.1 ^ 3 + ab.1 * ab.2)

end Transformer.Grokking.FlatSpectrum
