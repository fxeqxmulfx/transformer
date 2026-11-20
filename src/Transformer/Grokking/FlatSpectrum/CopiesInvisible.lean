module

public import Transformer.Grokking.FlatSpectrum.CopiesCovariance

/-!
# Theorem B for `k` copies: the other pairs scale every covariance by `p^{−m}`

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV6. For a prime `p` and
`q(x, y) = P(x) + x y` with `P(0) = 0`, such as `x³ + x y`, `E ψ_r(q) = 1/p` for every `r ≠ 0`:
for `x ≠ 0` the sum over `y` vanishes, and `x = 0` contributes `p`
(`mean_stdAddChar_poly_add_mul`). So `m` independent copies have `E ψ_r = p^{−m}`
(`mean_stdAddChar_copies`), and adding them to the label scales the covariance of every readout
with every function of the other pairs by exactly `p^{−m}` (`cov_comp_add_copies`; for `k`
copies of `x³ + x y` and any set `S` of pairs, `cov_copies_cube_add_mul`).

A remainder whose character means are all `ε` has the law `(1 − ε) U + ε δ₀`
(`mean_comp_of_mean_stdAddChar`, `mean_ite_eq_of_mean_stdAddChar`): one pair has
`P(q = 0) = (2p − 1)/p²`, and `k` pairs, with `ε = p^{−k}`, are closer to uniform.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {α β : Type*} [Fintype α] [Fintype β] {p : ℕ}

section Mixture

variable [NeZero p]

/-- **The law of a remainder with constant character means**: if `E ψ_r(w) = ε` for every
`r ≠ 0`, then `E φ(w) = (1 − ε) E φ + ε φ(0)`, the law `(1 − ε) U + ε δ₀`. -/
theorem mean_comp_of_mean_stdAddChar [Nonempty β] {ε : ℂ} {w : β → ZMod p}
    (hw : ∀ r ≠ 0, mean (fun b => ZMod.stdAddChar (r * w b)) = ε) (φ : ZMod p → ℂ) :
    mean (fun b => φ (w b)) = (1 - ε) * mean φ + ε * φ 0 := by
  have h0 := sum_fourierCoeff_mul φ 0
  have h1 := mean_comp_add φ w 0
  simp only [zero_add, mul_zero, AddChar.map_zero_eq_one, mul_one] at h0 h1
  have hsum : ∑ r ∈ univ.erase 0, fourierCoeff φ r * mean (fun b => ZMod.stdAddChar (r * w b)) =
      (∑ r ∈ univ.erase 0, fourierCoeff φ r) * ε := by
    rw [sum_mul]
    exact sum_congr rfl fun r hr => by rw [hw r (ne_of_mem_erase hr)]
  rw [h1, ← h0, ← add_sum_erase _ _ (mem_univ (0 : ZMod p)),
    ← add_sum_erase _ _ (mem_univ (0 : ZMod p)), hsum, fourierCoeff_zero]
  simp only [zero_mul, AddChar.map_zero_eq_one, mean_const]
  ring

example (φ : ZMod 5 → ℂ) := mean_comp_of_mean_stdAddChar (β := Unit) (w := fun _ => 0)
  (ε := 1) (fun r _ => by simp [mean]) φ

/-- The law itself: `P(w = t) = (1 − ε)/p + ε [t = 0]`. -/
theorem mean_ite_eq_of_mean_stdAddChar [Nonempty β] {ε : ℂ} {w : β → ZMod p}
    (hw : ∀ r ≠ 0, mean (fun b => ZMod.stdAddChar (r * w b)) = ε) (t : ZMod p) :
    mean (fun b => if w b = t then (1 : ℂ) else 0) =
      (1 - ε) / p + ε * if t = 0 then 1 else 0 := by
  rw [mean_comp_of_mean_stdAddChar hw fun s => if s = t then 1 else 0]
  have : mean (fun s : ZMod p => if s = t then (1 : ℂ) else 0) = 1 / p := by
    simp [mean, ZMod.card]
  rw [this]
  by_cases ht : t = 0
  · subst ht
    simp only [ite_true]
    ring
  · simp only [Ne.symm ht, ht, ite_false]
    ring

/-- A constant remainder is the point mass at `0`. -/
example (t : ZMod 5) :
    mean (fun _ : Unit => if (0 : ZMod 5) = t then (1 : ℂ) else 0) =
      (1 - 1) / 5 + 1 * if t = 0 then 1 else 0 :=
  mean_ite_eq_of_mean_stdAddChar (w := fun _ => 0) (fun r _ => by simp [mean]) t

end Mixture

variable [Fact p.Prime]

/-- **One pair has `E ψ_r(q) = 1/p`**: for `q(x, y) = P(x) + x y` with `P(0) = 0` and `r ≠ 0`,
the sum over `y` vanishes for `x ≠ 0`, and `x = 0` contributes `p`. -/
theorem mean_stdAddChar_poly_add_mul (P : ZMod p → ZMod p) (hP : P 0 = 0) {r : ZMod p}
    (hr : r ≠ 0) :
    mean (fun z : ZMod p × ZMod p => ZMod.stdAddChar (r * (P z.1 + z.1 * z.2))) =
      (p : ℂ)⁻¹ := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (Fact.out : p.Prime).ne_zero
  have hrow (x : ZMod p) : ∑ y, ZMod.stdAddChar (r * (P x + x * y)) =
      if x = 0 then (p : ℂ) else 0 := by
    have (y : ZMod p) : r * (P x + x * y) = r * P x + y * (r * x) := by ring
    simp only [this, AddChar.map_add_eq_mul, ← mul_sum, sum_stdAddChar_mul, mul_eq_zero, hr,
      false_or]
    split_ifs with hx
    · simp [hx, hP]
    · simp
  rw [mean, Fintype.sum_prod_type]
  simp only [hrow, sum_ite_eq', mem_univ, ite_true, Fintype.card_prod, ZMod.card, Nat.cast_mul]
  field_simp

example := @mean_stdAddChar_poly_add_mul 7 ⟨Nat.prime_seven⟩ (fun x => x ^ 3) (by simp) 1
  (by decide)

/-- **`m` independent copies**: `E ψ_r(∑_j (P(x_j) + x_j y_j)) = p^{−m}` for every `r ≠ 0`. -/
theorem mean_stdAddChar_copies (P : ZMod p → ZMod p) (hP : P 0 = 0) {τ : Type*} [Fintype τ]
    [DecidableEq τ] {r : ZMod p} (hr : r ≠ 0) :
    mean (fun z : τ → ZMod p × ZMod p =>
      ZMod.stdAddChar (r * ∑ j, (P (z j).1 + (z j).1 * (z j).2))) =
      (p : ℂ)⁻¹ ^ Fintype.card τ := by
  have := mean_stdAddChar_sum_pi (X := fun _ : τ => ZMod p × ZMod p)
    (fun _ z => P z.1 + z.1 * z.2) r
  simp only [mean_stdAddChar_poly_add_mul P hP hr, prod_const, card_univ] at this
  exact this

example := @mean_stdAddChar_copies 7 ⟨Nat.prime_seven⟩ (fun x => x ^ 3) (by simp) (Fin 3) _ _ 1
  (by decide)

/-- **NV6 for `P(x) + x y`**: adding `m` independent copies of `P(x) + x y` with `P(0) = 0` to
the label scales the covariance of every readout `g` with every `h` of the other blocks by
exactly `p^{−m}`. -/
theorem cov_comp_add_copies (P : ZMod p → ZMod p) (hP : P 0 = 0) {τ : Type*} [Fintype τ]
    [DecidableEq τ] (g : ZMod p → ℂ) (u : α → ZMod p) (h : α → ℂ) :
    cov (fun az : α × (τ → ZMod p × ZMod p) =>
        g (u az.1 + ∑ j, (P (az.2 j).1 + (az.2 j).1 * (az.2 j).2))) (fun az => h az.1) =
      (p : ℂ)⁻¹ ^ Fintype.card τ * cov (fun a => g (u a)) h :=
  cov_comp_add_of_mean_stdAddChar (fun _ hr => mean_stdAddChar_copies P hP hr) g u h

example := @cov_comp_add_copies Unit _ 7 ⟨Nat.prime_seven⟩ (fun x => x ^ 3) (by simp) (Fin 2) _ _

/-- **NV6 for `k` copies of `x³ + x y`**: for the label `f = ∑ᵢ (xᵢ³ + xᵢ yᵢ)`, a set `S` of the
pairs (indexed by `σ`) and the other `m` pairs (indexed by `τ`), every readout `g` and every `h`
of the pairs in `S`, `Cov(g(f), h(z_S)) = p^{−m} Cov(g(f_S), h)`, where `f_S` is the label of
the pairs in `S` alone. -/
theorem cov_copies_cube_add_mul {σ τ : Type*} [Fintype σ] [DecidableEq σ] [Fintype τ]
    [DecidableEq τ] (g : ZMod p → ℂ) (h : (σ → ZMod p × ZMod p) → ℂ) :
    cov (fun z : (σ → ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
        g (∑ i, ((z.1 i).1 ^ 3 + (z.1 i).1 * (z.1 i).2) +
          ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2))) (fun z => h z.1) =
      (p : ℂ)⁻¹ ^ Fintype.card τ *
        cov (fun zS => g (∑ i, ((zS i).1 ^ 3 + (zS i).1 * (zS i).2))) h :=
  cov_comp_add_copies (fun x => x ^ 3) (zero_pow three_ne_zero) g
    (fun zS : σ → ZMod p × ZMod p => ∑ i, ((zS i).1 ^ 3 + (zS i).1 * (zS i).2)) h

/-- One pair is in `S` and two are not, at `p = 7`: the factor is `7⁻²`. -/
example (g : ZMod 7 → ℂ) (h : (Fin 1 → ZMod 7 × ZMod 7) → ℂ) :
    cov (fun z : (Fin 1 → ZMod 7 × ZMod 7) × (Fin 2 → ZMod 7 × ZMod 7) =>
        g (∑ i, ((z.1 i).1 ^ 3 + (z.1 i).1 * (z.1 i).2) +
          ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2))) (fun z => h z.1) =
      (7 : ℂ)⁻¹ ^ 2 * cov (fun zS => g (∑ i, ((zS i).1 ^ 3 + (zS i).1 * (zS i).2))) h := by
  have := @cov_copies_cube_add_mul 7 ⟨Nat.prime_seven⟩ (Fin 1) (Fin 2) _ _ _ _ g h
  simpa using this

end Transformer.Grokking.FlatSpectrum
