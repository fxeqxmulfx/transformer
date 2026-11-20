module

public import Transformer.Grokking.FlatSpectrum.Conditional
public import Transformer.Grokking.FlatSpectrum.BilinearKernel

/-!
# The covariance of a readout of a sum with one of its independent parts

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV6. The label is
`f = u(a) + w(b)` for independent uniform `a` and `b`, a readout `g(f)` of it is compared with a
function `h(a)` of one part, and `Cov(A, B) = E[A conj B] − E A conj(E B)`. Expanding
`g = ∑_r ĝ(r) ψ_r` (`sum_fourierCoeff_mul`) and averaging over `b` first gives

    Cov(g(f), h(a)) = ∑_{r ≠ 0} ĝ(r) · E ψ_r(w(b)) · Cov(ψ_r(u(a)), h(a))

(`cov_comp_add_eq_sum`). For independent blocks, `w = ∑_j q_j(z_j)`, the middle factor is
`∏_j E ψ_r(q_j(z_j))` (`mean_stdAddChar_sum_pi`, `cov_comp_add_sum_pi`). When every
`E ψ_r(w)` with `r ≠ 0` is one number `ε`, the covariance is that of the task without `w`,
scaled by exactly `ε` (`cov_comp_add_of_mean_stdAddChar`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {α β : Type*} [Fintype α] [Fintype β]

/-- The covariance `Cov(A, B) = E[A conj B] − E A conj(E B)` on a finite uniform space,
`review/flat_spectrum.md`, NV6. -/
noncomputable def cov (A B : α → ℂ) : ℂ :=
  mean (fun x => A x * star (B x)) - mean A * star (mean B)

lemma mean_add (A B : α → ℂ) : mean (fun x => A x + B x) = mean A + mean B := by
  simp only [mean, sum_add_distrib, add_div]

lemma mean_sub (A B : α → ℂ) : mean (fun x => A x - B x) = mean A - mean B := by
  simp only [mean, sum_sub_distrib, sub_div]

lemma mean_const_mul (c : ℂ) (A : α → ℂ) : mean (fun x => c * A x) = c * mean A := by
  simp only [mean, ← mul_sum, mul_div_assoc]

lemma mean_mul_const (c : ℂ) (A : α → ℂ) : mean (fun x => A x * c) = mean A * c := by
  simp only [mean, ← sum_mul, div_mul_eq_mul_div]

lemma mean_sum {ι : Type*} (s : Finset ι) (A : ι → α → ℂ) :
    mean (fun x => ∑ i ∈ s, A i x) = ∑ i ∈ s, mean (A i) := by
  simp only [mean, ← sum_div]
  rw [sum_comm]

lemma mean_const [Nonempty α] (c : ℂ) : mean (fun _ : α => c) = c := by
  rw [mean, sum_const, card_univ, nsmul_eq_mul, mul_div_cancel_left₀]
  exact_mod_cast Fintype.card_ne_zero

lemma mean_star (A : α → ℂ) : mean (fun x => star (A x)) = star (mean A) := by
  rw [mean, mean, star_div₀, star_sum, star_natCast]

/-- The mean over `α × β` is the mean over `α` of the means over `β`. -/
lemma mean_prod (F : α × β → ℂ) : mean F = mean fun a => mean fun b => F (a, b) := by
  simp only [mean, Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul, ← sum_div, div_div,
    mul_comm]

/-- Independent factors: `E[A(a) B(b)] = E A · E B`. -/
lemma mean_mul_prod (A : α → ℂ) (B : β → ℂ) :
    mean (fun ab : α × β => A ab.1 * B ab.2) = mean A * mean B := by
  rw [mean_prod]
  simp only [mean_const_mul, mean_mul_const]

/-- `Cov(A, B) = E[(A − E A) conj B]`. -/
lemma cov_eq_mean_sub_mul (A B : α → ℂ) :
    cov A B = mean fun x => (A x - mean A) * star (B x) := by
  simp only [cov, sub_mul, mean_sub, mean_const_mul, mean_star]

lemma cov_const_left (c : ℂ) (B : α → ℂ) : cov (fun _ => c) B = 0 := by
  cases isEmpty_or_nonempty α
  · simp [cov, mean]
  · rw [cov, mean_const_mul, mean_star, mean_const, sub_self]

lemma cov_sum_left {ι : Type*} (s : Finset ι) (c : ι → ℂ) (A : ι → α → ℂ) (B : α → ℂ) :
    cov (fun x => ∑ i ∈ s, c i * A i x) B = ∑ i ∈ s, c i * cov (A i) B := by
  simp only [cov, sum_mul, mean_sum, mul_assoc, mean_const_mul, mul_sub, sum_sub_distrib]

/-- **Averaging out an independent part**: `Cov(F(a, b), h(a)) = Cov(E_b F(a, b), h(a))`. -/
lemma cov_prod_fst [Nonempty β] (F : α × β → ℂ) (h : α → ℂ) :
    cov F (fun ab => h ab.1) = cov (fun a => mean fun b => F (a, b)) h := by
  rw [cov, cov, mean_prod F, mean_prod fun ab : α × β => h ab.1,
    mean_prod fun ab : α × β => F ab * star (h ab.1)]
  simp only [mean_mul_const, mean_const]

variable {p : ℕ} [NeZero p]

/-- The Fourier coefficient `ĝ(r) = E_s g(s) ψ(−r s)` on `ℤ/p`, so that `g = ∑_r ĝ(r) ψ_r`
(`sum_fourierCoeff_mul`). -/
noncomputable def fourierCoeff (g : ZMod p → ℂ) (r : ZMod p) : ℂ :=
  mean fun s => g s * ZMod.stdAddChar (-(r * s))

lemma fourierCoeff_zero (g : ZMod p → ℂ) : fourierCoeff g 0 = mean g := by
  simp [fourierCoeff]

/-- **Fourier inversion** on `ℤ/p`: `g(t) = ∑_r ĝ(r) ψ(r t)`. -/
lemma sum_fourierCoeff_mul (g : ZMod p → ℂ) (t : ZMod p) :
    ∑ r, fourierCoeff g r * ZMod.stdAddChar (r * t) = g t := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne p)
  have h (r s : ZMod p) : g s * ZMod.stdAddChar (-(r * s)) / p * ZMod.stdAddChar (r * t) =
      g s * ZMod.stdAddChar (r * (t - s)) / p := by
    rw [show r * (t - s) = -(r * s) + r * t by ring, AddChar.map_add_eq_mul]
    ring
  simp only [fourierCoeff, mean, ZMod.card, sum_div, sum_mul, h]
  rw [sum_comm]
  simp only [← sum_div, ← mul_sum, sum_stdAddChar_mul, sub_eq_zero, mul_ite, mul_zero,
    sum_ite_eq, mem_univ, ite_true]
  field_simp

/-- A readout `∑_r c_r ψ_r(u)` of `u` has the covariance `∑_{r ≠ 0} c_r Cov(ψ_r(u), h)` with `h`:
the constant term drops out. -/
lemma cov_sum_stdAddChar (c : ZMod p → ℂ) (u : α → ZMod p) (h : α → ℂ) :
    cov (fun a => ∑ r, c r * ZMod.stdAddChar (r * u a)) h =
      ∑ r ∈ univ.erase 0, c r * cov (fun a => ZMod.stdAddChar (r * u a)) h := by
  rw [cov_sum_left, sum_erase]
  simp only [zero_mul, AddChar.map_zero_eq_one, cov_const_left, mul_zero]

/-- `Cov(g(u), h) = ∑_{r ≠ 0} ĝ(r) Cov(ψ_r(u), h)`. -/
lemma cov_comp_eq_sum (g : ZMod p → ℂ) (u : α → ZMod p) (h : α → ℂ) :
    cov (fun a => g (u a)) h =
      ∑ r ∈ univ.erase 0, fourierCoeff g r * cov (fun a => ZMod.stdAddChar (r * u a)) h := by
  rw [← cov_sum_stdAddChar]
  simp only [sum_fourierCoeff_mul]

/-- Averaging a readout of `t + w` over `w`: `E_b g(t + w(b)) = ∑_r ĝ(r) E ψ_r(w) ψ_r(t)`. -/
lemma mean_comp_add (g : ZMod p → ℂ) (w : β → ZMod p) (t : ZMod p) :
    mean (fun b => g (t + w b)) = ∑ r, fourierCoeff g r *
      mean (fun b => ZMod.stdAddChar (r * w b)) * ZMod.stdAddChar (r * t) := by
  have (b : β) : g (t + w b) = ∑ r, fourierCoeff g r * ZMod.stdAddChar (r * t) *
      ZMod.stdAddChar (r * w b) := by
    rw [← sum_fourierCoeff_mul g (t + w b)]
    simp only [mul_add, AddChar.map_add_eq_mul, mul_assoc]
  simp only [this, mean_sum, mean_const_mul]
  exact sum_congr rfl fun r _ => by ring

/-- **NV6, the identity**: for the label `u(a) + w(b)` of independent uniform `a` and `b`, any
readout `g` and any `h` of `a`, `Cov(g(u + w), h) = ∑_{r ≠ 0} ĝ(r) E ψ_r(w) Cov(ψ_r(u), h)`. -/
theorem cov_comp_add_eq_sum [Nonempty β] (g : ZMod p → ℂ) (u : α → ZMod p) (w : β → ZMod p)
    (h : α → ℂ) :
    cov (fun ab : α × β => g (u ab.1 + w ab.2)) (fun ab => h ab.1) =
      ∑ r ∈ univ.erase 0, fourierCoeff g r * mean (fun b => ZMod.stdAddChar (r * w b)) *
        cov (fun a => ZMod.stdAddChar (r * u a)) h := by
  rw [cov_prod_fst, ← cov_sum_stdAddChar]
  simp only [mean_comp_add]

example (g : ZMod 5 → ℂ) (h : ZMod 5 → ℂ) :
    cov (fun ab : ZMod 5 × ZMod 5 => g (ab.1 + ab.2)) (fun ab => h ab.1) =
      ∑ r ∈ univ.erase 0, fourierCoeff g r * mean (fun b : ZMod 5 => ZMod.stdAddChar (r * b)) *
        cov (fun a => ZMod.stdAddChar (r * a)) h :=
  cov_comp_add_eq_sum g id id h

section Blocks

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {X : ι → Type*} [∀ j, Fintype (X j)]

/-- **Independent blocks multiply the means of characters**:
`E ψ_r(∑_j q_j(z_j)) = ∏_j E ψ_r(q_j(z_j))` for independent uniform blocks `z_j`. -/
lemma mean_stdAddChar_sum_pi (q : ∀ j, X j → ZMod p) (r : ZMod p) :
    mean (fun z : ∀ j, X j => ZMod.stdAddChar (r * ∑ j, q j (z j))) =
      ∏ j, mean fun x => ZMod.stdAddChar (r * q j x) := by
  simp only [mean, mul_sum, stdAddChar_sum, Fintype.card_pi, Nat.cast_prod, prod_div_distrib]
  rw [Fintype.prod_sum]

/-- **NV6 for blocks**: for the label `u(a) + ∑_j q_j(z_j)` of independent uniform blocks and any
`h` of `a`, `Cov(g(f), h) = ∑_{r ≠ 0} ĝ(r) ∏_j E ψ_r(q_j(z_j)) Cov(ψ_r(u), h)`. -/
theorem cov_comp_add_sum_pi [∀ j, Nonempty (X j)] (g : ZMod p → ℂ) (u : α → ZMod p)
    (q : ∀ j, X j → ZMod p) (h : α → ℂ) :
    cov (fun az : α × (∀ j, X j) => g (u az.1 + ∑ j, q j (az.2 j))) (fun az => h az.1) =
      ∑ r ∈ univ.erase 0, fourierCoeff g r * (∏ j, mean fun x => ZMod.stdAddChar (r * q j x)) *
        cov (fun a => ZMod.stdAddChar (r * u a)) h := by
  rw [cov_comp_add_eq_sum g u (fun z : ∀ j, X j => ∑ j, q j (z j)) h]
  simp only [mean_stdAddChar_sum_pi]

end Blocks

/-- **NV6, the exact ratio**: if every `E ψ_r(w)` with `r ≠ 0` equals `ε`, then adding `w` to the
label scales the covariance of every readout with every `h` of `a` by exactly `ε`. -/
theorem cov_comp_add_of_mean_stdAddChar [Nonempty β] {ε : ℂ} {w : β → ZMod p}
    (hw : ∀ r ≠ 0, mean (fun b => ZMod.stdAddChar (r * w b)) = ε) (g : ZMod p → ℂ)
    (u : α → ZMod p) (h : α → ℂ) :
    cov (fun ab : α × β => g (u ab.1 + w ab.2)) (fun ab => h ab.1) =
      ε * cov (fun a => g (u a)) h := by
  rw [cov_comp_add_eq_sum, cov_comp_eq_sum, mul_sum]
  refine sum_congr rfl fun r hr => ?_
  rw [hw r (ne_of_mem_erase hr)]
  ring

/-- A constant remainder `w = 0` has `ε = 1`. -/
example (g : ZMod 5 → ℂ) (u : ZMod 5 → ZMod 5) (h : ZMod 5 → ℂ) :
    cov (fun ab : ZMod 5 × Unit => g (u ab.1 + 0)) (fun ab => h ab.1) =
      1 * cov (fun a => g (u a)) h :=
  cov_comp_add_of_mean_stdAddChar (w := fun _ => 0) (fun r _ => by simp [mean]) g u h

end Transformer.Grokking.FlatSpectrum
