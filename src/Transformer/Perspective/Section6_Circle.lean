/-
# §7 — Dynamics on the circle

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §7 of the survey:

* `eq:onangles`        — angular form of `USA` on `𝕊^1`,
* `e:kuramoto`         — Kuramoto model,
* `e:energykuramoto`   — Kuramoto energy `𝖥(θ)`,
* `eq:energyF`         — interaction energy `𝖤_β` on the torus,
* `e:variantkuramoto`  — generalized Kuramoto with non-linearity `h`,
* and the identification of `eq:onangles` with the gradient flow of `𝖤_β`.

Open Problem `o:strictsaddle` of §7 — apart from the global maxima, every
critical point of `𝖤_β` on `𝕋^n` is a strict saddle — is in
`Perspective.StrictSaddle`, which needs Appendix B's second derivative and so
comes after this file.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Add

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- Tuple of `n` angles on the torus `𝕋 = ℝ/2π ℤ`. -/
abbrev Angles (n : ℕ) : Type := Idx n → ℝ

/-- **Equation (eq:onangles).** Angular form of `USA` on the circle:

  `θ̇_i(t) = -(1/n) Σ_j exp(β cos(θ_i - θ_j)) sin(θ_i - θ_j)`. -/
def angularUSA (β : ℝ) (θ : ℝ → Angles n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => θ s i)
      (-(1 / (n : ℝ)) *
        ∑ j : Idx n,
          Real.exp (β * Real.cos (θ t i - θ t j)) * Real.sin (θ t i - θ t j)) t

/-- The integrand kernel `h_β(θ) = e^{β cos θ}`. -/
noncomputable def h_β (β θ : ℝ) : ℝ := Real.exp (β * Real.cos θ)

/-- **Equation (e:kuramoto).** Classical Kuramoto model:

  `θ̇_i(t) = ω_i + (K/n) Σ_j sin(θ_j - θ_i)`. -/
def kuramoto (K : ℝ) (ω : Idx n → ℝ) (θ : ℝ → Angles n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => θ s i)
      (ω i + (K / (n : ℝ)) *
        ∑ j : Idx n, Real.sin (θ t j - θ t i)) t

/-- **Equation (e:energykuramoto).** Kuramoto energy:

  `𝖥(θ) = (K/(2n²)) Σ_i Σ_j cos(θ_i - θ_j)`. -/
noncomputable def kuramotoEnergy (K : ℝ) (θ : Angles n) : ℝ :=
  (K / (2 * (n : ℝ)^2)) *
    ∑ i : Idx n, ∑ j : Idx n, Real.cos (θ i - θ j)

/-- **Equation (eq:energyF).** Toroidal interaction energy:

  `𝖤_β(θ) = (1/(2 β n²)) Σ_i Σ_j exp(β cos(θ_i - θ_j))`. -/
noncomputable def torusEnergy (β : ℝ) (θ : Angles n) : ℝ :=
  (1 / (2 * β * (n : ℝ)^2)) *
    ∑ i : Idx n, ∑ j : Idx n, Real.exp (β * Real.cos (θ i - θ j))

/-- **The `i`-th partial derivative of the toroidal energy.**

Differentiating `exp(β cos(θ_a - θ_b))` in `θ_i` touches the `a = i` row and
the `b = i` column.  The kernel is even and its derivative odd, so the two
contribute the same thing, and the factor `2β` they produce cancels the
`1/(2β n²)` in front, leaving `n⁻²  Σ_j exp(β cos(θ_j - θ_i)) sin(θ_j - θ_i)`. -/
theorem hasDerivAt_torusEnergy (β : ℝ) (hβ : β ≠ 0) (θ : Angles n) (i : Idx n) :
    HasDerivAt (fun x : ℝ => torusEnergy n β (Function.update θ i x))
      ((1 / (n : ℝ)^2) *
        ∑ j : Idx n, Real.exp (β * Real.cos (θ j - θ i)) * Real.sin (θ j - θ i))
      (θ i) := by
  have hupd : ∀ a : Idx n,
      HasDerivAt (fun x : ℝ => Function.update θ i x a) (if a = i then (1 : ℝ) else 0)
        (θ i) := by
    intro a
    by_cases h : a = i
    · subst h; simpa using hasDerivAt_id' (x := θ a)
    · simpa [Function.update_of_ne h, h] using hasDerivAt_const (θ i) (θ a)
  have hterm : ∀ a b : Idx n,
      HasDerivAt (fun x : ℝ =>
          Real.exp (β * Real.cos (Function.update θ i x a - Function.update θ i x b)))
        (Real.exp (β * Real.cos (θ a - θ b)) * (β * -Real.sin (θ a - θ b)) *
            (if a = i then (1 : ℝ) else 0)
          - Real.exp (β * Real.cos (θ a - θ b)) * (β * -Real.sin (θ a - θ b)) *
            (if b = i then (1 : ℝ) else 0)) (θ i) := by
    intro a b
    have h := (((hupd a).sub (hupd b)).cos.const_mul β).exp
    simp only [Pi.sub_apply, Function.update_eq_self] at h
    convert h using 1
    ring
  have hsum := (HasDerivAt.sum fun a (_ : a ∈ Finset.univ) =>
    HasDerivAt.sum fun b (_ : b ∈ Finset.univ) => hterm a b).const_mul
      (1 / (2 * β * (n : ℝ)^2))
  simp only [torusEnergy]
  convert hsum using 1
  · funext x
    simp [Finset.sum_apply]
  simp only [Finset.sum_sub_distrib, mul_ite, mul_one, mul_zero,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [Finset.sum_comm]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  have hodd : ∀ j : Idx n,
      Real.exp (β * Real.cos (θ i - θ j)) * (β * -Real.sin (θ i - θ j))
        = β * (Real.exp (β * Real.cos (θ j - θ i)) * Real.sin (θ j - θ i)) := by
    intro j
    rw [show θ i - θ j = -(θ j - θ i) by ring, Real.cos_neg, Real.sin_neg]
    ring
  have hodd' : ∀ j : Idx n,
      Real.exp (β * Real.cos (θ j - θ i)) * (β * -Real.sin (θ j - θ i))
        = -(β * (Real.exp (β * Real.cos (θ j - θ i)) * Real.sin (θ j - θ i))) := by
    intro j; ring
  rw [Finset.sum_congr rfl fun j _ => hodd j, Finset.sum_congr rfl fun j _ => hodd' j]
  rw [← Finset.mul_sum, Finset.sum_neg_distrib, ← Finset.mul_sum]
  field_simp
  ring

/-- **The dynamics `eq:onangles` is the gradient flow** `θ̇(t) = n ∇ 𝖤_β(θ(t))`.

The right-hand side of `angularUSA` is exactly `n` times the partial derivative
computed above, which is the sense in which §7 calls the angular model a
gradient ascent on the toroidal energy. -/
theorem angularUSA_is_gradient_flow
    (β : ℝ) (hβ : β ≠ 0) (hn : (n : ℝ) ≠ 0) (θ : ℝ → Angles n)
    (hθ : angularUSA n β θ) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => θ s i)
      ((n : ℝ) * deriv (fun x : ℝ => torusEnergy n β (Function.update (θ t) i x))
        (θ t i)) t := by
  have hgrad := (hasDerivAt_torusEnergy n β hβ (θ t) i).deriv
  rw [hgrad]
  have h := hθ t i
  convert h using 1
  have hodd : ∀ j : Idx n,
      Real.exp (β * Real.cos (θ t j - θ t i)) * Real.sin (θ t j - θ t i)
        = -(Real.exp (β * Real.cos (θ t i - θ t j)) * Real.sin (θ t i - θ t j)) := by
    intro j
    rw [show θ t j - θ t i = -(θ t i - θ t j) by ring, Real.cos_neg, Real.sin_neg]
    ring
  rw [Finset.sum_congr rfl fun j _ => hodd j, Finset.sum_neg_distrib]
  field_simp

/-- **Equation (e:variantkuramoto).** *Generalized Kuramoto*:

  `θ̇_i(t) = ω_i + (K/n) Σ_j h(θ_j - θ_i)`. -/
def variantKuramoto
    (K : ℝ) (ω : Idx n → ℝ) (h : ℝ → ℝ) (θ : ℝ → Angles n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => θ s i)
      (ω i + (K / (n : ℝ)) *
        ∑ j : Idx n, h (θ t j - θ t i)) t

/-- **The kernel is even**, which is the whole content of the Bessel expansion
`h_β(θ) = Σ_{k ∈ ℤ} I_k(β) e^{i k θ}` that §7 uses: because `h_β` is even, only
cosines survive and the coefficients are real.  The modified Bessel functions
`I_k` themselves are not in Mathlib and the expansion is not formalized here. -/
theorem h_β_neg (β θ : ℝ) : h_β β (-θ) = h_β β θ := by
  rw [h_β, h_β, Real.cos_neg]

/-- **And it is strictly positive.**  The interaction the kernel defines is
therefore attractive at every angle, which is what makes `torusEnergy` a sum of
positive terms. -/
theorem h_β_pos (β θ : ℝ) : 0 < h_β β θ := Real.exp_pos _

/-! ### The hypotheses are satisfiable -/

/-- `hasDerivAt_torusEnergy` and `angularUSA_is_gradient_flow` ask for `β ≠ 0`,
for `n ≠ 0` and for a solution of `eq:onangles`.  A single token is one: with
`n = 1` the only interaction is a token with itself, `sin 0 = 0`, so the angle
stands still and the constant curve solves the dynamics. -/
example : angularUSA 1 1 (fun _ _ => (0 : ℝ)) ∧ (1 : ℝ) ≠ 0 ∧ ((1 : ℕ) : ℝ) ≠ 0 := by
  refine ⟨fun t i => ?_, one_ne_zero, by norm_num⟩
  simpa using hasDerivAt_const t (0 : ℝ)

end Perspective
end Transformer
