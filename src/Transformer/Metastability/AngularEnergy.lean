/-
# Metastability — the angular energy on `𝕋^n` and its gradient

On the circle the configuration `x_i = (cos θ_i, sin θ_i)` turns the
interaction energy `Metastability.Eβ` into a function of the angles alone,

  `𝖤_β(Θ) = (1 / (2 β e^β n²)) Σ_i Σ_j exp(β cos(θ_i - θ_j))`,

because `⟨x_i, x_j⟩ = cos(θ_i - θ_j)`.  This file is that function and its
partial derivatives: `hasDerivAt_angularEβ` proves that `∂_{θ_ℓ} 𝖤_β` is

  `-(1/n²) Σ_m sin(θ_ℓ - θ_m) e^{β (cos(θ_ℓ - θ_m) - 1)}`,

which is the expression §3.2 of arXiv:2410.06833v1 writes on the right-hand
side of the PL inequality `lem: PL.borjan`.  The two `e^{-β}`, one from the
normalization of `𝖤_β` and one from the exponent, cancel against the `2 β`.
-/

import Transformer.Basic
import Transformer.Metastability.Basic
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

/-- The point `(cos θ, sin θ)` of the circle `𝕊^1`. -/
noncomputable def circlePoint (θ : ℝ) : SSphere 2 :=
  ⟨(EuclideanSpace.equiv (Fin 2) ℝ).symm ![Real.cos θ, Real.sin θ], by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq]
    simp only [Fin.sum_univ_two, Real.norm_eq_abs, sq_abs]
    norm_num [Real.sin_sq_add_cos_sq]⟩

/-- `⟨(cos θ, sin θ), (cos φ, sin φ)⟩ = cos(θ - φ)`: the inner product of two
circle points is the cosine of the angle between them. -/
theorem inner_circlePoint (θ φ : ℝ) :
    inner (𝕜 := ℝ) ((circlePoint θ : EucSpace 2)) ((circlePoint φ : EucSpace 2))
      = Real.cos (θ - φ) := by
  simp only [circlePoint, PiLp.inner_apply, RCLike.inner_apply, conj_trivial,
    Fin.sum_univ_two, Real.cos_sub]
  simp [Matrix.cons_val_zero, Matrix.cons_val_one]
  ring

/-- **The interaction energy in angular coordinates.**

`Metastability.Eβ` evaluated at `x_i = (cos θ_i, sin θ_i)`, where
`⟨x_i, x_j⟩ = cos(θ_i - θ_j)`.  Source: arXiv:2410.06833v1, §3.2. -/
noncomputable def angularEβ (n : ℕ) (β : ℝ) (Θ : Idx n → ℝ) : ℝ :=
  (1 / (2 * β * Real.exp β * (n : ℝ)^2)) *
    ∑ i : Idx n, ∑ j : Idx n, Real.exp (β * Real.cos (Θ i - Θ j))

/-- **The `ℓ`-th partial derivative of `angularEβ`**, as §3.2 writes it:

  `∂_{θ_ℓ} 𝖤_β(Θ) = -(1/n²) Σ_m sin(θ_ℓ - θ_m) e^{β (cos(θ_ℓ - θ_m) - 1)}`.

That this *is* the derivative is `hasDerivAt_angularEβ`. -/
noncomputable def angularGrad (n : ℕ) (β : ℝ) (Θ : Idx n → ℝ) (l : Idx n) : ℝ :=
  -((1 / (n : ℝ)^2) *
    ∑ m : Idx n, Real.sin (Θ l - Θ m)
      * Real.exp (β * (Real.cos (Θ l - Θ m) - 1)))

/-- Moving the first argument of one interaction term. -/
theorem hasDerivAt_pairFst (β a x : ℝ) :
    HasDerivAt (fun s : ℝ => Real.exp (β * Real.cos (s - a)))
      (-(β * Real.sin (x - a) * Real.exp (β * Real.cos (x - a)))) x := by
  have h1 : HasDerivAt (fun s : ℝ => s - a) 1 x := (hasDerivAt_id x).sub_const a
  have h2 : HasDerivAt (fun s : ℝ => Real.cos (s - a)) (-Real.sin (x - a) * 1) x := h1.cos
  simpa [mul_comm, mul_left_comm, mul_assoc] using ((h2.const_mul β).exp)

/-- Moving the second argument of one interaction term. -/
theorem hasDerivAt_pairSnd (β a x : ℝ) :
    HasDerivAt (fun s : ℝ => Real.exp (β * Real.cos (a - s)))
      (β * Real.sin (a - x) * Real.exp (β * Real.cos (a - x))) x := by
  have h1 : HasDerivAt (fun s : ℝ => a - s) (-1) x :=
    HasDerivAt.const_sub a (hasDerivAt_id x)
  have h2 : HasDerivAt (fun s : ℝ => Real.cos (a - s)) (-Real.sin (a - x) * -1) x := h1.cos
  simpa [mul_comm, mul_left_comm, mul_assoc] using ((h2.const_mul β).exp)

/-- **`angularGrad` is the partial derivative of `angularEβ`.**

Differentiating `𝖤_β` in `θ_ℓ` touches the terms with `i = ℓ` and those with
`j = ℓ`; the two families are equal, which is where the factor `2` cancelling
the `2 β` of the normalization comes from.  Source: arXiv:2410.06833v1, §3.2
(the gradient in `lem: PL.borjan`). -/
theorem hasDerivAt_angularEβ
    (n : ℕ) (β : ℝ) (hβ : β ≠ 0) (Θ : Idx n → ℝ) (l : Idx n) :
    HasDerivAt (fun s : ℝ => angularEβ n β (Function.update Θ l s))
      (angularGrad n β Θ l) (Θ l) := by
  classical
  set a : Idx n → ℝ := fun j =>
    -(β * Real.sin (Θ l - Θ j) * Real.exp (β * Real.cos (Θ l - Θ j))) with ha
  set b : Idx n → ℝ := fun i =>
    β * Real.sin (Θ i - Θ l) * Real.exp (β * Real.cos (Θ i - Θ l)) with hb
  have hterm : ∀ i j : Idx n,
      HasDerivAt
        (fun s : ℝ => Real.exp (β * Real.cos
          (Function.update Θ l s i - Function.update Θ l s j)))
        ((if i = l then a j else 0) + (if j = l then b i else 0)) (Θ l) := by
    intro i j
    rcases eq_or_ne i l with hi | hi <;> rcases eq_or_ne j l with hj | hj
    · simpa [hi, hj, ha, hb] using
        (hasDerivAt_const (Θ l) (Real.exp β))
    · simpa [hi, hj, ha, hb, Function.update_of_ne hj] using
        hasDerivAt_pairFst β (Θ j) (Θ l)
    · simpa [hi, hj, ha, hb, Function.update_of_ne hi] using
        hasDerivAt_pairSnd β (Θ i) (Θ l)
    · simpa [hi, hj, Function.update_of_ne hi, Function.update_of_ne hj] using
        (hasDerivAt_const (Θ l) (Real.exp (β * Real.cos (Θ i - Θ j))))
  have hrow : ∀ i : Idx n, HasDerivAt
      (fun s : ℝ => ∑ j : Idx n,
        Real.exp (β * Real.cos
          (Function.update Θ l s i - Function.update Θ l s j)))
      (∑ j : Idx n, ((if i = l then a j else 0) + (if j = l then b i else 0)))
      (Θ l) := by
    intro i
    have h := HasDerivAt.sum (u := (Finset.univ : Finset (Idx n)))
      (A := fun j (s : ℝ) => Real.exp (β * Real.cos
        (Function.update Θ l s i - Function.update Θ l s j)))
      (fun j _ => hterm i j)
    simpa [Finset.sum_fn] using h
  have hsum : HasDerivAt
      (fun s : ℝ => ∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * Real.cos
          (Function.update Θ l s i - Function.update Θ l s j)))
      (∑ i : Idx n, ∑ j : Idx n,
        ((if i = l then a j else 0) + (if j = l then b i else 0))) (Θ l) :=
    by
    have h := HasDerivAt.sum (u := (Finset.univ : Finset (Idx n)))
      (A := fun i (s : ℝ) => ∑ j : Idx n, Real.exp (β * Real.cos
        (Function.update Θ l s i - Function.update Θ l s j)))
      (fun i _ => hrow i)
    simpa [Finset.sum_fn] using h
  have hmul := HasDerivAt.const_mul (1 / (2 * β * Real.exp β * (n : ℝ)^2)) hsum
  refine hmul.congr_deriv ?_
  -- the derivative computed above is `angularGrad`
  have hA : ∀ i : Idx n, (∑ j : Idx n, (if i = l then a j else 0))
      = if i = l then ∑ j : Idx n, a j else 0 := by
    intro i; split_ifs with h <;> simp
  have hsplit : (∑ i : Idx n, ∑ j : Idx n,
      ((if i = l then a j else 0) + (if j = l then b i else 0)))
      = (∑ j : Idx n, a j) + ∑ i : Idx n, b i := by
    simp only [Finset.sum_add_distrib, hA, Finset.sum_ite_eq' Finset.univ l,
      Finset.mem_univ, ite_true]
  have hb_eq : (∑ i : Idx n, b i) = ∑ j : Idx n, a j := by
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [ha, hb]
    rw [show Θ i - Θ l = -(Θ l - Θ i) by ring, Real.sin_neg, Real.cos_neg]
    ring
  set S : ℝ := ∑ j : Idx n, Real.sin (Θ l - Θ j)
    * Real.exp (β * Real.cos (Θ l - Θ j)) with hS
  have hsa : (∑ j : Idx n, a j) = -β * S := by
    rw [hS, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by simp only [ha]; ring
  have hga : angularGrad n β Θ l
      = (1 / (n : ℝ)^2) * (-((Real.exp β)⁻¹ * S)) := by
    have h1 : (∑ m : Idx n, Real.sin (Θ l - Θ m)
          * Real.exp (β * (Real.cos (Θ l - Θ m) - 1)))
        = (Real.exp β)⁻¹ * S := by
      rw [hS, Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [mul_sub, mul_one, Real.exp_sub, div_eq_mul_inv]
      ring
    rw [angularGrad, h1]
    ring
  have hconst : (1 / (2 * β * Real.exp β * (n : ℝ)^2))
      = (1 / (n : ℝ)^2) * (1 / (2 * β * Real.exp β)) := by
    rw [one_div, one_div, one_div, ← mul_inv]
    ring_nf
  rw [hsplit, hb_eq, hsa, hga, hconst, mul_assoc]
  congr 1
  field_simp
  ring

/-- **The angular energy is the interaction energy of the circle
configuration.**

`𝖤_β` at `x_i = (cos θ_i, sin θ_i)` is `angularEβ`, since
`⟨x_i, x_j⟩ = cos(θ_i - θ_j)`.  This is what licenses reading §3.2 of
arXiv:2410.06833v1 on the torus. -/
theorem Eβ_circlePoint (n : ℕ) (β : ℝ) (Θ : Idx n → ℝ) :
    Eβ 2 n β (fun i => circlePoint (Θ i)) = angularEβ n β Θ := by
  simp only [Eβ, angularEβ, inner_circlePoint]

end Metastability
end Transformer
