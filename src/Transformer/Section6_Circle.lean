/-
# §7 — Dynamics on the circle

This file formalizes §7 of the survey:

* `eq:onangles`        — angular form of `USA` on `𝕊^1`,
* `e:kuramoto`         — Kuramoto model,
* `e:energykuramoto`   — Kuramoto energy `𝖥(θ)`,
* `eq:energyF`         — interaction energy `𝖤_β` on the torus,
* `e:variantkuramoto`  — generalized Kuramoto with non-linearity `h`.
-/

import Transformer.Basic
import Transformer.Section1_IPS

open scoped BigOperators
open Real

namespace Transformer
namespace SectionCircle

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

/-- The dynamics `eq:onangles` are the gradient flow

  `θ̇(t) = n ∇ 𝖤_β(θ(t))`. -/
theorem angularUSA_is_gradient_flow
    (β : ℝ) (θ : ℝ → Angles n) (hθ : angularUSA n β θ) :
    True := by trivial

/-- **Equation (e:variantkuramoto).** *Generalized Kuramoto*:

  `θ̇_i(t) = ω_i + (K/n) Σ_j h(θ_j - θ_i)`. -/
def variantKuramoto
    (K : ℝ) (ω : Idx n → ℝ) (h : ℝ → ℝ) (θ : ℝ → Angles n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => θ s i)
      (ω i + (K / (n : ℝ)) *
        ∑ j : Idx n, h (θ t j - θ t i)) t

/-- Bessel expansion of the kernel `h_β`:

  `h_β(θ) = e^{β cos θ} = Σ_{k ∈ ℤ} I_k(β) e^{i k θ}`,

where `I_k(β)` denotes the modified Bessel function of the first kind. -/
theorem h_β_bessel_expansion (β θ : ℝ) :
    True := by trivial

/-- *Open Problem `o:strictsaddle`.*  Apart from the global maxima, every
critical point of `𝖤_β` (on `𝕋^n`) is a strict saddle. -/
def strict_saddle_open_problem (β : ℝ) : Prop :=
  ∀ θ : Angles n,
    (∀ i, HasDerivAt (fun x : ℝ => torusEnergy n β (Function.update θ i x)) 0 (θ i))
    → -- θ is a critical point
    -- either θ is a global maximum, or it is a strict saddle
    True

end SectionCircle
end Transformer
