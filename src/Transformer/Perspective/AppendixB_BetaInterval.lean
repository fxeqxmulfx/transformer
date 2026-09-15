/-
# Appendix B — the angle `τ_β^*`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The kernels `g_β` of Appendix B, and the angle `τ_β^*` at which `g_β` changes
sign: the unique solution of `β sin² τ = (d - 1) cos τ` in `[0, π/2)`.

The block Hessian on the circle (`eq: taylor2`, `eq: taylor3`) is in
`Perspective.AppendixB_Taylor`; the `d`-dimensional statements
(`eq: claim.yury`, `eq: dr1`, `e:Hessianincoord`, `eq: metric.grad`,
`eq: metric.hess`) are in `Perspective.AppendixB_HighD`.
-/

import Transformer.Basic
import Transformer.Perspective.Section4_LargeBeta
import Transformer.Perspective.Section1_IPS

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The kernel `g_β` used in the proof for `d = 2`:

  `g_β(τ) = (cos τ - β sin² τ) · e^{β cos τ}`. -/
noncomputable def g_β_2d (β τ : ℝ) : ℝ :=
  (Real.cos τ - β * Real.sin τ ^ 2) * Real.exp (β * Real.cos τ)

/-- The kernel `g_β` used for general `d`:

  `g_β(ζ) = e^{β cos ζ} ((d - 1) cos ζ - β sin² ζ)`. -/
noncomputable def g_β_d (d : ℕ) (β ζ : ℝ) : ℝ :=
  Real.exp (β * Real.cos ζ) * (((d : ℝ) - 1) * Real.cos ζ - β * Real.sin ζ ^ 2)

/-- `cos τ_β^*`: the positive root of `β c² + (d-1) c - β = 0`, written out by
the quadratic formula.

The equation `β sin² τ = (d-1) cos τ` of the survey is this quadratic in
`c = cos τ`, since `sin² τ = 1 - c²`. -/
noncomputable def cosTauStar (d : ℕ) (β : ℝ) : ℝ :=
  (Real.sqrt (((d : ℝ) - 1) ^ 2 + 4 * β ^ 2) - ((d : ℝ) - 1)) / (2 * β)

/-- The solution `τ_β^* ∈ [0, π/2)` of `β sin² τ = (d - 1) cos τ`.

It is `arccos` of the root above; `τ_β_star_spec` proves that it solves the
equation and lies in `[0, π/2)`, and `τ_β_star_unique` that it is the only
such solution. -/
noncomputable def τ_β_star (d : ℕ) (β : ℝ) : ℝ := Real.arccos (cosTauStar d β)

/-- The discriminant is a square: `(√((d-1)² + 4β²))² = (d-1)² + 4β²`. -/
theorem sq_sqrt_disc (d : ℕ) (β : ℝ) :
    Real.sqrt (((d : ℝ) - 1) ^ 2 + 4 * β ^ 2) ^ 2 = ((d : ℝ) - 1) ^ 2 + 4 * β ^ 2 :=
  Real.sq_sqrt (by positivity)

/-- The hypotheses `0 < β` and `1 ≤ d` of the lemmas below are satisfiable. -/
example : (0 : ℝ) < 1 ∧ 1 ≤ 2 := ⟨one_pos, one_le_two⟩

/-- `cos τ_β^* > 0`, which is what puts `τ_β^*` strictly below `π/2`. -/
theorem cosTauStar_pos (d : ℕ) (β : ℝ) (hβ : 0 < β) (hd : 1 ≤ d) :
    0 < cosTauStar d β := by
  have hD : (0 : ℝ) ≤ (d : ℝ) - 1 := by
    have : (1 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
    linarith
  have hlt : ((d : ℝ) - 1) < Real.sqrt (((d : ℝ) - 1) ^ 2 + 4 * β ^ 2) :=
    Real.lt_sqrt_of_sq_lt (by nlinarith)
  exact div_pos (by linarith) (by linarith)

/-- `cos τ_β^* ≤ 1`, so `τ_β^*` is a genuine angle. -/
theorem cosTauStar_le_one (d : ℕ) (β : ℝ) (hβ : 0 < β) (hd : 1 ≤ d) :
    cosTauStar d β ≤ 1 := by
  have hD : (0 : ℝ) ≤ (d : ℝ) - 1 := by
    have : (1 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
    linarith
  have hle : Real.sqrt (((d : ℝ) - 1) ^ 2 + 4 * β ^ 2) ≤ 2 * β + ((d : ℝ) - 1) := by
    have h1 : ((d : ℝ) - 1) ^ 2 + 4 * β ^ 2 ≤ (2 * β + ((d : ℝ) - 1)) ^ 2 := by nlinarith
    calc Real.sqrt (((d : ℝ) - 1) ^ 2 + 4 * β ^ 2)
        ≤ Real.sqrt ((2 * β + ((d : ℝ) - 1)) ^ 2) := Real.sqrt_le_sqrt h1
      _ = 2 * β + ((d : ℝ) - 1) := Real.sqrt_sq (by linarith)
  rw [cosTauStar, div_le_one (by linarith)]
  linarith

/-- `cos τ_β^*` solves `β c² + (d-1) c - β = 0`. -/
theorem cosTauStar_quadratic (d : ℕ) (β : ℝ) (hβ : 0 < β) :
    β * cosTauStar d β ^ 2 + ((d : ℝ) - 1) * cosTauStar d β - β = 0 := by
  have hS := sq_sqrt_disc d β
  rw [cosTauStar]
  set S := Real.sqrt (((d : ℝ) - 1) ^ 2 + 4 * β ^ 2) with hSdef
  field_simp
  nlinarith [hS, sq_nonneg β]

/-- **The solution of `β sin² τ = (d-1) cos τ`.**  `τ_β^*` lies in `[0, π/2)`
and solves the equation. -/
theorem τ_β_star_spec (d : ℕ) (β : ℝ) (hβ : 0 < β) (hd : 1 ≤ d) :
    τ_β_star d β ∈ Set.Ico 0 (π / 2) ∧
      β * Real.sin (τ_β_star d β) ^ 2 = ((d : ℝ) - 1) * Real.cos (τ_β_star d β) := by
  have hpos := cosTauStar_pos d β hβ hd
  have hle := cosTauStar_le_one d β hβ hd
  have hcos : Real.cos (τ_β_star d β) = cosTauStar d β :=
    Real.cos_arccos (by linarith) hle
  refine ⟨⟨Real.arccos_nonneg _, Real.arccos_lt_pi_div_two.mpr hpos⟩, ?_⟩
  have hsin : Real.sin (τ_β_star d β) ^ 2 = 1 - cosTauStar d β ^ 2 := by
    have h := Real.sin_sq_add_cos_sq (τ_β_star d β)
    rw [hcos] at h
    linarith
  rw [hsin, hcos]
  have := cosTauStar_quadratic d β hβ
  linarith

/-- **Uniqueness.**  `τ_β^*` is the only solution of `β sin² τ = (d-1) cos τ`
in `[0, π/2)`: the quadratic has a single positive root. -/
theorem τ_β_star_unique
    (d : ℕ) (β : ℝ) (hβ : 0 < β) (hd : 1 ≤ d)
    (τ : ℝ) (hτ : τ ∈ Set.Ico 0 (π / 2))
    (h : β * Real.sin τ ^ 2 = ((d : ℝ) - 1) * Real.cos τ) :
    τ = τ_β_star d β := by
  have hD : (0 : ℝ) ≤ (d : ℝ) - 1 := by
    have : (1 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
    linarith
  have hcospos : 0 < Real.cos τ :=
    Real.cos_pos_of_mem_Ioo ⟨by linarith [hτ.1, Real.pi_pos], hτ.2⟩
  have hquad : β * Real.cos τ ^ 2 + ((d : ℝ) - 1) * Real.cos τ - β = 0 := by
    have hpyth := Real.sin_sq_add_cos_sq τ
    nlinarith [h, hpyth]
  have hquad' := cosTauStar_quadratic d β hβ
  have hstar := cosTauStar_pos d β hβ hd
  have heq : Real.cos τ = cosTauStar d β := by
    have hfac : (Real.cos τ - cosTauStar d β) *
        (β * (Real.cos τ + cosTauStar d β) + ((d : ℝ) - 1)) = 0 := by nlinarith
    have hposfac : 0 < β * (Real.cos τ + cosTauStar d β) + ((d : ℝ) - 1) := by nlinarith
    rcases mul_eq_zero.mp hfac with h1 | h2
    · linarith
    · linarith
  rw [τ_β_star, ← heq, Real.arccos_cos hτ.1 (by linarith [hτ.2, Real.pi_pos])]

end Perspective
end Transformer
