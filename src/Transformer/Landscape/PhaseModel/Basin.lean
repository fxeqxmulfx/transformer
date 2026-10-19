module

public import Transformer.Landscape.PhaseModel.Abelian.Basin

/-!
# Phase model — critical points of `V_full` near the zero `0`

Let `n ≥ 4` and let `ψ` be a critical point of the energy `V_full` of the
full division table where the Hessian has a nonnegative diagonal, as at a
local minimizer. If `E = Σ_x (1 - cos ψ_x) < n/8`, then `cos ψ_t > 1/2` for
every `t` (`half_lt_cos_of_gradient_eq_zero`). With
`θ_{x,y} = ψ_x - ψ_y - ψ_{x-y}` and
`λ_t = Σ_y cos θ_{t,y} + Σ_x cos θ_{x,t} + Σ_y cos θ_{t+y,y}`, the identity
`λ_t cos ψ_t + sin ψ_t ∂_t V = Σ_y cos (ψ_y + ψ_{t-y}) +
Σ_x cos (ψ_x - ψ_{x-t}) + Σ_y cos (ψ_{t+y} - ψ_y)` and the chord inequality
give `λ_t cos ψ_t ≥ 3n - 12E > 3n/2`, while `λ_t ≤ 3n` and, by the Hessian,
`λ_t ≥ -6`. Each `ψ_t` is then within `π/3` of `2πℤ`, every relation of
`ψ` within `π` of `2πℤ`, and a critical point with every `cos ψ_t > 1/2` is
a period (`mem_periods_of_forall_half_lt_cos`).

Both theorems are those of `Landscape.PhaseModel.Abelian.Basin` at `G = Fin n`, with
`|G| = n`; for `n = 0` every vector is a period.

Source: item M6.8 of `landscape_plan.md`.
-/

@[expose] public section

open Real

namespace Transformer.Landscape.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- **Near the zero `0`, a critical point of `V_full` with a nonnegative
Hessian diagonal has every `cos ψ_t > 1/2`**, for `n ≥ 4`, when
`Σ_x (1 - cos ψ_x) < n/8`. -/
theorem half_lt_cos_of_gradient_eq_zero (hn : 4 ≤ n) {ψ : EucSpace n}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ = 0)
    (hH : ∀ t, 0 ≤ ⟪fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))))
      ψ (PiLp.single 2 t 1), PiLp.single 2 t 1⟫)
    (hE : ∑ x, (1 - cos (ψ.ofLp x)) < n / 8) (t : Fin n) : 1 / 2 < cos (ψ.ofLp t) := by
  have : NeZero n := ⟨by omega⟩
  exact Abelian.half_lt_cos_of_gradient_eq_zero (G := Fin n) (by rwa [Fintype.card_fin]) hg hH
    (by rwa [Fintype.card_fin]) t

/-- **A critical point of `V_full` with every `cos ψ_t > 1/2` is a period**:
rounding each `ψ_t` to `2πℤ` leaves a remainder below `π/3`, every relation
of `ψ` is then within `π` of that of a period, a zero of `V_full`, so it
vanishes (`inner_sub_eq_zero_of_gradient_eq_zero`), and only `0` is
orthogonal to every row. -/
theorem mem_periods_of_forall_half_lt_cos {ψ : EucSpace n}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ = 0)
    (hc : ∀ t, 1 / 2 < cos (ψ.ofLp t)) : ψ ∈ periods n := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact fun t => t.elim0
  have : NeZero n := ⟨hn.ne'⟩
  exact Abelian.mem_periods_of_forall_half_lt_cos (G := Fin n) hg hc

/-- The hypotheses of `half_lt_cos_of_gradient_eq_zero` hold for `n = 4` at
the zero `ψ = 0`, where the Hessian is `h ↦ Σ_r ⟪m_r, h⟫ m_r` and
`Σ_x (1 - cos 0) = 0 < 4/8`. -/
example (t : Fin 4) : 1 / 2 < cos ((0 : EucSpace 4).ofLp t) :=
  half_lt_cos_of_gradient_eq_zero le_rfl
    (gradient_eq_zero_of_phaseEnergy_eq_zero _ (by simp [phaseEnergy]))
    (fun t => by
      simp only [divisionEnergy]
      rw [inner_fderiv_gradient_phaseEnergy]
      exact Finset.sum_nonneg fun r _ => by
        rw [inner_zero_right, cos_zero, one_mul]
        exact sq_nonneg _)
    (by norm_num) t

/-- The hypotheses of `mem_periods_of_forall_half_lt_cos` hold for `n = 4`
at the zero `ψ = 0`, where every `cos ψ_t = 1`. -/
example : (0 : EucSpace 4) ∈ periods 4 :=
  mem_periods_of_forall_half_lt_cos
    (gradient_eq_zero_of_phaseEnergy_eq_zero _ (by simp [phaseEnergy])) fun t => by norm_num

end Transformer.Landscape.PhaseModel
