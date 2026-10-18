/-
# Phase model — coordinates on the full division table

The phase `ψ_t` enters the relations `θ_{x,y} = ψ_x - ψ_y - ψ_{x-y}` of the
full division table as `ψ_x` in the rows `t / y`, as `ψ_y` in the rows
`x / t`, and as `ψ_{x-y}` in the rows `(t + y) / y`.  This gives the
coordinates of the gradient of `V_full`
(`inner_gradient_divisionEnergy_single`) and the diagonal of its Hessian
(`inner_fderiv_gradient_divisionEnergy_single`): there the rows `t / t` and
`t / 0`, where `ψ_t` enters twice with opposite signs, drop out, and the row
`2t / t`, where it enters twice with the same sign, counts four times.  The
chord inequality (`one_sub_cos_add_le`, `one_sub_cos_sub_le`) bounds the
energy of a sum of phases by those of the summands, also summed over
`Fin n` (`sub_le_sum_cos_add`, `sub_le_sum_cos_sub`).

Each theorem is that of `Landscape.PhaseModel.Abelian.Coordinates` at `G = Fin n`,
with `|G| = n`; those using subtraction in `Fin n` need `n ≠ 0`, the
statements without `NeZero n` being empty for `n = 0`.

Source: item M6.8 of `landscape_plan.md`.
-/

import Transformer.Landscape.PhaseModel.Abelian.Coordinates

open Real

namespace Transformer.Landscape.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- The sum over the pairs with first entry `t`. -/
theorem sum_ite_fst (t : Fin n) (f : Fin n × Fin n → ℝ) :
    ∑ p, (if p.1 = t then f p else 0) = ∑ y, f (t, y) :=
  Abelian.sum_ite_fst t f

/-- The sum over the pairs with second entry `t`. -/
theorem sum_ite_snd (t : Fin n) (f : Fin n × Fin n → ℝ) :
    ∑ p, (if p.2 = t then f p else 0) = ∑ x, f (x, t) :=
  Abelian.sum_ite_snd t f

/-- The sum over the pairs with difference `t`: these are the pairs
`(t + y, y)`. -/
theorem sum_ite_sub [NeZero n] (t : Fin n) (f : Fin n × Fin n → ℝ) :
    ∑ p, (if p.1 - p.2 = t then f p else 0) = ∑ y, f (t + y, y) :=
  Abelian.sum_ite_sub t f

/-- The coordinate `t` of the row of `x / y`, which is `e_x - e_y - e_{x-y}`. -/
theorem inner_intVec_divisionRow_single (p : Fin n × Fin n) (t : Fin n) :
    ⟪intVec (divisionRow p), PiLp.single 2 t (1 : ℝ)⟫ =
      (if p.1 = t then 1 else 0) - (if p.2 = t then 1 else 0) -
        (if p.1 - p.2 = t then 1 else 0) := by
  have : NeZero n := ⟨fun h => (h ▸ t).elim0⟩
  exact Abelian.inner_intVec_divisionRow_single p t

/-- **The chord inequality summed**:
`Σ_y cos (a_y + b_y) ≥ n - 2 Σ_y (1 - cos a_y) - 2 Σ_y (1 - cos b_y)`. -/
theorem sub_le_sum_cos_add (a b : Fin n → ℝ) :
    n - 2 * ∑ y, (1 - cos (a y)) - 2 * ∑ y, (1 - cos (b y)) ≤ ∑ y, cos (a y + b y) := by
  simpa only [Fintype.card_fin] using Abelian.sub_le_sum_cos_add a b

/-- **The chord inequality summed**:
`Σ_y cos (a_y - b_y) ≥ n - 2 Σ_y (1 - cos a_y) - 2 Σ_y (1 - cos b_y)`. -/
theorem sub_le_sum_cos_sub (a b : Fin n → ℝ) :
    n - 2 * ∑ y, (1 - cos (a y)) - 2 * ∑ y, (1 - cos (b y)) ≤ ∑ y, cos (a y - b y) := by
  simpa only [Fintype.card_fin] using Abelian.sub_le_sum_cos_sub a b

variable [NeZero n]

/-- **The coordinates of the gradient of `V_full`**: with
`θ_{x,y} = ψ_x - ψ_y - ψ_{x-y}`,
`∂_t V = Σ_y sin θ_{t,y} - Σ_x sin θ_{x,t} - Σ_y sin θ_{t+y,y}`. -/
theorem inner_gradient_divisionEnergy_single (ψ : EucSpace n) (t : Fin n) :
    ⟪gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ, PiLp.single 2 t 1⟫ =
      ∑ y, sin (ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y)) -
        ∑ x, sin (ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t)) -
          ∑ y, sin (ψ.ofLp (t + y) - ψ.ofLp y - ψ.ofLp t) :=
  Abelian.inner_gradient_divisionEnergy_single ψ t

/-- **The diagonal of the Hessian of `V_full`**: with
`θ_{x,y} = ψ_x - ψ_y - ψ_{x-y}`, `⟪H e_t, e_t⟫` is
`Σ_y cos θ_{t,y} + Σ_x cos θ_{x,t} + Σ_y cos θ_{t+y,y} - 4 cos ψ_0 +
2 cos (ψ_{2t} - 2 ψ_t)`. -/
theorem inner_fderiv_gradient_divisionEnergy_single (ψ : EucSpace n) (t : Fin n) :
    ⟪fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n)))) ψ
        (PiLp.single 2 t 1), PiLp.single 2 t 1⟫ =
      ∑ y, cos (ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y)) +
        ∑ x, cos (ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t)) +
          ∑ y, cos (ψ.ofLp (t + y) - ψ.ofLp y - ψ.ofLp t) - 4 * cos (ψ.ofLp 0) +
            2 * cos (ψ.ofLp (t + t) - 2 * ψ.ofLp t) :=
  Abelian.inner_fderiv_gradient_divisionEnergy_single ψ t

/-- The hypothesis `NeZero n` of `sum_ite_sub` holds for `n = 4`: the pairs
of `Fin 4` with difference `1` are the four pairs `(1 + y, y)`. -/
example : ∑ p : Fin 4 × Fin 4, (if p.1 - p.2 = 1 then (1 : ℝ) else 0) = 4 := by
  rw [sum_ite_sub]
  norm_num

/-- The hypothesis `NeZero n` of the last two theorems holds for `n = 4`; at
the phases `0` every relation is `0`, so the diagonal of the Hessian of
`V_full` is `3 · 4 - 4 + 2 = 10`. -/
example (t : Fin 4) :
    ⟪fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (Fin 4 × Fin 4)))) 0
        (PiLp.single 2 t 1), PiLp.single 2 t 1⟫ = 10 := by
  rw [inner_fderiv_gradient_divisionEnergy_single]
  simp only [WithLp.ofLp_zero, Pi.zero_apply, sub_zero, mul_zero, cos_zero, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  norm_num

end Transformer.Landscape.PhaseModel
