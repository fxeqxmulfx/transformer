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
chord inequality `1 - cos (a ± b) ≤ 2 (1 - cos a) + 2 (1 - cos b)`
(`one_sub_cos_add_le`, `one_sub_cos_sub_le`) bounds the energy of a sum of
phases by those of the summands, also summed over `Fin n`
(`sub_le_sum_cos_add`, `sub_le_sum_cos_sub`).
-/

import Transformer.PhaseModel.Lift
import Transformer.PhaseModel.Periods

open Real

namespace Transformer.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- The sum over the pairs with first entry `t`. -/
theorem sum_ite_fst (t : Fin n) (f : Fin n × Fin n → ℝ) :
    ∑ p, (if p.1 = t then f p else 0) = ∑ y, f (t, y) := by
  rw [Fintype.sum_prod_type, Finset.sum_comm]
  simp [Finset.sum_ite_eq']

/-- The sum over the pairs with second entry `t`. -/
theorem sum_ite_snd (t : Fin n) (f : Fin n × Fin n → ℝ) :
    ∑ p, (if p.2 = t then f p else 0) = ∑ x, f (x, t) := by
  rw [Fintype.sum_prod_type]
  simp [Finset.sum_ite_eq']

/-- The sum over the pairs with difference `t`: these are the pairs
`(t + y, y)`. -/
theorem sum_ite_sub [NeZero n] (t : Fin n) (f : Fin n × Fin n → ℝ) :
    ∑ p, (if p.1 - p.2 = t then f p else 0) = ∑ y, f (t + y, y) := by
  rw [Fintype.sum_prod_type_right]
  simp [sub_eq_iff_eq_add, Finset.sum_ite_eq']

/-- The coordinate `t` of the row of `x / y`, which is `e_x - e_y - e_{x-y}`. -/
theorem inner_intVec_divisionRow_single (p : Fin n × Fin n) (t : Fin n) :
    ⟪intVec (divisionRow p), PiLp.single 2 t (1 : ℝ)⟫ =
      (if p.1 = t then 1 else 0) - (if p.2 = t then 1 else 0) -
        (if p.1 - p.2 = t then 1 else 0) := by
  rw [inner_intVec_divisionRow]
  simp

/-- **The chord inequality** for a sum: `1 - cos (a + b) ≤ 2 (1 - cos a) +
2 (1 - cos b)`, the triangle inequality `|e^{i(a+b)} - 1| ≤ |e^{ia} - 1| +
|e^{ib} - 1|` squared, with `(u + v)² ≤ 2u² + 2v²`. -/
theorem one_sub_cos_add_le (a b : ℝ) : 1 - cos (a + b) ≤ 2 * (1 - cos a) + 2 * (1 - cos b) := by
  rw [cos_add]
  nlinarith [sin_sq_add_cos_sq a, sin_sq_add_cos_sq b, sq_nonneg (sin a - sin b),
    sq_nonneg (1 - cos a), sq_nonneg (1 - cos b),
    mul_nonneg (sub_nonneg.2 (cos_le_one a)) (sub_nonneg.2 (cos_le_one b))]

/-- **The chord inequality** for a difference: `1 - cos (a - b) ≤
2 (1 - cos a) + 2 (1 - cos b)`. -/
theorem one_sub_cos_sub_le (a b : ℝ) : 1 - cos (a - b) ≤ 2 * (1 - cos a) + 2 * (1 - cos b) := by
  rw [cos_sub]
  nlinarith [sin_sq_add_cos_sq a, sin_sq_add_cos_sq b, sq_nonneg (sin a + sin b),
    sq_nonneg (1 - cos a), sq_nonneg (1 - cos b),
    mul_nonneg (sub_nonneg.2 (cos_le_one a)) (sub_nonneg.2 (cos_le_one b))]

/-- **The chord inequality summed**:
`Σ_y cos (a_y + b_y) ≥ n - 2 Σ_y (1 - cos a_y) - 2 Σ_y (1 - cos b_y)`. -/
theorem sub_le_sum_cos_add (a b : Fin n → ℝ) :
    n - 2 * ∑ y, (1 - cos (a y)) - 2 * ∑ y, (1 - cos (b y)) ≤ ∑ y, cos (a y + b y) := by
  have h := Finset.sum_le_sum fun y (_ : y ∈ Finset.univ) => one_sub_cos_add_le (a y) (b y)
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at h ⊢
  linarith

/-- **The chord inequality summed**:
`Σ_y cos (a_y - b_y) ≥ n - 2 Σ_y (1 - cos a_y) - 2 Σ_y (1 - cos b_y)`. -/
theorem sub_le_sum_cos_sub (a b : Fin n → ℝ) :
    n - 2 * ∑ y, (1 - cos (a y)) - 2 * ∑ y, (1 - cos (b y)) ≤ ∑ y, cos (a y - b y) := by
  have h := Finset.sum_le_sum fun y (_ : y ∈ Finset.univ) => one_sub_cos_sub_le (a y) (b y)
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at h ⊢
  linarith

variable [NeZero n]

/-- **The coordinates of the gradient of `V_full`**: with
`θ_{x,y} = ψ_x - ψ_y - ψ_{x-y}`,
`∂_t V = Σ_y sin θ_{t,y} - Σ_x sin θ_{x,t} - Σ_y sin θ_{t+y,y}`. -/
theorem inner_gradient_divisionEnergy_single (ψ : EucSpace n) (t : Fin n) :
    ⟪gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ, PiLp.single 2 t 1⟫ =
      ∑ y, sin (ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y)) -
        ∑ x, sin (ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t)) -
          ∑ y, sin (ψ.ofLp (t + y) - ψ.ofLp y - ψ.ofLp t) := by
  rw [divisionEnergy, gradient_phaseEnergy]
  simp only [sum_inner, real_inner_smul_left, inner_intVec_divisionRow_single]
  rw [Finset.sum_coe_sort Finset.univ fun p => sin ⟪intVec (divisionRow p), ψ⟫ *
    ((if p.1 = t then 1 else 0) - (if p.2 = t then 1 else 0) - (if p.1 - p.2 = t then 1 else 0))]
  simp only [mul_sub, mul_ite, mul_one, mul_zero, Finset.sum_sub_distrib]
  rw [sum_ite_fst, sum_ite_snd, sum_ite_sub]
  simp only [inner_intVec_divisionRow, add_sub_cancel_right]

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
            2 * cos (ψ.ofLp (t + t) - 2 * ψ.ofLp t) := by
  rw [divisionEnergy, inner_fderiv_gradient_phaseEnergy]
  simp only [inner_intVec_divisionRow_single]
  rw [Finset.sum_coe_sort Finset.univ fun p => cos ⟪intVec (divisionRow p), ψ⟫ *
    ((if p.1 = t then 1 else 0) - (if p.2 = t then 1 else 0) - (if p.1 - p.2 = t then 1 else 0)) ^ 2]
  set c := fun p : Fin n × Fin n => cos ⟪intVec (divisionRow p), ψ⟫ with hc
  have h : ∀ p : Fin n × Fin n, c p *
      ((if p.1 = t then 1 else 0) - (if p.2 = t then 1 else 0) -
        (if p.1 - p.2 = t then 1 else 0)) ^ 2 =
      (if p.1 = t then c p else 0) + (if p.2 = t then c p else 0) +
        (if p.1 - p.2 = t then c p else 0) -
          2 * (if p.1 = t then (if p.2 = t then c p else 0) else 0) -
            2 * (if p.1 = t then (if p.1 - p.2 = t then c p else 0) else 0) +
              2 * (if p.2 = t then (if p.1 - p.2 = t then c p else 0) else 0) := fun p => by
    split_ifs <;> ring
  rw [Finset.sum_congr rfl fun p _ => h p]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [sum_ite_fst, sum_ite_snd, sum_ite_sub, sum_ite_fst, sum_ite_fst, sum_ite_snd]
  simp only [sub_eq_self, sub_eq_iff_eq_add, Finset.sum_ite_eq', Finset.mem_univ, ite_true, hc,
    inner_intVec_divisionRow, add_sub_cancel_right, sub_self, sub_zero, zero_sub, cos_neg,
    sub_sub_cancel_left]
  rw [show ψ.ofLp (t + t) - ψ.ofLp t - ψ.ofLp t = ψ.ofLp (t + t) - 2 * ψ.ofLp t by ring]
  ring

/-- The hypothesis `NeZero n` of `sum_ite_sub` and of the last two theorems
holds for `n = 4`; at the phases `0` every relation is `0`, so the diagonal
of the Hessian of `V_full` is `3 · 4 - 4 + 2 = 10`. -/
example (t : Fin 4) :
    ⟪fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (Fin 4 × Fin 4)))) 0
        (PiLp.single 2 t 1), PiLp.single 2 t 1⟫ = 10 := by
  rw [inner_fderiv_gradient_divisionEnergy_single]
  simp only [WithLp.ofLp_zero, Pi.zero_apply, sub_zero, mul_zero, cos_zero, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  norm_num

end Transformer.PhaseModel
