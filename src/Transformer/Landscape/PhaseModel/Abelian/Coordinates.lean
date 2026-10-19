module

public import Transformer.Landscape.PhaseModel.Abelian.Basic
public import Transformer.Landscape.PhaseModel.Lift

/-!
# Phase model over a finite abelian group — coordinates on the full table

As for `ℤ/n` (`Landscape.PhaseModel.Coordinates`), the phase `ψ_t` enters the
relations `θ_{x,y} = ψ_x - ψ_y - ψ_{x-y}` of the full division table of `G`
as `ψ_x` in the rows `t / y`, as `ψ_y` in the rows `x / t`, and as `ψ_{x-y}`
in the rows `(t + y) / y`. This gives the coordinates of the gradient of
`V_full` (`inner_gradient_divisionEnergy_single`) and the diagonal of its
Hessian (`inner_fderiv_gradient_divisionEnergy_single`): there the rows
`t / t` and `t / 0`, where `ψ_t` enters twice with opposite signs, drop out,
and the row `2t / t`, where it enters twice with the same sign, counts four
times. The chord inequality `1 - cos (a ± b) ≤ 2 (1 - cos a) + 2 (1 - cos b)`
summed over `G` bounds the energy of a sum of phases by those of the
summands (`sub_le_sum_cos_add`, `sub_le_sum_cos_sub`).
-/

@[expose] public section

open Real

namespace Transformer.Landscape.PhaseModel.Abelian

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {G : Type*} [Fintype G]

/-- **The chord inequality summed** over `G`:
`Σ_y cos (a_y + b_y) ≥ |G| - 2 Σ_y (1 - cos a_y) - 2 Σ_y (1 - cos b_y)`. -/
theorem sub_le_sum_cos_add (a b : G → ℝ) :
    Fintype.card G - 2 * ∑ y, (1 - cos (a y)) - 2 * ∑ y, (1 - cos (b y)) ≤
      ∑ y, cos (a y + b y) := by
  have h := Finset.sum_le_sum fun y (_ : y ∈ Finset.univ) => one_sub_cos_add_le (a y) (b y)
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, nsmul_eq_mul, mul_one] at h ⊢
  linarith

/-- **The chord inequality summed** over `G`:
`Σ_y cos (a_y - b_y) ≥ |G| - 2 Σ_y (1 - cos a_y) - 2 Σ_y (1 - cos b_y)`. -/
theorem sub_le_sum_cos_sub (a b : G → ℝ) :
    Fintype.card G - 2 * ∑ y, (1 - cos (a y)) - 2 * ∑ y, (1 - cos (b y)) ≤
      ∑ y, cos (a y - b y) := by
  have h := Finset.sum_le_sum fun y (_ : y ∈ Finset.univ) => one_sub_cos_sub_le (a y) (b y)
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, nsmul_eq_mul, mul_one] at h ⊢
  linarith

variable [DecidableEq G]

/-- The sum over the pairs with first entry `t`. -/
theorem sum_ite_fst (t : G) (f : G × G → ℝ) :
    ∑ p, (if p.1 = t then f p else 0) = ∑ y, f (t, y) := by
  rw [Fintype.sum_prod_type, Finset.sum_comm]
  simp [Finset.sum_ite_eq']

/-- The sum over the pairs with second entry `t`. -/
theorem sum_ite_snd (t : G) (f : G × G → ℝ) :
    ∑ p, (if p.2 = t then f p else 0) = ∑ x, f (x, t) := by
  rw [Fintype.sum_prod_type]
  simp [Finset.sum_ite_eq']

variable [AddCommGroup G]

/-- The sum over the pairs with difference `t`: these are the pairs
`(t + y, y)`. -/
theorem sum_ite_sub (t : G) (f : G × G → ℝ) :
    ∑ p, (if p.1 - p.2 = t then f p else 0) = ∑ y, f (t + y, y) := by
  rw [Fintype.sum_prod_type_right]
  simp [sub_eq_iff_eq_add, Finset.sum_ite_eq']

/-- The coordinate `t` of the row of `x / y`, which is `e_x - e_y - e_{x-y}`. -/
theorem inner_intVec_divisionRow_single (p : G × G) (t : G) :
    ⟪intVec (divisionRow p), PiLp.single 2 t (1 : ℝ)⟫ =
      (if p.1 = t then 1 else 0) - (if p.2 = t then 1 else 0) -
        (if p.1 - p.2 = t then 1 else 0) := by
  rw [inner_intVec_divisionRow]
  simp

/-- **The coordinates of the gradient of `V_full`**: with
`θ_{x,y} = ψ_x - ψ_y - ψ_{x-y}`,
`∂_t V = Σ_y sin θ_{t,y} - Σ_x sin θ_{x,t} - Σ_y sin θ_{t+y,y}`. -/
theorem inner_gradient_divisionEnergy_single (ψ : EuclideanSpace ℝ G) (t : G) :
    ⟪gradient (divisionEnergy (Finset.univ : Finset (G × G))) ψ, PiLp.single 2 t 1⟫ =
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
theorem inner_fderiv_gradient_divisionEnergy_single (ψ : EuclideanSpace ℝ G) (t : G) :
    ⟪fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (G × G)))) ψ
        (PiLp.single 2 t 1), PiLp.single 2 t 1⟫ =
      ∑ y, cos (ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y)) +
        ∑ x, cos (ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t)) +
          ∑ y, cos (ψ.ofLp (t + y) - ψ.ofLp y - ψ.ofLp t) - 4 * cos (ψ.ofLp 0) +
            2 * cos (ψ.ofLp (t + t) - 2 * ψ.ofLp t) := by
  rw [divisionEnergy, inner_fderiv_gradient_phaseEnergy]
  simp only [inner_intVec_divisionRow_single]
  rw [Finset.sum_coe_sort Finset.univ fun p => cos ⟪intVec (divisionRow p), ψ⟫ *
    ((if p.1 = t then 1 else 0) - (if p.2 = t then 1 else 0) -
      (if p.1 - p.2 = t then 1 else 0)) ^ 2]
  set c := fun p : G × G => cos ⟪intVec (divisionRow p), ψ⟫ with hc
  have h : ∀ p : G × G, c p *
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

/-- In the Klein four-group `ℤ/2 × ℤ/2`, at the phases `0` every relation is
`0`, so the diagonal of the Hessian of `V_full` is `3 · 4 - 4 + 2 = 10`. -/
example (t : ZMod 2 × ZMod 2) :
    ⟪fderiv ℝ (gradient (divisionEnergy
        (Finset.univ : Finset ((ZMod 2 × ZMod 2) × (ZMod 2 × ZMod 2))))) 0
        (PiLp.single 2 t 1), PiLp.single 2 t 1⟫ = 10 := by
  rw [inner_fderiv_gradient_divisionEnergy_single]
  simp only [WithLp.ofLp_zero, Pi.zero_apply, sub_zero, mul_zero, cos_zero, Finset.sum_const,
    Finset.card_univ, Fintype.card_prod, ZMod.card, nsmul_eq_mul, mul_one]
  norm_num

end Transformer.Landscape.PhaseModel.Abelian
