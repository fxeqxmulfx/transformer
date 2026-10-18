/-
# Phase model over a finite abelian group — critical points near `0`

As for `ℤ/n` (`PhaseModel.Basin`), with `n = |G| ≥ 4`: let `ψ` be a critical
point of the energy `V_full` of the full division table of `G` where the
Hessian has a nonnegative diagonal, as at a local minimizer.  If
`E = Σ_x (1 - cos ψ_x) < n/8`, then `cos ψ_t > 1/2` for every `t`
(`half_lt_cos_of_gradient_eq_zero`).  With `θ_{x,y} = ψ_x - ψ_y - ψ_{x-y}`
and `λ_t = Σ_y cos θ_{t,y} + Σ_x cos θ_{x,t} + Σ_y cos θ_{t+y,y}`, the
identity `λ_t cos ψ_t + sin ψ_t ∂_t V = Σ_y cos (ψ_y + ψ_{t-y}) +
Σ_x cos (ψ_x - ψ_{x-t}) + Σ_y cos (ψ_{t+y} - ψ_y)` and the chord inequality
give `λ_t cos ψ_t ≥ 3n - 12E > 3n/2`, while `λ_t ≤ 3n` and, by the Hessian,
`λ_t ≥ -6`.  Each `ψ_t` is then within `π/3` of `2πℤ`, every relation of
`ψ` within `π` of `2πℤ`, and a critical point with every `cos ψ_t > 1/2` is
a period (`mem_periods_of_forall_half_lt_cos`).
-/

import Transformer.PhaseModel.Abelian.Coordinates

open Real

namespace Transformer.PhaseModel.Abelian

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]

/-- **Near the zero `0`, a critical point of `V_full` with a nonnegative
Hessian diagonal has every `cos ψ_t > 1/2`**, for `|G| ≥ 4`, when
`Σ_x (1 - cos ψ_x) < |G|/8`. -/
theorem half_lt_cos_of_gradient_eq_zero (hn : 4 ≤ Fintype.card G) {ψ : EuclideanSpace ℝ G}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (G × G))) ψ = 0)
    (hH : ∀ t, 0 ≤ ⟪fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (G × G))))
      ψ (PiLp.single 2 t 1), PiLp.single 2 t 1⟫)
    (hE : ∑ x, (1 - cos (ψ.ofLp x)) < Fintype.card G / 8) (t : G) :
    1 / 2 < cos (ψ.ofLp t) := by
  have hgt := inner_gradient_divisionEnergy_single ψ t
  rw [hg, inner_zero_left] at hgt
  have hHt := hH t
  rw [inner_fderiv_gradient_divisionEnergy_single] at hHt
  set A := ∑ y, cos (ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y)) with hA
  set B := ∑ x, cos (ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t)) with hB
  set C := ∑ y, cos (ψ.ofLp (t + y) - ψ.ofLp y - ψ.ofLp t) with hC
  have hle : ∀ f : G → ℝ, ∑ y, cos (f y) ≤ Fintype.card G := fun f => by
    calc ∑ y, cos (f y) ≤ ∑ _y : G, (1 : ℝ) := Finset.sum_le_sum fun y _ => cos_le_one _
      _ = Fintype.card G := by simp
  have h1 : A * cos (ψ.ofLp t) + sin (ψ.ofLp t) * ∑ y, sin (ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y)) =
      ∑ y, cos (ψ.ofLp y + ψ.ofLp (t - y)) := by
    rw [hA, Finset.sum_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [show ψ.ofLp y + ψ.ofLp (t - y) = -(ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y) - ψ.ofLp t) by ring,
      cos_neg, cos_sub (ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y))]
    ring
  have h2 : B * cos (ψ.ofLp t) - sin (ψ.ofLp t) * ∑ x, sin (ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t)) =
      ∑ x, cos (ψ.ofLp x - ψ.ofLp (x - t)) := by
    rw [hB, Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [show ψ.ofLp x - ψ.ofLp (x - t) = ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t) + ψ.ofLp t by ring,
      cos_add (ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t))]
    ring
  have h3 : C * cos (ψ.ofLp t) - sin (ψ.ofLp t) * ∑ y, sin (ψ.ofLp (t + y) - ψ.ofLp y - ψ.ofLp t) =
      ∑ y, cos (ψ.ofLp (t + y) - ψ.ofLp y) := by
    rw [hC, Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun y _ => ?_
    have h := cos_add (ψ.ofLp (t + y) - ψ.ofLp y - ψ.ofLp t) (ψ.ofLp t)
    rw [sub_add_cancel] at h
    rw [h]
    ring
  have e1 : ∑ y, (1 - cos (ψ.ofLp (t - y))) = ∑ x, (1 - cos (ψ.ofLp x)) :=
    Equiv.sum_comp (Equiv.subLeft t) fun y => 1 - cos (ψ.ofLp y)
  have e2 : ∑ x, (1 - cos (ψ.ofLp (x - t))) = ∑ x, (1 - cos (ψ.ofLp x)) :=
    Equiv.sum_comp (Equiv.subRight t) fun y => 1 - cos (ψ.ofLp y)
  have e3 : ∑ y, (1 - cos (ψ.ofLp (t + y))) = ∑ x, (1 - cos (ψ.ofLp x)) :=
    Equiv.sum_comp (Equiv.addLeft t) fun y => 1 - cos (ψ.ofLp y)
  have c1 := sub_le_sum_cos_add (fun y => ψ.ofLp y) fun y => ψ.ofLp (t - y)
  have c2 := sub_le_sum_cos_sub (fun x => ψ.ofLp x) fun x => ψ.ofLp (x - t)
  have c3 := sub_le_sum_cos_sub (fun y => ψ.ofLp (t + y)) fun y => ψ.ofLp y
  have hL : (A + B + C) * cos (ψ.ofLp t) = ∑ y, cos (ψ.ofLp y + ψ.ofLp (t - y)) +
      ∑ x, cos (ψ.ofLp x - ψ.ofLp (x - t)) + ∑ y, cos (ψ.ofLp (t + y) - ψ.ofLp y) := by
    linear_combination h1 + h2 + h3 + sin (ψ.ofLp t) * hgt
  have hlow : -6 ≤ A + B + C := by
    linarith [neg_one_le_cos (ψ.ofLp 0), cos_le_one (ψ.ofLp (t + t) - 2 * ψ.ofLp t)]
  have hup : A + B + C ≤ 3 * Fintype.card G := by
    linarith [hle fun y => ψ.ofLp t - ψ.ofLp y - ψ.ofLp (t - y),
      hle fun x => ψ.ofLp x - ψ.ofLp t - ψ.ofLp (x - t),
      hle fun y => ψ.ofLp (t + y) - ψ.ofLp y - ψ.ofLp t]
  have hn' : (4 : ℝ) ≤ Fintype.card G := by exact_mod_cast hn
  have hrot : 3 * Fintype.card G - 12 * ∑ x, (1 - cos (ψ.ofLp x)) ≤
      (A + B + C) * cos (ψ.ofLp t) := by
    rw [hL]
    simp only at c1 c2 c3
    linarith
  by_contra hc
  push Not at hc
  rcases le_or_gt (cos (ψ.ofLp t)) 0 with h0 | h0
  · nlinarith [mul_nonneg (by linarith : 0 ≤ A + B + C + 6) (by linarith : 0 ≤ -cos (ψ.ofLp t)),
      neg_one_le_cos (ψ.ofLp t)]
  · nlinarith [mul_nonneg (sub_nonneg.2 hup) h0.le]

/-- **A critical point of `V_full` with every `cos ψ_t > 1/2` is a period**:
rounding each `ψ_t` to `2πℤ` leaves a remainder below `π/3`, every relation
of `ψ` is then within `π` of that of a period, a zero of `V_full`, so it
vanishes (`inner_sub_eq_zero_of_gradient_eq_zero`), and only `0` is
orthogonal to every row. -/
theorem mem_periods_of_forall_half_lt_cos {ψ : EuclideanSpace ℝ G}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (G × G))) ψ = 0)
    (hc : ∀ t, 1 / 2 < cos (ψ.ofLp t)) : ψ ∈ periods G := by
  set v : EuclideanSpace ℝ G :=
    WithLp.toLp 2 fun t => (round (ψ.ofLp t / (2 * π)) : ℝ) * (2 * π) with hv
  have hvp : v ∈ periods G := fun t => ⟨round (ψ.ofLp t / (2 * π)), rfl⟩
  have hφ : ∀ t, |ψ.ofLp t - v.ofLp t| < π / 3 := by
    intro t
    have h1 : |ψ.ofLp t - v.ofLp t| ≤ π := by
      have hr := abs_sub_round (ψ.ofLp t / (2 * π))
      have he : ψ.ofLp t - v.ofLp t =
          2 * π * (ψ.ofLp t / (2 * π) - round (ψ.ofLp t / (2 * π))) := by
        simp only [hv, WithLp.ofLp_toLp]
        field_simp
      rw [he, abs_mul, abs_of_pos (by positivity)]
      nlinarith [pi_pos]
    have h2 : cos (ψ.ofLp t - v.ofLp t) = cos (ψ.ofLp t) := by
      simp only [hv, WithLp.ofLp_toLp]
      exact cos_sub_int_mul_two_pi _ _
    by_contra h3
    push Not at h3
    have h4 := cos_le_cos_of_nonneg_of_le_pi (by positivity) h1 h3
    rw [cos_abs, h2, cos_pi_div_three] at h4
    linarith [hc t]
  have hz : divisionEnergy (Finset.univ : Finset (G × G)) v = 0 :=
    (phaseEnergy_eq_zero_iff _ v).2 fun r => exists_inner_intVec_eq _ hvp
  have hrel : ∀ r : (Finset.univ : Finset (G × G)),
      |⟪intVec (divisionRow r.1), ψ - v⟫| < π := by
    intro r
    rw [inner_intVec_divisionRow]
    simp only [WithLp.ofLp_sub, Pi.sub_apply]
    have h1 := abs_sub (ψ.ofLp r.1.1 - v.ofLp r.1.1 - (ψ.ofLp r.1.2 - v.ofLp r.1.2))
      (ψ.ofLp (r.1.1 - r.1.2) - v.ofLp (r.1.1 - r.1.2))
    have h2 := abs_sub (ψ.ofLp r.1.1 - v.ofLp r.1.1) (ψ.ofLp r.1.2 - v.ofLp r.1.2)
    linarith [hφ r.1.1, hφ r.1.2, hφ (r.1.1 - r.1.2)]
  have h0 := inner_sub_eq_zero_of_gradient_eq_zero
    (fun p : (Finset.univ : Finset (G × G)) => intVec (divisionRow p.1)) hg hz hrel
  have hψv := eq_zero_of_forall_inner_divisionRow (ψ - v) fun p => h0 ⟨p, Finset.mem_univ p⟩
  rw [sub_eq_zero] at hψv
  rw [hψv]
  exact hvp

/-- The hypotheses of the theorems above hold in the Klein four-group
`ℤ/2 × ℤ/2` at the zero `ψ = 0`, where every `cos ψ_t = 1` and the Hessian
is `h ↦ Σ_r ⟪m_r, h⟫ m_r`. -/
example : let V := divisionEnergy (Finset.univ : Finset ((ZMod 2 × ZMod 2) × (ZMod 2 × ZMod 2)))
    4 ≤ Fintype.card (ZMod 2 × ZMod 2) ∧ gradient V 0 = 0 ∧
      (∀ t, 0 ≤ ⟪fderiv ℝ (gradient V) 0 (PiLp.single 2 t 1), PiLp.single 2 t 1⟫) ∧
      ∑ x, (1 - cos ((0 : EuclideanSpace ℝ (ZMod 2 × ZMod 2)).ofLp x)) <
        (Fintype.card (ZMod 2 × ZMod 2) : ℝ) / 8 ∧
      ∀ t, 1 / 2 < cos ((0 : EuclideanSpace ℝ (ZMod 2 × ZMod 2)).ofLp t) := by
  intro V
  have h0 : V 0 = 0 := by simp [V, divisionEnergy, phaseEnergy]
  refine ⟨by simp, gradient_eq_zero_of_phaseEnergy_eq_zero _ h0, fun t => ?_,
    by simp, fun t => by norm_num⟩
  simp only [V, divisionEnergy]
  rw [inner_fderiv_gradient_phaseEnergy]
  exact Finset.sum_nonneg fun r _ => by
    rw [inner_zero_right, cos_zero, one_mul]
    exact sq_nonneg _

end Transformer.PhaseModel.Abelian
