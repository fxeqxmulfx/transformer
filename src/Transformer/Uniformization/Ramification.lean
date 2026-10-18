/-
# Local uniformization: ramified coordinates

The maps `ram w' q b t = w' + (±t_j^q)_j` with signs `b`. Every point `x'` with
`|x'_j - w'_j| ≤ τ^q` is `ram w' q b t` for the signs of `x' - w'` and
`t_j = |x'_j - w'_j|^{1/q} ≤ τ`, and a monomial in `ram w' q b t - w'` is a sign
times a monomial in `t`. Auxiliary for the Newton step of local uniformization for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Uniformization.NormalCrossing
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base)

variable {n : ℕ}

/-- The sign vector `(±1)_j` of `b`. -/
def signVec (b : Fin n → Bool) : Base n := fun j => if b j then 1 else -1

theorem abs_signVec (b : Fin n → Bool) (j : Fin n) : |signVec b j| = 1 := by
  unfold signVec
  split_ifs <;> simp

theorem abs_monomial_signVec (b : Fin n → Bool) (α : Fin n → ℕ) :
    |monomial α (signVec b)| = 1 := by
  rw [monomial, Finset.abs_prod]
  simp [abs_pow, abs_signVec]

theorem monomial_signVec_ne_zero (b : Fin n → Bool) (α : Fin n → ℕ) :
    monomial α (signVec b) ≠ 0 := by
  intro h
  have h1 := abs_monomial_signVec b α
  rw [h, abs_zero] at h1
  exact zero_ne_one h1

/-- The ramified coordinates `w' + (±t_j^q)_j`. -/
def ram (w' : Base n) (q : ℕ) (b : Fin n → Bool) (t : Base n) : Base n :=
  w' + fun j => signVec b j * t j ^ q

/-- A monomial in ramified coordinates is a sign times a monomial. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem monomial_ram_sub (w' : Base n) (q : ℕ) (b : Fin n → Bool) (α : Fin n → ℕ)
    (t : Base n) :
    monomial α (ram w' q b t - w') = monomial α (signVec b) * monomial (q • α) t := by
  have h : ram w' q b t - w' = signVec b * fun j => t j ^ q := by
    funext j
    simp [ram]
  rw [h, monomial_mul]
  simp only [monomial, Pi.smul_apply, smul_eq_mul, pow_mul]

theorem analyticAt_ram (w' : Base n) (q : ℕ) (b : Fin n → Bool) (t : Base n) :
    AnalyticAt ℝ (ram w' q b) t := by
  refine analyticAt_const.add (analyticAt_pi_iff.2 fun j => ?_)
  exact analyticAt_const.mul
    (((ContinuousLinearMap.proj j : Base n →L[ℝ] ℝ).analyticAt t).pow q)

theorem continuous_ram (w' : Base n) (q : ℕ) (b : Fin n → Bool) :
    Continuous (ram w' q b) :=
  continuous_iff_continuousAt.2 fun t => (analyticAt_ram w' q b t).continuousAt

/-- Ramified coordinates of size `τ` stay within `τ ^ q` of `w'`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_ram_sub_le (w' : Base n) (q : ℕ) (b : Fin n → Bool) {t : Base n} {τ : ℝ}
    (hτ : 0 ≤ τ) (ht : ∀ j, |t j| ≤ τ) : ‖ram w' q b t - w'‖ ≤ τ ^ q := by
  refine (pi_norm_le_iff_of_nonneg (pow_nonneg hτ q)).2 fun j => ?_
  simp only [ram, add_sub_cancel_left, Real.norm_eq_abs, abs_mul, abs_signVec, one_mul,
    abs_pow]
  exact pow_le_pow_left₀ (abs_nonneg _) (ht j) q

/-- Every point within `τ ^ q` of `w'` in each coordinate is in ramified
coordinates of size `τ`. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem exists_ram_eq (w' x' : Base n) {q : ℕ} (hq : q ≠ 0) {τ : ℝ} (hτ : 0 ≤ τ)
    (hx : ∀ j, |x' j - w' j| ≤ τ ^ q) : ∃ b t, ram w' q b t = x' ∧ ∀ j, |t j| ≤ τ := by
  refine ⟨fun j => decide (w' j ≤ x' j), fun j => |x' j - w' j| ^ ((q : ℝ)⁻¹), ?_,
    fun j => ?_⟩
  · funext j
    simp only [ram, Pi.add_apply, signVec, Real.rpow_inv_natCast_pow (abs_nonneg _) hq]
    by_cases h : w' j ≤ x' j
    · simp [h, abs_of_nonneg (sub_nonneg.2 h)]
    · simp [h, abs_of_neg (sub_neg.2 (not_le.1 h))]
  · have h0 : 0 ≤ |x' j - w' j| ^ ((q : ℝ)⁻¹) := Real.rpow_nonneg (abs_nonneg _) _
    rw [abs_of_nonneg h0, ← pow_le_pow_iff_left₀ h0 hτ hq,
      Real.rpow_inv_natCast_pow (abs_nonneg _) hq]
    exact hx j

end Transformer.Uniformization
