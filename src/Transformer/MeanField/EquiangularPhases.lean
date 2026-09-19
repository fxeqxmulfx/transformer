/-
# The equiangular attention weights at `β_n = γ log n`: the three phases

The diagonal weight `a = e^β / Z` of the equiangular configuration satisfies
`a = 1 / (1 + (n-1) e^{β(ρ-1)})`, so at `β_n = γ log n` its behaviour is set by
the exponent `1 - γ(1-ρ)` of `n`: the auxiliary quantity `(n-1) e^{β(ρ-1)}`
diverges, tends to `1`, or vanishes, and `a` tends to `0`, `1/2` or `1`.

That trichotomy is the whole content of the phase transition; the reduction of
the cosine to `a` is in `Transformer.MeanField.EquiangularLimit`, which proves
the theorem on top of this file.

Source: arXiv:2512.01868v4, §6, `thm: long-context-phase-transition`.
-/

import Transformer.MeanField.EquiangularWeights

open scoped BigOperators
open Filter Topology

namespace Transformer
namespace MeanField

/-! ### The limit of the diagonal weight `a` -/

/-- `a = 1 / (1 + (n-1) e^{β(ρ-1)})`: dividing through by `e^β` leaves a
quantity in which only the *gap* `β(1-ρ)` appears. -/
theorem equiDiag_eq_one_div (n : ℕ) (β ρ : ℝ) :
    equiDiag n β ρ = 1 / (1 + ((n : ℝ) - 1) * Real.exp (β * ρ - β)) := by
  have he := Real.exp_pos β
  have hd : 1 + ((n : ℝ) - 1) * (Real.exp (β * ρ) / Real.exp β)
      = (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * ρ)) / Real.exp β := by
    field_simp
  rw [equiDiag, Real.exp_sub, hd, one_div_div]

/-- At `β_n = γ log n` the quantity `(n-1) e^{β(ρ-1)}` becomes
`((m+1)/(m+2)) · e^{(1 - γ(1-ρ)) log(m+2)}`: the exponent of `n` is
`1 - γ(1-ρ)`, and its sign is the phase transition. -/
theorem equiDiag_seq_eq (γ ρ : ℝ) (m : ℕ) :
    equiDiag (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ
      = 1 / (1 + (((m : ℝ) + 1) / ((m : ℝ) + 2))
          * Real.exp ((1 - γ * (1 - ρ)) * Real.log ((m : ℝ) + 2))) := by
  have hcast : ((m + 2 : ℕ) : ℝ) = (m : ℝ) + 2 := by push_cast; ring
  have hpos : (0 : ℝ) < (m : ℝ) + 2 := by positivity
  set L := Real.log ((m : ℝ) + 2) with hLdef
  have hexp : Real.exp L = (m : ℝ) + 2 := by rw [hLdef, Real.exp_log hpos]
  rw [equiDiag_eq_one_div _ _ _, hcast,
    show (m : ℝ) + 2 - 1 = (m : ℝ) + 1 from by ring]
  congr 2
  rw [show (1 - γ * (1 - ρ)) * L = L + (γ * L * ρ - γ * L) by ring, Real.exp_add, hexp]
  field_simp

/-- `log (m+2) → ∞`. -/
theorem tendsto_log_natAdd_two :
    Tendsto (fun m : ℕ => Real.log ((m : ℝ) + 2)) atTop atTop := by
  exact Real.tendsto_log_atTop.comp
    (tendsto_atTop_add_const_right _ 2 tendsto_natCast_atTop_atTop)

/-- `(m+1)/(m+2) → 1`, and it stays in `[1/2, 1]`. -/
theorem tendsto_ratio_natAdd :
    Tendsto (fun m : ℕ => ((m : ℝ) + 1) / ((m : ℝ) + 2)) atTop (nhds 1) := by
  have hcast : Tendsto (fun m : ℕ => (m : ℝ) + 2) atTop atTop :=
    tendsto_atTop_add_const_right _ 2 tendsto_natCast_atTop_atTop
  have hinv : Tendsto (fun m : ℕ => 1 / ((m : ℝ) + 2)) atTop (nhds 0) := by
    simpa [Pi.inv_def, one_div] using hcast.inv_tendsto_atTop
  have heq : ∀ m : ℕ, ((m : ℝ) + 1) / ((m : ℝ) + 2) = 1 - 1 / ((m : ℝ) + 2) := by
    intro m
    have : (0 : ℝ) < (m : ℝ) + 2 := by positivity
    field_simp
    ring
  simpa [heq] using tendsto_const_nhds.sub hinv

theorem ratio_natAdd_mem (m : ℕ) :
    1 / 2 ≤ ((m : ℝ) + 1) / ((m : ℝ) + 2) ∧ ((m : ℝ) + 1) / ((m : ℝ) + 2) ≤ 1 := by
  have hpos : (0 : ℝ) < (m : ℝ) + 2 := by positivity
  have hm : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  constructor
  · rw [le_div_iff₀ hpos]; linarith
  · rw [div_le_one hpos]; linarith


/-! ### The three phases

Write `c = γ(1-ρ)`.  The exponent of `n` in `(n-1) e^{β(ρ-1)}` is `1 - c`, so
that quantity diverges, converges to `1`, or vanishes according to the sign of
`1 - c`, and `a = 1/(1 + ·)` tends to `0`, `1/2` or `1`. -/

/-- **Subcritical, `γ(1-ρ) < 1`:** the diagonal weight vanishes -- the
attention matrix spreads out uniformly. -/
theorem tendsto_equiDiag_of_lt (γ ρ : ℝ) (h : γ * (1 - ρ) < 1) :
    Tendsto (fun m : ℕ => equiDiag (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ)
      atTop (nhds 0) := by
  have hE : Tendsto
      (fun m : ℕ => Real.exp ((1 - γ * (1 - ρ)) * Real.log ((m : ℝ) + 2)))
      atTop atTop :=
    Real.tendsto_exp_atTop.comp
      (tendsto_log_natAdd_two.const_mul_atTop (by linarith))
  have hM : Tendsto
      (fun m : ℕ => 1 + ((m : ℝ) + 1) / ((m : ℝ) + 2)
        * Real.exp ((1 - γ * (1 - ρ)) * Real.log ((m : ℝ) + 2))) atTop atTop := by
    refine tendsto_atTop_add_const_left _ 1 (tendsto_atTop_mono (fun m => ?_)
      (hE.const_mul_atTop (by norm_num : (0 : ℝ) < 1 / 2)))
    exact mul_le_mul_of_nonneg_right (ratio_natAdd_mem m).1 (Real.exp_pos _).le
  have hinv := hM.inv_tendsto_atTop
  exact Filter.Tendsto.congr (fun m => by rw [Pi.inv_apply, equiDiag_seq_eq, one_div]) hinv

/-- **Critical, `γ(1-ρ) = 1`:** the diagonal weight tends to `1/2` -- a token
splits its attention evenly between itself and everything else. -/
theorem tendsto_equiDiag_of_eq (γ ρ : ℝ) (h : γ * (1 - ρ) = 1) :
    Tendsto (fun m : ℕ => equiDiag (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ)
      atTop (nhds (1 / 2)) := by
  have hM : Tendsto
      (fun m : ℕ => 1 + ((m : ℝ) + 1) / ((m : ℝ) + 2)
        * Real.exp ((1 - γ * (1 - ρ)) * Real.log ((m : ℝ) + 2)))
      atTop (nhds 2) := by
    have heq : ∀ m : ℕ, 1 + ((m : ℝ) + 1) / ((m : ℝ) + 2)
        = 1 + ((m : ℝ) + 1) / ((m : ℝ) + 2)
          * Real.exp ((1 - γ * (1 - ρ)) * Real.log ((m : ℝ) + 2)) := by
      intro m; rw [h]; norm_num
    have h2 : Tendsto (fun m : ℕ => 1 + ((m : ℝ) + 1) / ((m : ℝ) + 2)) atTop (nhds (1 + 1)) :=
      tendsto_const_nhds.add tendsto_ratio_natAdd
    have h3 : (1 : ℝ) + 1 = 2 := by norm_num
    rw [h3] at h2
    exact Filter.Tendsto.congr heq h2
  have hdiv := (tendsto_const_nhds (x := (1 : ℝ)) (f := atTop (α := ℕ))).div hM
    (by norm_num : (2 : ℝ) ≠ 0)
  exact Filter.Tendsto.congr (fun m => (equiDiag_seq_eq γ ρ m).symm) hdiv

/-- **Supercritical, `γ(1-ρ) > 1`:** the diagonal weight tends to `1` -- each
token attends only to itself, so nothing moves. -/
theorem tendsto_equiDiag_of_gt (γ ρ : ℝ) (h : 1 < γ * (1 - ρ)) :
    Tendsto (fun m : ℕ => equiDiag (m + 2) (γ * Real.log ((m : ℝ) + 2)) ρ)
      atTop (nhds 1) := by
  have hE : Tendsto
      (fun m : ℕ => Real.exp ((1 - γ * (1 - ρ)) * Real.log ((m : ℝ) + 2)))
      atTop (nhds 0) :=
    Real.tendsto_exp_atBot.comp
      (tendsto_log_natAdd_two.const_mul_atTop_of_neg (by linarith))
  have hM : Tendsto
      (fun m : ℕ => 1 + ((m : ℝ) + 1) / ((m : ℝ) + 2)
        * Real.exp ((1 - γ * (1 - ρ)) * Real.log ((m : ℝ) + 2)))
      atTop (nhds 1) := by
    simpa using
      (tendsto_const_nhds (x := (1 : ℝ)) (f := atTop (α := ℕ))).add
        (tendsto_ratio_natAdd.mul hE)
  have hdiv := (tendsto_const_nhds (x := (1 : ℝ)) (f := atTop (α := ℕ))).div hM
    (by norm_num : (1 : ℝ) ≠ 0)
  have h1 : (1 : ℝ) / 1 = 1 := by norm_num
  rw [h1] at hdiv
  exact Filter.Tendsto.congr (fun m => (equiDiag_seq_eq γ ρ m).symm) hdiv

end MeanField
end Transformer
