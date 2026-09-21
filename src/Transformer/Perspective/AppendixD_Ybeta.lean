/-
# Appendix D — the decay of `1 - γ_β(t)`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:ybetacloseto1`, the Grönwall estimate for the scalar equation `eq: ybeta`
itself.  It is proved here, on the range its own constant points at, from
`eq: ybeta` alone: the properties of the solution the survey's proof uses —
`γ_β` stays in `[0, 1]`, increases, and has passed `1/2` by time `n e^β / 2` —
are in `Perspective.AppendixD_YbetaInvariance`.

`not_forall_ybeta_close_to_1` refutes the form the survey prints, which puts
the estimate on all of `t ≥ 0`.

The stability estimates of the first half of Appendix D are in
`Perspective.AppendixD_PhaseTransition`.
-/

import Transformer.Perspective.AppendixD_YbetaInvariance
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Complex.ExponentialBounds

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- **Equation (e:ybetacloseto1).**

  `1 - γ_β(t) ≤ (1/2) exp( n² e^β / (2(n + e^{β/2})) - n t / (n + e^{β/2}) )`.

**What the source says and what is changed here.**

*The range is `t ≥ n e^β / 2`, not `t ≥ 0`.*  The survey's right-hand side is
`(1/2) exp(-λ (t - n e^β/2))` with `λ = n/(n + e^{β/2})`, so at `t = n e^β/2`
it asserts `γ_β ≥ 1/2` and before that time it asserts less than `1` only by
the width of the exponential.  At `n = 2`, `β = 0`, where `eq: ybeta` is
solved in closed form by `tanh`, the printed form already fails at `t = 0`:
it reads `1 ≤ (1/2) e^{2/3} ≈ 0.974`.  This is `not_forall_ybeta_close_to_1`.

*`n ≥ 1` instead of `n ≥ 2`.*  The theorem the estimate serves fixes `n ≥ 2`;
the proof needs only `n ≥ 1`.

*The proof is the survey's.*  `γ_β` stays in `[0, 1]` and is nondecreasing
(`ybetaODE_SA.mem_Ico`, `ybetaODE_SA.monotoneOn`), and it has passed `1/2` at
`n e^β / 2` (`ybetaODE_SA.half_le`), hence `γ_β ≥ 1/2` past
`n e^β / 2`, hence `e^{β γ_β} ≥ e^{β/2}` and `(n-1) γ_β + 1 ≥ (n+1)/2`, and
the rate

  `2 e^{βγ} ((n-1)γ + 1) / (e^β + (n-1) e^{βγ})  ≥  (n+1)/(n - 1 + e^{β/2})
                                                 ≥  n/(n + e^{β/2}) = λ`

is the survey's `λ`.  The constant `n² e^β / (2(n + e^{β/2}))` is exactly
`λ · n e^β / 2`, so nothing is lost.

Source: arXiv:2312.10794v5, Appendix D, `e:ybetacloseto1`. -/
theorem ybeta_close_to_1 (hn : 0 < n) (β : ℝ) (hβ : 0 ≤ β) (γ : ℝ → ℝ)
    (hγ : ybetaODE_SA n β γ) :
    ∀ t : ℝ, (n : ℝ) * Real.exp β / 2 ≤ t →
      1 - γ t
        ≤ (1/2 : ℝ) * Real.exp
            ((n : ℝ)^2 * Real.exp β
                / (2 * ((n : ℝ) + Real.exp (β / 2)))
              - ((n : ℝ) * t) / ((n : ℝ) + Real.exp (β / 2))) := by
  have hinv : ∀ t : ℝ, 0 ≤ t → 0 ≤ γ t ∧ γ t ≤ 1 := fun t ht =>
    ⟨(hγ.mem_Ico hn t ht).1, (hγ.mem_Ico hn t ht).2.le⟩
  have hhalf : (1/2 : ℝ) ≤ γ ((n : ℝ) * Real.exp β / 2) := hγ.half_le hn hβ
  have hnR : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hqpos : (0 : ℝ) < Real.exp (β / 2) := Real.exp_pos _
  have hEb : (0 : ℝ) < Real.exp β := Real.exp_pos _
  have hq2 : Real.exp (β / 2) ^ 2 = Real.exp β := by
    rw [sq, ← Real.exp_add]; congr 1; ring
  have hnq : ((n : ℝ) + Real.exp (β / 2)) ≠ 0 := by positivity
  set T0 : ℝ := (n : ℝ) * Real.exp β / 2 with hT0def
  have hT0nn : (0 : ℝ) ≤ T0 := by rw [hT0def]; positivity
  set lam : ℝ := (n : ℝ) / ((n : ℝ) + Real.exp (β / 2)) with hlamdef
  have hlampos : 0 < lam := by rw [hlamdef]; positivity
  have hderiv : ∀ t : ℝ, HasDerivAt γ
      (2 * Real.exp (β * γ t) * (1 - γ t) * (((n : ℝ) - 1) * γ t + 1)
        / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ t))) t := hγ.2
  have hDpos : ∀ t : ℝ,
      (0 : ℝ) < Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ t) := by
    intro t
    have : (0 : ℝ) ≤ ((n : ℝ) - 1) * Real.exp (β * γ t) :=
      mul_nonneg (by linarith) (Real.exp_pos _).le
    linarith
  have hmono : MonotoneOn γ (Set.Ici (0 : ℝ)) := hγ.monotoneOn hn
  have hhalft : ∀ t : ℝ, T0 ≤ t → (1/2 : ℝ) ≤ γ t := fun t ht =>
    hhalf.trans (hmono (Set.mem_Ici.mpr hT0nn) (Set.mem_Ici.mpr (hT0nn.trans ht)) ht)
  -- past the half-way time the rate is at least `lam`
  have hrate : ∀ t : ℝ, T0 ≤ t →
      lam * (1 - γ t)
        ≤ 2 * Real.exp (β * γ t) * (1 - γ t) * (((n : ℝ) - 1) * γ t + 1)
            / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ t)) := by
    intro t ht
    obtain ⟨hg0, hg1⟩ := hinv t (hT0nn.trans ht)
    have hgh : (1/2 : ℝ) ≤ γ t := hhalft t ht
    have hspos : (0 : ℝ) < Real.exp (β * γ t) := Real.exp_pos _
    have hsq : Real.exp (β / 2) ≤ Real.exp (β * γ t) :=
      Real.exp_le_exp.mpr (by nlinarith [mul_nonneg hβ (by linarith : (0 : ℝ) ≤ γ t - 1/2)])
    have hD : (0 : ℝ) < Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ t) := hDpos t
    have hkey : lam
        ≤ 2 * Real.exp (β * γ t) * (((n : ℝ) - 1) * γ t + 1)
            / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ t)) := by
      rw [hlamdef, div_le_div_iff₀ (by positivity) hD]
      have hA : Real.exp (β * γ t) * ((n : ℝ) + 1) * ((n : ℝ) + Real.exp (β / 2))
          ≤ 2 * Real.exp (β * γ t) * (((n : ℝ) - 1) * γ t + 1)
              * ((n : ℝ) + Real.exp (β / 2)) := by
        nlinarith [mul_nonneg
          (mul_nonneg (by linarith : (0 : ℝ) ≤ (n : ℝ) - 1)
            (by linarith : (0 : ℝ) ≤ 2 * γ t - 1))
          (by positivity :
            (0 : ℝ) ≤ Real.exp (β * γ t) * ((n : ℝ) + Real.exp (β / 2)))]
      have hB : (n : ℝ) * (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ t))
          ≤ Real.exp (β * γ t) * ((n : ℝ) + 1) * ((n : ℝ) + Real.exp (β / 2)) := by
        rw [← hq2]
        nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ Real.exp (β * γ t) - Real.exp (β / 2))
          (by positivity :
            (0 : ℝ) ≤ (n : ℝ) * Real.exp (β / 2) + 2 * (n : ℝ) + Real.exp (β / 2))]
      linarith
    calc lam * (1 - γ t)
        ≤ (2 * Real.exp (β * γ t) * (((n : ℝ) - 1) * γ t + 1)
            / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ t))) * (1 - γ t) :=
          mul_le_mul_of_nonneg_right hkey (by linarith)
      _ = 2 * Real.exp (β * γ t) * (1 - γ t) * (((n : ℝ) - 1) * γ t + 1)
            / (Real.exp β + ((n : ℝ) - 1) * Real.exp (β * γ t)) := by ring
  -- Grönwall: `(1 - γ) e^{lam ·}` is nonincreasing past `T0`
  have hzderiv : ∀ t : ℝ, T0 ≤ t →
      ∃ c : ℝ, HasDerivAt (fun r : ℝ => (1 - γ r) * Real.exp (lam * r)) c t ∧ c ≤ 0 := by
    intro t ht
    obtain ⟨R, hR, hRle⟩ : ∃ R : ℝ, HasDerivAt γ R t ∧ lam * (1 - γ t) ≤ R :=
      ⟨_, hderiv t, hrate t ht⟩
    have hexp : HasDerivAt (fun r : ℝ => Real.exp (lam * r))
        (Real.exp (lam * t) * lam) t := by
      simpa using ((hasDerivAt_id t).const_mul lam).exp
    refine ⟨(0 - R) * Real.exp (lam * t) + (1 - γ t) * (Real.exp (lam * t) * lam),
      ((hasDerivAt_const t (1 : ℝ)).sub hR).mul hexp, ?_⟩
    nlinarith [mul_nonneg (Real.exp_pos (lam * t)).le (sub_nonneg.mpr hRle)]
  have hz_anti : AntitoneOn (fun r : ℝ => (1 - γ r) * Real.exp (lam * r))
      (Set.Ici T0) := by
    refine antitoneOn_of_deriv_nonpos (convex_Ici _) ?_ ?_ ?_
    · intro s hs
      obtain ⟨c, hc, -⟩ := hzderiv s (Set.mem_Ici.mp hs)
      exact hc.continuousAt.continuousWithinAt
    · intro s hs
      rw [interior_Ici] at hs
      obtain ⟨c, hc, -⟩ := hzderiv s (Set.mem_Ioi.mp hs).le
      exact hc.differentiableAt.differentiableWithinAt
    · intro s hs
      rw [interior_Ici] at hs
      obtain ⟨c, hc, hcle⟩ := hzderiv s (Set.mem_Ioi.mp hs).le
      rw [hc.deriv]
      exact hcle
  intro t ht
  have hkey : (1 - γ t) * Real.exp (lam * t) ≤ (1 - γ T0) * Real.exp (lam * T0) :=
    hz_anti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr ht) ht
  have hEt : (0 : ℝ) < Real.exp (lam * t) := Real.exp_pos _
  have hstep : 1 - γ t ≤ (1/2 : ℝ) * Real.exp (lam * (T0 - t)) := by
    refine le_of_mul_le_mul_right ?_ hEt
    have hprod : (1/2 : ℝ) * Real.exp (lam * (T0 - t)) * Real.exp (lam * t)
        = (1/2 : ℝ) * Real.exp (lam * T0) := by
      rw [mul_assoc, ← Real.exp_add]; congr 2; ring
    rw [hprod]
    calc (1 - γ t) * Real.exp (lam * t)
        ≤ (1 - γ T0) * Real.exp (lam * T0) := hkey
      _ ≤ (1/2 : ℝ) * Real.exp (lam * T0) := by
          nlinarith [Real.exp_pos (lam * T0)]
  refine hstep.trans (le_of_eq ?_)
  have hexpeq : lam * (T0 - t)
      = (n : ℝ)^2 * Real.exp β / (2 * ((n : ℝ) + Real.exp (β / 2)))
        - ((n : ℝ) * t) / ((n : ℝ) + Real.exp (β / 2)) := by
    rw [hlamdef, hT0def]
    field_simp
  rw [hexpeq]

/-- The hypotheses of `ybeta_close_to_1` are satisfiable: `tanh` solves
`eq: ybeta` at `n = 2`, `β = 0`. -/
example : (0 : ℕ) < 2 ∧ (0 : ℝ) ≤ 0 ∧ ybetaODE_SA 2 0 Real.tanh :=
  ⟨two_pos, le_rfl, ybetaODE_SA_two_zero⟩

/-- **On all of `t ≥ 0` the estimate is false.**

The survey states `e:ybetacloseto1` "for any `t ≥ 0`", within
`thm: phase.transition.curve`, which fixes `β ≥ 0` and `n ≥ 2`.  At `n = 2`,
`β = 0` the solution of `eq: ybeta` is `tanh`, and at `t = 0` the bound reads
`1 ≤ (1/2) e^{2/3}`, false since `2/3 < log 2`.  The estimate is about the
regime past the half-way time `n e^β / 2`, which is `1` here;
`ybeta_close_to_1` proves it there.

Source: arXiv:2312.10794v5, Appendix D, `e:ybetacloseto1`. -/
theorem not_forall_ybeta_close_to_1 :
    ¬ ∀ (n : ℕ) (β : ℝ) (γ : ℝ → ℝ), 2 ≤ n → 0 ≤ β → ybetaODE_SA n β γ →
        ∀ t : ℝ, 0 ≤ t →
          1 - γ t
            ≤ (1/2 : ℝ) * Real.exp
                ((n : ℝ)^2 * Real.exp β
                    / (2 * ((n : ℝ) + Real.exp (β / 2)))
                  - ((n : ℝ) * t) / ((n : ℝ) + Real.exp (β / 2))) := by
  intro h
  have hineq := h 2 0 Real.tanh le_rfl le_rfl ybetaODE_SA_two_zero 0 le_rfl
  have hval : ((2 : ℕ) : ℝ) ^ 2 * Real.exp 0 / (2 * (((2 : ℕ) : ℝ) + Real.exp (0 / 2)))
      - (((2 : ℕ) : ℝ) * 0) / (((2 : ℕ) : ℝ) + Real.exp (0 / 2)) = 2 / 3 := by
    norm_num
  rw [hval, Real.tanh_zero] at hineq
  have hlt : Real.exp (2 / 3 : ℝ) < 2 := by
    calc Real.exp (2 / 3 : ℝ) < Real.exp (Real.log 2) :=
          Real.exp_lt_exp.mpr (by linarith [Real.log_two_gt_d9])
      _ = 2 := Real.exp_log two_pos
  linarith

end Perspective
end Transformer
