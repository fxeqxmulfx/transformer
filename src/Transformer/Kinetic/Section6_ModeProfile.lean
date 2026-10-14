/-
# Kinetic theory for Transformers — one mode at the output layer

The proof of `thm:U-shape` of arXiv:2605.09213v1 writes the profile `g_{a_n}(t,1;σ₀)` of
`prop:Bessel` at the output layer as `h_{c_n}(Y(σ₀))`, with `c_n = a_n t` and

  `h_c(y) = (λ/(e^λ - 1) + λe^{-y}) ψ_c(y)`,  `ψ_c(y) = √(c/y) I₁(2√(cy)) = Σ_k c^{k+1}y^k/(k!(k+1)!)`.

Here are `ψ_c` as an entire series (`psi`), its Bessel form (`sqrt_mul_besselI`), the identity of
the profile (`hardyProfile_one_eq`), and the first two derivatives of `h_c` (`modeDeriv`,
`modeDeriv2`), the second being `λ/(e^λ-1) ψ_c'' + λe^{-y}(ψ_c'' - 2ψ_c' + ψ_c)` as in the source.
-/

import Transformer.Kinetic.Hardy
import Transformer.Kinetic.Section6_PowerSeries

open Real

namespace Transformer
namespace Kinetic

/-- The coefficients `c^{k+1}/(k!(k+1)!)` of `ψ_c`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def psiCoeffs (c : ℝ) (k : ℕ) : ℝ :=
  c ^ (k + 1) / (k.factorial * (k + 1).factorial)

/-- `ψ_c(y) = Σ_k c^{k+1} y^k / (k!(k+1)!)`, the proof of `thm:U-shape`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def psi (c y : ℝ) : ℝ := powerSum (psiCoeffs c) y

/-- `ψ_c'`, termwise.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def psiDeriv (c y : ℝ) : ℝ := powerSum (derivCoeffs (psiCoeffs c)) y

/-- `ψ_c''`, termwise.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def psiDeriv2 (c y : ℝ) : ℝ := powerSum (derivCoeffs (derivCoeffs (psiCoeffs c))) y

/-- `h_c(y) = (λ/(e^λ - 1) + λe^{-y}) ψ_c(y)`, one mode of the proof of `thm:U-shape`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def modeProfile (lam c y : ℝ) : ℝ :=
  (lam / (Real.exp lam - 1) + lam * Real.exp (-y)) * psi c y

/-- `h_c' = -λe^{-y} ψ_c + (λ/(e^λ - 1) + λe^{-y}) ψ_c'`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def modeDeriv (lam c y : ℝ) : ℝ :=
  -(lam * Real.exp (-y)) * psi c y + (lam / (Real.exp lam - 1) + lam * Real.exp (-y)) * psiDeriv c y

/-- `h_c'' = λ/(e^λ - 1) ψ_c'' + λe^{-y}(ψ_c'' - 2ψ_c' + ψ_c)`, the source's "direct
differentiation".
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
noncomputable def modeDeriv2 (lam c y : ℝ) : ℝ :=
  lam / (Real.exp lam - 1) * psiDeriv2 c y +
    lam * Real.exp (-y) * (psiDeriv2 c y - 2 * psiDeriv c y + psi c y)

/-- The original Bessel-profile coefficients are entire at every real
mode parameter. Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem isEntire_psiCoeffs (c : ℝ) : IsEntire (psiCoeffs c) := by
  intro R hR
  refine Summable.of_nonneg_of_le (fun _ => by positivity) (fun k => ?_)
    ((Real.summable_pow_div_factorial (|c| * R)).mul_left |c|)
  have h1 : (1 : ℝ) ≤ (k + 1).factorial := by exact_mod_cast Nat.factorial_pos (k + 1)
  rw [psiCoeffs, abs_div, abs_pow,
    abs_of_pos (by positivity : (0 : ℝ) < k.factorial * (k + 1).factorial)]
  calc |c| ^ (k + 1) / (k.factorial * (k + 1).factorial) * R ^ k
      ≤ |c| ^ (k + 1) / (k.factorial * 1) * R ^ k := by gcongr
    _ = |c| * ((|c| * R) ^ k / k.factorial) := by rw [mul_one, mul_pow, pow_succ]; ring

/-- The Bessel form: `√(c/y) I₁(2√(cy)) = ψ_c(y)` for `c ≥ 0` and `y > 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem sqrt_mul_besselI {c y : ℝ} (hc : 0 ≤ c) (hy : 0 < y) :
    Real.sqrt (c / y) * besselI 1 (2 * Real.sqrt (c * y)) = psi c y := by
  have hs : Real.sqrt (c / y) * Real.sqrt (c * y) = c := by
    rw [← Real.sqrt_mul (div_nonneg hc hy.le), show c / y * (c * y) = c * c by field_simp]
    exact Real.sqrt_mul_self hc
  rw [besselI, psi, powerSum, ← tsum_mul_left]
  congr 1
  ext k
  rw [show 2 * Real.sqrt (c * y) / 2 = Real.sqrt (c * y) by ring, pow_succ, pow_mul,
    Real.sq_sqrt (mul_nonneg hc hy.le), psiCoeffs]
  calc Real.sqrt (c / y) * ((c * y) ^ k * Real.sqrt (c * y) /
        ((k.factorial : ℝ) * ((k + 1).factorial : ℝ)))
      = (Real.sqrt (c / y) * Real.sqrt (c * y)) * (c * y) ^ k /
        ((k.factorial : ℝ) * ((k + 1).factorial : ℝ)) := by ring
    _ = _ := by rw [hs, mul_pow, pow_succ]; ring

example := sqrt_mul_besselI le_rfl one_pos

/-- `e^{-Y(σ₀)} = (e^{λσ₀} - 1)/(e^λ - 1)` at the output layer, for `0 < σ₀`, `λ > 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem exp_neg_Yfun_one {lam σ₀ : ℝ} (hlam : 0 < lam) (hσ₀ : 0 < σ₀) :
    Real.exp (-Yfun lam 1 σ₀) = (Real.exp (lam * σ₀) - 1) / (Real.exp lam - 1) := by
  have h₁ : 0 < Real.exp lam - 1 := by linarith [Real.add_one_lt_exp hlam.ne']
  have h₂ : 0 < Real.exp (lam * σ₀) - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hlam hσ₀))
  rw [Yfun, ite_eq_right hlam.ne', mul_one, Real.exp_neg, Real.exp_log (div_pos h₁ h₂), inv_div]

example := exp_neg_Yfun_one one_pos one_pos

/-- `Y(σ₀) > 0` at the output layer, for `0 < σ₀ < 1`, `λ > 0`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem Yfun_one_pos {lam σ₀ : ℝ} (hlam : 0 < lam) (hσ₀ : 0 < σ₀) (hσ₁ : σ₀ < 1) :
    0 < Yfun lam 1 σ₀ := by
  have h₂ : 0 < Real.exp (lam * σ₀) - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hlam hσ₀))
  have h₃ : Real.exp (lam * σ₀) < Real.exp lam :=
    Real.exp_lt_exp.2 (by nlinarith)
  rw [Yfun, ite_eq_right hlam.ne', mul_one]
  exact Real.log_pos ((one_lt_div h₂).2 (by linarith))

example := Yfun_one_pos one_pos one_half_pos one_half_lt_one

/-- **The profile at the output layer**: for `0 < σ₀ < 1`, `λ > 0` and `at ≥ 0`, the profile
`g_a(t,1;σ₀)` of `eq:Bessel-full` is `h_{at}(Y(σ₀))`, the display before the proof of
`thm:U-shape`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem hardyProfile_one_eq {lam a t σ₀ : ℝ} (hlam : 0 < lam) (hσ₀ : 0 < σ₀) (hσ₁ : σ₀ < 1)
    (hat : 0 ≤ a * t) : hardyProfile lam a t 1 σ₀ = modeProfile lam (a * t) (Yfun lam 1 σ₀) := by
  have h₁ : 0 < Real.exp lam - 1 := by linarith [Real.add_one_lt_exp hlam.ne']
  have hY := Yfun_one_pos hlam hσ₀ hσ₁
  rw [hardyProfile, mul_assoc, sqrt_mul_besselI hat hY,
    modeProfile, exp_neg_Yfun_one hlam hσ₀]
  congr 1
  rw [graphon, ite_eq_left hσ₁, ite_eq_right hlam.ne', mul_one, show -(lam * (1 - σ₀)) = lam * σ₀ - lam by
    ring, Real.exp_sub, Real.exp_neg]
  have h₀ : Real.exp lam ≠ 0 := (Real.exp_pos lam).ne'
  have h₄ : 1 - (Real.exp lam)⁻¹ ≠ 0 := by
    rw [sub_ne_zero, ne_comm, inv_ne_one]
    exact (Real.one_lt_exp_iff.2 hlam).ne'
  field_simp
  ring

example := hardyProfile_one_eq (a := 1) (t := 1) one_pos one_half_pos one_half_lt_one (by norm_num)

/-- The weight `λ/(e^λ - 1) + λe^{-y}` has derivative `-λe^{-y}`.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem hasDerivAt_modeWeight (lam y : ℝ) :
    HasDerivAt (fun y => lam / (Real.exp lam - 1) + lam * Real.exp (-y))
      (-(lam * Real.exp (-y))) y := by
  have h := ((hasDerivAt_id' (x := y)).neg.exp.const_mul lam).const_add (lam / (Real.exp lam - 1))
  convert h using 1
  dsimp
  ring

/-- The actual first derivative of the original output-mode profile.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem hasDerivAt_modeProfile (lam c y : ℝ) :
    HasDerivAt (modeProfile lam c) (modeDeriv lam c y) y :=
  (hasDerivAt_modeWeight lam y).mul (hasDerivAt_powerSum (isEntire_psiCoeffs c) y)

/-- The actual second derivative of the original output-mode profile.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem hasDerivAt_modeDeriv (lam c y : ℝ) :
    HasDerivAt (modeDeriv lam c) (modeDeriv2 lam c y) y := by
  have hψ := hasDerivAt_powerSum (isEntire_psiCoeffs c) y
  have hψ' := hasDerivAt_powerSum (isEntire_psiCoeffs c).derivCoeffs y
  have hn : HasDerivAt (fun y => -(lam * Real.exp (-y))) (lam * Real.exp (-y)) y := by
    convert ((hasDerivAt_id' (x := y)).neg.exp.const_mul lam).neg using 1
    dsimp
    ring
  convert (hn.mul hψ).add ((hasDerivAt_modeWeight lam y).mul hψ') using 1
  · funext z
    rfl
  · unfold modeDeriv2 psiDeriv2 psiDeriv psi
    ring

/-- The actual derivative function is the original differentiated series.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem deriv_modeProfile (lam c : ℝ) : deriv (modeProfile lam c) = modeDeriv lam c :=
  funext fun y => (hasDerivAt_modeProfile lam c y).deriv

/-- The original second derivative equals the source's displayed formula.
Source: arXiv:2605.09213v1, §6, proof of `thm:U-shape`. -/
theorem deriv2_modeProfile (lam c y : ℝ) : deriv^[2] (modeProfile lam c) y = modeDeriv2 lam c y := by
  rw [Function.iterate_succ_apply, Function.iterate_one, deriv_modeProfile]
  exact (hasDerivAt_modeDeriv lam c y).deriv

end Kinetic
end Transformer
