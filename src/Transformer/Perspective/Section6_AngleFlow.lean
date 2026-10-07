/-
# §6.2 — Global existence of the scalar angle equation

Auxiliary to arXiv:2312.10794v5, *A mathematical perspective on Transformers*,
`thm: orthogonal` and `eq: ybeta`.

The scalar drift is smooth and vanishes at `-1/(n-1)` and `1`. The interval
between those equilibria contains zero. `GlobalFlow.exists_global_Icc` gives
a genuine solution for every real time, with no assumed ODE solution.
-/

import Transformer.GlobalFlow.Interval
import Mathlib.Analysis.Calculus.ContDiff.RCLike
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Real Set

namespace Transformer
namespace Perspective

/-- The scalar ODE driving the angle between pairwise orthogonal particles
under `SA`:

  `γ̇_β(t) = 2 e^{β γ_β(t)} (1 - γ_β(t)) ((n-1) γ_β(t) + 1)
             / (e^β + (n-1) e^{β γ_β(t)})`,
  `γ_β(0) = 0`.

Source: arXiv:2312.10794v5, §6.2, **Equation (eq: ybeta)**. -/
def ybetaODE_SA (n : ℕ) (β : ℝ) (γ : ℝ → ℝ) : Prop :=
  γ 0 = 0 ∧
  ∀ t : ℝ, HasDerivAt γ
    (2 * Real.exp (β * γ t) * (1 - γ t) * ((n - 1 : ℝ) * γ t + 1)
      / (Real.exp β + (n - 1 : ℝ) * Real.exp (β * γ t))) t

/-- The scalar ODE for `USA` (eq: ybetaUSA):

  `γ̇_β(t) = (2/n) e^{β γ_β(t)} (1 - γ_β(t)) ((n-1) γ_β(t) + 1)`.

Source: arXiv:2312.10794v5, §6.2, `eq: ybetaUSA`. -/
def ybetaODE_USA (n : ℕ) (β : ℝ) (γ : ℝ → ℝ) : Prop :=
  γ 0 = 0 ∧
  ∀ t : ℝ, HasDerivAt γ
    ((2 / (n : ℝ)) * Real.exp (β * γ t) * (1 - γ t) * ((n - 1 : ℝ) * γ t + 1)) t

/-- **A solution of `eq: ybeta`.**  At `n = 1` and `β = 0` the equation is
`γ̇ = 2(1 - γ)`, `γ(0) = 0`, whose solution is `γ(t) = 1 - e^{-2t}`: the scalar
approaches one at an exponential rate.

This elementary solution witnesses `ybetaODE_SA` at one particle. The
existence theorem below also covers every `n ≥ 2` and every real `β`.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`. -/
theorem ybetaODE_SA_one_zero :
    ybetaODE_SA 1 0 (fun t => 1 - Real.exp (-2 * t)) := by
  refine ⟨by simp, fun t => ?_⟩
  have hlin : HasDerivAt (fun s : ℝ => -2 * s) (-2 : ℝ) t := by
    simpa using HasDerivAt.const_mul (-2 : ℝ) (hasDerivAt_id t)
  have h : HasDerivAt (fun s : ℝ => 1 - Real.exp (-2 * s))
      (-(Real.exp (-2 * t) * -2)) t := hlin.exp.const_sub 1
  refine h.congr_deriv ?_
  norm_num
  ring

/-- The angle drift from arXiv:2312.10794v5, §6.2, `eq: ybeta`,
written as an autonomous scalar field. -/
noncomputable def saAngleDrift (n : ℕ) (β x : ℝ) : ℝ :=
  2 * Real.exp (β * x) * (1 - x) * ((n - 1 : ℝ) * x + 1)
    / (Real.exp β + (n - 1 : ℝ) * Real.exp (β * x))

/-- The scalar partition function is positive when `n ≥ 2`.

Source: arXiv:2312.10794v5, §6.2, denominator of `eq: ybeta`. -/
theorem saAngleDen_pos (n : ℕ) (β x : ℝ) (hn : 2 ≤ n) :
    0 < Real.exp β + (n - 1 : ℝ) * Real.exp (β * x) := by
  have hnreal : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hn' : (0 : ℝ) < n - 1 := by linarith
  positivity

/-- The angle field is continuously differentiable on the real line.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`; positivity of its
denominator justifies the scalar uniqueness argument. -/
theorem contDiff_saAngleDrift (n : ℕ) (β : ℝ) (hn : 2 ≤ n) :
    ContDiff ℝ 1 (saAngleDrift n β) := by
  have he : ContDiff ℝ 1 (fun x : ℝ => Real.exp (β * x)) :=
    (contDiff_const.mul contDiff_id).exp
  exact (((contDiff_const.mul he).mul (contDiff_const.sub contDiff_id)).mul
    ((contDiff_const.mul contDiff_id).add contDiff_const)).div
      (contDiff_const.add (contDiff_const.mul he)) (fun x => (saAngleDen_pos n β x hn).ne')

/-- The lower endpoint `-1/(n-1)` is an equilibrium of the angle equation.

Source: arXiv:2312.10794v5, §6.2, the factor `(n-1)γ + 1` in `eq: ybeta`. -/
theorem saAngleDrift_lower (n : ℕ) (β : ℝ) (hn : 2 ≤ n) :
    saAngleDrift n β (-(n - 1 : ℝ)⁻¹) = 0 := by
  have hn' : (n - 1 : ℝ) ≠ 0 := by
    have hnreal : (2 : ℝ) ≤ n := by exact_mod_cast hn
    linarith
  simp [saAngleDrift, hn']

/-- Consensus, `γ = 1`, is an equilibrium of the angle equation.

Source: arXiv:2312.10794v5, §6.2, the factor `1 - γ` in `eq: ybeta`. -/
theorem saAngleDrift_one (n : ℕ) (β : ℝ) : saAngleDrift n β 1 = 0 := by
  simp [saAngleDrift]

/-- The angle equation has a solution on all of `ℝ`, remaining in
`[-1/(n-1), 1]`. Both endpoints are equilibria, so the compact-interval
existence lemma applies to this field itself.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`. The source considers
`β ≥ 0` and nonnegative times; existence here holds for every real `β`
and in both time directions. -/
theorem exists_saAngle (n : ℕ) (β : ℝ) (hn : 2 ≤ n) :
    ∃ γ : ℝ → ℝ, γ 0 = 0 ∧ ∀ t, HasDerivAt γ (saAngleDrift n β (γ t)) t ∧
      γ t ∈ Icc (-(n - 1 : ℝ)⁻¹) 1 := by
  have hnreal : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hn' : (0 : ℝ) < n - 1 := by linarith
  exact Transformer.GlobalFlow.exists_global_Icc
    (contDiff_saAngleDrift n β hn).locallyLipschitz
    (saAngleDrift_lower n β hn) (saAngleDrift_one n β)
    ⟨neg_nonpos.mpr (inv_nonneg.mpr hn'.le), zero_le_one⟩

/-- The particle-count hypotheses of the denominator, smoothness, lower
endpoint and existence lemmas hold at two particles. -/
example : 2 ≤ 2 := le_rfl

/-- The scalar drift is nonnegative throughout its invariant interval.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`. The two linear factors
are nonnegative between their zeros, and the partition function is positive. -/
theorem saAngleDrift_nonneg (n : ℕ) (β x : ℝ) (hn : 2 ≤ n)
    (hx : x ∈ Icc (-(n - 1 : ℝ)⁻¹) 1) : 0 ≤ saAngleDrift n β x := by
  have hnreal : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hn' : (0 : ℝ) < n - 1 := by linarith
  have hlo : (n - 1 : ℝ) * (-(n - 1 : ℝ)⁻¹) = -1 := by simp [hn'.ne']
  have hmul := mul_le_mul_of_nonneg_left hx.1 hn'.le
  rw [hlo] at hmul
  have hfac : 0 ≤ (n - 1 : ℝ) * x + 1 := by linarith
  have hqx : 0 ≤ 1 - x := sub_nonneg.mpr hx.2
  unfold saAngleDrift
  exact div_nonneg (by positivity) (saAngleDen_pos n β x hn).le

/-- Zero lies in the two-particle invariant interval, witnessing both
hypotheses of the nonnegative-drift lemma. -/
example : 2 ≤ 2 ∧ (0 : ℝ) ∈ Icc (-(2 - 1 : ℝ)⁻¹) 1 := by norm_num

/-- Every angle solution remaining in the invariant interval is monotone.

Source: arXiv:2312.10794v5, §6.2, `eq: ybeta`; for nonnegative times
this also gives `0 ≤ γ(t) ≤ 1` from the initial condition `γ(0) = 0`. -/
theorem saAngle_monotone (n : ℕ) (β : ℝ) (hn : 2 ≤ n) (γ : ℝ → ℝ)
    (hγ : ∀ t, HasDerivAt γ (saAngleDrift n β (γ t)) t)
    (hb : ∀ t, γ t ∈ Icc (-(n - 1 : ℝ)⁻¹) 1) : Monotone γ := by
  exact monotone_of_hasDerivAt_nonneg hγ fun t => saAngleDrift_nonneg n β (γ t) hn (hb t)

/-- The existence and monotonicity hypotheses hold for an actual solution
at two particles and positive temperature. -/
example : ∃ γ : ℝ → ℝ, 2 ≤ 2 ∧ ybetaODE_SA 2 1 γ ∧
    (∀ t, γ t ∈ Icc (-(2 - 1 : ℝ)⁻¹) 1) ∧ Monotone γ := by
  obtain ⟨γ, h0, hγ⟩ := exists_saAngle 2 1 (by norm_num)
  exact ⟨γ, le_rfl, ⟨h0, fun t => (hγ t).1⟩, fun t => (hγ t).2,
    saAngle_monotone 2 1 le_rfl γ (fun t => (hγ t).1) (fun t => (hγ t).2)⟩

end Perspective
end Transformer
