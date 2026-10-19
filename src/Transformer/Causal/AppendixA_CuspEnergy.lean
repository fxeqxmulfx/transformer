module

public import Transformer.Causal.AppendixA_Cusp
public import Transformer.Causal.SequentialFlow
public import Mathlib.Analysis.Calculus.ContDiff.Deriv
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Periodic

/-!
# Actual periodic C¹ energies with a continuous cusp gradient

ArXiv:2411.04990v2, Appendix A, `lemma:convergence`, requires
C¹ periodic energies, bounded positive C¹ normalizations, and
prefix dependence. The construction here supplies those original
regularity and sequential inputs for two actual angular particles.

The primitive is the actual interval integral of the proved
continuous periodic cusp force. The fundamental theorem of calculus
proves its derivative at every real angle. Continuity of that
actual derivative proves C¹ regularity, including the cusp angles.
Oddness gives zero integral over one symmetric period, so the
primitive is genuinely periodic as well.

The actual energies are `cos ψ₀` and
`cos ψ₁ + sin² ψ₀ · primitive(ψ₁)`. Both are C¹ in the full
configuration and have the source's original period in each angle.
With normalization one, their actual negative angular derivatives
are `sin ψ₀` and `sin ψ₁ - sin² ψ₀ · cuspForce(ψ₁)`.
The isolated-critical-point and Jacobian conditions are separate
checks; no conclusion of the printed convergence lemma is used.
-/

@[expose] public section

open Real MeasureTheory
namespace Transformer.Causal

/-- The actual primitive is an interval integral, not an assigned derivative.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, C¹ regularity audit. -/
noncomputable def cuspPrimitive (y : ℝ) : ℝ := ∫ s in (0 : ℝ)..y, cuspForce s

/-- The fundamental theorem gives the actual derivative everywhere.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, regularity audit. -/
theorem hasDerivAt_cuspPrimitive (y : ℝ) : HasDerivAt cuspPrimitive (cuspForce y) y := by
  exact intervalIntegral.integral_hasDerivAt_right (continuous_cuspForce.intervalIntegrable 0 y)
    continuous_cuspForce.stronglyMeasurable.stronglyMeasurableAtFilter continuous_cuspForce.continuousAt

/-- The original cusp primitive really is C¹ at every angular point.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, its printed C¹ assumption. -/
@[fun_prop] theorem contDiff_cuspPrimitive : ContDiff ℝ 1 cuspPrimitive := by
  apply contDiff_one_iff_deriv.mpr
  refine ⟨fun y => (hasDerivAt_cuspPrimitive y).differentiableAt, ?_⟩
  have heq : deriv cuspPrimitive = cuspForce := funext (fun y => (hasDerivAt_cuspPrimitive y).deriv)
  rw [heq]
  exact continuous_cuspForce

/-- The primitive is fixed at zero at the original angular origin.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, energy construction. -/
@[simp] theorem cuspPrimitive_zero : cuspPrimitive 0 = 0 := by simp [cuspPrimitive]

/-- Oddness of the actual force makes its actual primitive even.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, energy construction. -/
theorem cuspPrimitive_neg (y : ℝ) : cuspPrimitive (-y) = cuspPrimitive y := by
  have h := intervalIntegral.integral_comp_neg (f := cuspForce) (a := (0 : ℝ)) (b := y)
  simp only [cuspForce_neg, intervalIntegral.integral_neg, neg_zero] at h
  rw [intervalIntegral.integral_symm (a := (0 : ℝ)) (b := -y)] at h
  exact neg_injective h.symm

/-- The actual force integrates to zero over a symmetric period.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, angular setting. -/
theorem integral_cuspForce_symmetric : (∫ s in -π..π, cuspForce s) = 0 := by
  have h := intervalIntegral.integral_comp_neg (f := cuspForce) (a := -π) (b := π)
  simp only [cuspForce_neg, intervalIntegral.integral_neg, neg_neg] at h
  linarith

/-- The actual primitive has the same circle period as the actual force.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, particles on the circle. -/
theorem cuspPrimitive_periodic : Function.Periodic cuspPrimitive (2 * π) := by
  intro y
  have hw := cuspForce_periodic.intervalIntegral_add_eq y (-π)
  rw [show -π + 2 * π = π by ring, integral_cuspForce_symmetric] at hw
  have hadd := intervalIntegral.integral_add_adjacent_intervals (μ := volume)
    (continuous_cuspForce.intervalIntegrable 0 y)
    (continuous_cuspForce.intervalIntegrable y (y + 2 * π))
  rw [hw, add_zero] at hadd
  exact hadd.symm

/-- The two actual prefix energies, retaining every angular argument used.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, regularity audit. -/
noncomputable def cuspEnergy (k : Idx 2) (ψ : Idx 2 → ℝ) : ℝ :=
  if k = 0 then Real.cos (ψ 0)
  else Real.cos (ψ 1) + Real.sin (ψ 0) ^ 2 * cuspPrimitive (ψ 1)

/-- The first energy depends on the original first angle alone.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, prefix condition. -/
@[simp] theorem cuspEnergy_zero (ψ : Idx 2 → ℝ) : cuspEnergy 0 ψ = Real.cos (ψ 0) := by simp [cuspEnergy]

/-- The second energy couples the two original angles through the actual primitive.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, prefix condition. -/
@[simp] theorem cuspEnergy_one (ψ : Idx 2 → ℝ) :
    cuspEnergy 1 ψ = Real.cos (ψ 1) + Real.sin (ψ 0) ^ 2 * cuspPrimitive (ψ 1) := by simp [cuspEnergy]

/-- Both actual energies satisfy exactly the printed C¹ hypothesis.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`. -/
theorem contDiff_cuspEnergy (k : Idx 2) : ContDiff ℝ 1 (cuspEnergy k) := by
  fin_cases k
  · change ContDiff ℝ 1 (fun ψ : Idx 2 → ℝ => Real.cos (ψ 0))
    fun_prop
  · change ContDiff ℝ 1 (fun ψ : Idx 2 → ℝ =>
      Real.cos (ψ 1) + Real.sin (ψ 0) ^ 2 * cuspPrimitive (ψ 1))
    fun_prop

/-- Both energies are periodic in each original angular coordinate.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, angular domain. -/
theorem periodicAngles_cuspEnergy : PeriodicAngles 2 cuspEnergy := by
  intro k i ψ
  fin_cases k <;> fin_cases i <;> simp [cuspEnergy, Function.update, cuspPrimitive_periodic (ψ 1)]

/-- The actual energy of each particle only sees its original prefix.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, sequential structure. -/
theorem dependsOnPrefix_cuspEnergy : DependsOnPrefix 2 cuspEnergy := by
  intro k ψ ψ' h
  fin_cases k
  · change Real.cos (ψ 0) = Real.cos (ψ' 0)
    rw [h 0 le_rfl]
  · change Real.cos (ψ 1) + Real.sin (ψ 0) ^ 2 * cuspPrimitive (ψ 1) =
      Real.cos (ψ' 1) + Real.sin (ψ' 0) ^ 2 * cuspPrimitive (ψ' 1)
    rw [h 0 (by decide), h 1 le_rfl]

/-- The first actual negative energy derivative is the original sine flow.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, actual ODE. -/
theorem seqVelocity_cuspEnergy_zero (ψ : Idx 2 → ℝ) :
    seqVelocity 2 cuspEnergy (fun _ _ => 1) 0 ψ = Real.sin (ψ 0) := by
  have heq : (fun u : ℝ => cuspEnergy 0 (Function.update ψ 0 u)) = Real.cos := by
    funext u
    simp
  simp only [seqVelocity, heq, (Real.hasDerivAt_cos (ψ 0)).deriv]
  ring

/-- The second actual derivative contains the proved continuous cusp force.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, actual ODE. -/
theorem seqVelocity_cuspEnergy_one (ψ : Idx 2 → ℝ) :
    seqVelocity 2 cuspEnergy (fun _ _ => 1) 1 ψ =
      Real.sin (ψ 1) - Real.sin (ψ 0) ^ 2 * cuspForce (ψ 1) := by
  have heq : (fun u : ℝ => cuspEnergy 1 (Function.update ψ 1 u)) =
      fun u => Real.cos u + Real.sin (ψ 0) ^ 2 * cuspPrimitive u := by
    funext u
    simp
  have hD : HasDerivAt (fun u => Real.cos u + Real.sin (ψ 0) ^ 2 * cuspPrimitive u)
      (-Real.sin (ψ 1) + Real.sin (ψ 0) ^ 2 * cuspForce (ψ 1)) (ψ 1) :=
    (Real.hasDerivAt_cos (ψ 1)).add
      ((hasDerivAt_cuspPrimitive (ψ 1)).const_mul (Real.sin (ψ 0) ^ 2))
  simp only [seqVelocity, heq, hD.deriv]
  ring

/-- The actual two-particle energies jointly satisfy C¹ regularity,
periodicity and prefix dependence. Normalization one is C¹ and lies
strictly between the genuine positive bounds one half and two. -/
example : (0 : ℝ) < 1 / 2 ∧
    (∀ k : Idx 2, ContDiff ℝ 1 (cuspEnergy k) ∧
      ContDiff ℝ 1 (fun _ : Idx 2 → ℝ => (1 : ℝ))) ∧
    (∀ k : Idx 2, ∀ ψ : Idx 2 → ℝ, (1 / 2 : ℝ) <
      (fun _ _ => (1 : ℝ)) k ψ ∧ (fun _ _ => (1 : ℝ)) k ψ < 2) ∧
    PeriodicAngles 2 cuspEnergy ∧ DependsOnPrefix 2 cuspEnergy :=
  ⟨by norm_num, fun k => ⟨contDiff_cuspEnergy k, contDiff_const⟩,
    fun _ _ => by norm_num, periodicAngles_cuspEnergy, dependsOnPrefix_cuspEnergy⟩

end Transformer.Causal
