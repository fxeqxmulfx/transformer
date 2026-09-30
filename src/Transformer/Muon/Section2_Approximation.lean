/-
# Muon — limits of the printed approximation

arXiv:2502.16982, §2.1. The source seeks a fixed point “near 1”, not an
exact polar iteration. The printed coefficients must not be treated as a
convergent iteration to exactly unit singular values.
-/

import Transformer.Muon.Section2_NewtonSchulz
import Transformer.Muon.AppendixA_RMS
import Mathlib.Topology.Algebra.Order.Field

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

/-- The printed coefficients send `1` to `0.701`, arXiv:2502.16982, §2.1,
after `eq:iteration`. Thus `1` is not a fixed point. -/
theorem printed_polynomial_one :
    schulzPolynomial (34445 / 10000) (-47750 / 10000) (20315 / 10000) 1 = 701 / 1000 := by
  norm_num [schulzPolynomial]

/-- Every limit of a Newton–Schulz scalar recurrence is a fixed point of its
polynomial, arXiv:2502.16982, §2.1, the convergence discussion. -/
theorem schulz_limit_fixed (α β γ L : ℝ) (x : ℕ → ℝ)
    (hstep : ∀ k, x (k + 1) = schulzPolynomial α β γ (x k))
    (hx : Filter.Tendsto x Filter.atTop (nhds L)) :
    schulzPolynomial α β γ L = L := by
  have hpoly : Continuous (schulzPolynomial α β γ) := by
    unfold schulzPolynomial
    fun_prop
  have hf := (hpoly.tendsto L).comp hx
  have hs : Filter.Tendsto (fun k => x (k + 1)) Filter.atTop (nhds L) :=
    (Filter.tendsto_add_atTop_iff_nat 1).mpr hx
  have hf' : Filter.Tendsto (fun k => x (k + 1)) Filter.atTop
      (nhds (schulzPolynomial α β γ L)) := by
    simpa only [hstep, Function.comp_def] using hf
  exact tendsto_nhds_unique hf' hs

/-- A convergent polynomial recurrence exists: the zero trajectory,
arXiv:2502.16982, §2.1. -/
example : (∀ k : ℕ, (fun _ : ℕ => (0 : ℝ)) (k + 1) =
    schulzPolynomial (34445 / 10000) (-47750 / 10000)
      (20315 / 10000) ((fun _ : ℕ => (0 : ℝ)) k)) ∧
    Filter.Tendsto (fun _ : ℕ => (0 : ℝ)) Filter.atTop (nhds 0) := by
  exact ⟨fun _ => by norm_num [schulzPolynomial], tendsto_const_nhds⟩

/-- Refutation of the exact-convergence interpretation: with the printed
coefficients no initial scalar yields convergence to `1`.
The source only says “near 1”; this theorem explains why exact polar-factor
identities cannot be applied to these iterates.
Source: arXiv:2502.16982, §2.1, `eq:iteration`. -/
theorem printed_iteration_not_tendsto_one (x₀ : ℝ) :
    ¬ Filter.Tendsto
      (scalarSchulz (34445 / 10000) (-47750 / 10000) (20315 / 10000) x₀)
      Filter.atTop (nhds 1) := by
  intro h
  have hf := schulz_limit_fixed (34445 / 10000) (-47750 / 10000) (20315 / 10000) 1
    _ (fun _ => rfl) h
  rw [printed_polynomial_one] at hf
  norm_num at hf

/-- One Newton–Schulz step already changes a unit singular value to `0.701`.
Source: arXiv:2502.16982, §2.1, `eq:iteration`. -/
theorem printed_step_identity :
    schulzStep (34445 / 10000) (-47750 / 10000) (20315 / 10000)
      (1 : Matrix (Fin 1) (Fin 1) ℝ) = (701 / 1000 : ℝ) • 1 := by
  ext i j
  fin_cases i
  fin_cases j
  norm_num [schulzStep, Matrix.mul_apply, pow_two]

/-- The finite polynomial approximation need not have the exact RMS of
Lemma 1, even for a full-rank one-dimensional input.
Source: arXiv:2502.16982, §2.1–2.2, `eq:iteration` and `lemma:updaterms`. -/
theorem printed_step_rms_not_exact :
    matrixRMS (schulzStep (34445 / 10000) (-47750 / 10000) (20315 / 10000)
      (1 : Matrix (Fin 1) (Fin 1) ℝ)) = 701 / 1000 ∧ (701 / 1000 : ℝ) ≠ 1 := by
  rw [printed_step_identity, matrixRMS_smul]
  norm_num [matrixRMS, squaredFrobenius]

/-- The actual five-step setting from §2.2 has RMS between `0.69` and
`0.70` on the one-dimensional identity input. All five printed polynomial
steps are evaluated with exact rational arithmetic.
Source: arXiv:2502.16982, §2.1, `eq:iteration`; §2.2, “Other Hyper-parameters”. -/
theorem five_step_identity_rms_bounds :
    (69 / 100 : ℝ) < matrixRMS (approximatePolar (1 : Matrix (Fin 1) (Fin 1) ℝ)) ∧
      matrixRMS (approximatePolar (1 : Matrix (Fin 1) (Fin 1) ℝ)) < 7 / 10 := by
  have hU : OrthonormalColumns (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
    simp [OrthonormalColumns]
  have h := schulzIterate_spectrum (34445 / 10000) (-47750 / 10000) (20315 / 10000)
    (1 : Matrix (Fin 1) (Fin 1) ℝ) (fun _ => 1) 1 hU hU 5
  norm_num [singularMatrix, squaredFrobenius] at h
  unfold approximatePolar
  norm_num only
  rw [h]
  norm_num [matrixRMS, squaredFrobenius, scalarSchulz, schulzPolynomial]

/-- Full-rank input does not make the printed five-step update's RMS equal
to the exact-polar RMS in Lemma 1. The source calls the iteration approximate;
this refutes interpreting that approximation as the exact lemma.
Source: arXiv:2502.16982, §2.1–2.2, `eq:iteration` and `lemma:updaterms`. -/
theorem five_step_rms_not_exact :
    (1 : Matrix (Fin 1) (Fin 1) ℝ).det ≠ 0 ∧
      matrixRMS (approximatePolar (1 : Matrix (Fin 1) (Fin 1) ℝ)) ≠ 1 := by
  refine ⟨by norm_num, ?_⟩
  have h := five_step_identity_rms_bounds.2
  linarith

end Transformer.Muon
