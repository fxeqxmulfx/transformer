import Transformer.Grokking.EffectiveTheory.Section3_Conservation
import Transformer.Grokking.EffectiveTheory.Section3_ScalarMode

/-!
# Recovering spectral decay through relative modes

Source: Liu et al., arXiv:2205.10343v2, section 3.2, "Time towards the
linear structure". Correction to the raw linear-flow deduction: the
normalized loss has a radial term, as proved in `Section3_Conservation`.
For the single-parallelogram specialization, that term cancels in the
ratio `(E₀ + E₂ - 2 E₁) / (E₀ - E₂)`.

The actual quotient-gradient equations imply a scalar linear ODE for this
ratio, with coefficient `12 / Z₀`. Conserved `Z₀` then gives an exact
exponential law. These are derived consequences of an explicitly stated
classical flow on the nonzero domain. The ground component `E₀ - E₂`
must remain nonzero; proving this from initial data alone is separate work.
The stationary nonzero-loss counterexample has zero ground component.
No transformer/AdamW trajectory is asserted to meet these effective premises.
-/

namespace Transformer.Grokking.EffectiveTheory

/-- Constraint mode divided by a ground component. Source correction:
arXiv:2205.10343v2, section 3.2; statements require a nonzero denominator. -/
noncomputable def relativeMode (x y z : ℝ) : ℝ := residual x y z / (x - z)

/-- The coefficient derived below is positive off the zero representation.
Source specialization: arXiv:2205.10343v2, section 3.2, positive mode rate. -/
theorem relative_rate_positive (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0) :
    0 < 12 / squaredNorm x y z := by
  have hn : 0 ≤ squaredNorm x y z := by unfold squaredNorm; positivity
  have hp := lt_of_le_of_ne hn (Ne.symm hZ)
  exact div_pos (by norm_num) hp

example : squaredNorm 1 0 0 ≠ 0 := by unfold squaredNorm; norm_num

/-- A satisfied parallelogram is stationary for the actual normalized
loss. Source: arXiv:2205.10343v2, section 3.2, loss-zero ground states. -/
theorem ground_gradient_zero (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0)
    (hr : residual x y z = 0) : quotientGradient x y z = (0, 0, 0) := by
  rw [quotientGradient_eq x y z hZ]
  unfold numerator
  rw [hr]
  norm_num

example : squaredNorm 1 0 (-1) ≠ 0 ∧ residual 1 0 (-1) = 0 := by
  unfold squaredNorm residual
  norm_num

/-- Corrected local spectral law derived from the actual quotient flow:
the radial correction cancels in a relative mode. Source:
arXiv:2205.10343v2, section 3.2; this replaces its raw linear-flow step
for the explicit one-constraint specialization. -/
theorem relativeMode_deriv_of_gradient_flow (x y z : ℝ → ℝ) (t : ℝ)
    (hZ : squaredNorm (x t) (y t) (z t) ≠ 0) (hA : x t - z t ≠ 0)
    (hx : HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t)
    (hy : HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t)
    (hz : HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) :
    HasDerivAt (fun s => relativeMode (x s) (y s) (z s))
      (-(12 / squaredNorm (x t) (y t) (z t)) * relativeMode (x t) (y t) (z t)) t := by
  have hn := (hx.add hz).sub (hy.const_mul 2)
  convert hn.div (hx.sub hz) hA using 1
  · rfl
  · rw [quotientGradient_eq (x t) (y t) (z t) hZ]
    dsimp
    unfold relativeMode
    field_simp
    unfold numerator residual squaredNorm
    ring

example : squaredNorm 1 0 (-1) ≠ 0 ∧ (1 : ℝ) - (-1) ≠ 0 ∧
    HasDerivAt (fun _ : ℝ => 1) (-(quotientGradient 1 0 (-1)).1) 0 ∧
    HasDerivAt (fun _ : ℝ => 0) (-(quotientGradient 1 0 (-1)).2.1) 0 ∧
    HasDerivAt (fun _ : ℝ => -1) (-(quotientGradient 1 0 (-1)).2.2) 0 := by
  have hg : quotientGradient 1 0 (-1) = (0, 0, 0) :=
    ground_gradient_zero 1 0 (-1) (by unfold squaredNorm; norm_num)
      (by unfold residual; norm_num)
  rw [hg]
  refine ⟨by unfold squaredNorm; norm_num, by norm_num, ?_, ?_, ?_⟩ <;>
    simpa using hasDerivAt_const (0 : ℝ) _

/-- Exact exponential evolution of the relative mode, using proved norm
conservation and scalar-ODE uniqueness. Source correction:
arXiv:2205.10343v2, section 3.2. Nonzero representation and nonvanishing
ground component are explicit domain premises, not a hidden convergence
assumption; classical-flow existence is not asserted here. -/
theorem relativeMode_exponential_of_gradient_flow (x y z : ℝ → ℝ)
    (hZ : ∀ t, squaredNorm (x t) (y t) (z t) ≠ 0)
    (hA : ∀ t, x t - z t ≠ 0)
    (hflow : ∀ t,
      HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) (t : ℝ) :
    relativeMode (x t) (y t) (z t) = relativeMode (x 0) (y 0) (z 0) *
      Real.exp (-(12 / squaredNorm (x 0) (y 0) (z 0)) * t) := by
  apply scalar_ode_solution (fun s => relativeMode (x s) (y s) (z s))
    (12 / squaredNorm (x 0) (y 0) (z 0))
  intro s
  have h := relativeMode_deriv_of_gradient_flow x y z s (hZ s) (hA s)
    (hflow s).1 (hflow s).2.1 (hflow s).2.2
  rw [norm_const_of_gradient_flow x y z hZ hflow 0 s] at h
  exact h

example : ∃ x y z : ℝ → ℝ,
    (∀ t, squaredNorm (x t) (y t) (z t) ≠ 0) ∧ (∀ t, x t - z t ≠ 0) ∧
    (∀ t, HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) := by
  refine ⟨fun _ => 1, fun _ => 0, fun _ => -1, ?_, ?_, ?_⟩
  · intro t
    unfold squaredNorm
    norm_num
  · intro t
    norm_num
  · intro t
    have hg : quotientGradient 1 0 (-1) = (0, 0, 0) :=
      ground_gradient_zero 1 0 (-1) (by unfold squaredNorm; norm_num)
        (by unfold residual; norm_num)
    refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

/-- The two gradients differ also at a centered nonstationary point with
a nonzero ground component. Source: arXiv:2205.10343v2, section 3.2,
counterexample to its raw linear-flow deduction beyond the stationary case. -/
theorem nonstationary_centered_example :
    (1 : ℝ) + (-1) + 0 = 0 ∧ squaredNorm 1 (-1) 0 = 2 ∧
      (1 : ℝ) - 0 ≠ 0 ∧ quotientGradient 1 (-1) 0 = (-3 / 2, -3 / 2, 3) ∧
      frozenGradient 1 (-1) 0 = (3, -6, 3) ∧
      quotientGradient 1 (-1) 0 ≠ frozenGradient 1 (-1) 0 := by
  have hZ : squaredNorm 1 (-1) 0 ≠ 0 := by unfold squaredNorm; norm_num
  rw [quotientGradient_eq 1 (-1) 0 hZ, frozenGradient_eq]
  unfold numerator residual squaredNorm
  norm_num

/-- Relative mode energy follows the doubled rate under the actual
quotient flow. Source correction: arXiv:2205.10343v2, section 3.2;
this is a mode-energy law, not an identification with held-out accuracy. -/
theorem relativeMode_energy_deriv (x y z : ℝ → ℝ) (t : ℝ)
    (hZ : squaredNorm (x t) (y t) (z t) ≠ 0) (hA : x t - z t ≠ 0)
    (hx : HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t)
    (hy : HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t)
    (hz : HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) :
    HasDerivAt (fun s => relativeMode (x s) (y s) (z s) ^ 2)
      (-(24 / squaredNorm (x t) (y t) (z t)) * relativeMode (x t) (y t) (z t) ^ 2) t := by
  convert (relativeMode_deriv_of_gradient_flow x y z t hZ hA hx hy hz).pow 2 using 1
  norm_num
  ring

example : squaredNorm 1 (-1) 0 ≠ 0 ∧ (1 : ℝ) - 0 ≠ 0 ∧
    HasDerivAt (fun s : ℝ => 1 + (3 / 2) * s) (-(quotientGradient 1 (-1) 0).1) 0 ∧
    HasDerivAt (fun s : ℝ => -1 + (3 / 2) * s) (-(quotientGradient 1 (-1) 0).2.1) 0 ∧
    HasDerivAt (fun s : ℝ => -3 * s) (-(quotientGradient 1 (-1) 0).2.2) 0 := by
  have hg := nonstationary_centered_example.2.2.2.1
  rw [hg]
  refine ⟨by unfold squaredNorm; norm_num, by norm_num, ?_, ?_, ?_⟩
  · convert ((hasDerivAt_id (0 : ℝ)).const_mul (3 / 2 : ℝ)).const_add 1 using 1 <;> norm_num
  · convert ((hasDerivAt_id (0 : ℝ)).const_mul (3 / 2 : ℝ)).const_add (-1) using 1 <;> norm_num
  · simpa using (hasDerivAt_id (0 : ℝ)).const_mul (-3 : ℝ)

end Transformer.Grokking.EffectiveTheory
