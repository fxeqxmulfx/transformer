import Transformer.Grokking.EffectiveTheory.Section3_QuotientGradient
import Mathlib.Analysis.Calculus.MeanValue

/-!
# Conserved norm, conditional centering and a failed linear-flow step

Source: Liu et al., arXiv:2205.10343v2, section 3.2, paragraph "Time
towards the linear structure", and appendix "Conservation laws of the
effective theory", equations `eq:Ei_dynamics_app` and `eq:Z0_conservation`.

For the actual normalized parallelogram loss, the radial gradient vanishes.
The sum of gradient components vanishes on centered embeddings; arbitrary
centroid conservation needs this missing hypothesis. A noncentered example
refutes the unqualified appendix claim, while retaining the norm law.

The centered point `(1, -2, 1)` is stationary with nonzero loss. Dividing
the numerator gradient by the norm gives a different vector there. Thus
dropping denominator derivatives does not yield the normalized-loss flow.
The source's `dR/dt = -H R` deduction needs revision before its eigenvalue
formula is transferred to the normalized flow or to an actual transformer.
-/

namespace Transformer.Grokking.EffectiveTheory

/-- Gradient obtained by treating the denominator as constant. Source:
arXiv:2205.10343v2, section 3.2, the linear-flow deduction; this is kept
distinct from the actual quotient gradient. -/
noncomputable def frozenGradient (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (deriv (fun u => numerator u y z) x / squaredNorm x y z,
    deriv (fun u => numerator x u z) y / squaredNorm x y z,
    deriv (fun u => numerator x y u) z / squaredNorm x y z)

/-- The purported linear field uses the numerator's actual gradient.
Source: arXiv:2205.10343v2, section 3.2, `H` as numerator Hessian over
`Z₀`; the following counterexample distinguishes it from the quotient. -/
theorem frozenGradient_eq (x y z : ℝ) :
    frozenGradient x y z =
      (2 * residual x y z / squaredNorm x y z,
        -4 * residual x y z / squaredNorm x y z,
        2 * residual x y z / squaredNorm x y z) := by
  unfold frozenGradient
  rw [(numerator_deriv_x x y z).deriv, (numerator_deriv_y x y z).deriv,
    (numerator_deriv_z x y z).deriv]

/-- The actual quotient gradient is orthogonal to the representation:
the precise instantaneous norm law. Source: arXiv:2205.10343v2,
appendix conservation laws, equation `eq:Z0_conservation`. -/
theorem quotientGradient_radial_zero (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0) :
    x * (quotientGradient x y z).1 + y * (quotientGradient x y z).2.1 +
      z * (quotientGradient x y z).2.2 = 0 := by
  rw [quotientGradient_eq x y z hZ]
  dsimp
  unfold numerator residual squaredNorm
  ring

example : squaredNorm 1 0 0 ≠ 0 := by
  unfold squaredNorm
  norm_num

/-- Correct sum-gradient formula, retaining the term omitted in the
appendix's centroid calculation. Source: arXiv:2205.10343v2, appendix
conservation laws. It vanishes for centered representations, not in general. -/
theorem quotientGradient_total (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0) :
    (quotientGradient x y z).1 + (quotientGradient x y z).2.1 +
      (quotientGradient x y z).2.2 =
        -(2 * numerator x y z / squaredNorm x y z ^ 2) * (x + y + z) := by
  rw [quotientGradient_eq x y z hZ]
  dsimp
  ring

example : squaredNorm 0 1 0 ≠ 0 := by
  unfold squaredNorm
  norm_num

/-- Corrected centroid law: a zero initial centroid has zero
instantaneous drift. Source: arXiv:2205.10343v2, appendix conservation
laws. The added centered hypothesis restricts its unqualified wording;
section 3.2's normalized centered embeddings meet this premise. -/
theorem centered_total_zero (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0)
    (hC : x + y + z = 0) :
    (quotientGradient x y z).1 + (quotientGradient x y z).2.1 +
      (quotientGradient x y z).2.2 = 0 := by
  rw [quotientGradient_total x y z hZ, hC]
  ring

example : squaredNorm 1 (-2) 1 ≠ 0 ∧ (1 : ℝ) + (-2) + 1 = 0 := by
  unfold squaredNorm
  norm_num

/-- Counterexample to the appendix's unqualified centroid conservation:
the sum of negative-gradient velocities is 2 at `(1, 0, 0)`. Source:
arXiv:2205.10343v2, appendix conservation laws. This does not contradict
the corrected law for initially centered representations. -/
theorem noncentered_centroid_counterexample :
    quotientGradient 1 0 0 = (0, -4, 2) ∧
      -((quotientGradient 1 0 0).1 + (quotientGradient 1 0 0).2.1 +
        (quotientGradient 1 0 0).2.2) = 2 := by
  have hZ : squaredNorm 1 0 0 ≠ 0 := by unfold squaredNorm; norm_num
  rw [quotientGradient_eq 1 0 0 hZ]
  unfold numerator residual squaredNorm
  norm_num

/-- Centered, nonzero-loss counterexample to replacing the normalized
gradient by the numerator gradient over `Z₀`. Source: arXiv:2205.10343v2,
section 3.2, the deduction that gradient descent is linear. Its actual
gradient is zero but the claimed frozen vector is nonzero. -/
theorem centered_stationary_counterexample :
    (1 : ℝ) + (-2) + 1 = 0 ∧ 0 < normalizedLoss 1 (-2) 1 ∧
      quotientGradient 1 (-2) 1 = (0, 0, 0) ∧
      frozenGradient 1 (-2) 1 = (2, -4, 2) ∧
      quotientGradient 1 (-2) 1 ≠ frozenGradient 1 (-2) 1 := by
  have hZ : squaredNorm 1 (-2) 1 ≠ 0 := by unfold squaredNorm; norm_num
  rw [quotientGradient_eq 1 (-2) 1 hZ, frozenGradient_eq]
  unfold normalizedLoss numerator residual squaredNorm
  norm_num

/-- The norm along an actual negative-quotient-gradient trajectory has
zero derivative wherever the representation is nonzero. Source:
arXiv:2205.10343v2, appendix conservation laws, `eq:Z0_conservation`.
The ODE premises identify the flow; conservation is proved from them. -/
theorem norm_deriv_of_gradient_flow (x y z : ℝ → ℝ) (t : ℝ)
    (hZ : squaredNorm (x t) (y t) (z t) ≠ 0)
    (hx : HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t)
    (hy : HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t)
    (hz : HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) :
    HasDerivAt (fun s => squaredNorm (x s) (y s) (z s)) 0 t := by
  have h := ((hx.pow 2).add (hy.pow 2)).add (hz.pow 2)
  convert h using 1
  · rfl
  · norm_num
    nlinarith [quotientGradient_radial_zero (x t) (y t) (z t) hZ]

example : squaredNorm 1 (-2) 1 ≠ 0 ∧
    HasDerivAt (fun _ : ℝ => 1) (-(quotientGradient 1 (-2) 1).1) 0 ∧
    HasDerivAt (fun _ : ℝ => -2) (-(quotientGradient 1 (-2) 1).2.1) 0 ∧
    HasDerivAt (fun _ : ℝ => 1) (-(quotientGradient 1 (-2) 1).2.2) 0 := by
  have hg := centered_stationary_counterexample.2.2.1
  rw [hg]
  refine ⟨by unfold squaredNorm; norm_num, ?_, ?_, ?_⟩ <;>
    simpa using hasDerivAt_const (0 : ℝ) _

/-- Correct centroid derivative along the same actual quotient-gradient
flow. Source: arXiv:2205.10343v2, appendix conservation laws, replacing
its omitted denominator term; initially zero centroid has zero drift. -/
theorem centroid_deriv_of_gradient_flow (x y z : ℝ → ℝ) (t : ℝ)
    (hZ : squaredNorm (x t) (y t) (z t) ≠ 0)
    (hx : HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t)
    (hy : HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t)
    (hz : HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) :
    HasDerivAt (fun s => x s + y s + z s)
      (2 * numerator (x t) (y t) (z t) / squaredNorm (x t) (y t) (z t) ^ 2 *
        (x t + y t + z t)) t := by
  convert (hx.add hy).add hz using 1
  have hs := quotientGradient_total (x t) (y t) (z t) hZ
  nlinarith

example : squaredNorm 1 (-2) 1 ≠ 0 ∧
    HasDerivAt (fun _ : ℝ => 1) (-(quotientGradient 1 (-2) 1).1) 0 ∧
    HasDerivAt (fun _ : ℝ => -2) (-(quotientGradient 1 (-2) 1).2.1) 0 ∧
    HasDerivAt (fun _ : ℝ => 1) (-(quotientGradient 1 (-2) 1).2.2) 0 := by
  have hg := centered_stationary_counterexample.2.2.1
  rw [hg]
  refine ⟨by unfold squaredNorm; norm_num, ?_, ?_, ?_⟩ <;>
    simpa using hasDerivAt_const (0 : ℝ) _

/-- Complete norm conservation for a classical normalized-gradient flow
on its nonzero domain. Source: arXiv:2205.10343v2, appendix conservation
laws, `eq:Z0_conservation`; existence of the flow is not assumed proved. -/
theorem norm_const_of_gradient_flow (x y z : ℝ → ℝ)
    (hZ : ∀ t, squaredNorm (x t) (y t) (z t) ≠ 0)
    (hflow : ∀ t,
      HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) (s t : ℝ) :
    squaredNorm (x t) (y t) (z t) = squaredNorm (x s) (y s) (z s) := by
  have hd : ∀ u, HasDerivAt (fun v => squaredNorm (x v) (y v) (z v)) 0 u := by
    intro u
    exact norm_deriv_of_gradient_flow x y z u (hZ u)
      (hflow u).1 (hflow u).2.1 (hflow u).2.2
  exact is_const_of_deriv_eq_zero (fun u => (hd u).differentiableAt)
    (fun u => (hd u).deriv) t s

example : ∃ x y z : ℝ → ℝ,
    (∀ t, squaredNorm (x t) (y t) (z t) ≠ 0) ∧
    (∀ t, HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) := by
  refine ⟨fun _ => 1, fun _ => -2, fun _ => 1, ?_, ?_⟩
  · intro t
    unfold squaredNorm
    norm_num
  · intro t
    have hg := centered_stationary_counterexample.2.2.1
    refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

end Transformer.Grokking.EffectiveTheory
