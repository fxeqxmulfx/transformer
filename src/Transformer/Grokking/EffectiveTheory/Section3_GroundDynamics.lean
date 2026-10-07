import Transformer.Grokking.EffectiveTheory.Section3_RelativeModes
import Transformer.Grokking.EffectiveTheory.Section3_BoundedGrowth

/-!
# Ground components cannot vanish from nonzero initial data

Source: Liu et al., arXiv:2205.10343v2, section 3.2, "Time towards the
linear structure". This strengthens the corrected relative-mode analysis
for the explicit one-parallelogram loss from `Section3_QuotientGradient`.

The ground component is `E₀ - E₂`. Its equation is derived from the actual
quotient gradient, including the denominator derivative. The squared
constraint residual is at most six times the squared representation norm,
so its growth coefficient lies between zero and `12 / Z₀`. Proved norm
conservation makes the upper bound uniform. A nonzero ground component at
time zero therefore stays nonzero; its squared amplitude is nondecreasing.

A classical flow on the nonzero representation domain is still a premise.
Existence and transfer to GPTMini/AdamW are not inferred from this bound.
-/

namespace Transformer.Grokking.EffectiveTheory

/-- Exact finite-dimensional residual bound for the single parallelogram.
Source specialization: arXiv:2205.10343v2, section 3.2, equation `eq:l_eff`.
The constant six comes from the residual coefficients `(1, -2, 1)`. -/
theorem numerator_le_six_norm (x y z : ℝ) :
    numerator x y z ≤ 6 * squaredNorm x y z := by
  unfold numerator residual squaredNorm
  nlinarith [sq_nonneg (2 * x + y), sq_nonneg (2 * z + y), sq_nonneg (x - z)]

/-- The radial growth coefficient has a finite, positive-norm bound.
Source correction: arXiv:2205.10343v2, section 3.2; the coefficient
retained when differentiating `ℓ₀ / Z₀`, not the frozen linear field. -/
theorem ground_coefficient_bound (x y z : ℝ) (hZ : squaredNorm x y z ≠ 0) :
    0 ≤ 2 * numerator x y z / squaredNorm x y z ^ 2 ∧
      2 * numerator x y z / squaredNorm x y z ^ 2 ≤ 12 / squaredNorm x y z := by
  have hn : 0 ≤ numerator x y z := by unfold numerator; positivity
  constructor
  · positivity
  · apply (div_le_iff₀ (sq_pos_of_ne_zero hZ)).mpr
    have hm : 12 / squaredNorm x y z * squaredNorm x y z ^ 2 =
        12 * squaredNorm x y z := by field_simp
    rw [hm]
    nlinarith [numerator_le_six_norm x y z]

example : squaredNorm 1 (-1) 0 ≠ 0 := by unfold squaredNorm; norm_num

/-- Actual ground-component derivative. Source correction:
arXiv:2205.10343v2, section 3.2, the quotient-gradient equation;
the ground coordinate grows through the radial correction. -/
theorem ground_deriv_of_gradient_flow (x y z : ℝ → ℝ) (t : ℝ)
    (hZ : squaredNorm (x t) (y t) (z t) ≠ 0)
    (hx : HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t)
    (hz : HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) :
    HasDerivAt (fun s => x s - z s)
      (2 * numerator (x t) (y t) (z t) / squaredNorm (x t) (y t) (z t) ^ 2 *
        (x t - z t)) t := by
  convert hx.sub hz using 1
  rw [quotientGradient_eq (x t) (y t) (z t) hZ]
  dsimp
  ring

example : squaredNorm 1 (-1) 0 ≠ 0 ∧
    HasDerivAt (fun s : ℝ => 1 + (3 / 2) * s) (-(quotientGradient 1 (-1) 0).1) 0 ∧
    HasDerivAt (fun s : ℝ => -3 * s) (-(quotientGradient 1 (-1) 0).2.2) 0 := by
  have hg := nonstationary_centered_example.2.2.2.1
  rw [hg]
  refine ⟨by unfold squaredNorm; norm_num, ?_, ?_⟩
  · convert ((hasDerivAt_id (0 : ℝ)).const_mul (3 / 2 : ℝ)).const_add 1 using 1 <;> norm_num
  · simpa using (hasDerivAt_id (0 : ℝ)).const_mul (-3 : ℝ)

/-- Initial ground-component nonvanishing suffices for every finite time.
Source correction: arXiv:2205.10343v2, section 3.2. The uniform coefficient
bound is proved from the actual loss and conserved norm, replacing the
all-time nonvanishing premise of the earlier relative-mode theorem. -/
theorem ground_nonzero_from_initial (x y z : ℝ → ℝ)
    (hZ : ∀ t, squaredNorm (x t) (y t) (z t) ≠ 0)
    (hflow : ∀ t,
      HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t)
    (h0 : x 0 - z 0 ≠ 0) : ∀ t, x t - z t ≠ 0 := by
  apply growth_nonvanishing_from_initial (fun t => x t - z t)
    (fun t => 2 * numerator (x t) (y t) (z t) / squaredNorm (x t) (y t) (z t) ^ 2)
    (12 / squaredNorm (x 0) (y 0) (z 0))
  · intro t
    exact ground_deriv_of_gradient_flow x y z t (hZ t) (hflow t).1 (hflow t).2.2
  · intro t
    have hn := norm_const_of_gradient_flow x y z hZ hflow 0 t
    change 0 ≤ 2 * numerator (x t) (y t) (z t) / squaredNorm (x t) (y t) (z t) ^ 2 ∧
      2 * numerator (x t) (y t) (z t) / squaredNorm (x t) (y t) (z t) ^ 2 ≤
        12 / squaredNorm (x 0) (y 0) (z 0)
    simpa only [hn] using ground_coefficient_bound (x t) (y t) (z t) (hZ t)
  · exact h0

example : ∃ x y z : ℝ → ℝ,
    (∀ t, squaredNorm (x t) (y t) (z t) ≠ 0) ∧
    (∀ t, HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) ∧ x 0 - z 0 ≠ 0 := by
  refine ⟨fun _ => 1, fun _ => 0, fun _ => -1, ?_, ?_, by norm_num⟩
  · intro t
    unfold squaredNorm
    norm_num
  · intro t
    have hg : quotientGradient 1 0 (-1) = (0, 0, 0) :=
      ground_gradient_zero 1 0 (-1) (by unfold squaredNorm; norm_num)
        (by unfold residual; norm_num)
    refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

/-- The actual ground-mode squared amplitude is nondecreasing. Source
correction: arXiv:2205.10343v2, section 3.2. Norm conservation can coexist
with this growth because the residual mode loses relative amplitude. -/
theorem ground_energy_monotone (x y z : ℝ → ℝ)
    (hZ : ∀ t, squaredNorm (x t) (y t) (z t) ≠ 0)
    (hflow : ∀ t,
      HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) :
    Monotone (fun t => (x t - z t) ^ 2) := by
  apply growth_energy_monotone (fun t => x t - z t)
    (fun t => 2 * numerator (x t) (y t) (z t) / squaredNorm (x t) (y t) (z t) ^ 2)
  · intro t
    exact ground_deriv_of_gradient_flow x y z t (hZ t) (hflow t).1 (hflow t).2.2
  · intro t
    exact (ground_coefficient_bound (x t) (y t) (z t) (hZ t)).1

example : ∃ x y z : ℝ → ℝ,
    (∀ t, squaredNorm (x t) (y t) (z t) ≠ 0) ∧
    (∀ t, HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) := by
  refine ⟨fun _ => 1, fun _ => 0, fun _ => -1, ?_, ?_⟩
  · intro t
    unfold squaredNorm
    norm_num
  · intro t
    have hg : quotientGradient 1 0 (-1) = (0, 0, 0) :=
      ground_gradient_zero 1 0 (-1) (by unfold squaredNorm; norm_num)
        (by unfold residual; norm_num)
    refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

/-- The corrected spectral formula now needs a nonzero initial ground
component, rather than nonvanishing at every time. Source correction:
arXiv:2205.10343v2, section 3.2, exponential mode evolution. -/
theorem relativeMode_exponential_from_initial_ground (x y z : ℝ → ℝ)
    (hZ : ∀ t, squaredNorm (x t) (y t) (z t) ≠ 0)
    (hflow : ∀ t,
      HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t)
    (h0 : x 0 - z 0 ≠ 0) (t : ℝ) :
    relativeMode (x t) (y t) (z t) = relativeMode (x 0) (y 0) (z 0) *
      Real.exp (-(12 / squaredNorm (x 0) (y 0) (z 0)) * t) := by
  have hA := ground_nonzero_from_initial x y z hZ hflow h0
  exact relativeMode_exponential_of_gradient_flow x y z hZ hA hflow t

example : ∃ x y z : ℝ → ℝ,
    (∀ t, squaredNorm (x t) (y t) (z t) ≠ 0) ∧
    (∀ t, HasDerivAt x (-(quotientGradient (x t) (y t) (z t)).1) t ∧
      HasDerivAt y (-(quotientGradient (x t) (y t) (z t)).2.1) t ∧
      HasDerivAt z (-(quotientGradient (x t) (y t) (z t)).2.2) t) ∧ x 0 - z 0 ≠ 0 := by
  refine ⟨fun _ => 1, fun _ => 0, fun _ => -1, ?_, ?_, by norm_num⟩
  · intro t
    unfold squaredNorm
    norm_num
  · intro t
    have hg : quotientGradient 1 0 (-1) = (0, 0, 0) :=
      ground_gradient_zero 1 0 (-1) (by unfold squaredNorm; norm_num)
        (by unfold residual; norm_num)
    refine ⟨?_, ?_, ?_⟩ <;> simpa [hg] using hasDerivAt_const t _

end Transformer.Grokking.EffectiveTheory
