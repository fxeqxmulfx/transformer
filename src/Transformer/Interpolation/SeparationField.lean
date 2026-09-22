/-
# Measure-to-measure interpolation — the separation field

The frozen perceptron of Step 2 of the proof of the interpolation theorem of
arXiv:2411.04551v3 (§4), `eq: neural.ode.separation`,

  `ẋ(t) = (⟨γ, x(t)⟩ - ε/2)_+ Proj_{x(t)} ω_+`,

its reading as the perceptron field at `𝐔_1 = 𝟏 γᵀ`, `b_1 = -(ε/2) 𝟏`,
`𝐖_1 𝟏 = ω_+`, and the two elementary tools `eq: Hartman.Grobman` is proved
with in `Interpolation.HartmanGrobman`: a fencing lemma, and the comparison of
the geodesic with the chordal distance on the sphere.

Source: arXiv:2411.04551v3, §4, Step 2.
-/

import Transformer.Interpolation.Settling
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.Analysis.InnerProductSpace.Continuous

open Real

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- The right-hand side of **`eq: neural.ode.separation`**,

  `(⟨γ, z⟩ - ε/2)_+ Proj_z ω_+`.

Source: arXiv:2411.04551v3, §4, Step 2, `eq: neural.ode.separation`. -/
noncomputable def separationField (γ ω : EucSpace d) (ε : ℝ) (z : EucSpace d) :
    EucSpace d :=
  max (inner (𝕜 := ℝ) γ z - ε / 2) 0 • proj d z ω

/-- **Fencing.**  A function that starts `≤ 0` and, wherever it is positive,
grows at most linearly in itself, stays `≤ 0`.  It is compared with the
barriers `δ e^{(|C|+1) t}`, `δ > 0`, by
`image_le_of_deriv_right_lt_deriv_boundary`. -/
theorem nonpos_of_deriv_le_mul (φ φ' : ℝ → ℝ) (C T : ℝ)
    (hφ : ∀ t ∈ Set.Icc 0 T, HasDerivAt φ (φ' t) t) (h0 : φ 0 ≤ 0)
    (hle : ∀ t ∈ Set.Ico 0 T, 0 < φ t → φ' t ≤ C * φ t) :
    ∀ t ∈ Set.Icc 0 T, φ t ≤ 0 := by
  have hB : ∀ δ : ℝ, 0 < δ → ∀ t ∈ Set.Icc (0 : ℝ) T,
      φ t ≤ δ * Real.exp ((|C| + 1) * t) := by
    intro δ hδ
    refine image_le_of_deriv_right_lt_deriv_boundary (f' := φ')
      (B' := fun s => δ * (Real.exp ((|C| + 1) * s) * (|C| + 1)))
      (fun s hs => (hφ s hs).continuousAt.continuousWithinAt)
      (fun s hs => (hφ s (Set.Ico_subset_Icc_self hs)).hasDerivWithinAt) ?_ ?_ ?_
    · simp only [mul_zero, Real.exp_zero, mul_one]
      linarith
    · intro s
      have h := ((hasDerivAt_id s).const_mul (|C| + 1)).exp.const_mul δ
      simpa using h
    · intro s hs heq
      have hE : 0 < δ * Real.exp ((|C| + 1) * s) := by positivity
      have h1 := hle s hs (heq ▸ hE)
      rw [heq] at h1
      have h2 : C * (δ * Real.exp ((|C| + 1) * s)) ≤ |C| * (δ * Real.exp ((|C| + 1) * s)) :=
        mul_le_mul_of_nonneg_right (le_abs_self C) hE.le
      nlinarith
  intro t ht
  by_contra hcon
  push Not at hcon
  have hE : 0 < Real.exp ((|C| + 1) * t) := Real.exp_pos _
  have h := hB (φ t / (2 * Real.exp ((|C| + 1) * t))) (by positivity) t ht
  rw [div_mul_eq_mul_div, mul_comm 2, ← div_div, mul_div_cancel_right₀ _ hE.ne'] at h
  linarith

/-- **The geodesic distance is at most `π/2` times the chordal one:**
`arccos ⟨x, y⟩ ≤ (π/2) ‖x - y‖` for unit vectors, by Jordan's inequality
`sin θ ≥ (2/π) θ` on `[0, π/2]`, since `‖x - y‖ = 2 sin(θ/2)`. -/
theorem arccos_inner_le_norm_sub {x y : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    Real.arccos (inner (𝕜 := ℝ) x y) ≤ π / 2 * ‖x - y‖ := by
  set u := inner (𝕜 := ℝ) x y with hu
  have hu1 : |u| ≤ 1 := by
    have := abs_real_inner_le_norm x y
    rwa [hx, hy, mul_one] at this
  set θ := Real.arccos u with hθ
  have hcos : Real.cos θ = u := Real.cos_arccos (abs_le.mp hu1).1 (abs_le.mp hu1).2
  have hθ0 : 0 ≤ θ := Real.arccos_nonneg u
  have hθπ : θ ≤ π := Real.arccos_le_pi u
  have hnorm : ‖x - y‖ ^ 2 = 2 - 2 * u := by
    rw [@norm_sub_sq_real, hx, hy, ← hu]
    ring
  have hhalf : Real.cos θ = 1 - 2 * Real.sin (θ / 2) ^ 2 := by
    rw [← Real.cos_two_mul_eq_one_sub]
    ring_nf
  have hsin0 : 0 ≤ Real.sin (θ / 2) :=
    Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith [Real.pi_pos])
  have hsinle : Real.sin (θ / 2) ≤ ‖x - y‖ / 2 := by
    nlinarith [norm_nonneg (x - y)]
  have hJ := Real.mul_le_sin (x := θ / 2) (by linarith) (by linarith)
  have hπ := Real.pi_pos
  rw [div_mul_eq_mul_div, div_le_iff₀ hπ] at hJ
  nlinarith

/-- The field of `eq: neural.ode.separation` tested against a vector `v`:
`⟨F(z), v⟩ = (⟨γ, z⟩ - ε/2)_+ (⟨ω_+, v⟩ - ⟨z, ω_+⟩ ⟨z, v⟩)`. -/
theorem inner_separationField (γ ω : EucSpace d) (ε : ℝ) (z v : EucSpace d) :
    inner (𝕜 := ℝ) (separationField d γ ω ε z) v
      = max (inner (𝕜 := ℝ) γ z - ε / 2) 0
          * (inner (𝕜 := ℝ) ω v - inner (𝕜 := ℝ) z ω * inner (𝕜 := ℝ) z v) := by
  rw [separationField, real_inner_smul_left, proj, inner_sub_left, real_inner_smul_left]

/-- **`eq: neural.ode.separation` is the perceptron flow at the parameters of
Step 2.**  With `𝐔_1 z = ⟨γ, z⟩ 𝟏`, `b_1 = -(ε/2) 𝟏` and `𝐖_1 𝟏 = ω_+`, the
perceptron field `Proj_z 𝐖_1 (𝐔_1 z + b_1)_+` is
`(⟨γ, z⟩ - ε/2)_+ Proj_z ω_+`.

Source: arXiv:2411.04551v3, §4, Step 2, the display before
`eq: neural.ode.separation`. -/
theorem perceptronField_eq_separationField (γ ω : EucSpace d) (ε : ℝ)
    (W U : ParamMatrix d) (b : EucSpace d)
    (hU : ∀ z, U z = inner (𝕜 := ℝ) γ z • (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1))
    (hb : b = (-(ε / 2)) • (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1))
    (hW : W ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1)) = ω) (z : EucSpace d) :
    perceptronField d W U b z = separationField d γ ω ε z := by
  have hrelu : (EuclideanSpace.equiv (Fin d) ℝ).symm
      (fun k => max ((EuclideanSpace.equiv (Fin d) ℝ (U z + b)) k) 0)
        = max (inner (𝕜 := ℝ) γ z - ε / 2) 0
            • (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1) := by
    ext k
    simp [hU, hb, sub_eq_add_neg]
  rw [perceptronField, hrelu, map_smul, hW, separationField, proj, proj, inner_smul_right,
    smul_sub, smul_smul]

/-- The hypotheses of `perceptronField_eq_separationField` are satisfiable for
every `γ` and `ε`: `𝐔_1 = 𝟏 γᵀ`, `b_1 = -(ε/2) 𝟏`, `𝐖_1 = Id`, `ω = 𝟏`. -/
example (γ : EucSpace d) (ε : ℝ) :
    ∃ (W U : ParamMatrix d) (b : EucSpace d),
      (∀ z, U z = inner (𝕜 := ℝ) γ z • (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1)) ∧
      b = (-(ε / 2)) • (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1) ∧
      W ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1))
        = (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1) :=
  ⟨ContinuousLinearMap.id ℝ _,
    (innerSL ℝ γ).smulRight ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1)), _,
    fun z => by simp, rfl, rfl⟩

end Interpolation
end Transformer
