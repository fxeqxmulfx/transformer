module

public import Transformer.Causal.Section4_AnisotropicCircle
public import Transformer.Landscape.GlobalFlow.Existence
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Genuine global solutions for the anisotropic causal field

Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`, and
`eqn:csa`, with `V=diag(2,1)`, `Q=I`, `K=diag(0,4)`, `β=1`.
The actual softmax weights are positive and sum to one. The sine
bound gives a linear growth estimate for the actual angular field,
so the proved global ODE existence theorem supplies real trajectories.

The original circle chart lifts each such trajectory to the actual
masked causal equation. Every initial pair on the original sphere
has an angular representative; no solution, forward invariance or
limiting behavior is postulated in the definition of the field.
-/

@[expose] public section

open Real

namespace Transformer.Causal

open Perceptron

/-- Every sine coordinate is bounded by the actual norm of the angle pair.
Source: arXiv:2411.04990v2, §4, the two-angle representation of `eqn:csa`. -/
theorem abs_sin_angle_le_norm (φ : Fin 2 → ℝ) (i : Fin 2) :
    |Real.sin (φ i)| ≤ ‖φ‖ := by
  exact Real.abs_sin_le_abs.trans (by simpa only [Real.norm_eq_abs] using norm_le_pi_norm φ i)

/-- The original anisotropic field has linear growth in the actual
angle-pair norm. Source: arXiv:2411.04990v2, §4, `eqn:csa`.
This bound follows from the original positive normalized weights. -/
theorem norm_anisotropicAngleField_le (φ : Fin 2 → ℝ) :
    ‖anisotropicAngleField φ‖ ≤ 3 * ‖φ‖ := by
  have hs : ∀ i : Fin 2, |Real.sin (φ i)| ≤ ‖φ‖ := abs_sin_angle_le_norm φ
  have hc : ∀ i : Fin 2, |Real.cos (φ i)| ≤ 1 := fun i => Real.abs_cos_le_one _
  have hself : |Real.cos (φ 1) * Real.sin (φ 1)| ≤ ‖φ‖ := by
    rw [abs_mul]
    calc
      _ ≤ 1 * ‖φ‖ := by gcongr; exact hc 1; exact hs 1
      _ = _ := one_mul _
  have hcross : |-2 * Real.cos (φ 0) * Real.sin (φ 1) + Real.sin (φ 0) * Real.cos (φ 1)| ≤
      3 * ‖φ‖ := by
    calc
      _ ≤ |-2 * Real.cos (φ 0) * Real.sin (φ 1)| + |Real.sin (φ 0) * Real.cos (φ 1)| := abs_add_le _ _
      _ = 2 * |Real.cos (φ 0)| * |Real.sin (φ 1)| + |Real.sin (φ 0)| * |Real.cos (φ 1)| := by
        simp only [abs_mul, abs_neg, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
      _ ≤ 2 * 1 * ‖φ‖ + ‖φ‖ * 1 := by gcongr; exact hc 0; exact hs 1; exact hs 0; exact hc 1
      _ = _ := by ring
  apply (pi_norm_le_iff_of_nonneg (by positivity)).mpr
  intro i
  fin_cases i
  · change ‖-Real.cos (φ 0) * Real.sin (φ 0)‖ ≤ 3 * ‖φ‖
    rw [Real.norm_eq_abs, abs_mul, abs_neg]
    calc
      _ ≤ 1 * ‖φ‖ := by gcongr; exact hc 0; exact hs 0
      _ ≤ _ := by nlinarith [norm_nonneg φ]
  · let a := Real.exp (4 * Real.sin (φ 1) ^ 2)
    let b := Real.exp (4 * Real.sin (φ 0) * Real.sin (φ 1))
    have ha : 0 < a := Real.exp_pos _
    have hb : 0 < b := Real.exp_pos _
    have hn : |-a * Real.cos (φ 1) * Real.sin (φ 1) +
        b * (-2 * Real.cos (φ 0) * Real.sin (φ 1) + Real.sin (φ 0) * Real.cos (φ 1))| ≤
        (a + b) * (3 * ‖φ‖) := by
      calc
        _ ≤ |-a * Real.cos (φ 1) * Real.sin (φ 1)| +
            |b * (-2 * Real.cos (φ 0) * Real.sin (φ 1) + Real.sin (φ 0) * Real.cos (φ 1))| := abs_add_le _ _
        _ = a * |Real.cos (φ 1) * Real.sin (φ 1)| +
            b * |-2 * Real.cos (φ 0) * Real.sin (φ 1) + Real.sin (φ 0) * Real.cos (φ 1)| := by
          simp only [abs_mul, abs_neg, abs_of_pos ha, abs_of_pos hb]
          ring
        _ ≤ a * ‖φ‖ + b * (3 * ‖φ‖) := by gcongr
        _ ≤ _ := by nlinarith [mul_nonneg ha.le (norm_nonneg φ)]
    change ‖(-a * Real.cos (φ 1) * Real.sin (φ 1) +
      b * (-2 * Real.cos (φ 0) * Real.sin (φ 1) + Real.sin (φ 0) * Real.cos (φ 1))) / (a + b)‖ ≤ _
    rw [Real.norm_eq_abs, abs_div, abs_of_pos (add_pos ha hb)]
    exact (div_le_iff₀ (add_pos ha hb)).mpr (by simpa only [mul_comm] using hn)

/-- The actual angular field vanishes at the original leading configuration.
Source: arXiv:2411.04990v2, §4, the leading eigenvector in `eqn:csa`. -/
@[simp] theorem anisotropicAngleField_zero : anisotropicAngleField 0 = 0 := by
  ext i
  fin_cases i <;> simp [anisotropicAngleField]

/-- A genuine solution exists for every original initial angle pair, for
all real times. Source: arXiv:2411.04990v2, §4, `eqn:csa`.
The ODE input is proved local Lipschitz regularity and actual linear growth. -/
theorem exists_anisotropicAngleFlow (φ₀ : Fin 2 → ℝ) :
    ∃ φ : ℝ → Fin 2 → ℝ, φ 0 = φ₀ ∧
      ∀ t : ℝ, HasDerivAt φ (anisotropicAngleField (φ t)) t := by
  exact Landscape.GlobalFlow.exists_global contDiff_anisotropicAngleField.locallyLipschitz
    (by norm_num : (0 : ℝ) ≤ 3) norm_anisotropicAngleField_le φ₀

/-- Each coordinate of a true angular solution has its actual field
derivative. Source: arXiv:2411.04990v2, §4, the coordinate velocities in `eqn:csa`. -/
theorem hasDerivAt_anisotropicAngle_coordinate (φ : ℝ → Fin 2 → ℝ)
    (hφ : ∀ t, HasDerivAt φ (anisotropicAngleField (φ t)) t) (t : ℝ) (i : Fin 2) :
    HasDerivAt (fun s => φ s i) (anisotropicAngleField (φ t) i) t :=
  (hasDerivAt_pi.mp (hφ t)) i

/-- The leader's second coordinate follows its actual diagonal-value
single-token equation. Source: arXiv:2411.04990v2, §3–§4,
the leader case of `eqn:csa` with `V=diag(2,1)`. -/
theorem hasDerivAt_anisotropic_leader_sin (φ : ℝ → Fin 2 → ℝ)
    (hφ : ∀ t, HasDerivAt φ (anisotropicAngleField (φ t)) t) (t : ℝ) :
    HasDerivAt (fun s => Real.sin (φ s 0))
      (-Real.cos (φ t 0) ^ 2 * Real.sin (φ t 0)) t := by
  have h : HasDerivAt (fun s => φ s 0) (-Real.cos (φ t 0) * Real.sin (φ t 0)) t :=
    hasDerivAt_anisotropicAngle_coordinate φ hφ t 0
  exact h.sin.congr_deriv (by ring)

/-- The leader's first coordinate has its original norm-preserving
derivative. Source: arXiv:2411.04990v2, §3–§4,
the single-token equation entering `thm1.5`. -/
theorem hasDerivAt_anisotropic_leader_cos (φ : ℝ → Fin 2 → ℝ)
    (hφ : ∀ t, HasDerivAt φ (anisotropicAngleField (φ t)) t) (t : ℝ) :
    HasDerivAt (fun s => Real.cos (φ s 0))
      (Real.cos (φ t 0) * Real.sin (φ t 0) ^ 2) t := by
  have h : HasDerivAt (fun s => φ s 0) (-Real.cos (φ t 0) * Real.sin (φ t 0)) t :=
    hasDerivAt_anisotropicAngle_coordinate φ hφ t 0
  exact h.cos.congr_deriv (by ring)

/-- The original masked causal equation has a genuine global solution
through every original sphere configuration for these actual matrices.
Source: arXiv:2411.04990v2, §4, `eqn:csa`.
Angular representatives come from the proved surjectivity of the circle
chart; no existence premise is added to the original equation. -/
theorem exists_anisotropic_CSA (X₀ : SphereTuple 2 2) :
    ∃ X : ℝ → SphereTuple 2 2, X 0 = X₀ ∧
      CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2)) anisotropicKeyMap anisotropicValueMap X := by
  choose φ₀ hφ₀ using fun i : Fin 2 => circlePoint_surjective (X₀ i)
  obtain ⟨φ, h0, hφ⟩ := exists_anisotropicAngleFlow φ₀
  refine ⟨fun t i => circlePoint (φ t i), ?_, CSA_of_anisotropicAngleFlow φ hφ⟩
  funext i
  change circlePoint (φ 0 i) = X₀ i
  rw [h0]
  exact hφ₀ i

/-- The solution hypotheses above hold for a genuine original leading
configuration. Its two points are unit sphere points and its angles solve
the actual field equation at every time. Source: arXiv:2411.04990v2,
§4, `eqn:csa` and the leading eigenvector in `thm1.5`. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : Fin 2 → ℝ))
      (anisotropicAngleField 0) t) ∧
    CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2)) anisotropicKeyMap anisotropicValueMap
      (fun _ i => circlePoint ((0 : Fin 2 → ℝ) i)) := by
  have h : ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : Fin 2 → ℝ)) (anisotropicAngleField 0) t := by
    intro t
    simpa only [anisotropicAngleField_zero] using hasDerivAt_const t (0 : Fin 2 → ℝ)
  exact ⟨h, CSA_of_anisotropicAngleFlow (fun _ => 0) h⟩

end Transformer.Causal
