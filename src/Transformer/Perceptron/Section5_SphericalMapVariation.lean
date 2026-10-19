module

public import Transformer.Perceptron.Section5_SphericalMapKernelBounds
public import Transformer.Perceptron.Section5_SphericalVariation

/-!
# Regularity of actual transported matrix interactions

ArXiv:2601.21366v2, §5.3, `thm: any.d` (i), and §3.1,
`rem: ext` (ii). Two original continuous tangent fields move
the two original integration points along their actual exponential
geodesics. The matrix kernel and its two actual derivatives
are continuous in this original pair, at every real time.

The pointwise operator-norm bounds give uniform domination once
the original fields have bounded speeds. Compactness subsequently
supplies those speed bounds and permits two genuine differentiations
under the original product probability measure. The two fields
may differ, so the same calculus also handles each actual one-point
transport needed in the global first-variation cancellation.
No finite atomic presentation or density is required.
The original matrix remains inside every bilinear exponent and
every derivative. The actual radial accelerations are retained.
-/

@[expose] public section

open Real MeasureTheory
namespace Transformer.Perceptron

/-- The original matrix kernel along the two actual Monge directions.
Source: arXiv:2601.21366v2, §2.2, §5.3 and §3.1, `rem: ext` (ii). -/
noncomputable def transportedMapPairKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (t : ℝ) (p : SSphere d × SSphere d) : ℝ :=
  sphereMapPairKernel B (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t

/-- The true first derivative with the same original fields.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
noncomputable def transportedMapPairForce {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (t : ℝ) (p : SSphere d × SSphere d) : ℝ :=
  sphereMapPairForce B (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t

/-- The true second derivative, including both radial accelerations.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
noncomputable def transportedMapPairCurvature {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (t : ℝ) (p : SSphere d × SSphere d) : ℝ :=
  sphereMapPairCurvature B (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t

/-- The original matrix kernel is continuous in the original pair.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem continuous_transportedMapPairKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (transportedMapPairKernel B ξ η t) := by
  have hX := (continuous_spherePosition_field ξ hξ t).comp
    (continuous_fst : Continuous (Prod.fst : SSphere d × SSphere d → SSphere d))
  have hY := (continuous_spherePosition_field η hη t).comp
    (continuous_snd : Continuous (Prod.snd : SSphere d × SSphere d → SSphere d))
  exact Real.continuous_exp.comp (hX.inner (B.toContinuousLinearMap.continuous.comp hY))

/-- Continuity of the actual exponent derivative in the original pair.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem continuous_transportedMapPairSlope {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (fun p : SSphere d × SSphere d =>
      sphereMapPairSlope B (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t) := by
  have hfst : Continuous (Prod.fst : SSphere d × SSphere d → SSphere d) := continuous_fst
  have hsnd : Continuous (Prod.snd : SSphere d × SSphere d → SSphere d) := continuous_snd
  have hX := (continuous_spherePosition_field ξ hξ t).comp hfst
  have hY := (continuous_spherePosition_field η hη t).comp hsnd
  have hV := (continuous_sphereVelocity_field ξ hξ t).comp hfst
  have hW := (continuous_sphereVelocity_field η hη t).comp hsnd
  exact (hV.inner (B.toContinuousLinearMap.continuous.comp hY)).add
    (hX.inner (B.toContinuousLinearMap.continuous.comp hW))

/-- Continuity retains the original matrix and radial accelerations.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem continuous_transportedMapPairAcceleration {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (fun p : SSphere d × SSphere d =>
      sphereMapPairAcceleration B (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (η p.2) t) := by
  have hfst : Continuous (Prod.fst : SSphere d × SSphere d → SSphere d) := continuous_fst
  have hsnd : Continuous (Prod.snd : SSphere d × SSphere d → SSphere d) := continuous_snd
  have hX := (continuous_spherePosition_field ξ hξ t).comp hfst
  have hY := (continuous_spherePosition_field η hη t).comp hsnd
  have hV := (continuous_sphereVelocity_field ξ hξ t).comp hfst
  have hW := (continuous_sphereVelocity_field η hη t).comp hsnd
  exact (((hV.inner (B.toContinuousLinearMap.continuous.comp hW)).const_mul 2)).sub
    ((((hξ.norm.comp continuous_fst).pow 2).add ((hη.norm.comp continuous_snd).pow 2)).mul
      (hX.inner (B.toContinuousLinearMap.continuous.comp hY)))

/-- The actual first derivative is continuous in the original pair.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem continuous_transportedMapPairForce {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (transportedMapPairForce B ξ η t) :=
  (continuous_transportedMapPairKernel B ξ η hξ hη t).mul
    (continuous_transportedMapPairSlope B ξ η hξ hη t)

/-- The actual second derivative is continuous in the original pair.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem continuous_transportedMapPairCurvature {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (hξ : Continuous ξ) (hη : Continuous η) (t : ℝ) :
    Continuous (transportedMapPairCurvature B ξ η t) :=
  (continuous_transportedMapPairKernel B ξ η hξ hη t).mul
    (((continuous_transportedMapPairSlope B ξ η hξ hη t).pow 2).add
      (continuous_transportedMapPairAcceleration B ξ η hξ hη t))

/-- Bounds on both original speeds dominate the two actual derivative
integrands at every real time. Source: arXiv:2601.21366v2, §5.3,
the domination for `thm: any.d` (i), and §3.1, `rem: ext` (ii). -/
theorem transportedMapPair_derivative_bounds {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0)
    (C D : ℝ) (hC : ∀ x, ‖ξ x‖ ≤ C) (hD : ∀ x, ‖η x‖ ≤ D) :
    (∀ t p, |transportedMapPairForce B ξ η t p| ≤
      Real.exp ‖B.toContinuousLinearMap‖ * (‖B.toContinuousLinearMap‖ * (C + D))) ∧
    (∀ t p, |transportedMapPairCurvature B ξ η t p| ≤ Real.exp ‖B.toContinuousLinearMap‖ *
      ((‖B.toContinuousLinearMap‖ ^ 2 + ‖B.toContinuousLinearMap‖) * (C + D) ^ 2)) := by
  constructor
  · intro t p
    apply (abs_sphereMapPairForce_le B (norm_eq_of_mem_sphere p.1) (hξ p.1)
      (norm_eq_of_mem_sphere p.2) (hη p.2) t).trans
    gcongr
    · exact hC p.1
    · exact hD p.2
  · intro t p
    apply (abs_sphereMapPairCurvature_le B (norm_eq_of_mem_sphere p.1) (hξ p.1)
      (norm_eq_of_mem_sphere p.2) (hη p.2) t).trans
    gcongr
    · exact hC p.1
    · exact hD p.2

/-- The original matrix exponential is continuous on the original product sphere.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3. -/
theorem continuous_expInnerMap_prod {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d) :
    Continuous (fun p : SSphere d × SSphere d =>
      Real.exp (inner (𝕜 := ℝ) (p.1 : EucSpace d) (B (p.2 : EucSpace d)))) :=
  Real.continuous_exp.comp ((continuous_subtype_val.comp continuous_fst).inner
    (B.toContinuousLinearMap.continuous.comp (continuous_subtype_val.comp continuous_snd)))

/-- The actual mixed curvature is continuous for the original continuous field.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem continuous_sphereMapPairMixedCurvature_field {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ : SSphere d → EucSpace d) (hcξ : Continuous ξ) :
    Continuous (fun p : SSphere d × SSphere d => sphereMapPairMixedCurvature B
      (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (ξ p.2)) := by
  have hfst : Continuous (Prod.fst : SSphere d × SSphere d → SSphere d) := continuous_fst
  have hsnd : Continuous (Prod.snd : SSphere d × SSphere d → SSphere d) := continuous_snd
  have hX := continuous_subtype_val.comp hfst
  have hY := continuous_subtype_val.comp hsnd
  have hV := hcξ.comp hfst
  have hW := hcξ.comp hsnd
  have hYB := B.toContinuousLinearMap.continuous.comp hY
  have hWB := B.toContinuousLinearMap.continuous.comp hW
  exact (continuous_expInnerMap_prod B).mul
    (((hV.inner hYB).mul (hX.inner hWB)).add (hV.inner hWB))

/-- The actual nonzero unit circle tangent jointly realizes all
continuity, tangency and speed-bound inputs, including identical fields.
Source: arXiv:2601.21366v2, §2.2 and §5.2, the circle transport. -/
example : Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2)
      (circleTangent (x : EucSpace 2)) = 0) ∧
    (∀ x : SSphere 2, ‖circleTangent (x : EucSpace 2)‖ ≤ 1) :=
  ⟨continuous_circleTangent.comp continuous_subtype_val,
    fun x => inner_circleTangent (x : EucSpace 2),
    fun x => by rw [norm_circleTangent, norm_eq_of_mem_sphere x]⟩

end Transformer.Perceptron
