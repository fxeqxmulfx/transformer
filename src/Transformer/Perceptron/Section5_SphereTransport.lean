module

public import Transformer.Perceptron.Section5_SphericalExponential
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Actual Monge geodesic transport in arbitrary dimension

ArXiv:2601.21366v2, §2.2, `T(t)(x)=exp_x(tξ(x))`, and §5.3,
the localization argument for `thm: any.d` (i). The actual exponential
curve from `Section5_SphericalExponential` gives a sphere-valued map
for every continuous tangent field. Its actual probability pushforward
realizes the original geodesic predicate, with the actual exponential
formula and the original measure.

Smooth spherical gradient fields are continuous. The displacement
bound follows from their actual constant geodesic speed. These
regularity facts supply the transport maps and bounds used later to
differentiate the original energy, without assuming a Hessian formula.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The original exponential transport as an actual sphere-valued map.
Source: arXiv:2601.21366v2, §2.2, the Monge exponential-map geodesic. -/
noncomputable def sphereTransport {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (t : ℝ) (x : SSphere d) : SSphere d :=
  ⟨spherePosition (x : EucSpace d) (ξ x) t,
    mem_sphere_zero_iff_norm.mpr (norm_spherePosition (norm_eq_of_mem_sphere x) (hξ x) t)⟩

/-- The actual transport agrees with the original spherical exponential.
Source: arXiv:2601.21366v2, §2.2, `T(t)(x)=exp_x(tξ(x))`. -/
theorem sphereTransport_eq_sphereExp {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (t : ℝ) (x : SSphere d) :
    (sphereTransport ξ hξ t x : EucSpace d) = sphereExp (x : EucSpace d) (t • ξ x) :=
  (sphereExp_smul_eq _ _ _).symm

/-- Continuous actual tangent fields give continuous original Monge maps.
Source: arXiv:2601.21366v2, §2.2 and §5.3, the geodesic transport maps. -/
theorem continuous_sphereTransport {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hc : Continuous ξ) (t : ℝ) :
    Continuous (sphereTransport ξ hξ t) := by
  apply Continuous.subtype_mk
  change Continuous (fun x : SSphere d => Real.cos (t * ‖ξ x‖) • (x : EucSpace d) +
    (t * Real.sinc (t * ‖ξ x‖)) • ξ x)
  exact ((Real.continuous_cos.comp (hc.norm.const_mul t)).smul continuous_subtype_val).add
    ((continuous_const.mul (Real.continuous_sinc.comp (hc.norm.const_mul t))).smul hc)

/-- The original transport starts at the actual original point.
Source: arXiv:2601.21366v2, §2.2, the geodesic issued from `μ`. -/
@[simp] theorem sphereTransport_zero {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0) (x : SSphere d) :
    sphereTransport ξ hξ 0 x = x := Subtype.ext (spherePosition_zero _ _)

/-- The original probability pushforward curve is an actual geodesic
issued from the same original probability and field.
Source: arXiv:2601.21366v2, §2.2, the Monge geodesic in `eq:strict2order`. -/
theorem isGeodesicFrom_sphereTransport {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0) (hc : Continuous ξ)
    (μ : Perspective.ProbSphere d) :
    IsGeodesicFrom ξ μ (fun t => μ.map (sphereTransport ξ hξ t)) := by
  intro t
  exact ⟨sphereTransport ξ hξ t, (continuous_sphereTransport ξ hξ hc t).measurable,
    sphereTransport_eq_sphereExp ξ hξ t, rfl⟩

/-- A smooth ambient function has a continuous actual gradient in every
dimension. Source: arXiv:2601.21366v2, §2.2, `eq:otto.tangent`, and §5.3. -/
theorem continuous_gradient_of_contDiff_any_d {d : ℕ} (f : EucSpace d → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) : Continuous (gradient f) :=
  (InnerProductSpace.toDual ℝ (EucSpace d)).symm.continuous.comp
    (hf.continuous_fderiv (by simp))

/-- The original smooth spherical gradient fields are continuous, so
their actual transport maps are measurable without a separate regularity
assumption. Source: arXiv:2601.21366v2, §2.2, `eq:otto.tangent`, and §5.3. -/
theorem continuous_of_isGradientField {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : IsGradientField ξ) : Continuous ξ := by
  obtain ⟨f, hf, heq⟩ := hξ
  have hfg : Continuous (fun x : SSphere d => gradient f (x : EucSpace d)) :=
    (continuous_gradient_of_contDiff_any_d f hf).comp continuous_subtype_val
  have hin : Continuous (fun x : SSphere d =>
      inner (𝕜 := ℝ) (x : EucSpace d) (gradient f (x : EucSpace d))) :=
    continuous_subtype_val.inner hfg
  have h : ξ = fun x : SSphere d => proj d (x : EucSpace d) (gradient f (x : EucSpace d)) :=
    funext heq
  rw [h]
  unfold proj
  exact hfg.sub (hin.smul continuous_subtype_val)

/-- The actual source gradient direction gives its actual Monge
pushforward geodesic. Source: arXiv:2601.21366v2, §2.2, `eq:strict2order`. -/
theorem isGeodesicFrom_gradientTransport {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : IsGradientField ξ) (μ : Perspective.ProbSphere d) :
    IsGeodesicFrom ξ μ (fun t => μ.map (sphereTransport ξ (inner_isGradientField hξ) t)) :=
  isGeodesicFrom_sphereTransport ξ (inner_isGradientField hξ) (continuous_of_isGradientField ξ hξ) μ

/-- The actual geodesic velocity of the original transport map.
Source: arXiv:2601.21366v2, §2.2 and §5.3, the transported first derivative. -/
theorem hasDerivAt_sphereTransport_coe {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (x : SSphere d) (t : ℝ) :
    HasDerivAt (fun t => (sphereTransport ξ hξ t x : EucSpace d))
      (sphereVelocity (x : EucSpace d) (ξ x) t) t := hasDerivAt_spherePosition _ _ _

/-- The actual velocity keeps the original field's pointwise norm.
Source: arXiv:2601.21366v2, §2.2 and §5.3, geodesic speed in the Hessian localization. -/
theorem norm_sphereTransport_velocity {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (x : SSphere d) (t : ℝ) :
    ‖sphereVelocity (x : EucSpace d) (ξ x) t‖ = ‖ξ x‖ :=
  norm_sphereVelocity (norm_eq_of_mem_sphere x) (hξ x) t

/-- The actual velocity is continuous in the original point for a
continuous field. Source: arXiv:2601.21366v2, §5.3, the transported
kernel derivatives in the localization argument for `thm: any.d` (i). -/
theorem continuous_sphereVelocity_field {d : ℕ} (ξ : SSphere d → EucSpace d)
    (hc : Continuous ξ) (t : ℝ) :
    Continuous (fun x : SSphere d => sphereVelocity (x : EucSpace d) (ξ x) t) := by
  unfold sphereVelocity
  exact (((hc.norm.mul (Real.continuous_sin.comp (hc.norm.const_mul t))).neg.smul
    continuous_subtype_val).add
      ((Real.continuous_cos.comp (hc.norm.const_mul t)).smul hc))

/-- The actual spherical displacement is bounded by time times initial
speed. Source: arXiv:2601.21366v2, §2.2 and §5.3, the localized transport estimate. -/
theorem norm_spherePosition_sub_le {d : ℕ} {x v : EucSpace d} (hx : ‖x‖ = 1)
    (hv : inner (𝕜 := ℝ) x v = 0) (t : ℝ) : ‖spherePosition x v t - x‖ ≤ ‖v‖ * |t| := by
  have h := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (s := univ) (fun r _ => (hasDerivAt_spherePosition x v r).hasDerivWithinAt)
    (fun r _ => (norm_sphereVelocity hx hv r).le) (convex_univ) (mem_univ 0) (mem_univ t)
  simpa only [spherePosition_zero, sub_zero, Real.norm_eq_abs] using h

/-- The nonzero continuous unit tangent on the actual circle witnesses
every continuity and tangency input of the general transport results.
Source: arXiv:2601.21366v2, §2.2 and §5.2, the exponential-map geodesic. -/
example : Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2) (circleTangent (x : EucSpace 2)) = 0) ∧
    IsGeodesicFrom (fun x : SSphere 2 => circleTangent (x : EucSpace 2))
      (Perspective.diracProb 2 (basePoint 1))
      (fun t => (Perspective.diracProb 2 (basePoint 1)).map
        (sphereTransport (fun x => circleTangent (x : EucSpace 2))
          (fun x => inner_circleTangent (x : EucSpace 2)) t)) :=
  ⟨continuous_circleTangent.comp continuous_subtype_val,
    fun x => inner_circleTangent (x : EucSpace 2),
    isGeodesicFrom_sphereTransport _ (fun x => inner_circleTangent (x : EucSpace 2))
      (continuous_circleTangent.comp continuous_subtype_val) _⟩

/-- The gradient-field regularity inputs are realized by the genuine
smooth constant potential and its zero gradient, in the nonempty sphere.
Source: arXiv:2601.21366v2, §2.2, the dense smooth-gradient subspace. -/
example : ContDiff ℝ (⊤ : ℕ∞) (fun _ : EucSpace 2 => (0 : ℝ)) ∧
    IsGradientField (fun _ : SSphere 2 => (0 : EucSpace 2)) :=
  ⟨contDiff_const, isGradientField_zero⟩

end Transformer.Perceptron
