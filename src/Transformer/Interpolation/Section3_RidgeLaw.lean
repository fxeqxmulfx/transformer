module

public import Transformer.Interpolation.Section3_RidgeFlowContinuity
public import Transformer.Interpolation.Section1_CompactTests

/-!
# Actual weak solutions transported by the single-neuron flow

ArXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf`, and
§3.2, `lem: perturbation`, Part 2.

Push the original spherical probability forward by the actual ReLU
ridge flow. Its proved continuity makes every flow map measurable.
For every original `C¹` ambient test, compact uniform differentiation
gives the derivative of its integral. The original test flux is
continuous in time, so the fundamental theorem of calculus gives
exactly the integrated weak equation in the source.

This constructs the measure-valued solutions required by the original
perturbation lemma. Separation of all weak solutions is a further step.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The probability transported by the actual single-neuron flow.
Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
noncomputable def ridgeLaw (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    Perspective.ProbSphere d := ν.map (fun x => ridgeFlow p u b x t)

/-- The constructed probability has the original flow pushforward as
its underlying measure. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem coe_ridgeLaw (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    (ridgeLaw p u b ν t : Measure (SSphere d)) =
      Measure.map (fun x => ridgeFlow p u b x t) (ν : Measure (SSphere d)) := rfl

/-- The actual transported law starts at the prescribed probability.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem ridgeLaw_zero (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d) :
    ridgeLaw p u b ν 0 = ν := by
  apply Subtype.ext
  change (ridgeLaw p u b ν 0 : Measure (SSphere d)) = (ν : Measure (SSphere d))
  rw [coe_ridgeLaw, show (fun x => ridgeFlow p u b x 0) = id from
    funext fun x => ridgeFlow_zero p u b x, Measure.map_id]

/-- Continuous tests integrate against the actual transported points.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem integral_ridgeLaw (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d)
    (f : EucSpace d → ℝ) (hf : Continuous f) (t : ℝ) :
    (∫ x : SSphere d, f x ∂(ridgeLaw p u b ν t : Measure (SSphere d))) =
      ∫ x : SSphere d, f (ridgeFlow p u b x t) ∂(ν : Measure (SSphere d)) := by
  rw [coe_ridgeLaw]
  exact integral_map (measurable_ridgeFlow_initial p u b t).aemeasurable
    (hf.comp continuous_subtype_val).measurable.aestronglyMeasurable

/-- The original gradient pairing, written using the ambient derivative.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
noncomputable def ridgeTestFlux (φ : EucSpace d → ℝ) (p u : EucSpace d) (b : ℝ)
    (x : EucSpace d) : ℝ := fderiv ℝ φ x (max (b - inner ℝ x u) 0 • proj d x p)

/-- The actual flux of a `C¹` test is continuous, including at the
ReLU kink. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf`. -/
theorem continuous_ridgeTestFlux (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ)
    (p u : EucSpace d) (b : ℝ) : Continuous (ridgeTestFlux φ p u b) := by
  have hfield : Continuous (fun x : EucSpace d => max (b - inner ℝ x u) 0 • proj d x p) := by
    unfold proj
    fun_prop
  exact (hφ.continuous_fderiv_apply (by decide)).comp (continuous_id.prodMk hfield)

/-- Every original `C¹` test integral along the transported law has
the actual network flux as derivative. Compact uniform differentiation
justifies exchanging differentiation and integration.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_ridgeLaw_test (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, φ x ∂(ridgeLaw p u b ν s : Measure (SSphere d)))
      (∫ x : SSphere d, ridgeTestFlux φ p u b x ∂(ridgeLaw p u b ν t : Measure (SSphere d))) t := by
  let F (s : ℝ) (x : SSphere d) := φ (ridgeFlow p u b x s)
  let G (s : ℝ) (x : SSphere d) := ridgeTestFlux φ p u b (ridgeFlow p u b x s)
  have hc : Continuous (fun q : ℝ × SSphere d => (ridgeFlow p u b q.2 q.1 : EucSpace d)) :=
    continuous_subtype_val.comp (continuous_ridgeFlow p u b)
  have hF : Continuous (Function.uncurry F) := hφ.continuous.comp hc
  have hflux := continuous_ridgeTestFlux φ hφ p u b
  have hG : Continuous (Function.uncurry G) := hflux.comp hc
  have hD (s : ℝ) (x : SSphere d) : HasDerivAt (fun r => F r x) (G s x) s :=
    (hφ.differentiable (by decide) (ridgeFlow p u b x s : EucSpace d)).hasFDerivAt
      |>.comp_hasDerivAt s (hasDerivAt_ridgeFlow p u b x s)
  have hd := hasDerivAt_spherical_compact_test (fun _ => ν) F G hF hG hD t 0
    (hasDerivAt_const t _) continuousAt_const
  have hm : (fun s => ∫ x : SSphere d, φ x ∂(ridgeLaw p u b ν s : Measure (SSphere d))) =
      (fun s => ∫ x, F s x ∂(ν : Measure (SSphere d))) :=
    funext fun s => integral_ridgeLaw p u b ν φ hφ.continuous s
  rw [hm, integral_ridgeLaw p u b ν (ridgeTestFlux φ p u b) hflux t]
  simpa only [zero_add] using hd

/-- The original test flux along the transported law is continuous.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_ridgeLaw_flux (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) :
    Continuous (fun s => ∫ x : SSphere d, ridgeTestFlux φ p u b x
      ∂(ridgeLaw p u b ν s : Measure (SSphere d))) := by
  have hflux := continuous_ridgeTestFlux φ hφ p u b
  have heq : (fun s => ∫ x : SSphere d, ridgeTestFlux φ p u b x
      ∂(ridgeLaw p u b ν s : Measure (SSphere d))) =
      (fun s => ∫ x : SSphere d, ridgeTestFlux φ p u b (ridgeFlow p u b x s)
        ∂(ν : Measure (SSphere d))) :=
    funext fun s => integral_ridgeLaw p u b ν _ hflux s
  rw [heq]
  exact continuous_spherical_integral_of_joint_continuous ν _
    (hflux.comp (continuous_subtype_val.comp (continuous_ridgeFlow p u b)))

/-- The constructed law solves the original integrated weak equation
for the actual single-neuron matrices and every real time.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`; existence for §3.2,
`lem: perturbation`, Part 2. -/
theorem cauchyPB_ridgeLaw (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (ν : Perspective.ProbSphere d) :
    cauchyPB d (fun _ => reluRidgeParams i p u b) (ridgeLaw p u b ν) := by
  intro φ hφ t
  have heq : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d (fun _ => reluRidgeParams i p u b) (ridgeLaw p u b ν s) s x)
      ∂(ridgeLaw p u b ν s : Measure (SSphere d))) =
      (fun s => ∫ x : SSphere d, ridgeTestFlux φ p u b x
        ∂(ridgeLaw p u b ν s : Measure (SSphere d))) := by
    funext s
    apply integral_congr_ae
    exact ae_of_all _ fun x => by
      dsimp only
      rw [fullVF_reluRidgeParams, inner_gradient_left]
      rfl
  rw [heq]
  have hi : IntervalIntegrable (fun s => ∫ x : SSphere d, ridgeTestFlux φ p u b x
      ∂(ridgeLaw p u b ν s : Measure (SSphere d))) volume 0 t :=
    (continuous_ridgeLaw_flux p u b ν φ hφ).intervalIntegrable 0 t
  refine ⟨hi, ?_⟩
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun s _ => hasDerivAt_ridgeLaw_test p u b ν φ hφ s) hi
  linarith

/-- The original single-neuron controls admit an actual global weak
solution from every spherical probability law. No existence hypothesis
is needed. Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem exists_cauchyPB_reluRidgeParams (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (ν : Perspective.ProbSphere d) :
    ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧
      cauchyPB d (fun _ => reluRidgeParams i p u b) μ :=
  ⟨ridgeLaw p u b ν, ridgeLaw_zero p u b ν, cauchyPB_ridgeLaw i p u b ν⟩

/-- A nonconstant ambient linear test jointly realizes the smoothness
and continuity premises of the transport and flux theorems. -/
example : ContDiff ℝ 1 (fun x : EucSpace 2 => inner ℝ (basePoint 1 : EucSpace 2) x) ∧
    Continuous (fun x : EucSpace 2 => inner ℝ (basePoint 1 : EucSpace 2) x) :=
  ⟨contDiff_const.inner ℝ contDiff_id, continuous_const.inner continuous_id⟩

end Transformer.Interpolation
