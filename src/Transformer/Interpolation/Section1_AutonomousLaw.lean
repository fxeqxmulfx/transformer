/-
# Actual weak laws from arbitrary spherical characteristic flows

ArXiv:2411.04551v3, §1, `eq: ips`, `eq: vf`, `eq: cauchy.pb`,
and the Lipschitz well-posedness argument.

Push the original initial probability along an actual characteristic flow.
Joint continuity makes every time slice measurable. The compact moving-test
rule differentiates each actual test integral, and the continuous original
flux gives the integrated weak equation by the fundamental theorem of calculus.

For a continuous field Lipschitz on the sphere, all required flow regularity
follows from the proved particle equation. Combined with the characteristic
identity for every weak solution, this gives existence and uniqueness from
any spherical probability. Several ReLU activations need no new weak-law proof.
-/

import Transformer.Interpolation.Section1_SphericalFlowBounds
import Transformer.Interpolation.Section1_AutonomousTransport

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original probability transported by the actual spherical flow.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
noncomputable def sphericalFlowLaw (Φ : SSphere d → ℝ → SSphere d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    Perspective.ProbSphere d := ν.map (fun x => Φ x t)

/-- The underlying law is the actual characteristic pushforward.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem coe_sphericalFlowLaw (Φ : SSphere d → ℝ → SSphere d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    (sphericalFlowLaw Φ ν t : Measure (SSphere d)) = Measure.map (fun x => Φ x t) (ν : Measure (SSphere d)) := rfl

/-- A flow starting at the identity retains the original initial law.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem sphericalFlowLaw_zero (Φ : SSphere d → ℝ → SSphere d) (hΦ₀ : ∀ x, Φ x 0 = x) (ν : Perspective.ProbSphere d) :
    sphericalFlowLaw Φ ν 0 = ν := by
  apply Subtype.ext
  change (sphericalFlowLaw Φ ν 0 : Measure (SSphere d)) = (ν : Measure (SSphere d))
  rw [coe_sphericalFlowLaw, show (fun x => Φ x 0) = id from funext hΦ₀, Measure.map_id]

/-- Continuous tests integrate against the actual transported points.
Source: arXiv:2411.04551v3, §1, the characteristic formula. -/
theorem integral_sphericalFlowLaw (Φ : SSphere d → ℝ → SSphere d)
    (hΦc : Continuous (fun q : ℝ × SSphere d => Φ q.2 q.1)) (ν : Perspective.ProbSphere d)
    (f : EucSpace d → ℝ) (hf : Continuous f) (t : ℝ) :
    (∫ x : SSphere d, f x ∂(sphericalFlowLaw Φ ν t : Measure (SSphere d))) =
      ∫ x : SSphere d, f (Φ x t) ∂(ν : Measure (SSphere d)) := by
  rw [coe_sphericalFlowLaw]
  exact integral_map ((hΦc.comp (continuous_const.prodMk continuous_id)).measurable.aemeasurable)
    ((hf.comp continuous_subtype_val).measurable.aestronglyMeasurable)

/-- The actual ambient derivative pairing of the original weak test.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
noncomputable def sphericalTestFlux (φ : EucSpace d → ℝ) (F : EucSpace d → EucSpace d) (x : EucSpace d) : ℝ :=
  fderiv ℝ φ x (F x)

/-- A continuous original field gives continuous actual test flux.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_sphericalTestFlux (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ)
    (F : EucSpace d → EucSpace d) (hF : Continuous F) : Continuous (sphericalTestFlux φ F) :=
  (hφ.continuous_fderiv_apply (by norm_num)).comp (continuous_id.prodMk hF)

/-- Uniform compact differentiation gives the original weak derivative
of the actual transported law. Source: arXiv:2411.04551v3,
§1, `eq: ips` and `eq: cauchy.pb`. -/
theorem hasDerivAt_sphericalFlowLaw_test (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (Φ : SSphere d → ℝ → SSphere d) (hΦc : Continuous (fun q : ℝ × SSphere d => Φ q.2 q.1))
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F (Φ x t)) t)
    (ν : Perspective.ProbSphere d) (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, φ x ∂(sphericalFlowLaw Φ ν s : Measure (SSphere d)))
      (∫ x : SSphere d, sphericalTestFlux φ F x ∂(sphericalFlowLaw Φ ν t : Measure (SSphere d))) t := by
  let A (s : ℝ) (x : SSphere d) := φ (Φ x s)
  let G (s : ℝ) (x : SSphere d) := sphericalTestFlux φ F (Φ x s)
  have hc : Continuous (fun q : ℝ × SSphere d => (Φ q.2 q.1 : EucSpace d)) := continuous_subtype_val.comp hΦc
  have hA : Continuous (Function.uncurry A) := hφ.continuous.comp hc
  have hflux := continuous_sphericalTestFlux φ hφ F hF
  have hG : Continuous (Function.uncurry G) := hflux.comp hc
  have hD (s : ℝ) (x : SSphere d) : HasDerivAt (fun r => A r x) (G s x) s :=
    (hφ.differentiable (by norm_num) (Φ x s : EucSpace d)).hasFDerivAt.comp_hasDerivAt s (hΦ x s)
  have hd := hasDerivAt_spherical_compact_test (fun _ => ν) A G hA hG hD t 0
    (hasDerivAt_const t _) continuousAt_const
  have hm : (fun s => ∫ x : SSphere d, φ x ∂(sphericalFlowLaw Φ ν s : Measure (SSphere d))) =
      (fun s => ∫ x, A s x ∂(ν : Measure (SSphere d))) :=
    funext fun s => integral_sphericalFlowLaw Φ hΦc ν φ hφ.continuous s
  rw [hm, integral_sphericalFlowLaw Φ hΦc ν (sphericalTestFlux φ F) hflux t]
  simpa only [zero_add] using hd

/-- The actual transported test flux is continuous in time.
Source: arXiv:2411.04551v3, §1, the time integral in `eq: cauchy.pb`. -/
theorem continuous_sphericalFlowLaw_flux (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (Φ : SSphere d → ℝ → SSphere d) (hΦc : Continuous (fun q : ℝ × SSphere d => Φ q.2 q.1))
    (ν : Perspective.ProbSphere d) (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) :
    Continuous (fun s => ∫ x : SSphere d, sphericalTestFlux φ F x ∂(sphericalFlowLaw Φ ν s : Measure (SSphere d))) := by
  have hf := continuous_sphericalTestFlux φ hφ F hF
  have he : (fun s => ∫ x : SSphere d, sphericalTestFlux φ F x ∂(sphericalFlowLaw Φ ν s : Measure (SSphere d))) =
      (fun s => ∫ x : SSphere d, sphericalTestFlux φ F (Φ x s) ∂(ν : Measure (SSphere d))) :=
    funext fun s => integral_sphericalFlowLaw Φ hΦc ν (sphericalTestFlux φ F) hf s
  rw [he]
  exact continuous_spherical_integral_of_joint_continuous ν _
    (hf.comp (continuous_subtype_val.comp hΦc))

/-- The actual characteristic pushforward solves the original integrated
weak equation for every real time. Source: arXiv:2411.04551v3,
§1, `eq: ips` and `eq: cauchy.pb`. -/
theorem cauchyPB_sphericalFlowLaw (θ : TimeParams d) (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (Φ : SSphere d → ℝ → SSphere d) (hΦc : Continuous (fun q : ℝ × SSphere d => Φ q.2 q.1))
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F (Φ x t)) t) (ν : Perspective.ProbSphere d) :
    cauchyPB d θ (sphericalFlowLaw Φ ν) := by
  intro φ hφ t
  have he : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d θ (sphericalFlowLaw Φ ν s) s x) ∂(sphericalFlowLaw Φ ν s : Measure (SSphere d))) =
      (fun s => ∫ x : SSphere d, sphericalTestFlux φ F x ∂(sphericalFlowLaw Φ ν s : Measure (SSphere d))) := by
    funext s
    apply integral_congr_ae
    exact ae_of_all _ fun x => by dsimp only [sphericalTestFlux]; rw [hfield, inner_gradient_left]
  rw [he]
  have hi := (continuous_sphericalFlowLaw_flux F hF Φ hΦc ν φ hφ).intervalIntegrable (μ := volume) 0 t
  refine ⟨hi, ?_⟩
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun s _ => hasDerivAt_sphericalFlowLaw_test F hF Φ hΦc hΦ ν φ hφ s) hi
  linarith

/-- An actual global particle flow supplies exactly one original weak
solution from each arbitrary initial probability. Its regularity is proved
from the original continuous Lipschitz field.
Source: arXiv:2411.04551v3, §1, well-posedness of `eq: cauchy.pb`. -/
theorem exists_unique_cauchyPB_autonomous (θ : TimeParams d) (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F (Φ x t)) t)
    (hΦ₀ : ∀ x, Φ x 0 = x) (ν : Perspective.ProbSphere d) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d θ μ := by
  have hΦc := continuous_sphericalFlow F hF K hK Φ hΦ hΦ₀
  have hμ := cauchyPB_sphericalFlowLaw θ F hF hfield Φ hΦc hΦ ν
  have h0 := sphericalFlowLaw_zero Φ hΦ₀ ν
  refine ⟨sphericalFlowLaw Φ ν, ⟨h0, hμ⟩, fun μ hμ' => ?_⟩
  exact cauchyPB_autonomous_unique θ F hF hfield K hK Φ hΦ hΦ₀
    (exists_lipschitzWith_sphericalFlow_initial F K hK Φ hΦ hΦ₀) μ _ hμ'.2 hμ (hμ'.1.trans h0.symm)

/-- A genuine projection neuron, its actual complete flow, a two-atom
initial law and a nonconstant linear test jointly realize all original
field, trajectory, continuity, initial-value and test hypotheses. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2) (K : ℝ≥0)
    (Φ : SSphere 2 → ℝ → SSphere 2) (ν : Perspective.ProbSphere 2) (φ : EucSpace 2 → ℝ),
    Continuous F ∧ LipschitzOnWith K F (Sphere 2) ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    Continuous (fun q : ℝ × SSphere 2 => Φ q.2 q.1) ∧
    (∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace 2)) (F (Φ x t)) t) ∧
    (∀ x, Φ x 0 = x) ∧ ContDiff ℝ 1 φ ∧ Continuous φ ∧ sphericalFlowLaw Φ ν 0 = ν := by
  let a : EucSpace 2 := basePoint 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) a 0 1
  let F (x : EucSpace 2) := proj 2 x a
  let Φ := ridgeFlow a 0 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  have hF : ContDiff ℝ 1 F := contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  obtain ⟨K, hK⟩ := GlobalFlow.exists_lipschitzOnWith_closedBall hF.locallyLipschitz 1
  have hsphere : Sphere 2 ⊆ Metric.closedBall 0 1 := fun x hx =>
    mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp hx).le
  refine ⟨θ, F, K, Φ, halfDirac 2 z (antipode 2 z), fun x => inner ℝ a x,
    hF.continuous, hK.mono hsphere, ?_, continuous_ridgeFlow _ _ _, ?_, ridgeFlow_zero _ _ _,
    contDiff_const.inner ℝ contDiff_id, continuous_const.inner continuous_id,
    sphericalFlowLaw_zero _ (ridgeFlow_zero _ _ _) _⟩
  · intro ρ s x
    rw [fullVF_reluRidgeParams]
    simp [F]
  · intro x t
    simpa [F, Φ] using hasDerivAt_ridgeFlow a 0 1 x t

end Transformer.Interpolation
