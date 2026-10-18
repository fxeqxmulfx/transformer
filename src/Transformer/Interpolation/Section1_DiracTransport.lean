/-
# Actual weak transport and uniqueness for a spherical Dirac law

ArXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`,
the atomic initial-data case of the source's well-posedness argument.

An actual particle path gives an actual Dirac-valued weak solution by the
chain rule and the fundamental theorem of calculus. For any other weak
solution with the same initial atom, the original squared-distance test
satisfies a linear growth bound and starts at zero. The fencing lemma
forces this nonnegative test to remain zero, and hence fixes the law.

Inverse time rescaling gives the same conclusion at negative times.
Thus every actual global particle path of an autonomous Lipschitz original
field determines exactly one global weak solution from its initial atom.
No regularity of the measure curve or uniqueness premise is added.
-/

import Transformer.Interpolation.Section1_SquaredDistance
import Transformer.Interpolation.Section1_TimeRescaling
import Transformer.Interpolation.SeparationField

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- An actual original particle trajectory yields an actual Dirac-valued
weak solution for every real time, with interval integrability proved.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem cauchyPB_dirac_autonomous (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (Y : ℝ → SSphere d)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t) :
    cauchyPB d θ (fun t => Perspective.diracProb d (Y t)) := by
  intro φ hφ t
  have hcY : Continuous (fun s => (Y s : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  let g (s : ℝ) := fderiv ℝ φ (Y s : EucSpace d) (F (Y s))
  have hg : Continuous g := (hφ.continuous_fderiv_apply (by norm_num)).comp
    (hcY.prodMk (hF.comp hcY))
  have heq : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d θ (Perspective.diracProb d (Y s)) s x)
      ∂(Perspective.diracProb d (Y s) : Measure (SSphere d))) = g := by
    funext s
    change (∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d θ (Perspective.diracProb d (Y s)) s x) ∂Measure.dirac (Y s)) = g s
    rw [integral_dirac, hfield, inner_gradient_left]
  rw [heq]
  have hi : IntervalIntegrable g volume 0 t := hg.intervalIntegrable 0 t
  refine ⟨hi, ?_⟩
  have hd (s : ℝ) : HasDerivAt (fun r => φ (Y r : EucSpace d)) (g s) s :=
    (hφ.differentiable (by norm_num) (Y s : EucSpace d)).hasFDerivAt.comp_hasDerivAt s (hY s)
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt (fun s _ => hd s) hi
  change (∫ x : SSphere d, φ x ∂Measure.dirac (Y t)) =
    (∫ x : SSphere d, φ x ∂Measure.dirac (Y 0)) + ∫ s in (0 : ℝ)..t, g s
  simp only [integral_dirac]
  linarith

/-- Every original weak solution started at the particle's initial atom
agrees with that particle's Dirac law at every nonnegative time.
Source: arXiv:2411.04551v3, §1, the Lipschitz well-posedness argument. -/
theorem cauchyPB_dirac_transport_nonneg (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d))
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (Y : ℝ → SSphere d)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t)
    (h0 : μ 0 = Perspective.diracProb d (Y 0)) (t : ℝ) (ht : 0 ≤ t) :
    μ t = Perspective.diracProb d (Y t) := by
  let e (s : ℝ) := ∫ x : SSphere d, ‖(x : EucSpace d) - (Y s : EucSpace d)‖ ^ 2 ∂(μ s : Measure (SSphere d))
  let e' (s : ℝ) := 2 * ∫ x : SSphere d, inner ℝ ((x : EucSpace d) - (Y s : EucSpace d))
    (F x - F (Y s)) ∂(μ s : Measure (SSphere d))
  have hd (s : ℝ) : HasDerivAt e (e' s) s := hasDerivAt_cauchyPB_squared_distance θ F hF hfield μ hμ Y hY s
  have hinit : e 0 ≤ 0 := by dsimp [e]; rw [h0]; simp [Perspective.diracProb]
  have hbound (s : ℝ) : e' s ≤ (2 * (K : ℝ)) * e s := by
    have h := mul_le_mul_of_nonneg_left
      (integral_inner_difference_le_squared_distance (μ s) F hF K hK (Y s)) (by norm_num : (0 : ℝ) ≤ 2)
    simpa only [mul_assoc] using h
  have he := nonpos_of_deriv_le_mul e e' (2 * (K : ℝ)) t
    (fun s _ => hd s) hinit (fun s _ _ => hbound s) t ⟨ht, le_rfl⟩
  have hn : 0 ≤ e t := integral_nonneg fun x => sq_nonneg _
  exact eq_diracProb_of_integral_squared_distance_eq_zero (μ t) (Y t) (le_antisymm he hn)

/-- The same exact atomic transport holds globally, by reversing the
original equation in time. No weak-solution uniqueness is assumed.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem cauchyPB_eq_dirac_transport (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d))
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (Y : ℝ → SSphere d)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t)
    (h0 : μ 0 = Perspective.diracProb d (Y 0)) :
    μ = fun t => Perspective.diracProb d (Y t) := by
  funext t
  by_cases ht : 0 ≤ t
  · exact cauchyPB_dirac_transport_nonneg θ F hF hfield K hK μ hμ Y hY h0 t ht
  have hfield' (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d) :
      fullVF d (rescaleTimeParams (-1) θ) ρ s x = -F x := by
    rw [fullVF_rescaleTimeParams, hfield, neg_one_smul]
  have hμ' : cauchyPB d (rescaleTimeParams (-1) θ) (fun s => μ (-s)) := by
    simpa only [neg_one_mul] using cauchyPB_rescaleTimeParams (-1) (by norm_num) θ μ hμ
  have hY' (s : ℝ) : HasDerivAt (fun r => (Y (-r) : EucSpace d)) (-F (Y (-s))) s := by
    simpa only [Function.comp_def, neg_smul, one_smul] using
      (hY (-s)).scomp s ((hasDerivAt_id s).neg)
  have h := cauchyPB_dirac_transport_nonneg (rescaleTimeParams (-1) θ) (fun x => -F x) hF.neg hfield'
    K hK.neg (fun s => μ (-s)) hμ' (fun s => Y (-s)) hY'
    (by simpa only [neg_zero] using h0) (-t) (by linarith)
  simpa only [neg_neg] using h

/-- Each actual particle trajectory supplies the unique original global
weak solution with its initial Dirac law.
Source: arXiv:2411.04551v3, §1, atomic well-posedness of `eq: cauchy.pb`. -/
theorem exists_unique_cauchyPB_autonomous_dirac (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d))
    (Y : ℝ → SSphere d)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = Perspective.diracProb d (Y 0) ∧ cauchyPB d θ μ := by
  refine ⟨fun t => Perspective.diracProb d (Y t), ⟨rfl, cauchyPB_dirac_autonomous θ F hF hfield Y hY⟩, ?_⟩
  intro μ hμ
  exact cauchyPB_eq_dirac_transport θ F hF hfield K hK μ hμ.2 Y hY hμ.1

/-- A genuine projection neuron, actual weak law and unit particle path,
and a compact-sphere Lipschitz bound jointly satisfy all dynamical premises.
Their actual common Dirac initial law has zero squared distance. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2)
    (μ : ℝ → Perspective.ProbSphere 2) (Y : ℝ → SSphere 2) (K : ℝ≥0),
    Continuous F ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    LipschitzOnWith K F (Sphere 2) ∧ cauchyPB 2 θ μ ∧
    (∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace 2)) (F (Y t)) t) ∧
    μ 0 = Perspective.diracProb 2 (Y 0) ∧ (0 : ℝ) ≤ 1 ∧
    (∫ x : SSphere 2, ‖(x : EucSpace 2) - (Y 0 : EucSpace 2)‖ ^ 2 ∂(μ 0 : Measure (SSphere 2))) = 0 := by
  let a : EucSpace 2 := basePoint 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) a 0 1
  let F (x : EucSpace 2) := proj 2 x a
  let μ := ridgeLaw a 0 1 (Perspective.diracProb 2 (basePoint 1))
  let Y : ℝ → SSphere 2 := fun _ => basePoint 1
  have hF : ContDiff ℝ 1 F := contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  obtain ⟨K, hK⟩ := Landscape.GlobalFlow.exists_lipschitzOnWith_closedBall hF.locallyLipschitz 1
  have hsphere : Sphere 2 ⊆ Metric.closedBall 0 1 := fun x hx =>
    mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp hx).le
  have hfield (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2) : fullVF 2 θ ρ s x = F x := by
    rw [fullVF_reluRidgeParams]
    simp [F]
  have hinit : μ 0 = Perspective.diracProb 2 (Y 0) := ridgeLaw_zero a 0 1 _
  refine ⟨θ, F, μ, Y, K, hF.continuous, hfield, hK.mono hsphere,
    cauchyPB_ridgeLaw _ _ _ _ _, ?_, hinit, zero_le_one, ?_⟩
  · intro t
    have ha : ‖a‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 1).2
    have he : F (Y t) = 0 := by
      change proj 2 a a = 0
      simpa only [one_smul] using proj_smul_self ha 1
    rw [he]
    exact hasDerivAt_const t _
  · rw [hinit]
    change (∫ x : SSphere 2, ‖(x : EucSpace 2) - (Y 0 : EucSpace 2)‖ ^ 2 ∂Measure.dirac (Y 0)) = 0
    simp

end Transformer.Interpolation
