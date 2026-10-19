module

public import Transformer.Interpolation.Section1_AutonomousWeakTests

/-!
# Smooth spacetime comparisons for the original weak equation

ArXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`,
the Lipschitz well-posedness argument, and the characteristic tests
used in §3.1, `lem: first.quadrant`.

A continuous original autonomous field gives continuous weak fluxes.
The compact moving-test rule therefore differentiates every ambient
`C¹` spacetime test along every actual weak solution. Its derivative
is the integral of the spacetime differential in direction `(1,F)`.
A nonpositive material derivative gives an actual comparison principle.

Particle paths are `C¹` in time even when the ReLU field has a spatial
kink. Consequently smooth functions of finitely many particle paths
are legitimate comparison tests without assuming spatial derivatives
of the original flow. This is the input for smooth transport barriers.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- An actual particle path of a continuous field has its original `C¹`
time regularity. No spatial derivative of the field is required.
Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem contDiff_spherical_particle (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (Y : ℝ → SSphere d)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t) :
    ContDiff ℝ 1 (fun s => (Y s : EucSpace d)) := by
  have hc : Continuous (fun s => (Y s : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  refine contDiff_one_iff_deriv.mpr ⟨fun s => (hY s).differentiableAt, ?_⟩
  have he : deriv (fun s => (Y s : EucSpace d)) = fun s => F (Y s) :=
    funext fun s => (hY s).deriv
  rw [he]
  exact hF.comp hc

/-- The actual weak equation differentiates every `C¹` spacetime test.
Its derivative is the spacetime differential in direction `(1,F(x))`.
The compact moving-test rule supplies uniform time differentiation, while
continuity of autonomous test fluxes comes from the original equation.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_cauchyPB_autonomous_spacetime_test (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (A : ℝ → EucSpace d → ℝ) (hA : ContDiff ℝ 1 (Function.uncurry A)) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, A s x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d)) (1, F x)
        ∂(μ t : Measure (SSphere d))) t := by
  let G (s : ℝ) (x : EucSpace d) := fderiv ℝ (Function.uncurry A) (s, x) (1, 0)
  have hGc : Continuous (Function.uncurry G) :=
    (hA.continuous_fderiv_apply (by norm_num)).comp (continuous_id.prodMk continuous_const)
  have hD (s : ℝ) (x : SSphere d) : HasDerivAt (fun r => A r x) (G s x) s := by
    dsimp only [G]
    simpa only [Function.comp_def, Function.uncurry_apply_pair, id_eq] using
      (hA.differentiable (by norm_num) (s, (x : EucSpace d))).hasFDerivAt.comp_hasDerivAt s
      ((hasDerivAt_id s).prodMk (hasDerivAt_const s (x : EucSpace d)))
  have hfixed := hasDerivAt_cauchyPB_autonomous_test θ F hF hfield μ hμ
    (A t) (hA.comp (contDiff_const.prodMk contDiff_id)) t
  have hx (x : SSphere d) : fderiv ℝ (A t) (x : EucSpace d) (F x) =
      fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d)) (0, F x) := by
    have hp : HasFDerivAt (fun y : EucSpace d => (t, y))
        ((0 : EucSpace d →L[ℝ] ℝ).prod (ContinuousLinearMap.id ℝ (EucSpace d))) (x : EucSpace d) :=
      (hasFDerivAt_const t (x : EucSpace d)).prodMk (hasFDerivAt_id (x : EucSpace d))
    rw [show fderiv ℝ (A t) (x : EucSpace d) =
      (fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d))).comp
        ((0 : EucSpace d →L[ℝ] ℝ).prod (ContinuousLinearMap.id ℝ (EucSpace d))) from
      ((hA.differentiable (by norm_num) _).hasFDerivAt.comp (x : EucSpace d) hp).fderiv]
    rfl
  have h := hasDerivAt_spherical_compact_test μ (fun s x => A s x) (fun s x => G s x)
    (hA.continuous.comp (continuous_fst.prodMk (continuous_subtype_val.comp continuous_snd)))
    (hGc.comp (continuous_fst.prodMk (continuous_subtype_val.comp continuous_snd))) hD t _ hfixed
    (continuous_cauchyPB_autonomous_test θ F hF hfield μ hμ (G t)
      (hGc.comp (continuous_const.prodMk continuous_id))).continuousAt
  have hc₀ : Continuous (fun x : SSphere d =>
      fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d)) (0, F x)) :=
    (hA.continuous_fderiv_apply (by norm_num)).comp
      ((continuous_const.prodMk continuous_subtype_val).prodMk
        (continuous_const.prodMk (hF.comp continuous_subtype_val)))
  have hc₁ : Continuous (fun x : SSphere d => G t x) :=
    hGc.comp (continuous_const.prodMk continuous_subtype_val)
  convert h using 1
  simp_rw [inner_gradient_left, hx]
  rw [← integral_add (integrable_spherical_continuous (μ t) _ hc₀)
    (integrable_spherical_continuous (μ t) _ hc₁)]
  apply integral_congr_ae
  exact ae_of_all _ fun x => by
    dsimp only [G]
    rw [← map_add]
    simp


/-- Every smooth test whose material derivative is nonpositive has a
nonincreasing integral along every actual original weak solution.
Source: arXiv:2411.04551v3, §1, the Lipschitz well-posedness argument
for `eq: cauchy.pb`; the comparison used for backward transport tests. -/
theorem cauchyPB_spacetime_test_le (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (A : ℝ → EucSpace d → ℝ) (hA : ContDiff ℝ 1 (Function.uncurry A))
    (hsub : ∀ (s : ℝ) (x : SSphere d),
      fderiv ℝ (Function.uncurry A) (s, (x : EucSpace d)) (1, F x) ≤ 0)
    (s t : ℝ) (hst : s ≤ t) :
    (∫ x : SSphere d, A t x ∂(μ t : Measure (SSphere d))) ≤
      ∫ x : SSphere d, A s x ∂(μ s : Measure (SSphere d)) := by
  have hd := hasDerivAt_cauchyPB_autonomous_spacetime_test θ F hF hfield μ hμ A hA
  exact antitone_of_deriv_nonpos (fun r => (hd r).differentiableAt)
    (fun r => by rw [(hd r).deriv]; exact integral_nonpos (hsub r)) hst

/-- A genuine projection neuron, its transported transverse atom, the
actual particle path, and the spacetime test `-t+⟨a,x⟩` jointly realize all
premises. The time order is nontrivial and both test variables are used. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2)
    (μ : ℝ → Perspective.ProbSphere 2) (A : ℝ → EucSpace 2 → ℝ) (Y : ℝ → SSphere 2),
    Continuous F ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    cauchyPB 2 θ μ ∧ ContDiff ℝ 1 (Function.uncurry A) ∧
    (∀ (s : ℝ) (x : SSphere 2), fderiv ℝ (Function.uncurry A) (s, (x : EucSpace 2)) (1, F x) ≤ 0) ∧
    (∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace 2)) (F (Y t)) t) ∧ (0 : ℝ) ≤ 1 := by
  let a : EucSpace 2 := basePoint 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) a 0 1
  let F (x : EucSpace 2) := proj 2 x a
  let μ := ridgeLaw a 0 1 (Perspective.diracProb 2 ⟨EuclideanSpace.single 1 1, by simp⟩)
  let Y := ridgeFlow a 0 1 (⟨EuclideanSpace.single 1 1, by simp⟩ : SSphere 2)
  let A (s : ℝ) (x : EucSpace 2) := -s + inner ℝ a x
  have ha : ‖a‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 1).2
  refine ⟨θ, F, μ, A, Y, ?_, ?_, cauchyPB_ridgeLaw _ _ _ _ _, ?_, ?_, ?_, zero_le_one⟩
  · unfold F proj
    fun_prop
  · intro ρ s x
    rw [fullVF_reluRidgeParams]
    simp [F]
  · exact contDiff_fst.neg.add (contDiff_const.inner ℝ contDiff_snd)
  · intro s x
    have hd : HasFDerivAt (Function.uncurry A)
        (-(ContinuousLinearMap.fst ℝ ℝ (EucSpace 2)) +
          (innerSL ℝ a).comp (ContinuousLinearMap.snd ℝ ℝ (EucSpace 2))) (s, (x : EucSpace 2)) :=
      hasFDerivAt_fst.neg.add ((innerSL ℝ a).hasFDerivAt.comp _ hasFDerivAt_snd)
    rw [hd.fderiv]
    change -1 + inner ℝ a (F x) ≤ 0
    have hb : inner ℝ a (F x) ≤ 1 := calc
      _ ≤ |inner ℝ a (F x)| := le_abs_self _
      _ ≤ ‖a‖ * ‖F x‖ := abs_real_inner_le_norm _ _
      _ ≤ 1 := by
        rw [ha, one_mul]
        exact (norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) a).trans ha.le
    linarith
  · intro t
    simpa [F, Y] using hasDerivAt_ridgeFlow a 0 1 (⟨EuclideanSpace.single 1 1, by simp⟩ : SSphere 2) t

end Transformer.Interpolation
