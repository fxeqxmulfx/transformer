module

public import Transformer.Interpolation.Section1_SpacetimeTests

/-!
# C1 spacetime tests for time-dependent weak transport

ArXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`.
The compact moving-test rule extends the original fixed-test weak
identity to globally C1 spacetime tests. Time dependence remains
explicit in the actual field. The inputs are continuity of original
continuous-test integrals and the differential weak identity for
original C1 tests; neither a transport formula nor atomicity is assumed.

Only continuity of each spatial field slice is needed in the weak
identity. Joint continuity of the field gives C1 regularity of actual
particle paths. These statements also cover clocks with only C1
regularity, as required for rank-one nonlinear mean attention in §3.2.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual differentiable particle paths are continuous on the sphere.
No regularity of the field beyond the displayed particle equation is
assumed. Auxiliary for arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem continuous_timeDependent_spherical_particle (F : ℝ → EucSpace d → EucSpace d)
    (Y : ℝ → SSphere d)
    (hY : ∀ t, HasDerivAt (fun s => (Y s : EucSpace d)) (F t (Y t)) t) : Continuous Y := by
  apply continuous_induced_rng.mpr
  exact continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt

/-- The original particle equation and joint continuity of its actual
velocity give C1 time regularity. No spatial derivative of the field
is required. Auxiliary for arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem contDiff_timeDependent_spherical_particle (F : ℝ → EucSpace d → EucSpace d)
    (hF : Continuous (Function.uncurry F)) (Y : ℝ → SSphere d)
    (hY : ∀ t, HasDerivAt (fun s => (Y s : EucSpace d)) (F t (Y t)) t) :
    ContDiff ℝ 1 (fun s => (Y s : EucSpace d)) := by
  have hc : Continuous (fun s => (Y s : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  refine contDiff_one_iff_deriv.mpr ⟨fun s => (hY s).differentiableAt, ?_⟩
  have he : deriv (fun s => (Y s : EucSpace d)) = fun s => F s (Y s) :=
    funext fun s => (hY s).deriv
  rw [he]
  exact hF.comp (continuous_id.prodMk hc)

/-- Original fixed-test weak derivatives and weak continuity imply the
C1 spacetime identity, with material direction `(1,F(t,x))`. Compact
uniform differentiation handles the time-dependent test; continuity
of each spatial field slice makes its flux integrable. Source:
arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_timeDependent_spacetime_test
    (F : ℝ → EucSpace d → EucSpace d) (hF : ∀ t, Continuous (F t))
    (μ : ℝ → Perspective.ProbSphere d)
    (hcont : ∀ (f : EucSpace d → ℝ), Continuous f →
      Continuous (fun s => ∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))))
    (hweak : ∀ (φ : EucSpace d → ℝ), ContDiff ℝ 1 φ → ∀ t,
      HasDerivAt (fun s => ∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d)))
        (∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d)) (F t x)
          ∂(μ t : Measure (SSphere d))) t)
    (A : ℝ → EucSpace d → ℝ) (hA : ContDiff ℝ 1 (Function.uncurry A)) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, A s x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d))
        (1, F t x)
          ∂(μ t : Measure (SSphere d))) t := by
  let G (s : ℝ) (x : EucSpace d) := fderiv ℝ (Function.uncurry A) (s, x) (1, 0)
  have hGc : Continuous (Function.uncurry G) :=
    (hA.continuous_fderiv_apply (by norm_num)).comp (continuous_id.prodMk continuous_const)
  have hD (s : ℝ) (x : SSphere d) : HasDerivAt (fun r => A r x) (G s x) s := by
    dsimp only [G]
    simpa only [Function.comp_def, Function.uncurry_apply_pair, id_eq] using
      (hA.differentiable (by norm_num) (s, (x : EucSpace d))).hasFDerivAt.comp_hasDerivAt s
      ((hasDerivAt_id s).prodMk (hasDerivAt_const s (x : EucSpace d)))
  have hfixed := hweak
    (A t) (hA.comp (contDiff_const.prodMk contDiff_id)) t
  have hx (x : SSphere d) : fderiv ℝ (A t) (x : EucSpace d) (F t x) =
      fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d)) (0, F t x) := by
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
    (hcont (G t)
      (hGc.comp (continuous_const.prodMk continuous_id))).continuousAt
  have hc₀ : Continuous (fun x : SSphere d =>
      fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d)) (0, F t x)) :=
    (hA.continuous_fderiv_apply (by norm_num)).comp
      ((continuous_const.prodMk continuous_subtype_val).prodMk
        (continuous_const.prodMk ((hF t).comp continuous_subtype_val)))
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

/-- A time-dependent projection field, its actual stationary nonzero
Dirac weak law and actual stationary particle jointly realize the
weak-continuity, all-test derivative and particle hypotheses. The
smooth test `sin(t)+⟨p,x⟩` depends on both spacetime variables. -/
example : ∃ (F : ℝ → EucSpace 2 → EucSpace 2) (μ : ℝ → Perspective.ProbSphere 2)
    (Y : ℝ → SSphere 2) (A : ℝ → EucSpace 2 → ℝ),
    Continuous (Function.uncurry F) ∧
    (∀ (f : EucSpace 2 → ℝ), Continuous f →
      Continuous (fun s => ∫ x : SSphere 2, f x ∂(μ s : Measure (SSphere 2)))) ∧
    (∀ (φ : EucSpace 2 → ℝ), ContDiff ℝ 1 φ → ∀ t,
      HasDerivAt (fun s => ∫ x : SSphere 2, φ x ∂(μ s : Measure (SSphere 2)))
        (∫ x : SSphere 2, inner ℝ (gradient φ (x : EucSpace 2)) (F t x)
          ∂(μ t : Measure (SSphere 2))) t) ∧
    (∀ t, HasDerivAt (fun s => (Y s : EucSpace 2)) (F t (Y t)) t) ∧
    ContDiff ℝ 1 (Function.uncurry A) := by
  let p : SSphere 2 := basePoint 1
  let F (t : ℝ) (x : EucSpace 2) := cos t • proj 2 x (p : EucSpace 2)
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  let A (s : ℝ) (x : EucSpace 2) := sin s + inner ℝ (p : EucSpace 2) x
  have hFp (t : ℝ) : F t p = 0 := by
    have hh := proj_smul_self (mem_sphere_zero_iff_norm.mp p.2) (1 : ℝ)
    change cos t • proj 2 (p : EucSpace 2) (p : EucSpace 2) = 0
    simpa only [one_smul, smul_zero] using congrArg (fun v => cos t • v) hh
  refine ⟨F, μ, fun _ => p, A, ?_, ?_, ?_, ?_, ?_⟩
  · exact (continuous_cos.comp continuous_fst).smul
      (continuous_const.sub ((continuous_snd.inner (𝕜 := ℝ) continuous_const).smul continuous_snd))
  · intro f _
    change Continuous (fun _ : ℝ => ∫ x : SSphere 2, f x ∂Measure.dirac p)
    exact continuous_const
  · intro φ _ t
    change HasDerivAt (fun _ : ℝ => ∫ x : SSphere 2, φ x ∂Measure.dirac p)
      (∫ x : SSphere 2, inner ℝ (gradient φ (x : EucSpace 2)) (F t x) ∂Measure.dirac p) t
    simp_rw [integral_dirac]
    rw [hFp, inner_zero_right]
    exact hasDerivAt_const t _
  · intro t
    rw [hFp]
    exact hasDerivAt_const t (p : EucSpace 2)
  · exact (contDiff_sin.comp contDiff_fst).add (contDiff_const.inner ℝ contDiff_snd)

end Transformer.Interpolation
