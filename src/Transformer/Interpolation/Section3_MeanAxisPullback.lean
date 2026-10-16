/-
# Conserved inverse-clock tests for every original mean-attention law

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.2,
`eq: cauchy.pb.reloaded`, Step 1 of `lem: induction.barycenter`.
Compose each original test with the proved smooth ambient extension
of the inverse projection flow at the actual weak-law clock.

The material derivative vanishes on the original sphere. This follows
by differentiating the actual group identity along a characteristic;
no extension identity off the sphere is asserted. The original C1
spacetime weak identity then makes every inverse-test integral constant
for all real times, including zero. No law transport is assumed.
-/

import Transformer.Interpolation.Section3_MeanAxisWeakTests

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A C1 clock gives admissible global C1 inverse-flow tests.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem contDiff_projectionPullback_clock (w : SSphere d) (q : ℝ → ℝ) (hq : ContDiff ℝ 1 q)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) :
    ContDiff ℝ 1 (fun r : ℝ × EucSpace d => φ (projectionExtension (w : EucSpace d) (-q r.1) r.2)) :=
  hφ.comp (((contDiff_projectionExtension (w : EucSpace d) (mem_sphere_zero_iff_norm.mp w.2)).of_le
    (by simp)).comp ((hq.comp contDiff_fst).neg.prodMk contDiff_snd))

/-- The inverse-clock test has zero actual material derivative on the
sphere. Source: arXiv:2411.04551v3, §1 and §3.2, characteristics. -/
theorem fderiv_projectionPullback_clock_zero (w : SSphere d) (q : ℝ → ℝ) (hq : ContDiff ℝ 1 q)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t c : ℝ) (ht : HasDerivAt q c t)
    (x : SSphere d) :
    fderiv ℝ (fun r : ℝ × EucSpace d => φ (projectionExtension (w : EucSpace d) (-q r.1) r.2))
      (t, (x : EucSpace d)) (1, c • proj d (x : EucSpace d) (w : EucSpace d)) = 0 := by
  let A (r : ℝ × EucSpace d) := φ (projectionExtension (w : EucSpace d) (-q r.1) r.2)
  let Y (s : ℝ) := projectionFlow (w : EucSpace d) (q s - q t) (x : EucSpace d)
  have hw : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hY : HasDerivAt Y (c • proj d (x : EucSpace d) (w : EucSpace d)) t := by
    simpa only [Y, sub_self, projectionFlow_zero, Function.comp_def] using
      (hasDerivAt_projectionFlow (w : EucSpace d) (x : EucSpace d) hw hx (q t - q t)).scomp t (ht.sub_const (q t))
  have he : ∀ s, A (s, Y s) = φ (projectionFlow (w : EucSpace d) (-q t) (x : EucSpace d)) := by
    intro s
    dsimp only [A, Y]
    rw [projectionExtension_eq _ _ hw (norm_projectionFlow _ _ hw hx _),
      projectionFlow_add _ _ hw hx]
    congr 2
    ring
  have hC := contDiff_projectionPullback_clock w q hq φ hφ
  have hp : (t, (x : EucSpace d)) = (t, Y t) := by simp only [Y, sub_self, projectionFlow_zero]
  have hd := (hC.differentiable (by decide) (t, (x : EucSpace d))).hasFDerivAt
  rw [hp] at hd
  have hcomp := hd.comp_hasDerivAt t ((hasDerivAt_id t).prodMk hY)
  have hconst : HasDerivAt (fun s => A (s, Y s)) 0 t := by
    have hfun : (fun s => A (s, Y s)) = fun _ => φ (projectionFlow (w : EucSpace d) (-q t) (x : EucSpace d)) := funext he
    rw [hfun]
    exact hasDerivAt_const t _
  have heq := hcomp.unique hconst
  rw [← hp] at heq
  exact heq

/-- Every original mean-attention weak law conserves each inverse-clock
test infinitesimally. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem hasDerivAt_meanAxis_pullback_integral (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d,
      φ (projectionExtension (w : EucSpace d) (-meanAxisWeakClock w κ μ s) (x : EucSpace d))
        ∂(μ s : Measure (SSphere d))) 0 t := by
  let q := meanAxisWeakClock w κ μ
  let A (s : ℝ) (x : EucSpace d) := φ (projectionExtension (w : EucSpace d) (-q s) x)
  have hq := contDiff_meanAxisWeakClock w κ μ hμ
  have hA : ContDiff ℝ 1 (Function.uncurry A) := contDiff_projectionPullback_clock w q hq φ hφ
  have hd := hasDerivAt_meanAxis_cauchyPB_spacetime w κ μ hμ A hA t
  apply hd.congr_deriv
  apply integral_eq_zero_of_ae
  exact ae_of_all _ fun x => by
    dsimp only [Pi.zero_apply]
    rw [fullVF_meanAxisParams]
    exact fderiv_projectionPullback_clock_zero w q hq φ hφ t
      (κ * meanAxisMoment (w : EucSpace d) (μ t)) (hasDerivAt_meanAxisWeakClock w κ μ hμ t) x

/-- Inverse-test integrals agree at every pair of real times along the
original weak law. Source: arXiv:2411.04551v3, §1 and §3.2. -/
theorem meanAxis_pullback_integral_eq (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (a b : ℝ) :
    (∫ x : SSphere d, φ (projectionExtension (w : EucSpace d) (-meanAxisWeakClock w κ μ b) (x : EucSpace d))
      ∂(μ b : Measure (SSphere d))) =
    ∫ x : SSphere d, φ (projectionExtension (w : EucSpace d) (-meanAxisWeakClock w κ μ a) (x : EucSpace d))
      ∂(μ a : Measure (SSphere d)) := by
  have hd := hasDerivAt_meanAxis_pullback_integral w κ μ hμ φ hφ
  exact is_const_of_deriv_eq_zero (fun s => (hd s).differentiableAt)
    (fun s => (hd s).deriv) b a

/-- The conserved inverse-test integral equals the original initial
test integral. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxis_pullback_integral_eq_initial (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    (∫ x : SSphere d, φ (projectionExtension (w : EucSpace d) (-meanAxisWeakClock w κ μ t) (x : EucSpace d))
      ∂(μ t : Measure (SSphere d))) = ∫ x : SSphere d, φ x ∂(μ 0 : Measure (SSphere d)) := by
  rw [meanAxis_pullback_integral_eq w κ μ hμ φ hφ 0 t]
  apply integral_congr_ae
  exact ae_of_all _ fun x => by
    dsimp only
    rw [meanAxisWeakClock_zero, neg_zero, projectionExtension_eq _ _
      (mem_sphere_zero_iff_norm.mp w.2) (mem_sphere_zero_iff_norm.mp x.2), projectionFlow_zero]

/-- The conserved inverse-test integrals are continuous at every real
time. Auxiliary for arXiv:2411.04551v3, §3.2, control-phase endpoints. -/
theorem continuous_meanAxis_pullback_integral (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) :
    Continuous (fun s => ∫ x : SSphere d,
      φ (projectionExtension (w : EucSpace d) (-meanAxisWeakClock w κ μ s) (x : EucSpace d))
        ∂(μ s : Measure (SSphere d))) :=
  continuous_iff_continuousAt.mpr fun s => (hasDerivAt_meanAxis_pullback_integral w κ μ hμ φ hφ s).continuousAt

/-- The actual inverse characteristic recovers the entire original initial
measure. Source: arXiv:2411.04551v3, §1 and §3.2, Step 1. -/
theorem cauchyPB_meanAxis_inverse_map (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) (t : ℝ) :
    Measure.map (sphereProjectionFlow w (-meanAxisWeakClock w κ μ t)) (μ t : Measure (SSphere d)) =
      (μ 0 : Measure (SSphere d)) := by
  have hc : Continuous (sphereProjectionFlow w (-meanAxisWeakClock w κ μ t)) :=
    (continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id)
  apply Perspective.measure_eq_of_integral_mono_eq
  intro k I
  have he := meanAxis_pullback_integral_eq_initial w κ μ hμ (Perspective.mono I)
    ((Perspective.contDiff_mono I ⊤).of_le (by simp)) t
  calc
    _ = ∫ x : SSphere d, Perspective.mono I
        (sphereProjectionFlow w (-meanAxisWeakClock w κ μ t) x : EucSpace d)
          ∂(μ t : Measure (SSphere d)) :=
      integral_map hc.measurable.aemeasurable
        ((Perspective.continuous_mono I).comp continuous_subtype_val).measurable.aestronglyMeasurable
    _ = _ := by
      have hx (x : SSphere d) : projectionExtension (w : EucSpace d)
          (-meanAxisWeakClock w κ μ t) (x : EucSpace d) =
          (sphereProjectionFlow w (-meanAxisWeakClock w κ μ t) x : EucSpace d) :=
        projectionExtension_eq _ _ (mem_sphere_zero_iff_norm.mp w.2) (mem_sphere_zero_iff_norm.mp x.2) _
      simpa only [hx] using he

/-- A nonconstant clock and original linear test, and an actual original
rank-one weak law jointly realize every regularity, clock-derivative and
weak-equation hypothesis. The clock is the actual axis-Dirac clock. -/
example : ∃ (w : SSphere 2) (q : ℝ → ℝ) (φ : EucSpace 2 → ℝ)
    (μ : ℝ → Perspective.ProbSphere 2) (t c : ℝ),
    ContDiff ℝ 1 q ∧ ContDiff ℝ 1 φ ∧ HasDerivAt q c t ∧
      cauchyPB 2 (fun _ => meanAxisParams (w : EucSpace 2) 1) μ := by
  let w : SSphere 2 := basePoint 1
  refine ⟨w, id, (fun x => inner ℝ (w : EucSpace 2) x),
    meanAxisLaw w 1 (Perspective.diracProb 2 w), 1, 1, contDiff_id,
    contDiff_const.inner ℝ contDiff_id, hasDerivAt_id 1, cauchyPB_meanAxisLaw w 1 _⟩

end Transformer.Interpolation
