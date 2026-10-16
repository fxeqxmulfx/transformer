/-
# Continuous tests for every original weak probability curve

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`.
The integrated original equation already gives continuity of C¹ test
integrals. Smooth approximation and the original probability mass one
extend this to every continuous ambient test. No field bound or time
regularity is imposed on the parameters or on the probability curve.

Compactness controls the approximation error uniformly over all laws.
Thus moving jointly continuous tests also have continuous integrals.
The finitely many scalar coordinates give continuity for vector tests.
These results apply across switches of the original controls, and are
independent of characteristic transport and nonlinear weak uniqueness.
-/

import Transformer.Interpolation.Section1_TestModulus
import Transformer.Interpolation.Section1_MovingTests
import Transformer.Interpolation.Section1_CompactTests

open Real MeasureTheory
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A uniform test error bounds the original probability integral error.
Source auxiliary: arXiv:2411.04551v3, §1, continuous weak tests. -/
theorem norm_spherical_test_integral_sub_le (ν : Perspective.ProbSphere d)
    (f g : EucSpace d → ℝ) (hf : Continuous f) (hg : Continuous g) (ε : ℝ)
    (hfg : ∀ x : SSphere d, ‖f x - g x‖ ≤ ε) :
    ‖(∫ x : SSphere d, f x ∂(ν : Measure (SSphere d))) -
      ∫ x : SSphere d, g x ∂(ν : Measure (SSphere d))‖ ≤ ε := by
  have hif : Integrable (fun x : SSphere d => f x) (ν : Measure (SSphere d)) :=
    integrable_spherical_continuous ν _ (hf.comp continuous_subtype_val)
  have hig : Integrable (fun x : SSphere d => g x) (ν : Measure (SSphere d)) :=
    integrable_spherical_continuous ν _ (hg.comp continuous_subtype_val)
  rw [← integral_sub hif hig]
  simpa only [probReal_univ, mul_one] using
    norm_integral_le_of_norm_le_const (μ := (ν : Measure (SSphere d))) (ae_of_all _ hfg)

/-- Every continuous ambient test pairs continuously with every original
weak probability curve. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`; continuity follows from its integrated C¹ tests. -/
theorem continuous_cauchyPB_continuous_test (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (f : EucSpace d → ℝ) (hf : Continuous f) :
    Continuous (fun t => ∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) := by
  apply continuous_iff_continuousAt.mpr
  intro t
  rw [Metric.continuousAt_iff]
  intro ε hε
  obtain ⟨g, hg, happ, -⟩ := hf.exists_contDiff_approx 1
    (continuous_const : Continuous (fun _ : EucSpace d => ε / 4)) (fun _ => by positivity)
  have herr (ν : Perspective.ProbSphere d) :
      ‖(∫ x : SSphere d, f x ∂(ν : Measure (SSphere d))) -
        ∫ x : SSphere d, g x ∂(ν : Measure (SSphere d))‖ ≤ ε / 4 :=
    norm_spherical_test_integral_sub_le ν f g hf hg.continuous (ε / 4)
      (fun x => by simpa only [dist_eq_norm, norm_sub_rev] using (happ (x : EucSpace d)).le)
  obtain ⟨δ, hδ, hlocal⟩ := Metric.continuousAt_iff.mp
    (continuous_cauchyPB_test θ μ hμ g hg).continuousAt (ε / 2) (by positivity)
  refine ⟨δ, hδ, fun s hs => ?_⟩
  have hgsmall := hlocal hs
  rw [dist_eq_norm] at hgsmall ⊢
  have he : (∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))) -
      (∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) =
      ((∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))) -
        (∫ x : SSphere d, g x ∂(μ s : Measure (SSphere d)))) +
      ((∫ x : SSphere d, g x ∂(μ s : Measure (SSphere d))) -
        (∫ x : SSphere d, g x ∂(μ t : Measure (SSphere d)))) +
      ((∫ x : SSphere d, g x ∂(μ t : Measure (SSphere d))) -
        (∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d)))) := by ring
  rw [he]
  have ht := herr (μ t)
  rw [norm_sub_rev] at ht
  calc
    _ ≤ (‖(∫ x : SSphere d, f x ∂(μ s : Measure (SSphere d))) -
          ∫ x : SSphere d, g x ∂(μ s : Measure (SSphere d))‖ +
        ‖(∫ x : SSphere d, g x ∂(μ s : Measure (SSphere d))) -
          ∫ x : SSphere d, g x ∂(μ t : Measure (SSphere d))‖) +
        ‖(∫ x : SSphere d, g x ∂(μ t : Measure (SSphere d))) -
          ∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))‖ :=
      (norm_add_le _ _).trans (add_le_add (norm_add_le _ _) le_rfl)
    _ < ε := by linarith [herr (μ s)]

/-- A jointly continuous ambient test varying in time has a continuous
integral along every original weak law, including at control switches.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_moving_test (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (F : ℝ × EucSpace d → ℝ) (hF : Continuous F) :
    Continuous (fun t => ∫ x : SSphere d, F (t, (x : EucSpace d))
      ∂(μ t : Measure (SSphere d))) := by
  have hc : Continuous (fun q : ℝ × SSphere d => F (q.1, (q.2 : EucSpace d))) :=
    hF.comp (continuous_fst.prodMk (continuous_subtype_val.comp continuous_snd))
  apply continuous_iff_continuousAt.mpr
  intro t
  exact continuousAt_bounded_weak_pairing (fun s => sphericalIntegralCLM (μ s))
    (fun s => norm_sphericalIntegralCLM_le (μ s))
    (boundedTestCurve (fun s (x : SSphere d) => F (s, (x : EucSpace d))) hc) t
    (continuous_boundedTestCurve_of_continuous _ hc).continuousAt
    (continuous_cauchyPB_continuous_test θ μ hμ (fun x => F (t, x))
      (hF.comp (continuous_const.prodMk continuous_id))).continuousAt

/-- Every continuous finite-dimensional ambient vector test also pairs
continuously with the original weak curve. Source auxiliary:
arXiv:2411.04551v3, §1, the moment form of `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_vector_test {m : ℕ} (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (f : EucSpace d → EucSpace m) (hf : Continuous f) :
    Continuous (fun t => ∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) := by
  let M (t : ℝ) := ∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))
  have hc (i : Fin m) : Continuous (fun t => (M t).ofLp i) := by
    have heq : (fun t => (M t).ofLp i) = (fun t => ∫ x : SSphere d,
        inner ℝ (EuclideanSpace.single i 1) (f x) ∂(μ t : Measure (SSphere d))) := by
      funext t
      have hi : Integrable (fun x : SSphere d => f x) (μ t : Measure (SSphere d)) :=
        (hf.comp continuous_subtype_val).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
      rw [integral_inner hi]
      simp only [EuclideanSpace.inner_single_left, map_one, one_mul]
      rfl
    rw [heq]
    exact continuous_cauchyPB_continuous_test θ μ hμ _ (continuous_const.inner hf)
  exact (PiLp.continuous_toLp 2 (fun _ : Fin m => ℝ)).comp (continuous_pi_iff.mpr hc)

/-- Squared distance to any continuous ambient path is a continuous
original weak test. No particle equation is required for this continuity.
Source auxiliary: arXiv:2411.04551v3, §1, the atomic-data estimates
for `eq: cauchy.pb`. -/
theorem continuous_cauchyPB_squared_distance_test (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (Y : ℝ → EucSpace d) (hY : Continuous Y) :
    Continuous (fun t => ∫ x : SSphere d, ‖(x : EucSpace d) - Y t‖ ^ 2
      ∂(μ t : Measure (SSphere d))) := by
  exact continuous_cauchyPB_moving_test θ μ hμ
    (fun q => ‖q.2 - Y q.1‖ ^ 2)
    ((continuous_snd.sub (hY.comp continuous_fst)).norm.pow 2)

/-- A nonzero actual weak law, nonconstant scalar tests at uniform
distance one, a test moving in time and space, and the original coordinate
vector test jointly realize every hypothesis of the continuity results. -/
example : ∃ (θ : TimeParams 2) (μ : ℝ → Perspective.ProbSphere 2)
    (f g : EucSpace 2 → ℝ) (F : ℝ × EucSpace 2 → ℝ) (v : EucSpace 2 → EucSpace 2)
    (Y : ℝ → EucSpace 2),
    cauchyPB 2 θ μ ∧ Continuous f ∧ Continuous g ∧
    (∀ x : SSphere 2, ‖f x - g x‖ ≤ 1) ∧ Continuous F ∧ Continuous v ∧ Continuous Y := by
  let p : EucSpace 2 := basePoint 1
  let f (x : EucSpace 2) := inner ℝ p x
  let g (x : EucSpace 2) := f x + 1
  let F (q : ℝ × EucSpace 2) := sin q.1 * f q.2
  have hf : Continuous f := continuous_const.inner continuous_id
  have hg : Continuous g := hf.add continuous_const
  have hF : Continuous F := (continuous_sin.comp continuous_fst).mul (hf.comp continuous_snd)
  refine ⟨pureAttentionParams 2 1, (fun _ => Perspective.diracProb 2 (basePoint 1)),
    f, g, F, id, (fun t => sin t • p), cauchyPB_pureAttention_dirac 1 (basePoint 1),
    hf, hg, ?_, hF, continuous_id, continuous_sin.smul continuous_const⟩
  intro x
  dsimp only [g]
  have he : f x - (f x + 1) = (-1 : ℝ) := by ring
  rw [he]
  norm_num

end Transformer.Interpolation
