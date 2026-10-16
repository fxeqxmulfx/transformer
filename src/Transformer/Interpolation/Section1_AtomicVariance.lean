/-
# Atomicity forced by the complete original weak equation

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: cauchy.pb`,
and §1.1, the original point-mass input setting.
With any constant tuple of all five original parameter blocks, the
integrated weak equation differentiates the actual vector barycenter.
Its spherical variance has derivative bounded by its value times a
parameter-dependent constant, by the actual tangent-field flux bound.

An initial unit Dirac has zero variance. The differential inequality
forces zero variance at every nonnegative time; reversing time gives
the same conclusion globally. Thus the actual original
weak law is a Dirac at its own unit barycenter. Neither an atomic
representation nor nonlinear weak uniqueness is assumed.
-/

import Transformer.Interpolation.Section1_FullWeakContinuity
import Transformer.Interpolation.Section1_VarianceFlux
import Transformer.Interpolation.Section1_TimeRescaling
import Transformer.Interpolation.SeparationField
import Mathlib.Analysis.Calculus.Deriv.Prod

open Real MeasureTheory
open scoped NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual vector barycenter has the original mean velocity as
derivative for constant complete controls. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_cauchyPB_constant_barycenter (A : Params d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => A) μ) (t : ℝ) :
    HasDerivAt (fun s => barycenter d (μ s))
      (∫ x : SSphere d, fullVF d (fun _ => A) (μ t) t x ∂(μ t : Measure (SSphere d))) t := by
  let v := ∫ x : SSphere d, fullVF d (fun _ => A) (μ t) t x ∂(μ t : Measure (SSphere d))
  have hi : Integrable (fun x : SSphere d => fullVF d (fun _ => A) (μ t) t x)
      (μ t : Measure (SSphere d)) :=
    (continuous_fullVF_sphere (fun _ => A) (μ t) t).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hc (i : Fin d) : HasDerivAt (fun s => (barycenter d (μ s)).ofLp i) (v.ofLp i) t := by
    have hd := hasDerivAt_cauchyPB_constant_params_test A μ hμ
      (fun x => inner ℝ (EuclideanSpace.single i 1) x) (contDiff_const.inner ℝ contDiff_id) t
    simp only [gradient_inner_linear] at hd
    convert hd using 1
    · funext s
      rw [integral_inner (integrable_spherical_coe (μ s))]
      simp only [EuclideanSpace.inner_single_left, map_one, one_mul]
      rfl
    · rw [integral_inner hi]
      simp only [EuclideanSpace.inner_single_left, map_one, one_mul]
      rfl
  have hpi := hasDerivAt_pi.mpr hc
  let L := (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.toContinuousLinearMap
  exact L.hasFDerivAt.comp_hasDerivAt t hpi

/-- The original spherical variance has the tangent mean-flux
derivative. Source auxiliary: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`, for constant complete parameters. -/
theorem hasDerivAt_cauchyPB_constant_variance (A : Params d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => A) μ) (t : ℝ) :
    HasDerivAt (fun s => 1 - ‖barycenter d (μ s)‖ ^ 2)
      (-2 * inner ℝ (barycenter d (μ t))
        (∫ x : SSphere d, fullVF d (fun _ => A) (μ t) t x ∂(μ t : Measure (SSphere d)))) t := by
  convert ((hasDerivAt_cauchyPB_constant_barycenter A μ hμ t).norm_sq).const_sub 1 using 1
  ring

/-- An actual weak law started at a unit atom has a unit original
barycenter at every nonnegative time. Source auxiliary:
arXiv:2411.04551v3, §1, the atomic case of `eq: cauchy.pb`. -/
theorem norm_barycenter_eq_one_of_constant_dirac (A : Params d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => A) μ)
    (p : SSphere d) (h0 : μ 0 = Perspective.diracProb d p) (t : ℝ) (ht : 0 ≤ t) :
    ‖barycenter d (μ t)‖ = 1 := by
  let K : ℝ≥0 := ⟨‖A.V‖ * (2 * exp (2 * ‖A.B‖) * ‖A.B‖) + ‖A.W‖ * ‖A.U‖ +
    2 * (‖A.V‖ + ‖A.W‖ * (‖A.U‖ + ‖A.b‖)), by positivity⟩
  let e (s : ℝ) := 1 - ‖barycenter d (μ s)‖ ^ 2
  let e' (s : ℝ) := -2 * inner ℝ (barycenter d (μ s))
    (∫ x : SSphere d, fullVF d (fun _ => A) (μ s) s x ∂(μ s : Measure (SSphere d)))
  have hder (s : ℝ) : HasDerivAt e (e' s) s := hasDerivAt_cauchyPB_constant_variance A μ hμ s
  have hbound (s : ℝ) : e' s ≤ (2 * (K : ℝ)) * e s := by
    have hh := neg_inner_mean_flux_le_variance (μ s)
      (fun x : SSphere d => fullVF d (fun _ => A) (μ s) s x) K
      (lipschitzWith_fullVF_sphere (fun _ => A) (μ s) s)
      (fun x => inner_fullVF_eq_zero (fun _ => A) (μ s) s x (mem_sphere_zero_iff_norm.mp x.2))
    dsimp only [e', e]
    linarith
  have hinit : e 0 ≤ 0 := by
    dsimp only [e]
    rw [h0]
    simp [barycenter, Perspective.diracProb, mem_sphere_zero_iff_norm.mp p.2]
  have he := nonpos_of_deriv_le_mul e e' (2 * (K : ℝ)) t
    (fun s _ => hder s) hinit (fun s _ _ => hbound s) t ⟨ht, le_rfl⟩
  have hn : ‖barycenter d (μ t)‖ ≤ 1 := norm_spherical_mean_le_one (μ t)
  have hpos := norm_nonneg (barycenter d (μ t))
  dsimp only [e] at he
  nlinarith

/-- Every original weak law from an atom remains an actual Dirac law
under arbitrary constant complete parameters. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem exists_dirac_of_cauchyPB_constant (A : Params d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => A) μ)
    (p : SSphere d) (h0 : μ 0 = Perspective.diracProb d p) (t : ℝ) (ht : 0 ≤ t) :
    ∃ q : SSphere d, (q : EucSpace d) = barycenter d (μ t) ∧ μ t = Perspective.diracProb d q := by
  have hn := norm_barycenter_eq_one_of_constant_dirac A μ hμ p h0 t ht
  let q : SSphere d := ⟨barycenter d (μ t), mem_sphere_zero_iff_norm.mpr hn⟩
  refine ⟨q, rfl, eq_diracProb_of_integral_squared_distance_eq_zero (μ t) q ?_⟩
  rw [integral_spherical_squared_distance, integral_inner (integrable_spherical_coe (μ t))]
  change 2 - 2 * inner ℝ (barycenter d (μ t)) (barycenter d (μ t)) = 0
  rw [real_inner_self_eq_norm_sq, hn]
  norm_num

/-- Atomicity holds also at negative times by the original exact time
rescaling. Source auxiliary: arXiv:2411.04551v3, §1,
the atomic case of `eq: cauchy.pb`. Every matrix block remains in the
equation; both output matrices change sign under time reversal. -/
theorem exists_dirac_of_cauchyPB_constant_all_time (A : Params d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => A) μ)
    (p : SSphere d) (h0 : μ 0 = Perspective.diracProb d p) (t : ℝ) :
    ∃ q : SSphere d, (q : EucSpace d) = barycenter d (μ t) ∧
      μ t = Perspective.diracProb d q := by
  by_cases ht : 0 ≤ t
  · exact exists_dirac_of_cauchyPB_constant A μ hμ p h0 t ht
  let A' : Params d := (rescaleTimeParams (-1) (fun _ => A)) 0
  have hμ' : cauchyPB d (fun _ => A') (fun s => μ (-s)) := by
    have hA : rescaleTimeParams (-1) (fun _ : ℝ => A) = (fun _ => A') := by
      funext s
      rfl
    rw [← hA]
    simpa only [neg_one_mul] using
      cauchyPB_rescaleTimeParams (-1) (by norm_num) (fun _ => A) μ hμ
  have h0' : μ (-0) = Perspective.diracProb d p := by
    simpa only [neg_zero] using h0
  have h := exists_dirac_of_cauchyPB_constant A' (fun s => μ (-s)) hμ' p h0'
    (-t) (by linarith)
  simpa only [neg_neg] using h

/-- A genuine original constant attention field and its stationary
nonzero unit atom supply all weak-law, initial-law and time hypotheses. -/
example : ∃ (A : Params 2) (μ : ℝ → Perspective.ProbSphere 2) (p : SSphere 2),
    cauchyPB 2 (fun _ => A) μ ∧ μ 0 = Perspective.diracProb 2 p ∧ (0 : ℝ) ≤ 1 := by
  let I := ContinuousLinearMap.id ℝ (EucSpace 2)
  let A : Params 2 := ⟨I, I, 0, 0, 0⟩
  let p : SSphere 2 := basePoint 1
  have hθ : pureAttentionParams 2 1 = (fun _ => A) := by
    funext t
    simp only [pureAttentionParams, one_smul]
    rfl
  refine ⟨A, (fun _ => Perspective.diracProb 2 p), p, ?_, rfl, zero_le_one⟩
  rw [← hθ]
  exact cauchyPB_pureAttention_dirac 1 p

end Transformer.Interpolation
