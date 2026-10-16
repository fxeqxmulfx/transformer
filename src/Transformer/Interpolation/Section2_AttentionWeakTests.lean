/-
# Moving tests for the original attention equation

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §2, `prop: targets.atoms`.
The actual weak equation differentiates C1 spacetime tests. Its material
derivative therefore controls test integrals along the evolving law.
Applying this to finite smooth transport cones uses the current field
and the actual particle velocities at that time.
-/

import Transformer.Interpolation.Section2_AttentionRegularity
import Transformer.Interpolation.Section1_FiniteTransport

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual attention particles are C1, by joint continuity of their
velocity. Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem contDiff_attention_particle (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (Y : ℝ → SSphere d)
    (hY : ∀ t, HasDerivAt (fun s => (Y s : EucSpace d))
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t (Y t)) t) :
    ContDiff ℝ 1 (fun s => (Y s : EucSpace d)) := by
  have hc : Continuous (fun s => (Y s : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  have hYs : Continuous Y := continuous_induced_rng.mpr hc
  refine contDiff_one_iff_deriv.mpr ⟨fun s => (hY s).differentiableAt, ?_⟩
  have he : deriv (fun s => (Y s : EucSpace d)) = fun s =>
      fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ s) s (Y s) :=
    funext fun s => (hY s).deriv
  rw [he]
  have hpair : Continuous (fun s : ℝ => (s, Y s)) := continuous_id.prodMk hYs
  simpa only [Function.comp_def] using (continuous_fullVF_attention_joint_sphere B μ hμ).comp hpair

/-- Moving C1 tests satisfy the original weak transport identity.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_attention_spacetime_test (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (A : ℝ → EucSpace d → ℝ) (hA : ContDiff ℝ 1 (Function.uncurry A)) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, A s x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d))
        (1, fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t x)
          ∂(μ t : Measure (SSphere d))) t := by
  let F (x : EucSpace d) := fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t x
  let G (s : ℝ) (x : EucSpace d) := fderiv ℝ (Function.uncurry A) (s, x) (1, 0)
  have hGc : Continuous (Function.uncurry G) :=
    (hA.continuous_fderiv_apply (by norm_num)).comp (continuous_id.prodMk continuous_const)
  have hD (s : ℝ) (x : SSphere d) : HasDerivAt (fun r => A r x) (G s x) s := by
    dsimp only [G]
    simpa only [Function.comp_def, Function.uncurry_apply_pair, id_eq] using
      (hA.differentiable (by norm_num) (s, (x : EucSpace d))).hasFDerivAt.comp_hasDerivAt s
      ((hasDerivAt_id s).prodMk (hasDerivAt_const s (x : EucSpace d)))
  have hfixed := hasDerivAt_attention_cauchyPB_test B μ hμ
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
    (continuous_attention_cauchyPB_test B μ hμ (G t)
      (hGc.comp (continuous_const.prodMk continuous_id))).continuousAt
  have hc₀ : Continuous (fun x : SSphere d =>
      fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d)) (0, F x)) :=
    (hA.continuous_fderiv_apply (by norm_num)).comp
      ((continuous_const.prodMk continuous_subtype_val).prodMk
        (continuous_const.prodMk ((continuous_fullVF_attention B (μ t) t).comp continuous_subtype_val)))
  have hc₁ : Continuous (fun x : SSphere d => G t x) :=
    hGc.comp (continuous_const.prodMk continuous_subtype_val)
  convert h using 1
  change (∫ x : SSphere d, fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d)) (1, F x)
      ∂(μ t : Measure (SSphere d))) =
    (∫ x : SSphere d, inner ℝ (gradient (A t) (x : EucSpace d)) (F x)
      ∂(μ t : Measure (SSphere d))) + ∫ x : SSphere d, G t x ∂(μ t : Measure (SSphere d))
  simp_rw [inner_gradient_left, hx]
  rw [← integral_add (integrable_spherical_continuous (μ t) _ hc₀)
    (integrable_spherical_continuous (μ t) _ hc₁)]
  apply integral_congr_ae
  exact ae_of_all _ fun x => by
    dsimp only [G, F]
    rw [← map_add]
    simp

/-- A nonpositive material derivative makes the actual test integral
nonincreasing. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem attention_spacetime_test_le (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (A : ℝ → EucSpace d → ℝ) (hA : ContDiff ℝ 1 (Function.uncurry A))
    (hsub : ∀ (s : ℝ) (x : SSphere d), fderiv ℝ (Function.uncurry A) (s, (x : EucSpace d))
      (1, fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ s) s x) ≤ 0)
    (s t : ℝ) (hst : s ≤ t) :
    (∫ x : SSphere d, A t x ∂(μ t : Measure (SSphere d))) ≤
      ∫ x : SSphere d, A s x ∂(μ s : Measure (SSphere d)) := by
  have hd := hasDerivAt_attention_spacetime_test B μ hμ A hA
  exact antitone_of_deriv_nonpos (fun r => (hd r).differentiableAt)
    (fun r => by rw [(hd r).deriv]; exact integral_nonpos (hsub r)) hst

section FiniteComparison

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- Finite smooth cones are subsolutions for the current attention field.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem fderiv_attention_finiteTransportTest_nonpos (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (L ε β T : ℝ) (hL : 0 ≤ L) (hε : 0 < ε) (hβ : 0 < β) (Y : ι → ℝ → SSphere d)
    (hY : ∀ i s, HasDerivAt (fun r => (Y i r : EucSpace d))
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ s) s (Y i s)) s)
    (f : SSphere d → ℝ) (t : ℝ) (x : SSphere d) :
    fderiv ℝ (finiteTransportTest (2 * exp (2 * ‖B‖) * ‖B‖ + 2) L ε β T Y f) (t, (x : EucSpace d))
      (1, fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ t) t x) ≤ 0 := by
  exact fderiv_finiteTransportTest_nonpos_at _ ⟨_, by positivity⟩
    (lipschitzOnWith_fullVF_attention_sphere B (μ t) t) L ε β T hL hε hβ Y
    (fun i => contDiff_attention_particle B μ hμ (Y i) (hY i)) f t (fun i => hY i t) x

/-- The evolving weak law obeys the finite transport-cone comparison.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem integral_attention_finiteTransportTest_le (B : ParamMatrix d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) μ)
    (L ε β T : ℝ) (hL : 0 ≤ L) (hε : 0 < ε) (hβ : 0 < β) (Y : ι → ℝ → SSphere d)
    (hY : ∀ i s, HasDerivAt (fun r => (Y i r : EucSpace d))
      (fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩) (μ s) s (Y i s)) s)
    (f : SSphere d → ℝ) (s t : ℝ) (hst : s ≤ t) :
    (∫ x : SSphere d, finiteTransportTest (2 * exp (2 * ‖B‖) * ‖B‖ + 2) L ε β T Y f
      (t, (x : EucSpace d)) ∂(μ t : Measure (SSphere d))) ≤
    ∫ x : SSphere d, finiteTransportTest (2 * exp (2 * ‖B‖) * ‖B‖ + 2) L ε β T Y f
      (s, (x : EucSpace d)) ∂(μ s : Measure (SSphere d)) := by
  let A (r : ℝ) (x : EucSpace d) :=
    finiteTransportTest (2 * exp (2 * ‖B‖) * ‖B‖ + 2) L ε β T Y f (r, x)
  have hC : ∀ i, ContDiff ℝ 1 (fun r => (Y i r : EucSpace d)) :=
    fun i => contDiff_attention_particle B μ hμ (Y i) (hY i)
  have hA : ContDiff ℝ 1 (Function.uncurry A) :=
    contDiff_finiteTransportTest _ _ _ _ _ hε Y hC f
  have hsub (r : ℝ) (x : SSphere d) :
      fderiv ℝ (Function.uncurry A) (r, (x : EucSpace d))
        (1, fullVF d (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace d), B, 0, 0, 0⟩)
          (μ r) r x) ≤ 0 :=
    fderiv_attention_finiteTransportTest_nonpos B μ hμ L ε β T hL hε hβ Y hY f r x
  exact attention_spacetime_test_le B μ hμ A hA hsub s t hst

end FiniteComparison

/-- A genuine attention weak law and actual particle centers jointly
realize the weak, particle, smooth-test and comparison hypotheses. The
terminal distance test is nonconstant on the circle. -/
example : ∃ (B : ParamMatrix 2) (μ : ℝ → Perspective.ProbSphere 2)
    (Y : Fin 2 → ℝ → SSphere 2) (A : ℝ → EucSpace 2 → ℝ),
    cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) μ ∧
    (∀ i s, HasDerivAt (fun r => (Y i r : EucSpace 2))
      (fullVF 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) (μ s) s (Y i s)) s) ∧
    ContDiff ℝ 1 (Function.uncurry A) ∧
    (∀ s (x : SSphere 2), fderiv ℝ (Function.uncurry A) (s, (x : EucSpace 2))
      (1, fullVF 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), B, 0, 0, 0⟩) (μ s) s x) ≤ 0) ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 := by
  let p : SSphere 2 := basePoint 1
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  let Y : Fin 2 → ℝ → SSphere 2 := fun _ _ => p
  let f : SSphere 2 → ℝ := fun x => dist x p
  have hθ : pureAttentionParams 2 1 =
      (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), ContinuousLinearMap.id ℝ (EucSpace 2), 0, 0, 0⟩) := by
    funext t
    simp only [pureAttentionParams, one_smul]
  have hμ : cauchyPB 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), ContinuousLinearMap.id ℝ (EucSpace 2), 0, 0, 0⟩) μ := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 p
  have hY (i : Fin 2) (s : ℝ) : HasDerivAt (fun r => (Y i r : EucSpace 2))
      (fullVF 2 (fun _ => ⟨ContinuousLinearMap.id ℝ (EucSpace 2), ContinuousLinearMap.id ℝ (EucSpace 2), 0, 0, 0⟩) (μ s) s (Y i s)) s := by
    rw [← hθ, fullVF_pureAttentionParams]
    change HasDerivAt (fun _ => (p : EucSpace 2)) (Perspective.vectorField 2 1 (Perspective.diracProb 2 p) p) s
    rw [Perspective.vectorField_diracProb_self]
    exact hasDerivAt_const s (p : EucSpace 2)
  let A (s : ℝ) (x : EucSpace 2) :=
    finiteTransportTest (2 * exp (2 * ‖(ContinuousLinearMap.id ℝ (EucSpace 2) : ParamMatrix 2)‖) * ‖(ContinuousLinearMap.id ℝ (EucSpace 2) : ParamMatrix 2)‖ + 2) 1 1 1 1 Y f (s, x)
  refine ⟨ContinuousLinearMap.id ℝ (EucSpace 2), μ, Y, A, hμ, hY, ?_, ?_, zero_le_one, zero_lt_one, zero_lt_one⟩
  · exact contDiff_finiteTransportTest _ _ _ _ _ zero_lt_one Y (fun _ => contDiff_const) f
  · exact fun s x => fderiv_attention_finiteTransportTest_nonpos (ContinuousLinearMap.id ℝ (EucSpace 2)) μ hμ 1 1 1 1
      zero_le_one zero_lt_one zero_lt_one Y hY f s x

end Transformer.Interpolation
