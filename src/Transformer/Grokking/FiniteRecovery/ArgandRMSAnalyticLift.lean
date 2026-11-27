module

public import Transformer.Grokking.FiniteRecovery.ArgandRMSDifferentiability
public import Transformer.Optimization.AnalyticLiftOperations

/-! The exact norm-plus-offset RMS has an analytic lift. At a vector
(x), choose (r=‖x‖) and a bounded vector multiplier. The correction
for (r²-‖x‖²) makes the derivative in the radius direction vanish.
The lift works at zero without differentiating the norm there. -/

@[expose] public section

open Filter Metric
open scoped Topology InnerProductSpace

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transformer.Optimization

variable {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

private noncomputable def radialLift (ε : ℝ) (p : E × (ℝ × E)) : E :=
  (p.2.1 + ε)⁻¹ • p.1 + (p.2.1 ^ 2 - ‖p.1‖ ^ 2) • p.2.2

private noncomputable def liftMultiplier (ε : ℝ) (x : E) : E :=
  ((‖x‖ + ε)⁻¹ ^ 2 / 2) • (‖x‖⁻¹ • x)

private theorem norm_smul_inverse_norm_smul (x : E) : ‖x‖ • (‖x‖⁻¹ • x) = x := by
  by_cases h : ‖x‖ = 0
  · simp [norm_eq_zero.mp h]
  · rw [smul_smul, mul_inv_cancel₀ h, one_smul]

private theorem radialLift_derivative {ε : ℝ} (hε : 0 < ε) (x : E) :
    HasFDerivAt (radialLift ε)
      ((fderiv ℝ (fun y : E => (‖y‖ + ε)⁻¹ • y) x).comp
        (ContinuousLinearMap.fst ℝ E (ℝ × E))) (x, (‖x‖, liftMultiplier ε x)) := by
  let P := E × (ℝ × E)
  let p : P := (x, (‖x‖, liftMultiplier ε x))
  let xL := ContinuousLinearMap.fst ℝ E (ℝ × E)
  let rL := (ContinuousLinearMap.fst ℝ ℝ E).comp (ContinuousLinearMap.snd ℝ E (ℝ × E))
  let bL := (ContinuousLinearMap.snd ℝ ℝ E).comp (ContinuousLinearMap.snd ℝ E (ℝ × E))
  let k := (‖x‖ + ε)⁻¹
  have hden : ‖x‖ + ε ≠ 0 := (add_pos_of_nonneg_of_pos (norm_nonneg x) hε).ne'
  have hk₀ : HasDerivAt (fun r : ℝ => (r + ε)⁻¹) (-(k ^ 2)) ‖x‖ := by
    have h := ((hasDerivAt_id ‖x‖).add_const ε).inv hden
    convert h using 1 <;> first | rfl | simp only [k, id_eq, inv_pow, neg_div, one_div]
  have hk := hk₀.comp_hasFDerivAt p (rL.hasFDerivAt (x := p))
  have hr := (rL.hasFDerivAt (x := p)).pow 2
  have hx := (hasStrictFDerivAt_norm_sq x).hasFDerivAt.comp p (xL.hasFDerivAt (x := p))
  have hb := (hr.sub hx).smul (bL.hasFDerivAt (x := p))
  have h := (hk.smul (xL.hasFDerivAt (x := p))).add hb
  rw [(hasFDerivAt_radial hε x).fderiv]
  convert h using 1
  · rfl
  apply ContinuousLinearMap.ext
  intro t
  simp only [ContinuousLinearMap.comp_apply, sub_apply, add_apply, smul_apply,
    ContinuousLinearMap.id_apply, ContinuousLinearMap.smulRight_apply, pow_one,
    Nat.reduceSub, nsmul_eq_mul, Nat.cast_ofNat, innerSL_apply_apply]
  change k • t.1 - k ^ 2 • (⟪‖x‖⁻¹ • x, t.1⟫_ℝ • x) =
    k • t.1 + (-(k ^ 2) * t.2.1) • x +
      ((‖x‖ ^ 2 - ‖x‖ ^ 2) • t.2.2 + ((2 * ‖x‖) • t.2.1 - 2 * ⟪x, t.1⟫_ℝ) • liftMultiplier ε x)
  simp only [sub_self, zero_smul, zero_add, smul_eq_mul]
  let u : E := ‖x‖⁻¹ • x
  have hu : ‖x‖ • u = x := norm_smul_inverse_norm_smul x
  have hi : ⟪u, t.1⟫_ℝ • x = ⟪x, t.1⟫_ℝ • u := by
    simp only [u, real_inner_smul_left, smul_smul, mul_comm]
  rw [hi]
  change _ = k • t.1 + (-(k ^ 2) * t.2.1) • x +
    ((2 * ‖x‖) * t.2.1 - 2 * ⟪x, t.1⟫_ℝ) • ((k ^ 2 / 2) • u)
  calc
    _ = k • t.1 - k ^ 2 • (⟪x, t.1⟫_ℝ • u) +
        (-(k ^ 2) * t.2.1) • x + (k ^ 2 * t.2.1) • x := by module
    _ = _ := by
      rw [congrArg (fun y : E => (k ^ 2 * t.2.1) • y) hu.symm]
      module

variable [FiniteDimensional ℝ E]

/-- The complete radial normalizer lifts at all inputs, including zero
head, embedding and residual activations. -/
theorem analyticLift_radial {ε : ℝ} (hε : 0 < ε) :
    HasAnalyticLift (fun x : E => (‖x‖ + ε)⁻¹ • x) := by
  let U : AnalyticLiftSpace := ⟨ℝ × E⟩
  refine ⟨fun x => (hasFDerivAt_radial hε x).differentiableAt, fun x z hx => ?_⟩
  let dir (n : ℕ) : E := ‖x n‖⁻¹ • x n
  have hc (n : ℕ) : dir n ∈ closedBall (0 : E) 1 := by
    apply mem_closedBall_zero_iff.mpr
    dsimp only [dir]
    rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr (norm_nonneg _))]
    by_cases h : ‖x n‖ = 0
    · simp [h]
    · rw [inv_mul_cancel₀ h]
  obtain ⟨u₀, _, σ, hσ, hu⟩ := (isCompact_closedBall (0 : E) 1).tendsto_subseq hc
  have hx' := hx.comp hσ.tendsto_atTop
  have hν := hx'.norm
  have hk := (hν.add_const ε).inv₀ (add_pos_of_nonneg_of_pos (norm_nonneg z) hε).ne'
  have hmult : Tendsto (fun n => liftMultiplier ε (x (σ n))) atTop
      (𝓝 (((‖z‖ + ε)⁻¹ ^ 2 / 2) • u₀)) :=
    ((hk.pow 2).div_const 2).smul hu
  let b₀ := ((‖z‖ + ε)⁻¹ ^ 2 / 2) • u₀
  let p₀ : E × U.carrier := (z, (‖z‖, b₀))
  refine ⟨U, radialLift ε, (fun n => (‖x (σ n)‖, liftMultiplier ε (x (σ n)))), (‖z‖, b₀),
    σ, hσ, hν.prodMk_nhds hmult, ?_, fun n => ⟨by simp [radialLift], radialLift_derivative hε _⟩⟩
  have ha : AnalyticAt ℝ (fun p : E × (ℝ × E) => p.2.1) p₀ :=
    (analyticAt_fst (p := p₀.2)).comp (f := Prod.snd) analyticAt_snd
  have hb : AnalyticAt ℝ (fun p : E × (ℝ × E) => p.2.2) p₀ :=
    (analyticAt_snd (p := p₀.2)).comp (f := Prod.snd) analyticAt_snd
  have hv : AnalyticAt ℝ (fun p : E × (ℝ × E) => ‖p.1‖ ^ 2) p₀ := by
    have hpair : AnalyticAt ℝ (fun p : E × (ℝ × E) => (p.1, p.1)) p₀ :=
      analyticAt_fst.prod analyticAt_fst
    have ht := ((innerSL ℝ : E →L[ℝ] E →L[ℝ] ℝ).analyticAt_bilinear (z, z)).comp_of_eq
      hpair (show (p₀.1, p₀.1) = (z, z) from rfl)
    have heq : (fun q : E × E => innerSL ℝ q.1 q.2) ∘
        (fun p : E × (ℝ × E) => (p.1, p.1)) = (fun p => ‖p.1‖ ^ 2) := by
      funext p
      exact real_inner_self_eq_norm_sq p.1
    rwa [heq] at ht
  have hd : ‖z‖ + ε ≠ 0 := (add_pos_of_nonneg_of_pos (norm_nonneg z) hε).ne'
  exact (((ha.add analyticAt_const).inv hd).smul analyticAt_fst).add (((ha.pow 2).sub hv).smul hb)

theorem analyticLift_argandRMS {c : ℕ} {ε : ℝ} (hε : 0 < ε) :
    HasAnalyticLift (argandRMS ε : CSpace c → CSpace c) := by
  have he (x : CSpace c) : argandRMS ε x = Real.sqrt c • ((‖x‖ + ε)⁻¹ • x) := by
    rw [argandRMS, smul_smul, div_eq_mul_inv]
    rfl
  rw [show (argandRMS ε : CSpace c → CSpace c) = _ from funext he]
  exact (HasAnalyticLift.const (Real.sqrt c)).smul (analyticLift_radial hε)

example := analyticLift_argandRMS (c := 64) (ε := 1 / 1000000) (by norm_num)

end Transformer.Grokking.FiniteRecovery.Argand
