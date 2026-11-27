module

public import Transformer.Optimization.AnalyticLiftInequality
public import Transformer.Optimization.AnalyticLiftOperations
public import Transformer.Optimization.FullMomentumConvergence

/-! The unclipped kinetic energy of an analytically lifted objective
satisfies the local inequality needed by the momentum convergence theorem. -/

@[expose] public section

open Filter Metric
open scoped Topology InnerProductSpace

namespace Transformer.Optimization

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

private theorem norm_fderiv_momentumEnergy_le {f : E → ℝ} (hf : Differentiable ℝ f)
    {η : ℝ} (hη : 0 ≤ η) (x v : E) :
    ‖fderiv ℝ (fun p : E × E => f p.1 + η / 2 * ‖p.2‖ ^ 2) (x, v)‖ ≤
      ‖gradient f x‖ + η * ‖v‖ := by
  let p := (x, v)
  let X := ContinuousLinearMap.fst ℝ E E
  let V := ContinuousLinearMap.snd ℝ E E
  have hd := (hf x).hasFDerivAt.comp p X.hasFDerivAt
  have hk : HasFDerivAt (fun v : E => η / 2 * ‖v‖ ^ 2) (η • innerSL ℝ v) v := by
    have h := (hasStrictFDerivAt_norm_sq v).hasFDerivAt.const_mul (η / 2)
    have heq : (η / 2) • (2 • innerSL ℝ v) = η • innerSL ℝ v := by
      apply ContinuousLinearMap.ext
      intro u
      simp only [two_smul, smul_apply, add_apply, smul_eq_mul]
      ring
    rwa [heq] at h
  have hkin := hk.comp p (V.hasFDerivAt (x := p))
  have hz := (hd.add hkin).fderiv
  have heq : fderiv ℝ (fun p : E × E => f p.1 + η / 2 * ‖p.2‖ ^ 2) (x, v) =
      (fderiv ℝ f x).comp X + (η • innerSL ℝ v).comp V := hz
  rw [heq]
  have hp (a : E →L[ℝ] ℝ) : ‖a.comp X‖ ≤ ‖a‖ := by
    exact (ContinuousLinearMap.opNorm_comp_le _ _).trans
      (by simpa only [mul_one] using
        mul_le_mul_of_nonneg_left (ContinuousLinearMap.norm_fst_le ℝ E E) (norm_nonneg a))
  have hv (a : E →L[ℝ] ℝ) : ‖a.comp V‖ ≤ ‖a‖ := by
    exact (ContinuousLinearMap.opNorm_comp_le _ _).trans
      (by simpa only [mul_one] using
        mul_le_mul_of_nonneg_left (ContinuousLinearMap.norm_snd_le ℝ E E) (norm_nonneg a))
  have hv' : ‖η • innerSL ℝ v‖ = η * ‖v‖ := by
    rw [norm_smul, Real.norm_of_nonneg hη]
    exact congrArg (η * ·) ((InnerProductSpace.toDual ℝ E).norm_map v)
  calc
    _ ≤ ‖(fderiv ℝ f x).comp X‖ + ‖(η • innerSL ℝ v).comp V‖ := norm_add_le ((fderiv ℝ f x).comp X) ((η • innerSL ℝ v).comp V)
    _ ≤ ‖fderiv ℝ f x‖ + ‖η • innerSL ℝ v‖ := add_le_add (hp _) (hv _)
    _ = _ := congrArg₂ (· + ·) ((InnerProductSpace.toDual ℝ E).symm.norm_map _).symm hv'

/-- Analytic lifting discharges the local energy hypothesis of
full-gradient momentum at every point. -/
theorem HasAnalyticLift.momentumEnergyInequality {f : E → ℝ} (hf : HasAnalyticLift f)
    {η : ℝ} (hη : 0 ≤ η) (z : E) : MomentumEnergyInequality f η z := by
  let g (p : E × E) : ℝ := f p.1 + η / 2 * ‖p.2‖ ^ 2
  have h1 : HasAnalyticLift (fun p : E × E => f p.1) :=
    hf.comp (HasAnalyticLift.clm (ContinuousLinearMap.fst ℝ E E))
  have h2 : HasAnalyticLift (fun p : E × E => p.2) :=
    HasAnalyticLift.clm (ContinuousLinearMap.snd ℝ E E)
  have hn : HasAnalyticLift (fun p : E × E => ‖p.2‖ ^ 2) := by
    simpa only [real_inner_self_eq_norm_sq] using h2.inner ℝ h2
  have hg : HasAnalyticLift g := h1.add ((HasAnalyticLift.const (η / 2)).mul hn)
  obtain ⟨a, m, ha, ha', hm, hloc⟩ := hg.local_fderiv_inequality (z, 0)
  obtain ⟨r, hr, hb⟩ := Metric.eventually_nhds_iff.mp hloc
  let K := max 1 η
  have hK : 0 < K := zero_lt_one.trans_le (le_max_left _ _)
  refine ⟨r, hr, m / K, div_pos hm hK, a, ha, ha', fun x hx v hv hpos => ?_⟩
  have hxv : dist (x, v) (z, 0) < r := by
    rw [Prod.dist_eq, dist_zero_right]
    exact max_lt (mem_ball.mp hx) hv
  have hz : g (z, 0) = f z := by simp [g]
  have h := hb hxv (by rw [hz]; exact sub_pos.mp hpos)
  rw [hz] at h
  have hN := norm_fderiv_momentumEnergy_le hf.differentiable hη x v
  have hbound : ‖gradient f x‖ + η * ‖v‖ ≤ K * (‖gradient f x‖ + ‖v‖) := by
    calc
      _ ≤ K * ‖gradient f x‖ + K * ‖v‖ := add_le_add
        (by simpa only [one_mul] using mul_le_mul_of_nonneg_right (le_max_left 1 η) (norm_nonneg _))
        (mul_le_mul_of_nonneg_right (le_max_right 1 η) (norm_nonneg _))
      _ = _ := (mul_add _ _ _).symm
  have hresult := h.trans (hN.trans hbound)
  apply (le_of_mul_le_mul_left _ hK)
  rw [← mul_assoc, show K * (m / K) = m by field_simp]
  exact hresult

end Transformer.Optimization
