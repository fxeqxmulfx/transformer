module

public import Transformer.Grokking.FiniteRecovery.ArgandAttention
public import Transformer.Optimization.C1Lipschitz

/-! For the offset outside the norm, the only potentially singular part
of the derivative is the operator x ⊗ x / ‖x‖. Its extension
at zero is Lipschitz, which gives the local gradient bounds needed by
the exact clipped momentum update. -/

@[expose] public section

open Filter
open scoped Topology NNReal

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2 Transformer.Optimization

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

private noncomputable def unitDirection (x : E) : E := ‖x‖⁻¹ • x

private theorem norm_unitDirection_le (x : E) : ‖unitDirection x‖ ≤ 1 := by
  by_cases hx : ‖x‖ = 0
  · simp [unitDirection, hx]
  · rw [unitDirection, norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr (norm_nonneg x)),
      inv_mul_cancel₀ hx]

private theorem norm_smul_unitDirection (x : E) : ‖x‖ • unitDirection x = x := by
  by_cases hx : ‖x‖ = 0
  · simp [norm_eq_zero.mp hx]
  · rw [unitDirection, smul_smul, mul_inv_cancel₀ hx, one_smul]

private noncomputable def normRankOne (x : E) : E →L[ℝ] E :=
  (innerSL ℝ (unitDirection x)).smulRight x

private theorem norm_sub_normRankOne_le {x y : E} (hxy : ‖x‖ ≤ ‖y‖) :
    ‖normRankOne x - normRankOne y‖ ≤ 3 * ‖x - y‖ := by
  have hdecomp : ‖x‖ • (unitDirection x - unitDirection y) =
      (x - y) + (‖y‖ - ‖x‖) • unitDirection y := by
    rw [smul_sub, norm_smul_unitDirection, sub_smul, norm_smul_unitDirection]
    abel
  have hprod : ‖x‖ * ‖unitDirection x - unitDirection y‖ ≤ 2 * ‖x - y‖ := by
    rw [← Real.norm_of_nonneg (norm_nonneg x), ← norm_smul, hdecomp]
    refine (norm_add_le _ _).trans ?_
    rw [norm_smul, Real.norm_of_nonneg (sub_nonneg.mpr hxy)]
    have h := mul_le_mul_of_nonneg_left (norm_unitDirection_le y) (sub_nonneg.mpr hxy)
    linarith [norm_sub_norm_le y x, norm_sub_rev y x]
  have hsplit : normRankOne x - normRankOne y =
      (innerSL ℝ (unitDirection x - unitDirection y)).smulRight x +
        (innerSL ℝ (unitDirection y)).smulRight (x - y) := by
    ext z
    simp only [normRankOne, sub_apply, ContinuousLinearMap.smulRight_apply,
      add_apply, map_sub, sub_smul, smul_sub]
    abel
  have hinner (u : E) : ‖innerSL ℝ u‖ ≤ ‖u‖ := by
    apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _)
    intro z
    exact abs_real_inner_le_norm _ _
  rw [hsplit]
  refine (norm_add_le _ _).trans ?_
  rw [ContinuousLinearMap.norm_smulRight_apply, ContinuousLinearMap.norm_smulRight_apply]
  have h₁ := mul_le_mul_of_nonneg_right (hinner (unitDirection x - unitDirection y)) (norm_nonneg x)
  have h₂ := mul_le_mul_of_nonneg_right
    ((hinner (unitDirection y)).trans (norm_unitDirection_le y)) (norm_nonneg (x - y))
  linarith [mul_comm ‖unitDirection x - unitDirection y‖ ‖x‖]

private theorem lipschitz_normRankOne : LipschitzWith 3 (normRankOne (E := E)) := by
  rw [lipschitzWith_iff_dist_le_mul]
  intro x y
  simp only [dist_eq_norm, NNReal.coe_ofNat]
  rcases le_total ‖x‖ ‖y‖ with h | h
  · exact norm_sub_normRankOne_le h
  · rw [norm_sub_rev, norm_sub_rev x y]
    exact norm_sub_normRankOne_le h

private theorem hasFDerivAt_norm_of_ne {x : E} (hx : x ≠ 0) :
    HasFDerivAt (norm : E → ℝ) (innerSL ℝ (unitDirection x)) x := by
  have h := ((contDiffAt_norm ℝ hx : ContDiffAt ℝ 1 _ x).differentiableAt one_ne_zero).hasFDerivAt
  have hpow := (h.pow 2).unique (hasStrictFDerivAt_norm_sq x).hasFDerivAt
  have hpos := norm_pos_iff.mpr hx
  have hF : fderiv ℝ (norm : E → ℝ) x = innerSL ℝ (unitDirection x) := by
    ext z
    have ht := congrArg (fun L : E →L[ℝ] ℝ => L z) hpow
    simp only [two_smul, Nat.reduceSub, pow_one, smul_apply, add_apply,
      smul_eq_mul, unitDirection, map_smul] at ht ⊢
    field_simp [hpos.ne']
    nlinarith only [ht]
  rwa [← hF]

/-- Exact real derivative of the norm-plus-offset radial map. The inverse
norm in the rank-one term is interpreted as zero at the origin. -/
theorem hasFDerivAt_radial {ε : ℝ} (hε : 0 < ε) (x : E) :
    HasFDerivAt (fun y : E => (‖y‖ + ε)⁻¹ • y)
      ((‖x‖ + ε)⁻¹ • ContinuousLinearMap.id ℝ E -
        ((‖x‖ + ε)⁻¹ ^ 2) • (innerSL ℝ (‖x‖⁻¹ • x)).smulRight x) x := by
  let k (x : E) := (‖x‖ + ε)⁻¹
  let radial (x : E) := k x • x
  have hden (x : E) : 0 < ‖x‖ + ε := add_pos_of_nonneg_of_pos (norm_nonneg _) hε
  have hk : Continuous k := (continuous_norm.add continuous_const).inv₀ (fun x => (hden x).ne')
  change HasFDerivAt radial (k x • ContinuousLinearMap.id ℝ E - k x ^ 2 • normRankOne x) x
  by_cases hx : x = 0
  · subst x
    have hzero : normRankOne (0 : E) = 0 := by ext; simp [normRankOne]
    simp only [hzero, smul_zero, sub_zero]
    rw [hasFDerivAt_iff_isLittleO]
    have hsmall : Tendsto (fun x : E => k x - k 0) (𝓝 0) (𝓝 0) := by
      simpa only [sub_self] using (hk.tendsto (0 : E)).sub_const (k 0)
    simpa only [radial, smul_zero, smul_apply,
      ContinuousLinearMap.id_apply, sub_zero, one_smul, ← sub_smul] using
      (Asymptotics.isLittleO_one_iff ℝ).2 hsmall |>.smul_isBigO
        (Asymptotics.isBigO_refl (fun x : E => x) (𝓝 0))
  · have hK : HasFDerivAt k (-(k x ^ 2) • innerSL ℝ (unitDirection x)) x := by
      have hh := (hasFDerivAt_inv (hden x).ne').comp x
        ((hasFDerivAt_norm_of_ne hx).add_const ε)
      have heq :
          (ContinuousLinearMap.toSpanSingleton ℝ (-((‖x‖ + ε) ^ 2)⁻¹)).comp
            (innerSL ℝ (unitDirection x)) =
              (-(k x ^ 2)) • innerSL ℝ (unitDirection x) := by
        ext z
        simp [k, inv_pow, mul_comm]
      rw [heq] at hh
      exact hh
    have H := hK.smul (hasFDerivAt_id x)
    change HasFDerivAt (k • id) _ x
    convert H using 1
    ext z
    simp [normRankOne, sub_eq_add_neg, smul_smul, neg_smul]

set_option maxHeartbeats 2000000 in
private theorem c1Lipschitz_radial {ε : ℝ} (hε : 0 < ε) :
    C1Lipschitz (fun x : E => (‖x‖ + ε)⁻¹ • x) := by
  let k (x : E) := (‖x‖ + ε)⁻¹
  let radial (x : E) := k x • x
  have hden (x : E) : 0 < ‖x‖ + ε := add_pos_of_nonneg_of_pos (norm_nonneg _) hε
  have hk : LocallyLipschitz k := fun x => by
    have hi : ContDiffAt ℝ 1 (fun y : ℝ => y⁻¹) (‖x‖ + ε) :=
      contDiffAt_id.inv (hden x).ne'
    obtain ⟨a, s, hs, hL⟩ := hi.exists_lipschitzOnWith
    have hn : LipschitzWith 1 (fun y : E => ‖y‖ + ε) := by
      rw [lipschitzWith_iff_dist_le_mul]
      intro u v
      rw [Real.dist_eq, add_sub_add_right_eq_sub, NNReal.coe_one, one_mul, dist_eq_norm]
      exact abs_norm_sub_norm_le u v
    refine ⟨a * 1, (fun y : E => ‖y‖ + ε) ⁻¹' s, hn.continuous.continuousAt hs, ?_⟩
    exact hL.comp hn.lipschitzOnWith (Set.mapsTo_preimage _ _)
  have hF (x : E) : HasFDerivAt radial
      (k x • ContinuousLinearMap.id ℝ E - k x ^ 2 • normRankOne x) x :=
    hasFDerivAt_radial hε x
  have hL : LocallyLipschitz (fderiv ℝ radial) := by
    rw [show fderiv ℝ radial = _ from funext (fun x => (hF x).fderiv)]
    have hopp : ContDiff ℝ 1 (fun p : ℝ × (E →L[ℝ] E) =>
        p.1 • ContinuousLinearMap.id ℝ E - (p.1 ^ 2) • p.2) :=
      (contDiff_fst.smul contDiff_const).sub ((contDiff_fst.pow 2).smul contDiff_snd)
    exact LocallyLipschitz.comp
      (f := fun p : ℝ × (E →L[ℝ] E) => p.1 • ContinuousLinearMap.id ℝ E - p.1 ^ 2 • p.2)
      (g := fun x => (k x, normRankOne x)) hopp.locallyLipschitz
      (hk.prodMk lipschitz_normRankOne.locallyLipschitz)
  exact ⟨contDiff_one_iff_fderiv.mpr ⟨fun x => (hF x).differentiableAt, hL.continuous⟩, hL⟩

/-- The actual complex RMS map has a locally Lipschitz real derivative
even at zero, where the norm itself fails to be differentiable. -/
theorem c1Lipschitz_argandRMS {c : ℕ} {ε : ℝ} (hε : 0 < ε) :
    C1Lipschitz (argandRMS ε : CSpace c → CSpace c) := by
  have he (x : CSpace c) : argandRMS ε x = Real.sqrt c • ((‖x‖ + ε)⁻¹ • x) := by
    rw [argandRMS, smul_smul, div_eq_mul_inv]
    rfl
  rw [show (argandRMS ε : CSpace c → CSpace c) = _ from funext he]
  have hScale : ContDiff ℝ 2 (fun y : CSpace c => Real.sqrt c • y) :=
    (contDiff_id : ContDiff ℝ 2 (fun y : CSpace c => y)).const_smul (Real.sqrt c)
  exact (C1Lipschitz.of_contDiff hScale).comp (c1Lipschitz_radial hε)

theorem contDiff_argandRMS {c : ℕ} {ε : ℝ} (hε : 0 < ε) :
    ContDiff ℝ 1 (argandRMS ε : CSpace c → CSpace c) :=
  (c1Lipschitz_argandRMS hε).contDiff

example := c1Lipschitz_argandRMS (c := 64) (ε := 1 / 1000000) (by norm_num)

end Transformer.Grokking.FiniteRecovery.Argand
