module

public import Transformer.Causal.Section4_SubspaceCovectors
public import Transformer.Causal.Section4_ScalarLimits

/-!
# Convergence of the original follower from positive overlap

ArXiv:2411.04990v2, §4, `thm2`, restricted to two tokens.
The exact spectral assumptions give a nonnegative self contribution
in the positive leading hemisphere. The converging actual leader gives
an error tending to zero in the remaining force. Uniformly positive
drift on every closed strip below overlap one then proves convergence
of the actual follower, without assuming its limit exists.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d : ℕ}

/-- In the original leading positive hemisphere, the original two-token
field has its positive consensus drift minus the actual leader error.
Source: arXiv:2411.04990v2, §4, `thm2`, restricted to two tokens. -/
theorem csaSecond_leading_field_lower (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 ≤ lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L) (Y : SphereTuple d 2)
    (hq : 0 ≤ inner ℝ (u : EucSpace d) (Y 1 : EucSpace d)) :
    lam * csaWeightFloor β Q K 2 *
        (1 - inner ℝ (u : EucSpace d) (Y 1 : EucSpace d) ^ 2) -
      ‖V‖ * ‖(Y 0 : EucSpace d) - (u : EucSpace d)‖ ≤
        inner ℝ (u : EucSpace d)
          (csaAmbientField β Q K V (fun j => (Y j : EucSpace d)) 1) := by
  let a := csaTokenWeight β Q K (fun j => (Y j : EucSpace d)) 1 0
  let b := csaTokenWeight β Q K (fun j => (Y j : EucSpace d)) 1 1
  let r := inner ℝ (u : EucSpace d) (Y 1 : EucSpace d)
  let e := ‖V‖ * ‖(Y 0 : EucSpace d) - (u : EucSpace d)‖
  have ha : csaWeightFloor β Q K 2 ≤ a := csaWeightFloor_le β Q K _
    (fun j => mem_sphere_zero_iff_norm.mp (Y j).2) 1 0 (by norm_num)
  have ha0 : 0 ≤ a := csaTokenWeight_nonneg β Q K _ 1 0
  have ha1 : a ≤ 1 := csaTokenWeight_le_one β Q K _ 1 0
  have hb0 : 0 ≤ b := csaTokenWeight_nonneg β Q K _ 1 1
  have hr := csaSphereOverlap_bounds u (Y 1)
  have hgap : 0 ≤ 1 - r ^ 2 := by dsimp [r]; nlinarith
  have herr : 0 ≤ e := by dsimp [e]; positivity
  have hi := (abs_le.mp (leadingOverlapInteraction_approx V lam u (Y 1) (Y 0) (heig _ hu))).1
  have hself : 0 ≤ inner ℝ (u : EucSpace d) (proj d (Y 1 : EucSpace d) (V (Y 1 : EucSpace d))) := by
    have hceil := rayleigh_le_leading V L lam heig hinv hdecay (Y 1 : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp (Y 1).2, one_pow, mul_one] at hceil
    rw [proj, inner_sub_right, real_inner_smul_right, inner_value_of_mem_leading V L lam heig hinv _ hu,
      real_inner_comm (V (Y 1 : EucSpace d)) (Y 1 : EucSpace d)]
    nlinarith
  rw [csaAmbientField_eq_sum, Fin.sum_univ_two, inner_add_right, real_inner_smul_right,
    real_inner_smul_right]
  change lam * csaWeightFloor β Q K 2 * (1 - r ^ 2) - e ≤ a * _ + b * _
  have hf : lam * csaWeightFloor β Q K 2 * (1 - r ^ 2) ≤ a * (lam * (1 - r ^ 2)) := by
    nlinarith [mul_le_mul_of_nonneg_right ha (mul_nonneg hlam hgap)]
  have hforce : lam * (1 - r ^ 2) - e ≤
      inner ℝ (u : EucSpace d) (proj d (Y 1 : EucSpace d) (V (Y 0 : EucSpace d))) := by
    dsimp [r, e]
    linarith
  have hamul := mul_le_mul_of_nonneg_left hforce ha0
  have he := mul_le_mul_of_nonneg_right ha1 herr
  have hs := mul_nonneg hb0 hself
  nlinarith

/-- The actual second-token overlap has the actual original projected
field as its derivative. Source: arXiv:2411.04990v2, §2, `eqn:csa`,
and §4, the scalar argument for `thm2`. -/
theorem hasDerivAt_csaSecond_fieldOverlap (β : ℝ) (Q K V : ParamMatrix d)
    (X : ℝ → SphereTuple d 2) (hX : CSA d 2 β Q K V X) (u : EucSpace d) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ u (X s 1 : EucSpace d))
      (inner ℝ u (csaAmbientField β Q K V (fun j => (X t j : EucSpace d)) 1)) t := by
  simpa! only [inner_zero_left, add_zero] using
    (hasDerivAt_const t u).inner ℝ ((CSA_iff_csaAmbientField β Q K V X).mp hX t 1)

/-- Eventual nonnegative leading overlap suffices for convergence of
the actual second token, under the original spectral assumptions and
the actual leader limit. Source: arXiv:2411.04990v2, §4, `thm2`, for
two original tokens; the scalar escape argument follows `thm1`. -/
theorem csaSecond_tendsto_of_eventually_nonneg_leading (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (X : ℝ → SphereTuple d 2) (hX : CSA d 2 β Q K V X)
    (hleader : Tendsto (fun t => (X t 0 : EucSpace d)) atTop (nhds (u : EucSpace d)))
    (hpos : ∀ᶠ t in atTop, 0 ≤ inner ℝ (u : EucSpace d) (X t 1 : EucSpace d)) :
    Tendsto (fun t => (X t 1 : EucSpace d)) atTop (nhds (u : EucSpace d)) := by
  let q : ℝ → ℝ := fun t => inner ℝ (u : EucSpace d) (X t 1 : EucSpace d)
  let q' : ℝ → ℝ := fun t => inner ℝ (u : EucSpace d)
    (csaAmbientField β Q K V (fun j => (X t j : EucSpace d)) 1)
  let A := lam * csaWeightFloor β Q K 2
  let e : ℝ → ℝ := fun t => ‖V‖ * ‖(X t 0 : EucSpace d) - (u : EucSpace d)‖
  have hA : 0 < A := mul_pos hlam (csaWeightFloor_pos β Q K (by norm_num : 0 < 2))
  have hq (t : ℝ) : HasDerivAt q (q' t) t :=
    hasDerivAt_csaSecond_fieldOverlap β Q K V X hX u t
  have hbound (t : ℝ) : -1 ≤ q t ∧ q t ≤ 1 := csaSphereOverlap_bounds u (X t 1)
  have he : Tendsto e atTop (nhds 0) := by
    have h := ((hleader.sub_const (u : EucSpace d)).norm).const_mul ‖V‖
    simpa only [sub_self, norm_zero, mul_zero] using h
  have hlim : Tendsto q atTop (nhds 1) := by
    apply tendsto_order.mpr
    constructor
    · intro a ha
      obtain ⟨b, hb, hb1⟩ := exists_between (max_lt (by norm_num : (0 : ℝ) < 1) ha)
      have hb0 : 0 ≤ b := (le_max_left 0 a).trans hb.le
      have hgap : 0 < 1 - b ^ 2 := by nlinarith
      let c := A * (1 - b ^ 2) / 2
      have hc : 0 < c := div_pos (mul_pos hA hgap) (by norm_num)
      obtain ⟨T, hT⟩ := eventually_atTop.mp (he.eventually (gt_mem_nhds hc))
      obtain ⟨s, hs⟩ := eventually_atTop.mp hpos
      have hdr (t : ℝ) (ht : T ≤ t) (h0 : 0 ≤ q t) (hbq : q t ≤ b) : c ≤ q' t := by
        have hfield := csaSecond_leading_field_lower β Q K V L lam hlam.le heig hinv
          hdecay u hu (X t) h0
        have hsq : q t ^ 2 ≤ b ^ 2 := by nlinarith
        have hmul := mul_le_mul_of_nonneg_left (sub_le_sub_left hsq 1) hA.le
        have herr := hT t ht
        change A * (1 - q t ^ 2) - e t ≤ q' t at hfield
        dsimp [c] at herr ⊢
        linarith
      have htop := csaScalar_escape_strip q q' hq hbound 0 b T c (max T s) hb0 hc
        (le_max_left _ _) (hs _ (le_max_right _ _)) hdr
      filter_upwards [htop] with t ht
      exact ((le_max_right 0 a).trans_lt hb).trans_le ht
    · intro a ha
      exact Eventually.of_forall (fun t => (hbound t).2.trans_lt ha)
  have hsq : Tendsto (fun t => ‖(X t 1 : EucSpace d) - (u : EucSpace d)‖ ^ 2)
      atTop (nhds 0) := by
    have h : Tendsto (fun t => 2 * (1 - q t)) atTop (nhds (2 * (1 - 1 : ℝ))) :=
      (tendsto_const_nhds.sub hlim).const_mul (2 : ℝ)
    simpa only [norm_sub_rev, csaSphere_dist_sq, q, sub_self, mul_zero] using h
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  simpa only [Function.comp_def, Real.sqrt_sq_eq_abs, abs_norm, Real.sqrt_zero] using
    Real.continuous_sqrt.continuousAt.tendsto.comp hsq

/-- Actual nonzero consensus satisfies the spectral, leader-limit and
positive-overlap premises together. Source: arXiv:2411.04990v2, §4,
`thm2`, the identity-value case with the full leading plane. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    ∃ (u : SSphere 2) (X : ℝ → SphereTuple 2 2),
      (0 : ℝ) < 1 ∧
      (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
        (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
        (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
        inner ℝ ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < (1 : ℝ) * ‖z‖ ^ 2) ∧
      (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
      CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) X ∧
      Tendsto (fun t => (X t 0 : EucSpace 2)) atTop (nhds (u : EucSpace 2)) ∧
      (∀ᶠ t in atTop, 0 ≤ inner ℝ (u : EucSpace 2) (X t 1 : EucSpace 2)) := by
  let u : SSphere 2 := basePoint 1
  refine ⟨u, fun _ _ => u, one_pos, fun v _ => by simp, fun z hz => hz, ?_,
    Submodule.mem_top, CSA_const_consensus β Q K u, tendsto_const_nhds, ?_⟩
  · intro z hz hz0
    exact (hz0 (by simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim
  · exact Eventually.of_forall fun _ => by rw [real_inner_self_eq_norm_sq]; positivity

end Transformer.Causal
