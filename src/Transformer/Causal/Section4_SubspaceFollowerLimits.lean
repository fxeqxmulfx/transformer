module

public import Transformer.Causal.Section4_SubspaceFollowerDrift
public import Transformer.Causal.Section4_PositiveReference

/-!
# A converging original prefix attracts its original positive follower

ArXiv:2411.04990v2, §4, a leading-hemisphere case of `thm2`.
The original spectral assumptions give the leading Rayleigh ceiling.
Together with actual prefix convergence this forces positive drift on
every closed nonnegative overlap strip below one. The original follower
therefore converges to the same leading limit once its overlap is
eventually nonnegative. A common positive initial leading covector
supplies this entry event by the preceding original scalar equations.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- Eventual nonnegative overlap and the actual original prefix limit
force the actual original last token to converge, for every nonempty
prefix. Source: arXiv:2411.04990v2, §4, the positive leading-hemisphere
case of `thm2`, using the scalar escape argument of `thm1`. -/
theorem csaLast_tendsto_of_eventually_nonneg_leading (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X)
    (hn : 1 ≤ n)
    (hprefix : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d)))
    (hpos : ∀ᶠ t in atTop, 0 ≤ inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d)) :
    Tendsto (fun t => (X t (Fin.last n) : EucSpace d))
      atTop (nhds (u : EucSpace d)) := by
  let q : ℝ → ℝ := fun t =>
    inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d)
  let q' : ℝ → ℝ := fun t => inner ℝ (u : EucSpace d)
    (csaAmbientField β Q K V (fun j => (X t j : EucSpace d)) (Fin.last n))
  let A := lam * csaWeightFloor β Q K (n + 1)
  let e : ℝ → ℝ := fun t =>
    ‖V‖ * ∑ j : Idx n, ‖(X t j.castSucc : EucSpace d) - (u : EucSpace d)‖
  have hA : 0 < A :=
    mul_pos hlam (csaWeightFloor_pos β Q K (Nat.zero_lt_succ n))
  have hq (t : ℝ) : HasDerivAt q (q' t) t :=
    hasDerivAt_csaLast_fieldOverlap β Q K V X hX u t
  have hbound (t : ℝ) : -1 ≤ q t ∧ q t ≤ 1 :=
    csaSphereOverlap_bounds u (X t (Fin.last n))
  have he : Tendsto e atTop (nhds 0) := by
    simpa only [mul_zero] using (tendsto_csaLeadingPrefixError X u hprefix).const_mul ‖V‖
  have hlim : Tendsto q atTop (nhds 1) := by
    apply tendsto_order.mpr
    constructor
    · intro a ha
      obtain ⟨b, hb, hb1⟩ := exists_between (max_lt (by norm_num : (0 : ℝ) < 1) ha)
      have hb0 : 0 ≤ b := (le_max_left 0 a).trans hb.le
      have hgap : 0 < 1 - b ^ 2 := by nlinarith
      let c := A * (1 - b ^ 2) / 2
      have hc : 0 < c :=
        div_pos (mul_pos hA hgap) (by norm_num)
      obtain ⟨T, hT⟩ :=
        eventually_atTop.mp (he.eventually (gt_mem_nhds hc))
      obtain ⟨s, hs⟩ := eventually_atTop.mp hpos
      have hdr (t : ℝ) (ht : T ≤ t) (h0 : 0 ≤ q t) (hbq : q t ≤ b) :
          c ≤ q' t := by
        have hfield := csaLastLeadingOverlap_lower β Q K V L lam hlam.le
          heig hinv hdecay u hu (X t) hn h0
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
  have hsq : Tendsto (fun t => ‖(X t (Fin.last n) : EucSpace d) - (u : EucSpace d)‖ ^ 2)
      atTop (nhds 0) := by
    have h : Tendsto (fun t => 2 * (1 - q t)) atTop (nhds (2 * (1 - 1 : ℝ))) :=
      (tendsto_const_nhds.sub hlim).const_mul (2 : ℝ)
    simpa only [norm_sub_rev, csaSphere_dist_sq, q, sub_self, mul_zero] using h
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  simpa only [Function.comp_def, Real.sqrt_sq_eq_abs, abs_norm, Real.sqrt_zero] using
    Real.continuous_sqrt.continuousAt.tendsto.comp hsq

/-- A common positive original leading covector makes the original
last token inherit its original prefix's leading limit. Neither entry
nor follower convergence is assumed. Source: arXiv:2411.04990v2, §4,
a pointwise leading-hemisphere case of `thm2`. -/
theorem csaLast_tendsto_of_positive_leading_covector (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (w : EucSpace d) (hw : w ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (hn : 1 ≤ n)
    (hpos : ∀ i, 0 < inner ℝ w (X 0 i : EucSpace d))
    (hprefix : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d))) :
    Tendsto (fun t => (X t (Fin.last n) : EucSpace d))
      atTop (nhds (u : EucSpace d)) :=
  csaLast_tendsto_of_eventually_nonneg_leading β Q K V L lam hlam
    heig hinv hdecay u hu X hX hn hprefix
    (csaLast_eventually_nonneg_leading_of_positive_covector β Q K V L lam hlam
      heig hinv u hu w hw X hX hn hpos hprefix)

/-- Actual consensus realizes the full spectral, common positive
covector, original prefix and positive-overlap premises jointly.
Source: arXiv:2411.04990v2, §4, the leading consensus in `thm2`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    ∃ (u : SSphere 2) (X : ℝ → SphereTuple 2 2),
      (0 : ℝ) < 1 ∧ (1 : ℕ) ≤ 1 ∧
      (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
        (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
        (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
        inner ℝ ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < (1 : ℝ) * ‖z‖ ^ 2) ∧
      (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
      CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) X ∧
      (∀ i, 0 < inner ℝ (u : EucSpace 2) (X 0 i : EucSpace 2)) ∧
      (∀ j : Idx 1, Tendsto (fun t => (X t j.castSucc : EucSpace 2)) atTop
        (nhds (u : EucSpace 2))) ∧
      (∀ᶠ t in atTop, 0 ≤ inner ℝ (u : EucSpace 2) (X t (1 : Idx 2) : EucSpace 2)) := by
  let u : SSphere 2 := basePoint 1
  refine ⟨u, fun _ _ => u, one_pos, le_rfl, fun v _ => by simp, fun z hz => hz, ?_,
    Submodule.mem_top, CSA_const_consensus β Q K u, ?_, fun _ => tendsto_const_nhds, ?_⟩
  · intro z hz hz0
    exact (hz0 (by
      simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim
  · intro i
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp u.2]
    norm_num
  · exact Eventually.of_forall fun _ => by
      rw [real_inner_self_eq_norm_sq]
      positivity

end Transformer.Causal
