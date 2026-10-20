module

public import Transformer.Causal.Section4_LeadingComponents

/-!
# Entry into the leading positive hemisphere for any original follower

ArXiv:2411.04990v2, §4, a leading-hemisphere case of `thm2`.
A positive original leading covector gives a positive bounded reference
component. Its actual forcing need not vanish: it is nonnegative by
causality and the proved sign preservation. While the longitudinal
overlap is negative, dividing by this reference produces a positive
derivative. The actual converging prefix therefore forces entry into
the positive hemisphere. No follower limit is assumed.
-/

@[expose] public section

open Real Set Filter
open scoped Topology BigOperators

namespace Transformer.Causal

variable {d n : ℕ}

/-- Nonnegative reference forcing suffices for positive longitudinal
entry; it need not vanish as it did for the transverse two-token
reference. Source: arXiv:2411.04990v2, §4, the scalar comparison for
`thm2` under the explicit positive leading-covector condition. -/
theorem eventually_nonneg_of_positive_reference (h f g r q : ℝ → ℝ)
    (hr : ∀ t, HasDerivAt r (g t + h t * r t) t)
    (hq : ∀ t, HasDerivAt q (f t + h t * q t) t)
    (hrpos : ∀ t, 0 ≤ t → 0 < r t) (M : ℝ) (hM : 0 < M)
    (hrle : ∀ t, r t ≤ M) (hg : ∀ t, 0 ≤ t → 0 ≤ g t)
    (c T : ℝ) (hc : 0 < c) (hT0 : 0 ≤ T) (hf : ∀ t, T ≤ t → c ≤ f t) :
    ∀ᶠ t in atTop, 0 ≤ q t := by
  have hex : ∃ s, T ≤ s ∧ 0 ≤ q s := by
    by_contra hn
    push Not at hn
    let z : ℝ → ℝ := fun t => q t / r t
    let a := c / M
    have ha : 0 < a := div_pos hc hM
    have hz (t : ℝ) (ht : T ≤ t) : z t < 0 :=
      div_neg_of_neg_of_pos (hn t ht) (hrpos t (hT0.trans ht))
    have hd (t : ℝ) (ht : T ≤ t) : HasDerivAt z
        ((f t * r t - q t * g t) / r t ^ 2) t := by
      convert (hq t).div (hr t) (hrpos t (hT0.trans ht)).ne' using 1
      ring
    have hb (t : ℝ) (ht : T ≤ t) : a ≤ (f t * r t - q t * g t) / r t ^ 2 := by
      have hp := hrpos t (hT0.trans ht)
      have hlow : a ≤ f t / r t :=
        (div_le_div_of_nonneg_left hc.le hp (hrle t)).trans
          (div_le_div_of_nonneg_right (hf t ht) hp.le)
      apply hlow.trans
      apply (le_div_iff₀ (sq_pos_of_pos hp)).mpr
      have he : f t / r t * r t ^ 2 = f t * r t := by field_simp
      rw [he]
      have hqg := mul_nonpos_of_nonpos_of_nonneg (hn t ht).le (hg t (hT0.trans ht))
      linarith
    have hlin (u : ℝ) (hu : T ≤ u) : a * (u - T) ≤ z u - z T := by
      apply (convex_Ici T).mul_sub_le_image_sub_of_le_deriv
        (fun t ht => (hd t ht).continuousAt.continuousWithinAt)
        (fun t ht => (hd t (interior_subset ht)).differentiableAt.differentiableWithinAt) ?_
        T (by simp) u hu hu
      intro t ht
      rw [(hd t (interior_subset ht)).deriv]
      exact hb t (interior_subset ht)
    let u := T + (1 - z T) / a
    have hu : T ≤ u := by
      have hnum : 0 ≤ 1 - z T := by linarith [hz T le_rfl]
      exact le_add_of_nonneg_right (div_nonneg hnum ha.le)
    have he : a * (u - T) = 1 - z T := by dsimp [u]; field_simp; ring
    have h := hlin u hu
    rw [he] at h
    linarith [hz u hu]
  obtain ⟨s, hTs, hqs⟩ := hex
  apply eventually_atTop.mpr
  refine ⟨s, csaScalar_forward_barrier q (fun t => f t + h t * q t) hq 0 T s hTs hqs ?_⟩
  intro t ht he
  rw [he, mul_zero, add_zero]
  exact hc.trans_le (hf t ht)

/-- A common positive original leading covector and a genuinely
converging original prefix force the actual last token into the
leading nonnegative hemisphere. Source: arXiv:2411.04990v2, §4,
a pointwise leading-hemisphere case of `thm2`. -/
theorem csaLast_eventually_nonneg_leading_of_positive_covector (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L) (w : EucSpace d) (hw : w ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (hn : 1 ≤ n)
    (hpos : ∀ i, 0 < inner ℝ w (X 0 i : EucSpace d))
    (hprefix : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d))) :
    ∀ᶠ t in atTop, 0 ≤ inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d) := by
  let r : ℝ → ℝ := fun t => inner ℝ w (X t (Fin.last n) : EucSpace d)
  let q : ℝ → ℝ := fun t => inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d)
  let h := csaLastLeadingCoefficient β Q K V lam X
  let f : ℝ → ℝ := fun t => lam * ∑ j : Idx n,
    csaTokenWeight β Q K (fun k => (X t k : EucSpace d)) (Fin.last n) j.castSucc *
      inner ℝ (u : EucSpace d) (X t j.castSucc : EucSpace d)
  let g : ℝ → ℝ := fun t => lam * ∑ j : Idx n,
    csaTokenWeight β Q K (fun k => (X t k : EucSpace d)) (Fin.last n) j.castSucc *
      inner ℝ w (X t j.castSucc : EucSpace d)
  have hsign := csaLeading_pos_of_initial_pos β Q K V L lam hlam.le heig hinv w hw _ X hX hpos
  have hrpos (t : ℝ) (ht : 0 ≤ t) : 0 < r t := hsign _ t ht
  have hrle (t : ℝ) : r t ≤ ‖w‖ := by
    calc
      r t ≤ |inner ℝ w (X t (Fin.last n) : EucSpace d)| := le_abs_self _
      _ ≤ ‖w‖ * ‖(X t (Fin.last n) : EucSpace d)‖ := abs_real_inner_le_norm _ _
      _ = ‖w‖ := by rw [mem_sphere_zero_iff_norm.mp (X t (Fin.last n)).2, mul_one]
  have hg (t : ℝ) (ht : 0 ≤ t) : 0 ≤ g t := mul_nonneg hlam.le
    (Finset.sum_nonneg fun j _ => mul_nonneg (csaTokenWeight_nonneg β Q K _ _ _) (hsign _ t ht).le)
  have hhalf : ∀ᶠ t in atTop, ∀ j : Idx n,
      (1 / 2 : ℝ) ≤ inner ℝ (u : EucSpace d) (X t j.castSucc : EucSpace d) := by
    rw [Filter.eventually_all]
    intro j
    have hj : Tendsto (fun t => inner ℝ (u : EucSpace d) (X t j.castSucc : EucSpace d))
        atTop (nhds (inner ℝ (u : EucSpace d) u)) := tendsto_const_nhds.inner (hprefix j)
    have hj' : Tendsto (fun t => inner ℝ (u : EucSpace d) (X t j.castSucc : EucSpace d))
        atTop (nhds 1) := by
      simpa only [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp u.2, one_pow] using hj
    exact (hj'.eventually (lt_mem_nhds (by norm_num : (1 / 2 : ℝ) < 1))).mono fun _ h => h.le
  obtain ⟨T, hT⟩ := eventually_atTop.mp hhalf
  let c := lam * csaWeightFloor β Q K (n + 1) / 2
  have hc : 0 < c := div_pos (mul_pos hlam (csaWeightFloor_pos β Q K (Nat.zero_lt_succ n))) (by norm_num)
  have hf (t : ℝ) (ht : max T 0 ≤ t) : c ≤ f t := by
    let j : Idx n := ⟨0, hn⟩
    have hh := hT t ((le_max_left _ _).trans ht)
    have hwf := csaWeightFloor_le β Q K (fun k => (X t k : EucSpace d))
      (fun k => mem_sphere_zero_iff_norm.mp (X t k).2) (Fin.last n) j.castSucc (by simp [j])
    have hsum := Finset.single_le_sum (fun k _ => mul_nonneg
      (csaTokenWeight_nonneg β Q K (fun l => (X t l : EucSpace d)) (Fin.last n) k.castSucc)
        ((by norm_num : (0 : ℝ) ≤ 1 / 2).trans (hh k)))
      (Finset.mem_univ j)
    have hterm := mul_le_mul hwf (hh j) (by norm_num : (0 : ℝ) ≤ 1 / 2)
      (csaTokenWeight_nonneg β Q K (fun k => (X t k : EucSpace d)) (Fin.last n) j.castSucc)
    have hmul := mul_le_mul_of_nonneg_left (hterm.trans hsum) hlam.le
    dsimp [f, c]
    nlinarith
  exact eventually_nonneg_of_positive_reference h f g r q
    (hasDerivAt_csaLast_leading β Q K V L lam heig hinv w hw X hX)
    (hasDerivAt_csaLast_leading β Q K V L lam heig hinv _ hu X hX)
    hrpos ‖w‖ ((hrpos 0 le_rfl).trans_le (hrle 0)) hrle hg c (max T 0) hc (le_max_right _ _) hf

/-- Positive bounded reference and nonconstant longitudinal overlap
jointly realize every scalar comparison premise.
Source: arXiv:2411.04990v2, §4, the positive reference argument for `thm2`. -/
example : ∃ h f g r q : ℝ → ℝ,
    (∀ t, HasDerivAt r (g t + h t * r t) t) ∧
    (∀ t, HasDerivAt q (f t + h t * q t) t) ∧
    (∀ t, (0 : ℝ) ≤ t → 0 < r t) ∧ (0 : ℝ) < 1 ∧ (∀ t, r t ≤ 1) ∧
    (∀ t, (0 : ℝ) ≤ t → 0 ≤ g t) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧
    (∀ t, (0 : ℝ) ≤ t → 1 ≤ f t) := by
  exact ⟨fun _ => 0, fun _ => 1, fun _ => 0, fun _ => 1, id,
    fun t => by simpa using hasDerivAt_const t (1 : ℝ),
    fun t => by simpa using hasDerivAt_id t,
    fun _ _ => one_pos, one_pos, fun _ => le_rfl, fun _ _ => le_rfl,
    one_pos, le_rfl, fun _ _ => le_rfl⟩

/-- Actual three-token consensus jointly realizes every original
positive-covector and prefix-limit premise of the entry theorem.
Source: arXiv:2411.04990v2, §4, the leading-hemisphere case of `thm2`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    ∀ᶠ t in atTop, 0 ≤ inner ℝ (basePoint 1 : EucSpace 2)
      ((fun (_ : ℝ) (_ : Idx 3) => basePoint 1) t (Fin.last 2) : EucSpace 2) := by
  apply csaLast_eventually_nonneg_leading_of_positive_covector (n := 2)
    β Q K (ContinuousLinearMap.id ℝ _)
    ⊤ 1 one_pos (fun v _ => by simp) (fun z hz => hz) (basePoint 1) Submodule.mem_top
    (basePoint 1) Submodule.mem_top (fun _ _ => basePoint 1)
    (CSA_const_consensus β Q K (basePoint 1)) (by norm_num)
  · intro i
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (basePoint 1).2]
    norm_num
  · exact fun _ => tendsto_const_nhds

end Transformer.Causal
