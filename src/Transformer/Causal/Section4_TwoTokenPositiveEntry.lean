module

public import Transformer.Causal.Section4_TwoTokenLeading

/-!
# The original follower enters the leading positive hemisphere

ArXiv:2411.04990v2, §4, `thm2`, restricted to two tokens.
An original leading covector transverse to the initial leader has a
positive homogeneous follower component. Its ratio with the original
longitudinal component cancels the actual nonlinear radial coefficient.
The positive original leader weight and the proved leader limit force
this ratio to grow, hence the actual follower's longitudinal overlap
becomes positive. This step does not assume a follower limit or a
positive-overlap event.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d : ℕ}

/-- A bounded positive denominator and a uniformly positive forcing make
the actual longitudinal/homogeneous ratio grow linearly on a tail.
Source: arXiv:2411.04990v2, §4, `thm2`, the two-token quotient calculation. -/
theorem quotient_tendsto_atTop_of_positive_forcing (h f r q : ℝ → ℝ)
    (hr : ∀ t, HasDerivAt r (h t * r t) t)
    (hq : ∀ t, HasDerivAt q (f t + h t * q t) t)
    (hrpos : ∀ t, 0 < r t) (M : ℝ) (hM : 0 < M) (hrle : ∀ t, r t ≤ M)
    (c T : ℝ) (hc : 0 < c) (hf : ∀ t, T ≤ t → c ≤ f t) :
    Tendsto (fun t => q t / r t) atTop atTop := by
  have hd (t : ℝ) : HasDerivAt (fun s => q s / r s) (f t / r t) t := by
    convert (hq t).div (hr t) (hrpos t).ne' using 1
    field_simp
    ring
  have hbound (t : ℝ) (ht : T ≤ t) : c / M ≤ f t / r t :=
    (div_le_div_of_nonneg_left hc.le (hrpos t) (hrle t)).trans
      (div_le_div_of_nonneg_right (hf t ht) (hrpos t).le)
  have hg := csaScalar_linear_growth (fun t => q t / r t) (fun t => f t / r t) hd
    (c / M) T hbound
  have hlin : Tendsto (fun t : ℝ => c / M * (t - T) + q T / r T) atTop atTop := by
    have hsub : Tendsto (fun t : ℝ => t - T) atTop atTop := by
      simpa only [sub_eq_add_neg, id_eq] using tendsto_atTop_add_const_right atTop (-T) tendsto_id
    exact tendsto_atTop_add_const_right atTop (q T / r T) (hsub.const_mul_atTop (div_pos hc hM))
  apply tendsto_atTop_mono' atTop
    ((eventually_ge_atTop T).mono fun t ht => ?_)
    hlin
  linarith [hg t ht]

/-- A positive denominator turns eventual positivity of the quotient into
eventual positivity of the original longitudinal component.
Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem eventually_pos_of_quotient_tendsto_atTop (q r : ℝ → ℝ)
    (hr : ∀ t, 0 < r t) (hlim : Tendsto (fun t => q t / r t) atTop atTop) :
    ∀ᶠ t in atTop, 0 < q t := by
  filter_upwards [hlim.eventually_gt_atTop 0] with t ht
  exact (div_pos_iff_of_pos_right (hr t)).mp ht

/-- A genuine nonzero transverse leading component forces eventual
positive longitudinal overlap for the actual second token. Only the
leader's limit, as supplied by `lemma1`, is an input; no follower limit
or entry assumption is used. Source: arXiv:2411.04990v2, §4, `thm2`,
restricted to two original tokens. -/
theorem csaSecond_eventually_pos_leading_of_transverse (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (v : EucSpace d) (hv : v ∈ L)
    (X : ℝ → SphereTuple d 2) (hX : CSA d 2 β Q K V X)
    (hzero : inner ℝ v (X 0 0 : EucSpace d) = 0)
    (hpos : 0 < inner ℝ v (X 0 1 : EucSpace d))
    (hleader : Tendsto (fun t => (X t 0 : EucSpace d)) atTop (nhds (u : EucSpace d))) :
    ∀ᶠ t in atTop, 0 < inner ℝ (u : EucSpace d) (X t 1 : EucSpace d) := by
  let r : ℝ → ℝ := fun t => inner ℝ v (X t 1 : EucSpace d)
  let q : ℝ → ℝ := fun t => inner ℝ (u : EucSpace d) (X t 1 : EucSpace d)
  let h := csaLeadingCoefficient β Q K V lam X
  let f : ℝ → ℝ := fun t => lam * csaTokenWeight β Q K (fun j => (X t j : EucSpace d)) 1 0 *
    inner ℝ (u : EucSpace d) (X t 0 : EucSpace d)
  have hr (t : ℝ) : HasDerivAt r (h t * r t) t := by
    simpa only [csaLeader_leading_orthogonal β Q K V L lam heig hinv v hv X hX hzero t,
      mul_zero, zero_add] using hasDerivAt_csaSecond_leading β Q K V L lam heig hinv v hv X hX t
  have hq (t : ℝ) : HasDerivAt q (f t + h t * q t) t :=
    hasDerivAt_csaSecond_leading β Q K V L lam heig hinv _ hu X hX t
  have hrpos (t : ℝ) : 0 < r t := csaSecond_pos_transverse_leading β Q K V L lam
    heig hinv v hv X hX hzero hpos t
  have hrle (t : ℝ) : r t ≤ ‖v‖ := by
    calc
      r t ≤ |inner ℝ v (X t 1 : EucSpace d)| := le_abs_self _
      _ ≤ ‖v‖ * ‖(X t 1 : EucSpace d)‖ := abs_real_inner_le_norm _ _
      _ = ‖v‖ := by rw [mem_sphere_zero_iff_norm.mp (X t 1).2, mul_one]
  have hM : 0 < ‖v‖ := (hrpos 0).trans_le (hrle 0)
  have hl : Tendsto (fun t => inner ℝ (u : EucSpace d) (X t 0 : EucSpace d)) atTop (nhds 1) := by
    have h : Tendsto (fun t => inner ℝ (u : EucSpace d) (X t 0 : EucSpace d)) atTop
        (nhds (inner ℝ (u : EucSpace d) u)) :=
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => (u : EucSpace d)) atTop
        (nhds (u : EucSpace d))).inner hleader
    simpa only [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp u.2, one_pow] using h
  obtain ⟨T, hT⟩ := eventually_atTop.mp (hl.eventually (lt_mem_nhds (by norm_num : (1 / 2 : ℝ) < 1)))
  let c := lam * csaWeightFloor β Q K 2 / 2
  have hc : 0 < c := div_pos (mul_pos hlam (csaWeightFloor_pos β Q K (by norm_num : 0 < 2))) (by norm_num)
  have hf (t : ℝ) (ht : T ≤ t) : c ≤ f t := by
    have hw := csaWeightFloor_le β Q K (fun j => (X t j : EucSpace d))
      (fun j => mem_sphere_zero_iff_norm.mp (X t j).2) (1 : Idx 2) 0 (by norm_num)
    have hw0 := csaTokenWeight_nonneg β Q K (fun j => (X t j : EucSpace d)) 1 0
    have hhalf := hT t ht
    calc
      c = (lam * csaWeightFloor β Q K 2) * (1 / 2) := by dsimp [c]; ring
      _ ≤ (lam * csaTokenWeight β Q K (fun j => (X t j : EucSpace d)) 1 0) *
          inner ℝ (u : EucSpace d) (X t 0 : EucSpace d) :=
        mul_le_mul (mul_le_mul_of_nonneg_left hw hlam.le) hhalf.le (by norm_num)
          (mul_nonneg hlam.le hw0)
      _ = f t := rfl
  exact eventually_pos_of_quotient_tendsto_atTop q r hrpos
    (quotient_tendsto_atTop_of_positive_forcing h f r q hr hq hrpos ‖v‖ hM hrle c T hc hf)

/-- A genuine orthogonal original pair with identity values realizes
the transverse initialization and the actual leader-limit hypothesis.
Source: arXiv:2411.04990v2, §4, `thm2`. -/
example : ∃ (u : SSphere 2) (v : EucSpace 2) (X : ℝ → SphereTuple 2 2),
    (0 : ℝ) < 1 ∧
    (∀ w ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) w = (1 : ℝ) • w) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    v ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    CSA 2 2 1 0 0 (ContinuousLinearMap.id ℝ _) X ∧
    inner ℝ v (X 0 0 : EucSpace 2) = 0 ∧ 0 < inner ℝ v (X 0 1 : EucSpace 2) ∧
    Tendsto (fun t => (X t 0 : EucSpace 2)) atTop (nhds (u : EucSpace 2)) := by
  let u : SSphere 2 := basePoint 1
  let v : SSphere 2 := ⟨EuclideanSpace.single (1 : Fin 2) 1, by simp⟩
  let X₀ : SphereTuple 2 2 := ![u, v]
  let X := csaFlow 1 0 0 (ContinuousLinearMap.id ℝ _) X₀
  refine ⟨u, v, X, one_pos, fun w _ => by simp, fun z hz => hz,
    Submodule.mem_top, Submodule.mem_top, (csaFlow_spec _ _ _ _ _).2, ?_, ?_, ?_⟩
  · simp [X, csaFlow_zero, X₀, u, v, basePoint, EuclideanSpace.inner_single_left]
  · simp [X, csaFlow_zero, X₀, v]
  · have he (t : ℝ) : (X t 0 : EucSpace 2) = (u : EucSpace 2) := by
      simp only [X, csaFlow_leader, X₀, Matrix.cons_val_zero]
    simpa only [he] using (tendsto_const_nhds : Tendsto (fun _ : ℝ => (u : EucSpace 2)) atTop
      (nhds (u : EucSpace 2)))

/-- A positive homogeneous component and its nonconstant forced companion
realize every integrating-factor, quotient, boundedness, and forcing
hypothesis jointly. Source: arXiv:2411.04990v2, §4, the two-token scalar argument. -/
example : ∃ (h f r q : ℝ → ℝ), Continuous h ∧
    (∀ t, HasDerivAt r (h t * r t) t) ∧
    (∀ t, HasDerivAt q (f t + h t * q t) t) ∧
    (∀ t, 0 < r t) ∧ (0 : ℝ) < 1 ∧ (∀ t, r t ≤ 1) ∧
    (0 : ℝ) < 1 ∧ (∀ t, (0 : ℝ) ≤ t → 1 ≤ f t) ∧
    Tendsto (fun t => q t / r t) atTop atTop := by
  refine ⟨fun _ => 0, fun _ => 1, fun _ => 1, id, continuous_const,
    fun t => by simpa using hasDerivAt_const t (1 : ℝ),
    fun t => by simpa using hasDerivAt_id t,
    fun _ => one_pos, one_pos, fun _ => le_rfl, one_pos, fun _ _ => le_rfl, ?_⟩
  convert! (tendsto_id : Tendsto (fun t : ℝ => t) atTop atTop) using 1
  ext t
  simp only [id_eq, div_one]

end Transformer.Causal
