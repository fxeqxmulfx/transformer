module

public import Transformer.Causal.Section4_ExpandingScalar
public import Transformer.Causal.Section4_LeadingForcing
public import Transformer.Causal.Section5_StrongCenterWitness

/-!
# Vanishing transverse components on the negative leading branch

ArXiv:2411.04990v2, §4, a reduction for the unproved case of `thm2`.
Longitudinal and transverse leading components have the same actual
diagonal coefficient. If the longitudinal component stays uniformly
negative, their quotient has a uniformly positive coefficient. The
transverse forcing tends to zero when the original prefix converges.
Boundedness therefore forces the transverse component to tend to zero.
This is a trajectory statement; it does not assert nullity of its basin.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- A bounded transverse component with vanishing forcing tends to
zero when its longitudinal companion stays uniformly negative with
uniformly positive forcing. The common actual radial coefficient
cancels in the quotient. Source: arXiv:2411.04990v2, §4, the leading
component equations of `thm2`; no follower limit is assumed. -/
theorem tendsto_transverse_of_negative_companion (h f g r q : ℝ → ℝ)
    (hr : ∀ t, HasDerivAt r (g t + h t * r t) t)
    (hq : ∀ t, HasDerivAt q (f t + h t * q t) t)
    (M c δ T : ℝ) (hc : 0 < c) (hδ : 0 < δ)
    (hrbound : ∀ t, T ≤ t → |r t| ≤ M)
    (hqbound : ∀ t, T ≤ t → -1 ≤ q t ∧ q t ≤ -δ)
    (hf : ∀ t, T ≤ t → c ≤ f t) (hg : Tendsto g atTop (nhds 0)) :
    Tendsto r atTop (nhds 0) := by
  let z : ℝ → ℝ := fun t => r t / q t
  let a : ℝ → ℝ := fun t => -f t / q t
  let b : ℝ → ℝ := fun t => g t / q t
  have hM : 0 ≤ M := (abs_nonneg (r T)).trans (hrbound T le_rfl)
  have hneg (t : ℝ) (ht : T ≤ t) : q t < 0 := by
    linarith [(hqbound t ht).2]
  have habs (t : ℝ) (ht : T ≤ t) : δ ≤ |q t| := by
    rw [abs_of_neg (hneg t ht)]
    linarith [(hqbound t ht).2]
  have hz (t : ℝ) (ht : T ≤ t) : |z t| ≤ M / δ := by
    calc
      |z t| = |r t| / |q t| := abs_div _ _
      _ ≤ M / |q t| := div_le_div_of_nonneg_right (hrbound t ht) (abs_nonneg _)
      _ ≤ M / δ := div_le_div_of_nonneg_left hM hδ (habs t ht)
  have hb : Tendsto b atTop (nhds 0) := by
    apply squeeze_zero_norm' ((eventually_ge_atTop T).mono fun t ht => ?_)
      (show Tendsto (fun t => |g t| / δ) atTop (nhds 0) from by
        simpa only [abs_zero, zero_div] using hg.abs.div_const δ)
    change |g t / q t| ≤ |g t| / δ
    rw [abs_div]
    exact div_le_div_of_nonneg_left (abs_nonneg _) hδ (habs t ht)
  have ha (t : ℝ) (ht : T ≤ t) : c ≤ a t := by
    have he : a t = f t / (-q t) := by dsimp [a]; ring
    rw [he]
    apply (le_div_iff₀ (neg_pos.mpr (hneg t ht))).mpr
    nlinarith [(hqbound t ht).1, hf t ht]
  have hd (t : ℝ) (ht : T ≤ t) : HasDerivAt z (a t * z t + b t) t := by
    convert (hr t).div (hq t) (ne_of_lt (hneg t ht)) using 1
    dsimp [a, b, z]
    field_simp
    ring
  have hzlim : Tendsto z atTop (nhds 0) :=
    tendsto_zero_of_bounded_expanding a b z (M / δ) c T hd hz hc ha hb
  apply squeeze_zero_norm' ((eventually_ge_atTop T).mono fun t ht => ?_)
    (show Tendsto (fun t => |z t|) atTop (nhds 0) from by
      simpa only [abs_zero] using hzlim.abs)
  have hqabs : |q t| ≤ 1 := by
    rw [abs_of_neg (hneg t ht)]
    linarith [(hqbound t ht).1]
  calc
    ‖r t‖ = |z t * q t| := by
      change ‖r t‖ = |r t / q t * q t|
      rw [div_mul_cancel₀ _ (ne_of_lt (hneg t ht)), Real.norm_eq_abs]
    _ = |z t| * |q t| := abs_mul _ _
    _ ≤ |z t| := by nlinarith [abs_nonneg (z t)]

/-- Every original leading component orthogonal to the prefix limit
tends to zero if the actual last longitudinal overlap stays uniformly
negative. No bound on the token count or hemisphere initialization is
imposed. Source: arXiv:2411.04990v2, §4, a necessary condition for an
exceptional trajectory in `thm2`; this is not a measure-zero assertion. -/
theorem csaLast_transverse_tendsto_of_uniformly_negative (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (v : EucSpace d) (hv : v ∈ L) (hvu : inner ℝ v u = 0)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d)))
    (δ : ℝ) (hδ : 0 < δ)
    (hq : ∀ᶠ t in atTop, inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d) ≤ -δ) :
    Tendsto (fun t => inner ℝ v (X t (Fin.last n) : EucSpace d)) atTop (nhds 0) := by
  obtain ⟨c, hc, hf⟩ := eventually_csaLastLeadingForcing_pos β Q K lam hlam X hn u hp
  obtain ⟨T, hT⟩ := eventually_atTop.mp (hf.and hq)
  apply tendsto_transverse_of_negative_companion (csaLastLeadingCoefficient β Q K V lam X)
    (csaLastLeadingForcing β Q K lam X u) (csaLastLeadingForcing β Q K lam X v)
    (fun t => inner ℝ v (X t (Fin.last n) : EucSpace d))
    (fun t => inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d))
    (hasDerivAt_csaLast_leading_forcing β Q K V L lam heig hinv v hv X hX)
    (hasDerivAt_csaLast_leading_forcing β Q K V L lam heig hinv _ hu X hX)
    ‖v‖ c δ T hc hδ ?_ ?_ (fun t ht => (hT t ht).1)
    (tendsto_csaLastLeadingForcing_zero β Q K lam X u v hvu hp)
  · intro t _
    simpa only [mem_sphere_zero_iff_norm.mp (X t (Fin.last n)).2, mul_one] using
      abs_real_inner_le_norm v (X t (Fin.last n) : EucSpace d)
  · intro t ht
    exact ⟨(csaSphereOverlap_bounds u (X t (Fin.last n))).1, (hT t ht).2⟩

/-- An exponentially decaying, initially nonzero transverse component
and a constant negative longitudinal companion satisfy every premise
jointly. The vanishing transverse forcing is not assumed to be zero.
Source: arXiv:2411.04990v2, §4, the negative-branch scalar reduction
of `thm2`. This example distinguishes convergence from exact vanishing. -/
example : ∃ h f g r q : ℝ → ℝ,
    (∀ t, HasDerivAt r (g t + h t * r t) t) ∧
    (∀ t, HasDerivAt q (f t + h t * q t) t) ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (∀ t, (0 : ℝ) ≤ t → |r t| ≤ 1) ∧
    (∀ t, (0 : ℝ) ≤ t → -1 ≤ q t ∧ q t ≤ -1) ∧
    (∀ t, (0 : ℝ) ≤ t → 1 ≤ f t) ∧
    Tendsto g atTop (nhds 0) ∧ r 0 ≠ 0 := by
  let r : ℝ → ℝ := fun t => Real.exp (-t)
  have hr (t : ℝ) : HasDerivAt r (-r t) t := by
    convert (hasDerivAt_id t).neg.exp using 1 <;> simp [r]
  have hlim : Tendsto r atTop (nhds 0) :=
    Real.tendsto_exp_atBot.comp tendsto_neg_atTop_atBot
  refine ⟨fun _ => 1, fun _ => 1, fun t => -2 * r t, r, fun _ => -1, ?_,
    fun t => by simpa using hasDerivAt_const t (-1 : ℝ), one_pos, one_pos,
    ?_, fun _ _ => ⟨le_rfl, le_rfl⟩, fun _ _ => le_rfl, ?_, ?_⟩
  · intro t
    convert hr t using 1
    ring
  · intro t ht
    rw [abs_of_pos (Real.exp_pos _)]
    exact Real.exp_le_one_iff.mpr (by linarith)
  · simpa only [mul_zero] using hlim.const_mul (-2 : ℝ)
  · simp [r]

/-- The actual stationary antipodal pair realizes every negative-branch
premise, with a nonzero leading covector orthogonal to the leader.
Source: arXiv:2411.04990v2, §4, the antipodal exception of `thm2`. -/
example : ∃ (X : ℝ → SphereTuple 2 2) (u : SSphere 2) (v : EucSpace 2),
    CSA 2 2 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) X ∧ (0 : ℝ) < 1 ∧ (1 : ℕ) ≤ 1 ∧
    (∀ w ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) w = (1 : ℝ) • w) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    v ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧ inner ℝ v u = 0 ∧
    (∀ j : Idx 1, Tendsto (fun t => (X t j.castSucc : EucSpace 2)) atTop
      (nhds (u : EucSpace 2))) ∧
    (∀ᶠ t in atTop, inner ℝ (u : EucSpace 2) (X t (Fin.last 1) : EucSpace 2) ≤ -(1 : ℝ)) := by
  let u : SSphere 2 := basePoint 1
  let v : EucSpace 2 := EuclideanSpace.single (1 : Fin 2) 1
  refine ⟨fun _ => antipodalCircleTuple u, u, v, csa_const_antipodal 1 u, one_pos, le_rfl,
    fun w _ => by simp, fun z hz => hz, Submodule.mem_top, Submodule.mem_top,
    ?_, ?_, Eventually.of_forall fun _ => ?_⟩
  · simp [u, v, basePoint, EuclideanSpace.inner_single_left]
  · intro j
    have hj : j = 0 := by ext; omega
    simpa only [hj, Fin.castSucc_zero, antipodalCircleTuple, Matrix.cons_val_zero] using
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => (u : EucSpace 2)) atTop (nhds (u : EucSpace 2)))
  · simp [antipodalCircleTuple, mem_sphere_zero_iff_norm.mp u.2]

end Transformer.Causal
