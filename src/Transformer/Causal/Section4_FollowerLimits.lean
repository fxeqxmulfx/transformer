/-
# A last follower converges to the leader or its antipode

ArXiv:2411.04990v2, §4, proof of `thm1`.
Assume that every actual preceding token converges to the original
fixed leader. The actual original prefix error then vanishes.
On every closed overlap strip strictly inside `(-1,1)`, the actual
last overlap derivative is eventually bounded below by a positive
constant obtained from the original causal weight floor.

The proved scalar barrier theorem yields an endpoint overlap limit.
The exact original unit-sphere distance identity turns that into
actual convergence to either the original leader or its original
antipode. This deterministic dichotomy does not assume the
manuscript's unstable-equilibrium avoidance step; that separate
measure argument is needed to eliminate the antipodal alternative.
-/

import Transformer.Causal.Section4_FollowerDrift
import Transformer.Causal.Section4_ScalarLimits

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- The actual original last overlap has a uniform eventually positive drift on every fixed interior strip.
Source: arXiv:2411.04990v2, §4, the final scalar induction in `thm1`.
Its explicit positive constant follows from the original weight floor and actual vanishing prefix error. -/
theorem csaLastOverlap_positive_strip (β : ℝ) (Q K : ParamMatrix d)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X)
    (hn : 1 ≤ n) (hp : ∀ j : Idx n, Tendsto (fun t => X t j.castSucc) atTop (𝓝 (X 0 0)))
    (a b : ℝ) (ha : -1 < a) (hb : b < 1) :
    ∃ T c : ℝ, 0 < c ∧ ∀ t, T ≤ t →
      a ≤ inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d) →
      inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d) ≤ b →
      c ≤ deriv (fun s => inner ℝ (X s (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d)) t := by
  let c := csaWeightFloor β Q K (n + 1) * ((1 - b) * (1 + a)) / 2
  have hf := csaWeightFloor_pos β Q K (Nat.zero_lt_succ n)
  have hc : 0 < c := div_pos (mul_pos hf (mul_pos (by linarith) (by linarith))) (by norm_num)
  have he := tendsto_csaPrefixError X (X 0 0) hp
  obtain ⟨T, hT⟩ := eventually_atTop.mp (he.eventually_lt_const hc)
  refine ⟨T, c, hc, ?_⟩
  intro t ht hta htb
  have hr := csaSphereOverlap_bounds (X t (Fin.last n)) (X 0 0)
  have hq : (1 - b) * (1 + a) ≤
      1 - inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d) ^ 2 := by
    have hmul := mul_le_mul (by linarith :
      1 - b ≤ 1 - inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d))
      (by linarith : 1 + a ≤ 1 + inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d))
      (by linarith : 0 ≤ 1 + a) (by linarith :
        0 ≤ 1 - inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d))
    nlinarith
  have hm := mul_le_mul_of_nonneg_left hq hf.le
  have hl := csaLastOverlap_deriv_lower β Q K X hX hn t
  have herr := hT t ht
  dsimp [c] at herr ⊢
  linarith

/-- The original last overlap has an endpoint limit whenever the original preceding tokens converge to the leader.
Source: arXiv:2411.04990v2, §4, the deterministic scalar part of the induction for `thm1`. -/
theorem csaLastOverlap_tendsto_endpoints (β : ℝ) (Q K : ParamMatrix d)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X)
    (hn : 1 ≤ n) (hp : ∀ j : Idx n, Tendsto (fun t => X t j.castSucc) atTop (𝓝 (X 0 0))) :
    Tendsto (fun t => inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d)) atTop (𝓝 (-1)) ∨
      Tendsto (fun t => inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d)) atTop (𝓝 1) := by
  apply csaScalar_tendsto_endpoints _ (fun t => deriv
    (fun s => inner ℝ (X s (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d)) t)
    (fun t => (hasDerivAt_csaLastOverlap β Q K X hX t).differentiableAt.hasDerivAt)
    (fun t => csaSphereOverlap_bounds (X t (Fin.last n)) (X 0 0))
  intro a b ha hab hb
  exact csaLastOverlap_positive_strip β Q K X hX hn hp a b ha hb

/-- An actual original overlap tending to one gives convergence to its actual unit anchor.
Source: arXiv:2411.04990v2, §4, the last step of the overlap proof of `thm1`. -/
theorem csaSphere_tendsto_of_overlap_one (x : ℝ → SSphere d) (a : SSphere d)
    (h : Tendsto (fun t => inner ℝ (x t : EucSpace d) (a : EucSpace d)) atTop (𝓝 1)) :
    Tendsto x atTop (𝓝 a) := by
  apply csaSphere_tendsto_of_overlap
  simpa only [sub_self] using (tendsto_const_nhds : Tendsto (fun _ : ℝ => (1 : ℝ)) atTop (𝓝 1)).sub h

/-- An actual original overlap tending to minus one gives convergence to the actual antipodal unit anchor.
Source: arXiv:2411.04990v2, §4, the antipodal alternative in the proof of `thm1`. -/
theorem csaSphere_tendsto_of_overlap_neg_one (x : ℝ → SSphere d) (a : SSphere d)
    (h : Tendsto (fun t => inner ℝ (x t : EucSpace d) (a : EucSpace d)) atTop (𝓝 (-1))) :
    Tendsto x atTop (𝓝 (Perspective.antipode d a)) := by
  apply csaSphere_tendsto_of_overlap
  simpa only [Perspective.antipode, inner_neg_right, sub_neg_eq_add, add_neg_cancel] using h.const_add 1

/-- Every actual original last follower converges to the original leader or its antipode when its original prefix converges.
Source: arXiv:2411.04990v2, §4, the scalar convergence step of `thm1`. -/
theorem CSA_last_tendsto_leader_or_antipode (β : ℝ) (Q K : ParamMatrix d)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X)
    (hn : 1 ≤ n) (hp : ∀ j : Idx n, Tendsto (fun t => X t j.castSucc) atTop (𝓝 (X 0 0))) :
    Tendsto (fun t => X t (Fin.last n)) atTop (𝓝 (X 0 0)) ∨
      Tendsto (fun t => X t (Fin.last n)) atTop (𝓝 (Perspective.antipode d (X 0 0))) := by
  rcases csaLastOverlap_tendsto_endpoints β Q K X hX hn hp with h | h
  · exact Or.inr (csaSphere_tendsto_of_overlap_neg_one _ _ h)
  · exact Or.inl (csaSphere_tendsto_of_overlap_one _ _ h)

/-- Once the actual antipodal limit is excluded, the original last follower necessarily converges to its original leader.
Source: arXiv:2411.04990v2, §4, the final combination of instability avoidance and scalar convergence in `thm1`. -/
theorem CSA_last_tendsto_of_not_antipode (β : ℝ) (Q K : ParamMatrix d)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X)
    (hn : 1 ≤ n) (hp : ∀ j : Idx n, Tendsto (fun t => X t j.castSucc) atTop (𝓝 (X 0 0)))
    (hnot : ¬ Tendsto (fun t => X t (Fin.last n)) atTop (𝓝 (Perspective.antipode d (X 0 0)))) :
    Tendsto (fun t => X t (Fin.last n)) atTop (𝓝 (X 0 0)) := by
  rcases CSA_last_tendsto_leader_or_antipode β Q K X hX hn hp with h | h
  · exact h
  · exact False.elim (hnot h)

/-- The actual canonical original flow has the same last-token dichotomy under its actual prefix convergence.
Source: arXiv:2411.04990v2, §4, the conditioned original flow of `thm1`. -/
theorem csaFlow_last_tendsto_leader_or_antipode (β : ℝ) (Q K : ParamMatrix d)
    (X₀ : SphereTuple d (n + 1)) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t j.castSucc)
      atTop (𝓝 (X₀ 0))) :
    Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t (Fin.last n)) atTop (𝓝 (X₀ 0)) ∨
      Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t (Fin.last n))
        atTop (𝓝 (Perspective.antipode d (X₀ 0))) := by
  have h0 := csaFlow_zero β Q K (ContinuousLinearMap.id ℝ _) X₀
  simpa only [h0] using CSA_last_tendsto_leader_or_antipode β Q K _
    (csaFlow_spec β Q K _ X₀).2 hn (by simpa only [h0] using hp)

/-- The actual unit leader and its actual antipode are distinct, even before the dimension hypothesis for null sets.
Source: arXiv:2411.04990v2, §4, the two distinct endpoint configurations in `thm1`. -/
theorem csaSphere_ne_antipode (a : SSphere d) : a ≠ Perspective.antipode d a := by
  intro he
  have h := congrArg (fun z : SSphere d => inner ℝ (z : EucSpace d) (a : EucSpace d)) he
  change inner ℝ (a : EucSpace d) (a : EucSpace d) = inner ℝ (-(a : EucSpace d)) (a : EucSpace d) at h
  rw [inner_neg_left, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp a.2, one_pow] at h
  norm_num at h

/-- Actual nonzero original consensus jointly realizes all solution, prefix-convergence and positive-strip premises.
Source: arXiv:2411.04990v2, §4, the original consensus equilibrium in `thm1`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧ (1 : ℕ) ≤ 1 ∧
      (∀ j : Idx 1, Tendsto (fun _ : ℝ => (fun _ : Idx 2 => basePoint 1) j.castSucc)
        atTop (𝓝 (basePoint 1 : SSphere 2))) ∧ (-1 : ℝ) < -1 / 2 ∧ (1 / 2 : ℝ) < 1 :=
  ⟨CSA_const_consensus β Q K (basePoint 1), le_rfl, fun _ => tendsto_const_nhds,
    by norm_num, by norm_num⟩

/-- The original leader and antipodal overlap-limit premises both have actual constant-path witnesses.
Source: arXiv:2411.04990v2, §4, the endpoint configurations in `thm1`. -/
example (a : SSphere 2) :
    Tendsto (fun _ : ℝ => inner ℝ (Perspective.antipode 2 a : EucSpace 2) (a : EucSpace 2))
      atTop (𝓝 (-1)) ∧ ¬ Tendsto (fun _ : ℝ => a) atTop (𝓝 (Perspective.antipode 2 a)) := by
  have hi : inner ℝ (Perspective.antipode 2 a : EucSpace 2) (a : EucSpace 2) = -1 := by
    change inner ℝ (-(a : EucSpace 2)) (a : EucSpace 2) = -1
    rw [inner_neg_left, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp a.2, one_pow]
  refine ⟨by simpa only [hi] using (tendsto_const_nhds : Tendsto (fun _ : ℝ => (-1 : ℝ)) atTop (𝓝 (-1))), ?_⟩
  intro h
  exact csaSphere_ne_antipode a (tendsto_nhds_unique tendsto_const_nhds h)

end Transformer.Causal
