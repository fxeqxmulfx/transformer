/-
# Original conditional antipodal fibers have at most one initial token

ArXiv:2411.04990v2, §4, proof of `thm1`, conditional instability.
For a fixed actual prefix converging to its original leader, the
last-token scalar dichotomy gives an actual real-time endpoint
limit for every original last initial token. An integer-time
antipodal limit rules out the leader alternative and hence also
gives the actual real-time antipodal limit.

Two such original initializations share their entire original prefix
trajectory. The proved original antipodal expansion therefore
forces their actual trajectories, and their original initial last
tokens, to agree. Thus the original integer-time antipodal event
has at most one last-token point in every original good-prefix
fiber. Its previously proved Borel measurability permits Fubini
for the original independent uniform initialization.
-/

import Transformer.Causal.Section4_AntipodalExpansion

open Real Filter Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- For an actual good original prefix, an original integer-time antipodal limit is an actual real-time antipodal limit.
Source: arXiv:2411.04990v2, §4, the two possible limits in `thm1`. -/
theorem csaFlow_last_antipode_of_nat (β : ℝ) (Q K : ParamMatrix d)
    (X₀ : SphereTuple d (n + 1)) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t j.castSucc)
      atTop (𝓝 (X₀ 0)))
    (h : Tendsto (fun m : ℕ => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ (m : ℝ) (Fin.last n))
      atTop (𝓝 (Perspective.antipode d (X₀ 0)))) :
    Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t (Fin.last n))
      atTop (𝓝 (Perspective.antipode d (X₀ 0))) := by
  rcases csaFlow_last_tendsto_leader_or_antipode β Q K X₀ hn hp with hgood | hbad
  · have hg := hgood.comp (tendsto_natCast_atTop_atTop : Tendsto (fun m : ℕ => (m : ℝ)) atTop atTop)
    have he := tendsto_nhds_unique hg h
    exact False.elim (csaSphere_ne_antipode (X₀ 0) he)
  · exact hbad

/-- Original initial tuples sharing a converging original prefix cannot both produce the same original antipodal limit unless equal.
Source: arXiv:2411.04990v2, §4, the conditional instability exclusion in `thm1`. -/
theorem csaFlow_initial_eq_of_antipodal_limits (β : ℝ) (Q K : ParamMatrix d)
    (X₀ Y₀ : SphereTuple d (n + 1)) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, X₀ j.castSucc = Y₀ j.castSucc)
    (hlim : ∀ j : Idx n, Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t j.castSucc)
      atTop (𝓝 (X₀ 0)))
    (hx : Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t (Fin.last n))
      atTop (𝓝 (Perspective.antipode d (X₀ 0))))
    (hy : Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) Y₀ t (Fin.last n))
      atTop (𝓝 (Perspective.antipode d (X₀ 0)))) : X₀ = Y₀ := by
  have h := CSA_eq_of_shared_prefix_antipodal_limits β Q K _ _
    (csaFlow_spec β Q K _ X₀).2 (csaFlow_spec β Q K _ Y₀).2 (X₀ 0) hn
    (fun t j => csaFlow_prefix_congr β Q K _ X₀ Y₀ hp t j) hlim hx hy
  simpa only [csaFlow_zero] using congrFun h 0

/-- Appending an actual last initial token preserves precisely the actual smaller original prefix flow.
Source: arXiv:2411.04990v2, §4, independence of preceding original tokens in `thm1`. -/
theorem csaFlow_snoc_prefix (β : ℝ) (Q K V : ParamMatrix d)
    (Y : SphereTuple d n) (z : SSphere d) (t : ℝ) (j : Idx n) :
    csaFlow β Q K V (Fin.snoc Y z) t j.castSucc = csaFlow β Q K V Y t j := by
  simpa only [Fin.snoc_castSucc] using csaFlow_castSucc β Q K V (Fin.snoc Y z) t j

/-- An actual nonempty original prefix has the same original leader after appending a last token.
Source: arXiv:2411.04990v2, §4, conditioning on the original fixed leader in `thm1`. -/
theorem csaSnoc_leader (Y : SphereTuple d n) (z : SSphere d) (hn : 1 ≤ n) :
    (Fin.snoc Y z : SphereTuple d (n + 1)) 0 = Y ⟨0, hn⟩ := by
  change (Fin.snoc Y z : SphereTuple d (n + 1)) (⟨0, hn⟩ : Idx n).castSucc = Y ⟨0, hn⟩
  rw [Fin.snoc_castSucc]

/-- For every actual original good prefix, the original measurable antipodal last-token fiber has at most one point.
Source: arXiv:2411.04990v2, §4, the conditional measure-zero step of `thm1`.
The fiber uses the original canonical flow and the actual integer-time antipodal convergence event. -/
theorem subsingleton_csaSnoc_antipodal_nat_fiber (β : ℝ) (Q K : ParamMatrix d)
    (Y : SphereTuple d n) (hn : 1 ≤ n)
    (hlim : ∀ j : Idx n, Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) Y t j)
      atTop (𝓝 (Y ⟨0, hn⟩))) :
    {z : SSphere d | Tendsto
      (fun m : ℕ => csaFlow β Q K (ContinuousLinearMap.id ℝ _) (Fin.snoc Y z) (m : ℝ) (Fin.last n))
      atTop (𝓝 (Perspective.antipode d ((Fin.snoc Y z : SphereTuple d (n + 1)) 0)))}.Subsingleton := by
  intro x hx y hy
  have hp (z : SSphere d) : ∀ j : Idx n, Tendsto
      (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) (Fin.snoc Y z) t j.castSucc)
      atTop (𝓝 ((Fin.snoc Y z : SphereTuple d (n + 1)) 0)) := by
    intro j
    simpa only [csaFlow_snoc_prefix, csaSnoc_leader Y z hn] using hlim j
  have hxr := csaFlow_last_antipode_of_nat β Q K (Fin.snoc Y x) hn (hp x) hx
  have hyr := csaFlow_last_antipode_of_nat β Q K (Fin.snoc Y y) hn (hp y) hy
  have he := csaFlow_initial_eq_of_antipodal_limits β Q K (Fin.snoc Y x) (Fin.snoc Y y) hn
    (fun j => by rw [Fin.snoc_castSucc, Fin.snoc_castSucc]) (hp x) hxr
    (by simpa only [csaSnoc_leader Y x hn, csaSnoc_leader Y y hn] using hyr)
  simpa only [Fin.snoc_last] using congrFun he (Fin.last n)

/-- With its original good prefix, a follower outside the original integer-time antipodal event converges to the original leader.
Source: arXiv:2411.04990v2, §4, the concluding deterministic implication of `thm1`. -/
theorem csaFlow_last_tendsto_of_not_antipodal_nat (β : ℝ) (Q K : ParamMatrix d)
    (X₀ : SphereTuple d (n + 1)) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t j.castSucc)
      atTop (𝓝 (X₀ 0)))
    (hnot : ¬ Tendsto (fun m : ℕ => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ (m : ℝ) (Fin.last n))
      atTop (𝓝 (Perspective.antipode d (X₀ 0)))) :
    Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t (Fin.last n)) atTop (𝓝 (X₀ 0)) := by
  rcases csaFlow_last_tendsto_leader_or_antipode β Q K X₀ hn hp with hgood | hbad
  · exact hgood
  · exact False.elim (hnot (hbad.comp tendsto_natCast_atTop_atTop))

/-- The original integer-time antipodal initial event is Borel on the original full tuple space.
Source: arXiv:2411.04990v2, §4, the measurable conditional exceptional set of `thm1`. -/
theorem measurableSet_csaAntipodalInitial (β : ℝ) (Q K : ParamMatrix d) :
    MeasurableSet {X₀ : SphereTuple d (n + 1) | Tendsto
      (fun m : ℕ => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ (m : ℝ) (Fin.last n))
      atTop (𝓝 (Perspective.antipode d (X₀ 0)))} :=
  measurableSet_csaFlow_antipode_nat β Q K (ContinuousLinearMap.id ℝ _) (Fin.last n) 0

/-- Genuine original one-token prefix flows converge, and actual original antipodal follower fibers are nonempty.
Source: arXiv:2411.04990v2, §4, the original antipodal equilibrium in `thm1`. -/
example (a : SSphere 2) : (1 : ℕ) ≤ 1 ∧
    (∀ j : Idx 1, Tendsto (fun t => csaFlow 1 (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _) (fun _ => a) t j) atTop (𝓝 a)) ∧
    Tendsto (fun m : ℕ => csaFlow 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (antipodalCircleTuple a) (m : ℝ) (Fin.last 1))
      atTop (𝓝 (Perspective.antipode 2 a)) := by
  refine ⟨le_rfl, ?_, ?_⟩
  · intro j
    simpa only [csaFlow_consensus] using (tendsto_const_nhds : Tendsto (fun _ : ℝ => a) atTop (𝓝 a))
  · have he := csaFlow_eq_of_CSA 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (fun _ => antipodalCircleTuple a) (csa_const_antipodal 1 a)
    have hl : antipodalCircleTuple a (Fin.last 1) = Perspective.antipode 2 a := by rfl
    simpa only [he, hl] using
      (tendsto_const_nhds : Tendsto (fun _ : ℕ => Perspective.antipode 2 a) atTop (𝓝 (Perspective.antipode 2 a)))

/-- Actual original consensus jointly realizes a good prefix and exclusion of the original antipodal event.
Source: arXiv:2411.04990v2, §4, the original consensus equilibrium in `thm1`. -/
example (a : SSphere 2) :
    (∀ j : Idx 1, Tendsto (fun t => csaFlow 1 (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _) (fun _ : Idx 2 => a) t j.castSucc)
        atTop (𝓝 a)) ∧
    ¬ Tendsto (fun m : ℕ => csaFlow 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (fun _ : Idx 2 => a) (m : ℝ) (Fin.last 1))
        atTop (𝓝 (Perspective.antipode 2 a)) := by
  constructor
  · intro j
    simpa only [csaFlow_consensus] using (tendsto_const_nhds : Tendsto (fun _ : ℝ => a) atTop (𝓝 a))
  · intro h
    have he : a = Perspective.antipode 2 a := tendsto_nhds_unique
      (tendsto_const_nhds : Tendsto (fun _ : ℕ => a) atTop (𝓝 a)) (by simpa only [csaFlow_consensus] using h)
    exact csaSphere_ne_antipode a he

end Transformer.Causal
