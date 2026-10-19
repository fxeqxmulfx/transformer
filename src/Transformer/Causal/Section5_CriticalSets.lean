module

public import Transformer.Causal.Section5_PrefixCurvature
public import Mathlib.Analysis.Calculus.Deriv.Inverse
public import Mathlib.Topology.DiscreteSubset
public import Mathlib.Topology.Compactness.Lindelof

/-!
# Isolated and countable critical sets for the actual frozen row

ArXiv:2411.04990v2, §5, the smooth frozen specialization of
Appendix A's critical-point isolation argument.
Once the actual predecessors lie in the derived narrow
neighborhoods, the proved last-row classification excludes
zero curvature at every critical point. The derivative of the
actual force is exactly that curvature. Thus its zeros form a
discrete subset of the real line, are countable, and are finite
in every bounded closed interval.

The actual normalized velocity has exactly those zeros because
its original partition stays strictly positive. These results
concern this smooth frozen row with clustered predecessors;
the false general printed C1 convergence lemma is not used.
The elementary simple-zero statements below also record their
precise derivative premises for the specialized argument.
-/

@[expose] public section

open Real Set

namespace Transformer.Causal

variable {n m : ℕ}

/-- Real zeros with a nonzero actual derivative are isolated as a subset of the real line.
Source: arXiv:2411.04990v2, Appendix A and §5, auxiliary simple-zero argument for the actual frozen interaction. -/
theorem frozen_simple_zeros_isDiscrete (f g : ℝ → ℝ) (hd : ∀ x, HasDerivAt f (g x) x)
    (hsimple : ∀ x, f x = 0 → g x ≠ 0) : IsDiscrete {x | f x = 0} := by
  rw [isDiscrete_iff_nhdsNE]
  intro x hx
  apply Filter.inf_principal_eq_bot.mpr
  exact (hd x).eventually_ne (hsimple x hx)

/-- The actual isolated real zeros of a differentiable function with nonzero derivative at each zero are countable.
Source: arXiv:2411.04990v2, Appendix A and §5, the simple-critical-point set used for the smooth frozen specialization. -/
theorem frozen_simple_zeros_countable (f g : ℝ → ℝ) (hd : ∀ x, HasDerivAt f (g x) x)
    (hsimple : ∀ x, f x = 0 → g x ≠ 0) : {x | f x = 0}.Countable :=
  (HereditarilyLindelofSpace.isLindelof _).countable_of_isDiscrete (frozen_simple_zeros_isDiscrete f g hd hsimple)

/-- In every bounded closed interval, the actual zeros are finite when every zero has nonzero derivative.
Source: arXiv:2411.04990v2, Appendix A and §5, compactness and simple critical points for the actual smooth frozen row. -/
theorem frozen_simple_zeros_finite_Icc (f g : ℝ → ℝ) (hd : ∀ x, HasDerivAt f (g x) x)
    (hsimple : ∀ x, f x = 0 → g x ≠ 0) (A B : ℝ) : (Icc A B ∩ {x | f x = 0}).Finite := by
  have hc : Continuous f := continuous_iff_continuousAt.mpr (fun x => (hd x).continuousAt)
  exact (isCompact_Icc.inter_right (isClosed_eq hc continuous_const)).finite
    ((frozen_simple_zeros_isDiscrete f g hd hsimple).mono inter_subset_right)

/-- The actual original last-row force zeros form a discrete subset of the real line once every actual predecessor lies in a narrow cluster.
Source: arXiv:2411.04990v2, §5, isolated original critical points in the smooth frozen specialization of `thm:fixed_centers`. -/
theorem frozen_last_force_zeros_isDiscrete (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    IsDiscrete {u | frozenLastForce β θ a p u = 0} := by
  apply frozen_simple_zeros_isDiscrete (frozenLastForce β θ a p) (frozenLastCurvature β θ a p)
    (hasDerivAt_frozenLastForce β θ a p)
  intro u hu
  rcases frozen_last_critical_dichotomy_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN
    hh hg p hpre u hu with hn | ⟨hp, hi⟩
  · exact hn.ne
  · exact hp.ne'

/-- The actual original last-row force has a countable critical set when all actual predecessors are in the proved narrow neighborhoods.
Source: arXiv:2411.04990v2, §5, critical-point isolation for `thm:fixed_centers`, from the proved original curvature dichotomy. -/
theorem frozen_last_force_zeros_countable (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    {u | frozenLastForce β θ a p u = 0}.Countable := by
  apply frozen_simple_zeros_countable (frozenLastForce β θ a p) (frozenLastCurvature β θ a p)
    (hasDerivAt_frozenLastForce β θ a p)
  intro u hu
  rcases frozen_last_critical_dichotomy_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN
    hh hg p hpre u hu with hn | ⟨hp, hi⟩
  · exact hn.ne
  · exact hp.ne'

/-- The actual original last-row force has only finitely many critical angles in every bounded closed interval when its actual predecessors are narrowly clustered.
Source: arXiv:2411.04990v2, §5, the isolated critical set for `thm:fixed_centers`, retaining the original force and all original masses. -/
theorem frozen_last_force_zeros_finite_Icc (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) (A B : ℝ) :
    (Icc A B ∩ {u | frozenLastForce β θ a p u = 0}).Finite := by
  apply frozen_simple_zeros_finite_Icc (frozenLastForce β θ a p) (frozenLastCurvature β θ a p)
    (hasDerivAt_frozenLastForce β θ a p)
  intro u hu
  rcases frozen_last_critical_dichotomy_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN
    hh hg p hpre u hu with hn | ⟨hp, hi⟩
  · exact hn.ne
  · exact hp.ne'

/-- The critical set of the exact original normalized last-row velocity is countable under the proved narrow-cluster hypotheses.
Source: arXiv:2411.04990v2, §5, the actual critical points of `thm:fixed_centers`; positive original normalization preserves the force zeros. -/
theorem frozen_last_velocity_zeros_countable (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    {u | frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0}.Countable := by
  have hset : {u | frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0} =
      {u | frozenLastForce β θ a p u = 0} := by
    ext u
    exact frozenVelocity_last_eq_zero_iff β θ a (fun j => le_trans zero_le_one (ha j)) p u
  rw [hset]
  exact frozen_last_force_zeros_countable N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN hh hg p hpre

/-- The actual real identity function has genuine simple zeros and supplies the joint derivative premises of the real isolation, countability, and compact finiteness estimates.
Source: arXiv:2411.04990v2, Appendix A and §5, the auxiliary simple-zero argument used for the smooth frozen row. -/
example : (∀ x : ℝ, HasDerivAt (fun u : ℝ => u) 1 x) ∧
    (∀ x : ℝ, (fun u : ℝ => u) x = 0 → (1 : ℝ) ≠ 0) := by
  exact ⟨fun x => hasDerivAt_id x, fun _ _ => one_ne_zero⟩

/-- Genuine original clustered predecessors satisfy all critical-set premises; the exact original normalized row therefore has countably many critical angles.
Source: arXiv:2411.04990v2, §5, the original two-center configuration for `thm:fixed_centers`. -/
example : {u | frozenVelocity 2 2 11 (![0, Real.pi]) (fun _ => 1)
    (Fin.snoc (fun _ : Idx 1 => 0) u) (Fin.last 1) = 0}.Countable := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  exact frozen_last_velocity_zeros_countable 3 11 ε (142 / 25) hβ hε hc hcap (![0, Real.pi]) (fun _ => 1)
    ha hsep hpair (by simpa only [Nat.cast_one] using hN) hh hg (fun _ : Idx 1 => 0) hpre

/-- The same genuine original row has finitely many critical force zeros in a full original angular period, with every narrow-cluster premise supplied.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual periodic critical set for `thm:fixed_centers`. -/
example : (Icc 0 (2 * Real.pi) ∩ {u | frozenLastForce 11 (![0, Real.pi] : Idx 2 → ℝ)
    (fun _ => 1) (fun _ : Idx 1 => 0) u = 0}).Finite := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  exact frozen_last_force_zeros_finite_Icc 3 11 ε (142 / 25) hβ hε hc hcap (![0, Real.pi]) (fun _ => 1)
    ha hsep hpair (by simpa only [Nat.cast_one] using hN) hh hg (fun _ : Idx 1 => 0) hpre 0 (2 * Real.pi)

end Transformer.Causal
