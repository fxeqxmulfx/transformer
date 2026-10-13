/-
# Endpoint limits of a uniform staircase profile

ArXiv:2410.06833v1, §6, `conj: saddle-to-saddle`.
For a continuous family, uniform convergence on full plateau intervals,
including the final unbounded plateau, forces its initial-temperature limit
to equal its long-time limit. The two limiting operations are kept separate:
the first is in the parameter and the second is in time at fixed parameter.

The source's printed problem excludes the final plateau. This obstruction
concerns the full-trajectory formulation in `HasStaircaseProfile`, which
adds it, as the source's `thm: staircase` does. No dynamical claim is assumed.

The comparison uses a fixed large parameter after choosing the tolerance.
At that parameter, continuity extends the first plateau's bound to time
zero and the final plateau's bound to its finite left endpoint. The
long-time limit then lies within the same tolerance of the last height.
The initial-temperature limit supplies a third bound at time zero.

All plateau heights agree by the preceding module, so these three errors
bound the distance between the endpoint limits. Choosing the tolerance to
be one quarter of a purported positive distance gives a contradiction.
This argument does not interchange the two limits or assume uniform
long-time convergence in temperature. Its only uniform input is precisely
the source's entire-interval plateau estimate.
-/

import Transformer.Metastability.Section6_UniformPlateaux
import Mathlib.Topology.MetricSpace.Pseudo.Defs
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Filter Topology

namespace Transformer.Metastability

/-- A continuous curve's uniform bound on the last open plateau extends to
its finite endpoint. Source: arXiv:2410.06833v1, §6, the last plateau of
`thm: staircase`, and the full-trajectory form of `conj: saddle-to-saddle`. -/
theorem continuous_final_plateau_bound (f : ℝ → ℝ) (hf : Continuous f)
    (a L ε : ℝ) (hbound : ∀ t, a < t → |f t - L| ≤ ε) :
    ∀ t, a ≤ t → |f t - L| ≤ ε := by
  have hclosed : IsClosed {t : ℝ | |f t - L| ≤ ε} :=
    isClosed_le ((hf.sub continuous_const).abs) continuous_const
  have hclosure : closure (Set.Ioi a) ⊆ {t : ℝ | |f t - L| ≤ ε} :=
    closure_minimal hbound hclosed
  rw [closure_Ioi] at hclosure
  exact hclosure

/-- The endpoint bound has a concrete witness on a nonempty half-line. -/
example : Continuous (fun _ : ℝ => (3 : ℝ)) ∧
    ∀ t : ℝ, 0 < t → |(fun _ : ℝ => (3 : ℝ)) t - 3| ≤ 0 :=
  ⟨continuous_const, fun _ _ => by norm_num⟩

/-- **A full uniform staircase has identical initial and final limits.**
The initial limit is taken in `β`, while the final limit is taken in time
for each `β > 1`. Bounds on the first and last plateaux extend to their
endpoints, and all limiting heights agree by `uniform_plateau_heights_eq`.

The conclusion is an obstruction to the full-trajectory uniform version,
not a change to the source's convergence quantifier.
Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`, with the final
plateau included as in `thm: staircase`. -/
theorem uniform_plateau_endpoint_limits_eq (f : ℝ → ℝ → ℝ)
    (hf : ∀ β, Continuous (f β)) (k : ℕ) (T : ℕ → ℝ) (L : ℕ → ℝ)
    (hk : 0 < k) (hT0 : T 0 = 0) (hT : ∀ i, i < k → T i < T (i + 1))
    (hlimit : ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      ∀ i : ℕ, i ≤ k → ∀ t : ℝ, T i < t → (i < k → t < T (i + 1)) →
        |f β t - L i| < ε)
    (p q : ℝ) (hinit : Tendsto (fun β => f β 0) atTop (nhds p))
    (hlong : ∀ β : ℝ, 1 < β → Tendsto (f β) atTop (nhds q)) : p = q := by
  have hlevels : L k = L 0 := uniform_plateau_heights_eq f hf k T L hT hlimit k le_rfl
  by_contra hne
  have hgap : 0 < |p - q| := abs_pos.mpr (sub_ne_zero.mpr hne)
  set ε := |p - q| / 4
  have hε : 0 < ε := by dsimp [ε]; positivity
  obtain ⟨B, hB⟩ := hlimit ε hε
  have hevent : ∀ᶠ β : ℝ in atTop,
      dist (f β 0) p < ε ∧ B < β ∧ 1 < β :=
    (Metric.tendsto_nhds.mp hinit ε hε).and
      ((eventually_gt_atTop B).and (eventually_gt_atTop 1))
  obtain ⟨β, hβinit, hβB, hβ1⟩ := hevent.exists
  rw [Real.dist_eq] at hβinit
  have hleft : |f β 0 - L 0| ≤ ε := by
    have hab : (0 : ℝ) < T 1 := by simpa only [hT0, zero_add] using hT 0 hk
    refine (continuous_plateau_bound_endpoints (f β) (hf β) 0 (T 1) (L 0) ε hab ?_).1
    intro t ht
    exact (hB β hβB 0 (Nat.zero_le k) t (by simpa [hT0] using ht.1)
      (fun _ => by simpa using ht.2)).le
  have hright : |q - L 0| ≤ ε := by
    have hbound := continuous_final_plateau_bound (f β) (hf β) (T k) (L k) ε
      (fun t ht => (hB β hβB k le_rfl t ht (fun hi => absurd hi (lt_irrefl k))).le)
    have he : ∀ᶠ t : ℝ in atTop, |f β t - L k| ≤ ε :=
      (eventually_ge_atTop (T k)).mono fun t ht => hbound t ht
    simpa only [hlevels] using le_of_tendsto ((hlong β hβ1).sub tendsto_const_nhds).abs he
  have htriangle : |p - q| ≤ |p - f β 0| + |f β 0 - L 0| + |L 0 - q| :=
    (abs_sub_le p (L 0) q).trans
      (add_le_add (abs_sub_le p (f β 0) (L 0)) le_rfl)
  rw [abs_sub_comm p (f β 0), abs_sub_comm (L 0) q] at htriangle
  dsimp [ε] at hβinit hleft hright
  linarith

/-- All inputs of the endpoint theorem hold simultaneously for the constant
family, one finite plateau followed by an unbounded one, and limits `p=q=3`.
Thus neither continuity nor either limiting hypothesis is unwitnessed. -/
example : (∀ β : ℝ, Continuous ((fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β)) ∧
    (0 : ℕ) < 1 ∧ ((0 : ℕ) : ℝ) = 0 ∧
    (∀ i : ℕ, i < 1 → (i : ℝ) < ((i + 1 : ℕ) : ℝ)) ∧
    (∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      ∀ i : ℕ, i ≤ 1 → ∀ t : ℝ, (i : ℝ) < t →
        (i < 1 → t < ((i + 1 : ℕ) : ℝ)) →
          |(fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β t - (fun _ : ℕ => (3 : ℝ)) i| < ε) ∧
    Tendsto (fun β : ℝ => (fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β 0)
      atTop (nhds 3) ∧
    (∀ β : ℝ, 1 < β → Tendsto ((fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β) atTop (nhds 3)) := by
  refine ⟨fun _ => continuous_const, Nat.one_pos, by norm_num, fun i _ => ?_, ?_,
    tendsto_const_nhds, fun _ _ => tendsto_const_nhds⟩
  · simp only [Nat.cast_add, Nat.cast_one]
    linarith
  · intro ε hε
    exact ⟨0, fun _ _ _ _ _ _ _ => by simpa using hε⟩

/-- **Distinct endpoint limits rule out the full uniform profile.**
There is no finite staircase partition, ending in an unbounded plateau,
along which such a continuous family has the uniform estimates of the
full-trajectory version of `conj: saddle-to-saddle`.

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`, with the final
plateau included as in `thm: staircase`. -/
theorem not_uniform_plateaux_of_distinct_endpoint_limits (f : ℝ → ℝ → ℝ)
    (hf : ∀ β, Continuous (f β)) (p q : ℝ)
    (hinit : Tendsto (fun β => f β 0) atTop (nhds p))
    (hlong : ∀ β : ℝ, 1 < β → Tendsto (f β) atTop (nhds q)) (hne : p ≠ q) :
    ¬ ∃ (k : ℕ) (T : ℕ → ℝ) (L : ℕ → ℝ),
      0 < k ∧ T 0 = 0 ∧ (∀ i, i < k → T i < T (i + 1)) ∧
      ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
        ∀ i : ℕ, i ≤ k → ∀ t : ℝ, T i < t → (i < k → t < T (i + 1)) →
          |f β t - L i| < ε := by
  rintro ⟨k, T, L, hk, hT0, hT, hlimit⟩
  exact hne (uniform_plateau_endpoint_limits_eq f hf k T L hk hT0 hT hlimit p q hinit hlong)

/-- The negative theorem's hypotheses have a concrete continuous witness:
`f β t = 1-exp(-t)` has initial value zero and tends to one. -/
example : ∃ f : ℝ → ℝ → ℝ, (∀ β, Continuous (f β)) ∧
    Tendsto (fun β => f β 0) atTop (nhds (0 : ℝ)) ∧
    (∀ β : ℝ, 1 < β → Tendsto (f β) atTop (nhds (1 : ℝ))) ∧ (0 : ℝ) ≠ 1 := by
  refine ⟨fun _ t => 1 - Real.exp (-t), fun _ => by fun_prop, ?_,
    fun _ _ => ?_, zero_ne_one⟩
  · simp
  · simpa using (tendsto_const_nhds.sub Real.tendsto_exp_neg_atTop_nhds_zero)

end Transformer.Metastability
