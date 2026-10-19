module

public import Mathlib.Topology.Algebra.Order.Group
public import Mathlib.Topology.Order.DenselyOrdered
public import Mathlib.Tactic

/-!
# Uniform convergence across the staircase jump times

ArXiv:2410.06833v1, §6, `conj: saddle-to-saddle` and `thm: staircase`.
Both displays ask for uniform convergence on entire open plateau intervals.
For continuous approximating curves, the bounds extend to their endpoints.
Consequently adjacent limiting heights agree: excluding the jump time itself
does not allow different one-sided limits under this uniform quantifier.

This module proves that obstruction independently of the attention dynamics.
It applies to continuous normalized energy curves and continuous time changes.
It does not replace the source's uniform convergence by convergence on compact
subintervals away from the jump times.
-/

@[expose] public section

namespace Transformer.Metastability

/-- An absolute-error bound on a nonempty open interval extends to both
endpoints for a continuous curve. Source: arXiv:2410.06833v1, §6, the uniform
convergence quantifiers in `conj: saddle-to-saddle` and `thm: staircase`. -/
theorem continuous_plateau_bound_endpoints (f : ℝ → ℝ) (hf : Continuous f)
    (a b L ε : ℝ) (hab : a < b)
    (hbound : ∀ t ∈ Set.Ioo a b, |f t - L| ≤ ε) :
    |f a - L| ≤ ε ∧ |f b - L| ≤ ε := by
  have hclosed : IsClosed {t : ℝ | |f t - L| ≤ ε} :=
    isClosed_le ((hf.sub continuous_const).abs) continuous_const
  have hclosure : closure (Set.Ioo a b) ⊆ {t : ℝ | |f t - L| ≤ ε} :=
    closure_minimal hbound hclosed
  rw [closure_Ioo hab.ne] at hclosure
  exact ⟨hclosure ⟨le_rfl, hab.le⟩, hclosure ⟨hab.le, le_rfl⟩⟩

/-- The endpoint-extension hypotheses hold on a constant curve. -/
example : Continuous (fun _ : ℝ => (3 : ℝ)) ∧ (0 : ℝ) < 1 ∧
    ∀ t ∈ Set.Ioo (0 : ℝ) 1, |(fun _ : ℝ => (3 : ℝ)) t - 3| ≤ 0 :=
  ⟨continuous_const, zero_lt_one, fun _ _ => by norm_num⟩

/-- **Adjacent plateau heights are at most twice the uniform error apart.**
The two error estimates meet at their common endpoint by continuity, and the
triangle inequality compares the proposed limiting heights there.
Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle` and `thm: staircase`. -/
theorem adjacent_plateau_heights_le (f : ℝ → ℝ) (hf : Continuous f)
    (a b c L R ε : ℝ) (hab : a < b) (hbc : b < c)
    (hleft : ∀ t ∈ Set.Ioo a b, |f t - L| ≤ ε)
    (hright : ∀ t ∈ Set.Ioo b c, |f t - R| ≤ ε) :
    |L - R| ≤ 2 * ε := by
  have hl := (continuous_plateau_bound_endpoints f hf a b L ε hab hleft).2
  have hr := (continuous_plateau_bound_endpoints f hf b c R ε hbc hright).1
  calc
    |L - R| ≤ |L - f b| + |f b - R| := abs_sub_le L (f b) R
    _ = |f b - L| + |f b - R| := by rw [abs_sub_comm L (f b)]
    _ ≤ ε + ε := add_le_add hl hr
    _ = 2 * ε := by ring

/-- The adjacent-interval hypotheses hold with equal heights and zero error. -/
example : Continuous (fun _ : ℝ => (3 : ℝ)) ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧
    (∀ t ∈ Set.Ioo (0 : ℝ) 1, |(fun _ : ℝ => (3 : ℝ)) t - 3| ≤ 0) ∧
    (∀ t ∈ Set.Ioo (1 : ℝ) 2, |(fun _ : ℝ => (3 : ℝ)) t - 3| ≤ 0) :=
  ⟨continuous_const, zero_lt_one, by norm_num,
    fun _ _ => by norm_num, fun _ _ => by norm_num⟩

/-- The limits on two full adjacent intervals coincide for a continuous
approximating family. Source: arXiv:2410.06833v1, §6, `thm: staircase`.
The interval endpoints themselves need not belong to either error estimate. -/
theorem uniform_adjacent_plateau_limits_eq (f : ℝ → ℝ → ℝ)
    (hf : ∀ β, Continuous (f β)) (a b c L R : ℝ) (hab : a < b) (hbc : b < c)
    (hlimit : ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      (∀ t ∈ Set.Ioo a b, |f β t - L| < ε) ∧
      (∀ t ∈ Set.Ioo b c, |f β t - R| < ε)) : L = R := by
  by_contra hne
  have hpos : 0 < |L - R| := abs_pos.mpr (sub_ne_zero.mpr hne)
  obtain ⟨B, hB⟩ := hlimit (|L - R| / 3) (by positivity)
  obtain ⟨hl, hr⟩ := hB (B + 1) (by linarith)
  have hbound := adjacent_plateau_heights_le (f (B + 1)) (hf (B + 1))
    a b c L R (|L - R| / 3) hab hbc
    (fun t ht => (hl t ht).le) (fun t ht => (hr t ht).le)
  linarith

/-- The two-interval limit hypotheses hold for a constant continuous family. -/
example : (∀ β : ℝ, Continuous ((fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β)) ∧
    (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧
    ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      (∀ t ∈ Set.Ioo (0 : ℝ) 1,
        |(fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β t - 3| < ε) ∧
      (∀ t ∈ Set.Ioo (1 : ℝ) 2,
        |(fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β t - 3| < ε) := by
  refine ⟨fun _ => continuous_const, zero_lt_one, by norm_num, ?_⟩
  intro ε hε
  exact ⟨0, fun _ _ => ⟨fun _ _ => by simpa using hε,
    fun _ _ => by simpa using hε⟩⟩

/-- **Different heights cannot be uniform limits on full adjacent intervals.**
This is a counterexample obstruction to the uniform jump quantifier, for
every continuous family, and so also for continuous attention energies.
Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle` and `thm: staircase`. -/
theorem not_uniform_limit_of_distinct_plateau_heights (f : ℝ → ℝ → ℝ)
    (hf : ∀ β, Continuous (f β)) (a b c L R : ℝ) (hab : a < b) (hbc : b < c)
    (hne : L ≠ R) :
    ¬ (∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      (∀ t ∈ Set.Ioo a b, |f β t - L| < ε) ∧
      (∀ t ∈ Set.Ioo b c, |f β t - R| < ε)) := by
  intro hlimit
  exact hne (uniform_adjacent_plateau_limits_eq f hf a b c L R hab hbc hlimit)

/-- The obstruction's hypotheses have a concrete nonempty instance: the
identity curve at every parameter, adjoining intervals `(0,1)` and `(1,2)`,
and proposed heights `0` and `1`. -/
example : (∀ β : ℝ, Continuous ((fun (_ : ℝ) (t : ℝ) => t) β)) ∧
    (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧ (0 : ℝ) ≠ 1 :=
  ⟨fun _ => continuous_id, zero_lt_one, by norm_num, zero_ne_one⟩

/-- **Uniform limits on the full staircase plateaux have no jumps.**
Let continuous curves converge uniformly to the height `L i` on every
`(T i, T (i+1))`, with the final plateau `(T k, ∞)`. Then all heights equal
`L 0`. This is the quantifier in `HasStaircaseProfile`, with continuity made
explicit; it is an obstruction, not a weakened version of the open problem.

Source: arXiv:2410.06833v1, §6, `conj: saddle-to-saddle`; compare the uniform
limit in `thm: staircase`, whose first plateau is indexed by `1`. -/
theorem uniform_plateau_heights_eq (f : ℝ → ℝ → ℝ)
    (hf : ∀ β, Continuous (f β)) (k : ℕ) (T : ℕ → ℝ) (L : ℕ → ℝ)
    (hT : ∀ i, i < k → T i < T (i + 1))
    (hlimit : ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      ∀ i : ℕ, i ≤ k → ∀ t : ℝ, T i < t → (i < k → t < T (i + 1)) →
        |f β t - L i| < ε) :
    ∀ i, i ≤ k → L i = L 0 := by
  have hadj : ∀ i, i < k → L i = L (i + 1) := by
    intro i hi
    by_contra hne
    have hpos : 0 < |L i - L (i + 1)| := abs_pos.mpr (sub_ne_zero.mpr hne)
    set ε := |L i - L (i + 1)| / 3
    have hε : 0 < ε := by dsimp [ε]; positivity
    obtain ⟨B, hB⟩ := hlimit ε hε
    let c := if i + 1 < k then T (i + 2) else T (i + 1) + 1
    have hbc : T (i + 1) < c := by
      dsimp [c]
      split_ifs with hnext
      · simpa only [Nat.add_assoc] using hT (i + 1) hnext
      · linarith
    have hbound := adjacent_plateau_heights_le (f (B + 1)) (hf (B + 1))
      (T i) (T (i + 1)) c (L i) (L (i + 1)) ε (hT i hi) hbc
      (fun t ht => (hB (B + 1) (by linarith) i hi.le t ht.1 (fun _ => ht.2)).le)
      (fun t ht => (hB (B + 1) (by linarith) (i + 1) (by omega) t ht.1
        (fun hnext => by simpa [c, hnext, Nat.add_assoc] using ht.2)).le)
    dsimp [ε] at hbound
    linarith
  intro i hi
  induction i with
  | zero => rfl
  | succ i ih =>
    exact (hadj i (by omega)).symm.trans (ih (by omega))

/-- All hypotheses of the no-jump theorem are simultaneously satisfiable:
a constant family, one jump-time label, `T i = i`, and identical heights.
The labeled partition is nonempty, although there is no actual height jump. -/
example : (∀ β : ℝ, Continuous ((fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β)) ∧
    (∀ i : ℕ, i < 1 → (i : ℝ) < ((i + 1 : ℕ) : ℝ)) ∧
    ∀ ε : ℝ, 0 < ε → ∃ B : ℝ, ∀ β : ℝ, B < β →
      ∀ i : ℕ, i ≤ 1 → ∀ t : ℝ, (i : ℝ) < t →
        (i < 1 → t < ((i + 1 : ℕ) : ℝ)) →
          |(fun (_ : ℝ) (_ : ℝ) => (3 : ℝ)) β t - (fun _ : ℕ => (3 : ℝ)) i| < ε := by
  refine ⟨fun _ => continuous_const, fun i _ => ?_, ?_⟩
  · simp only [Nat.cast_add, Nat.cast_one]
    linarith
  · intro ε hε
    exact ⟨0, fun _ _ _ _ _ _ _ => by simpa using hε⟩

end Transformer.Metastability
