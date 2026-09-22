/-
# The emergence of clusters in self-attention dynamics — `l:boundedxn`

§7 of arXiv:2305.05465v6: if an extreme token stays bounded, its row of the
self-attention matrix tends to the standard basis row of that token.

**What the source says and what is carried here.**  The second sentence of
`l:boundedxn` says "if `x_1(t)` is bounded then `P_11(t) → 1` and
`P_{1j}(t) → 0` for any `j ∈ [n-1]`".  As written it is false: `1 ∈ [n-1]`
whenever `n ≥ 2`, so it asks for `P_11(t) → 0` and `P_11(t) → 1` at once.  The
range is a copy of the first sentence's, where `[n-1] = [n] \ {n}` is correct;
symmetrically it must be `[n] \ {1}`, and that is how it is carried.  Nothing
else is changed.

The proof is the source's: by `l:onlyone` every other token is unbounded and
runs off like `-c e^t` (`e:convdeladiff`, `tendsto_softmaxWeight_of_bounded`);
the diagonal entry follows by the row sum.  The case of `x_1` is the mirror
image under `x ↦ -x`.

Source: arXiv:2305.05465v6, `l:boundedxn`.
-/

import Transformer.Clusters.Section7_BoundedXNLimit
import Transformer.Clusters.Section7_Bounded

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {n m : ℕ}

/-- By `l:onlyone`, a token other than a bounded one is unbounded. -/
theorem not_isBoundedToken_of_ne (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) {i j : Idx (m + 1)}
    (hi : IsBoundedToken X i) (hji : j ≠ i) : ¬ IsBoundedToken X j := by
  intro hj
  have hsub : ({j, i} : Set (Idx (m + 1))) ⊆ {k | IsBoundedToken X k} := by
    intro k hk
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hk
    rcases hk with rfl | rfl
    exacts [hj, hi]
  have h2 := Set.ncard_le_ncard hsub (Set.toFinite _)
  rw [Set.ncard_pair hji] at h2
  have := ncard_boundedTokens_le_one X hX hord
  omega

/-- The coordinates of an ordered solution stay ordered for `t ≥ 0`. -/
theorem coord_mono (X : ℝ → Idx (m + 1) → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (hord : IsOrderedConfig (X 0)) (t : ℝ) (ht : 0 ≤ t) (i j : Idx (m + 1)) (hij : i ≤ j) :
    X t i 0 ≤ X t j 0 := by
  rcases hij.lt_or_eq with h | h
  · exact (isOrderedConfig_of_nonneg X hX hord t ht i j h).le
  · rw [h]

/-- **Lemma (l:boundedxn), the largest token.**  If `x_n(t)` stays bounded
then `P_nn(t) → 1` and `P_nj(t) → 0` for every `j ∈ [n-1]`.

Source: arXiv:2305.05465v6, `l:boundedxn`. -/
theorem tendsto_attention_of_bounded_last (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0))
    (hb : IsBoundedToken X (Fin.last m)) :
    Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) (Fin.last m) (Fin.last m))
        atTop (nhds 1) ∧
      ∀ j : Idx (m + 1), j ≠ Fin.last m →
        Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) (Fin.last m) j)
          atTop (nhds 0) := by
  have hle := coord_mono X hX hord
  have hoff : ∀ j : Idx (m + 1), j ≠ Fin.last m →
      Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) (Fin.last m) j)
        atTop (nhds 0) := by
    intro j hj
    have hjv : (j : ℕ) < m := by
      have h2 : (j : ℕ) ≠ m := fun h => hj (Fin.ext (by simp [h]))
      have := j.isLt; omega
    set J : Idx (m + 1) := ⟨m - 1, by omega⟩
    have hJN : J ≠ Fin.last m := fun h => by
      have := congrArg Fin.val h; simp only [J, Fin.val_last] at this; omega
    have hJ : ∀ k : Idx (m + 1), k ≠ Fin.last m → k ≤ J := fun k hk => by
      have h2 : (k : ℕ) ≠ m := fun h => hk (Fin.ext (by simp [h]))
      have := k.isLt
      rw [Fin.le_iff_val_le_val]; simp only [J]; omega
    obtain ⟨R, hR⟩ := hb
    obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant (m + 1) (Nat.succ_pos m)
    rcases unbounded_tendsto_atTop_or_atBot X hX hord J
        (not_isBoundedToken_of_ne X hX hord ⟨R, hR⟩ hJN) with h | h
    · obtain ⟨t, ht, ht0⟩ := ((h.1.eventually (eventually_gt_atTop R)).and
        (eventually_ge_atTop 0)).exists
      linarith [hle t ht0 J (Fin.last m) (Fin.le_last J), (abs_le.1 (hR t ht0)).2]
    obtain ⟨t₁, ht₁⟩ := (h.1.eventually (eventually_lt_atBot (-A))).exists
    obtain ⟨c, hc, hev⟩ := exists_exp_upper_bound A hA hAeq X hX J t₁ ht₁
    have := tendsto_softmaxWeight_of_bounded (fun t k => X t k 0)
      (fun t k => hasDerivAt_coord X hX t k) (Fin.last m) J 0 hJN
      (fun t ht k => hle t ht k _ (Fin.le_last k)) (fun t ht k hk => hle t ht k J (hJ k hk))
      (fun t ht k => hle t ht 0 k (Fin.zero_le k)) hR hc hev j hj
    simpa only [attentionMatrix_one_eq] using this
  refine ⟨?_, hoff⟩
  simpa using tendsto_attention_of_tendsto_others (Nat.succ_pos m) 1 1 X (Fin.last m)
    (Fin.last m) (fun _ => 0) hoff

/-- **Lemma (l:boundedxn), the smallest token.**  If `x_1(t)` stays bounded
then `P_11(t) → 1` and `P_{1j}(t) → 0` for every `j ∈ [n] \ {1}`.

The source writes `j ∈ [n-1]` here, copying the range of its first sentence;
that range contains `j = 1` and would contradict `P_11(t) → 1`.  The
symmetric range is `[n] \ {1}`, and that is what is stated.

Source: arXiv:2305.05465v6, `l:boundedxn`, second sentence, corrected. -/
theorem tendsto_attention_of_bounded_first (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0))
    (hb : IsBoundedToken X 0) :
    Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) 0 0) atTop (nhds 1) ∧
      ∀ j : Idx (m + 1), j ≠ 0 →
        Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) 0 j) atTop (nhds 0) := by
  have hle := coord_mono X hX hord
  have hoff : ∀ j : Idx (m + 1), j ≠ 0 →
      Tendsto (fun t => attentionMatrix (1 : ParamMatrix 1) 1 (X t) 0 j) atTop (nhds 0) := by
    intro j hj
    have hjv : 1 ≤ (j : ℕ) := by
      have : (j : ℕ) ≠ 0 := fun h => hj (Fin.ext (by simp [h]))
      omega
    have hm : 1 < m + 1 := lt_of_le_of_lt hjv j.isLt
    set J : Idx (m + 1) := ⟨1, hm⟩
    have hJN : J ≠ 0 := fun h => by
      have := congrArg Fin.val h; simp only [J, Fin.val_zero] at this; omega
    have hJ : ∀ k : Idx (m + 1), k ≠ 0 → J ≤ k := fun k hk => by
      have : (k : ℕ) ≠ 0 := fun h => hk (Fin.ext (by simp [h]))
      rw [Fin.le_iff_val_le_val]; simp only [J]; omega
    obtain ⟨R, hR⟩ := hb
    obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant (m + 1) (Nat.succ_pos m)
    rcases unbounded_tendsto_atTop_or_atBot X hX hord J
        (not_isBoundedToken_of_ne X hX hord ⟨R, hR⟩ hJN) with h | h
    swap
    · obtain ⟨t, ht, ht0⟩ := ((h.1.eventually (eventually_lt_atBot (-R))).and
        (eventually_ge_atTop 0)).exists
      linarith [hle t ht0 0 J (Fin.zero_le J), (abs_le.1 (hR t ht0)).1]
    obtain ⟨t₁, ht₁⟩ := (h.1.eventually (eventually_gt_atTop A)).exists
    obtain ⟨c, hc, hev⟩ := exists_exp_lower_bound A hA hAeq X hX J t₁ ht₁
    have := tendsto_softmaxWeight_of_bounded (fun t k => -X t k 0)
      (fun t k => by
        convert (hasDerivAt_coord X hX t k).fun_neg using 1
        simp only [mul_neg, neg_mul, neg_neg, Finset.sum_neg_distrib])
      0 J (Fin.last m) hJN (fun t ht k => neg_le_neg (hle t ht 0 k (Fin.zero_le k)))
      (fun t ht k hk => neg_le_neg (hle t ht J k (hJ k hk)))
      (fun t ht k => neg_le_neg (hle t ht k _ (Fin.le_last k)))
      (R := R) (fun t ht => by simpa using hR t ht) hc
      (hev.mono fun t h => neg_le_neg h) j hj
    simpa only [neg_mul_neg, attentionMatrix_one_eq] using this
  refine ⟨?_, hoff⟩
  simpa using tendsto_attention_of_tendsto_others (Nat.succ_pos m) 1 1 X 0 0 (fun _ => 0) hoff

/-- The configuration pinned at the origin keeps every token bounded; used to
witness the hypotheses of both halves of `l:boundedxn`. -/
theorem isBoundedToken_zero (i : Idx n) :
    IsBoundedToken (fun _ _ => (0 : EucSpace 1)) i :=
  ⟨0, fun _ _ => by simp⟩

/-- The hypotheses of both halves of `l:boundedxn` are satisfiable. -/
example :
    IdNonrescaledDynamics (n := 1) (fun _ _ => (0 : EucSpace 1)) ∧
      IsOrderedConfig (n := 1) (fun _ => (0 : EucSpace 1)) ∧
      IsBoundedToken (fun _ _ => (0 : EucSpace 1)) (Fin.last 0) ∧
      IsBoundedToken (fun _ _ => (0 : EucSpace 1)) (0 : Idx 1) :=
  ⟨idNonrescaledDynamics_zero 1 1, isOrderedConfig_subsingleton _,
    isBoundedToken_zero _, isBoundedToken_zero _⟩

end Clusters
end Transformer
