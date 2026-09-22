/-
# The emergence of clusters in self-attention dynamics — `l:onlyone`

§7 of arXiv:2305.05465v6: at most one token of a solution of `e:Idnonresca`
in `d = 1` stays bounded.

The source states `#ℬ ∈ {0,1}` for `ℬ = {i ∈ [n] : x_i ∈ L^∞([0,+∞))}`; for a
cardinality that is `#ℬ ≤ 1`, which is what is carried.

The proof follows the source's case split on the extreme tokens, each of
which is bounded or runs off to `±∞` (`unbounded_tendsto_atTop_or_atBot`):

* both extremes bounded — impossible, the configuration spreads
  (`not_gap_of_all_bounded`);
* `x_n → +∞` — two bounded tokens cannot keep their gap
  (`not_gap_of_bounded`); `x_1 → -∞` is the mirror image under `x ↦ -x`;
* `x_n → -∞` or `x_1 → +∞` — impossible beside a bounded token, by the order.

Source: arXiv:2305.05465v6, `l:onlyone` and its proof.
-/

import Transformer.Clusters.Section7_OnlyOneSpread
import Mathlib.Data.Set.Card

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- **Lemma (l:onlyone).**  At most one token stays uniformly bounded:
`#ℬ ∈ {0,1}` for `ℬ = {i ∈ [n] : x_i ∈ L^∞([0,+∞))}`, carried as `#ℬ ≤ 1`.

Source: arXiv:2305.05465v6, `l:onlyone`. -/
theorem ncard_boundedTokens_le_one (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) :
    {i : Idx (m + 1) | IsBoundedToken X i}.ncard ≤ 1 := by
  set x : ℝ → Idx (m + 1) → ℝ := fun t i => X t i 0
  have hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t :=
    fun t k => hasDerivAt_coord X hX t k
  obtain ⟨g, hg, hgap⟩ := exists_gap X hX hord
  have hle : ∀ t, 0 ≤ t → ∀ i j : Idx (m + 1), i ≤ j → x t i ≤ x t j := by
    intro t ht i j hij
    rcases hij.lt_or_eq with h | h
    · exact (isOrderedConfig_of_nonneg X hX hord t ht i j h).le
    · rw [h]
  have htok : ∀ j, (∃ R, ∀ t, 0 ≤ t → |x t j| ≤ R) ∨ Tendsto (fun t => x t j) atTop atTop ∨
      Tendsto (fun t => x t j) atTop atBot := by
    intro j
    by_cases hb : IsBoundedToken X j
    · exact Or.inl hb
    · rcases unbounded_tendsto_atTop_or_atBot X hX hord j hb with h | h
      · exact Or.inr (Or.inl h.1)
      · exact Or.inr (Or.inr h.1)
  have key : ∀ i j : Idx (m + 1), i < j → IsBoundedToken X i → IsBoundedToken X j → False := by
    intro i j hij hi hj
    have hgij : ∀ t, 0 ≤ t → x t i + g ≤ x t j := fun t ht => hgap t ht i j hij
    obtain ⟨Ri, hRi⟩ := hi
    obtain ⟨Rj, hRj⟩ := hj
    rcases htok (Fin.last m) with ⟨RL, hRL⟩ | hL | hL
    · rcases htok 0 with ⟨R0, hR0⟩ | h0 | h0
      · refine not_gap_of_all_bounded x hder i j (R := max R0 RL) (fun t ht k => ?_) hg hgij
        have h1 := hle t ht 0 k (Fin.zero_le k)
        have h2 := hle t ht k (Fin.last m) (Fin.le_last k)
        have h3 := abs_le.1 (hR0 t ht)
        have h4 := abs_le.1 (hRL t ht)
        exact abs_le.2 ⟨by linarith [le_max_left R0 RL], by linarith [le_max_right R0 RL]⟩
      · obtain ⟨t, ht, ht0⟩ :=
          ((h0.eventually (eventually_gt_atTop Ri)).and (eventually_ge_atTop 0)).exists
        linarith [hle t ht0 0 i (Fin.zero_le i), (abs_le.1 (hRi t ht0)).2]
      · refine not_gap_of_bounded (fun t k => -x t k) (fun t k => ?_) (Fin.last m) 0
          (fun t ht k => neg_le_neg (hle t ht k (Fin.last m) (Fin.le_last k)))
          (fun t ht k => neg_le_neg (hle t ht 0 k (Fin.zero_le k)))
          (tendsto_neg_atBot_atTop.comp h0) (fun k => ?_) j i
          ⟨Rj, fun t ht => by simpa using hRj t ht⟩ ⟨Ri, fun t ht => by simpa using hRi t ht⟩ hg
          (fun t ht => by linarith [hgij t ht])
        · convert (hder t k).fun_neg using 1
          simp only [mul_neg, neg_mul, neg_neg, Finset.sum_neg_distrib]
        · rcases htok k with ⟨R, hR⟩ | h | h
          · exact Or.inl ⟨R, fun t ht => by simpa using hR t ht⟩
          · exact Or.inr (Or.inr (tendsto_neg_atTop_atBot.comp h))
          · exact Or.inr (Or.inl (tendsto_neg_atBot_atTop.comp h))
    · exact not_gap_of_bounded x hder 0 (Fin.last m)
        (fun t ht k => hle t ht 0 k (Fin.zero_le k))
        (fun t ht k => hle t ht k (Fin.last m) (Fin.le_last k)) hL htok i j
        ⟨Ri, hRi⟩ ⟨Rj, hRj⟩ hg hgij
    · obtain ⟨t, ht, ht0⟩ :=
        ((hL.eventually (eventually_lt_atBot (-Rj))).and (eventually_ge_atTop 0)).exists
      linarith [hle t ht0 j (Fin.last m) (Fin.le_last j), (abs_le.1 (hRj t ht0)).1]
  by_contra hcon
  push Not at hcon
  obtain ⟨a, ha, b, hb, hab⟩ := (Set.one_lt_ncard_iff_nontrivial_and_finite.1 hcon).1
  rcases lt_or_gt_of_ne hab with h | h
  · exact key a b h ha hb
  · exact key b a h hb ha

/-- The hypotheses of `ncard_boundedTokens_le_one` are satisfiable. -/
example :
    IdNonrescaledDynamics (n := 1) (fun _ _ => (0 : EucSpace 1)) ∧
      IsOrderedConfig (n := 1) (fun _ => (0 : EucSpace 1)) :=
  ⟨idNonrescaledDynamics_zero 1 1, isOrderedConfig_subsingleton _⟩

end Clusters
end Transformer
