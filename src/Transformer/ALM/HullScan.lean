/-
# The merge walk is a constant-time step, not a scan

After the binary search, `HullHalf::query`
(`transformer_vm/attention/hull2d_cht.h`, lines 277-303) does something the
cost accounting of `Transformer.ALM.BinSearch` never mentions: it walks left
from the winning line while the neighbour's score equals the best, then walks
right on the same condition, merging every tied line's `HullMeta` into one.
Written as it stands the walk is unbounded — two `while` loops over a
`std::set` — and the `log₂ n + 1` of `bcount_le_log` would be a lie if the
walks could run long.

They cannot, and the reason is that the score is a strictly concave function
of the key.  `tie_iff_midpoint` says two distinct keys tie only at the
midpoint query, so a third key would have to satisfy `k₂ = 2q - k₁ = k₃`:
`not_tie_three` rules it out.  Hence:

* `argmaxSet_card_le_two` — at most two lines are ever tied, whatever the
  query and however many keys are stored;
* `tie_adjacent` — and when two are, no key lies between them, so they are
  neighbours in the sorted order the hull keeps;
* `scan_left_step` / `scan_right_step` — therefore each `while` loop executes
  at most one iteration, and `scan_merge_count_le_one` says `combined` is the
  merge of at most two `Meta`s.

So the walk is `O(1)` on top of the search, and the published bound stands.
What that merge resolves to is `Transformer.ALM.HullResolve`.  The sorted order
is `Transformer.ALM.KeyOrder`'s; the metas are `Transformer.ALM.TieBreak`'s.
-/

import Transformer.ALM.KeyOrder
import Transformer.ALM.TieBreak

namespace Transformer
namespace ALM

open Classical

variable {n : ℕ}

/-! ### Ties among scalar keys -/

/-- The score the hull compares, as a function of the key: `2kq - k²`, which
is `lineEval (liftKey k) q` of `Transformer.ALM.Duality`. -/
theorem lineEval_liftKey (k q : ℝ) : lineEval (liftKey k) q = 2 * k * q - k ^ 2 := by
  unfold lineEval liftKey
  ring

/-- **Two keys tie only at their midpoint.**  The real-key form of
`sScore_eq_iff` of `Transformer.ALM.TieBreak`: the hull's `s == best_score`
test succeeds for exactly one query per pair of lines. -/
theorem tie_iff_midpoint {k₁ k₂ : ℝ} (q : ℝ) (hne : k₁ ≠ k₂) :
    lineEval (liftKey k₁) q = lineEval (liftKey k₂) q ↔ 2 * q = k₁ + k₂ := by
  rw [lineEval_liftKey, lineEval_liftKey]
  constructor
  · intro h
    have hsub : (k₁ - k₂) * (2 * q - (k₁ + k₂)) = 0 := by nlinarith [h]
    rcases mul_eq_zero.mp hsub with h₀ | h₀
    · exact absurd (sub_eq_zero.mp h₀) hne
    · linarith
  · intro h; linear_combination (k₁ - k₂) * h

/-- The hypothesis is satisfiable: two distinct keys, tying at their
midpoint. -/
example : lineEval (liftKey (0 : ℝ)) 1 = lineEval (liftKey (2 : ℝ)) 1 := by
  rw [tie_iff_midpoint 1 (by norm_num)]
  norm_num

/-- **Three lines never tie.**  Strict concavity in one sentence: two ties
with the same query force `k₂ = 2q - k₁ = k₃`. -/
theorem not_tie_three {k₁ k₂ k₃ q : ℝ} (h₁₂ : k₁ ≠ k₂) (h₁₃ : k₁ ≠ k₃) (h₂₃ : k₂ ≠ k₃)
    (t₁₂ : lineEval (liftKey k₁) q = lineEval (liftKey k₂) q)
    (t₁₃ : lineEval (liftKey k₁) q = lineEval (liftKey k₃) q) : False := by
  have e₁₂ := (tie_iff_midpoint q h₁₂).mp t₁₂
  have e₁₃ := (tie_iff_midpoint q h₁₃).mp t₁₃
  exact h₂₃ (by linarith)

/-- The hypotheses are satisfiable: three distinct keys, two of which tie. -/
example : (0 : ℝ) ≠ 2 ∧ (0 : ℝ) ≠ 3 ∧ (2 : ℝ) ≠ 3 ∧
    lineEval (liftKey (0 : ℝ)) 1 = lineEval (liftKey (2 : ℝ)) 1 :=
  ⟨by norm_num, by norm_num, by norm_num, by rw [tie_iff_midpoint 1 (by norm_num)]; norm_num⟩

/-- **A tied pair has nothing between it.**  A key strictly inside the
interval is strictly closer to the midpoint, so it beats both — it would have
been the winner, and the walk never reaches past it. -/
theorem tie_adjacent {k₁ k k₂ q : ℝ} (h₁ : k₁ < k) (h₂ : k < k₂)
    (ht : lineEval (liftKey k₁) q = lineEval (liftKey k₂) q)
    (hne : k₁ ≠ k₂) :
    lineEval (liftKey k₁) q < lineEval (liftKey k) q := by
  have hmid := (tie_iff_midpoint q hne).mp ht
  rw [lineEval_liftKey, lineEval_liftKey]
  nlinarith [mul_pos (sub_pos.mpr h₁) (sub_pos.mpr h₂)]

/-- The hypotheses are satisfiable: `0 < 1 < 2`, tied at the query `1`. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧
    lineEval (liftKey (0 : ℝ)) 1 = lineEval (liftKey (2 : ℝ)) 1 ∧ (0 : ℝ) ≠ 2 :=
  ⟨by norm_num, by norm_num, by rw [tie_iff_midpoint 1 (by norm_num)]; norm_num, by norm_num⟩

/-! ### What the walk can reach -/

variable (K : ℕ → ℝ) (q : ℝ) (N : ℕ)

/-- The lines the merge walk would like to collect: every index in range whose
score equals the best. -/
noncomputable def argmaxSet : Finset ℕ :=
  (Finset.range (N + 1)).filter
    (fun j => ∀ i ≤ N, lineEval (liftKey (K i)) q ≤ lineEval (liftKey (K j)) q)

lemma mem_argmaxSet {j : ℕ} :
    j ∈ argmaxSet K q N ↔
      j ≤ N ∧ ∀ i ≤ N, lineEval (liftKey (K i)) q ≤ lineEval (liftKey (K j)) q := by
  simp [argmaxSet]

/-- Two winners score the same, by antisymmetry. -/
lemma argmaxSet_tie {j₁ j₂ : ℕ} (h₁ : j₁ ∈ argmaxSet K q N) (h₂ : j₂ ∈ argmaxSet K q N) :
    lineEval (liftKey (K j₁)) q = lineEval (liftKey (K j₂)) q := by
  rw [mem_argmaxSet] at h₁ h₂
  exact le_antisymm (h₂.2 _ h₁.1) (h₁.2 _ h₂.1)

/-- **At most two lines are ever tied.**  However many keys the hull holds and
whatever the query, `combined` merges at most two `Meta`s. -/
theorem argmaxSet_card_le_two (hstep : ∀ j, K j < K (j + 1)) :
    (argmaxSet K q N).card ≤ 2 := by
  rcases Nat.lt_or_ge (argmaxSet K q N).card 3 with hlt | hge
  · omega
  exfalso
  obtain ⟨s, hs, hcard3⟩ := Finset.exists_subset_card_eq hge
  obtain ⟨a, b, c, hab, hac, hbc, rfl⟩ := Finset.card_eq_three.mp hcard3
  have hmono : StrictMono K := strictMono_nat_of_lt_succ hstep
  have ha := hs (show a ∈ ({a, b, c} : Finset ℕ) by simp)
  have hb := hs (show b ∈ ({a, b, c} : Finset ℕ) by simp)
  have hc := hs (show c ∈ ({a, b, c} : Finset ℕ) by simp)
  exact not_tie_three (fun h => hab (hmono.injective h)) (fun h => hac (hmono.injective h))
    (fun h => hbc (hmono.injective h)) (argmaxSet_tie K q N ha hb)
    (argmaxSet_tie K q N ha hc)

/-- **And two winners are neighbours.**  With the keys sorted, no index lies
strictly between two tied ones, so `std::prev` and `std::next` already reach
everything the walk can collect. -/
theorem argmaxSet_adjacent (hstep : ∀ j, K j < K (j + 1))
    {j₁ j₂ : ℕ} (h₁ : j₁ ∈ argmaxSet K q N) (h₂ : j₂ ∈ argmaxSet K q N)
    (hlt : j₁ < j₂) : j₁ + 1 = j₂ := by
  have hmono : StrictMono K := strictMono_nat_of_lt_succ hstep
  by_contra hne
  have hmid : j₁ + 1 < j₂ := lt_of_le_of_ne (Nat.succ_le_of_lt hlt) hne
  have hbound := (mem_argmaxSet K q N).mp h₂
  have h1b := (mem_argmaxSet K q N).mp h₁
  have hwin : lineEval (liftKey (K j₁)) q < lineEval (liftKey (K (j₁ + 1))) q :=
    tie_adjacent (hmono (Nat.lt_succ_self j₁)) (hmono hmid)
      (argmaxSet_tie K q N h₁ h₂) (fun h => absurd (hmono.injective h) (Nat.ne_of_lt hlt))
  exact absurd (h1b.2 (j₁ + 1) (le_trans (le_of_lt hmid) hbound.1)) (not_le.mpr hwin)

/-! ### So each `while` loop runs at most once -/

/-- **The left walk takes at most one step.**  `itL` moves from `best_it` to
`std::prev(best_it)` and then the test fails. -/
theorem scan_left_step (hstep : ∀ j, K j < K (j + 1)) {j b : ℕ}
    (hj : j ∈ argmaxSet K q N) (hb : b ∈ argmaxSet K q N) (hlt : j < b) : j + 1 = b :=
  argmaxSet_adjacent K q N hstep hj hb hlt

/-- **The right walk takes at most one step.** -/
theorem scan_right_step (hstep : ∀ j, K j < K (j + 1)) {j b : ℕ}
    (hb : b ∈ argmaxSet K q N) (hj : j ∈ argmaxSet K q N) (hlt : b < j) : b + 1 = j :=
  argmaxSet_adjacent K q N hstep hb hj hlt

/-- **The walk's cost.**  Besides the winner there is at most one tied line,
so the two `while` loops together perform at most one `merge`: the walk is
`O(1)` on top of the binary search, and `bcount_le_log` remains the cost of a
query. -/
theorem scan_merge_count_le_one (hstep : ∀ j, K j < K (j + 1)) {b : ℕ}
    (hb : b ∈ argmaxSet K q N) : ((argmaxSet K q N).erase b).card ≤ 1 := by
  have hcard := argmaxSet_card_le_two K q N hstep
  have herase := Finset.card_erase_of_mem hb
  omega

/-- **And exactly which lines it collects.**  If a neighbour ties, the winner
and that neighbour are all there is: the walk stops with the pair, never
continuing into a third line. -/
theorem argmaxSet_eq_pair (hstep : ∀ j, K j < K (j + 1)) {b : ℕ}
    (hb : b ∈ argmaxSet K q N) (hb1 : b + 1 ∈ argmaxSet K q N) :
    argmaxSet K q N = {b, b + 1} := by
  refine Finset.eq_of_subset_of_card_le ?_ ?_
  · intro j hj
    rcases lt_trichotomy j b with hlt | rfl | hgt
    · exfalso
      have h2 := argmaxSet_adjacent K q N hstep hj hb1 (by omega)
      omega
    · simp
    · have := argmaxSet_adjacent K q N hstep hb hj hgt
      simp [← this]
  · rw [Finset.card_pair (by omega)]
    exact Finset.one_lt_card.mpr ⟨b, hb, b + 1, hb1, by omega⟩

/-- The hypotheses of the walk are satisfiable, and not vacuously: the keys
`j ↦ j` are sorted, and at the query `1/2` the lines `0` and `1` really do
tie, so `argmaxSet` really is a pair and the merge walk really does run. -/
example :
    (∀ j : ℕ, (fun i : ℕ => (i : ℝ)) j < (fun i : ℕ => (i : ℝ)) (j + 1)) ∧
      (0 : ℕ) ∈ argmaxSet (fun i : ℕ => (i : ℝ)) (1 / 2) 1 ∧
      (1 : ℕ) ∈ argmaxSet (fun i : ℕ => (i : ℝ)) (1 / 2) 1 := by
  refine ⟨fun j => by push_cast; linarith, ?_, ?_⟩ <;>
  · rw [mem_argmaxSet]
    refine ⟨by omega, fun i hi => ?_⟩
    interval_cases i <;> simp [lineEval_liftKey]

end ALM
end Transformer
