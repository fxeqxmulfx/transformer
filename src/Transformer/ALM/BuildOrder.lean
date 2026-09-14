/-
# What the arrival order costs, and what it cannot cost

`Transformer.ALM.HullBuild` prices a build of the hull: two ordered-container
searches per key and one unit per erase, amortized to `3n(log₂ n + 1)` by
`buildCost_le`.  That theorem is stated of *one* pop sequence, and a pop
sequence is what one arrival order produced.  Nothing in it said the next
order could not be worse, and the question is not idle — `cache.rs` reads its
keys from a learned projection, so no order is guaranteed, and
`vm-rs/alm-hull/src/bin/alm-stress.rs` measures five of them on the same
262 144 keys and finds a two-fold spread, `0.069 s` to `0.153 s`.  The old
`Vec` port spread thirty-one-fold on the same table, which is why it is gone.

`buildPrices m n` is every price a build of `m` keys can be charged, over every
pop sequence of that length and so over every arrival order.  Three statements
about it: `buildPrices_le`, no order escapes the bound; `buildPrices_spread`,
two orders differ by at most one unit per key; and `buildPrices_ratio`, so the
worst order costs at most `1 + 1/(2(log₂ n + 1))` of the best — under three
per cent at the size the stress test drives (`stress_spread_le`).
`buildPrices_nontrivial` is there so that none of this is vacuous: the order
does move the price, just not far.

And on the paraboloid it does not move it at all.  `Transformer.ALM.HullCover`
proves the erase test never fires on lifted keys, so the pop sequence is forced
to be zeros and `buildPrices_paraboloid` collapses the whole set to a single
number.  The five orders of the stress table are charged *the same price*, and
the two-fold between them is therefore not comparisons: it is the memory the
comparisons walk, which is what `tree.rs` is kept for and what the vector lost.

Source: `vm-rs/alm-hull/src/bin/alm-stress.rs` (the measured table) and
`vm-rs/alm-hull/src/envelope.rs`; `transformer_vm/attention/hull2d_cht.h`,
lines 143-195.
-/

import Transformer.ALM.HullCover

namespace Transformer
namespace ALM

/-! ### Every price a build of `m` keys can be charged -/

/-- The prices `buildCost` can hand a build of `m` keys into a hull of `n`.
The arrival order decides how the erase loops distribute their work, hence
which pop sequence the build produces; quantifying over all sequences of the
right length quantifies over every order, and over more besides. -/
def buildPrices (m n : ℕ) : Set ℕ := {c | ∃ ps : List ℕ, ps.length = m ∧ buildCost ps n = c}

/-- Some order is charged something: the build that erases nothing. -/
theorem buildPrices_nonempty (m n : ℕ) : (buildPrices m n).Nonempty :=
  ⟨buildCost (List.replicate m 0) n, List.replicate m 0, List.length_replicate, rfl⟩

/-- **The searches are paid whatever the order.**  Two per key is the floor,
and the erase term can only add to it. -/
theorem buildPrices_ge {m n c : ℕ} (h : c ∈ buildPrices m n) :
    2 * m * (Nat.log 2 n + 1) ≤ c := by
  obtain ⟨ps, hps, rfl⟩ := h
  rw [← hps]
  exact Nat.le_add_right _ _

/-- **And no order escapes the bound.**  `buildCost_le` asks only for the
number of keys, so the `O(n log n)` of `Transformer.ALM.HullBuild` is a
statement about every arrival order at once and not about the one the
reference traces happen to have. -/
theorem buildPrices_le {m n c : ℕ} (h : c ∈ buildPrices m n) (hm : m ≤ n) :
    c ≤ 3 * n * (Nat.log 2 n + 1) := by
  obtain ⟨ps, hps, rfl⟩ := h
  exact buildCost_le ps n (hps ▸ hm)

/-! ### How far apart two orders can be -/

/-- **The whole of the order-dependence is one unit per key.**  Two builds of
the same keys in different orders perform the same searches; only the erase
term moves, and `pops_le_length` caps it at one erase per key.  So the spread
between orders is additive in `m` and cannot grow with `log n`. -/
theorem buildPrices_spread {m n c d : ℕ} (hc : c ∈ buildPrices m n)
    (hd : d ∈ buildPrices m n) : c ≤ d + m := by
  obtain ⟨ps, hps, rfl⟩ := hc
  obtain ⟨qs, hqs, rfl⟩ := hd
  have ha : (runState ps).2 ≤ m := hps ▸ pops_le_length ps
  unfold buildCost
  rw [hps, hqs]
  omega

/-- **So relatively the worst order costs `1 + 1/(2(log₂ n + 1))` of the
best.**  The floor of `buildPrices_ge` is `2m(log₂ n + 1)` and the spread of
`buildPrices_spread` is `m`, so the ratio is bounded by the two together —
stated without division, as a comparison of products. -/
theorem buildPrices_ratio {m n c d : ℕ} (hc : c ∈ buildPrices m n)
    (hd : d ∈ buildPrices m n) :
    c * (2 * (Nat.log 2 n + 1)) ≤ d * (2 * (Nat.log 2 n + 1) + 1) := by
  obtain ⟨ps, hps, rfl⟩ := hc
  obtain ⟨qs, hqs, rfl⟩ := hd
  have ha : (runState ps).2 ≤ m := hps ▸ pops_le_length ps
  unfold buildCost
  rw [hps, hqs]
  nlinarith [ha, Nat.zero_le (runState qs).2, Nat.zero_le (Nat.log 2 n + 1),
    Nat.zero_le m]

/-- **And it does move.**  Three keys whose last insertion erases two lines
against three that erase none: the two bounds above are not bounds on a
constant. -/
theorem buildPrices_nontrivial (n : ℕ) :
    ∃ c ∈ buildPrices 3 n, ∃ d ∈ buildPrices 3 n, c ≠ d := by
  refine ⟨buildCost [0, 0, 2] n, ⟨[0, 0, 2], rfl, rfl⟩,
    buildCost [0, 0, 0] n, ⟨[0, 0, 0], rfl, rfl⟩, ?_⟩
  have h₁ : (runState [0, 0, 2]).2 = 2 := rfl
  have h₂ : (runState [0, 0, 0]).2 = 0 := rfl
  unfold buildCost
  rw [h₁, h₂]
  simp only [List.length_cons, List.length_nil]
  omega

/-- The hypotheses are satisfiable at once: at three keys into a hull of three
the two prices of `buildPrices_nontrivial` are `14` and `12`, both inside the
`18` of `buildPrices_le`. -/
example : buildCost [0, 0, 2] 3 ∈ buildPrices 3 3 ∧ buildCost [0, 0, 0] 3 ∈ buildPrices 3 3 ∧
    buildCost [0, 0, 2] 3 ≤ 3 * 3 * (Nat.log 2 3 + 1) :=
  ⟨⟨[0, 0, 2], rfl, rfl⟩, ⟨[0, 0, 0], rfl, rfl⟩, buildCost_le [0, 0, 2] 3 (by norm_num)⟩

/-! ### At the size the stress test drives -/

/-- The stress table's `262 144` is `2^18`. -/
theorem log_two_stress : Nat.log 2 262144 = 18 := by
  rw [show (262144 : ℕ) = 2 ^ 18 by norm_num, Nat.log_pow (by norm_num)]

/-- **Under three per cent, against a measured two-fold.**  At the size
`alm-stress` reports, the tariff permits the worst arrival order to cost
`39/38` of the best.  The measured spread is `0.153 / 0.069`, a factor of
`2.2`, so whatever the five orders differ in, it is not the comparisons this
model counts. -/
theorem stress_spread_le {c d : ℕ} (hc : c ∈ buildPrices 262144 262144)
    (hd : d ∈ buildPrices 262144 262144) : c * 38 ≤ d * 39 := by
  have h := buildPrices_ratio hc hd
  rw [log_two_stress] at h
  norm_num at h
  exact h

/-- And in absolute terms every one of the five orders is charged under
`3 · 262144 · 19`. -/
theorem stress_le {c : ℕ} (hc : c ∈ buildPrices 262144 262144) :
    c ≤ 3 * 262144 * 19 := by
  have h := buildPrices_le hc (le_refl _)
  rwa [log_two_stress] at h

/-- The hypotheses are satisfiable at that size: the build that erases
nothing. -/
example : buildCost (List.replicate 262144 0) 262144 ∈ buildPrices 262144 262144 :=
  ⟨List.replicate 262144 0, List.length_replicate, rfl⟩

/-! ### And on the paraboloid the order costs nothing at all -/

/-- **One price, not a band.**  `Transformer.ALM.HullCover` proves the erase
test never fires on lifted keys, so every arrival order produces the same pop
sequence — zeros — and the set of prices collapses to a point.  The five orders
of the stress table are charged identically, and the two-fold between them is
memory traffic rather than work this tariff can see. -/
theorem buildPrices_paraboloid (m n : ℕ) :
    {c | ∃ ps : List ℕ, ps.length = m ∧ (∀ p ∈ ps, p = 0) ∧ buildCost ps n = c}
      = {2 * m * (Nat.log 2 n + 1)} := by
  ext c
  constructor
  · rintro ⟨ps, hps, hno, rfl⟩
    rw [buildCost_of_no_erase ps n hno, hps]
    rfl
  · rintro rfl
    exact ⟨List.replicate m 0, List.length_replicate,
      fun p hp => List.eq_of_mem_replicate hp, by
        rw [buildCost_of_no_erase _ n fun p hp => List.eq_of_mem_replicate hp,
          List.length_replicate]⟩

/-- The hypotheses are satisfiable, and of the data the machine builds from:
three distinct keys, whose lifts no erase rule touches. -/
example : ([0, 0, 0] : List ℕ).length = 3 ∧ (∀ p ∈ ([0, 0, 0] : List ℕ), p = 0) ∧
    buildCost [0, 0, 0] 3 = 2 * 3 * (Nat.log 2 3 + 1) :=
  ⟨rfl, by decide, buildCost_of_no_erase [0, 0, 0] 3 (by decide)⟩

end ALM
end Transformer
