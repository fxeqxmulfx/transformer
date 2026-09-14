/-
# What this port pays, which is less than the tariff charges

`Transformer.ALM.HullBuild` prices a build at two ordered-container searches
per key, and `Transformer.ALM.BuildOrder` shows no arrival order escapes that.
Both are tariffs for the C++ container: `std::map::insert` after a
`lower_bound` searches a second time, because it is handed a hint and not a
position.

`vm-rs/alm-hull/src/envelope.rs` does not.  One `lower_bound_slope` per
`add_line`, and everything after it — `next`, `prev`, `erase`,
`insert_before` — is a cursor walk from the node that search returned.  And
`vm-rs/alm-hull/src/tree.rs` holds the two ends of the envelope in a field, so
`lower_bound_slope` answers without descending at all whenever the arriving
slope falls outside the span already covered.  `alm-stress` counts those: of
its five arrival orders, three descend on no key whatsoever.

So this file prices what the port runs.  `portCost` charges one search per
key, and that search is a logarithm only when the key lands strictly inside
the span — `stepCost`.  `portCost_le_buildCost` is the claim `envelope.rs`
makes and nothing proved; `portCost_le` lands at `n(log₂ n + 1) + n`, a third
under `buildCost_le`; and `portCost_of_all_ends` is linear with no logarithm
at all, which is the near-sorted path every reference trace takes.

That sharpness costs something, and the cost is the point.  `buildCost` is
blind to the fast path, so on the paraboloid it collapses to one number
(`buildPrices_paraboloid`) and the spread `alm-stress` measures has to be
charged to memory.  A tariff that can see the fast path sees a band instead:
`portCost_moves_on_the_paraboloid` produces two orders of the same lifted keys
whose prices differ, and `portPrices_ratio` bounds the band by `log₂ n + 2` —
`20` at the stress size, against a measured `4.1`.  Most of the measured
spread is comparisons after all; the tariff simply could not see them.

Source: `vm-rs/alm-hull/src/envelope.rs` (`add_line`),
`vm-rs/alm-hull/src/tree.rs` (`lower_bound_slope` and the cached `ends`), and
`vm-rs/alm-hull/src/bin/alm-stress.rs` (the measured table and its descent
columns).
-/

import Transformer.ALM.BuildOrder

namespace Transformer
namespace ALM

/-! ### One insertion's search -/

/-- What one `add_line` pays to find its insertion point in a container of at
most `2^(L-1)` lines: `L` comparisons when it has to descend, and one when the
cached ends answer it.  `atEnd` is the test `lower_bound_slope` opens with —
the arriving slope is above every slope stored, or below every one. -/
def stepCost (L : ℕ) (atEnd : Bool) : ℕ := if atEnd then 1 else L

lemma one_le_stepCost {L : ℕ} (hL : 1 ≤ L) (e : Bool) : 1 ≤ stepCost L e := by
  unfold stepCost; cases e <;> simp [hL]

/-- The hypothesis is satisfiable, at every size a container can have: the
comparison count `Transformer.ALM.BinSearch` charges is never zero. -/
example (n : ℕ) : 1 ≤ Nat.log 2 n + 1 := Nat.le_add_left 1 (Nat.log 2 n)

/-- The searches of a whole build, one flag per key in insertion order. -/
def searchCost (n : ℕ) (es : List Bool) : ℕ :=
  (es.map (stepCost (Nat.log 2 n + 1))).sum

/-- Every key pays for its own search, whether it descends or not. -/
lemma searchCost_ge (n : ℕ) (es : List Bool) : es.length ≤ searchCost n es := by
  induction es with
  | nil => simp [searchCost]
  | cons e es ih =>
      have h := one_le_stepCost (Nat.le_add_left 1 (Nat.log 2 n)) e
      simp only [searchCost, List.map_cons, List.sum_cons, List.length_cons] at *
      omega

/-- And none pays more than a full descent. -/
lemma searchCost_le (n : ℕ) (es : List Bool) :
    searchCost n es ≤ es.length * (Nat.log 2 n + 1) := by
  induction es with
  | nil => simp [searchCost]
  | cons e es ih =>
      have h : stepCost (Nat.log 2 n + 1) e ≤ Nat.log 2 n + 1 := by
        unfold stepCost; cases e <;> simp
      simp only [searchCost, List.map_cons, List.sum_cons, List.length_cons] at *
      calc stepCost (Nat.log 2 n + 1) e + (es.map (stepCost (Nat.log 2 n + 1))).sum
          ≤ (Nat.log 2 n + 1) + es.length * (Nat.log 2 n + 1) := Nat.add_le_add h ih
        _ = (es.length + 1) * (Nat.log 2 n + 1) := by ring

/-- The whole build, as this port runs it: one search per key, plus the erase
loops, which `pops_le_length` has already amortized. -/
def portCost (n : ℕ) (es : List Bool) (ps : List ℕ) : ℕ :=
  searchCost n es + (runState ps).2

/-! ### The port is cheaper than the tariff -/

/-- **One search per key, not two.**  This is what the second `lower_bound`
of `std::map::insert` costs and this port does not spend: `insert_before` is
handed the cursor the first search returned.  The claim is made in
`envelope.rs`; here it is as an inequality against the tariff
`hullIndex.build` is still declared with. -/
theorem portCost_le_buildCost (n : ℕ) (es : List Bool) (ps : List ℕ)
    (h : es.length = ps.length) : portCost n es ps ≤ buildCost ps n := by
  have hs := searchCost_le n es
  rw [h] at hs
  have hdouble : 2 * ps.length * (Nat.log 2 n + 1)
      = ps.length * (Nat.log 2 n + 1) + ps.length * (Nat.log 2 n + 1) := by ring
  unfold portCost buildCost
  omega

/-- The hypothesis is satisfiable and the inequality is strict: two lifted
keys arriving in order pay `2`, where the tariff charges `12`. -/
example : portCost 4 [true, true] [0, 0] < buildCost [0, 0] 4 := by
  have hlog : Nat.log 2 4 = 2 := by
    rw [show (4 : ℕ) = 2 ^ 2 by norm_num, Nat.log_pow (by norm_num)]
  have hs : searchCost 4 [true, true] = 2 := by
    unfold searchCost stepCost; norm_num
  unfold portCost buildCost
  rw [hs, hlog]
  norm_num [runState, stepState]

/-- **And so the build is under `n(log₂ n + 1) + n`.**  A third below
`buildCost_le`, and the gap is a real one: the second search was never run. -/
theorem portCost_le {n : ℕ} {es : List Bool} {ps : List ℕ}
    (hes : es.length ≤ n) (hps : ps.length ≤ n) :
    portCost n es ps ≤ n * (Nat.log 2 n + 1) + n := by
  have hs : searchCost n es ≤ n * (Nat.log 2 n + 1) :=
    le_trans (searchCost_le n es) (Nat.mul_le_mul_right _ hes)
  have hp : (runState ps).2 ≤ n := le_trans (pops_le_length ps) hps
  unfold portCost
  omega

/-- The hypotheses are satisfiable, and the bound is not the trivial one: two
keys into a hull of four are charged `2 · 3 + 2 = 8`, where the two of them
really pay `2`. -/
example : ([true, true] : List Bool).length ≤ 4 ∧ ([0, 0] : List ℕ).length ≤ 4 ∧
    portCost 4 [true, true] [0, 0] ≤ 4 * (Nat.log 2 4 + 1) + 4 :=
  ⟨by norm_num, by norm_num, portCost_le (by norm_num) (by norm_num)⟩

/-! ### And on the near-sorted path there is no logarithm at all -/

/-- When no key descends, the searches cost one apiece. -/
theorem searchCost_of_all_ends {n : ℕ} {es : List Bool} (h : ∀ e ∈ es, e = true) :
    searchCost n es = es.length := by
  induction es with
  | nil => simp [searchCost]
  | cons e es ih =>
      have he : e = true := h e (List.mem_cons_self ..)
      simp only [searchCost, List.map_cons, List.sum_cons, List.length_cons] at *
      rw [he, ih fun a ha => h a (List.mem_cons_of_mem e ha)]
      unfold stepCost
      norm_num
      omega

/-- **Linear, with no logarithm.**  On an arrival order whose every key lands
outside the span already covered — `ascending`, `descending` and `inside-out`
of the stress table, which `alm-stress` now counts and finds descend on no key
at all — the cached ends answer every search, and a build of lifted keys costs
one unit each.  `tree.rs` keeps those two cursors for exactly this. -/
theorem portCost_of_all_ends {n : ℕ} {es : List Bool} {ps : List ℕ}
    (h : ∀ e ∈ es, e = true) (hp : ∀ p ∈ ps, p = 0) :
    portCost n es ps = es.length := by
  unfold portCost
  rw [searchCost_of_all_ends h, runState_of_no_erase ps hp]
  rfl

/-- The hypotheses are satisfiable together, and of the data the compiler
emits: three lifted keys in position order erase nothing and descend nowhere,
and the whole build costs three. -/
example : portCost 262144 [true, true, true] [0, 0, 0] = 3 :=
  portCost_of_all_ends (by decide) (by decide)

/-! ### How far apart two arrival orders can be -/

/-- Every price this port can be charged for `m` keys: over every pattern of
descents, and so over every arrival order. -/
def portPrices (m n : ℕ) : Set ℕ :=
  {c | ∃ (es : List Bool) (ps : List ℕ),
    es.length = m ∧ ps.length = m ∧ portCost n es ps = c}

theorem portPrices_nonempty (m n : ℕ) : (portPrices m n).Nonempty :=
  ⟨portCost n (List.replicate m true) (List.replicate m 0),
    List.replicate m true, List.replicate m 0,
    List.length_replicate, List.length_replicate, rfl⟩

/-- **One unit per key is the floor.**  No order gets a key in for free. -/
theorem portPrices_ge {m n c : ℕ} (h : c ∈ portPrices m n) : m ≤ c := by
  obtain ⟨es, ps, hes, _, rfl⟩ := h
  have := searchCost_ge n es
  rw [hes] at this
  unfold portCost
  omega

/-- And `log₂ n + 2` per key is the ceiling. -/
theorem portPrices_le {m n c : ℕ} (h : c ∈ portPrices m n) :
    c ≤ m * (Nat.log 2 n + 2) := by
  obtain ⟨es, ps, hes, hps, rfl⟩ := h
  have hs := searchCost_le n es
  have hp := pops_le_length ps
  rw [hes] at hs
  rw [hps] at hp
  have hexp : m * (Nat.log 2 n + 2) = m * (Nat.log 2 n + 1) + m := by ring
  unfold portCost
  omega

/-- **So the worst order costs `log₂ n + 2` of the best.**  Not the three per
cent `buildPrices_ratio` permits: a tariff that can see the fast path can see
that missing it is a logarithm, and this is how wide the band really is. -/
theorem portPrices_ratio {m n c d : ℕ} (hc : c ∈ portPrices m n)
    (hd : d ∈ portPrices m n) : c ≤ d * (Nat.log 2 n + 2) :=
  le_trans (portPrices_le hc) (Nat.mul_le_mul_right _ (portPrices_ge hd))

/-- The hypotheses are satisfiable at once: at the stress size, the all-ends
build and a build that descends everywhere are both prices of the same keys. -/
example : portCost 262144 (List.replicate 262144 true) (List.replicate 262144 0)
      ∈ portPrices 262144 262144 ∧
    portCost 262144 (List.replicate 262144 false) (List.replicate 262144 0)
      ∈ portPrices 262144 262144 :=
  ⟨⟨_, _, List.length_replicate, List.length_replicate, rfl⟩,
    ⟨_, _, List.length_replicate, List.length_replicate, rfl⟩⟩

/-! ### And on the paraboloid the band does not collapse -/

/-- **The order still moves the price, with no erase anywhere.**
`buildPrices_paraboloid` collapses the lifted build to a single number, which
forced the measured spread to be charged entirely to memory.  It collapses
because `buildCost` cannot see a search it does not have to run.  This one
can: the same `m` lifted keys, erasing nothing either way, cost `m` when every
key lands at an end and `m(log₂ n + 1)` when none does. -/
theorem portCost_moves_on_the_paraboloid {m n : ℕ} (hm : 1 ≤ m) (hn : 2 ≤ n) :
    ∃ (es es' : List Bool) (ps : List ℕ), es.length = m ∧ es'.length = m ∧
      ps.length = m ∧ (∀ p ∈ ps, p = 0) ∧ portCost n es ps ≠ portCost n es' ps := by
  have hno : ∀ p ∈ List.replicate m (0 : ℕ), p = 0 := fun p hp => List.eq_of_mem_replicate hp
  refine ⟨List.replicate m true, List.replicate m false, List.replicate m 0,
    List.length_replicate, List.length_replicate, List.length_replicate, hno, ?_⟩
  have hlo : portCost n (List.replicate m true) (List.replicate m 0) = m := by
    rw [portCost_of_all_ends (fun e he => List.eq_of_mem_replicate he) hno,
      List.length_replicate]
  have hhi : portCost n (List.replicate m false) (List.replicate m 0)
      = m * (Nat.log 2 n + 1) := by
    unfold portCost
    rw [runState_of_no_erase _ hno]
    unfold searchCost stepCost
    rw [List.map_replicate, List.sum_replicate]
    norm_num [Nat.smul_one_eq_cast]
  have hL : 1 ≤ Nat.log 2 n := Nat.log_pos (by norm_num) hn
  have hwide : m * 2 ≤ m * (Nat.log 2 n + 1) := Nat.mul_le_mul_left m (by omega)
  rw [hlo, hhi]
  omega

/-- The hypotheses are satisfiable, at the size the stress test drives. -/
example : 1 ≤ 262144 ∧ 2 ≤ 262144 := by norm_num

/-- **Twenty, against a measured four point one.**  At the stress size the
sharper tariff permits the worst arrival order twenty times the best, where
`stress_spread_le` permitted three per cent.  The measured table sits between
them — `0.030 s` to `0.123 s` — so the descent counts explain most of the
spread and cannot explain all of it: `shuffled` and `outside-in` descend on
the same keys to within fifty and still differ by half again, and that
remainder is the memory the identical comparisons reach for. -/
theorem stress_port_ratio {c d : ℕ} (hc : c ∈ portPrices 262144 262144)
    (hd : d ∈ portPrices 262144 262144) : c ≤ d * 20 := by
  have h := portPrices_ratio hc hd
  rwa [log_two_stress] at h

/-- The hypothesis is satisfiable: the all-ends build is one of those prices,
and by `portCost_of_all_ends` it is `262144` exactly. -/
example : portCost 262144 (List.replicate 262144 true) (List.replicate 262144 0)
    ∈ portPrices 262144 262144 :=
  ⟨_, _, List.length_replicate, List.length_replicate, rfl⟩

end ALM
end Transformer
