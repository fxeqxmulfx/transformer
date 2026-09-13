/-
# What building the hull costs

`Transformer.ALM.HullIndex` priced a query and proved the price paid.  The
other half of `NNIndex` — `build` — was a number with nothing behind it, the
same kind of fiction as `freeIndex`'s zero prices in
`Transformer.ALM.Independence`.

`HullHalf::add_line` (`transformer_vm/attention/hull2d_cht.h`, lines 143-195)
does three things per key: one `lower_bound`, one `insert`, and then the two
erase loops

    while (isect(y, z)) z = lines.erase(z);
    while ((y = x) != lines.begin() && (--x)->p >= y->p) isect(x, lines.erase(y));

which can each run many times on a single call — enough to make one insertion
cost `Θ(n)`.  What keeps the build near-linear is amortization: a line is
inserted once and erased at most once, so the erases over the whole build are
at most the insertions.
`pops_add_size` is that invariant, stated exactly — pops plus the surviving
lines equal the keys seen — and `pops_le_length` is its consequence.  On the
paraboloid the loops never run at all (`liftKey_not_dominated`), and
`runState_of_no_erase` draws the conclusion that was left in that theorem's
docstring: the container ends holding one line per key, which is the range the
query's binary search is proved over.

From there `buildCost_le` charges two ordered-container searches per key, at
`Nat.log 2 n + 1` comparisons each by `Transformer.ALM.BinSearch`, plus one
unit per erase (erasing at a known iterator is amortized constant), and lands
under `3n(log₂ n + 1)` — which is what `hullIndex.build` now declares, so
`hullIndex_build_paid` is a payment and not a promise.

At `m ≠ 1` the price stays zero because nothing is built, and that is the last
section: `hullAns_eq_bfAns` and `hullIndex_agrees_with_bruteForce` say the
index is then the exhaustive scan of `Transformer.ALM.LookupIndex` outright.
The hull is the hull in dimension one and nowhere else, which is the honest
reading of `reduction_dimension_even`.
-/

import Transformer.ALM.HullIndex

namespace Transformer
namespace ALM

/-! ### The amortized invariant -/

/-- One `add_line`, as it acts on `(lines.size(), erases so far)`: the erase
loops remove `p` lines — never more than are there — and then the new line is
inserted. -/
def stepState (st : ℕ × ℕ) (p : ℕ) : ℕ × ℕ := (st.1 - p + 1, st.2 + min p st.1)

/-- A whole build: one `p` per key, in insertion order. -/
def runState (ps : List ℕ) : ℕ × ℕ := ps.foldl stepState (0, 0)

/-- **Nothing is erased twice.**  After every prefix of the build, the lines
erased so far plus the lines still standing are exactly the lines inserted so
far — a line leaves the container at most once because it entered it once. -/
theorem foldl_stepState_count (ps : List ℕ) (st : ℕ × ℕ) :
    (ps.foldl stepState st).2 + (ps.foldl stepState st).1
      = st.2 + st.1 + ps.length := by
  induction ps generalizing st with
  | nil => simp
  | cons p ps ih =>
      rw [List.foldl_cons, ih (stepState st p)]
      simp only [stepState, List.length_cons]
      omega

/-- The invariant at the start of the build, where the container is empty. -/
theorem pops_add_size (ps : List ℕ) :
    (runState ps).2 + (runState ps).1 = ps.length := by
  simpa [runState] using foldl_stepState_count ps (0, 0)

/-- **So the erase loops are amortized constant.**  However the two `while`
loops of `add_line` distribute their work, over the whole build they run at
most once per key. -/
theorem pops_le_length (ps : List ℕ) : (runState ps).2 ≤ ps.length := by
  have := pops_add_size ps
  omega

/-- And a single `add_line` really can erase many lines, so the bound above is
amortized and not per call: three keys inserted, the third erasing two. -/
example : (runState [0, 0, 2]).2 = 2 ∧ (runState [0, 0, 2]).1 = 1 := by
  constructor <;> rfl

/-! ### And on the paraboloid it never fires at all -/

/-- A build that erases nothing erases nothing, at any starting state. -/
theorem foldl_stepState_pops_of_no_erase (ps : List ℕ) (st : ℕ × ℕ)
    (h : ∀ p ∈ ps, p = 0) : (ps.foldl stepState st).2 = st.2 := by
  induction ps generalizing st with
  | nil => simp
  | cons p ps ih =>
      rw [List.foldl_cons, ih (stepState st p) (fun q hq => h q (List.mem_cons_of_mem p hq))]
      simp [stepState, h p (List.mem_cons_self ..)]

/-- **Every lifted key stays in the container.**  `liftKey_not_dominated` says
the erase test never fires on three lifted keys in increasing order; this is
what that buys: the hull ends holding one line per key inserted, so the
`lower_bound` of `argmax` really does search all of them — the range
`Transformer.ALM.Hull` and `Transformer.ALM.KeyOrder` assume it searches.

Source: `hull2d_cht.h`, lines 143-195. -/
theorem runState_of_no_erase (ps : List ℕ) (h : ∀ p ∈ ps, p = 0) :
    runState ps = (ps.length, 0) := by
  have hpop : (runState ps).2 = 0 := foldl_stepState_pops_of_no_erase ps (0, 0) h
  have hsum := pops_add_size ps
  refine Prod.ext ?_ hpop
  omega

/-- The hypothesis is the content of `liftKey_not_dominated`, and it does hold
of the lift: the middle of the three lifted keys `0, 1, 2` is somewhere
strictly above the envelope of the outer two, so no `add_line` on a sorted
lifted family erases anything. -/
example : (∀ p ∈ ([0, 0, 0] : List ℕ), p = 0) ∧
    ¬ ∀ x : ℝ, lineEval (liftKey 1) x
      ≤ max (lineEval (liftKey 0) x) (lineEval (liftKey 2) x) :=
  ⟨by decide, liftKey_not_dominated (by norm_num) (by norm_num)⟩

/-! ### The price of the build -/

/-- The comparisons a build performs: `lower_bound` and `insert` per key, each
a search of an ordered container of at most `n` lines, plus one unit for each
erase. -/
def buildCost (ps : List ℕ) (n : ℕ) : ℕ :=
  2 * ps.length * (Nat.log 2 n + 1) + (runState ps).2

/-- **The build is `O(n log n)`, amortization included.**  The erase loops
contribute `n` in total, not `n` per call. -/
theorem buildCost_le (ps : List ℕ) (n : ℕ) (h : ps.length ≤ n) :
    buildCost ps n ≤ 3 * n * (Nat.log 2 n + 1) := by
  have hpop : (runState ps).2 ≤ n := le_trans (pops_le_length ps) h
  have hlen : 2 * ps.length * (Nat.log 2 n + 1) ≤ 2 * n * (Nat.log 2 n + 1) :=
    Nat.mul_le_mul_right _ (by omega)
  have hn : n ≤ n * (Nat.log 2 n + 1) := Nat.le_mul_of_pos_right n (by omega)
  unfold buildCost
  calc 2 * ps.length * (Nat.log 2 n + 1) + (runState ps).2
      ≤ 2 * n * (Nat.log 2 n + 1) + n := Nat.add_le_add hlen hpop
    _ ≤ 2 * n * (Nat.log 2 n + 1) + n * (Nat.log 2 n + 1) := Nat.add_le_add_left hn _
    _ = 3 * n * (Nat.log 2 n + 1) := by ring

/-- The hypothesis is satisfiable: a build of three keys into a hull of size
three or more. -/
example : ([0, 0, 2] : List ℕ).length ≤ 3 := by norm_num

/-- **And on the paraboloid the price is exactly the two searches per key.**
With no erase to amortize, `buildCost` loses its third term: building the hull
of `n` lifted keys is `2n(log₂ n + 1)` comparisons and nothing else. -/
theorem buildCost_of_no_erase (ps : List ℕ) (n : ℕ) (h : ∀ p ∈ ps, p = 0) :
    buildCost ps n = 2 * ps.length * (Nat.log 2 n + 1) := by
  simp [buildCost, runState_of_no_erase ps h]

/-- **The declared price covers it.**  `hullIndex.build` is no longer a number
chosen to make an inequality go through: every build of at most `n` keys fits
inside it. -/
theorem hullIndex_build_paid (ps : List ℕ) (n : ℕ) (h : ps.length ≤ n) :
    ((buildCost ps n : ℕ) : ℝ) ≤ hullIndex.build n 1 := by
  have hb : hullIndex.build n 1 = 3 * (n : ℝ) * ((Nat.log 2 n : ℝ) + 1) := by
    show (if (1 : ℕ) = 1 then 3 * (n : ℝ) * ((Nat.log 2 n : ℝ) + 1) else 0) = _
    rw [if_pos rfl]
  rw [hb]
  have := buildCost_le ps n h
  have hcast : ((3 * n * (Nat.log 2 n + 1) : ℕ) : ℝ) = 3 * (n : ℝ) * ((Nat.log 2 n : ℝ) + 1) := by
    push_cast
    ring
  calc ((buildCost ps n : ℕ) : ℝ) ≤ ((3 * n * (Nat.log 2 n + 1) : ℕ) : ℝ) := by exact_mod_cast this
    _ = 3 * (n : ℝ) * ((Nat.log 2 n : ℝ) + 1) := hcast

/-- The hypothesis is satisfiable, and the payment is not trivial: a build of
three keys against a hull of three costs `14` and is charged `18`. -/
example : ((buildCost [0, 0, 2] 3 : ℕ) : ℝ) ≤ hullIndex.build 3 1 :=
  hullIndex_build_paid [0, 0, 2] 3 (by norm_num)


/-! ### Where the hull is actually the hull -/

/-- **Outside dimension one there is no hull.**  `hullAns` matches on the key
dimension and takes the scanning branch everywhere but `m = 1`; this is that
match, as a statement. -/
theorem hullAns_eq_bfAns : ∀ {m n : ℕ} [Nonempty (Fin n)], m ≠ 1 →
    ∀ (K : Fin n → EucSpace m) (q : EucSpace m), hullAns K q = bfAns K q
  | 0, _, _, _, _, _ => rfl
  | 1, _, _, hm, _, _ => absurd rfl hm
  | _ + 2, _, _, _, _, _ => rfl

/-- **And no saving either.**  Both prices fall back to `bruteForce`'s, so
outside dimension one `hullIndex` is the exhaustive scan under another name.
What dodges the barrier of `Transformer.ALM.Hardness` is therefore a claim
about a single dimension — `reduction_dimension_even` — and not an index that
beats a scan wherever it is asked. -/
theorem hullIndex_agrees_with_bruteForce (n m : ℕ) (hm : m ≠ 1) :
    hullIndex.build n m = bruteForce.build n m ∧
      hullIndex.query n m = bruteForce.query n m := by
  constructor
  · show (if m = 1 then 3 * (n : ℝ) * ((Nat.log 2 n : ℝ) + 1) else 0) = 0
    rw [if_neg hm]
  · show (if m = 1 then (Nat.log 2 n : ℝ) + 1 else (n : ℝ) * (m : ℝ))
      = (n : ℝ) * (m : ℝ)
    rw [if_neg hm]

/-- The hypothesis is satisfiable, and the agreement is not the empty claim:
the reduction of `Transformer.ALM.LookupIndex` queries at `d + d`, which is
even, so this is the branch it always takes. -/
example (d : ℕ) : hullIndex.query 8 (d + d + 2) = bruteForce.query 8 (d + d + 2) :=
  (hullIndex_agrees_with_bruteForce 8 (d + d + 2) (by omega)).2

end ALM
end Transformer
