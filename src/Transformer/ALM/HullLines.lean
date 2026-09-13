/-
# The search, over the lines the code actually stores

`Transformer.ALM.BinSearch` proves `hull_bsearch_isGreatest`: the binary
search returns a maximizer, and pays `log₂ n + 1` comparisons for it.  But it
says so only for the lines the *paraboloid lift* produces,
`liftKey k = (2k, -k²)`, whose breakpoint is a midpoint.  The implementation
stores no such thing: `_HullCHT` in `transformer_vm/attention/hull2d_cht.h`
holds arbitrary lines `(m, b)` and runs `lower_bound` over their intersection
abscissas `isect`.  The geometry for that is already proved —
`Transformer.ALM.Query`'s `lowerBound_isGreatest` assumes nothing but the two
invariants the file states — but no theorem ran the *procedure* on it, so
"the search is correct for the lines the code stores" was prose.

`bsearch_lines_isGreatest` is that theorem: `hull_bsearch_isGreatest` with the
lift removed, the comparison being `x ≤ isect(Lⱼ, Lⱼ₊₁)` exactly as in the
code, and the same logarithmic bound.

Three more statements cover the three branches of `query` (lines 255-315),
which `Transformer.ALM.Duality` reduces one at a time but which nothing then
searched:

* `planar_bsearch_of_pos` — `qy > 0`: the upper hull, queried at `m = qx/qy`.
* `planar_bsearch_of_neg` — `qy < 0`: the code negates `(m, b)` and asks the
  *lower* hull, which is the same search run on the negated family.  This is
  the branch `isGreatest_dot_iff_of_neg` reduced and nothing used.
* `planar_argmax_of_snd_eq_zero`, `planar_argmax_of_snd_eq_zero_neg` —
  `qy == 0`: the degenerate branch the code answers at `±INF`, where the
  maximizer is an extreme slope and no search happens at all.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 11-17 (the reduction),
95-98 and 203-215 (the invariants and `argmax`), 255-315 (`query`).
-/

import Transformer.ALM.BinSearch

namespace Transformer
namespace ALM

/-! ### The search, with no lift -/

/-- **`lower_bound` over the stored breakpoints, for arbitrary lines.**  Under
the two invariants of `hull2d_cht.h` — slopes increasing, intersection
abscissas increasing — the index the binary search returns maximizes the line
value at `x` over the whole stored range, and it cost `log₂ n + 1`
comparisons.  Source: `hull2d_cht.h`, lines 95-98 and 203-215. -/
theorem bsearch_lines_isGreatest (L : ℕ → ℝ × ℝ) (x : ℝ) (n : ℕ)
    (hslope : ∀ j, (L j).1 < (L (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1))) :
    let p : ℕ → Bool := fun j => decide (x ≤ interX (L j) (L (j + 1)))
    (∀ j ≤ n, lineEval (L j) x ≤ lineEval (L (bsearch p 0 n)) x) ∧
      bcount n ≤ Nat.log 2 n + 1 := by
  intro p
  have hp : ∀ a b, a ≤ b → p a = true → p b = true := by
    intro a b hab ha
    have := hbp a b hab
    simp only [p, decide_eq_true_eq] at ha ⊢
    linarith
  refine ⟨lowerBound_isGreatest L x (fun j _ => hslope j) (fun a b hab _ => hbp a b hab)
    (bsearch p 0 n) (by simpa using bsearch_le p 0 n) (fun j hj => ?_) (fun hin => ?_),
    bcount_le_log n⟩
  · have := bsearch_lt p hp 0 n j (Nat.zero_le j) hj
    simp only [p, decide_eq_false_iff_not, not_le] at this
    exact this.le
  · have := bsearch_ge_of_lt p 0 n (by simpa using hin)
    simpa [p] using this

/-! ### Negating a line -/

/-- The lower hull stores `(-m, -b)`; its line value is the negated one. -/
theorem lineEval_neg (k : ℝ × ℝ) (x : ℝ) : lineEval (-k) x = -lineEval k x := by
  unfold lineEval
  simp only [Prod.fst_neg, Prod.snd_neg]
  ring

/-- Negating both lines leaves their crossing abscissa where it was, so the
lower hull's breakpoints are the breakpoints of the original lines. -/
theorem interX_neg (k k' : ℝ × ℝ) : interX (-k) (-k') = interX k k' := by
  unfold interX
  simp only [Prod.fst_neg, Prod.snd_neg]
  rw [show -k'.2 - -k.2 = -(k'.2 - k.2) by ring,
    show -k.1 - -k'.1 = -(k.1 - k'.1) by ring, neg_div_neg_eq]

/-- Negating the key negates the planar score. -/
theorem dot_neg (q k : ℝ × ℝ) : dot q (-k) = -dot q k := by
  unfold dot
  simp only [Prod.fst_neg, Prod.snd_neg]
  ring

/-! ### The three branches of `query` -/

/-- **`qy > 0`.**  The upper hull, queried at `m = qx / qy`: the line the
binary search lands on is a planar argmax, at logarithmic cost.  Source:
`hull2d_cht.h`, lines 11-17 and 255-315. -/
theorem planar_bsearch_of_pos (L : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : 0 < q.2) (n : ℕ)
    (hslope : ∀ j, (L j).1 < (L (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1))) :
    let p : ℕ → Bool := fun j => decide (q.1 / q.2 ≤ interX (L j) (L (j + 1)))
    (∀ j ≤ n, dot q (L j) ≤ dot q (L (bsearch p 0 n))) ∧ bcount n ≤ Nat.log 2 n + 1 := by
  intro p
  exact ⟨fun j hj => (dot_le_dot_iff_of_pos hq _ _).mpr
    ((bsearch_lines_isGreatest L (q.1 / q.2) n hslope hbp).1 j hj), bcount_le_log n⟩

/-- **`qy < 0`.**  Here the code pushes `(-m, -b)` into a second hull and
queries that one.  `N` is what the lower hull stores, so the keys of the
instance are `-N j`; the very same search over `N` returns a planar argmax of
the original keys, because negation reverses the order twice.  Source:
`hull2d_cht.h`, lines 11-17 and 255-315. -/
theorem planar_bsearch_of_neg (N : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : q.2 < 0) (n : ℕ)
    (hslope : ∀ j, (N j).1 < (N (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (N a) (N (a + 1)) ≤ interX (N b) (N (b + 1))) :
    let p : ℕ → Bool := fun j => decide (q.1 / q.2 ≤ interX (N j) (N (j + 1)))
    (∀ j ≤ n, dot q (-N j) ≤ dot q (-N (bsearch p 0 n))) ∧ bcount n ≤ Nat.log 2 n + 1 := by
  intro p
  refine ⟨fun j hj => ?_, bcount_le_log n⟩
  have hmax := (bsearch_lines_isGreatest N (q.1 / q.2) n hslope hbp).1 j hj
  rw [dot_le_dot_iff_of_neg hq, lineEval_neg, lineEval_neg, neg_le_neg_iff]
  exact hmax

/-- The mirror of `dot_le_dot_iff_of_snd_eq_zero`: with `qy = 0` and `qx < 0`
the planar argmax is the *smallest* slope. -/
theorem dot_le_dot_iff_of_snd_eq_zero_neg {q : ℝ × ℝ} (hq : q.2 = 0) (hq1 : q.1 < 0)
    (k k' : ℝ × ℝ) : dot q k ≤ dot q k' ↔ k'.1 ≤ k.1 := by
  rw [dot_of_snd_eq_zero q k hq, dot_of_snd_eq_zero q k' hq]
  exact mul_le_mul_left_of_neg hq1

/-- **`qy == 0`, `qx > 0`.**  The intercept drops out, so the argmax is the
last stored line — the code's query at `+INF`, answered with no search.
Source: `hull2d_cht.h`, lines 255-315. -/
theorem planar_argmax_of_snd_eq_zero (L : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : q.2 = 0)
    (hq1 : 0 < q.1) (n : ℕ) (hslope : ∀ j, (L j).1 < (L (j + 1)).1) :
    ∀ j ≤ n, dot q (L j) ≤ dot q (L n) := by
  intro j hj
  rw [dot_le_dot_iff_of_snd_eq_zero hq hq1]
  exact le_of_step_lt (n := n) (K := fun i => (L i).1) (fun i _ => hslope i) hj le_rfl

/-- **`qy == 0`, `qx < 0`.**  The same branch at `-INF`: the argmax is the
first stored line.  Source: `hull2d_cht.h`, lines 255-315. -/
theorem planar_argmax_of_snd_eq_zero_neg (L : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : q.2 = 0)
    (hq1 : q.1 < 0) (n : ℕ) (hslope : ∀ j, (L j).1 < (L (j + 1)).1) :
    ∀ j ≤ n, dot q (L j) ≤ dot q (L 0) := by
  intro j hj
  rw [dot_le_dot_iff_of_snd_eq_zero_neg hq hq1]
  exact le_of_step_lt (n := n) (K := fun i => (L i).1) (fun i _ => hslope i)
    (Nat.zero_le j) hj

/-! ### The hypotheses are satisfiable -/

/-- A concrete family of lines: the lifted keys `0, 1, 2, …`. -/
private def parabLine (j : ℕ) : ℝ × ℝ := liftKey (j : ℝ)

private lemma parabLine_interX (j : ℕ) :
    interX (parabLine j) (parabLine (j + 1)) = (j : ℝ) + 1 / 2 := by
  show interX (liftKey (j : ℝ)) (liftKey ((j + 1 : ℕ) : ℝ)) = _
  rw [interX_liftKey (by push_cast; linarith)]
  push_cast
  ring

private lemma parabLine_slope (j : ℕ) : (parabLine j).1 < (parabLine (j + 1)).1 := by
  simp only [parabLine, liftKey]
  push_cast
  linarith

private lemma parabLine_bp (a b : ℕ) (hab : a ≤ b) :
    interX (parabLine a) (parabLine (a + 1)) ≤ interX (parabLine b) (parabLine (b + 1)) := by
  rw [parabLine_interX, parabLine_interX]
  have : (a : ℝ) ≤ (b : ℝ) := by exact_mod_cast hab
  linarith

/-- The invariants hold, and the search is not trivial: on these lines the
walk for `x = 2.5` over `[0, 4)` lands on the line of key `2`. -/
example :
    (∀ j, (parabLine j).1 < (parabLine (j + 1)).1) ∧
      (∀ a b, a ≤ b → interX (parabLine a) (parabLine (a + 1))
        ≤ interX (parabLine b) (parabLine (b + 1))) ∧
      bsearch (fun j => decide ((2.5 : ℝ) ≤ interX (parabLine j) (parabLine (j + 1)))) 0 4
        = 2 :=
  ⟨parabLine_slope, parabLine_bp, by
    norm_num [bsearch_succ, bsearch_zero, parabLine_interX]⟩

/-- Satisfiable for a query with positive second coordinate. -/
example : (0 : ℝ) < ((5, 2) : ℝ × ℝ).2 ∧
    (∀ j, (parabLine j).1 < (parabLine (j + 1)).1) ∧
      ∀ a b, a ≤ b → interX (parabLine a) (parabLine (a + 1))
        ≤ interX (parabLine b) (parabLine (b + 1)) :=
  ⟨by norm_num, parabLine_slope, parabLine_bp⟩

/-- And for one with negative second coordinate, the lower-hull branch. -/
example : ((5, -2) : ℝ × ℝ).2 < 0 ∧
    (∀ j, (parabLine j).1 < (parabLine (j + 1)).1) ∧
      ∀ a b, a ≤ b → interX (parabLine a) (parabLine (a + 1))
        ≤ interX (parabLine b) (parabLine (b + 1)) :=
  ⟨by norm_num, parabLine_slope, parabLine_bp⟩

/-- The degenerate branch is reachable in both directions, and the two sides
really disagree: at `q = (1, 0)` the last line wins, at `q = (-1, 0)` the
first. -/
example : ((1, 0) : ℝ × ℝ).2 = 0 ∧ (0 : ℝ) < ((1, 0) : ℝ × ℝ).1 ∧
    ((-1, 0) : ℝ × ℝ).2 = 0 ∧ ((-1, 0) : ℝ × ℝ).1 < 0 ∧
    dot (1, 0) (parabLine 0) < dot (1, 0) (parabLine 1) ∧
    dot (-1, 0) (parabLine 1) < dot (-1, 0) (parabLine 0) := by
  refine ⟨rfl, by norm_num, rfl, by norm_num, ?_, ?_⟩ <;>
    simp only [dot, parabLine, liftKey] <;> norm_num

end ALM
end Transformer
