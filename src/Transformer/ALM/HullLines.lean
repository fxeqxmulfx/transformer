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

Negating a line is the other half of what the code does with its stored
lines, and `Transformer.ALM.HullBranch` runs the search over the three
branches of `query` from here.

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

/-! ### The hypotheses are satisfiable -/

/-- A concrete family of lines: the lifted keys `0, 1, 2, …`, the lines
`hull2d_cht.h` stores for the scalar keys `0, 1, 2, …`.

Source: `hull2d_cht.h`, lines 11-17. -/
def parabLine (j : ℕ) : ℝ × ℝ := liftKey (j : ℝ)

/-- Its breakpoints are the midpoints `j + 1/2`. -/
lemma parabLine_interX (j : ℕ) :
    interX (parabLine j) (parabLine (j + 1)) = (j : ℝ) + 1 / 2 := by
  show interX (liftKey (j : ℝ)) (liftKey ((j + 1 : ℕ) : ℝ)) = _
  rw [interX_liftKey (by push_cast; linarith)]
  push_cast
  ring

/-- Its slopes increase. -/
lemma parabLine_slope (j : ℕ) : (parabLine j).1 < (parabLine (j + 1)).1 := by
  simp only [parabLine, liftKey]
  push_cast
  linarith

/-- And so do its breakpoints, which are the two invariants of the file. -/
lemma parabLine_bp (a b : ℕ) (hab : a ≤ b) :
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
end ALM
end Transformer
