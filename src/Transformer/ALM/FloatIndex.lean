/-
# The index survives finite precision

`Transformer.ALM.FloatHull` proves that the hull's individual comparisons are
decided correctly in floating point: `fp_query_branch` says a query separated
from a breakpoint takes the branch exact arithmetic would take.  But the
object the complexity barrier of `Transformer.ALM.Hardness` is compared
against is `hullIndex`, and no statement connected the two — `FPArith`
appeared nowhere outside `FloatHull`.  So "finite precision does not change
the answer" was proved of a comparison and asserted of an index.

Here it is proved of the index.  `fpSearch` is the binary search of
`Transformer.ALM.BinSearch` with every breakpoint computed by an `FPArith`
instead of exactly — that is, `lower_bound` as the implementation runs it.
`fpSearch_eq` says it returns the same index as the exact search whenever the
query clears the breakpoints *inside the searched window* by more than the
breakpoint error — which is all the search inspects, by `bsearch_congr` — and
`fpProbe_eq_hullProbe` carries that to `hullIndex`: the key the machine
returns in floating point is the key the index returns, so
`hullIdx_isGreatest` and the query price hold of the running code.

`fp_hullIndex_longDouble` puts numbers on it: at `long double` accuracy and
keys below `2^56`, a query further than `1/16` from every breakpoint is
answered exactly.  The excluded queries are the ones where it does not
matter — at a breakpoint the two keys score the same
(`lineEval_eq_at_midpoint`).

Source: `transformer_vm/attention/hull2d_cht.h`, lines 60-70 (`isect`) and
203-215 (the search that calls it).
-/

import Transformer.ALM.HullIndex
import Transformer.ALM.FloatHull

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### The search as the implementation runs it -/

/-- `lower_bound(q)` over the breakpoints as computed, not as defined. -/
noncomputable def fpSearch (F : FPArith) (K : ℕ → ℝ) (q : ℝ) (N : ℕ) : ℕ :=
  bsearch (fun j => decide (q ≤ F.isect (liftKey (K j)) (liftKey (K (j + 1))))) 0 N

/-- **A separated query is answered exactly.**  If every computed breakpoint
in the searched window is within `δ` of the true one and the query clears
every such breakpoint by more than `δ`, the floating-point search returns the
index the exact search returns: the same comparisons, hence the same path.
Only the window matters, which is what `bsearch_congr` supplies. -/
theorem fpSearch_eq (F : FPArith) (K : ℕ → ℝ) (q : ℝ) (N : ℕ) (δ : ℝ)
    (herr : ∀ j < N, |F.isect (liftKey (K j)) (liftKey (K (j + 1)))
      - (K j + K (j + 1)) / 2| ≤ δ)
    (hsep : ∀ j < N, δ < |q - (K j + K (j + 1)) / 2|) :
    fpSearch F K q N
      = bsearch (fun j => decide (q ≤ (K j + K (j + 1)) / 2)) 0 N := by
  refine bsearch_congr _ _ 0 N (fun j _ hlt => ?_)
  rw [Nat.zero_add] at hlt
  refine decide_eq_decide.mpr ?_
  refine cmp_of_sep (a := q) (b := (K j + K (j + 1)) / 2) (δ₁ := 0) (by simp)
    (herr j hlt) ?_
  rw [zero_add]
  exact hsep j hlt

/-- The same with the error expressed as `Transformer.ALM.FloatHull` bounds
it: a relative `u` on keys of magnitude at most `M`. -/
theorem fpSearch_eq_of_bounded (F : FPArith) (K : ℕ → ℝ) (q M : ℝ) (N : ℕ)
    (hstep : ∀ j, K j < K (j + 1)) (hb : ∀ j ≤ N, |K j| ≤ M)
    (hsep : ∀ j < N, F.u * M < |q - (K j + K (j + 1)) / 2|) :
    fpSearch F K q N
      = bsearch (fun j => decide (q ≤ (K j + K (j + 1)) / 2)) 0 N :=
  fpSearch_eq F K q N (F.u * M)
    (fun j hj => isect_liftKey_error F (ne_of_lt (hstep j)) (hb j (le_of_lt hj))
      (hb (j + 1) hj)) hsep

/-- The hypotheses are satisfiable: exact arithmetic makes no error at all,
the keys `j ↦ j` stay within `5` over the window searched, and the query `0`
clears every breakpoint in it. -/
example : fpSearch exactArith (fun j : ℕ => (j : ℝ)) 0 5
    = bsearch (fun j => decide ((0 : ℝ) ≤
        ((fun i : ℕ => (i : ℝ)) j + (fun i : ℕ => (i : ℝ)) (j + 1)) / 2)) 0 5 := by
  refine fpSearch_eq_of_bounded exactArith (fun j : ℕ => (j : ℝ)) 0 5 5
    (fun j => by push_cast; linarith) (fun j hj => ?_) (fun j _ => ?_)
  · rw [abs_of_nonneg (by positivity)]
    exact_mod_cast hj
  · have hm : (0 : ℝ) < ((j : ℝ) + ((j + 1 : ℕ) : ℝ)) / 2 := by push_cast; positivity
    simp only [zero_sub, abs_neg, exactArith]
    rwa [abs_of_pos hm, zero_mul]

/-! ### And so the index is unchanged -/

/-- The index's probe, run in floating point. -/
noncomputable def fpProbe (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ) : ℕ :=
  fpSearch F (sortedKey K) q (keyCard K - 1)

/-- **The machine in floating point answers what the index answers.**  Under
the separation condition the computed probe is the exact one, so every
statement `Transformer.ALM.HullIndex` proves about `hullIndex` — exactness and
the query price alike — is a statement about the code that runs. -/
theorem fpProbe_eq_hullProbe (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ) (q M : ℝ)
    (hb : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M)
    (hsep : ∀ j < keyCard K - 1,
      F.u * M < |q - (sortedKey K j + sortedKey K (j + 1)) / 2|) :
    fpProbe F K q = hullProbe K q :=
  fpSearch_eq_of_bounded F (sortedKey K) q M (keyCard K - 1) (sortedKey_lt_succ K) hb hsep

/-- **The same key comes back.**  What the floating-point search returns is
the key `hullIdx` names, hence the key `hullIndex.ans` returns in dimension
one. -/
theorem fp_hullIndex_key (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ) (q M : ℝ)
    (hb : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M)
    (hsep : ∀ j < keyCard K - 1,
      F.u * M < |q - (sortedKey K j + sortedKey K (j + 1)) / 2|) :
    sortedKey K (fpProbe F K q) = K (hullIdx K q) := by
  rw [fpProbe_eq_hullProbe F K q M hb hsep, hullIdx_spec K q]

/-- **And it is still an exact argmax.**  Finite precision loses no
optimality: the key the running code returns beats every stored key. -/
theorem fp_hullIndex_isGreatest (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ) (q M : ℝ)
    (hb : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M)
    (hsep : ∀ j < keyCard K - 1,
      F.u * M < |q - (sortedKey K j + sortedKey K (j + 1)) / 2|)
    (i : Fin n) :
    lineEval (liftKey (K i)) q ≤ lineEval (liftKey (sortedKey K (fpProbe F K q))) q := by
  rw [fpProbe_eq_hullProbe F K q M hb hsep]
  exact hullProbe_isGreatest K q i

/-- **The hypotheses never rule out a family of keys.**  Finitely many keys
are bounded, and the smallest of them clears every breakpoint, since each
breakpoint lies strictly above it — so for exact arithmetic the conditions of
`fpProbe_eq_hullProbe` are met by every family at the query `sortedKey K 0`,
and they are not vacuous. -/
example [Nonempty (Fin n)] (K : Fin n → ℝ) :
    (∃ M : ℝ, ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ M) ∧
      ∀ j < keyCard K - 1, exactArith.u * 0
        < |sortedKey K 0 - (sortedKey K j + sortedKey K (j + 1)) / 2| := by
  constructor
  · obtain ⟨b, -, hb⟩ := Finset.exists_max_image (Finset.range (keyCard K))
      (fun j => |sortedKey K j|) ⟨0, by simp [keyCard_pos K]⟩
    refine ⟨|sortedKey K b|, fun j hj => hb j (Finset.mem_range.mpr ?_)⟩
    have hpos := keyCard_pos K
    omega
  · intro j _
    have hmono : StrictMono (sortedKey K) := strictMono_nat_of_lt_succ (sortedKey_lt_succ K)
    have h₁ : sortedKey K 0 ≤ sortedKey K j := hmono.monotone (Nat.zero_le j)
    have h₂ : sortedKey K 0 < sortedKey K (j + 1) := hmono (Nat.succ_pos j)
    rw [show exactArith.u * 0 = 0 from by simp, abs_of_neg (by linarith)]
    linarith

/-! ### With the numbers the implementation runs on -/

/-- **`long double` answers the index exactly.**  At a relative accuracy of
`2^-60` — above the unit roundoff of the 80-bit format, with room for the
subtractions inside `isect` — and keys below `2^56`, any query further than
`1/16` from every breakpoint gets the index's own answer.  This is
`fp_longDouble` of `Transformer.ALM.FloatHull` on the query side. -/
theorem fp_hullIndex_longDouble (F : FPArith) [Nonempty (Fin n)] (K : Fin n → ℝ) (q : ℝ)
    (hu : F.u ≤ 1 / 2 ^ (60 : ℕ))
    (hb : ∀ j ≤ keyCard K - 1, |sortedKey K j| ≤ 2 ^ (56 : ℕ))
    (hsep : ∀ j < keyCard K - 1,
      1 / 16 < |q - (sortedKey K j + sortedKey K (j + 1)) / 2|) :
    fpProbe F K q = hullProbe K q := by
  refine fpProbe_eq_hullProbe F K q (2 ^ (56 : ℕ)) hb (fun j hj => ?_)
  have hpos : (0 : ℝ) < 2 ^ (56 : ℕ) := by positivity
  have h1 := mul_le_mul_of_nonneg_right hu hpos.le
  have hval : (1 / 2 ^ (60 : ℕ)) * (2 : ℝ) ^ (56 : ℕ) = 1 / 16 := by norm_num
  rw [hval] at h1
  linarith [hsep j hj]

/-- The accuracy hypothesis is satisfiable: exact arithmetic has `u = 0`. -/
example : exactArith.u ≤ 1 / 2 ^ (60 : ℕ) := by norm_num [exactArith]

end ALM
end Transformer
