/-
# The container really does hold every key

`Transformer.ALM.HullBuild` prices the build with the erase loops amortized
away, and `Transformer.ALM.Hull` proves `liftKey_not_dominated`: of three
lifted keys in increasing order the middle one is never under the envelope of
the outer two, so the test

    while (isect(y, z)) z = lines.erase(z);

never fires on the paraboloid.  That was as far as either file went, and the
conclusion it exists for — the hull of `n` keys holds `n` lines, so the
`lower_bound` of `argmax` searches all of them — lived in a docstring.

Here it is drawn.  `runState_of_no_erase` turns "no pop" into the container's
final size, `keyCard_eq_of_injective` says how many positions the query's
search covers, and `hull_covers_every_key` puts the two equal: for distinct
keys the lines the build leaves are exactly the positions
`Transformer.ALM.KeyOrder` sorts and `Transformer.ALM.Hull` searches, at a
price of two ordered-container searches per key and nothing else.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 143-195.
-/

import Transformer.ALM.HullBuild

namespace Transformer
namespace ALM

variable {n : ℕ}

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

/-- **And on the paraboloid the price is exactly the two searches per key.**
With no erase to amortize, `buildCost` loses its third term: building the hull
of `n` lifted keys is `2n(log₂ n + 1)` comparisons and nothing else. -/
theorem buildCost_of_no_erase (ps : List ℕ) (n : ℕ) (h : ∀ p ∈ ps, p = 0) :
    buildCost ps n = 2 * ps.length * (Nat.log 2 n + 1) := by
  simp [buildCost, runState_of_no_erase ps h]

/-! ### And so it holds one line per key the search covers -/

/-- **The build leaves exactly the array the query searches.**  Distinct keys,
nothing erased: the container ends with one line per key, `keyCard K` of them,
which is the length of the sorted array `hullProbe` runs its binary search
over — so the search's window is the whole family, and the price of the build
is the two searches per key with no amortized term at all.

Source: `hull2d_cht.h`, lines 143-195 (`add_line`) and 203-215 (`argmax`). -/
theorem hull_covers_every_key [Nonempty (Fin n)] (K : Fin n → ℝ)
    (hinj : Function.Injective K) (ps : List ℕ) (hlen : ps.length = n)
    (hno : ∀ p ∈ ps, p = 0) :
    (runState ps).1 = keyCard K ∧ (runState ps).2 = 0 ∧
      buildCost ps n = 2 * keyCard K * (Nat.log 2 n + 1) := by
  refine ⟨?_, ?_, ?_⟩ <;>
    simp [runState_of_no_erase ps hno, buildCost_of_no_erase ps n hno, hlen,
      keyCard_eq_of_injective K hinj]

/-- The hypotheses hold together, and of the data the machine builds from: the
three distinct keys `0, 1, 2` are inserted by three `add_line` calls that erase
nothing, because `liftKey_not_dominated` forbids the erase test from firing on
their lifts. -/
example : Function.Injective (fun j : Fin 3 => ((j : ℕ) : ℝ)) ∧
    ([0, 0, 0] : List ℕ).length = 3 ∧ (∀ p ∈ ([0, 0, 0] : List ℕ), p = 0) ∧
    ¬ ∀ x : ℝ, lineEval (liftKey 1) x
      ≤ max (lineEval (liftKey 0) x) (lineEval (liftKey 2) x) := by
  refine ⟨fun a b h => Fin.ext ?_, by norm_num, by decide,
    liftKey_not_dominated (by norm_num) (by norm_num)⟩
  simp only at h
  exact_mod_cast h

end ALM
end Transformer
