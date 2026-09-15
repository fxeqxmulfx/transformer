/-
# RASP: the immediate API of `select`, `aggregate` and `selector_width`

Weiss, Goldberg, Yahav — arXiv:2106.06981v2, "Thinking Like Transformers".

The row-level rewrites every RASP program is read through, plus the two
identities §3.1 asserts about the built-ins: `length` is *computed*, not
primitive, and `selector_width(select(tokens,tokens,==))` is the histogram.
-/

import Transformer.RASP.Defs

namespace Transformer
namespace RASP

variable {n : ℕ} {α : Type*}

@[simp] lemma mem_selected {S : Selector n} {i j : Fin n} :
    j ∈ selected S i ↔ S i j = true := by
  simp [selected]

@[simp] lemma selected_selectAll (i : Fin n) : selected (selectAll n) i = Finset.univ := by
  ext j; simp [selectAll]

@[simp] lemma card_selected_selectAll (i : Fin n) :
    (selected (selectAll n) i).card = n := by
  simp

lemma selected_selectZero (hn : 0 < n) (i : Fin n) :
    selected (selectZero n) i = {(⟨0, hn⟩ : Fin n)} := by
  ext j; simp [selectZero, Fin.ext_iff]

/-- `select_all` selects everything, so `selector_width` of it is the input
length (§3). -/
theorem selectorWidth_selectAll (i : Fin n) : selectorWidth (selectAll n) i = (n : ℝ) := by
  simp [selectorWidth]

/-- Aggregating over `select_all` averages the whole sequence (§3): this is
the shape of `frac_as = aggregate(select_all, 1 if tokens=="a" else 0)`. -/
theorem aggregate_selectAll (hn : 0 < n) (v : Seq n ℝ) (d : ℝ) (i : Fin n) :
    aggregate (selectAll n) v d i = (∑ j, v j) / (n : ℝ) := by
  simp only [aggregate, selected_selectAll, Finset.card_univ, Fintype.card_fin]
  rw [ite_eq_right hn.ne']

/-- A row that selects exactly one position aggregates to the value there. -/
theorem aggregate_of_selected_eq_singleton {S : Selector n} {i j : Fin n}
    (h : selected S i = {j}) (v : Seq n ℝ) (d : ℝ) : aggregate S v d i = v j := by
  simp [aggregate, h]

/-- The non-numeric aggregation on a row that selects exactly one position
returns that value (footnote to §3). -/
theorem aggregateOne_of_selected_eq_singleton {S : Selector n} {i j : Fin n}
    (h : selected S i = {j}) (v : Seq n α) (d : α) : aggregateOne S v d i = v j := by
  have hc : (selected S i).card = 1 := by rw [h]; simp
  have hspec := (Finset.card_eq_one.mp hc).choose_spec
  have hchoose : (Finset.card_eq_one.mp hc).choose = j :=
    Finset.singleton_injective (hspec.symm.trans h)
  simp only [aggregateOne, dite_eq_left hc, hchoose]

/-- A row that selects nothing aggregates to the default (§3). -/
theorem aggregate_of_selected_eq_empty {S : Selector n} {i : Fin n}
    (h : selected S i = ∅) (v : Seq n ℝ) (d : ℝ) : aggregate S v d i = d := by
  simp [aggregate, h]

lemma sum_light0 (hn : 0 < n) : ∑ j, light0 n j = 1 := by
  rw [Finset.sum_eq_single (⟨0, hn⟩ : Fin n)]
  · simp [light0]
  · intro b _ hb
    have : (b : ℕ) ≠ 0 := fun h => hb (Fin.ext h)
    simp [light0, this]
  · intro h
    exact absurd (Finset.mem_univ _) h

/-- **`length` is a RASP program, not a primitive.**  §3.1 gives

    length = 1/aggregate(select_all, indicator(indices == 0)),

and this is that identity: aggregating the indicator of position `0` over
all positions yields `1/n` everywhere, whose reciprocal is the length. -/
theorem length_eq_one_div_aggregate (hn : 0 < n) (i : Fin n) :
    1 / aggregate (selectAll n) (light0 n) 0 i = length n i := by
  rw [aggregate_selectAll hn, sum_light0 hn, length, one_div_one_div]

/-- **The histogram.**  §3: `selector_width(select(tokens,tokens,==))`
counts, at each position, the positions carrying the same token — for
instance `hist("hello") = [1,1,2,2,1]`. -/
theorem selectorWidth_sameToken [DecidableEq α] (t : Seq n α) (i : Fin n) :
    selectorWidth (sel t t (fun a b => decide (a = b))) i
      = ((Finset.univ.filter (fun j => t j = t i)).card : ℝ) := by
  simp [selectorWidth, selected, sel]

/-- The hypotheses above are satisfiable: on `"hello"` of length `5` the
histogram is `[1,1,2,2,1]`, the length program returns `5`, and
`select_all` selects all five positions. -/
example :
    let t : Seq 5 (Fin 4) := ![0, 1, 2, 2, 3]
    selectorWidth (sel t t (fun a b => decide (a = b))) 2 = 2 ∧
      (1 / aggregate (selectAll 5) (light0 5) 0 0 = length 5 0) ∧
      selectorWidth (selectAll 5) 0 = 5 := by
  refine ⟨?_, length_eq_one_div_aggregate (by norm_num) 0, selectorWidth_selectAll 0⟩
  have : selected (sel ![(0 : Fin 4), 1, 2, 2, 3] ![(0 : Fin 4), 1, 2, 2, 3]
      (fun a b => decide (a = b))) 2 = {2, 3} := by decide
  have hc : ({2, 3} : Finset (Fin 5)).card = 2 := by decide
  rw [selectorWidth, this, hc]
  norm_num

end RASP
end Transformer
