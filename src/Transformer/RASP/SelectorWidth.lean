/-
# RASP: `selector_width` is not a primitive

Weiss, Goldberg, Yahav — arXiv:2106.06981v2, "Thinking Like Transformers",
Figure 8 (Appendix, "Implementation of the powerful RASP operation
`selector_width`").

§3.1 says that `selector_width` "can be implemented such that it compiles to
either one or two selectors, depending on whether or not one can assume a
beginning-of-sequence token is added to the input sequence", and Figure 8 is
that implementation:

    light0      = indicator(indices == 0);
    or0         = sel or select_eq(indices, 0);
    and0        = sel and select_eq(indices, 0);
    or0_width   = 1 / aggregate(or0, light0);
    and0_width  = aggregate(and0, light0, 0);
    bos_res     = or0_width - 1;
    nobos_res   = bos_res + and0_width;

The claim to check is that `nobos_res` is the width of `sel`, and that
`bos_res` is its width with position `0` — the BOS position, which is never
to be counted — removed.  Both hold exactly, for every selector.

The mechanism is the one §4 calls out under "Use of Separator Tokens": a
row of `or0` always selects position `0`, so the mass `light0` puts there is
diffused by exactly the row's width, and the reciprocal recovers it.  The
`and0` term is the correction that a neutral position would have made
unnecessary — "without such neutral positions, counting requires an
additional head".
-/

import Transformer.RASP.Basic

namespace Transformer
namespace RASP

variable {n : ℕ}

/-- `or0 = sel or select_eq(indices, 0)` (Figure 8). -/
def or0 (S : Selector n) : Selector n := Selector.or' S (selectZero n)

/-- `and0 = sel and select_eq(indices, 0)` (Figure 8). -/
def and0 (S : Selector n) : Selector n := Selector.and' S (selectZero n)

/-- `or0_width = 1 / aggregate(or0, light0)` (Figure 8). -/
noncomputable def or0Width (S : Selector n) : Seq n ℝ :=
  fun i => 1 / aggregate (or0 S) (light0 n) 0 i

/-- `and0_width = aggregate(and0, light0, 0)` (Figure 8). -/
noncomputable def and0Width (S : Selector n) : Seq n ℝ := aggregate (and0 S) (light0 n) 0

/-- `bos_res = or0_width - 1` (Figure 8): the value returned when a
beginning-of-sequence token may be assumed. -/
noncomputable def bosRes (S : Selector n) : Seq n ℝ := fun i => or0Width S i - 1

/-- `nobos_res = bos_res + and0_width` (Figure 8): the value returned
otherwise. -/
noncomputable def noBosRes (S : Selector n) : Seq n ℝ := fun i => bosRes S i + and0Width S i

lemma selected_or0 (hn : 0 < n) (S : Selector n) (i : Fin n) :
    selected (or0 S) i = insert (⟨0, hn⟩ : Fin n) (selected S i) := by
  ext j
  simp [or0, Selector.or', selectZero, Fin.ext_iff, or_comm]

lemma selected_and0 (hn : 0 < n) (S : Selector n) (i : Fin n) :
    selected (and0 S) i =
      if S i ⟨0, hn⟩ = true then {(⟨0, hn⟩ : Fin n)} else ∅ := by
  ext j
  have hj : (selectZero n i j = true) ↔ j = (⟨0, hn⟩ : Fin n) := by
    simp [selectZero, Fin.ext_iff]
  simp only [mem_selected, and0, Selector.and', Bool.and_eq_true, hj]
  by_cases h : S i ⟨0, hn⟩ = true
  · rw [ite_eq_left h]
    simp only [Finset.mem_singleton]
    exact ⟨fun hx => hx.2, fun hx => ⟨hx ▸ h, hx⟩⟩
  · rw [ite_eq_right h]
    simp only [Finset.notMem_empty, iff_false, not_and]
    intro h1 h2
    exact h (h2 ▸ h1)

/-- Position `0` is always in an `or0` row: that is what makes the diffused
signal `light0` measurable at every query position. -/
lemma card_selected_or0 (hn : 0 < n) (S : Selector n) (i : Fin n) :
    (selected (or0 S) i).card = ((selected S i).erase ⟨0, hn⟩).card + 1 := by
  have hins : insert (⟨0, hn⟩ : Fin n) (selected S i)
      = insert (⟨0, hn⟩ : Fin n) ((selected S i).erase ⟨0, hn⟩) := by
    ext x
    by_cases hx : x = (⟨0, hn⟩ : Fin n) <;> simp [hx]
  rw [selected_or0 hn, hins, Finset.card_insert_of_notMem (Finset.notMem_erase _ _)]

lemma sum_light0_or0 (hn : 0 < n) (S : Selector n) (i : Fin n) :
    ∑ j ∈ selected (or0 S) i, light0 n j = 1 := by
  rw [Finset.sum_eq_single (⟨0, hn⟩ : Fin n)]
  · simp [light0]
  · intro b _ hb
    have : (b : ℕ) ≠ 0 := fun h => hb (Fin.ext h)
    simp [light0, this]
  · intro h
    exact absurd (by simp [selected_or0 hn]) h

/-- **The `or0` width is the width of `or0`.**  Aggregating the position-`0`
indicator over a row that always contains position `0` returns the
reciprocal of the row's width, so the reciprocal of that is the width. -/
theorem or0Width_eq (hn : 0 < n) (S : Selector n) (i : Fin n) :
    or0Width S i = (((selected S i).erase ⟨0, hn⟩).card : ℝ) + 1 := by
  have hcard := card_selected_or0 hn S i
  have hpos : ((selected (or0 S) i).card : ℝ) ≠ 0 := by
    rw [hcard]; positivity
  rw [or0Width, aggregate, ite_eq_right (by rw [hcard]; omega), sum_light0_or0 hn,
    one_div_one_div, hcard]
  push_cast
  ring

/-- **The `and0` width is the correction term.**  `and0` selects position `0`
exactly at the query positions whose `sel` row already contained it. -/
theorem and0Width_eq (hn : 0 < n) (S : Selector n) (i : Fin n) :
    and0Width S i = if S i ⟨0, hn⟩ = true then 1 else 0 := by
  by_cases h : S i ⟨0, hn⟩ = true
  · rw [and0Width, aggregate_of_selected_eq_singleton (by rw [selected_and0 hn, ite_eq_left h]),
      ite_eq_left h, light0, ite_eq_left rfl]
  · rw [and0Width, aggregate_of_selected_eq_empty (by rw [selected_and0 hn, ite_eq_right h]),
      ite_eq_right h]

/-- **Figure 8 computes `selector_width`.**  Without a beginning-of-sequence
token, `nobos_res` is the width of the selector, at every query position and
for every selector at all. -/
theorem noBosRes_eq_selectorWidth (hn : 0 < n) (S : Selector n) (i : Fin n) :
    noBosRes S i = selectorWidth S i := by
  rw [noBosRes, bosRes, or0Width_eq hn, and0Width_eq hn, selectorWidth]
  by_cases h : S i ⟨0, hn⟩ = true
  · have hmem : (⟨0, hn⟩ : Fin n) ∈ selected S i := by simpa using h
    rw [ite_eq_left h, Finset.card_erase_of_mem hmem]
    have hone : 1 ≤ (selected S i).card := Finset.card_pos.2 ⟨_, hmem⟩
    have hcast : (((selected S i).card - 1 : ℕ) : ℝ) = ((selected S i).card : ℝ) - 1 := by
      push_cast [hone]
      ring
    rw [hcast]; ring
  · have hmem : (⟨0, hn⟩ : Fin n) ∉ selected S i := by simpa using h
    rw [ite_eq_right h, Finset.erase_eq_of_notMem hmem]
    ring

/-- **And with one, it computes the width less the BOS position.**  The
comment in Figure 8 — "if has bos, remove bos from width (doesn't count, even
if chosen by sel)" — is exact: `bos_res` is the width of the selector on the
positions after the first, whether or not the selector chose the first. -/
theorem bosRes_eq (hn : 0 < n) (S : Selector n) (i : Fin n) :
    bosRes S i = (((selected S i).erase ⟨0, hn⟩).card : ℝ) := by
  rw [bosRes, or0Width_eq hn]; ring

/-- The hypotheses are satisfiable: on three positions with
`sel = select(indices, indices, <)`, query `2` selects `{0, 1}`, so its width
is `2` and its width without the BOS position is `1`. -/
example :
    let S : Selector 3 := sel (indices 3) (indices 3) (fun a b => decide (a < b))
    noBosRes S 2 = 2 ∧ bosRes S 2 = 1 := by
  intro S
  have hsel : selected S 2 = {0, 1} := by
    ext j
    fin_cases j <;> simp [S, selected, sel, indices]
  have hc : ({0, 1} : Finset (Fin 3)).card = 2 := by decide
  have he : (({0, 1} : Finset (Fin 3)).erase ⟨0, by norm_num⟩).card = 1 := by decide
  refine ⟨?_, ?_⟩
  · rw [noBosRes_eq_selectorWidth (by norm_num), selectorWidth, hsel, hc]; norm_num
  · rw [bosRes_eq (by norm_num), hsel, he]; norm_num

end RASP
end Transformer
