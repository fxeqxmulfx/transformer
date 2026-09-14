/-
# The erase keeps the value and can drop the winner

`Transformer.ALM.HullErase` proves that what `add_line` erases costs the
machine no exactness, and `Transformer.ALM.HullPrune` carries that through a
whole build: the container standing at the end holds a line that is highest at
every query among all the lines ever inserted.  Both are statements about the
*value*.  The head does not return a value.  It returns a payload, merged over
every line that attains the maximum — `HullHalf::query` walks left and right
from the winner while the score is equal and merges each neighbour's `vsum`,
`vlast` and `last_seq` — so what a query depends on is the whole tie set, and
nothing so far says the erase preserves *that*.

It does not.  When three stored lines meet at one point the middle one is a
maximizer there and the breakpoint test fires on it anyway: the value at that
query survives in its two neighbours, the payload does not.  `todo3.md` §3a
measures it on three collinear keys — `Latest` answers `101` where the brute
head answers the erased line's `102`, `Average` answers `100.5` where the
brute head answers `101` — and `vm-rs/alm-hull/tests/differential.rs` is the
witness.

Three lines meet at a point exactly when the three keys they came from are
collinear (`concurrent_iff_collinear`), which is the hypothesis
`Transformer.ALM.GeneralPosition` then states and discharges for the machine.

Source: `todo3.md` §3a; `transformer_vm/attention/hull2d_cht.h`, lines 143-195
(the erase loops) and 276-306 (the merge walk).
-/

import Transformer.ALM.HullErase

namespace Transformer
namespace ALM

/-! ### The tie set -/

open scoped Classical in
/-- The lines of `s` that are highest at `x`: exactly what the merge walk of
`HullHalf::query` collects, and what the answer is a function of. -/
noncomputable def tieSet (s : Finset (ℝ × ℝ)) (x : ℝ) : Finset (ℝ × ℝ) :=
  s.filter fun l => ∀ l' ∈ s, lineEval l' x ≤ lineEval l x

lemma mem_tieSet {s : Finset (ℝ × ℝ)} {x : ℝ} {l : ℝ × ℝ} :
    l ∈ tieSet s x ↔ l ∈ s ∧ ∀ l' ∈ s, lineEval l' x ≤ lineEval l x := by
  classical
  simp [tieSet]

lemma tieSet_subset (s : Finset (ℝ × ℝ)) (x : ℝ) : tieSet s x ⊆ s :=
  fun _ h => (mem_tieSet.mp h).1

/-! ### Where lines meet -/

/-- Lines of different slope meet only at their stored breakpoint. -/
lemma eq_interX_of_lineEval_eq {l l' : ℝ × ℝ} (h : l.1 ≠ l'.1) {x : ℝ}
    (hx : lineEval l x = lineEval l' x) : x = interX l l' := by
  unfold lineEval at hx
  unfold interX
  rw [eq_div_iff (sub_ne_zero_of_ne h)]
  linear_combination hx

/-- Three lines are *concurrent* when one query sees all three at the same
height.  This is the degenerate case the erase loops do not distinguish. -/
def Concurrent (l₁ l₂ l₃ : ℝ × ℝ) : Prop :=
  ∃ x, lineEval l₁ x = lineEval l₂ x ∧ lineEval l₂ x = lineEval l₃ x

/-- **Concurrency is collinearity of the keys.**  A key `(kx, ky)` is stored as
the line `x ↦ kx·x + ky`, so three lines meeting at one point is the vanishing
determinant of the three key points — "no three keys collinear", the hypothesis
`todo3.md` §3a says is nowhere stated. -/
theorem concurrent_iff_collinear {l₁ l₂ l₃ : ℝ × ℝ} (h₁₂ : l₁.1 < l₂.1) :
    Concurrent l₁ l₂ l₃ ↔
      (l₂.1 - l₁.1) * (l₃.2 - l₁.2) = (l₃.1 - l₁.1) * (l₂.2 - l₁.2) := by
  constructor
  · rintro ⟨x, h12, h23⟩
    unfold lineEval at h12 h23
    have hb2 : l₂.2 - l₁.2 = (l₁.1 - l₂.1) * x := by linear_combination -h12
    have hb3 : l₃.2 - l₁.2 = (l₁.1 - l₃.1) * x := by linear_combination -h12 - h23
    rw [hb2, hb3]; ring
  · intro hdet
    refine ⟨interX l₁ l₂, lineEval_interX l₁ l₂ (by linarith), ?_⟩
    have hd : l₁.1 - l₂.1 ≠ 0 := sub_ne_zero_of_ne (ne_of_lt h₁₂)
    unfold lineEval interX
    field_simp
    linear_combination hdet

/-- Concurrent lines cross at one abscissa, so the two breakpoints the erase
loop compares are equal and the test — which is `≤`, not `<` — fires. -/
lemma interX_eq_of_concurrent {l₁ l₂ l₃ : ℝ × ℝ} (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1)
    (h : Concurrent l₁ l₂ l₃) : interX l₂ l₃ = interX l₁ l₂ := by
  obtain ⟨x, h12, h23⟩ := h
  rw [← eq_interX_of_lineEval_eq (ne_of_lt h₁₂) h12,
    ← eq_interX_of_lineEval_eq (ne_of_lt h₂₃) h23]

/-! ### What the erase costs when three lines meet -/

/-- **The value survives the erase; the winner need not.**  At a query where
three stored lines agree, the middle one is a maximizer, the breakpoint test
fires on it, and afterwards the envelope has the same value at that query —
`sup'_erase_of_le`, the theorem `Transformer.ALM.HullErase` exists for — while
the tie set the payloads are merged over has lost a member.  Two heads told to
merge over the maximizers therefore answer differently, which is what
`todo3.md` §3a measured between `HullKVCache` and `BruteAttentionHead`. -/
theorem erase_keeps_the_value_and_drops_the_winner
    {s : Finset (ℝ × ℝ)} {l₁ l₂ l₃ : ℝ × ℝ} {x : ℝ}
    (h₁ : l₁ ∈ s) (h₂ : l₂ ∈ s) (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1)
    (h12 : lineEval l₁ x = lineEval l₂ x) (h23 : lineEval l₂ x = lineEval l₃ x)
    (hwin : ∀ l ∈ s, lineEval l x ≤ lineEval l₂ x) :
    interX l₂ l₃ ≤ interX l₁ l₂ ∧ l₂ ∈ tieSet s x ∧ l₂ ∉ tieSet (s.erase l₂) x ∧
      ∀ hne : (s.erase l₂).Nonempty,
        s.sup' ⟨l₂, h₂⟩ (fun a => lineEval a x)
          = (s.erase l₂).sup' hne (fun a => lineEval a x) := by
  classical
  have hne12 : l₁ ≠ l₂ := fun h => absurd (congrArg Prod.fst h) (ne_of_lt h₁₂)
  have h₁e : l₁ ∈ s.erase l₂ := Finset.mem_erase.mpr ⟨hne12, h₁⟩
  refine ⟨(interX_eq_of_concurrent h₁₂ h₂₃ ⟨x, h12, h23⟩).le, mem_tieSet.mpr ⟨h₂, hwin⟩,
    fun h => Finset.notMem_erase l₂ s (tieSet_subset _ _ h), fun hne => ?_⟩
  exact sup'_erase_of_le h₂ hne x ⟨l₁, h₁e, le_of_eq h12.symm⟩


/-! ### The hypotheses are satisfiable -/

/-- The degenerate triple: `y = -x`, `y = 0` and `y = x` meet at the origin.
It satisfies every hypothesis of `erase_keeps_the_value_and_drops_the_winner`
at the query `x = 0` — where all three score `0`, the flat line is a winner,
and the breakpoint test erases it — and with it those of
`eq_interX_of_lineEval_eq`, `interX_eq_of_concurrent` and
`concurrent_iff_collinear`. -/
example :
    ((-1, 0) : ℝ × ℝ).1 < ((0, 0) : ℝ × ℝ).1 ∧ ((0, 0) : ℝ × ℝ).1 < ((1, 0) : ℝ × ℝ).1 ∧
      lineEval ((-1, 0) : ℝ × ℝ) 0 = lineEval ((0, 0) : ℝ × ℝ) 0 ∧
      lineEval ((0, 0) : ℝ × ℝ) 0 = lineEval ((1, 0) : ℝ × ℝ) 0 ∧
      ((-1, 0) : ℝ × ℝ) ∈ ({(-1, 0), (0, 0), (1, 0)} : Finset (ℝ × ℝ)) ∧
      ((0, 0) : ℝ × ℝ) ∈ ({(-1, 0), (0, 0), (1, 0)} : Finset (ℝ × ℝ)) ∧
      ∀ l ∈ ({(-1, 0), (0, 0), (1, 0)} : Finset (ℝ × ℝ)),
        lineEval l 0 ≤ lineEval ((0, 0) : ℝ × ℝ) 0 := by
  refine ⟨by norm_num, by norm_num, by norm_num [lineEval], by norm_num [lineEval],
    by simp, by simp, fun l hl => ?_⟩
  simp only [Finset.mem_insert, Finset.mem_singleton] at hl
  rcases hl with rfl | rfl | rfl <;> norm_num [lineEval]

end ALM
end Transformer
