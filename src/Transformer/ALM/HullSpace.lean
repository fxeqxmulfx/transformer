/-
# The envelope names the keys, so nothing smaller answers

`Transformer.ALM.HullLift` proves that the container the machine builds keeps
one line per distinct key: no erase rule applies to a lifted family, so the
build ends with `keyCard K` lines.  That is a statement about `add_line`.  It
leaves open the question `todo3.md` §5 actually asks — whether a *different*
container could do better, since the measured `128` bytes per token and the
"several GB of hulls" of the Sudoku demo are a property of the construction
only if no cleverer representation exists.

None does, in the representation the head is defined in.  At the query `x = k`
the envelope of a lifted family is exactly `k²`, and `k²` is the most any line
of it can score there, so a line attaining the maximum at two distinct keys
would be a chord of the parabola and would overshoot the envelope at their
midpoint.  So the attaining line is different for every key
(`card_le_of_envelope_eq`): any family of lines with the head's envelope has at
least one line per distinct key it was fed, whatever built it.

The linear space is therefore not a defect of `hull2d_cht.h`.  It is what a
head keyed on `position` costs, and `wasm/interpreter.py` has at least thirteen
of them.

Source: `todo3.md` §5; `transformer_vm/wasm/interpreter.py` lines 304, 330, 334,
415-417 and 481.
-/

import Transformer.ALM.HullLift

namespace Transformer
namespace ALM

/-! ### The envelope of a lifted family, read at one of its own keys -/

/-- No lifted line reaches above the parabola: the shortfall at `x` is the
squared distance from `x` to the key. -/
theorem lineEval_liftKey_le_sq (k x : ℝ) : lineEval (liftKey k) x ≤ x ^ 2 := by
  have h : x ^ 2 - lineEval (liftKey k) x = (x - k) ^ 2 := by
    unfold lineEval liftKey; ring
  nlinarith [sq_nonneg (x - k)]

/-- So neither does their envelope. -/
theorem envelope_le_sq {S : Finset ℝ} (hS : S.Nonempty) (x : ℝ) :
    S.sup' hS (fun k => lineEval (liftKey k) x) ≤ x ^ 2 :=
  Finset.sup'_le _ _ fun k _ => lineEval_liftKey_le_sq k x

/-- **And at a stored key it is reached exactly.**  The envelope of a lifted
family is `k²` at every key `k` the family holds — the tangency
`lineEval_liftKey_self`, now as a property of the whole container. -/
theorem envelope_eq_sq_of_mem {S : Finset ℝ} (hS : S.Nonempty) {k : ℝ} (hk : k ∈ S) :
    S.sup' hS (fun k' => lineEval (liftKey k') k) = k ^ 2 :=
  le_antisymm (envelope_le_sq hS k)
    (lineEval_liftKey_self k ▸ Finset.le_sup' (fun k' => lineEval (liftKey k') k) hk)

/-! ### One line per key, for any representation at all -/

/-- **A chord of the parabola overshoots its midpoint.**  A line touching the
parabola at two distinct keys is above it halfway between them, by a quarter of
the squared key distance.  This is the whole obstruction: such a line cannot
stay under an envelope that is itself under the parabola. -/
lemma lineEval_midpoint_gt {l : ℝ × ℝ} {k₁ k₂ : ℝ} (hne : k₁ ≠ k₂)
    (h₁ : lineEval l k₁ = k₁ ^ 2) (h₂ : lineEval l k₂ = k₂ ^ 2) :
    ((k₁ + k₂) / 2) ^ 2 < lineEval l ((k₁ + k₂) / 2) := by
  have hpos : 0 < (k₁ - k₂) ^ 2 / 4 := by
    have : k₁ - k₂ ≠ 0 := sub_ne_zero_of_ne hne
    positivity
  have hid : lineEval l ((k₁ + k₂) / 2) - ((k₁ + k₂) / 2) ^ 2 = (k₁ - k₂) ^ 2 / 4 := by
    unfold lineEval at h₁ h₂ ⊢
    linear_combination h₁ / 2 + h₂ / 2
  linarith

/-- **So no family of lines with the head's envelope is smaller than its key
set.**  Whatever the container, however it was built, if its upper envelope
agrees at every query with the envelope of the lifted keys `S` then it holds at
least `S.card` lines: the line attaining the maximum at `k` attains it at no
other key of `S`, so `k ↦ that line` is injective.  This is the lower bound
`todo3.md` §5 asks for — the linear state is forced by the queries, not chosen
by `add_line`. -/
theorem card_le_of_envelope_eq {S : Finset ℝ} (hS : S.Nonempty) {L : Finset (ℝ × ℝ)}
    (hL : L.Nonempty)
    (henv : ∀ x : ℝ, L.sup' hL (fun l => lineEval l x)
      = S.sup' hS (fun k => lineEval (liftKey k) x)) :
    S.card ≤ L.card := by
  classical
  have hpick : ∀ k ∈ S, ∃ l, l ∈ L ∧ lineEval l k = k ^ 2 := by
    intro k hk
    obtain ⟨l, hlL, hl⟩ := Finset.exists_mem_eq_sup' hL (fun l => lineEval l k)
    exact ⟨l, hlL, by rw [← hl, henv k, envelope_eq_sq_of_mem hS hk]⟩
  choose! pick hpickL hpickEq using hpick
  refine Finset.card_le_card_of_injOn pick hpickL ?_
  intro a ha b hb hab
  by_contra hne
  have h₁ : lineEval (pick a) a = a ^ 2 := hpickEq a ha
  have h₂ : lineEval (pick a) b = b ^ 2 := hab ▸ hpickEq b hb
  have hgt := lineEval_midpoint_gt hne h₁ h₂
  have hle : lineEval (pick a) ((a + b) / 2) ≤ ((a + b) / 2) ^ 2 := by
    refine le_trans ?_ (envelope_le_sq hS ((a + b) / 2))
    rw [← henv]
    exact Finset.le_sup' (fun l => lineEval l ((a + b) / 2)) (hpickL a ha)
  linarith

/-! ### The hypotheses are satisfiable -/

/-- The container the machine actually builds witnesses the bound's hypothesis
with equality: its lines are exactly the lifted keys, so its envelope is the
envelope of the keys and the bound says it can hold no fewer lines. -/
example (S : Finset ℝ) (hS : S.Nonempty) (x : ℝ) :
    (S.image liftKey).sup' (hS.image liftKey) (fun l => lineEval l x)
      = S.sup' hS (fun k => lineEval (liftKey k) x) :=
  Finset.sup'_image _ _

/-- And it is a nonempty container of a nonempty key set: the three keys
`0, 1, 2` of `Transformer.ALM.HullLift`. -/
example : ({0, 1, 2} : Finset ℝ).Nonempty ∧
    (({0, 1, 2} : Finset ℝ).image liftKey).Nonempty :=
  ⟨⟨0, by simp⟩, ⟨liftKey 0, by simp⟩⟩

/-- And the chord's hypotheses: the line `y = x` touches the parabola at `0`
and at `1`, and sits a quarter above it at the midpoint. -/
example : (0 : ℝ) ≠ 1 ∧ lineEval ((1, 0) : ℝ × ℝ) 0 = (0 : ℝ) ^ 2 ∧
    lineEval ((1, 0) : ℝ × ℝ) 1 = (1 : ℝ) ^ 2 := by
  refine ⟨by norm_num, by norm_num [lineEval], by norm_num [lineEval]⟩

end ALM
end Transformer
