/-
# On the paraboloid, nothing is ever erased

`Transformer.ALM.Hull.liftKey_not_dominated` says the breakpoint test of
`add_line` is false for three lifted keys in increasing order, and
`Transformer.ALM.HullCover` draws the container's final size from that — but
only through `hno : ∀ p ∈ ps, p = 0`, a hypothesis of the counter model
`runState`, justified in prose and discharged nowhere.  Between the geometric
fact and the counter there was no theorem: `ps` is a list of numbers, and
nothing tied its entries to the erase relation the lines actually undergo.

`Transformer.ALM.HullPrune` supplies the relation, so the gap can be closed
directly and in a stronger form than the loop conditions give.  A lifted key
is the tangent of the paraboloid at its own abscissa, so it *strictly* beats
every other lifted key there (`lineEval_liftKey_lt_of_ne`); hence no lifted
line is pointwise dominated by the rest of a lifted family, hence no
`EraseStep` applies to such a family at all — not the breakpoint erase, not
the equal-slope erase, and not any other rule a later implementation might add.
`erasesTo_eq_of_lift` iterates that over a whole chain, and
`build_eq_of_lift` over a whole build: the container the build leaves is
exactly the set of lines inserted into it, and by `build_card_eq_keyCard` it
holds `keyCard K` lines — one per distinct key, which is what the binary
search of `argmax` assumes it is searching.

None of which is a statement about the shipped model, because `_to_2d_key`
never emits a lift: it adds `LATEST_ALPHA * inv_log_pos` to the intercept of
every live key and subtracts `BIG` from a cleared one.
`Transformer.ALM.HullMark` carries the argument below through the first term,
which is too small to revive an erase, and `Transformer.ALM.HullClear` shows
the second is not — a cleared entry *is* erased, and that is how it leaves.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 143-195.
-/

import Transformer.ALM.HullCover

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### A lifted key owns its own abscissa -/

/-- The lift is faithful: the slope `2k` already determines the key. -/
theorem liftKey_injective : Function.Injective liftKey := by
  intro k k' h
  have : 2 * k = 2 * k' := congrArg Prod.fst h
  linarith

/-- At `x = k` the lifted key evaluates to `k²`: the paraboloid's tangent
touches it there. -/
theorem lineEval_liftKey_self (k : ℝ) : lineEval (liftKey k) k = k ^ 2 := by
  unfold lineEval liftKey
  ring

/-- **And every other lifted key is strictly below it there.**  The shortfall
is exactly `(k - k')²`, so the inequality is strict as soon as the keys
differ.  This is the reason the hull of a lifted family is the whole family:
no line of it is redundant at the one query that its own key names. -/
theorem lineEval_liftKey_lt_of_ne {k k' : ℝ} (h : k ≠ k') :
    lineEval (liftKey k') k < lineEval (liftKey k) k := by
  have hpos : 0 < (k - k') ^ 2 := by
    have : k - k' ≠ 0 := sub_ne_zero_of_ne h
    positivity
  unfold lineEval liftKey
  nlinarith

/-- The hypothesis is satisfiable, and the gap is the squared key distance:
at `x = 1` the lift of `0` falls one below the lift of `1`. -/
example : lineEval (liftKey (0 : ℝ)) 1 < lineEval (liftKey (1 : ℝ)) 1 :=
  lineEval_liftKey_lt_of_ne (by norm_num)

/-! ### So no erase rule at all can fire -/

/-- A family of lifted keys, as the build's container holds it. -/
def Lifted (s : Finset (ℝ × ℝ)) : Prop := ∀ l ∈ s, ∃ k : ℝ, l = liftKey k

/-- **No erase step applies to a lifted family.**  Not merely that the two
tests of `add_line` are false — that *no* rule of the form "this line is
matched by a surviving one at every query" can hold, because the line under
test strictly beats all the survivors at its own key.  So the prose claim
"the erase loops never fire on the paraboloid" is a theorem about the erase
relation and not about the particular comparisons the loops perform. -/
theorem not_eraseStep_of_lift {s t : Finset (ℝ × ℝ)} (hs : Lifted s) :
    ¬ EraseStep s t := by
  rintro ⟨l, hl, rfl, hdom⟩
  obtain ⟨k, rfl⟩ := hs l hl
  obtain ⟨l', hl', hle⟩ := hdom k
  obtain ⟨k', rfl⟩ := hs l' (Finset.mem_of_mem_erase hl')
  have hne : k' ≠ k := fun h => (Finset.mem_erase.mp hl').1 (congrArg liftKey h)
  exact absurd hle (not_le.mpr (lineEval_liftKey_lt_of_ne (Ne.symm hne)))

/-- The hypothesis is satisfiable, and the conclusion is not vacuous on the
same data an erase *does* apply to elsewhere: the three lifted keys `0, 1, 2`
form a lifted family, while the three concurrent lines of
`Transformer.ALM.HullErase` admit an erase. -/
example : Lifted ({liftKey 0, liftKey 1, liftKey 2} : Finset (ℝ × ℝ)) := by
  intro l hl
  simp only [Finset.mem_insert, Finset.mem_singleton] at hl
  rcases hl with rfl | rfl | rfl
  exacts [⟨0, rfl⟩, ⟨1, rfl⟩, ⟨2, rfl⟩]

/-- **And so no chain of them removes anything.**  However many erase loops
run over the build, a lifted container comes out of them unchanged. -/
theorem erasesTo_eq_of_lift {s t : Finset (ℝ × ℝ)} (h : ErasesTo s t) (hs : Lifted s) :
    t = s := by
  induction h with
  | refl => rfl
  | tail _ hstep ih => exact absurd hstep (not_eraseStep_of_lift (ih ▸ hs))

/-! ### And the build keeps every line it inserts -/

/-- **The container is the whole insertion history.**  Over an arbitrary
interleaving of insertions and erases — `BuildStep` of
`Transformer.ALM.HullPrune` — a build that only ever inserts lifted keys ends
with the container equal to the set of lines it was given.  This is
`hno : ∀ p ∈ ps, p = 0` discharged: not assumed of a counter, but proved of
the lines. -/
theorem build_eq_of_lift {p : Finset (ℝ × ℝ) × Finset (ℝ × ℝ)}
    (h : Relation.ReflTransGen BuildStep (∅, ∅) p) (hs : Lifted p.2) : p.1 = p.2 := by
  induction h with
  | refl => rfl
  | tail _ hstep ih =>
      rename_i b _ _
      rcases hstep with ⟨l, rfl⟩ | ⟨hers, h2⟩
      · have hb : Lifted b.2 := fun x hx => hs x (Finset.mem_insert_of_mem hx)
        simp only []
        rw [ih hb]
      · have hb : Lifted b.2 := h2 ▸ hs
        exact absurd hers (not_eraseStep_of_lift ((ih hb) ▸ hb))

/-- **So it holds one line per distinct key.**  The build of a key family
ends with a container of `keyCard K` lines, which is the length of the sorted
array `hullProbe` binary-searches: the search window really is the whole
family, with no appeal to the counter model at all. -/
theorem build_card_eq_keyCard {c : Finset (ℝ × ℝ)} (K : Fin n → ℝ)
    (h : Relation.ReflTransGen BuildStep (∅, ∅) (c, (keySet K).image liftKey)) :
    c.card = keyCard K := by
  have heq : c = (keySet K).image liftKey :=
    build_eq_of_lift (p := (c, (keySet K).image liftKey)) h (fun l hl => by
      obtain ⟨k, -, hk⟩ := Finset.mem_image.mp hl
      exact ⟨k, hk.symm⟩)
  rw [heq]
  exact Finset.card_image_of_injective _ liftKey_injective

/-- The hypothesis is satisfiable: inserting the single key `0` builds the
container `{liftKey 0}`, whose one line is the one distinct key of `Fin 1`. -/
example : (insert (liftKey (0 : ℝ)) (∅ : Finset (ℝ × ℝ))).card
    = keyCard (fun _ : Fin 1 => (0 : ℝ)) := by
  refine build_card_eq_keyCard (fun _ : Fin 1 => (0 : ℝ)) ?_
  have hs : (keySet (fun _ : Fin 1 => (0 : ℝ))).image liftKey
      = insert (liftKey (0 : ℝ)) (∅ : Finset (ℝ × ℝ)) := by
    simp [keySet]
  rw [hs]
  exact Relation.ReflTransGen.single (Or.inl ⟨liftKey (0 : ℝ), rfl⟩)

end ALM
end Transformer
