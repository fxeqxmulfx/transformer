/-
# One erase, and then all of them

`Transformer.ALM.HullErase` proves that each erase `add_line` performs leaves
the lookup unchanged: whichever of the two loop conditions fired, the best
value over the pruned family is still the best value over the family before
the erase (`erase_slope_eq_preserves_isGreatest`,
`erase_interX_le_preserves_isGreatest`).

That is a statement about a single call.  The container the queries are run
against is the result of a whole build — thousands of erases, interleaved with
insertions — and "so a query answered against the hull is answered against
every line ever inserted" was the conclusion drawn in prose from a one-step
lemma.  One step preserving a property does not by itself give it after many:
the induction has to be done, and after each erase the family the *next* erase
is justified against is a different one.

`EraseStep` is one loop iteration as a relation on containers, `erases_*` its
reflexive-transitive closure, and `erases_preserves_isGreatest` the induction:
however long the sequence of erases, the surviving lines still contain a
maximizer over the original family.  `eraseStep_of_slope_eq` and
`eraseStep_of_interX_le` say the two loop conditions each produce such a step,
so the chain is the one the code walks and not an abstraction of it.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 143-195.
-/

import Transformer.ALM.HullErase

namespace Transformer
namespace ALM

/-- **One iteration of an erase loop.**  A line leaves the container, and some
line that stays is at least as high as it at *every* query — which is what the
loop conditions of `add_line` guarantee and what the queries need. -/
def EraseStep (s t : Finset (ℝ × ℝ)) : Prop :=
  ∃ l ∈ s, t = s.erase l ∧ ∀ x : ℝ, ∃ l' ∈ t, lineEval l x ≤ lineEval l' x

/-- **The early-return test produces such a step.**  A new line parallel to a
stored one and no higher is dropped, and the stored one dominates it
everywhere. -/
theorem eraseStep_of_slope_eq {s : Finset (ℝ × ℝ)} {l l' : ℝ × ℝ}
    (hl : l ∈ s) (hmem : l' ∈ s.erase l) (hm : l.1 = l'.1) (hb : l.2 ≤ l'.2) :
    EraseStep s (s.erase l) :=
  ⟨l, hl, rfl, fun x => ⟨l', hmem, lineEval_le_of_slope_eq hm hb x⟩⟩

/-- **And so does the breakpoint test.**  When
`isect(l₁,l₂).p ≥ isect(l₂,l₃).p` fires the middle line is at or below one of
its surviving neighbours at every query — `exists_ge_of_interX_le`. -/
theorem eraseStep_of_interX_le {s : Finset (ℝ × ℝ)} {l₁ l₂ l₃ : ℝ × ℝ}
    (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1) (htest : interX l₂ l₃ ≤ interX l₁ l₂)
    (hl₂ : l₂ ∈ s) (h₁ : l₁ ∈ s.erase l₂) (h₃ : l₃ ∈ s.erase l₂) :
    EraseStep s (s.erase l₂) :=
  ⟨l₂, hl₂, rfl, fun x => exists_ge_of_interX_le h₁₂ h₂₃ htest h₁ h₃ x⟩

/-- The hypotheses of both are satisfiable at once: two parallel lines with
the lower one erased, and three concurrent lines with the middle one erased. -/
example :
    EraseStep ({((0:ℝ), (0:ℝ)), (0, 1)} : Finset (ℝ × ℝ))
        (({((0:ℝ), (0:ℝ)), (0, 1)} : Finset (ℝ × ℝ)).erase (0, 0)) ∧
      EraseStep ({((-1:ℝ), (0:ℝ)), (0, 0), (1, 0)} : Finset (ℝ × ℝ))
        (({((-1:ℝ), (0:ℝ)), (0, 0), (1, 0)} : Finset (ℝ × ℝ)).erase (0, 0)) := by
  constructor
  · exact eraseStep_of_slope_eq (l := ((0 : ℝ), (0 : ℝ))) (l' := ((0 : ℝ), (1 : ℝ)))
      (by simp) (by simp) rfl (by norm_num)
  · exact eraseStep_of_interX_le (l₁ := ((-1 : ℝ), (0 : ℝ))) (l₃ := ((1 : ℝ), (0 : ℝ)))
      (by norm_num) (by norm_num) (by norm_num [interX]) (by simp) (by simp) (by simp)

/-- A whole build's erases: any number of them, in any order. -/
def ErasesTo (s t : Finset (ℝ × ℝ)) : Prop := Relation.ReflTransGen EraseStep s t

/-- **Pruning loses no argmax, however often it prunes.**  After an arbitrary
sequence of erase steps the surviving lines still contain a line that is
highest at `x` among *all* the lines the build ever held — so the query the
machine answers against the final container is the query it would have
answered against the whole input.  This is the induction the one-step
statements of `Transformer.ALM.HullErase` do not give: each step is justified
against the family standing at that moment, and the maximizer is carried
forward through them all. -/
theorem erases_preserves_isGreatest {s t : Finset (ℝ × ℝ)} (h : ErasesTo s t)
    (hne : t.Nonempty) (x : ℝ) : ∃ a ∈ t, ∀ b ∈ s, lineEval b x ≤ lineEval a x := by
  induction h with
  | refl => exact Finset.exists_max_image _ _ hne
  | tail _ hbc ih =>
      obtain ⟨l, hl, rfl, hdom⟩ := hbc
      obtain ⟨a, ha, hmax⟩ := ih ⟨l, hl⟩
      obtain ⟨a', ha', hmax'⟩ := erase_preserves_isGreatest hl hne x (hdom x)
      exact ⟨a', ha', fun z hz => (hmax z hz).trans (hmax' a ha)⟩

/-- The hypotheses are satisfiable by a chain of genuine length: three
parallel lines, the lowest erased and then the middle one, and the query at
`x = 0` is still answered by a line that is standing at the end. -/
example :
    ∃ a ∈ (({((0:ℝ), (0:ℝ)), (0, 1), (0, 2)} : Finset (ℝ × ℝ)).erase (0, 0)).erase (0, 1),
      ∀ b ∈ ({((0:ℝ), (0:ℝ)), (0, 1), (0, 2)} : Finset (ℝ × ℝ)),
        lineEval b 0 ≤ lineEval a 0 := by
  refine erases_preserves_isGreatest (Relation.ReflTransGen.tail
    (Relation.ReflTransGen.single
      (eraseStep_of_slope_eq (l := ((0 : ℝ), (0 : ℝ))) (l' := ((0 : ℝ), (2 : ℝ)))
        (by simp) (by simp) rfl (by norm_num)))
    (eraseStep_of_slope_eq (l := ((0 : ℝ), (1 : ℝ))) (l' := ((0 : ℝ), (2 : ℝ)))
      (by simp) (by simp) rfl (by norm_num))) ⟨((0 : ℝ), (2 : ℝ)), by simp⟩ 0

end ALM
end Transformer
