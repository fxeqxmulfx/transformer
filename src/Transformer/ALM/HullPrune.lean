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

A build is not a sequence of erases either: `add_line` inserts, then erases,
then the next call inserts again, and the family a query has to be answered
against grows while the container shrinks.  `BuildStep` is one call on both at
once, and `build_isGreatest_of_inserted` is the build's correctness — the
container standing at the end holds a line that is highest at every query
among all the lines ever inserted.  `Transformer.ALM.HullBuild` prices that
build and `Transformer.ALM.HullCover` counts what it holds; neither says it
answers anything.

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

/-! ### And the insertions in between -/

/-- **One call of `add_line`, on the container and on the log.**  Either a new
line is inserted — into the container and into the record of everything ever
inserted alike — or one of the erase loops drops a line from the container,
leaving that record untouched. -/
def BuildStep (p q : Finset (ℝ × ℝ) × Finset (ℝ × ℝ)) : Prop :=
  (∃ l, q = (insert l p.1, insert l p.2)) ∨ (EraseStep p.1 q.1 ∧ q.2 = p.2)

/-- The container is empty only before anything was inserted, and while it is
not, it holds a maximizer over every line inserted so far.  The first half is
what makes the second inductive: an erase step cannot empty the container,
since the line it drops is dominated by one that stays. -/
private lemma build_invariant (x : ℝ) {p : Finset (ℝ × ℝ) × Finset (ℝ × ℝ)}
    (h : Relation.ReflTransGen BuildStep (∅, ∅) p) :
    (p.1 = ∅ → p.2 = ∅) ∧
      (p.1.Nonempty → ∃ a ∈ p.1, ∀ b ∈ p.2, lineEval b x ≤ lineEval a x) := by
  induction h with
  | refl => exact ⟨fun _ => rfl, fun hne => absurd hne (by simp)⟩
  | tail _ hstep ih =>
      rename_i b c _
      rcases hstep with ⟨l, rfl⟩ | ⟨hers, h2⟩
      · refine ⟨fun hc => absurd hc (by simp), fun _ => ?_⟩
        simp only
        rcases Finset.eq_empty_or_nonempty b.1 with hb | hb
        · refine ⟨l, Finset.mem_insert_self l b.1, fun z hz => ?_⟩
          rw [ih.1 hb] at hz
          rw [Finset.mem_insert] at hz
          rcases hz with rfl | hz
          · exact le_rfl
          · exact absurd hz (by simp)
        · obtain ⟨a, ha, hmax⟩ := ih.2 hb
          rcases le_total (lineEval a x) (lineEval l x) with hle | hle
          · refine ⟨l, Finset.mem_insert_self l b.1, fun z hz => ?_⟩
            rcases Finset.mem_insert.mp hz with rfl | hz
            · exact le_rfl
            · exact (hmax z hz).trans hle
          · refine ⟨a, Finset.mem_insert_of_mem ha, fun z hz => ?_⟩
            rcases Finset.mem_insert.mp hz with rfl | hz
            · exact hle
            · exact hmax z hz
      · obtain ⟨l, hl, hct, hdom⟩ := hers
        obtain ⟨l', hl', -⟩ := hdom x
        refine ⟨fun hc => absurd hl' (by rw [hc]; simp), fun _ => ?_⟩
        obtain ⟨a, ha, hmax⟩ := ih.2 ⟨l, hl⟩
        rw [hct] at hl' ⊢
        obtain ⟨a', ha', hmax'⟩ := erase_preserves_isGreatest hl ⟨l', hl'⟩ x (hct ▸ hdom x)
        exact ⟨a', ha', fun z hz => (hmax z (h2 ▸ hz)).trans (hmax' a ha)⟩

/-- **The hull the build ends with answers every query the input could ask.**
However insertions and erases interleave — and `add_line` interleaves them on
every call — the container standing at the end contains a line that is highest
at `x` among all the lines ever inserted.  This is the correctness of the
build, as against `Transformer.ALM.HullBuild`, which prices it, and
`Transformer.ALM.HullCover`, which counts what it holds. -/
theorem build_isGreatest_of_inserted {c s : Finset (ℝ × ℝ)}
    (h : Relation.ReflTransGen BuildStep (∅, ∅) (c, s)) (hne : c.Nonempty) (x : ℝ) :
    ∃ a ∈ c, ∀ b ∈ s, lineEval b x ≤ lineEval a x :=
  (build_invariant x h).2 hne

/-- The hypothesis is satisfiable by a build that really builds: two lines
inserted one after the other, the first then erased by the equal-slope test,
and the query at `x = 0` still answered against both. -/
example :
    ∃ a ∈ (insert ((0:ℝ), (1:ℝ)) (insert ((0:ℝ), (0:ℝ)) (∅ : Finset (ℝ × ℝ)))).erase (0, 0),
      ∀ b ∈ insert ((0:ℝ), (1:ℝ)) (insert ((0:ℝ), (0:ℝ)) (∅ : Finset (ℝ × ℝ))),
        lineEval b 0 ≤ lineEval a 0 := by
  refine build_isGreatest_of_inserted (Relation.ReflTransGen.tail
    (Relation.ReflTransGen.tail
      (Relation.ReflTransGen.single (Or.inl ⟨((0 : ℝ), (0 : ℝ)), rfl⟩))
      (Or.inl ⟨((0 : ℝ), (1 : ℝ)), rfl⟩))
    (Or.inr ⟨eraseStep_of_slope_eq (l := ((0 : ℝ), (0 : ℝ))) (l' := ((0 : ℝ), (1 : ℝ)))
      (by simp) (by simp) rfl (by norm_num), rfl⟩)) ⟨((0 : ℝ), (1 : ℝ)), by simp⟩ 0

end ALM
end Transformer
