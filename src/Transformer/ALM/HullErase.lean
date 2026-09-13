/-
# The erase loops of `add_line`, and why they change nothing

`Transformer.ALM.Envelope` proves three things separately: the equal-slope
test is sound (`lineEval_le_of_slope_eq`), the breakpoint test is sound and
necessary (`dominated_iff_le_at_interX`, `interX_le_interX_iff`), and erasing
a pointwise-dominated line leaves the upper envelope alone
(`sup'_erase_of_le`).  It then says in prose that "applying this at every `x`
for a line the two tests above reject is what makes the erase loops correct" —
and never applies it.  The composition was the claim; the pieces were the
theorems.

This file makes the composition a theorem, once for each of the two erases
`_HullCHT::add_line` performs (`transformer_vm/attention/hull2d_cht.h`,
lines 132-139 for the equal-slope early return, lines 140-192 for the two
breakpoint loops).  In both cases the conclusion is the strongest one
available: not that the erased line is dominated, but that the envelope of
the family is *equal* at every query to the envelope of the family without
it.  `erase_preserves_isGreatest` then carries that to the lookup: the best
score over the pruned hull is the best score over everything it was built
from, so the erase loops cost the machine no exactness.

Lines are the pairs `(slope, intercept)` of `Transformer.ALM.lineEval`.
-/

import Transformer.ALM.Envelope

namespace Transformer
namespace ALM

/-! ### The equal-slope erase -/

/-- **The early return of `add_line` is envelope-preserving.**  When the new
line shares a slope with a stored one and does not beat its intercept, the
code returns without inserting; dropping it from the family leaves the
envelope unchanged at every query. -/
theorem sup'_erase_of_slope_eq {s : Finset (ℝ × ℝ)} {l l' : ℝ × ℝ}
    (hl : l ∈ s) (hmem : l' ∈ s.erase l) (hm : l.1 = l'.1) (hb : l.2 ≤ l'.2) (x : ℝ) :
    s.sup' ⟨l, hl⟩ (fun a => lineEval a x)
      = (s.erase l).sup' ⟨l', hmem⟩ (fun a => lineEval a x) :=
  sup'_erase_of_le hl ⟨l', hmem⟩ x ⟨l', hmem, lineEval_le_of_slope_eq hm hb x⟩

/-- The hypotheses are satisfiable: two parallel lines, the lower one erased. -/
example : ((0:ℝ), (0:ℝ)).1 = ((0:ℝ), (1:ℝ)).1 ∧ ((0:ℝ), (0:ℝ)).2 ≤ ((0:ℝ), (1:ℝ)).2 ∧
    ((0:ℝ), (1:ℝ)) ∈ ({((0:ℝ), (0:ℝ)), (0, 1)} : Finset (ℝ × ℝ)).erase (0, 0) := by
  refine ⟨rfl, by norm_num, ?_⟩
  simp

/-! ### The breakpoint erase -/

/-- A line strictly between two others in slope is distinct from both. -/
private lemma ne_of_fst_lt {l l' : ℝ × ℝ} (h : l.1 < l'.1) : l' ≠ l :=
  fun he => absurd (congrArg Prod.fst he) (ne_of_gt h)

/-- **The erase loops of `add_line` are envelope-preserving.**  When the
breakpoint comparison `isect(l₁,l₂).p ≥ isect(l₂,l₃).p` fires — which is the
loop condition, computed from the two stored breakpoints alone — the middle
line may be erased with no effect on the envelope at any query whatsoever.
This is `sup'_erase_of_le` applied at every `x` through
`dominated_iff_le_at_interX`, the step `Transformer.ALM.Envelope` only
described. -/
theorem sup'_erase_of_interX_le {s : Finset (ℝ × ℝ)} {l₁ l₂ l₃ : ℝ × ℝ}
    (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1) (htest : interX l₂ l₃ ≤ interX l₁ l₂)
    (hl₂ : l₂ ∈ s) (h₁ : l₁ ∈ s.erase l₂) (h₃ : l₃ ∈ s.erase l₂) (x : ℝ) :
    s.sup' ⟨l₂, hl₂⟩ (fun a => lineEval a x)
      = (s.erase l₂).sup' ⟨l₁, h₁⟩ (fun a => lineEval a x) := by
  refine sup'_erase_of_le hl₂ ⟨l₁, h₁⟩ x ?_
  have hdom := (dominated_iff_le_at_interX h₁₂ h₂₃).mpr
    ((interX_le_interX_iff h₁₂ h₂₃).mp htest) x
  rcases max_cases (lineEval l₁ x) (lineEval l₃ x) with ⟨he, -⟩ | ⟨he, -⟩
  · exact ⟨l₁, h₁, by rwa [he] at hdom⟩
  · exact ⟨l₃, h₃, by rwa [he] at hdom⟩

/-- The hypotheses are satisfiable: `y = -x`, `y = 0` and `y = x` all meet at
the origin, so both breakpoints are `0` and the loop condition holds with
equality — the case the code's `>=` is written for. -/
example :
    ((-1:ℝ), (0:ℝ)).1 < ((0:ℝ), (0:ℝ)).1 ∧ ((0:ℝ), (0:ℝ)).1 < ((1:ℝ), (0:ℝ)).1 ∧
      interX ((0:ℝ), (0:ℝ)) (1, 0) ≤ interX ((-1:ℝ), (0:ℝ)) (0, 0) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  norm_num [interX]

/-! ### And so the lookup is unharmed -/

/-- **Pruning loses no argmax.**  Whatever the erased line, if some surviving
line matches it at `x` then the best value over the pruned family is the best
value over the whole one — so a query answered against the hull is answered
against every line ever inserted.  This is what the erase loops have to
preserve, stated on the lookup rather than on the envelope. -/
theorem erase_preserves_isGreatest {s : Finset (ℝ × ℝ)} {l : ℝ × ℝ} (hl : l ∈ s)
    (hne : (s.erase l).Nonempty) (x : ℝ)
    (hdom : ∃ l' ∈ s.erase l, lineEval l x ≤ lineEval l' x) :
    ∃ a ∈ s.erase l, ∀ b ∈ s, lineEval b x ≤ lineEval a x := by
  obtain ⟨a, ha, hmax⟩ := Finset.exists_max_image (s.erase l) (fun c => lineEval c x) hne
  refine ⟨a, ha, fun b hb => ?_⟩
  rcases eq_or_ne b l with rfl | hbl
  · obtain ⟨l', hl', hle⟩ := hdom
    exact hle.trans (hmax l' hl')
  · exact hmax b (Finset.mem_erase.mpr ⟨hbl, hb⟩)

/-- The hypotheses are satisfiable: the family `{(0,0), (0,1)}` with the lower
of the two parallel lines erased, dominated at `x = 0` by the one that stays. -/
example :
    ∃ a ∈ ({((0:ℝ), (0:ℝ)), (0, 1)} : Finset (ℝ × ℝ)).erase (0, 0),
      ∀ b ∈ ({((0:ℝ), (0:ℝ)), (0, 1)} : Finset (ℝ × ℝ)), lineEval b 0 ≤ lineEval a 0 :=
  erase_preserves_isGreatest (by simp) ⟨(0, 1), by simp⟩ 0
    ⟨(0, 1), by simp, by simp [lineEval]⟩

end ALM
end Transformer
