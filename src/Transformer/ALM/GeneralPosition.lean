/-
# Keys in general position, and why the machine has them

`Transformer.ALM.TieSet` shows what the erase loops of `add_line` can cost: at
a query where three stored lines agree, the test fires on the middle one, the
envelope keeps its value and the tie set the payloads are merged over loses a
member.  `todo3.md` §3a asks for the hypothesis that rules this out to be
stated, since `HullKVCache` is offered as a general drop-in and neither post
nor code names it.

Here it is.  `erase_preserves_tieSet`: if the three lines of a loop iteration
are not concurrent — equivalently, if the three keys are not collinear — then
the rejected line is a maximizer at no query at all, and the erase preserves
the tie set and not merely the value.  So the hull head and the brute head
agree for keys in general position, and anything else needs the brute head or
a hull that keeps collinear vertices.

`liftKey_not_concurrent` is why the released machine never meets the
degenerate case: two lifted keys cross at their midpoint, so three of them
concurrent would need `k₁ = k₃`.  That is the strict concavity of
`k ↦ (2k, -k²)` and nothing the code checks — it is the hypothesis that fails
the moment a head is keyed on something the compiler did not lift.

Source: `todo3.md` §3a; `transformer_vm/attention/hull2d_cht.h`, lines 143-195.
-/

import Transformer.ALM.TieSet
import Transformer.ALM.Hull

namespace Transformer
namespace ALM

/-! ### What the erase costs when the three lines do not meet -/

/-- With the breakpoint test satisfied strictly, the middle line is strictly
below the envelope of its two neighbours at *every* query, so it is a
maximizer nowhere. -/
lemma lt_max_of_lt_at_interX {l₁ l₂ l₃ : ℝ × ℝ} (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1)
    (h : lineEval l₂ (interX l₁ l₃) < lineEval l₁ (interX l₁ l₃)) (x : ℝ) :
    lineEval l₂ x < max (lineEval l₁ x) (lineEval l₃ x) := by
  have hne : l₁.1 ≠ l₃.1 := by linarith
  have heq : lineEval l₁ (interX l₁ l₃) = lineEval l₃ (interX l₁ l₃) :=
    lineEval_interX l₁ l₃ hne
  set s := interX l₁ l₃ with hs
  rcases le_total x s with hx | hx
  · refine lt_of_lt_of_le ?_ (le_max_left _ _)
    have hmul : (l₂.1 - l₁.1) * x ≤ (l₂.1 - l₁.1) * s :=
      mul_le_mul_of_nonneg_left hx (by linarith)
    unfold lineEval at h ⊢
    nlinarith [hmul]
  · refine lt_of_lt_of_le ?_ (le_max_right _ _)
    have hmul : (l₃.1 - l₂.1) * s ≤ (l₃.1 - l₂.1) * x :=
      mul_le_mul_of_nonneg_left hx (by linarith)
    rw [heq] at h
    unfold lineEval at h ⊢
    nlinarith [hmul]

/-- **In general position the erase preserves the tie set.**  If the three
lines of one loop iteration are not concurrent — equivalently, the three keys
are not collinear — then the line the test rejects is a maximizer at no query
at all, and dropping it changes neither the value nor the set of lines the
merge walk collects.  This is the hypothesis `todo3.md` §3a asks to be stated:
under it the hull head and the brute head answer alike. -/
theorem erase_preserves_tieSet {s : Finset (ℝ × ℝ)} {l₁ l₂ l₃ : ℝ × ℝ}
    (h₁ : l₁ ∈ s) (h₃ : l₃ ∈ s) (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1)
    (hfire : interX l₂ l₃ ≤ interX l₁ l₂) (hgp : ¬ Concurrent l₁ l₂ l₃) (x : ℝ) :
    tieSet (s.erase l₂) x = tieSet s x := by
  classical
  have hne13 : l₁.1 ≠ l₃.1 := by linarith
  have hle : lineEval l₂ (interX l₁ l₃) ≤ lineEval l₁ (interX l₁ l₃) :=
    (interX_le_interX_iff h₁₂ h₂₃).mp hfire
  have hlt : lineEval l₂ (interX l₁ l₃) < lineEval l₁ (interX l₁ l₃) := by
    refine lt_of_le_of_ne hle fun heq => hgp ⟨interX l₁ l₃, heq.symm, ?_⟩
    rw [heq]
    exact lineEval_interX l₁ l₃ hne13
  have hdom : ∀ y : ℝ, lineEval l₂ y < max (lineEval l₁ y) (lineEval l₃ y) :=
    lt_max_of_lt_at_interX h₁₂ h₂₃ hlt
  have hbeat : ∃ l ∈ s.erase l₂, lineEval l₂ x < lineEval l x := by
    rcases max_cases (lineEval l₁ x) (lineEval l₃ x) with ⟨hmax, _⟩ | ⟨hmax, _⟩
    · exact ⟨l₁, Finset.mem_erase.mpr
        ⟨fun h => absurd (congrArg Prod.fst h) (ne_of_lt h₁₂), h₁⟩, hmax ▸ hdom x⟩
    · exact ⟨l₃, Finset.mem_erase.mpr
        ⟨fun h => absurd (congrArg Prod.fst h) (ne_of_gt h₂₃), h₃⟩, hmax ▸ hdom x⟩
  obtain ⟨l, hl, hlt'⟩ := hbeat
  obtain ⟨hlne, hls⟩ := Finset.mem_erase.mp hl
  ext l₀
  simp only [mem_tieSet, Finset.mem_erase]
  constructor
  · rintro ⟨⟨_, h₀s⟩, hbest⟩
    refine ⟨h₀s, fun l' hl' => ?_⟩
    rcases eq_or_ne l' l₂ with rfl | hne
    · exact le_of_lt (lt_of_lt_of_le hlt' (hbest l ⟨hlne, hls⟩))
    · exact hbest l' ⟨hne, hl'⟩
  · rintro ⟨h₀s, hbest⟩
    have h₀ne : l₀ ≠ l₂ := by
      rintro rfl
      exact absurd (hbest l hls) (not_le.mpr hlt')
    exact ⟨⟨h₀ne, h₀s⟩, fun l' hl' => hbest l' hl'.2⟩

/-! ### Why the machine never meets the degenerate case -/

/-- **The paraboloid is in general position.**  Two lifted keys cross at their
midpoint, so three of them meeting at one point would need `k₁ = k₃`.  Nothing
in `add_line` checks this; it is the strict concavity of `k ↦ (2k, -k²)` that
keeps `HullKVCache` equivalent to the attention it replaces, and it is the
hypothesis that fails the moment a head is keyed on anything else. -/
theorem liftKey_not_concurrent {k₁ k₂ k₃ : ℝ} (h₁₂ : k₁ < k₂) (h₂₃ : k₂ < k₃) :
    ¬ Concurrent (liftKey k₁) (liftKey k₂) (liftKey k₃) := by
  have s₁₂ : (liftKey k₁).1 < (liftKey k₂).1 := by unfold liftKey; simp; linarith
  have s₂₃ : (liftKey k₂).1 < (liftKey k₃).1 := by unfold liftKey; simp; linarith
  intro h
  have := interX_eq_of_concurrent s₁₂ s₂₃ h
  rw [interX_liftKey (ne_of_lt h₂₃), interX_liftKey (ne_of_lt h₁₂)] at this
  linarith

/-! ### The hypotheses are satisfiable -/

/-- The triple that keeps its tie set: `y = -x`, `y = -1`, `y = x`.  The
breakpoint test still fires — the flat line is redundant — but the three keys
`(-1, 0)`, `(0, -1)`, `(1, 0)` are not collinear, so the erased line was a
maximizer nowhere.  These are the hypotheses of `erase_preserves_tieSet` and
`lt_max_of_lt_at_interX`. -/
example :
    ((-1, 0) : ℝ × ℝ).1 < ((0, -1) : ℝ × ℝ).1 ∧ ((0, -1) : ℝ × ℝ).1 < ((1, 0) : ℝ × ℝ).1 ∧
      interX ((0, -1) : ℝ × ℝ) (1, 0) ≤ interX ((-1, 0) : ℝ × ℝ) (0, -1) ∧
      ¬ Concurrent ((-1, 0) : ℝ × ℝ) (0, -1) (1, 0) ∧
      lineEval ((0, -1) : ℝ × ℝ) (interX ((-1, 0) : ℝ × ℝ) (1, 0))
        < lineEval ((-1, 0) : ℝ × ℝ) (interX ((-1, 0) : ℝ × ℝ) (1, 0)) := by
  refine ⟨by norm_num, by norm_num, by norm_num [interX], ?_, by norm_num [interX, lineEval]⟩
  rw [concurrent_iff_collinear (by norm_num)]
  norm_num

/-- And the lift's own hypotheses: three keys in increasing order. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 := by norm_num

end ALM
end Transformer
