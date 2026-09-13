/-
# Why the hull may discard a line

`Transformer.ALM.Duality` reduces the planar argmax to a one-dimensional
upper envelope.  The machine never stores that envelope as a list of all
lines: `_HullCHT::add_line` in `transformer_vm/attention/hull2d_cht.h`
(lines 122-192) erases a line as soon as it is no longer on the envelope,
using two tests and no others.

* Equal slopes (`add_line`, lines 132-139): the line with the smaller
  intercept is dominated everywhere.
* Three consecutive slopes (`isect`, lines 123-137, and the erase loops of
  `add_line`): the middle line is dropped when the breakpoints cross, i.e.
  when `isect(x,y).p ≥ isect(y,z).p`.

This file proves that both tests are sound and, for the second one, that it
is also *necessary* — the hull keeps exactly the lines that are somewhere
strictly best.  It then shows that erasing a pointwise-dominated line leaves
the envelope unchanged, which is the invariant `add_line` maintains.

Lines are the pairs `(slope, intercept)` of `Transformer.ALM.lineEval`.
-/

import Transformer.ALM.Duality

namespace Transformer
namespace ALM

/-- The abscissa at which two lines of different slope meet.  This is the
`x->p` computed by `isect`: `(y.b - x.b) / (x.m - y.m)`. -/
noncomputable def interX (l l' : ℝ × ℝ) : ℝ := (l'.2 - l.2) / (l.1 - l'.1)

/-- At `interX l l'` the two lines agree, provided their slopes differ. -/
theorem lineEval_interX (l l' : ℝ × ℝ) (h : l.1 ≠ l'.1) :
    lineEval l (interX l l') = lineEval l' (interX l l') := by
  unfold lineEval interX
  field_simp [sub_ne_zero_of_ne h]
  ring

/-! ### The equal-slope test -/

/-- **Equal slopes.**  A line is dominated everywhere by a parallel line with
a larger intercept.  This is the `it->b >= b` early return of `add_line`. -/
theorem lineEval_le_of_slope_eq {l l' : ℝ × ℝ} (hm : l.1 = l'.1) (hb : l.2 ≤ l'.2)
    (x : ℝ) : lineEval l x ≤ lineEval l' x := by
  unfold lineEval
  rw [hm]
  linarith

/-! ### The three-line test -/

/-- **Soundness and necessity of the breakpoint test, geometric form.**  For
three lines of strictly increasing slope, the middle one is nowhere above the
upper envelope of the outer two exactly when it lies below them at their
crossing point. -/
theorem dominated_iff_le_at_interX {l₁ l₂ l₃ : ℝ × ℝ}
    (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1) :
    (∀ x, lineEval l₂ x ≤ max (lineEval l₁ x) (lineEval l₃ x)) ↔
      lineEval l₂ (interX l₁ l₃) ≤ lineEval l₁ (interX l₁ l₃) := by
  have hne : l₁.1 ≠ l₃.1 := by linarith
  have heq : lineEval l₁ (interX l₁ l₃) = lineEval l₃ (interX l₁ l₃) :=
    lineEval_interX l₁ l₃ hne
  constructor
  · intro h
    have hx := h (interX l₁ l₃)
    rwa [← heq, max_self] at hx
  · intro h x
    set s := interX l₁ l₃ with hs
    rcases le_total x s with hx | hx
    · refine le_trans ?_ (le_max_left _ _)
      have hmul : (l₂.1 - l₁.1) * x ≤ (l₂.1 - l₁.1) * s :=
        mul_le_mul_of_nonneg_left hx (by linarith)
      unfold lineEval at h ⊢
      nlinarith [hmul]
    · refine le_trans ?_ (le_max_right _ _)
      have hmul : (l₃.1 - l₂.1) * s ≤ (l₃.1 - l₂.1) * x :=
        mul_le_mul_of_nonneg_left hx (by linarith)
      rw [heq] at h
      unfold lineEval at h ⊢
      nlinarith [hmul]

/-- **The test as the code computes it.**  The geometric condition above is
the breakpoint comparison `isect(l₁,l₂).p ≥ isect(l₂,l₃).p` that `add_line`
loops on.  Neither side mentions the outer crossing point, so the hull can
decide with the two breakpoints it already stores. -/
theorem interX_le_interX_iff {l₁ l₂ l₃ : ℝ × ℝ}
    (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1) :
    interX l₂ l₃ ≤ interX l₁ l₂ ↔
      lineEval l₂ (interX l₁ l₃) ≤ lineEval l₁ (interX l₁ l₃) := by
  have d₁ : (0:ℝ) < l₂.1 - l₁.1 := by linarith
  have d₂ : (0:ℝ) < l₃.1 - l₂.1 := by linarith
  have d₃ : (0:ℝ) < l₃.1 - l₁.1 := by linarith
  have e₁ : interX l₁ l₂ = (l₁.2 - l₂.2) / (l₂.1 - l₁.1) := by
    unfold interX; rw [div_eq_div_iff (by linarith) (by linarith)]; ring
  have e₂ : interX l₂ l₃ = (l₂.2 - l₃.2) / (l₃.1 - l₂.1) := by
    unfold interX; rw [div_eq_div_iff (by linarith) (by linarith)]; ring
  have e₃ : interX l₁ l₃ = (l₁.2 - l₃.2) / (l₃.1 - l₁.1) := by
    unfold interX; rw [div_eq_div_iff (by linarith) (by linarith)]; ring
  have left : interX l₂ l₃ ≤ interX l₁ l₂ ↔
      (l₂.2 - l₃.2) * (l₂.1 - l₁.1) ≤ (l₁.2 - l₂.2) * (l₃.1 - l₂.1) := by
    rw [e₁, e₂, div_le_div_iff₀ d₂ d₁]
  have right : lineEval l₂ (interX l₁ l₃) ≤ lineEval l₁ (interX l₁ l₃) ↔
      0 ≤ (l₁.2 - l₂.2) * (l₃.1 - l₁.1) - (l₂.1 - l₁.1) * (l₁.2 - l₃.2) := by
    rw [e₃]
    unfold lineEval
    rw [← sub_nonneg]
    have hrw : l₁.1 * ((l₁.2 - l₃.2) / (l₃.1 - l₁.1)) + l₁.2
          - (l₂.1 * ((l₁.2 - l₃.2) / (l₃.1 - l₁.1)) + l₂.2)
        = ((l₁.2 - l₂.2) * (l₃.1 - l₁.1) - (l₂.1 - l₁.1) * (l₁.2 - l₃.2))
            / (l₃.1 - l₁.1) := by
      field_simp; ring
    rw [hrw, le_div_iff₀ d₃, zero_mul]
  rw [left, right]
  constructor <;> intro h <;> nlinarith [h]

/-! ### Erasing a dominated line leaves the envelope alone -/

/-- **The invariant `add_line` maintains.**  If at `x` some other retained
line does at least as well as `l`, then dropping `l` does not change the
envelope's value at `x`.  Applying this at every `x` for a line the two tests
above reject is what makes the erase loops correct. -/
theorem sup'_erase_of_le {s : Finset (ℝ × ℝ)} {l : ℝ × ℝ} (hl : l ∈ s)
    (hne : (s.erase l).Nonempty) (x : ℝ)
    (hdom : ∃ l' ∈ s.erase l, lineEval l x ≤ lineEval l' x) :
    s.sup' ⟨l, hl⟩ (fun a => lineEval a x)
      = (s.erase l).sup' hne (fun a => lineEval a x) := by
  obtain ⟨l', hl', hle⟩ := hdom
  refine le_antisymm (Finset.sup'_le _ _ fun a ha => ?_)
    (Finset.sup'_le _ _ fun a ha =>
      Finset.le_sup' (fun b => lineEval b x) (Finset.mem_of_mem_erase ha))
  rcases eq_or_ne a l with rfl | hal
  · exact le_trans hle (Finset.le_sup' (fun b => lineEval b x) hl')
  · exact Finset.le_sup' (fun b => lineEval b x) (Finset.mem_erase.mpr ⟨hal, ha⟩)

/-! ### Satisfiability of the hypotheses -/

private lemma eval_flat (y : ℝ) : lineEval ((0:ℝ), (0:ℝ)) y = 0 := by simp [lineEval]

private lemma eval_down (y : ℝ) : lineEval ((-1:ℝ), (0:ℝ)) y = -y := by simp [lineEval]

private lemma eval_up (y : ℝ) : lineEval ((1:ℝ), (0:ℝ)) y = y := by simp [lineEval]

/-- Three lines of increasing slope with the middle one redundant: `y = -x`,
`y = 0` and `y = x` all meet at the origin, so the flat line never rises above
the envelope of the other two and the test fires. -/
example :
    ((-1, 0) : ℝ × ℝ).1 < ((0, 0) : ℝ × ℝ).1 ∧ ((0, 0) : ℝ × ℝ).1 < ((1, 0) : ℝ × ℝ).1 ∧
      ∀ x : ℝ, lineEval (0, 0) x ≤ max (lineEval (-1, 0) x) (lineEval (1, 0) x) := by
  refine ⟨by norm_num, by norm_num, fun x => ?_⟩
  rw [eval_flat, eval_down, eval_up]
  rcases le_total x 0 with hx | hx
  · exact le_trans (by linarith) (le_max_left (-x) x)
  · exact le_trans (by linarith) (le_max_right (-x) x)

/-- And a middle line that must be kept: lifting its intercept puts it strictly
above the other two at `x = 0`, so the test does not fire and the hypothesis of
`dominated_iff_le_at_interX` is not vacuous. -/
example : ¬ (∀ x : ℝ, lineEval (0, 1) x ≤ max (lineEval (-1, 0) x) (lineEval (1, 0) x)) := by
  intro h
  have hx := h 0
  rw [eval_down, eval_up] at hx
  simp [lineEval] at hx
  linarith

/-- The erasure hypotheses are satisfiable: a two-line family in which the
erased line is the worse one at `x = 0`. -/
example :
    ∃ l' ∈ ({((0:ℝ), (0:ℝ)), (0, 1)} : Finset (ℝ × ℝ)).erase (0, 0),
      lineEval ((0:ℝ), (0:ℝ)) 0 ≤ lineEval l' 0 := by
  refine ⟨(0, 1), ?_, ?_⟩
  · simp
  · simp [lineEval]

end ALM
end Transformer
