/-
# General position for the keys the compiler actually emits

`Transformer.ALM.GeneralPosition` names the hypothesis the hull head needs and
discharges it — for `liftKey k = (2k, -k²)`.  The compiler does not emit that.
`Transformer.ALM.HullMark` is the key it does emit: `markKey δ k` raises the
intercept by the recency term `δ = LATEST_ALPHA · inv_log_pos p`, and the whole
point of that term is that it *moves the keys off the paraboloid*.  Strict
concavity is what `liftKey_not_concurrent` runs on, and a perturbation is
exactly what can spend it.

So the general-position hypothesis is not inherited, and until now it was
carried only by `vm-rs/alm-hull/tests/differential.rs`, which samples keys and
skips the collinear draws.  `markKey_not_concurrent` settles it by arithmetic
instead, and says what the perturbation costs: three marked keys are concurrent
only if `δ` spreads further than the product of the two key gaps.  On integer
keys that product is at least one, and the shipped offset stays under
`0.3/log 2 = 0.433`, so the bound holds with a factor of two to spare — the
same margin `Transformer.ALM.HullMark` needs for a different reason.

`marked_not_concurrent` is the statement over the family `Marked A s` the
container actually holds.  `marked_breakpoints_ordered` is the stronger fact
beneath it: on that family the breakpoint test of `add_line` never fires at
all, because the middle of any three marked keys is strictly ahead of both
others at its own key.  So the erase that could drop a tied winner never runs
on the machine's keys, and the disagreement of
`three_collinear_keys_are_where_the_hull_head_stops_being_exact` is ruled out
rather than sampled away.

An earlier `marked_erase_preserves_tieSet` took the test firing as a hypothesis
next to `Marked A s`; the two are contradictory, so it held vacuously and is
replaced by `marked_breakpoints_ordered`.

Source: `todo3.md` §3a; `transformer_vm/graph/core.py` lines 293-319;
`vm-rs/alm-hull/tests/differential.rs`.
-/

import Transformer.ALM.GeneralPosition
import Transformer.ALM.HullMark
import Transformer.ALM.HullMono

namespace Transformer
namespace ALM

/-! ### Three marked keys never meet at a point -/

/-- **The perturbation cannot spend the gap.**  Writing `a = k₂ - k₁` and
`b = k₃ - k₂`, three marked lines meeting at one abscissa would force
`a·b·(a + b) = b(δ₁ - δ₂) - a(δ₂ - δ₃)`, whose right side is at most
`A·(a + b)`.  With integer keys `a` and `b` are at least one, so the left side
is at least `a + b`, and `A < 1` closes it.

This is the hypothesis `todo3.md` §3a asks for, on the keys the compiler emits
rather than on the lift: `Transformer.ALM.GeneralPosition.liftKey_not_concurrent`
is the case `δ = 0`, where strict concavity does the work alone.

Three of the six bounds on the offsets are not needed and are not taken.  Only
the outer two keys may not be raised past `A`, and only the middle one may not
be pushed below zero: the middle offset enters both differences with the sign
that helps, so lowering it can only make the two lines meet further apart. -/
theorem markKey_not_concurrent {A δ₁ δ₂ δ₃ k₁ k₂ k₃ : ℝ} (hA : A < 1)
    (hδ₁' : δ₁ ≤ A) (hδ₂ : 0 ≤ δ₂) (hδ₃' : δ₃ ≤ A)
    (hs₁₂ : 1 ≤ k₂ - k₁) (hs₂₃ : 1 ≤ k₃ - k₂) :
    ¬ Concurrent (markKey δ₁ k₁) (markKey δ₂ k₂) (markKey δ₃ k₃) := by
  rintro ⟨x, h12, h23⟩
  unfold lineEval markKey at h12 h23
  have e1 : (k₂ - k₁) * (2 * x - (k₁ + k₂)) = δ₁ - δ₂ := by linear_combination -h12
  have e2 : (k₃ - k₂) * (2 * x - (k₂ + k₃)) = δ₂ - δ₃ := by linear_combination -h23
  have key : (k₂ - k₁) * (k₃ - k₂) * ((k₂ - k₁) + (k₃ - k₂))
      = (k₃ - k₂) * (δ₁ - δ₂) - (k₂ - k₁) * (δ₂ - δ₃) := by
    linear_combination (k₃ - k₂) * e1 - (k₂ - k₁) * e2
  have hsum : (0 : ℝ) < (k₂ - k₁) + (k₃ - k₂) := by linarith
  have hupper : (k₃ - k₂) * (δ₁ - δ₂) - (k₂ - k₁) * (δ₂ - δ₃)
      ≤ ((k₂ - k₁) + (k₃ - k₂)) * A := by nlinarith
  have hab : (1 : ℝ) ≤ (k₂ - k₁) * (k₃ - k₂) := by nlinarith
  have hbig : (k₂ - k₁) + (k₃ - k₂)
      ≤ (k₂ - k₁) * (k₃ - k₂) * ((k₂ - k₁) + (k₃ - k₂)) := by nlinarith
  have hlt : ((k₂ - k₁) + (k₃ - k₂)) * A < (k₂ - k₁) + (k₃ - k₂) := by nlinarith
  linarith

/-- The hypotheses are satisfiable at the shipped spread, and not vacuously:
the three offsets differ, and the middle key is raised the furthest. -/
example : ¬ Concurrent (markKey 0 (0 : ℝ)) (markKey 0.4 1) (markKey 0.2 2) :=
  markKey_not_concurrent (A := 0.4) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- And the bound is the shipped constant: `0.3 · inv_log_pos p < 0.4329 < 1`
at every position, which is `Transformer.ALM.HullMark.marked_sep_of_shipped`
carried into general position. -/
example {δ₁ δ₂ δ₃ k₁ k₂ k₃ : ℝ}
    (hδ₁' : δ₁ ≤ 0.3 / Real.log 2) (hδ₂ : 0 ≤ δ₂) (hδ₃' : δ₃ ≤ 0.3 / Real.log 2)
    (hs₁₂ : 1 ≤ k₂ - k₁) (hs₂₃ : 1 ≤ k₃ - k₂) :
    ¬ Concurrent (markKey δ₁ k₁) (markKey δ₂ k₂) (markKey δ₃ k₃) :=
  markKey_not_concurrent marked_sep_of_shipped hδ₁' hδ₂ hδ₃' hs₁₂ hs₂₃

/-! ### Over the family the container holds -/

/-- **The family the compiler emits is in general position.**  `Marked A s`
says every line of `s` is a marked integer key with an offset in `[0, A]`, and
distinct integer keys are a unit apart, so `markKey_not_concurrent` applies to
any three of them taken in increasing slope. -/
theorem marked_not_concurrent {A : ℝ} {s : Finset (ℝ × ℝ)} (hA : A < 1) (hs : Marked A s)
    {l₁ l₂ l₃ : ℝ × ℝ} (h₁ : l₁ ∈ s) (h₂ : l₂ ∈ s) (h₃ : l₃ ∈ s)
    (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1) : ¬ Concurrent l₁ l₂ l₃ := by
  obtain ⟨z₁, δ₁, -, hδ₁', rfl⟩ := hs.perturbed l₁ h₁
  obtain ⟨z₂, δ₂, hδ₂, -, rfl⟩ := hs.perturbed l₂ h₂
  obtain ⟨z₃, δ₃, -, hδ₃', rfl⟩ := hs.perturbed l₃ h₃
  rw [markKey_fst, markKey_fst] at h₁₂ h₂₃
  have hz₁₂ : z₁ < z₂ := by exact_mod_cast (by linarith : (z₁ : ℝ) < z₂)
  have hz₂₃ : z₂ < z₃ := by exact_mod_cast (by linarith : (z₂ : ℝ) < z₃)
  have hs₁₂ : (1 : ℝ) ≤ (z₂ : ℝ) - (z₁ : ℝ) := by
    have : (z₁ : ℝ) + 1 ≤ (z₂ : ℝ) := by exact_mod_cast Int.add_one_le_iff.mpr hz₁₂
    linarith
  have hs₂₃ : (1 : ℝ) ≤ (z₃ : ℝ) - (z₂ : ℝ) := by
    have : (z₂ : ℝ) + 1 ≤ (z₃ : ℝ) := by exact_mod_cast Int.add_one_le_iff.mpr hz₂₃
    linarith
  exact markKey_not_concurrent hA hδ₁' hδ₂ hδ₃' hs₁₂ hs₂₃

/-- **On the family the compiler emits, the breakpoint test never fires.**
For three members of `Marked A s` in increasing slope, the middle one is kept:
`interX l₁ l₂ < interX l₂ l₃`.  At its own key `k₂` the middle line is
strictly ahead of both others (`lineEval_markKey_lt`), and a line strictly
ahead of both neighbours somewhere is ahead of them at their crossing.

So `add_line` erases nothing from a marked family, and the question of what an
erase does to the tie set the payload merge walks never arises on it.

Source: `todo3.md` §3a; `hull2d_cht.h`, lines 122-192, the breakpoint test. -/
theorem marked_breakpoints_ordered {A : ℝ} {s : Finset (ℝ × ℝ)} (hA : A < 1)
    (hs : Marked A s) {l₁ l₂ l₃ : ℝ × ℝ} (h₁ : l₁ ∈ s) (h₂ : l₂ ∈ s) (h₃ : l₃ ∈ s)
    (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1) : interX l₁ l₂ < interX l₂ l₃ := by
  obtain ⟨z₁, δ₁, hδ₁, hδ₁', rfl⟩ := hs.perturbed l₁ h₁
  obtain ⟨z₂, δ₂, hδ₂, hδ₂', rfl⟩ := hs.perturbed l₂ h₂
  obtain ⟨z₃, δ₃, hδ₃, hδ₃', rfl⟩ := hs.perturbed l₃ h₃
  have h₁₂' := h₁₂
  have h₂₃' := h₂₃
  rw [markKey_fst, markKey_fst] at h₁₂' h₂₃'
  have hz₁₂ : z₁ < z₂ := by exact_mod_cast (by linarith : (z₁ : ℝ) < z₂)
  have hz₂₃ : z₂ < z₃ := by exact_mod_cast (by linarith : (z₂ : ℝ) < z₃)
  have hs₁₂ : (1 : ℝ) ≤ (z₂ : ℝ) - (z₁ : ℝ) := by
    have : (z₁ : ℝ) + 1 ≤ (z₂ : ℝ) := by exact_mod_cast Int.add_one_le_iff.mpr hz₁₂
    linarith
  have hs₂₃ : (1 : ℝ) ≤ (z₃ : ℝ) - (z₂ : ℝ) := by
    have : (z₂ : ℝ) + 1 ≤ (z₃ : ℝ) := by exact_mod_cast Int.add_one_le_iff.mpr hz₂₃
    linarith
  set l₁ := markKey δ₁ (z₁ : ℝ)
  set l₂ := markKey δ₂ (z₂ : ℝ)
  set l₃ := markKey δ₃ (z₃ : ℝ)
  set x₀ : ℝ := (z₂ : ℝ)
  have e₁ : lineEval l₁ x₀ < lineEval l₂ x₀ :=
    lineEval_markKey_lt (by nlinarith) (by linarith)
  have e₃ : lineEval l₃ x₀ < lineEval l₂ x₀ :=
    lineEval_markKey_lt (by nlinarith) (by linarith)
  refine (interX_lt_interX_iff h₁₂ h₂₃).mpr ?_
  set m := interX l₁ l₃
  have hm : lineEval l₁ m = lineEval l₃ m := lineEval_interX l₁ l₃ (h₁₂.trans h₂₃).ne
  have lin : ∀ (l : ℝ × ℝ) (x y : ℝ), lineEval l y = lineEval l x + l.1 * (y - x) := by
    intro l x y; unfold lineEval; ring
  rcases le_total x₀ m with hx | hx
  · rw [lin l₁ x₀ m, lin l₂ x₀ m]
    nlinarith
  · rw [hm, lin l₃ x₀ m, lin l₂ x₀ m]
    nlinarith

/-- The hypotheses of `marked_not_concurrent` and `marked_breakpoints_ordered`
are satisfiable, at the shipped spread and with three different offsets: the
keys `0, 1, 2` raised by `0, 0.4, 0.2`. -/
example :
    ¬ Concurrent (markKey 0 ((0 : ℤ) : ℝ)) (markKey 0.4 ((1 : ℤ) : ℝ)) (markKey 0.2 ((2 : ℤ) : ℝ)) ∧
      interX (markKey 0 ((0 : ℤ) : ℝ)) (markKey 0.4 ((1 : ℤ) : ℝ))
        < interX (markKey 0.4 ((1 : ℤ) : ℝ)) (markKey 0.2 ((2 : ℤ) : ℝ)) := by
  have hs : Marked 0.4 ({markKey 0 ((0 : ℤ) : ℝ), markKey 0.4 ((1 : ℤ) : ℝ),
      markKey 0.2 ((2 : ℤ) : ℝ)} : Finset (ℝ × ℝ)) := by
    constructor
    · intro l hl
      simp only [Finset.mem_insert, Finset.mem_singleton] at hl
      rcases hl with rfl | rfl | rfl
      exacts [⟨0, 0, le_rfl, by norm_num, rfl⟩, ⟨1, 0.4, by norm_num, le_rfl, rfl⟩,
        ⟨2, 0.2, by norm_num, by norm_num, rfl⟩]
    · intro l hl l' hl' hf
      simp only [Finset.mem_insert, Finset.mem_singleton] at hl hl'
      rcases hl with rfl | rfl | rfl <;> rcases hl' with rfl | rfl | rfl <;>
        first | rfl | (exfalso; simp only [markKey_fst] at hf; norm_num at hf)
  have hlt : ∀ a b : ℤ, a < b → ∀ δ δ' : ℝ,
      (markKey δ (a : ℝ)).1 < (markKey δ' (b : ℝ)).1 := fun a b h _ _ => by
    rw [markKey_fst, markKey_fst]
    have : (a : ℝ) < b := by exact_mod_cast h
    linarith
  exact ⟨marked_not_concurrent (by norm_num) hs (by simp) (by simp) (by simp)
      (hlt 0 1 (by norm_num) _ _) (hlt 1 2 (by norm_num) _ _),
    marked_breakpoints_ordered (by norm_num) hs (by simp) (by simp) (by simp)
      (hlt 0 1 (by norm_num) _ _) (hlt 1 2 (by norm_num) _ _)⟩

end ALM
end Transformer
