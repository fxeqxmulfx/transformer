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
container actually holds, and `marked_erase_preserves_tieSet` is the payoff:
the erase inside `add_line` drops a line that is a maximizer at no query, so
the tie set the payload merge walks is preserved and the hull head answers what
the brute head answers.  That is the disagreement of
`three_collinear_keys_are_where_the_hull_head_stops_being_exact` ruled out on
the machine's own keys rather than sampled away.

Source: `todo3.md` §3a; `transformer_vm/graph/core.py` lines 293-319;
`vm-rs/alm-hull/tests/differential.rs`.
-/

import Transformer.ALM.GeneralPosition
import Transformer.ALM.HullMark

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

/-- **So the erase preserves the payload and not only the value.**  On the
family the compiler emits, the line `add_line`'s breakpoint test rejects is a
maximizer at no query at all, so the tie set `HullHalf::query` merges over is
unchanged and the hull head returns what the brute head returns.

`Transformer.ALM.TieSet.erase_drops_the_winner` is the failure this rules out,
and `vm-rs/alm-hull/tests/differential.rs` exhibits it on three collinear keys
— keys no compiled head holds. -/
theorem marked_erase_preserves_tieSet {A : ℝ} {s : Finset (ℝ × ℝ)} (hA : A < 1)
    (hs : Marked A s) {l₁ l₂ l₃ : ℝ × ℝ} (h₁ : l₁ ∈ s) (h₂ : l₂ ∈ s) (h₃ : l₃ ∈ s)
    (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1) (hfire : interX l₂ l₃ ≤ interX l₁ l₂) (x : ℝ) :
    tieSet (s.erase l₂) x = tieSet s x :=
  erase_preserves_tieSet h₁ h₃ h₁₂ h₂₃ hfire (marked_not_concurrent hA hs h₁ h₂ h₃ h₁₂ h₂₃) x

end ALM
end Transformer
