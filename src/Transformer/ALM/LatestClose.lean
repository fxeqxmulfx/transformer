/-
# And the window shuts, at a position that can be named

`Transformer.ALM.LatestWindow` bounds the perturbation on both sides and shows
the released `LATEST_ALPHA = 0.3` inside them.  The upper side is a constant and
holds forever.  The lower side is not: it asks that one step of `invLogPos` stay
wider than the rounding, and the steps shrink like `1/((p+2) log²(p+2))` while
the rounding does not shrink at all.

So the guarantee has a shelf life, and this file gives it a name.  Past
`⌈exp (α / 2η)⌉` a rounding inside the budget makes the *stale* write win — not
a weaker bound, an inverted answer, with nothing to announce it.  `todo3.md` §2a
measures where that sits on the released model: the closest runner-up on `hello`
is `6.026e-6` against a noise floor of `4.3e-15`, which puts the crossing at
`p ≈ 8·10^10` on heads keyed on small integers, and — §4b — already behind us on
a head keyed on a 32-bit WebAssembly value, where `ulp(ky) ≈ 16` key steps.

That is what makes `HullMeta::last_seq` worth keeping even though the released
weights never reach it: it is the only mechanism left once the perturbation is
gone, and `Transformer.ALM.HullResolve` is what says it answers.

Source: `todo3.md` §2a and §4b; `transformer_vm/evaluator.py:230`.
-/

import Transformer.ALM.LatestWindow

namespace Transformer
namespace ALM

/-- The step the perturbation takes between consecutive positions is
`1/log (p+2) - 1/log (p+3)`, so at most `1/log (p+2)` — and that tends to zero
while the rounding does not. -/
lemma invLogPos_step_le (p : ℕ) :
    invLogPos (p + 1) - invLogPos p ≤ 1 / Real.log ((p : ℝ) + 2) := by
  have hc : ((p + 1 : ℕ) : ℝ) + 2 = (p : ℝ) + 3 := by push_cast; ring
  have h3 : 0 < Real.log ((p : ℝ) + 3) := by
    have hp : (0 : ℝ) ≤ (p : ℝ) := Nat.cast_nonneg p
    exact Real.log_pos (by linarith)
  have hpos : 0 < 1 / Real.log ((p : ℝ) + 3) := by positivity
  simp only [invLogPos, hc]; linarith

/-- **So for every scale and every rounding there is a position past which the
stale write can win.**  Fix the perturbation scale `α` and a rounding budget
`η > 0`.  Past `P` the step `invLogPos` takes between consecutive positions is
narrower than the rounding, and then a rounding inside the budget — `+η` on the
earlier write, `-η` on the later — makes the *earlier* one score at least as
high.  Nothing announces it: the head returns a stale value and the query looks
like any other.  This is §2a's "past some length the ordering is lost silently
and the answer becomes whichever the rounding prefers", and it is why the
unreached fallback `HullMeta::last_seq` has to stay reachable. -/
theorem past_the_window_rounding_decides {α η : ℝ} (hα : 0 < α) (hη : 0 < η)
    (q : ℝ) (k : ℤ) :
    ∃ P : ℕ, ∀ p : ℕ, P ≤ p → ∃ ν₁ ν₂ : ℝ, |ν₁| ≤ η ∧ |ν₂| ≤ η ∧
      writeScore α q k (p + 1) ν₂ ≤ writeScore α q k p ν₁ := by
  have ht : 0 < α / (2 * η) := by positivity
  refine ⟨⌈Real.exp (α / (2 * η))⌉₊, fun p hp => ⟨η, -η, ?_, ?_, ?_⟩⟩
  · rw [abs_of_nonneg hη.le]
  · rw [abs_neg, abs_of_nonneg hη.le]
  · have hPp : Real.exp (α / (2 * η)) ≤ (p : ℝ) + 2 := by
      have h1 : Real.exp (α / (2 * η)) ≤ (⌈Real.exp (α / (2 * η))⌉₊ : ℝ) := Nat.le_ceil _
      have h2 : ((⌈Real.exp (α / (2 * η))⌉₊ : ℕ) : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
      linarith
    have hlog : α / (2 * η) ≤ Real.log ((p : ℝ) + 2) := by
      have := Real.log_le_log (Real.exp_pos _) hPp
      rwa [Real.log_exp] at this
    have hinv : 1 / Real.log ((p : ℝ) + 2) ≤ 2 * η / α := by
      have h := one_div_le_one_div_of_le ht hlog
      rwa [one_div_div] at h
    have hbound : α * (invLogPos (p + 1) - invLogPos p) ≤ 2 * η := by
      have h1 := mul_le_mul_of_nonneg_left (invLogPos_step_le p) hα.le
      have h2 := mul_le_mul_of_nonneg_left hinv hα.le
      have h3 : α * (2 * η / α) = 2 * η := by field_simp
      linarith
    have hexp : α * (invLogPos (p + 1) - invLogPos p)
        = α * invLogPos (p + 1) - α * invLogPos p := by rw [mul_sub]
    simp only [writeScore]; linarith

/-- Satisfiable at the released constant and a rounding of the size §2a
measures on `hello` — `4.3e-15` key steps, the floor the matvec leaves. -/
example : (0 : ℝ) < 0.3 ∧ (0 : ℝ) < 4.3e-15 := by norm_num

end ALM
end Transformer
