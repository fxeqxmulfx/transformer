/-
# The other half keeps two lines, whatever the trace does

A head is two containers: `HullHalf` inserts every key into an upper envelope
and the negated key into a lower one, and a query goes to whichever half the
sign of `qy` selects.  `Transformer.ALM.HullLift` prices the upper half — under
the lift `k ↦ (2k, -k²)` no erase rule applies at all, so it ends holding one
line per distinct key.  This file prices the other half, and the answer is the
opposite: `2`.

The reason is the same concavity read the other way.  Negating a lifted key
turns the score into `k² - 2kx`, which is *convex* in the key, so a key with
one neighbour on each side is below the higher of them at every query and the
erase loop discards it — `eraseStep_neg_liftKey`.  What survives is the two
extremes, and `lower_envelope_eq_extremes` shows they answer every query the
whole family would have.

That is `vm-rs/alm-hull/src/bin/alm-stress.rs` measured: `262144` keys in five
arrival orders, `upper 262144` and `lower 2` on every line of the table.  The
`128` bytes per token `todo3.md` §5 measures is therefore the upper half alone,
and halving the container is not among the ways to reclaim it.

Source: `todo3.md` §5; `transformer_vm/attention/hull2d_cht.h` (the two halves);
`vm-rs/alm-hull/src/bin/alm-stress.rs`.
-/

import Transformer.ALM.HullLift
import Transformer.ALM.HullLines

namespace Transformer
namespace ALM

/-! ### The negated lift is convex in the key -/

/-- What the lower half stores: the negated lifted key scores `k² - 2kx`. -/
theorem lineEval_neg_liftKey (k x : ℝ) : lineEval (-liftKey k) x = k ^ 2 - 2 * k * x := by
  rw [lineEval_neg]
  unfold lineEval liftKey
  ring

/-- **A key between two others is below the higher of them, everywhere.**  The
second difference of `k ↦ k² - 2kx` is the product of the three key gaps and
does not depend on the query, so the middle line of a negated triple is
strictly dominated at every `x` at once — where on the upper half
(`liftKey_not_dominated`) it was dominated at none. -/
theorem neg_liftKey_dominated {k₁ k₂ k₃ : ℝ} (h₁₂ : k₁ < k₂) (h₂₃ : k₂ < k₃) (x : ℝ) :
    lineEval (-liftKey k₂) x
      < max (lineEval (-liftKey k₁) x) (lineEval (-liftKey k₃) x) := by
  simp only [lineEval_neg_liftKey]
  have hid : ((k₃ - k₂) * (k₁ ^ 2 - 2 * k₁ * x) + (k₂ - k₁) * (k₃ ^ 2 - 2 * k₃ * x))
      - (k₃ - k₁) * (k₂ ^ 2 - 2 * k₂ * x) = (k₃ - k₂) * (k₂ - k₁) * (k₃ - k₁) := by ring
  have hpos : 0 < (k₃ - k₂) * (k₂ - k₁) * (k₃ - k₁) := by
    have h₁₃ : k₁ < k₃ := h₁₂.trans h₂₃
    have : 0 < (k₃ - k₂) * (k₂ - k₁) := mul_pos (by linarith) (by linarith)
    exact mul_pos this (by linarith)
  rcases le_total (k₁ ^ 2 - 2 * k₁ * x) (k₃ ^ 2 - 2 * k₃ * x) with h | h
  · refine lt_of_lt_of_le ?_ (le_max_right _ _)
    nlinarith
  · refine lt_of_lt_of_le ?_ (le_max_left _ _)
    nlinarith

/-- **So the erase loop fires on it.**  With both neighbours in the container,
the middle negated key satisfies the erase relation of
`Transformer.ALM.HullPrune`: some surviving line is at least as high at every
query.  This is the loop of `add_line` doing on the lower half exactly what
`not_eraseStep_of_lift` forbids it on the upper. -/
theorem eraseStep_neg_liftKey {s : Finset (ℝ × ℝ)} {k₁ k₂ k₃ : ℝ}
    (h₁₂ : k₁ < k₂) (h₂₃ : k₂ < k₃) (h₂ : -liftKey k₂ ∈ s)
    (h₁ : -liftKey k₁ ∈ s.erase (-liftKey k₂)) (h₃ : -liftKey k₃ ∈ s.erase (-liftKey k₂)) :
    EraseStep s (s.erase (-liftKey k₂)) := by
  refine ⟨-liftKey k₂, h₂, rfl, fun x => ?_⟩
  rcases max_cases (lineEval (-liftKey k₁) x) (lineEval (-liftKey k₃) x) with
    ⟨hmax, _⟩ | ⟨hmax, _⟩
  · exact ⟨-liftKey k₁, h₁, le_of_lt (hmax ▸ neg_liftKey_dominated h₁₂ h₂₃ x)⟩
  · exact ⟨-liftKey k₃, h₃, le_of_lt (hmax ▸ neg_liftKey_dominated h₁₂ h₂₃ x)⟩

/-! ### And the extremes answer for everyone -/

/-- **The lower envelope is the envelope of its two extreme keys.**  However
many keys the trace inserted, at every query the highest negated line is the
smallest key's or the largest key's. -/
theorem lower_envelope_eq_extremes {S : Finset ℝ} (hS : S.Nonempty) (x : ℝ) :
    S.sup' hS (fun k => lineEval (-liftKey k) x)
      = max (lineEval (-liftKey (S.min' hS)) x) (lineEval (-liftKey (S.max' hS)) x) := by
  refine le_antisymm (Finset.sup'_le _ _ fun k hk => ?_) ?_
  · rcases eq_or_lt_of_le (S.min'_le k hk) with heq | hlo
    · exact heq ▸ le_max_left _ _
    rcases eq_or_lt_of_le (S.le_max' k hk) with heq | hhi
    · exact heq ▸ le_max_right _ _
    exact (neg_liftKey_dominated hlo hhi x).le
  · exact max_le (Finset.le_sup' (fun k => lineEval (-liftKey k) x) (S.min'_mem hS))
      (Finset.le_sup' (fun k => lineEval (-liftKey k) x) (S.max'_mem hS))

/-- **So two lines are enough, at every length.**  The lower half of a head
that has seen `S` needs a subfamily of at most two lines to answer every query
the whole of `S` would have answered — the container `alm-stress` reports as
`lower 2` at every size and in every arrival order. -/
theorem lower_two_lines_suffice {S : Finset ℝ} (hS : S.Nonempty) :
    ∃ T ⊆ S, T.card ≤ 2 ∧ ∃ hT : T.Nonempty, ∀ x : ℝ,
      S.sup' hS (fun k => lineEval (-liftKey k) x)
        = T.sup' hT (fun k => lineEval (-liftKey k) x) := by
  classical
  refine ⟨{S.min' hS, S.max' hS}, ?_, Finset.card_insert_le _ _ |>.trans (by simp),
    ⟨S.min' hS, Finset.mem_insert_self _ _⟩, fun x => ?_⟩
  · intro k hk
    simp only [Finset.mem_insert, Finset.mem_singleton] at hk
    rcases hk with rfl | rfl
    exacts [S.min'_mem hS, S.max'_mem hS]
  · rw [lower_envelope_eq_extremes hS x,
      Finset.sup'_insert (H := Finset.singleton_nonempty _), Finset.sup'_singleton]

/-! ### The hypotheses are satisfiable -/

/-- Three keys in order, the middle one dominated and erased: `0 < 1 < 2`, with
the container holding all three negated lifted keys. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧
    -liftKey (1 : ℝ) ∈ ({-liftKey 0, -liftKey 1, -liftKey 2} : Finset (ℝ × ℝ)) ∧
    -liftKey (0 : ℝ) ∈ ({-liftKey 0, -liftKey 1, -liftKey 2} : Finset (ℝ × ℝ)).erase
      (-liftKey 1) ∧
    -liftKey (2 : ℝ) ∈ ({-liftKey 0, -liftKey 1, -liftKey 2} : Finset (ℝ × ℝ)).erase
      (-liftKey 1) := by
  refine ⟨by norm_num, by norm_num, by simp, ?_, ?_⟩ <;>
    · rw [Finset.mem_erase]
      refine ⟨fun h => ?_, by simp⟩
      have := congrArg (fun p => (p : ℝ × ℝ).1) h
      norm_num [liftKey] at this

/-- And a nonempty key set for the envelope: the three keys above. -/
example : ({0, 1, 2} : Finset ℝ).Nonempty := ⟨0, by simp⟩

end ALM
end Transformer
