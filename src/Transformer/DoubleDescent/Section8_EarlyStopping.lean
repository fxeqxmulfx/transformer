import Transformer.DoubleDescent.Section5_Curves
import Lean.Elab.Tactic.Omega

/-!
# Optimal early stopping

arXiv:1912.02292v1, Section 8, paragraph "Early stopping", and Appendix E.2.
The oracle chooses the best *test* error observed through a finite horizon,
as in the paper's plots. A practical validation rule is a different procedure.
This oracle's error cannot increase as the horizon grows. Its minimum need
not remove double descent as a function of model size, as the paper notes.
-/

namespace Transformer.DoubleDescent

/-- Section 8 and Appendix E.2: best error through the given training horizon,
including epoch zero. This is a finite minimum over actual iterates. -/
noncomputable def bestThrough (error : ℕ → ℝ) : ℕ → ℝ
  | 0 => error 0
  | t + 1 => min (bestThrough error t) (error (t + 1))

/-- Section 8 and Appendix E.2: optimal early stopping is no worse than any
of the available checkpoints. -/
theorem bestThrough_le (error : ℕ → ℝ) {t horizon : ℕ} (ht : t ≤ horizon) :
    bestThrough error horizon ≤ error t := by
  induction horizon with
  | zero =>
    have : t = 0 := by omega
    simp [this, bestThrough]
  | succ horizon ih =>
    by_cases he : t = horizon + 1
    · simp [he, bestThrough]
    · exact (min_le_left _ _).trans (ih (by omega))

/-- Section 8: a checkpoint inside the horizon exists. -/
example : (0 : ℕ) ≤ 1 := by decide

/-- Section 8 and Appendix E.2: adding checkpoints cannot worsen the oracle
early-stopped test error. This says nothing about varying model or sample size. -/
theorem bestThrough_antitone (error : ℕ → ℝ) : Antitone (bestThrough error) := by
  intro a b hab
  induction b with
  | zero =>
    have : a = 0 := by omega
    simp [this]
  | succ b ih =>
    by_cases he : a = b + 1
    · simp [he]
    · exact (min_le_left _ _).trans (ih (by omega))

/-- Section 8 and Appendix E.2: the finite minimum is attained by a checkpoint,
so it does not rely on a fictitious error value or an unattained infimum. -/
theorem bestThrough_attained (error : ℕ → ℝ) (horizon : ℕ) :
    ∃ t ≤ horizon, bestThrough error horizon = error t := by
  induction horizon with
  | zero => exact ⟨0, le_rfl, rfl⟩
  | succ horizon ih =>
    rcases ih with ⟨t, ht, he⟩
    by_cases h : bestThrough error horizon ≤ error (horizon + 1)
    · exact ⟨t, by omega, by rw [bestThrough, min_eq_left h, he]⟩
    · exact ⟨horizon + 1, le_rfl, by rw [bestThrough, min_eq_right (le_of_not_ge h)]⟩

/-- Section 8: strictly increasing checkpoint errors are minimized at epoch
zero, even when their baseline depends non-monotonically on model size. -/
theorem bestThrough_increasing_offset (baseline : ℝ) (horizon : ℕ) :
    bestThrough (fun t => baseline + (t : ℝ)) horizon = baseline := by
  induction horizon with
  | zero => simp [bestThrough]
  | succ horizon ih =>
    rw [bestThrough, ih, min_eq_left]
    exact le_add_of_nonneg_right (Nat.cast_nonneg _)

/-- Section 8: the source explicitly allows model-wise double descent even
with optimal early stopping. This mathematical witness refutes an unrestricted
claim that taking the best checkpoint always removes it; it is not a CNN log. -/
theorem earlyStopping_can_preserve_modelDoubleDescent :
    ∃ error : ℕ → ℕ → ℝ,
      HasDoubleDescent (fun width => bestThrough (error width) 10) ∧
      (∀ width, error width 0 < error width 1) := by
  let baseline : ℕ → ℝ := fun n =>
    if n = 0 then 3 else if n = 1 then 1 else if n = 2 then 4 else 0
  refine ⟨fun width t => baseline width + (t : ℝ), ?_, ?_⟩
  · simp_rw [bestThrough_increasing_offset]
    refine ⟨0, 1, 2, 3, ?_⟩
    norm_num [baseline]
  · intro width
    norm_num

end Transformer.DoubleDescent
