module

public import Transformer.Optimization.PAGE.SummableSteps

/-! A contraction estimate for PAGE with a quadratic penalty about a reference
point. A refresh applies a contracting full-gradient map. Between refreshes,
the size of an update contracts by the same factor. The distance to the fixed
point plus half the previous displacement pays for the entire path length. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Filter Metric Set Finset
open scoped Topology

variable {E : Type*} [NormedAddCommGroup E] [CompleteSpace E]

/-- A statement about the actual weights alone. The two hypotheses on
successive updates are supplied by the sampled PAGE recurrence. -/
theorem converges_of_refresh_or_correction (w : ℕ → E) (c z : E) {R : ℝ} (hR : 0 < R)
    (hz : dist z c ≤ R/4) (h₀ : w 0 ∈ closedBall c R)
    (h₁ : dist (w 1) z + dist (w 1) (w 0)/2 ≤ R/2)
    (hstep : ∀ t, w t ∈ closedBall c R → w (t+1) ∈ closedBall c R →
      dist (w (t+2)) z ≤ (1/8) * dist (w (t+1)) z ∨
      dist (w (t+2)) (w (t+1)) ≤ (1/8) * dist (w (t+1)) (w t)) :
    (∀ t, w t ∈ closedBall c R) ∧ ∃ star ∈ closedBall c R, Tendsto w atTop (𝓝 star) := by
  let d (t : ℕ) := dist (w t) z
  let v (t : ℕ) := dist (w (t+1)) (w t)
  let energy (t : ℕ) := d (t+1) + v t / 2
  have nonneg (t : ℕ) : 0 ≤ energy t := by dsimp [energy,d,v]; positivity
  have dec (t : ℕ) (ht : w t ∈ closedBall c R) (hn : w (t+1) ∈ closedBall c R) :
      energy (t+1) + v t/4 ≤ energy t := by
    rcases hstep t ht hn with h | h
    · have tri := dist_triangle_right (w (t+2)) (w (t+1)) z
      dsimp [energy,d,v]
      linarith [dist_nonneg (x := w (t+1)) (y := w t), dist_nonneg (x := w (t+1)) (y := z)]
    · have tri := dist_triangle (w (t+2)) (w (t+1)) z
      dsimp [energy,d,v]
      linarith [dist_nonneg (x := w (t+1)) (y := w t)]
  have bounds (t : ℕ) : w t ∈ closedBall c R ∧ energy t ≤ R/2 := by
    induction t with
    | zero => exact ⟨h₀,h₁⟩
    | succ t ih =>
      have hn : w (t+1) ∈ closedBall c R := by
        have h := ih.2
        dsimp [energy,d,v] at h
        exact (by linarith [dist_triangle (w (t+1)) z c, dist_nonneg (x := w (t+1)) (y := w t)] :
          dist (w (t+1)) c ≤ R)
      exact ⟨hn,by linarith [dec t ih.1 hn, ih.2, show 0 ≤ v t from dist_nonneg]⟩
  have length (n : ℕ) : (∑ t ∈ range n, v t)/4 + energy n ≤ energy 0 := by
    induction n with
    | zero => simp
    | succ n ih =>
      rw [sum_range_succ]
      linarith [dec n (bounds n).1 (bounds (n+1)).1]
  have hs := summable_of_sum_range_le (f := v) (c := 4*energy 0) (fun _ => dist_nonneg)
    (fun n => by linarith [length n, nonneg n])
  obtain ⟨star,ht⟩ := cauchySeq_tendsto_of_complete (cauchySeq_of_dist_le_of_summable v
    (fun i => (dist_comm _ _).le) hs)
  exact ⟨fun t => (bounds t).1,star,isClosed_closedBall.mem_of_tendsto ht
    (Eventually.of_forall fun t => (bounds t).1),ht⟩

example := converges_of_refresh_or_correction (fun _ : ℕ => (0 : ℝ)) 0 0 (R := 1)
  (by norm_num) (by norm_num) (by norm_num [Metric.mem_closedBall]) (by norm_num) (by simp)

end Transformer.Optimization.PAGE
