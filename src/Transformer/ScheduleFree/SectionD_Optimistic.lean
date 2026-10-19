module

public import Transformer.ScheduleFree.SectionE_Regret

/-!
# The concrete optimistic algorithm

arXiv:2405.15682v4, Appendix D.1, Lemma 1. We follow the step size in
the displayed recursion; the paragraph's preceding, different formula
for `η_t` is not used. Real division supplies the zero-denominator
convention, so the all-zero oracle leaves every iterate zero.

The printed lemma omits `‖x_star‖≤D`, although its proof explicitly uses
that inequality. `optimistic_comparator_counterexample` refutes the
unrestricted statement. The corrected bounded-comparator lemma is
stated separately; its proof remains an explicit task, not a premise
silently assumed by the accelerated conversion theorem.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace Transformer.ScheduleFree

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Appendix D.1: the two iterates and the past oracle statistics. -/
structure OptimisticState (E : Type*) where
  z : E
  auxiliary : E
  previous : E
  peak : ℝ
  energy : ℝ

/-- Appendix D.1: clipped optimistic mirror-descent step, including both projections. -/
def optimisticStep (D : ℝ) (s : OptimisticState E) (g : E) : OptimisticState E :=
  let δ := g - s.previous
  let clipped := s.previous + (min s.peak ‖δ‖ / ‖δ‖) • δ
  let peak := max s.peak ‖δ‖
  let energy := s.energy + ‖clipped - s.previous‖ ^ 2
  let η := D / Real.sqrt (peak ^ 2 + energy)
  let auxiliary := projectBall D (s.auxiliary - η • clipped)
  ⟨projectBall D (auxiliary - η • g), auxiliary, g, peak, energy⟩

/-- Appendix D.1: all buffers start at zero. -/
def optimisticRun (D : ℝ) (g : ℕ → E) : ℕ → OptimisticState E
  | 0 => ⟨0, 0, 0, 0, 0⟩
  | t + 1 => optimisticStep D (optimisticRun D g t) (g t)

/-- `g_0=0` in the paper's one-based notation. -/
def previousOracle (g : ℕ → E) : ℕ → E
  | 0 => 0
  | t + 1 => g t

/-- Counterexample to Lemma 1 without the comparator-radius condition.
At `T=1`, `D=1`, `g_1=1`, `z_1=0`, and `x_star=-11`, regret is 11,
larger than `7√2`; all algorithm recursions are the actual ones above. -/
theorem optimistic_comparator_counterexample :
    7 * (1 : ℝ) * Real.sqrt (2 * (∑ t ∈ Finset.range 1,
      ‖(1 : ℝ) - previousOracle (fun _ => (1 : ℝ)) t‖ ^ 2)) <
        regret (fun t => (optimisticRun 1 (fun _ => (1 : ℝ)) t).z) (fun _ => 1) (-11) 1 := by
  norm_num [regret, optimisticRun, previousOracle, Finset.sum_range_succ]
  nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2), Real.sqrt_nonneg 2]

/-- Appendix D.1, Lemma 1, corrected by the missing comparator condition
`‖u‖≤D` used in its proof. No proved declaration depends on this unfinished lemma. -/
theorem optimistic_regret {D : ℝ} (hD : 0 ≤ D) (g : ℕ → E) (u : E) (hu : ‖u‖ ≤ D) (T : ℕ) :
    regret (fun t => (optimisticRun D g t).z) g u T ≤
      7 * D * Real.sqrt (2 * ∑ t ∈ Finset.range T, ‖g t - previousOracle g t‖ ^ 2) := by
  sorry

example : 0 ≤ (1 : ℝ) ∧ ‖(0 : ℝ)‖ ≤ 1 := by norm_num

end Transformer.ScheduleFree
