/-
# Why the always-rejecting safeguards behave as SGD

Source implementation: benchmark common.py, DirectionOptimizer.step.
This extension permits a different differentiable minibatch loss at every
update and arbitrary stateful hybrid proposals. It is a trajectory identity,
not a stochastic convergence theorem. The finite rejection hypothesis must
be verified on the actual run; it is not asserted for every candidate.
The model uses real arithmetic and actual mathematical derivatives; it
does not certify floating-point guard decisions or PyTorch autograd.
The free learning rate and finite horizon match the experiment's controls.
State updates continue even when their proposed direction is discarded.

Algorithm references: training correction to arXiv:2502.16982, Section 2.2,
and arXiv:2602.02016v2, Sections 2-4.
-/

import Transformer.Optimization.Basic

open scoped InnerProductSpace

noncomputable section

namespace Transformer.OptimizerBenchmark

open Transformer.Optimization

variable {E S : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- SGD on an indexed sequence of actual batch objectives with fixed rate.
Source: benchmark common.py, step; training extension to arXiv:2502.16982,
Section 2.2, and arXiv:2602.02016v2, Section 2, Algorithm 1. -/
def batchSGD (rate : ℝ) (loss : ℕ → E → ℝ) (initial : E) : ℕ → E
  | 0 => initial
  | t + 1 =>
      let x := batchSGD rate loss initial t
      x - rate • gradient (loss t) x

/-- The literal benchmark safeguard with state updates and minibatch-indexed
losses. A free rate matches the executable; imposing rate=sigma/L belongs
to separate convergence results. Source: benchmark common.py, step;
training correction to arXiv:2502.16982, Section 2.2, and
arXiv:2602.02016v2, Sections 2-4. -/
def guardedBatchRun (σ rate : ℝ) (loss : ℕ → E → ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial : E) : ℕ → S × E
  | 0 => (initialState, initial)
  | t + 1 =>
      let state := guardedBatchRun σ rate loss propose initialState initial t
      let g := gradient (loss t) state.2
      let candidate := propose state.1 state.2 g
      (candidate.1, state.2 - rate • descentGuard σ g candidate.2)

/-- Rejection at each actual state up to a finite horizon forces equality
with SGD on the same batch sequence and initialization. Proposal states may
continue to change and may contain Adam, Muon or DASH buffers. Source:
REPORT.md, Implementation and proof scope; common.py, step; correction to
arXiv:2502.16982, Section 2.2, and arXiv:2602.02016v2, Sections 2-4. -/
theorem guardedBatchRun_eq_sgd (σ rate : ℝ) (loss : ℕ → E → ℝ)
    (propose : S → E → E → S × E) (initialState : S) (initial : E) (T : ℕ)
    (hreject : ∀ t < T,
      let state := guardedBatchRun σ rate loss propose initialState initial t
      let g := gradient (loss t) state.2
      ¬ (σ * ‖g‖ ^ 2 ≤ ⟪g, (propose state.1 state.2 g).2⟫_ℝ ∧
        ‖(propose state.1 state.2 g).2‖ ≤ ‖g‖)) :
    ∀ t ≤ T, (guardedBatchRun σ rate loss propose initialState initial t).2 =
      batchSGD rate loss initial t := by
  intro t
  induction t with
  | zero => intro ht; rfl
  | succ t ih =>
      intro ht
      have hprev := ih (by omega)
      have hbad := hreject t (by omega)
      dsimp only at hbad
      simp only [guardedBatchRun, batchSGD]
      rw [descentGuard_rejects σ _ _ hbad, hprev]

/-- A genuine nonconstant quadratic at x=1 rejects an uphill proposal;
the one-step rejection hypothesis is nonempty. The optimizer state is
incremented even when its proposed direction is discarded. Source:
benchmark common.py, step; training correction to arXiv:2502.16982,
Section 2.2, and arXiv:2602.02016v2, Sections 2-4. -/
example : ∀ t < 1,
    let state := guardedBatchRun (1 / 2) (1 / 10) (fun _ => quadratic)
      (fun (s : ℕ) (x g : ℝ) => (s + 1, -g - x)) 0 1 t
    let g := gradient quadratic state.2
    ¬ ((1 / 2) * ‖g‖ ^ 2 ≤ ⟪g, -g - state.2⟫_ℝ ∧
      ‖-g - state.2‖ ≤ ‖g‖) := by
  intro t ht
  have hz : t = 0 := by omega
  subst t
  norm_num [guardedBatchRun, quadratic_gradient]

end Transformer.OptimizerBenchmark
