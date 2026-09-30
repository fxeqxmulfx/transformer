/-
# DASH — the finite numerical candidate on a quadratic cycle

arXiv:2602.02016v2, §2.2–2.3 and §3.3–3.5. The counterexample uses
preconditioner EMA `β=1/2`, positive regularizer `ε=1/4`, no gradient
momentum and current squared gradients for the Adam reference (`ν=0`).
Both roots use one actual NDB step per call and nonzero PI starts.
-/

import Transformer.DASH.Section2_TrainingScalarRoot

open scoped Matrix

noncomputable section

namespace Transformer.DASH

open Optimization

/-- Fixed nonzero PI starts for the scalar training example,
arXiv:2602.02016v2, §3.4–3.5. -/
def cycleStarts : Fin 1 → Fin 1 → ℝ := fun _ _ => 1

/-- A scalar block with squared gradient `1/16` is grafted to twice its
current gradient, despite retaining both actual finite NDB calls and
the current histories. The left/right nonnegativity invariant is also
preserved. Source: arXiv:2602.02016v2, §2–3, training counterexample. -/
theorem dash_cycle_candidate (state : TrainingState 1 1) (s : ℝ)
    (hleft : 0 ≤ state.left 0 0) (hright : 0 ≤ state.right 0 0) (hs : s ^ 2 = 1 / 16) :
    (dashTrainingCandidate (1 / 2) 0 0 (1 / 4) cycleStarts cycleStarts 0 1 state
      (fromMatrix (s • (1 : Matrix (Fin 1) (Fin 1) ℝ)))).2 =
        fromMatrix ((2 : ℝ) • (s • (1 : Matrix (Fin 1) (Fin 1) ℝ))) ∧
    0 ≤ (dashTrainingCandidate (1 / 2) 0 0 (1 / 4) cycleStarts cycleStarts 0 1 state
      (fromMatrix (s • (1 : Matrix (Fin 1) (Fin 1) ℝ)))).1.left 0 0 ∧
    0 ≤ (dashTrainingCandidate (1 / 2) 0 0 (1 / 4) cycleStarts cycleStarts 0 1 state
      (fromMatrix (s • (1 : Matrix (Fin 1) (Fin 1) ℝ)))).1.right 0 0 := by
  let G := s • (1 : Matrix (Fin 1) (Fin 1) ℝ)
  let A : Matrix (Fin 1) (Fin 1) ℝ := fun i j => G i j ^ 2
  let l := 1 / 2 * state.left 0 0 + 1 / 2 * s ^ 2 + 1 / 4
  let r := 1 / 2 * state.right 0 0 + 1 / 2 * s ^ 2 + 1 / 4
  have hl : 0 < l := by dsimp only [l]; nlinarith [sq_nonneg s]
  have hr : 0 < r := by dsimp only [r]; nlinarith [sq_nonneg s]
  have hL : leftEma (1 / 2) state.left G + (1 / 4 : ℝ) • 1 = l • 1 := by
    ext i j
    fin_cases i
    fin_cases j
    simp [leftEma, G, l]
    ring
  have hR : rightEma (1 / 2) state.right G + (1 / 4 : ℝ) • 1 = r • 1 := by
    ext i j
    fin_cases i
    fin_cases j
    simp [rightEma, G, r]
    ring
  have hreference : adamGraftingDirection (1 / 4) A G = (2 : ℝ) • G := by
    ext i j
    fin_cases i
    fin_cases j
    norm_num [adamGraftingDirection, A, G, hs]
    ring
  have hG : 0 < frobeniusNorm G := by
    have hsne : s ≠ 0 := by intro h; rw [h] at hs; norm_num at hs
    dsimp only [G]
    rw [frobeniusNorm_smul]
    have hunit : frobeniusNorm (1 : Matrix (Fin 1) (Fin 1) ℝ) = 1 := by
      norm_num [frobeniusNorm, Muon.squaredFrobenius]
    rw [hunit, mul_one]
    exact abs_pos.mpr hsne
  have hU : preconditionedGradient
      (guardedInverseFourth (l • (1 : Matrix (Fin 1) (Fin 1) ℝ))
        (pooledRayleigh (l • 1) cycleStarts 0) 1) G
      (guardedInverseFourth (r • (1 : Matrix (Fin 1) (Fin 1) ℝ))
        (pooledRayleigh (r • 1) cycleStarts 0) 1) =
      (finiteScalarRoot l * finiteScalarRoot r) • G := by
    have hpoolL : pooledRayleigh (l • (1 : Matrix (Fin 1) (Fin 1) ℝ)) cycleStarts 0 = l :=
      scalar_unit_pool l
    have hpoolR : pooledRayleigh (r • (1 : Matrix (Fin 1) (Fin 1) ℝ)) cycleStarts 0 = r :=
      scalar_unit_pool r
    rw [hpoolL, hpoolR, guardedInverseFourth_scalar l hl, guardedInverseFourth_scalar r hr]
    simp [preconditionedGradient, smul_smul, mul_comm]
  refine ⟨?_, ?_, ?_⟩
  · change fromMatrix (graft (preconditionedGradient
        (guardedInverseFourth (leftEma (1 / 2) state.left G + (1 / 4 : ℝ) • 1)
          (pooledRayleigh (leftEma (1 / 2) state.left G + (1 / 4 : ℝ) • 1)
            cycleStarts 0) 1) G
        (guardedInverseFourth (rightEma (1 / 2) state.right G + (1 / 4 : ℝ) • 1)
          (pooledRayleigh (rightEma (1 / 2) state.right G + (1 / 4 : ℝ) • 1)
            cycleStarts 0) 1)) (adamGraftingDirection (1 / 4)
              (0 • state.accumulator + (1 - 0) • A)
              (0 • state.momentum + (1 - 0) • G))) = _
    simp only [zero_smul, zero_add, sub_zero, one_smul, hL, hR, hU, hreference]
    rw [graft_positive_smul _ 2 G
      (mul_pos (finiteScalarRoot_pos l hl) (finiteScalarRoot_pos r hr)) (by norm_num) hG]
  · change 0 ≤ (leftEma (1 / 2) state.left G) 0 0
    simp [leftEma, G]
    nlinarith [sq_nonneg s]
  · change 0 ≤ (rightEma (1 / 2) state.right G) 0 0
    simp [rightEma, G]
    nlinarith [sq_nonneg s]

/-- The candidate's hypotheses hold for zero-initialized source states
and a nonzero scalar gradient. Source: arXiv:2602.02016v2, §2–3,
training counterexample. -/
example : 0 ≤ (initialTrainingState : TrainingState 1 1).left 0 0 ∧
    0 ≤ (initialTrainingState : TrainingState 1 1).right 0 0 ∧ (1 / 4 : ℝ) ^ 2 = 1 / 16 := by
  norm_num [initialTrainingState]

end Transformer.DASH
