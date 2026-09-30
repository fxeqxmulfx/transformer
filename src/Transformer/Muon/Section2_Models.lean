/-
# Muon is Scalable for LLM Training — the optimizer

Liu et al., arXiv:2502.16982, §2.1–2.2. The iteration is over real matrices;
bf16 rounding and the measured training performance are separate empirical
questions. In particular, a finite Newton–Schulz iterate is not identified
with an exact polar factor.
-/

import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.Muon

variable {a b : ℕ}

/-- Gradient momentum, arXiv:2502.16982, §2.1, the first line of `eq:Ot`. -/
def momentumStep (μ : ℝ) (M G : Matrix (Fin a) (Fin b) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  μ • M + G

/-- The Nesterov input in the footnote to `eq:Ot`, arXiv:2502.16982, §2.1.
`M` is the previous momentum, so the new momentum is calculated first. -/
def nesterovInput (μ : ℝ) (M G : Matrix (Fin a) (Fin b) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  μ • momentumStep μ M G + G

/-- The singular-value polynomial of `eq:iteration`, arXiv:2502.16982, §2.1. -/
def schulzPolynomial (α β γ x : ℝ) : ℝ := α * x + β * x ^ 3 + γ * x ^ 5

/-- One Newton–Schulz step, arXiv:2502.16982, §2.1, `eq:iteration`. -/
def schulzStep (α β γ : ℝ) (X : Matrix (Fin a) (Fin b) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  α • X + β • ((X * X.transpose) * X) + γ • ((X * X.transpose) ^ 2 * X)

/-- Squared Frobenius norm, as used in arXiv:2502.16982, §2.1 and Appendix A. -/
def squaredFrobenius (X : Matrix (Fin a) (Fin b) ℝ) : ℝ :=
  ∑ i, ∑ j, X i j ^ 2

/-- Entrywise RMS from Appendix A of arXiv:2502.16982.
Nonempty matrix dimensions are explicit hypotheses whenever cancellation
of the denominator is needed. -/
def matrixRMS (X : Matrix (Fin a) (Fin b) ℝ) : ℝ :=
  Real.sqrt (squaredFrobenius X / ((a : ℝ) * b))

/-- Initial normalization, arXiv:2502.16982, §2.1, before `eq:iteration`.
The total real division also specifies the zero-input case. -/
def schulzInitial (M : Matrix (Fin a) (Fin b) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  (Real.sqrt (squaredFrobenius M))⁻¹ • M

/-- Finite Newton–Schulz iteration, arXiv:2502.16982, §2.1, `eq:iteration`. -/
def schulzIterate (α β γ : ℝ) (M : Matrix (Fin a) (Fin b) ℝ) : ℕ → Matrix (Fin a) (Fin b) ℝ
  | 0 => schulzInitial M
  | k + 1 => schulzStep α β γ (schulzIterate α β γ M k)

/-- The five-step implementation specified in arXiv:2502.16982, §2.1–2.2. -/
def approximatePolar (M : Matrix (Fin a) (Fin b) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  schulzIterate (34445 / 10000) (-47750 / 10000) (20315 / 10000) M 5

/-- Decoupled weight decay, arXiv:2502.16982, §2.2, `equation:weightdecay`. -/
def decayStep (η wd : ℝ) (W O : Matrix (Fin a) (Fin b) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  W - η • (O + wd • W)

/-- Shape-dependent scaling, arXiv:2502.16982, §2.2, “Consistent update RMS”. -/
def shapeScale (a b : ℕ) : ℝ := Real.sqrt (max (a : ℝ) b)

/-- The adjusted update, arXiv:2502.16982, §2.2, “Matching update RMS of AdamW”. -/
def adjustedUpdate (O : Matrix (Fin a) (Fin b) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  ((1 / 5 : ℝ) * shapeScale a b) • O

/-- Direct RMS normalization, arXiv:2502.16982, §3.1, “Update Norm”. -/
def normalizedUpdate (O : Matrix (Fin a) (Fin b) ℝ) : Matrix (Fin a) (Fin b) ℝ :=
  ((1 / 5 : ℝ) / matrixRMS O) • O

/-- The complete practical Muon step including Nesterov momentum and weight
decay, arXiv:2502.16982, §2.1–2.2. The returned pair is `(new momentum, new weight)`. -/
def muonStep (μ η wd : ℝ) (M W G : Matrix (Fin a) (Fin b) ℝ) :
    Matrix (Fin a) (Fin b) ℝ × Matrix (Fin a) (Fin b) ℝ :=
  (momentumStep μ M G, decayStep η wd W (adjustedUpdate (approximatePolar (nesterovInput μ M G))))

/-- Weight decay is equivalent to shrinking the old weights and then taking
the update, arXiv:2502.16982, §2.2, `equation:weightdecay`. -/
theorem decayStep_eq (η wd : ℝ) (W O : Matrix (Fin a) (Fin b) ℝ) :
    decayStep η wd W O = (1 - η * wd) • W - η • O := by
  ext i j
  simp only [decayStep, Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  ring

/-- The Nesterov footnote expands to `μ² M + (1 + μ) G`, arXiv:2502.16982, §2.1. -/
theorem nesterovInput_eq (μ : ℝ) (M G : Matrix (Fin a) (Fin b) ℝ) :
    nesterovInput μ M G = μ ^ 2 • M + (1 + μ) • G := by
  ext i j
  simp only [nesterovInput, momentumStep, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  ring

/-- The zero momentum remains zero under every finite Newton–Schulz iteration,
arXiv:2502.16982, §2.1. This includes the normalization's zero-input case. -/
theorem schulzIterate_zero (α β γ : ℝ) (k : ℕ) :
    schulzIterate α β γ (0 : Matrix (Fin a) (Fin b) ℝ) k = 0 := by
  induction k with
  | zero => simp [schulzIterate, schulzInitial]
  | succ k ih => simp [schulzIterate, schulzStep, ih]

/-- An arbitrary normalization scale cannot turn a zero update into a nonzero
one, arXiv:2502.16982, §3.1, “Update Norm”. -/
theorem normalizedUpdate_zero :
    normalizedUpdate (0 : Matrix (Fin a) (Fin b) ℝ) = 0 := by
  simp [normalizedUpdate]

end Transformer.Muon
