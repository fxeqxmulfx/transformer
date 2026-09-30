/-
# DASH — temporal preconditioners and regularized Shampoo updates

arXiv:2602.02016v2, §2, `algorithm:default-shampoo` and “Grafting”.
The optimizer's inverse roots are exact EVD powers over the reals.
-/

import Transformer.DASH.Section3_EVDExistence

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {m n : ℕ}

/-- Left EMA over the gradient history, initialized at zero,
with `G t` denoting the manuscript's `G_(t+1)` so that the first update
uses `G 0`. Source: arXiv:2602.02016v2, §2, Algorithm 1. -/
def leftHistory (β : ℝ) (G : ℕ → Matrix (Fin m) (Fin n) ℝ) :
    ℕ → Matrix (Fin m) (Fin m) ℝ
  | 0 => 0
  | t + 1 => leftEma β (leftHistory β G t) (G t)

/-- Right EMA over the same gradient history,
with `G t` denoting the manuscript's `G_(t+1)`.
Source: arXiv:2602.02016v2, §2, Algorithm 1. -/
def rightHistory (β : ℝ) (G : ℕ → Matrix (Fin m) (Fin n) ℝ) :
    ℕ → Matrix (Fin n) (Fin n) ℝ
  | 0 => 0
  | t + 1 => rightEma β (rightHistory β G t) (G t)

/-- Both entire preconditioner histories are positive semidefinite.
Source: arXiv:2602.02016v2, §2, Algorithm 1. -/
theorem history_posSemidef (β : ℝ) (G : ℕ → Matrix (Fin m) (Fin n) ℝ)
    (hβ : 0 ≤ β) (hβ' : β ≤ 1) (t : ℕ) :
    (leftHistory β G t).PosSemidef ∧ (rightHistory β G t).PosSemidef := by
  induction t with
  | zero => exact ⟨Matrix.PosSemidef.zero, Matrix.PosSemidef.zero⟩
  | succ t ih => exact ⟨leftEma_posSemidef β _ _ hβ hβ' ih.1,
      rightEma_posSemidef β _ _ hβ hβ' ih.2⟩

/-- Valid EMA parameters exist, arXiv:2602.02016v2, §2. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Adding the source's positive `εI` makes any PSD preconditioner positive
definite, so that the inverse roots have a positive spectral domain.
Source: arXiv:2602.02016v2, §2, Algorithm 1's regularization term. -/
theorem regularized_posDef (A : Matrix (Fin n) (Fin n) ℝ)
    (ε : ℝ) (hA : A.PosSemidef) (hε : 0 < ε) : (A + ε • 1).PosDef := by
  have hdiag : (Matrix.diagonal (fun _ : Fin n => ε)).PosDef :=
    Matrix.PosDef.diagonal (fun _ => hε)
  have heq : ε • (1 : Matrix (Fin n) (Fin n) ℝ) = Matrix.diagonal (fun _ => ε) := by
    ext i j
    by_cases h : i = j <;> simp [Matrix.diagonal, h]
  rw [heq]
  exact Matrix.PosDef.posSemidef_add hA hdiag

/-- Regularization hypotheses are satisfiable, arXiv:2602.02016v2, §2. -/
example : (0 : Matrix (Fin 1) (Fin 1) ℝ).PosSemidef ∧ (0 : ℝ) < 1 :=
  ⟨Matrix.PosSemidef.zero, by norm_num⟩

/-- Exact EVD inverse root on a positive-definite input,
arXiv:2602.02016v2, §2 and §3.1. -/
def evdInverseRoot (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.PosDef) (p : ℕ) :
    Matrix (Fin n) (Fin n) ℝ :=
  spectralPower (evdFrame A hA.1) hA.1.eigenvalues (-(1 / (p : ℝ)))

/-- The full regularized parameter step in Algorithm 1,
arXiv:2602.02016v2, §2, `θ_(t+1)=θ_t-η(L_t+εI)^(-1/4)G_t(R_t+εI)^(-1/4)`. -/
def shampooStep (η ε : ℝ) (θ G : Matrix (Fin m) (Fin n) ℝ)
    (L : Matrix (Fin m) (Fin m) ℝ) (R : Matrix (Fin n) (Fin n) ℝ)
    (hL : L.PosSemidef) (hR : R.PosSemidef) (hε : 0 < ε) : Matrix (Fin m) (Fin n) ℝ :=
  θ - η • preconditionedGradient
    (evdInverseRoot (L + ε • 1) (regularized_posDef L ε hL hε) 4) G
    (evdInverseRoot (R + ε • 1) (regularized_posDef R ε hR hε) 4)

/-- A zero gradient leaves the parameter unchanged, regardless of the
nonzero accumulated preconditioners, arXiv:2602.02016v2, §2, Algorithm 1. -/
theorem shampooStep_zero (η ε : ℝ) (θ : Matrix (Fin m) (Fin n) ℝ)
    (L : Matrix (Fin m) (Fin m) ℝ) (R : Matrix (Fin n) (Fin n) ℝ)
    (hL : L.PosSemidef) (hR : R.PosSemidef) (hε : 0 < ε) :
    shampooStep η ε θ 0 L R hL hR hε = θ := by
  simp [shampooStep, preconditionedGradient]

/-- Zero-step hypotheses are satisfiable, arXiv:2602.02016v2, §2. -/
example : (0 : Matrix (Fin 1) (Fin 1) ℝ).PosSemidef ∧
    (0 : Matrix (Fin 1) (Fin 1) ℝ).PosSemidef ∧ (0 : ℝ) < 1 :=
  ⟨Matrix.PosSemidef.zero, Matrix.PosSemidef.zero, by norm_num⟩

/-- The elementwise grafting direction supplied in the source's paragraph;
`M=G` without EMA, and `M` is the gradient momentum when EMA is enabled.
Source: arXiv:2602.02016v2, §2, “Grafting”, `P_t=G_t/(ε+√A_t)`. -/
def adamGraftingDirection (ε : ℝ) (A M : Matrix (Fin m) (Fin n) ℝ) :
    Matrix (Fin m) (Fin n) ℝ := fun i j => M i j / (ε + Real.sqrt (A i j))

/-- Positive `ε` keeps every elementwise denominator strictly positive,
arXiv:2602.02016v2, §2, the Adam grafting expression. -/
theorem grafting_denominator_pos (ε a : ℝ) (hε : 0 < ε) : 0 < ε + Real.sqrt a := by
  linarith [Real.sqrt_nonneg a]

/-- Grafting-denominator hypotheses are satisfiable, arXiv:2602.02016v2, §2. -/
example : (0 : ℝ) < 1 := by norm_num

end Transformer.DASH
