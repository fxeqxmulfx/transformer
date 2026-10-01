/-
# IC-EoT: selective input expansion and the actual MPC rollout

arXiv:2603.22095v2, §4.1, Eqs. (31)–(39). The history is represented
by its predicted-variable and extended-control columns. This is a coordinate
reindexing of the source's concatenated matrix, not a change in the dynamics.
Only controls are duplicated with a negative sign. Every predicted output,
including physically exogenous variables, is substituted recursively.
-/

import Transformer.ICEoT.Section3_Encoder

noncomputable section

namespace Transformer.ICEoT

/-- Model-input columns partitioned as in Eqs. (33), (34), §4.1. -/
abbrev History (n y u : ℕ) :=
  Sequence (n + 1) y × ((Fin (n + 1) × (Bool × Fin u)) → ℝ)

/-- The finite control sequence of Eqs. (31), (32), §4.1. -/
abbrev Controls (np u : ℕ) := Sequence np u

/-- Monotonicity only in the recursively substituted columns, exactly the
hypothesis of §4.1, Corollary 2. Extended controls are held fixed. -/
def PredictedMonotone {P C O : Type*}
    (f : ((P → ℝ) × (C → ℝ)) → O → ℝ) : Prop :=
  ∀ p q c, p ≤ q → f (p, c) ≤ f (q, c)

/-- Convexity and the restricted monotonicity assumed in §4.1, Corollary 2. -/
def PredictorConditions {n y u : ℕ} (f : History n y u → Fin y → ℝ) : Prop :=
  ConvexOn ℝ Set.univ f ∧ PredictedMonotone f

/-- An affine function's combination identity, used for control columns;
§4.1, Eq. (33) and the proof of Corollary 2. -/
def AffineCombinations {E V : Type*} [AddCommGroup E] [Module ℝ E]
    [AddCommGroup V] [Module ℝ V] (f : E → V) : Prop :=
  ∀ x y (a b : ℝ), a + b = 1 → f (a • x + b • y) = a • f x + b • f y

/-- Drop the oldest row and append the current row; §4.1, Eq. (35). -/
def shiftHistory {n : ℕ} {J : Type*} (H : (Fin (n + 1) × J) → ℝ)
    (row : J → ℝ) : (Fin (n + 1) × J) → ℝ :=
  fun ir => if hi : ir.1.val < n then H (⟨ir.1.val + 1, by omega⟩, ir.2) else row ir.2

/-- The control lift `[u,-u]` in Eq. (33), §4.1. -/
def extendControl {u : ℕ} (x : Fin u → ℝ) : (Bool × Fin u) → ℝ :=
  fun r => if r.1 then -x r.2 else x r.2

/-- The control at a finite-horizon stage, Eqs. (31), (32), §4.1.
The zero extension beyond the horizon only makes the recursion total;
the MPC problem uses stages `k < np` exclusively. -/
def stageControl {np u : ℕ} (U : Controls np u) (k : ℕ) : Fin u → ℝ :=
  fun r => if hk : k < np then U (⟨k, hk⟩, r) else 0

/-- The histories and current model output in Eqs. (34)–(36), §4.1. -/
structure PredictionState (n y u : ℕ) where
  predictedHistory : Sequence (n + 1) y
  controlHistory : (Fin (n + 1) × (Bool × Fin u)) → ℝ
  prediction : Fin y → ℝ

/-- The actual recursively substituted predictor, Eqs. (34)–(36), §4.1.
The initial measured history and current measurement are fixed parameters. -/
def mpcRun {np n y u : ℕ} (f : History n y u → Fin y → ℝ)
    (H0 : History n y u) (y0 : Fin y → ℝ) (U : Controls np u) :
    ℕ → PredictionState n y u
  | 0 => ⟨H0.1, H0.2, y0⟩
  | k + 1 =>
    let old := mpcRun f H0 y0 U k
    let ph := shiftHistory old.predictedHistory old.prediction
    let ch := shiftHistory old.controlHistory (extendControl (stageControl U k))
    ⟨ph, ch, f (ph, ch)⟩

/-- A concrete affine predictor using both state and control, to witness
the hypotheses of §4.1, Corollary 2. -/
def witnessPredictor (H : History 0 1 1) : Fin 1 → ℝ :=
  fun _ => H.1 (0, 0) + H.2 (0, (false, 0))

/-- The rollout assumptions are satisfiable; §4.1, Corollary 2. -/
theorem witnessPredictor_conditions : PredictorConditions witnessPredictor := by
  constructor
  · refine ⟨convex_univ, ?_⟩
    intro x hx y hy a b ha hb hab r
    simp only [witnessPredictor, Prod.smul_fst, Prod.smul_snd, Prod.fst_add,
      Prod.snd_add, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    exact le_of_eq (by ring)
  · intro p q c h r
    exact add_le_add (h (0, 0)) le_rfl

end Transformer.ICEoT
