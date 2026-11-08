module

public import Transformer.Grokking.GradientEvidence.ReadoutLeakage

/-!
# Residual dynamics under actual readout descent and changing subspaces

Source: `review/flat_spectrum.md`, *Residual learning dynamics*, RD3–RD4.
Euclidean descent with decay contracts directions absent from all fixed
training features. Outside that case, the objective controls a forcing term.
Changing the learned basis adds a separate term; it is never assumed fixed
when interpreting the saved end-to-end model trajectories.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset FlatSpectrum.FeatureCodes

variable {I C K : Type*} [Fintype I] [Fintype C] [Fintype K]
  [DecidableEq C] [DecidableEq K]

/-- Euclidean descent on the actual fixed-feature CE, with decoupled decay,
RD3. This is not substituted for the saved models' AdamW updates. -/
def readoutStep (T : I → K → ℝ) (y : I → C) (η ρ : ℝ) (E : C → K → ℝ) : C → K → ℝ :=
  fun c k => (1 - η * ρ) * E c k - η * readoutGradient T y E c k

/-- Projecting a step retains both decay and the actual gradient forcing, RD3. -/
theorem projected_readoutStep (T : I → K → ℝ) (y : I → C) (η ρ : ℝ)
    (E : C → K → ℝ) (Q : K → K → ℝ) (c : C) (k : K) :
    matrixApply Q (readoutStep T y η ρ E c) k =
      (1 - η * ρ) * matrixApply Q (E c) k - η * matrixApply Q (readoutGradient T y E c) k := by
  exact matrixApply_sub_smul Q (E c) (readoutGradient T y E c) (1 - η * ρ) η k

/-- If Q annihilates every training feature, its actual CE forcing vanishes,
RD3. An answer group's centered span alone does not ensure this hypothesis. -/
theorem projected_gradient_zero (T : I → K → ℝ) (y : I → C)
    (E : C → K → ℝ) (Q : K → K → ℝ)
    (h : ∀ i k, matrixApply Q (T i) k = 0) (c : C) (k : K) :
    matrixApply Q (readoutGradient T y E c) k = 0 := by
  rw [projected_readoutGradient]
  simp only [h, mul_zero, sum_const_zero, zero_div]

/-- The feature-span condition derives pure decay of the residual, RD3. -/
theorem projected_step_of_annihilation (T : I → K → ℝ) (y : I → C)
    (η ρ : ℝ) (E : C → K → ℝ) (Q : K → K → ℝ)
    (h : ∀ i k, matrixApply Q (T i) k = 0) (c : C) (k : K) :
    matrixApply Q (readoutStep T y η ρ E c) k =
      (1 - η * ρ) * matrixApply Q (E c) k := by
  rw [projected_readoutStep, projected_gradient_zero T y E Q h]
  ring

/-- Iterated actual readout-loss gradient steps at fixed training features,
RD3. The gradient is recomputed at every readout. -/
def readoutRun (T : I → K → ℝ) (y : I → C) (η ρ : ℝ) (E : C → K → ℝ) : ℕ → C → K → ℝ
  | 0 => E
  | n + 1 => readoutStep T y η ρ (readoutRun T y η ρ E n)

/-- **Unobserved fixed-feature directions decay geometrically**, RD3.
At zero decay the same formula proves invariance, not contraction. -/
theorem projected_readoutRun (T : I → K → ℝ) (y : I → C)
    (η ρ : ℝ) (E : C → K → ℝ) (Q : K → K → ℝ)
    (h : ∀ i k, matrixApply Q (T i) k = 0) (n : ℕ) (c : C) (k : K) :
    matrixApply Q (readoutRun T y η ρ E n c) k =
      (1 - η * ρ) ^ n * matrixApply Q (E c) k := by
  induction n with
  | zero => simp [readoutRun]
  | succ n ih =>
    change matrixApply Q (readoutStep T y η ρ (readoutRun T y η ρ E n) c) k = _
    rw [projected_step_of_annihilation T y η ρ _ Q h, ih, pow_succ]
    ring

/-- Scalar decay/forcing estimate used coordinatewise, RD3. -/
theorem decay_forcing_sq_le (η ρ u g : ℝ) (hη : 0 ≤ η) (hρ : 0 < ρ) (hs : η * ρ ≤ 1) :
    ((1 - η * ρ) * u - η * g) ^ 2 ≤ (1 - η * ρ) * u ^ 2 + (η / ρ) * g ^ 2 := by
  have hn : 0 ≤ (η / ρ) * (1 - η * ρ) * (ρ * u + g) ^ 2 :=
    mul_nonneg (mul_nonneg (div_nonneg hη hρ.le) (by linarith)) (sq_nonneg _)
  have he : (1 - η * ρ) * u ^ 2 + (η / ρ) * g ^ 2 -
      ((1 - η * ρ) * u - η * g) ^ 2 = (η / ρ) * (1 - η * ρ) * (ρ * u + g) ^ 2 := by
    field_simp
    ring
  linarith

/-- **A loss-controlled forcing bound for residual energy**, RD3.
Small loss bounds the forcing; pure contraction additionally needs decay
and control of the projected training-feature energy. -/
theorem readout_step_energy_le (T : I → K → ℝ) (y : I → C)
    (η ρ : ℝ) (E : C → K → ℝ) (Q : K → K → ℝ)
    (hη : 0 ≤ η) (hρ : 0 < ρ) (hs : η * ρ ≤ 1) :
    matrixEnergy (fun c => matrixApply Q (readoutStep T y η ρ E c)) ≤
      (1 - η * ρ) * matrixEnergy (fun c => matrixApply Q (E c)) +
      (2 * η / ρ) * readoutLoss T y E *
        (matrixEnergy (fun i => matrixApply Q (T i)) / Fintype.card I) := by
  calc
    _ ≤ ∑ c, ∑ k, ((1 - η * ρ) * (matrixApply Q (E c) k) ^ 2 +
        (η / ρ) * (matrixApply Q (readoutGradient T y E c) k) ^ 2) := by
      simp only [matrixEnergy, projected_readoutStep]
      exact sum_le_sum fun c _ => sum_le_sum fun k _ => decay_forcing_sq_le η ρ _ _ hη hρ hs
    _ = (1 - η * ρ) * matrixEnergy (fun c => matrixApply Q (E c)) +
        (η / ρ) * matrixEnergy (fun c => matrixApply Q (readoutGradient T y E c)) := by
      simp only [matrixEnergy, sum_add_distrib, mul_sum]
    _ ≤ (1 - η * ρ) * matrixEnergy (fun c => matrixApply Q (E c)) +
        (η / ρ) * (2 * readoutLoss T y E *
          (matrixEnergy (fun i => matrixApply Q (T i)) / Fintype.card I)) := by
      exact add_le_add_right (mul_le_mul_of_nonneg_left
        (projected_readoutGradient_energy_le T y E Q) (div_nonneg hη hρ.le)) _
    _ = _ := by ring

omit [Fintype I] [Fintype C] [DecidableEq C] [DecidableEq K] in
/-- Exact finite change: fixed-basis parameter movement plus basis movement,
RD4. This identity does not identify checkpoint differences with one update. -/
theorem moving_residual_identity (Q Q' : K → K → ℝ) (w w' : K → ℝ) (k : K) :
    matrixApply Q' w' k = matrixApply Q w k +
      matrixApply Q (fun l => w' l - w l) k +
      (matrixApply Q' w' k - matrixApply Q w' k) := by
  simp only [matrixApply, linearScore, mul_sub, sum_sub_distrib]
  ring

example (n : ℕ) := projected_readoutRun
  (fun (i k : Bool) => if k then (0 : ℝ) else if i then 2 else 1) id (1 / 2) (1 / 2)
  (fun (c k : Bool) => if k then (if c then 2 else -1) else 1)
  (fun (k l : Bool) => if k ∧ l then (1 : ℝ) else 0)
  (by intro i k; cases k <;> norm_num [matrixApply, linearScore]) n true true
example := projected_gradient_zero
  (fun (i k : Bool) => if k then (0 : ℝ) else if i then 2 else 1) id
  (fun (c k : Bool) => if k then (if c then 2 else -1) else 1)
  (fun (k l : Bool) => if k ∧ l then (1 : ℝ) else 0)
  (by intro i k; cases k <;> norm_num [matrixApply, linearScore]) true true
example := projected_step_of_annihilation
  (fun (i k : Bool) => if k then (0 : ℝ) else if i then 2 else 1) id (1 / 2) (1 / 2)
  (fun (c k : Bool) => if k then (if c then 2 else -1) else 1)
  (fun (k l : Bool) => if k ∧ l then (1 : ℝ) else 0)
  (by intro i k; cases k <;> norm_num [matrixApply, linearScore]) true true
example := decay_forcing_sq_le (1 / 2) (1 / 2) 2 1 (by norm_num) (by norm_num) (by norm_num)
example := readout_step_energy_le
  (fun (i : Bool) (_ : Fin 1) => if i then (2 : ℝ) else 1) id (1 / 2) (1 / 2)
  (fun (c : Bool) (_ : Fin 1) => if c then (1 : ℝ) else -1)
  (fun (_ : Fin 1) (_ : Fin 1) => 1) (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.GradientEvidence
