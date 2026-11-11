module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch.Gradient

/-!
# Exact singleton oracles can remain at fifty percent accuracy

Source: `review/flat_spectrum.md`, *Repairing minibatch and full-batch failures*,
MF1. The run starts just after the native algorithm's first step of size one.
At every later step it selects the first tied key, with arbitrary retained
decoder history and any feasible convex step size. Every such oracle is exact
for the current singleton CE gradient, but all iterates ignore the separating
slot. Every deterministic Boolean readout gets exactly one of the two examples
right. A feasible channel using the other slot fits both with unit logit gap.
This is an optimization failure on observed, representable inputs.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch

open scoped BigOperators

/-- MF1: the first-key tie rule, starting after the first size-one FW update.
The label order and all unobserved decoder choices remain arbitrary. -/
noncomputable def firstKeyRun (η : ℕ → ℝ) (label : ℕ → Bool)
    (history : ℕ → Bool → Bool → Bool) : ℕ → Channel Unit Bool Bool Bool
  | 0 => vertex (fun _ : Unit => false) (fun _ => observedDecoder (label 0) (history 0))
  | t + 1 => (1 - η t) • firstKeyRun η label history t +
      η t • vertex (fun _ : Unit => false)
        (fun _ => observedDecoder (label (t + 1)) (history (t + 1)))

/-- Each iterate respects the same joint convex domain as the original model. -/
theorem feasible_firstKeyRun (η : ℕ → ℝ) (hη : ∀ t, 0 ≤ η t ∧ η t ≤ 1)
    (label : ℕ → Bool) (history : ℕ → Bool → Bool → Bool) (t : ℕ) :
    Feasible (firstKeyRun η label history t) := by
  induction t with
  | zero => exact feasible_vertex _ _
  | succ t ih => exact feasible_oracle_step ih _ _ _ (hη t).1 (hη t).2

/-- The ignored slot remains identically zero under every update, MF1. -/
theorem firstKeyRun_separating_slot_zero (η : ℕ → ℝ) (label : ℕ → Bool)
    (history : ℕ → Bool → Bool → Bool) (t : ℕ) (v c : Bool) :
    firstKeyRun η label history t () true v c = 0 := by
  induction t with
  | zero => simp [firstKeyRun, vertex]
  | succ t ih => simp [firstKeyRun, vertex, ih]

/-- Ignoring the separating slot makes the two complete logit vectors equal. -/
theorem logits_eq_of_separating_slot_zero {W : Channel Unit Bool Bool Bool}
    (h : ∀ v c, W () true v c = 0) :
    logits W () (input false) = logits W () (input true) := by
  funext c
  simp [logits, input, h]

/-- **Every iterate gives the same logits to opposite labels**, MF1. -/
theorem firstKeyRun_logits_eq (η : ℕ → ℝ) (label : ℕ → Bool)
    (history : ℕ → Bool → Bool → Bool) (t : ℕ) :
    logits (firstKeyRun η label history t) () (input false) =
      logits (firstKeyRun η label history t) () (input true) :=
  logits_eq_of_separating_slot_zero (firstKeyRun_separating_slot_zero η label history t)

/-- **Exactly one correct answer out of two**, MF1, for any deterministic
readout of the logits, including any fixed argmax tie rule. -/
theorem firstKeyRun_one_correct (η : ℕ → ℝ) (label : ℕ → Bool)
    (history : ℕ → Bool → Bool → Bool) (t : ℕ) (read : (Bool → ℝ) → Bool) :
    (if read (logits (firstKeyRun η label history t) () (input false)) = false
      then 1 else 0 : ℕ) +
    (if read (logits (firstKeyRun η label history t) () (input true)) = true
      then 1 else 0 : ℕ) = 1 := by
  rw [firstKeyRun_logits_eq]
  cases read (logits (firstKeyRun η label history t) () (input true)) <;> decide

/-- The separating vertex is feasible and gives the true class logit one and
the other class logit zero on both examples, MF1. -/
theorem separating_vertex_fits :
    Feasible (vertex (fun _ : Unit => true) (fun _ _ => (id : Bool → Bool))) ∧
    ∀ b c : Bool, logits (vertex (fun _ : Unit => true)
      (fun _ _ => (id : Bool → Bool))) () (input b) c =
      if c = b then 1 else 0 := by
  refine ⟨feasible_vertex _ _, ?_⟩
  intro b c
  cases b <;> cases c <;> norm_num [logits, vertex, input]

/-! Explicit feasible runs satisfy every hypothesis. The failure statements
allow arbitrary label order rather than relying on an alternating-data premise. -/

example (t : ℕ) := feasible_firstKeyRun (fun _ => (1 : ℝ) / 2)
  (fun _ => by norm_num) (fun i => decide (i % 2 = 0)) (fun _ _ => id) t

example := logits_eq_of_separating_slot_zero
  (W := vertex (fun _ : Unit => false) (fun _ _ => id))
  (by intro v c; simp [vertex])

example (t : ℕ) := firstKeyRun_one_correct (fun _ => (1 : ℝ) / 2)
  (fun i => decide (i % 2 = 0)) (fun _ _ => id) t (fun z => decide (z false < z true))

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch
