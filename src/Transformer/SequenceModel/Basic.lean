module

public import Transformer.SequenceModel.Defs

/-!
# Source response without an architectural restriction

Source: `review/flat_spectrum.md`, *General positional models*, G1. The
finite response applies to arbitrary inputs, positional maps and attention.
It controls parameter movement, not logit rank or learning success. The
nonlinear forward remains outside the parameter-space decomposition.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel

open ScheduleFree Rotate90 Forcing
open scoped BigOperators

variable {I X V : Type*} [Fintype I] [Fintype V]

/-- G1: every finite sampled trajectory is its homogeneous matrix evolution
plus the transported history of its actual normalized gradients. -/
theorem history_sum (forward : Weights I → X → V → ℝ)
    (samples : ℕ → Finset X) (target : X → V) (p : AdamW.Parameters)
    (offset : ℕ) (s : AdamW.TrainingState I) (n : ℕ) :
    buffers (trajectory forward samples target p offset s n) =
      propagate (transitions p offset s.history) 0 n (buffers s) +
        ∑ j ∈ Finset.range n, propagate (transitions p offset s.history)
          (j + 1) (n - 1 - j) (forcing p offset (sampledOracle forward samples target) s j) := by
  rw [trajectory_eq_run]
  exact run_history_sum p offset (sampledOracle forward samples target) s n

/-- G1: gradients affect an arbitrary nonlinear predictor through their
parameter response. This does not assert an additive response in logits. -/
theorem score_response (forward : Weights I → X → V → ℝ)
    (samples : ℕ → Finset X) (target : X → V) (p : AdamW.Parameters)
    (offset : ℕ) (s : AdamW.TrainingState I) (n : ℕ) (x : X) (v : V) :
    forward (Learning.evaluated p (trajectory forward samples target p offset s n)) x v =
      forward (Learning.evaluated p (run p offset (fun _ _ => 0) s n) +
        WithLp.toLp 2 (readEvaluation p (response (transitions p offset s.history)
          (forcing p offset (sampledOracle forward samples target) s) n))) x v := by
  rw [trajectory_eq_run]
  congr 2
  change WithLp.toLp 2 (AdamW.evaluation p (run _ _ _ _ _)) = _
  rw [evaluation_response]
  rfl

omit [Fintype I] [Fintype V] in
/-- G1: arbitrary output tables under zero gradients stay in a common
two-parameter family of full networks, even when input length is large. -/
theorem zero_scores_plane (forward : Weights I → X → V → ℝ)
    (p : AdamW.Parameters) (offset : ℕ) (s : AdamW.TrainingState I) (n : ℕ) :
    ∃ a b : ℝ, ∀ x v,
      forward (Learning.evaluated p (run p offset (fun _ _ => 0) s n)) x v =
        forward (WithLp.toLp 2 (a • s.point + b • s.base)) x v := by
  obtain ⟨a, b, hab⟩ := Submodule.mem_span_pair.mp (zero_evaluation_span p offset s n)
  refine ⟨a, b, ?_⟩
  intro x v
  change forward (WithLp.toLp 2 (AdamW.evaluation p _)) x v = _
  rw [← hab]

example := history_sum (I := Fin 3) (X := Fin 12 → Fin 3)
  (fun w x q => w.ofLp q * (x 0).val + w.ofLp (x 11))
  (fun _ => {fun _ => 1}) (fun x => x 0)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 400
  ⟨fun i => i.val, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩ 1100

example := zero_scores_plane (I := Fin 3) (X := Fin 100 → Fin 3)
  (fun w x q => Real.exp (w.ofLp q * (x 99).val))
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 400
  ⟨fun i => i.val, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩ 1100

end Transformer.SequenceModel
