module

public import Transformer.Rotate90.Forcing.Span
public import Transformer.Rotate90.Learning.Trajectory

/-!
# Gradient response of the actual finite Rotate90 model

Source: `review/flat_spectrum.md`, *Gradient forcing*, F3--F4. The oracle
differentiates the jointly varying tied embedding and every layer matrix.
The existing analyticity theorem justifies this gradient at positive RMS
epsilon. The algebraic identities below also hold for the total Lean gradient
at other epsilon values; no executable claim is made for invalid RMS settings.
Zero-gradient controls have two scalar degrees of freedom in parameter space,
while their predictions still use the entire nonlinear network.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Forcing

open Learning ScheduleFree
open scoped BigOperators

variable {V K H D : Type*} [Fintype V] [Fintype K] [Fintype H] [Fintype D]

/-- The full sampled-loss gradient at the actual query, in source coordinates. -/
def sampledOracle (eps : ℝ) (order : List D) (samples : ℕ → Finset (V × V))
    (target : V × V → V) (t : ℕ) (y : Index V K H D → ℝ) : Index V K H D → ℝ :=
  (gradient (loss eps order (samples t) target) (WithLp.toLp 2 y)).ofLp

/-- At zero offset this is precisely the existing actual-model trajectory. -/
theorem run_eq_learning (eps : ℝ) (order : List D) (samples : ℕ → Finset (V × V))
    (target : V × V → V) (p : AdamW.Parameters)
    (s : AdamW.TrainingState (Index V K H D)) (n : ℕ) :
    run p 0 (sampledOracle eps order samples target) s n =
      Learning.runFrom eps order samples target p s n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [run, Learning.runFrom, sampledOracle, Nat.zero_add, ih]

/-- F4: full actual-model training has the proved finite transported-gradient
history, without fixing the embedding or linearizing any layer. -/
theorem actual_history_sum (eps : ℝ) (order : List D) (samples : ℕ → Finset (V × V))
    (target : V × V → V) (p : AdamW.Parameters)
    (s : AdamW.TrainingState (Index V K H D)) (n : ℕ) :
    buffers (Learning.runFrom eps order samples target p s n) =
      propagate (transitions p 0 s.history) 0 n (buffers s) +
        ∑ j ∈ Finset.range n, propagate (transitions p 0 s.history)
          (j + 1) (n - 1 - j) (forcing p 0 (sampledOracle eps order samples target) s j) := by
  rw [← run_eq_learning]
  exact run_history_sum p 0 (sampledOracle eps order samples target) s n

/-- The difference of actual evaluation parameter vectors is exactly the
recovered response; zero-gradient decay may itself move the comparison point. -/
theorem actual_evaluation_response (eps : ℝ) (order : List D)
    (samples : ℕ → Finset (V × V)) (target : V × V → V) (p : AdamW.Parameters)
    (s : AdamW.TrainingState (Index V K H D)) (n : ℕ) :
    evaluated p (Learning.runFrom eps order samples target p s n) -
        evaluated p (run p 0 (fun _ _ => 0) s n) =
      WithLp.toLp 2 (readEvaluation p (response (transitions p 0 s.history)
        (forcing p 0 (sampledOracle eps order samples target) s) n)) := by
  rw [← run_eq_learning]
  ext i
  change AdamW.evaluation p (run _ _ _ _ _) i - AdamW.evaluation p (run _ _ _ _ _) i = _
  rw [evaluation_response]
  simp only [Pi.add_apply, add_sub_cancel_left]

omit [Fintype V] [Fintype K] [Fintype H] [Fintype D] in
/-- A zero-gradient evaluation uses one common pair of coefficients for every
trainable scalar parameter, including all layers and tied embedding entries. -/
theorem zero_parameter_plane (p : AdamW.Parameters) (offset : ℕ)
    (s : AdamW.TrainingState (Index V K H D)) (n : ℕ) :
    ∃ a b : ℝ, evaluated p (run p offset (fun _ _ => 0) s n) =
      WithLp.toLp 2 (a • s.point + b • s.base) := by
  obtain ⟨a, b, hab⟩ := Submodule.mem_span_pair.mp (zero_evaluation_span p offset s n)
  exact ⟨a, b, congrArg (WithLp.toLp 2) hab.symm⟩

omit [Fintype V] [Fintype D] in
/-- The entire table of actual nonlinear scores lies in that two-parameter
family. This is not a claim that the score matrix has rank at most two. -/
theorem zero_scores_plane (eps : ℝ) (order : List D) (p : AdamW.Parameters) (offset : ℕ)
    (s : AdamW.TrainingState (Index V K H D)) (n : ℕ) :
    ∃ a b : ℝ, ∀ u v q : V,
      score eps order (evaluated p (run p offset (fun _ _ => 0) s n)) u v q =
        score eps order (WithLp.toLp 2 (a • s.point + b • s.base)) u v q := by
  obtain ⟨a, b, hab⟩ := zero_parameter_plane p offset s n
  exact ⟨a, b, fun u v q => congrArg (fun w => score eps order w u v q) hab⟩

/-- New gradients change the actual logits through the recovered response
inside the full nonlinear forward pass, not by adding a linear logit response. -/
theorem actual_score_response (eps : ℝ) (order : List D)
    (samples : ℕ → Finset (V × V)) (target : V × V → V) (p : AdamW.Parameters)
    (s : AdamW.TrainingState (Index V K H D)) (n : ℕ) (u v q : V) :
    score eps order (evaluated p (Learning.runFrom eps order samples target p s n)) u v q =
      score eps order
        (evaluated p (run p 0 (fun _ _ => 0) s n) +
          WithLp.toLp 2 (readEvaluation p (response (transitions p 0 s.history)
            (forcing p 0 (sampledOracle eps order samples target) s) n))) u v q := by
  have h := actual_evaluation_response eps order samples target p s n
  have heq : evaluated p (Learning.runFrom eps order samples target p s n) =
      evaluated p (run p 0 (fun _ _ => 0) s n) +
        WithLp.toLp 2 (readEvaluation p (response (transitions p 0 s.history)
          (forcing p 0 (sampledOracle eps order samples target) s) n)) := by
    rw [← h]
    abel
  exact congrArg (fun w => score eps order w u v q) heq

example := actual_history_sum (K := Fin 8) (H := Fin 43)
  (1/8388608) (List.finRange 3)
  (fun _ => {(0, 1), (1, 2)} : ℕ → Finset (ZMod 3 × ZMod 3))
  (fun r => (r.1 / r.2) ^ 2)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩
  ⟨fun _ => 1, fun _ => 2, ⟨fun _ => 1, 1, 3/1000⟩⟩ 5
example := zero_scores_plane (V := ZMod 3) (K := Fin 8) (H := Fin 43)
  (1/8388608) (List.finRange 3)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 400
  ⟨fun i => if i.isLeft then 1 else 0,
    fun i => if i.isLeft then 0 else 1, ⟨fun _ => 1, 1, 3/1000⟩⟩ 1100

end Transformer.Rotate90.Forcing
