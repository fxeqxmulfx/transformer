module

public import Transformer.Rotate90.Forcing.Defs

/-!
# Finite source evolution and transported gradient history

Source: `review/flat_spectrum.md`, *Gradient forcing*, F2. Matrix products
and forced response are defined independently of the optimizer trajectory.
The source identity identifies their sum with both actual buffers. The
transported impulses use the actual normalized gradients, not predictions
or gradients evaluated along the homogeneous control.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Forcing

open ScheduleFree
open scoped BigOperators

section LinearResponse

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- The ordered product of n transitions, beginning at absolute index start. -/
def propagate (A : ℕ → E →ₗ[ℝ] E) (start : ℕ) : ℕ → E →ₗ[ℝ] E
  | 0 => LinearMap.id
  | n + 1 => (A (start + n)).comp (propagate A start n)

/-- An independently accumulated response to a supplied sequence of impulses. -/
def response (A : ℕ → E →ₗ[ℝ] E) (b : ℕ → E) : ℕ → E
  | 0 => 0
  | n + 1 => A n (response A b n) + b n

/-- F2: every earlier impulse is transported through all later transitions. -/
theorem response_sum (A : ℕ → E →ₗ[ℝ] E) (b : ℕ → E) (n : ℕ) :
    response A b n = ∑ j ∈ Finset.range n, propagate A (j + 1) (n - 1 - j) (b j) := by
  induction n with
  | zero => simp [response]
  | succ n ih =>
    rw [response, ih, map_sum, Finset.sum_range_succ]
    have hn : n + 1 - 1 - n = 0 := by omega
    rw [hn, propagate, LinearMap.id_apply]
    congr 1
    apply Finset.sum_congr rfl
    intro j hj
    have hjn : j < n := Finset.mem_range.mp hj
    have hlen : n + 1 - 1 - j = (n - 1 - j) + 1 := by omega
    have hindex : j + 1 + (n - 1 - j) = n := by omega
    rw [hlen, propagate, LinearMap.comp_apply, hindex]

/-- Zero input has zero response; the homogeneous matrices can still move data. -/
theorem response_zero (A : ℕ → E →ₗ[ℝ] E) (n : ℕ) : response A 0 n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [response, ih]

end LinearResponse

variable {I : Type*}

/-- The complete matrix sequence is fixed by rates, time and initial history. -/
def transitions (p : AdamW.Parameters) (offset : ℕ) (h : AdamW.History I) (n : ℕ) :
    Buffers I →ₗ[ℝ] Buffers I :=
  transition p (offset + n) (clock p offset h n)

/-- Every impulse uses the active trajectory's query, gradient and variance. -/
def forcing (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I) (n : ℕ) : Buffers I :=
  let state := run p offset oracle s n
  impulse p (offset + n) state.history
    (normalizedInput p (offset + n) state (oracle (offset + n) state.point))

/-- F2 on the actual source: homogeneous evolution plus its accumulated forcing. -/
theorem run_split (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I) (n : ℕ) :
    buffers (run p offset oracle s n) =
      propagate (transitions p offset s.history) 0 n (buffers s) +
        response (transitions p offset s.history) (forcing p offset oracle s) n := by
  induction n with
  | zero => simp [run, propagate, response]
  | succ n ih =>
    change buffers (AdamW.trainingStep _ _ _ _) = _
    rw [step_split, transition_clock, ih]
    simp only [propagate, response, Nat.zero_add, LinearMap.comp_apply, map_add]
    change transitions p offset s.history n _ +
      transitions p offset s.history n _ + forcing p offset oracle s n = _
    rw [add_assoc]

/-- In particular the source's finite past gradient history determines the
exact response through its chronologically ordered transition products. -/
theorem run_history_sum (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I) (n : ℕ) :
    buffers (run p offset oracle s n) =
      propagate (transitions p offset s.history) 0 n (buffers s) +
        ∑ j ∈ Finset.range n, propagate (transitions p offset s.history)
          (j + 1) (n - 1 - j) (forcing p offset oracle s j) := by
  rw [run_split, response_sum]

/-- No gradient impulse remains when zero tensors are supplied to the source. -/
theorem forcing_zero (p : AdamW.Parameters) (offset : ℕ)
    (s : AdamW.TrainingState I) (n : ℕ) : forcing p offset (fun _ _ => 0) s n = 0 := by
  simp [forcing, normalizedInput_zero, impulse]

/-- The actual zero-gradient control equals the independent homogeneous product. -/
theorem run_zero (p : AdamW.Parameters) (offset : ℕ) (s : AdamW.TrainingState I) (n : ℕ) :
    buffers (run p offset (fun _ _ => 0) s n) =
      propagate (transitions p offset s.history) 0 n (buffers s) := by
  rw [run_split]
  have hb : forcing p offset (fun _ _ => 0) s = 0 := by
    funext j
    exact forcing_zero p offset s j
  rw [hb, response_zero, add_zero]

/-- F4's evaluation difference is the linear recovery of the actual response.
The two paths use different gradient oracles but the same scalar clock. -/
theorem evaluation_response (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I) (n : ℕ) :
    AdamW.evaluation p (run p offset oracle s n) =
      AdamW.evaluation p (run p offset (fun _ _ => 0) s n) +
        readEvaluation p (response (transitions p offset s.history) (forcing p offset oracle s) n) := by
  rw [← readEvaluation_buffers, ← readEvaluation_buffers, run_split, run_zero, map_add]

example := run_history_sum (I := Fin 3)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 400 (fun _ y => y)
  ⟨fun i => i.val, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩ 1100
example := evaluation_response (I := Fin 3)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ 400 (fun _ y => y)
  ⟨fun i => i.val, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩ 1100

end Transformer.Rotate90.Forcing
