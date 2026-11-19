module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Width.Basic

/-!
# A bounded three-coordinate recurrence outside every plain two-state mixture

Source: `review/convex/moments.md`, CM110. Three explicit coordinate transfers
are linear contractions, with unit initial state and bounded actual readout.
Their eight-word obstruction is nonzero. This is an expressivity witness,
not a supplied task program, a native initialization or an MQAR lower bound.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Width

namespace Witness

/-- CM110: all three complex coordinates are retained in the witness state. -/
abbrev State := ℂ × (ℂ × ℂ)

/-- CM110: the complete witness's second coordinate readout. -/
noncomputable def second : State →L[ℂ] ℂ :=
  (ContinuousLinearMap.fst ℂ ℂ ℂ).comp (ContinuousLinearMap.snd ℂ ℂ (ℂ × ℂ))

/-- CM110: the complete witness's third coordinate readout. -/
noncomputable def third : State →L[ℂ] ℂ :=
  (ContinuousLinearMap.snd ℂ ℂ ℂ).comp (ContinuousLinearMap.snd ℂ ℂ (ℂ × ℂ))

/-- CM110: the full matrices E01, E10 and E12, acting in sequence order. -/
noncomputable def transition (o : Fin 3) : State →L[ℂ] State :=
  if o = 0 then second.prod (0 : State →L[ℂ] ℂ × ℂ)
  else if o = 1 then (0 : State →L[ℂ] ℂ).prod ((ContinuousLinearMap.fst ℂ ℂ (ℂ × ℂ)).prod 0)
  else (0 : State →L[ℂ] ℂ).prod (third.prod 0)

/-- CM110: actual complete numerical execution, including the zero entries. -/
theorem transition_apply (o : Fin 3) (v : State) :
    transition o v = if o = 0 then (v.2.1, 0, 0)
      else if o = 1 then (0, v.1, 0) else (0, v.2.2, 0) := by
  by_cases h0 : o = 0
  · simp [transition, h0, second, ContinuousLinearMap.fst, ContinuousLinearMap.snd]
    rfl
  · by_cases h1 : o = 1 <;>
      simp [transition, h0, h1, third, ContinuousLinearMap.fst, ContinuousLinearMap.snd]

/-- CM110: each full witness transition is contractive in the product norm. -/
theorem transition_norm_le (o : Fin 3) : ‖transition o‖ ≤ 1 := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by norm_num)
  intro v
  have h0 := norm_fst_le v
  have h1 := (norm_fst_le v.2).trans (norm_snd_le v)
  have h2 := (norm_snd_le v.2).trans (norm_snd_le v)
  rw [one_mul, transition_apply]
  split_ifs <;> simp [h0, h1, h2, norm_nonneg]

/-- CM110: squared Euclidean energy includes every real and imaginary entry. -/
noncomputable def energy (v : State) : ℝ := ‖v.1‖ ^ 2 + ‖v.2.1‖ ^ 2 + ‖v.2.2‖ ^ 2

/-- CM110: the same complete coordinate transfers also contract Euclidean
energy, so the separation is not obtained by expanding state norms. -/
theorem transition_energy_le (o : Fin 3) (v : State) :
    energy (transition o v) ≤ energy v := by
  rw [transition_apply]
  split_ifs <;> simp only [energy, norm_zero, zero_pow (by decide : 2 ≠ 0),
    zero_add, add_zero] <;> nlinarith [sq_nonneg ‖v.1‖, sq_nonneg ‖v.2.1‖,
      sq_nonneg ‖v.2.2‖]

/-- CM110: the witness has unit initial energy and one distinguished original
class readout; all three objects still index actual transitions and outputs. -/
noncomputable def parameters : Parameters (Fin 3) State :=
  ⟨(0, 0, 1), transition, fun output => if output = 0 then second else 0⟩

/-- CM110: a genuine unit-capacity learned-recurrence parameter point. -/
theorem parameters_unit : Unit parameters := by
  refine ⟨?_, transition_norm_le, ?_⟩
  · norm_num [parameters, Prod.norm_def]
  · intro output
    change ‖(if output = 0 then second else 0)‖ ≤ 1
    split_ifs
    · apply ContinuousLinearMap.opNorm_le_bound _ (by norm_num)
      intro v
      simpa [second, one_mul, ContinuousLinearMap.fst, ContinuousLinearMap.snd] using
        (norm_fst_le v.2).trans (norm_snd_le v)
    · simp

example : Unit parameters := parameters_unit

/-- CM110: the three-coordinate initial state also has unit Euclidean energy. -/
theorem initial_energy : energy parameters.initial = 1 := by
  norm_num [energy, parameters]

/-- CM110: every original class readout is bounded by complete state energy. -/
theorem readout_energy_le (output : Fin 3) (v : State) :
    ‖parameters.readout output v‖ ^ 2 ≤ energy v := by
  by_cases h : output = 0 <;>
    simp [parameters, h, second, ContinuousLinearMap.fst, ContinuousLinearMap.snd,
      energy] <;> nlinarith [sq_nonneg ‖v.1‖, sq_nonneg ‖v.2.1‖, sq_nonneg ‖v.2.2‖]

/-- CM110: the same witness remains within Euclidean energy at every depth. -/
theorem run_energy_le (v : State) (xs : List (Fin 3)) :
    energy (runFrom parameters v xs) ≤ energy v := by
  induction xs generalizing v with
  | nil => exact le_rfl
  | cons o xs ih =>
    exact (ih (transition o v)).trans (transition_energy_le o v)

/-- CM110: every executed witness word obeys the original capacity bound. -/
theorem score_bound (xs : List (Fin 3)) (output : Fin 3) :
    |score parameters xs output| ≤ 1 :=
  LearnedRecurrent.score_bound parameters parameters_unit xs output

/-- CM110: the eight actual length-five words violate the two-state identity. -/
theorem obstruction_value :
    obstruction (fun xs => score parameters xs 0) 0 1 2 = -1 := by
  norm_num [obstruction, score, complex, runFrom, parameters, transition_apply,
    second, ContinuousLinearMap.fst, ContinuousLinearMap.snd]

/-- CM110: no finite signed mixture of arbitrary complete two-state blocks
equals this bounded three-state sequence function on every original word. -/
theorem not_two_state_mixture {J : Type*} [Fintype J]
    (f : J → Parameters (Fin 3) Pair.State) (w : J → ℝ) :
    ¬ (∀ xs, (∑ j, w j * score (f j) xs 0) = score parameters xs 0) := by
  intro h
  have hz := mixture_obstruction_zero f w (0 : Fin 3) 1 2 0
  have he : (fun xs => ∑ j, w j * score (f j) xs 0) =
      (fun xs => score parameters xs 0) := funext h
  rw [he, obstruction_value] at hz
  norm_num at hz

/-- CM110: even approximate uniform score recovery has error at least one eighth.
This is a score-function obstruction, not a benchmark classification bound. -/
theorem uniform_error_lower_bound {J : Type*} [Fintype J]
    (f : J → Parameters (Fin 3) Pair.State) (w : J → ℝ) (ε : ℝ)
    (h : ∀ xs, |(∑ j, w j * score (f j) xs 0) - score parameters xs 0| ≤ ε) :
    1 / 8 ≤ ε := by
  have hb := obstruction_lipschitz _ _ ε h (0 : Fin 3) 1 2
  rw [mixture_obstruction_zero, obstruction_value] at hb
  norm_num at hb
  linarith

example : 1 / 8 ≤ (1 : ℝ) := by
  apply uniform_error_lower_bound (J := Fin 1) ![zeroParameters] ![1] 1
  intro xs
  simpa [zero_score] using score_bound xs 0

end Witness

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Width
