import Transformer.GPTMini.Convex.Structured.MarkovChain

/-!
# Finite learned-energy confidence bounds

Source: Structured.Basic's actual affine Gibbs distribution at 19ff013
and the freely learned state rows at 11899f6. A derived complete-energy
gap bounds every rival's true probability and therefore the selected
configuration's mass. These are local operator inequalities, not a
Basis solver theorem with a correct encoder or correct logits assumed.
Raw semantic constructions must discharge the local energy inequalities.

For the causal state head, the same bounds apply to actual row softmax
and to explicit finite freely trainable sharp-row assignments. No
infinite logits, hard argmax transitions, special optimizer or inferred
state correctness is an input to these computations. Propagation of
confidence along raw paths and complete Basis task capability remain
subsequent obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {E R : Type*} [AddCommGroup E] [Module ℝ E] [Fintype R] [Nonempty R]

/-- A real rival's probability is bounded by its true complete-energy deficit.
Source: the genuine positive partition includes the selected configuration, and exponentiation preserves the local gap. -/
theorem probability_rival_le (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E)
    (selected rival : R) (gap : ℝ)
    (hgap : energy linear offset θ rival ≤ energy linear offset θ selected - gap) :
    probability linear offset θ rival ≤ Real.exp (-gap) := by
  unfold probability
  apply (div_le_iff₀ (partition_pos linear offset θ)).2
  calc
    Real.exp (energy linear offset θ rival) ≤ Real.exp (energy linear offset θ selected - gap) :=
      Real.exp_le_exp.mpr hgap
    _ = Real.exp (-gap) * Real.exp (energy linear offset θ selected) := by
      rw [sub_eq_add_neg, Real.exp_add]
      ring
    _ ≤ Real.exp (-gap) * partition linear offset θ :=
      mul_le_mul_of_nonneg_left (partition_ge linear offset θ selected) (Real.exp_pos _).le

example : energy (fun state : Fin 2 => (LinearMap.proj state : (Fin 2 → ℝ) →ₗ[ℝ] ℝ))
    (fun _ => 0) ![1, 0] 1 ≤
    energy (fun state : Fin 2 => (LinearMap.proj state : (Fin 2 → ℝ) →ₗ[ℝ] ℝ))
      (fun _ => 0) ![1, 0] 0 - 1 := by
  norm_num [energy, LinearMap.proj_apply]

/-- An actual finite complete configuration retains mass at least one minus the true number-of-choices tail bound.
Source: all actual rival probabilities, full Gibbs normalization and the derived finite exponential energy-gap inequality. -/
theorem probability_gap (linear : R → E →ₗ[ℝ] ℝ) (offset : R → ℝ) (θ : E)
    (selected : R) (gap : ℝ)
    (hgap : ∀ rival, rival ≠ selected → energy linear offset θ rival ≤ energy linear offset θ selected - gap) :
    1 - (Fintype.card R : ℝ) * Real.exp (-gap) ≤ probability linear offset θ selected := by
  have hs : (∑ rival ∈ Finset.univ.erase selected, probability linear offset θ rival) +
      probability linear offset θ selected = 1 := by
    rw [Finset.sum_erase_add _ _ (Finset.mem_univ selected), probability_sum]
  have htail : (∑ rival ∈ Finset.univ.erase selected, probability linear offset θ rival) ≤
      ∑ _ ∈ Finset.univ.erase selected, Real.exp (-gap) := by
    apply Finset.sum_le_sum
    intro rival hr
    exact probability_rival_le linear offset θ selected rival gap (hgap rival (Finset.mem_erase.mp hr).1)
  rw [Finset.sum_const, nsmul_eq_mul] at htail
  have hcard : ((Finset.univ.erase selected).card : ℝ) ≤ (Fintype.card R : ℝ) := by
    exact_mod_cast (Finset.card_erase_le (s := Finset.univ) (a := selected))
  have hbound := htail.trans (mul_le_mul_of_nonneg_right hcard (Real.exp_pos (-gap)).le)
  linarith

example : ∀ rival : Fin 2, rival ≠ 0 →
    energy (fun state : Fin 2 => (LinearMap.proj state : (Fin 2 → ℝ) →ₗ[ℝ] ℝ))
      (fun _ => 0) ![1, 0] rival ≤
    energy (fun state : Fin 2 => (LinearMap.proj state : (Fin 2 → ℝ) →ₗ[ℝ] ℝ))
      (fun _ => 0) ![1, 0] 0 - 1 := by
  intro rival hr
  fin_cases rival
  · exact False.elim (hr rfl)
  · norm_num [energy, LinearMap.proj_apply]

variable {S : Type*} [Fintype S] [Nonempty S]

omit [Nonempty S] in
/-- A genuinely computed transition row is the same finite affine Gibbs distribution on free state logits.
Source: actual coordinate linear reads, proving the state/generic-probability bridge without a supplied row invariant. -/
theorem stateRow_probability (score : S → ℝ) (state : S) :
    stateRow score state = probability (fun next => LinearMap.proj next) (fun _ => 0) score state := by
  simp only [stateRow, probability, partition, energy, LinearMap.proj_apply, add_zero]

/-- The actual transition probability of a rival state has the same finite exponential deficit bound.
Source: true row softmax, positive normalization and the local raw-logit inequality. -/
theorem stateRow_rival_le (score : S → ℝ) (selected rival : S) (gap : ℝ)
    (hgap : score rival ≤ score selected - gap) : stateRow score rival ≤ Real.exp (-gap) := by
  rw [stateRow_probability]
  apply probability_rival_le (selected := selected)
  simpa only [energy, LinearMap.proj_apply, add_zero] using hgap

example : (![1, 0] : Fin 2 → ℝ) 1 ≤ (![1, 0] : Fin 2 → ℝ) 0 - 1 := by norm_num

/-- A true learned transition-row gap gives a quantitative actual next-state probability bound.
Source: the real row/Gibbs identity and finite competitor bound, without an assumed encoded semantic state. -/
theorem stateRow_gap (score : S → ℝ) (selected : S) (gap : ℝ)
    (hgap : ∀ rival, rival ≠ selected → score rival ≤ score selected - gap) :
    1 - (Fintype.card S : ℝ) * Real.exp (-gap) ≤ stateRow score selected := by
  rw [stateRow_probability]
  apply probability_gap
  intro rival hr
  simpa only [energy, LinearMap.proj_apply, add_zero] using hgap rival hr

/-- No actual normalized transition-row entry exceeds one.
Source: the same genuine row/Gibbs distribution and its positive finite partition. -/
theorem stateRow_le_one (score : S → ℝ) (state : S) : stateRow score state ≤ 1 := by
  rw [stateRow_probability]
  exact probability_le_one _ _ _ _

/-- A finite free raw-logit assignment favors one state without hard transitions or infinite temperatures.
Source: the proposed learned row parameters; this is a capacity witness, not the architecture's fixed transition rule. -/
def sharpRowLogits (selected : S) (gain : ℝ) (state : S) : ℝ :=
  if state = selected then gain else 0

omit [Fintype S] [Nonempty S] in
/-- The finite raw sharp-row assignment derives its gap against every distinct physical state.
Source: actual learned-logit witness coordinates, before computing any inferred state probability. -/
theorem sharpRowLogits_gap (selected rival : S) (gain : ℝ) (hr : rival ≠ selected) :
    sharpRowLogits selected gain rival ≤ sharpRowLogits selected gain selected - gain := by
  simp only [sharpRowLogits, ite_eq_right hr, ite_true]
  linarith

example : (1 : Fin 6) ≠ 0 := by decide

/-- The actual finite sharp-row witness has a uniformly derived target-state mass bound.
Source: explicit raw logits and the true row gap, not a probability assignment substituted for softmax. -/
theorem stateRow_sharp (selected : S) (gain : ℝ) :
    1 - (Fintype.card S : ℝ) * Real.exp (-gain) ≤ stateRow (sharpRowLogits selected gain) selected := by
  apply stateRow_gap
  intro rival hr
  exact sharpRowLogits_gap selected rival gain hr

example : ∀ rival : Fin 6, rival ≠ 0 →
    sharpRowLogits (0 : Fin 6) 1 rival ≤ sharpRowLogits (0 : Fin 6) 1 0 - 1 := by
  intro rival hr
  simp only [sharpRowLogits, ite_eq_right hr, ite_true]
  norm_num

/-- Every six-state transition witness admits the same finite confidence bound at an actual ordinary real gain.
Source: the compact freely trainable six-state table, with no exact zero/one state transitions assumed. -/
example : 1 - 6 * Real.exp (-(12 : ℝ)) ≤ stateRow (sharpRowLogits (4 : Fin 6) 12) 4 := by
  simpa only [Fintype.card_fin, Nat.cast_ofNat] using stateRow_sharp (4 : Fin 6) 12

end
end Transformer.GPTMini.Convex.Structured
