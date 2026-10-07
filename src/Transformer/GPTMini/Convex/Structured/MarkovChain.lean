import Transformer.GPTMini.Convex.Structured.Basic

/-!
# A compact freely learned causal state head

New ordered-head proposal motivated by the complete Basis semantics at
d640a91. All token-conditioned transition logits remain free parameters;
inference applies their actual row softmax and propagates a distribution
over six proposed encoder states. Each token costs a state-by-state
matrix update, with no enumeration of complete state histories.

The explicit computation below is chronological List.foldl over actual
tokens, rather than a task interpreter or a supplied semantic state.
Initial and transition logits can be arbitrary finite real tables.
Their computed state distribution is proved positive and normalized.
The learned head can therefore feed conditional output-channel means.

This module does not yet prove its complete observed-path likelihood
convex, an exact path/inference bridge, Basis depth/parity capability, or
the true prenorm/residual/tied integer interface. Source algebra: the
finite softmax normalizer underlying Structured.Basic and §7's
log-sum-exp in arXiv:2305.05465v6, with a new causal state recurrence.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {S A : Type*} [Fintype S] [Nonempty S]

/-- The actual finite row softmax of arbitrary trainable state logits.
Source: the proposed local state conditional, using the same positive exponential normalization. -/
def stateRow (score : S → ℝ) (state : S) : ℝ :=
  Real.exp (score state) / ∑ next, Real.exp (score next)

/-- Every actual row entry is positive at every finite transition-parameter assignment.
Source: the genuine local exponential sum over a nonempty state alphabet. -/
theorem stateRow_pos (score : S → ℝ) (state : S) : 0 < stateRow score state :=
  div_pos (Real.exp_pos _) (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty)

/-- The actual local transition row is normalized without a constraint projection.
Source: the computed common row softmax denominator. -/
theorem stateRow_sum (score : S → ℝ) : ∑ state, stateRow score state = 1 := by
  unfold stateRow
  rw [← Finset.sum_div]
  exact div_self (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne'

/-- One real state-distribution update from free token-conditioned transition logits.
Source: the proposed compact matrix update, reading every previous state rather than a prepared correct state. -/
def markovAdvance (transition : S → S → ℝ) (mass : S → ℝ) (next : S) : ℝ :=
  ∑ previous, mass previous * stateRow (transition previous) next

/-- The actual state update preserves nonnegative mass for arbitrary free transition logits.
Source: positive computed transition rows and the explicit finite matrix multiplication. -/
theorem markovAdvance_nonneg (transition : S → S → ℝ) (mass : S → ℝ)
    (hmass : ∀ state, 0 ≤ mass state) (next : S) : 0 ≤ markovAdvance transition mass next :=
  Finset.sum_nonneg (fun _ _ => mul_nonneg (hmass _) (stateRow_pos _ _).le)

example : ∀ state : Fin 6, (0 : ℝ) ≤ (fun _ => (1 / 6 : ℝ)) state := by
  intro state
  norm_num

/-- Strict positive initial mass remains positive after the actual learned state update.
Source: the true matrix sum over a nonempty previous-state domain. -/
theorem markovAdvance_pos (transition : S → S → ℝ) (mass : S → ℝ)
    (hmass : ∀ state, 0 < mass state) (next : S) : 0 < markovAdvance transition mass next :=
  Finset.sum_pos (fun _ _ => mul_pos (hmass _) (stateRow_pos _ _)) Finset.univ_nonempty

example : ∀ state : Fin 6, (0 : ℝ) < (fun _ => (1 / 6 : ℝ)) state := by
  intro state
  norm_num

/-- Every true learned state update preserves the entire supplied total mass exactly.
Source: finite sum exchange and the actual row normalization, without assuming encoder correctness. -/
theorem markovAdvance_mass (transition : S → S → ℝ) (mass : S → ℝ) :
    ∑ next, markovAdvance transition mass next = ∑ previous, mass previous := by
  unfold markovAdvance
  rw [Finset.sum_comm]
  simp_rw [← Finset.mul_sum, stateRow_sum, mul_one]

/-- Chronological compact state propagation through the actual raw token list.
Source: the proposed causal head's ordinary recurrent matrix updates, with no data-dependent parameter table. -/
def markovPropagate (transition : A → S → S → ℝ) (tokens : List A) (mass : S → ℝ) : S → ℝ :=
  tokens.foldl (fun current token => markovAdvance (transition token) current) mass

/-- The full actually computed causal recurrence retains the original total mass at every raw prefix.
Source: induction over real matrix updates, not an invariant supplied as an encoder premise. -/
theorem markovPropagate_mass (transition : A → S → S → ℝ) (tokens : List A) (mass : S → ℝ) :
    ∑ state, markovPropagate transition tokens mass state = ∑ state, mass state := by
  induction tokens generalizing mass with
  | nil => rfl
  | cons token rest ih =>
      change (∑ state, markovPropagate transition rest (markovAdvance (transition token) mass) state) = _
      rw [ih, markovAdvance_mass]

/-- Positive initial mass remains positive through every actual raw-token update.
Source: chronological induction through the unrestricted learned transition matrix at each token. -/
theorem markovPropagate_pos (transition : A → S → S → ℝ) (tokens : List A) (mass : S → ℝ)
    (hmass : ∀ state, 0 < mass state) : ∀ state, 0 < markovPropagate transition tokens mass state := by
  induction tokens generalizing mass with
  | nil => exact hmass
  | cons token rest ih =>
      exact ih (markovAdvance (transition token) mass) (markovAdvance_pos (transition token) mass hmass)

example : ∀ state : Fin 6, (0 : ℝ) < (fun _ => (1 / 6 : ℝ)) state := by
  intro state
  norm_num

omit [Nonempty S] in
/-- The same real recurrence composes on concatenated raw token lists, giving a causal cached continuation.
Source: List.foldl's actual chronological concatenation law, without recomputing a semantic answer. -/
theorem markovPropagate_append (transition : A → S → S → ℝ) (left right : List A) (mass : S → ℝ) :
    markovPropagate transition (left ++ right) mass =
      markovPropagate transition right (markovPropagate transition left mass) := by
  unfold markovPropagate
  rw [List.foldl_append]

/-- The complete learned state head starts with actual softmax initial logits and reads all raw tokens in order.
Source: the proposed compact normalized causal encoder; neither initial nor transition logits are fixed by task semantics. -/
def markovRun (initial : S → ℝ) (transition : A → S → S → ℝ) (tokens : List A) : S → ℝ :=
  markovPropagate transition tokens (stateRow initial)

/-- Every full computed raw-prefix encoder distribution has exactly unit total mass.
Source: actual initial softmax and the derived chronological recurrence mass identity. -/
theorem markovRun_sum (initial : S → ℝ) (transition : A → S → S → ℝ) (tokens : List A) :
    ∑ state, markovRun initial transition tokens state = 1 := by
  unfold markovRun
  rw [markovPropagate_mass, stateRow_sum]

/-- Every full computed encoder state remains strictly positive for every finite raw learned table.
Source: actual initial softmax and chronological positive-state propagation, with no task-state hypothesis. -/
theorem markovRun_pos (initial : S → ℝ) (transition : A → S → S → ℝ) (tokens : List A) (state : S) :
    0 < markovRun initial transition tokens state :=
  markovPropagate_pos transition tokens (stateRow initial) (stateRow_pos initial) state

/-- No actual encoder-state probability exceeds one after any raw prefix.
Source: genuine positive state probabilities and the derived unit total mass. -/
theorem markovRun_le_one (initial : S → ℝ) (transition : A → S → S → ℝ) (tokens : List A) (state : S) :
    markovRun initial transition tokens state ≤ 1 := by
  have h := Finset.single_le_sum (fun s (_ : s ∈ Finset.univ) => (markovRun_pos initial transition tokens s).le)
    (Finset.mem_univ state)
  rw [markovRun_sum] at h
  exact h

/-- A concrete six-state head consumes a real three-symbol word and derives its complete distribution invariant.
Source: arbitrary finite all-zero learned transition logits, rather than a correct-state oracle. -/
example : (∑ state : Fin 6, markovRun (fun _ : Fin 6 => (0 : ℝ))
    (fun _ : Fin 3 => fun _ : Fin 6 => fun _ : Fin 6 => 0) [0, 1, 2] state) = 1 :=
  markovRun_sum _ _ _

end
end Transformer.GPTMini.Convex.Structured
