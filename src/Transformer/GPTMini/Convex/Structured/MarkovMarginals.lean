import Transformer.GPTMini.Convex.Structured.MarkovTraining

/-!
# Exact compact state-path marginalization

Source: the actual freely learned causal row-softmax proposal at 11899f6,
with its convex fully observed path likelihood. Complete state histories
are proof objects only: inference stores one state vector and performs
one state-by-state matrix update per raw token. The identities here hold
for every finite initial/transition table, without a solved task-state
premise or an oracle intermediate state during inference.

The endpoint expectation of the exact normalized path model is proved
equal to markovRun's actual compact causal computation. A backward
recurrence is used only to establish this algebraic identity. Neither
path enumeration nor the backward recurrence is required by the forward
implementation. Conditional learned value emission, raw Basis capability
and the residual/tied integer interface remain separate obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {S A : Type*} [Fintype S] [Nonempty S]

/-- The actual endpoint of a post-token state history, retaining the initial state for an empty word.
Source: the chronological path representation in MarkovTraining, with its last physical state. -/
def statePathEnd (start : S) : (n : ℕ) → (Fin n → S) → S
  | 0, _ => start
  | n + 1, path => path (Fin.last n)

omit [Fintype S] [Nonempty S] in
/-- Splitting the first state of a real history leaves exactly the same final state.
Source: the true last-state definition and Fin.cons's physical indexing. -/
theorem statePathEnd_cons (start head : S) (n : ℕ) (tail : Fin n → S) :
    statePathEnd start (n + 1) (Fin.cons head tail) = statePathEnd head n tail := by
  cases n with
  | zero =>
      change (Fin.cons head tail : Fin 1 → S) 0 = head
      exact Fin.cons_zero _ _
  | succ n =>
      change (Fin.cons head tail : Fin (n + 2) → S) (Fin.last (n + 1)) = tail (Fin.last n)
      rw [← Fin.succ_last, Fin.cons_succ]

omit [Nonempty S] in
/-- All histories can be split into their first state and remaining history, without changing any statistic.
Source: the genuine Fin.cons equivalence; the product decomposition is used only in the proof of contraction. -/
theorem sum_statePath_cons (n : ℕ) (statistic : (Fin (n + 1) → S) → ℝ) :
    (∑ path, statistic path) = ∑ head, ∑ tail, statistic (Fin.cons head tail) := by
  calc
    (∑ path, statistic path) = ∑ pair : S × (Fin n → S), statistic (Fin.cons pair.1 pair.2) := by
      symm
      convert (Fin.consEquiv (fun _ : Fin (n + 1) => S)).sum_comp statistic
      rfl
    _ = ∑ head, ∑ tail, statistic (Fin.cons head tail) := Fintype.sum_prod_type _

/-- The probability of every full learned post-token path sums to one, without a global path partition.
Source: chronological path decomposition into a normalized row and an independently normalized suffix. -/
theorem conditionalStatePath_sum (transition : A → S → S → ℝ) (start : S) (tokens : List A) :
    ∑ path : Fin tokens.length → S, conditionalStatePath transition start tokens path = 1 := by
  induction tokens generalizing start with
  | nil =>
      change (∑ _ : Fin 0 → S, (1 : ℝ)) = 1
      exact Fintype.sum_unique _
  | cons token rest ih =>
      have hsplit : (∑ path : Fin (token :: rest).length → S,
          conditionalStatePath transition start (token :: rest) path) =
          ∑ head, ∑ tail : Fin rest.length → S,
            stateRow (transition token start) head * conditionalStatePath transition head rest tail := by
        convert sum_statePath_cons rest.length
          (fun path => conditionalStatePath transition start (token :: rest) path) using 1
        simp only [conditionalStatePath, Fin.cons_zero, Fin.tail_cons]
      rw [hsplit]
      simp_rw [← Finset.mul_sum, ih, mul_one]
      exact stateRow_sum _

/-- A finite backward expectation through the same actual freely learned rows.
Source: the chronological conditional expectation recurrence, used only to prove exact forward contraction. -/
def markovBackward (transition : A → S → S → ℝ) : List A → (S → ℝ) → S → ℝ
  | [], code => code
  | token :: rest, code => fun start =>
      ∑ next, stateRow (transition token start) next * markovBackward transition rest code next

omit [Nonempty S] in
/-- An arbitrary endpoint statistic's full path expectation equals the small backward recurrence exactly.
Source: complete path probabilities, physical endpoint indexing and finite first-state/suffix sum decomposition. -/
theorem conditionalStatePath_mean (transition : A → S → S → ℝ) (start : S)
    (tokens : List A) (code : S → ℝ) :
    (∑ path : Fin tokens.length → S, conditionalStatePath transition start tokens path *
      code (statePathEnd start tokens.length path)) = markovBackward transition tokens code start := by
  induction tokens generalizing start with
  | nil =>
      change (∑ _ : Fin 0 → S, (1 : ℝ) * code start) = code start
      simpa only [one_mul] using (Fintype.sum_unique (fun _ : Fin 0 → S => (1 : ℝ) * code start))
  | cons token rest ih =>
      have hsplit : (∑ path : Fin (token :: rest).length → S,
          conditionalStatePath transition start (token :: rest) path *
            code (statePathEnd start (token :: rest).length path)) =
          ∑ head, ∑ tail : Fin rest.length → S,
            stateRow (transition token start) head * conditionalStatePath transition head rest tail *
              code (statePathEnd head rest.length tail) := by
        convert sum_statePath_cons rest.length
          (fun path => conditionalStatePath transition start (token :: rest) path *
            code (statePathEnd start (rest.length + 1) path)) using 1
        simp only [conditionalStatePath, Fin.cons_zero, Fin.tail_cons, statePathEnd_cons]
      rw [hsplit]
      simp_rw [mul_assoc, ← Finset.mul_sum, ih]
      rfl

omit [Nonempty S] in
/-- Forward state propagation and backward endpoint expectation are exactly dual compact computations.
Source: the same learned matrix rows, with finite sum exchange at each chronological raw token. -/
theorem markovPropagate_backward (transition : A → S → S → ℝ) (tokens : List A)
    (mass code : S → ℝ) :
    (∑ start, mass start * markovBackward transition tokens code start) =
      ∑ last, markovPropagate transition tokens mass last * code last := by
  induction tokens generalizing mass with
  | nil => rfl
  | cons token rest ih =>
      change (∑ start, mass start * ∑ next,
        stateRow (transition token start) next * markovBackward transition rest code next) =
          ∑ last, markovPropagate transition rest (markovAdvance (transition token) mass) last * code last
      rw [← ih]
      simp only [markovAdvance, Finset.mul_sum, Finset.sum_mul, mul_assoc]
      exact Finset.sum_comm

/-- The complete initial-state/path model is normalized for every unrestricted learned initial and transition table.
Source: actual initial softmax combined with the derived normalized complete conditional paths. -/
theorem initialStatePath_sum (initial : S → ℝ) (transition : A → S → S → ℝ) (tokens : List A) :
    (∑ start, ∑ path : Fin tokens.length → S,
      stateRow initial start * conditionalStatePath transition start tokens path) = 1 := by
  simp only [← Finset.mul_sum, conditionalStatePath_sum, mul_one, stateRow_sum]

omit [Nonempty S] in
/-- The actual compact inference state expectation equals the same full path distribution trained by observed-state NLL.
Source: exact conditional-path contraction and forward/backward duality, with no inference state labels. -/
theorem markovRun_path_mean (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (code : S → ℝ) :
    (∑ start, ∑ path : Fin tokens.length → S,
      stateRow initial start * conditionalStatePath transition start tokens path *
        code (statePathEnd start tokens.length path)) =
      ∑ last, markovRun initial transition tokens last * code last := by
  simp only [mul_assoc, ← Finset.mul_sum, conditionalStatePath_mean]
  exact markovPropagate_backward transition tokens (stateRow initial) code

omit [Nonempty S] in
/-- Each real inference coordinate is exactly the probability of that endpoint in the complete chronological path model.
Source: indicator endpoint expectation of the same learned normalized path distribution, without a supplied encoder state. -/
theorem markovRun_endpoint (initial : S → ℝ) (transition : A → S → S → ℝ)
    (tokens : List A) (target : S) :
    (∑ start, ∑ path : Fin tokens.length → S,
      stateRow initial start * conditionalStatePath transition start tokens path *
        (if statePathEnd start tokens.length path = target then 1 else 0)) =
      markovRun initial transition tokens target := by
  rw [markovRun_path_mean initial transition tokens (fun last => if last = target then 1 else 0)]
  simp only [mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- A six-state encoder's path expectation is computed without storing its exponentially many histories.
Source: arbitrary finite shared initial/transition tables, with the actual three-token compact inference. -/
example (initial : Fin 6 → ℝ) (transition : Fin 3 → Fin 6 → Fin 6 → ℝ) :
    (∑ start, ∑ path : Fin 3 → Fin 6,
      stateRow initial start * conditionalStatePath transition start [0, 1, 2] path *
        (if statePathEnd start 3 path = 4 then (1 : ℝ) else 0)) =
      ∑ last, markovRun initial transition [0, 1, 2] last * (if last = 4 then 1 else 0) := by
  convert markovRun_path_mean initial transition [0, 1, 2] (fun last => if last = 4 then 1 else 0)

end
end Transformer.GPTMini.Convex.Structured
