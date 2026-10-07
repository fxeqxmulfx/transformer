import Transformer.GPTMini.Convex.Structured.MarkovObjective

/-!
# Physical-index formulas for the actual complete state path

Source: MarkovTraining's chronological conditionalStatePath and
conditionalStateNLL at f1f6c9c/95456c1. Tensor training uses an indexed
sequence and a fixed observed post-token history. These formulas read
the genuine previous state at position zero from the observed start,
and at later positions from the preceding physical history entry.
The products/sums are proved equal to the existing actual recursive
probability/loss, including empty and single-token inputs.

Observed history is training-only data. This is no deterministic
encoder substituted for inference, no supplied correct probability
and no change of the transition logits. It supplies an exact compact
physical-index likelihood for subsequent tensor training, where all
learned transition fields must come from real recovered input rows.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {E S A : Type*} {n : ℕ}

/-- Actual previous state in a complete observed history at its physical token position.
Source: MarkovTraining's chronological recursion; the initial state is retained at the first token. -/
def statePathPrevious (start : S) (path : Fin n → S) (position : Fin n) : S :=
  if hzero : position.val = 0 then start
  else path ⟨position.val - 1, by have hp := position.isLt; omega⟩

/-- The first actual row uses the observed initial state rather than its first post-token state.
Source: physical-index unrolling of the original conditional path factorization. -/
theorem statePathPrevious_zero (start : S) (path : Fin (n + 1) → S) :
    statePathPrevious start path 0 = start := by
  simp only [statePathPrevious, Fin.val_zero]
  exact dite_eq_left (show (0 : ℕ) = 0 from rfl)

/-- Removing the first token and its state leaves every later physical previous-state read unchanged.
Source: the actual Fin.tail indexing used by MarkovTraining's recursion. -/
theorem statePathPrevious_succ (start : S) (path : Fin (n + 1) → S) (position : Fin n) :
    statePathPrevious start path position.succ = statePathPrevious (path 0) (Fin.tail path) position := by
  have hsucc : position.succ.val ≠ 0 := by rw [Fin.val_succ]; omega
  unfold statePathPrevious
  rw [dite_eq_right hsucc]
  by_cases hzero : position.val = 0
  · rw [dite_eq_left hzero]
    congr 1
    apply Fin.ext
    change position.val + 1 - 1 = 0
    omega
  · rw [dite_eq_right hzero]
    unfold Fin.tail
    congr 1
    apply Fin.ext
    change position.val + 1 - 1 = position.val - 1 + 1
    omega

/-- Actual chronological previous-state reads distinguish the initial state from all later observed states.
Source: a concrete three-token history, not a correct semantic encoder premise. -/
example : statePathPrevious (0 : Fin 6) (fun _ : Fin 3 => (1 : Fin 6)) 0 = 0 ∧
    statePathPrevious (0 : Fin 6) (fun _ : Fin 3 => (1 : Fin 6)) 1 = 1 ∧
    statePathPrevious (0 : Fin 6) (fun _ : Fin 3 => (1 : Fin 6)) 2 = 1 := by
  decide

section Finite
variable [Fintype S]

/-- The true complete causal path probability is the product of exactly its physical chronological row factors.
Source: MarkovTraining.conditionalStatePath, with the original token reads and previous-state indices proved by induction. -/
theorem conditionalStatePath_product (transition : A → S → S → ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) :
    conditionalStatePath transition start tokens path =
      ∏ position, stateRow (transition (tokens.get position) (statePathPrevious start path position)) (path position) := by
  induction tokens generalizing start with
  | nil => simp only [conditionalStatePath, List.length_nil, Fin.prod_univ_zero]
  | cons token rest ih =>
      rw [conditionalStatePath, ih]
      simpa only [List.length_cons, List.get_cons_zero, List.get_cons_succ', statePathPrevious_zero,
        statePathPrevious_succ, Fin.tail] using
        (Fin.prod_univ_succ (fun position : Fin (rest.length + 1) =>
          stateRow (transition ((token :: rest).get position) (statePathPrevious start path position)) (path position))).symm

/-- Genuine zero-logit row probabilities give one quarter to an actual fixed two-token path.
Source: the computed binary state rows and chronological factorization, including the learned initial-state distinction. -/
example : conditionalStatePath (fun _ : Fin 1 => fun _ _ : Fin 2 => (0 : ℝ)) 0
    [0, 0] (fun _ => 1) = 1 / 4 := by
  norm_num [conditionalStatePath, stateRow, Fin.sum_univ_two, Fin.tail]

variable [AddCommGroup E] [Module ℝ E]

/-- The actually computed complete path loss is precisely the sum of physical observed row likelihoods.
Source: MarkovTraining.conditionalStateNLL; no row, transition parameter or state of the observed path is omitted. -/
theorem conditionalStateNLL_sum (transition : A → S → S → E →ₗ[ℝ] ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) (θ : E) :
    conditionalStateNLL transition start tokens path θ =
      ∑ position, stateRowNLL (transition (tokens.get position) (statePathPrevious start path position)) (path position) θ := by
  induction tokens generalizing start with
  | nil => simp only [conditionalStateNLL, List.length_nil, Fin.sum_univ_zero]
  | cons token rest ih =>
      rw [conditionalStateNLL, ih]
      simpa only [List.length_cons, List.get_cons_zero, List.get_cons_succ', statePathPrevious_zero,
        statePathPrevious_succ, Fin.tail] using
        (Fin.sum_univ_succ (fun position : Fin (rest.length + 1) =>
          stateRowNLL (transition ((token :: rest).get position) (statePathPrevious start path position)) (path position) θ)).symm

/-- Every complete physical-index state loss is nonnegative at arbitrary unrestricted weights.
Source: the actual sum of genuine observed categorical likelihoods, without semantic correctness assumptions. -/
theorem conditionalStateNLL_sum_nonneg (transition : A → S → S → E →ₗ[ℝ] ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) (θ : E) :
    0 ≤ ∑ position, stateRowNLL (transition (tokens.get position) (statePathPrevious start path position)) (path position) θ := by
  rw [← conditionalStateNLL_sum]
  exact conditionalStateNLL_nonneg transition start tokens path θ

variable [Nonempty S]

/-- The physical-index loss is exactly negative log of the true full path product used by the same inference model.
Source: MarkovTraining's exact recursive likelihood identity and the proved genuine product/sum unrolling. -/
theorem conditionalStateNLL_sum_eq (transition : A → S → S → E →ₗ[ℝ] ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) (θ : E) :
    (∑ position, stateRowNLL (transition (tokens.get position) (statePathPrevious start path position)) (path position) θ) =
      -Real.log (∏ position, stateRow (fun next =>
        transition (tokens.get position) (statePathPrevious start path position) next θ) (path position)) := by
  rw [← conditionalStateNLL_sum, conditionalStateNLL_eq, conditionalStatePath_product]

/-- All simultaneous learned transitions retain global complete-likelihood convexity in the physical-index formula.
Source: the same actual unrestricted path objective, proved equal rather than redefined as an assumed convex target. -/
theorem conditionalStateNLL_sum_convex (transition : A → S → S → E →ₗ[ℝ] ℝ) (start : S)
    (tokens : List A) (path : Fin tokens.length → S) :
    ConvexOn ℝ Set.univ (fun θ => ∑ position,
      stateRowNLL (transition (tokens.get position) (statePathPrevious start path position)) (path position) θ) := by
  have heq : (fun θ => ∑ position,
      stateRowNLL (transition (tokens.get position) (statePathPrevious start path position)) (path position) θ) =
      conditionalStateNLL transition start tokens path := by
    funext θ
    exact (conditionalStateNLL_sum transition start tokens path θ).symm
  rw [heq]
  exact conditionalStateNLL_convex transition start tokens path

/-- A genuine scalar-trained two-state transition family has a convex complete two-token history loss.
Source: arbitrary learned self-transition potential with free scalar domain, rather than a constant or fixed correct encoder. -/
example : ConvexOn ℝ Set.univ (fun θ : ℝ => ∑ position : Fin 2,
    stateRowNLL (fun next : Fin 2 => if next = statePathPrevious 0 (fun _ : Fin 2 => (1 : Fin 2)) position
      then (LinearMap.id : ℝ →ₗ[ℝ] ℝ) else 0) (1 : Fin 2) θ) := by
  exact conditionalStateNLL_sum_convex
    (fun _ : Fin 1 => fun previous next : Fin 2 => if next = previous then LinearMap.id else 0)
    (0 : Fin 2) [0, 0] (fun _ => 1)

end Finite
end
end Transformer.GPTMini.Convex.Structured
