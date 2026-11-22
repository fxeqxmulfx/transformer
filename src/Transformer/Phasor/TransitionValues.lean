module

public import Transformer.Phasor.SuccessorDisplacement

/-!
# A value table indexed by changes in the causal prefix

Source: the transition-count experiment in `experiments/phasor_value_transport/README.md`.
The summary counts changes of token identity on completed edges. It shifts one
existing table address and applies the same read to any input vocabulary.
-/

@[expose] public section

namespace Transformer.Phasor

variable {α β : Type*} {D d H F p : ℕ}

/-- Number of changes across the first `n` adjacent pairs, ending at token `n`. -/
def transitionCount [DecidableEq α] (xs : ℕ → α) : ℕ → ℕ
  | 0 => 0
  | n+1 => transitionCount xs n + if xs n = xs (n+1) then 0 else 1

/-- Summaries compose across any two consecutive groups of edges. The
shared endpoint supplies the left token of the second group. -/
theorem transitionCount_split [DecidableEq α] (xs : ℕ → α) (m n : ℕ) :
    transitionCount xs (m+n) =
      transitionCount xs m + transitionCount (fun i => xs (m+i)) n := by
  induction n with
  | zero => simp [transitionCount]
  | succ n ih => simp [transitionCount, ih, Nat.add_assoc]

/-- The streaming counter measures changes in identity in every alphabet.
An injective renaming, even to another vocabulary, preserves it. -/
theorem transitionCount_map [DecidableEq α] [DecidableEq β] (f : α → β)
    (hf : Function.Injective f) (xs : ℕ → α) (n : ℕ) :
    transitionCount (fun i => f (xs i)) n = transitionCount xs n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [transitionCount, hf.eq_iff, ih]

/-- At each length the counter is zero precisely when the whole observed
segment is constant. In particular, longer repeated runs contribute no edges. -/
theorem transitionCount_eq_zero [DecidableEq α] (xs : ℕ → α) (n : ℕ) :
    transitionCount xs n = 0 ↔ ∀ i ≤ n, xs i = xs 0 := by
  induction n with
  | zero => simp [transitionCount]
  | succ n ih =>
    constructor
    · intro h i hi
      have earlier : transitionCount xs n = 0 := by
        unfold transitionCount at h
        omega
      have last : xs n = xs (n+1) := by
        by_contra hne
        simp [transitionCount, hne] at h
      by_cases le : i ≤ n
      · exact ih.mp earlier i le
      · have eq : i = n+1 := by omega
        rw [eq, ← last]
        exact ih.mp earlier n (by omega)
    · intro h
      have earlier := ih.mpr (fun i hi => h i (by omega))
      have last := (h n (by omega)).trans (h (n+1) (by omega)).symm
      simp only [transitionCount, last, ite_true, earlier, Nat.add_zero]

/-- Use the count to translate the address of the squared-product table
modulo the vocabulary size. All embeddings, the other tables, the gain and
the backbone remain trainable under the same parameterization. -/
def PairTables.withTransitions [NeZero p] (T : PairTables D p) (shift : Fin p) : PairTables D p :=
  { T with values := fun k a => T.values k (if k = .productPowers 1 1 then a + shift else a) }

/-- At position `i` the counter ends at `i-1`. The new table read composes
with any fixed strength of the tied successor displacement. -/
noncomputable def Model.transitionLogit [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (strength gain eps scale : ℝ) (xs : ℕ → Fin p)
    (L i : ℕ) (c : Fin p) : ℝ :=
  M.displacedSuccessorLogit (T.withTransitions (Fin.ofNat p (transitionCount xs (i-1))))
    strength gain eps scale xs L i c

/-- The entire logit is the same parameter function on any prefix whose
completed edges are constant, including every position of a two-token task. -/
theorem Model.transitionLogit_eq_of_constant [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (strength gain eps scale : ℝ) (xs : ℕ → Fin p)
    (L i : ℕ) (hc : ∀ j ≤ i-1, xs j = xs 0) (c : Fin p) :
    M.transitionLogit T strength gain eps scale xs L i c =
      M.displacedSuccessorLogit T strength gain eps scale xs L i c := by
  have zero := (transitionCount_eq_zero xs (i-1)).mpr hc
  simp [transitionLogit, zero, PairTables.withTransitions]

example (xs : ℕ → ℕ) (n : ℕ) :
    transitionCount (fun i => xs i + 17) n = transitionCount xs n :=
  transitionCount_map (fun a : ℕ => a + 17) (by intro a b h; dsimp at h; omega) xs n

example : transitionCount (fun i => (![0,0,2,2,0,1] : Fin 6 → Fin 3) (Fin.ofNat 6 i)) 5 = 3 := by
  decide

example (M : Model 128 32 4 171 97) (T : PairTables 128 97) (xs : ℕ → Fin 97)
    (L : ℕ) (c : Fin 97) :
    M.transitionLogit T 1 (1/2) 1 1 xs L 1 c =
      M.displacedSuccessorLogit T 1 (1/2) 1 1 xs L 1 c :=
  M.transitionLogit_eq_of_constant T _ _ _ _ xs L 1 (by simp) c

end Transformer.Phasor
