module

public import Transformer.Phasor.SharedValues

/-!
# Causal prefix addresses for the shared value residual

Source: the prefix-address comparison in `experiments/phasor_value_transport/README.md`.
A single sum-address table can use the preceding prefix in place of the
preceding token. The two candidate folds share the trained parameters and the
solver of the pair-address model. Each fold uses only preceding inputs.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {D d H F p : ℕ}

/-- Two candidate prefix summaries on the natural representatives of tokens. -/
inductive PrefixFold
  | xor | sum
  deriving DecidableEq

/-- The empty prefix has value zero. Reduction to the vocabulary is done at
lookup, so bits of an intermediate XOR are retained even above the modulus. -/
def PrefixFold.word (kind : PrefixFold) (tokens : ℕ → Fin p) : ℕ → ℕ
  | 0 => 0
  | i + 1 =>
    let a := kind.word tokens i
    match kind with
    | .xor => a ^^^ (tokens i).val
    | .sum => a + (tokens i).val

/-- For arbitrary lengths, either scan can process two consecutive chunks
separately and combine their states. This includes every bit plane of a
multisymbol alphabet, with no distinguished input or output tokens. -/
theorem PrefixFold.word_split (kind : PrefixFold) (xs : ℕ → Fin p) (m n : ℕ) :
    kind.word xs (m+n) =
      match kind with
      | .xor => kind.word xs m ^^^ kind.word (fun j => xs (m+j)) n
      | .sum => kind.word xs m + kind.word (fun j => xs (m+j)) n := by
  cases kind <;> induction n with
  | zero => simp [word]
  | succ n ih => simp [word, ih, Nat.xor_assoc, Nat.add_assoc]

/-- Count tokens whose natural representative has bit `k` set. -/
def bitOccurrences (xs : ℕ → Fin p) (k : ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => bitOccurrences xs k n + if (xs n).val.testBit k then 1 else 0

/-- At any length and for every bit plane, the XOR scan represents the parity
of that bit's occurrences across the entire prefix. -/
theorem PrefixFold.xor_testBit (xs : ℕ → Fin p) (k n : ℕ) :
    ((PrefixFold.xor.word xs n).testBit k = true) ↔ bitOccurrences xs k n % 2 = 1 := by
  induction n with
  | zero => simp [word, bitOccurrences]
  | succ n ih =>
    cases earlier : (PrefixFold.xor.word xs n).testBit k <;>
      cases last : (xs n).val.testBit k <;>
      simp [word, bitOccurrences, Nat.testBit_xor, earlier, last] at * <;> omega

/-- The summary depends only on tokens strictly before the reading position. -/
theorem PrefixFold.word_congr (kind : PrefixFold) (xs ys : ℕ → Fin p) (i : ℕ)
    (h : ∀ j < i, xs j = ys j) : kind.word xs i = kind.word ys i := by
  induction i with
  | zero => rfl
  | succ i ih =>
    have earlier := ih (fun j hj => h j (by omega))
    have last := h i (by omega)
    cases kind <;> simp only [word, earlier, last]

/-- At position zero use the token itself, as in the original pair read. -/
def PrefixFold.left [NeZero p] (kind : PrefixFold) (tokens : ℕ → Fin p) : ℕ → Fin p
  | 0 => tokens 0
  | i + 1 => Fin.ofNat p (kind.word tokens (i + 1))

/-- One of the existing nine power-sum tables changes its left address.
The chosen exponents are a single architecture choice at every position. -/
def PrefixFold.address [NeZero p] (kind : PrefixFold) (tokens : ℕ → Fin p)
    (slot : Fin 3 × Fin 3) (i : ℕ) (k : PairKey) : Fin p :=
  if k = .sumPowers slot.1 slot.2 then k.address (kind.left tokens i) (tokens i)
  else k.address (tokens (i-1)) (tokens i)

/-- The trainable tables and the order of their contributions are shared with
the pair read. Only the indices depend on the prefix. -/
noncomputable def PairTables.prefixRead [NeZero p] (T : PairTables D p)
    (kind : PrefixFold) (slot : Fin 3 × Fin 3) (tokens : ℕ → Fin p) (i : ℕ) : EucSpace D :=
  T.right (tokens i) + ∑ k, T.values k (kind.address tokens slot i k)

/-- Tied score of the prefix-address candidate at an arbitrary position. -/
noncomputable def Model.prefixLogit [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (kind : PrefixFold) (slot : Fin 3 × Fin 3) (gain eps scale : ℝ)
    (tokens : ℕ → Fin p) (L i : ℕ) (c : Fin p) : ℝ :=
  let a := tokens (i-1)
  let b := tokens i
  let w := Perspective.softmaxWeight (M.logits eps scale tokens L (i-1))
  ⟪M.embedding c, M.feature eps scale tokens L i + gain • GPTMini.rmsNormEps eps
    (transportedValue M.embedding w (a*b) + T.prefixRead kind slot tokens i)⟫

/-- Both prefix folds give the identical pair address at both positions of
every two-token input, for all vocabularies and every table. -/
theorem PrefixFold.address_eq_pair [NeZero p] (kind : PrefixFold) (xs : ℕ → Fin p)
    (slot : Fin 3 × Fin 3) (i : ℕ) (hi : i ≤ 1) (k : PairKey) :
    kind.address xs slot i k = k.address (xs (i-1)) (xs i) := by
  have same : kind.left xs i = xs (i-1) := by
    rcases Nat.le_one_iff_eq_zero_or_eq_one.mp hi with rfl | rfl <;>
      cases kind <;> simp [left, word]
  simp [address, same]

/-- The complete logit as a function of all trainable parameters agrees with
the incumbent on one- and two-token inputs. This includes the NV11 objective. -/
theorem Model.prefixLogit_eq_shared [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (kind : PrefixFold) (slot : Fin 3 × Fin 3) (gain eps scale : ℝ)
    (xs : ℕ → Fin p) (L i : ℕ) (hi : i ≤ 1)
    (c : Fin p) :
    M.prefixLogit T kind slot gain eps scale xs L i c = M.sharedLogit T gain eps scale xs L i c := by
  simp only [Model.prefixLogit, Model.sharedLogit, Model.sharedFeature, PairTables.prefixRead,
    PairTables.read, kind.address_eq_pair xs slot i hi]

/-- A concrete NV11 read at position one instantiates the two-token result. -/
example (M : Model 128 32 4 171 97) (T : PairTables 128 97) (xs : ℕ → Fin 97)
    (L : ℕ) (c : Fin 97) :
    M.prefixLogit T .xor (0,0) (1/4) 1 1 xs L 1 c = M.sharedLogit T (1/4) 1 1 xs L 1 c :=
  M.prefixLogit_eq_shared T .xor (0,0) _ _ _ xs L 1 (by decide) c

/-- A differing third token has no effect on the two-token prefix summary. -/
example (kind : PrefixFold) (xs : ℕ → Fin 97) :
    kind.word xs 2 = kind.word (fun j => if j < 2 then xs j else 0) 2 :=
  kind.word_congr _ _ 2 (by intro j hj; simp [hj])

example (kind : PrefixFold) (xs : ℕ → Fin 16) (k : PairKey) :
    kind.address xs (2,2) 0 k = k.address (xs 0) (xs 0) :=
  kind.address_eq_pair xs (2,2) 0 (by decide) k

end Transformer.Phasor
