module

public import Transformer.Grokking.Protocol.Gradient
public import Transformer.Grokking.DivisionOrbits.FiniteCells
public import Mathlib.Algebra.Field.ZMod
public import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Division modulo 97 as the source lists, tokenizes and supervises it

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, the
binary operation `x / y (mod p)` for `0 ≤ x < p`, `0 < y < p` with `p = 97`, and "a transformer
with 2 layers, width 128, and 4 attention heads"; openai/grok at 3d64b1d8, `grok/data.py`:
`ArithmeticTokenizer`, `ArithmeticDataset._make_binary_operation_data` and
`ArithmeticIterator`; `experiments/grok_division/PROTOCOL.md`, *The task and its tokens* and
*The loss and the accuracy*.

The decoder has 239 tokens, width 128, 4 heads of width 32, hidden width 512, 2 blocks, and
reads 6 tokens (`paperSizes`). An equation (`Equation`) is the quotient `c` and the denominator
`b = 1, …, 96`, the latter kept as `b - 1`; its numerator is `b c mod 97` (`numerator`), so the
numerator `0` occurs and the denominator `0` does not. The source lists the `97 · 96 = 9312`
equations with the quotient in the outer loop (`equations`, a bijection). In the field of 97
elements the operands divide to the quotient (`inDivisionCell_operands`), which ties the list
to `DivisionOrbits`.

The tokens are `<|eos|>` (`0`), `=` (`1`), `/` (`10`) and `22 + r` for the residue `r`. The
model reads `<|eos|> a / b = c` (`read`). The source trains on the two tokens to the right of
`=`: the answer after `=`, and `<|eos|>` after the answer (`supervised`), so a batch of `n`
equations has `2 n` targets (`targetCount_supervised`). The paper's wording, "only on the
answer part", is the answer alone (`supervisedAnswer`, `n` targets): the source's loss is not
that one, and the targets below follow the source.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Grokking.DivisionOrbits

/-- **The sizes of the paper's decoder.** -/
abbrev paperSizes : Sizes := ⟨239, 128, 4, 32, 512, 2, 6⟩

/-- **An equation** `a / b = c` modulo 97: the quotient `c`, and the denominator `b` as
`b - 1`. -/
abbrev Equation := Fin 97 × Fin 96

/-- The denominator `b`, from `1` to `96`. -/
def denominator (x : Equation) : Fin 97 := ⟨(x.2 : ℕ) + 1, by have := x.2.isLt; omega⟩

/-- The numerator `a = b c mod 97`. -/
def numerator (x : Equation) : Fin 97 :=
  ⟨((x.2 : ℕ) + 1) * (x.1 : ℕ) % 97, Nat.mod_lt _ (by norm_num)⟩

/-- **The equations in the order of the source's list**: the quotient in the outer loop. -/
def equations : Fin (97 * 96) → Equation := finProdFinEquiv.symm

/-- Every equation is listed once. -/
lemma equations_bijective : Function.Bijective equations := finProdFinEquiv.symm.bijective

/-- **The operands divide to the quotient** in the field of 97 elements, and the denominator
is not zero: the equation is in the cell of its answer of `DivisionOrbits`. -/
theorem inDivisionCell_operands (x : Equation) :
    InDivisionCell ((x.1 : ℕ) : ZMod 97)
      (((numerator x : ℕ) : ZMod 97), ((denominator x : ℕ) : ZMod 97)) := by
  have hd : ((denominator x : ℕ) : ZMod 97) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    have := x.2.isLt
    exact Nat.not_dvd_of_pos_of_lt (Nat.succ_pos _) (by simp only [denominator]; omega)
  have h := generated_in_cell ((x.1 : ℕ) : ZMod 97) _ hd
  simpa [divisionInput, numerator, denominator] using h

/-- The token `<|eos|>`: index `0` of the tokenizer's list. -/
def eosToken : Fin 239 := ⟨0, by norm_num⟩

/-- The token `=`: index `1`. -/
def eqToken : Fin 239 := ⟨1, by norm_num⟩

/-- The token `/`: index `10`, among the operators in sorted order. -/
def divToken : Fin 239 := ⟨10, by norm_num⟩

/-- The token of a residue. -/
def residueToken (r : Fin 97) : Fin 239 := ⟨22 + r, by have := r.isLt; omega⟩

lemma residueToken_injective : Function.Injective residueToken := fun r s h => by
  have := congrArg Fin.val h
  simp only [residueToken] at this
  exact Fin.ext (by omega)

/-- **The tokens read**: `<|eos|> a / b = c`. -/
def read (x : Equation) : Fin 6 → Fin 239 :=
  ![eosToken, residueToken (numerator x), divToken, residueToken (denominator x), eqToken,
    residueToken x.1]

/-- **Equations are read as different rows of tokens.** -/
theorem read_injective : Function.Injective read := fun x y h => by
  have h3 := residueToken_injective (congrFun h 3)
  have h5 := residueToken_injective (congrFun h 5)
  have hv := congrArg Fin.val h3
  simp only [denominator] at hv
  exact Prod.ext h5 (Fin.ext (by omega))

/-- **The source's supervision of an equation**: after `=` the answer, and after the answer
the closing `<|eos|>`. -/
def supervised (x : Equation) : Example paperSizes :=
  ⟨read x, [((4 : Fin 6), residueToken x.1), ((5 : Fin 6), eosToken)]⟩

/-- The supervision of the paper's wording: the answer alone. -/
def supervisedAnswer (x : Equation) : Example paperSizes :=
  ⟨read x, [((4 : Fin 6), residueToken x.1)]⟩

/-- **The source's mean is over two targets for each equation.** -/
theorem targetCount_supervised (B : List Equation) :
    targetCount (B.map supervised) = 2 * B.length := by
  induction B with
  | nil => rfl
  | cons x B ih =>
    simp only [targetCount, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    rw [ih]
    simp only [supervised, List.length_cons, List.length_nil]
    ring

/-- The paper's wording has one target for each equation. -/
theorem targetCount_supervisedAnswer (B : List Equation) :
    targetCount (B.map supervisedAnswer) = B.length := by
  induction B with
  | nil => rfl
  | cons x B ih =>
    simp only [targetCount, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    rw [ih]
    simp only [supervisedAnswer, List.length_cons, List.length_nil]
    ring

/-- **The logits of the answer**: those at the position of `=`. -/
noncomputable def answerLogits (θ : Param paperSizes → ℝ) (x : Equation) : Fin 239 → ℝ :=
  logits θ (relu paperSizes) (read x) (4 : Fin 6)

/-- The logits of the closing token: those at the position of the answer. -/
noncomputable def closingLogits (θ : Param paperSizes → ℝ) (x : Equation) : Fin 239 → ℝ :=
  logits θ (relu paperSizes) (read x) (5 : Fin 6)

/-- **The gradient of an update**: what the source's backward pass returns for the batch's
supervised equations. -/
noncomputable def divisionGrad (B : List Equation) (θ : Param paperSizes → ℝ) :
    Param paperSizes → ℝ :=
  sourceGrad (B.map supervised) θ

/-! The witnesses: `3 / 5 = 20` modulo 97, since `5 · 20 = 100 = 3`, read as the tokens
`0, 25, 10, 27, 1, 42`; and the numerator `0`, which is the quotient `0`. -/

example : numerator (20, 4) = 3 ∧ denominator (20, 4) = 5 := by decide

example : read (20, 4) = ![0, 25, 10, 27, 1, 42] := by decide

example : numerator (0, 4) = 0 := by decide

example : equations ⟨96 * 20 + 4, by norm_num⟩ = (20, 4) := by decide

example := inDivisionCell_operands (20, 4)

example := targetCount_supervised [(20, 4), (0, 4)]

end Transformer.Grokking.Protocol
