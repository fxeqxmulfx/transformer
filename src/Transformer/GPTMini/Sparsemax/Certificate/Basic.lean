import Mathlib.Tactic.NormNum

/-!
# A finite certificate for modular-division predictions

Experimental extension of arXiv:2211.11052v1, §4, modular division, with
prime 193 and a frozen 25% training split. A record contains the numerator,
nonzero denominator, predicted numeral (193 denotes a non-numeral), and
whether EOS is correct. Kernel evaluation checks the supplied table; its
export from a PyTorch checkpoint is a separately measured trust boundary.
-/

namespace Transformer.GPTMini.Sparsemax.Certificate

/-- Lossless packing on the recorded bounds: numerator and denominator
are below 193, prediction is below 194, and EOS is a bit. Source: §4's
modular-division task, adapted to the frozen prime-193 experiment. -/
def pack (numerator denominator prediction eos : ℕ) : ℕ :=
  ((numerator * 193 + denominator) * 194 + prediction) * 2 + eos

/-- EOS bit of a packed §4 modular-division prediction record. -/
def eos (code : ℕ) : ℕ := code % 2

/-- Predicted numeral; 193 denotes any non-numeral token. Source: §4's
modular-division task, complete-RHS evaluation extension. -/
def prediction (code : ℕ) : ℕ := code / 2 % 194

/-- Nonzero denominator in a packed record of the §4 task extension. -/
def denominator (code : ℕ) : ℕ := code / 2 / 194 % 193

/-- Numerator in a packed record of the §4 task extension. -/
def numerator (code : ℕ) : ℕ := code / 2 / 194 / 193

/-- Bounds required by the prime-193 division task. Source: §4,
experiment adaptation; the final table is checked against these bounds. -/
def validInput (code : ℕ) : Bool :=
  decide (numerator code < 193 ∧ 0 < denominator code)

/-- Complete-RHS correctness: the predicted numeral solves `y*z = x`
modulo 193 and is followed by the correct EOS. Source: §4's division
task, exhaustive complete-RHS evaluation of the frozen experiment. -/
def correctRHS (code : ℕ) : Bool :=
  decide (prediction code < 193 ∧ eos code = 1 ∧
    denominator code * prediction code % 193 = numerator code)

/-- Number of supplied rows; batches keep kernel reduction shallow.
Source: §4's finite division evaluation, experiment adaptation. -/
def rowCount (batches : List (List ℕ)) : ℕ :=
  batches.foldl (fun count batch => count + batch.length) 0

/-- Number of correct complete answers in the supplied prediction table.
Source: §4's division accuracy, complete-RHS experiment adaptation. -/
def correctCount (batches : List (List ℕ)) : ℕ :=
  batches.foldl (fun count batch => count + batch.countP correctRHS) 0

/-- Every supplied record has a valid modular-division input. Source:
§4's finite division domain, prime-193 experiment adaptation. -/
def allInputsValid (batches : List (List ℕ)) : Bool :=
  batches.all (fun batch => batch.all validInput)

/-- Every supplied record predicts EOS correctly. Source: §4's division
task, complete-RHS evaluation extension. -/
def allEOSCorrect (batches : List (List ℕ)) : Bool :=
  batches.all (fun batch => batch.all (fun code => decide (eos code = 1)))

/-- Exact accuracy of the supplied finite table. Source: §4's test
accuracy, with no floating-point division or rounding in this definition. -/
def accuracy (batches : List (List ℕ)) : ℚ :=
  (correctCount batches : ℚ) / rowCount batches

end Transformer.GPTMini.Sparsemax.Certificate
