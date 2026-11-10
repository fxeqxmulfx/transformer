module

public import Transformer.Grokking.FlatSpectrum.Defs

/-!
# Associative recall as a kernel: definitions

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), item AR1,
for the task of Arora et al., arXiv:2312.04927v1, §3, Definition `def: general-AR`, and its
Appendix `app:retnet-proof`: "The AR problem takes key-value pairs … along with a query `q`
appended at the end as input and the goal is to output `v_i` if `q = k_i`".

A context binds each key of `κ` to a value of `ν`: a dictionary `π : κ → ν`. The answer to the
query `q` is `π q`, the evaluation map. For a readout `φ` of the answer, the **recall kernel** is
`K(q, π) = φ(π q)` (`recallKernel`), with a row for each queried key and a column for each
dictionary: `|κ|` rows and `|ν|^|κ|` columns. It is the task kernel `φ(f(a, b))` of the note for
the operation `f(q, π) = π q`, whose two arguments have different types.

A model reads the context before the query. Its readout `F q x` at the query has **cut width**
`d` (`HasCutWidth`) when `F q x = g q + ∑_{r < d} a_r(q) b_r(x)`: beyond a term of the query
alone, `d` numbers `b_r(x)` are all that crosses from the context `x` to the query `q`, and the
query combines them linearly.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

/-- **The recall kernel** `K(q, π) = φ(π q)` of a readout `φ` of the value that the dictionary
`π` binds to the queried key `q` (`review/flat_spectrum.md`, AR1). -/
def recallKernel {κ ν : Type*} (φ : ν → ℂ) : Matrix κ (κ → ν) ℂ :=
  Matrix.of fun q π => φ (π q)

/-- The readout `F` of a query `q` after a context `x` has **cut width** `d`:
`F q x = g q + ∑_{r < d} a_r(q) b_r(x)` (`review/flat_spectrum.md`, AR1). -/
def HasCutWidth {Q C : Type*} (d : ℕ) (F : Q → C → ℝ) : Prop :=
  ∃ (g : Q → ℝ) (a : Q → Fin d → ℝ) (b : C → Fin d → ℝ),
    ∀ q x, F q x = g q + ∑ r, a q r * b x r

end Transformer.Grokking.FlatSpectrum.Recall
