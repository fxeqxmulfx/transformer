/-
# RASP: sequences, selectors, and the three core operations

Weiss, Goldberg, Yahav — arXiv:2106.06981v2, "Thinking Like Transformers"
(ICML 2021).

The paper introduces a programming language, RASP, whose primitives are meant
to be exactly the primitives of a transformer encoder: element-wise maps for
the feed-forward sublayers, and a `select` / `aggregate` pair for an attention
head.  It carries no theorem environments; its mathematical content is the
semantics of those primitives (§3) and what a program's shape implies about
the number of layers and heads needed to run it (§3.1, §4).

A RASP computation over a length-`n` input manipulates only sequences of
length `n` and matrices of size `n × n` (§2).  The paper's own objects are
the *functions* producing those — s-ops and selectors — but it states that
"the conceptual model to bear in mind is that of operations over sequences
and selection matrices", and that is what this file fixes: `n` is a
parameter, and every operation acts on the sequences themselves.  The
functional layer is recovered in `RASP.Compilation`, which needs the syntax
of a program and not only its value.

**A note on indices.**  §2 says the selection matrix satisfies
`S[i][j] = p(k[i], q[j])`, but both the worked example directly below it and
the definition of `aggregate` ("each position `s[i]` combines the values in
`v` according to row `i` in `S`") require the transpose: rows are queries and
columns are keys.  Reading `sel([0,1,2],[1,2,3],<)` with the displayed matrix
settles it — row `0` is `[T,F,F]`, which is `0 < 1, 1 < 1, 2 < 1`, i.e. all
keys against query `q₀ = 1`.  We take the convention of the example and of
`aggregate`: `S i j = p (k j) (q i)`.
-/

import Transformer.Basic

namespace Transformer
namespace RASP

variable {n : ℕ} {α β γ : Type*}

/-- A sequence of length `n`: the value of an s-op at one input (§3).  All
sequences in a RASP computation have the length of the input. -/
abbrev Seq (n : ℕ) (α : Type*) : Type _ := Fin n → α

/-- A selection matrix on `n` positions: `S i j` says that query position `i`
attends to key position `j` (§3).  This is the value of a selector at one
input, and compiles to the attention pattern of one head (§3.1). -/
abbrev Selector (n : ℕ) : Type := Fin n → Fin n → Bool

/-- `select`: from a key sequence, a query sequence and a predicate on pairs,
the matrix `S i j = p (k j) (q i)` (§3, and the note on indices above). -/
def sel (k : Seq n α) (q : Seq n β) (p : α → β → Bool) : Selector n :=
  fun i j => p (k j) (q i)

/-- The keys selected by query position `i`: row `i` of the selection
matrix, as a finite set of positions (§3). -/
def selected (S : Selector n) (i : Fin n) : Finset (Fin n) :=
  Finset.univ.filter (fun j => S i j = true)

/-- `aggregate` with default `d`: at each position, the average of the values
selected by that row; `d` where the row selects nothing (§3, and its
footnote on the default). -/
noncomputable def aggregate (S : Selector n) (v : Seq n ℝ) (d : ℝ := 0) : Seq n ℝ :=
  fun i =>
    if (selected S i).card = 0 then d
    else (∑ j ∈ selected S i, v j) / ((selected S i).card : ℝ)

/-- `aggregate` on values that are not numbers.  The footnote to §3 says that
where exactly one position is selected RASP passes that value through
directly, without averaging; that is the only case in which a non-numeric
aggregation is defined, and the default is returned elsewhere.  This is what
`reverse = aggregate(flip, tokens)` uses. -/
noncomputable def aggregateOne (S : Selector n) (v : Seq n α) (d : α) : Seq n α :=
  fun i =>
    if h : (selected S i).card = 1 then v (Finset.card_eq_one.mp h).choose else d

/-- `selector_width`: the number of positions each row selects (§3).  It is
not a primitive — `RASP.SelectorWidth` derives it from `select` and
`aggregate`, which is how it compiles. -/
def selectorWidth (S : Selector n) : Seq n ℝ := fun i => ((selected S i).card : ℝ)

/-! ### Built-in s-ops and selector algebra -/

/-- The built-in s-op `indices`: `indices("hi") = [0,1]` (§3). -/
def indices (n : ℕ) : Seq n ℝ := fun i => ((i : ℕ) : ℝ)

/-- The built-in s-op `length`, as a sequence.  `RASP.Basic` proves that it
is the value of the RASP program `1/aggregate(select_all, indicator(indices
== 0))` given in §3.1, so it need not be a primitive. -/
def length (n : ℕ) : Seq n ℝ := fun _ => (n : ℝ)

/-- `select_all = select(1,1,==)`: every position attends to every position
(§3). -/
def selectAll (n : ℕ) : Selector n := fun _ _ => true

/-- `select_eq(indices, 0)`: every position attends to position `0` alone.
This is the selector `selector_width` adds to its argument (Figure 8). -/
def selectZero (n : ℕ) : Selector n := fun _ j => decide ((j : ℕ) = 0)

/-- `indicator(indices == 0)`, written `light0` in Figure 8. -/
def light0 (n : ℕ) : Seq n ℝ := fun j => if (j : ℕ) = 0 then 1 else 0

/-- `indicator`: a boolean sequence read as `0`/`1` (§3). -/
def ind (b : Seq n Bool) : Seq n ℝ := fun i => if b i then 1 else 0

/-- Element-wise conjunction of selectors (§3, "selector manipulations"). -/
def Selector.and' (S T : Selector n) : Selector n := fun i j => S i j && T i j

/-- Element-wise disjunction of selectors (§3, "selector manipulations"). -/
def Selector.or' (S T : Selector n) : Selector n := fun i j => S i j || T i j

/-- Element-wise negation of a selector (§3, "selector manipulations"). -/
def Selector.not' (S : Selector n) : Selector n := fun i j => !S i j

end RASP
end Transformer
