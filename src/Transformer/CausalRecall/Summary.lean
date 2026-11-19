module

public import Transformer.CausalRecall.Basic

/-!
# Causal models that read the past through a summary

In no paper. A layer (`Layer`) gives the position `i` with state `z` the new
state `combine i z (read zs)`, where `zs` are the states before it and the
summary `read zs` lies in a set `R`. The summary never sees the position's own
state, so a query placed there cannot steer what is read; what the position
does with the summary afterwards is arbitrary. A model (`Model`) embeds each
token with its position, applies its `L` layers in turn and reads an output
off every position.

Appending a position leaves the states before it unchanged
(`Layer.apply_append_singleton`, `run_append_singleton`), so the output at an
appended token depends only on that token, its index and the `L` summaries the
layers read from the context (`Model.outputs_append_singleton`). A model that
answers every key of every dictionary written in contexts of one length `n`
therefore has `|V|^|K| ≤ |R|^L` (`Model.card_pow_le`), and the counting of
`Transformer.CausalRecall.Basic` bounds the dictionaries it answers in part
(`Model.card_answered_mul_le`).
-/

@[expose] public section

namespace Transformer.CausalRecall

open Finset

variable {X Z R Y : Type*}

/-- A layer of a causal model: the position `i` with state `z`, after the states
`zs`, gets the state `combine i z (read zs)`. Its own state never enters `read`. -/
structure Layer (Z R : Type*) where
  /-- The summary of the states before a position. -/
  read : List Z → R
  /-- The new state of the position `i` from its state and the summary. -/
  combine : ℕ → Z → R → Z

namespace Layer

/-- The layer at the positions of `zs`, which follow the states `pre`. -/
def scan (f : Layer Z R) : List Z → List Z → List Z
  | _, [] => []
  | pre, z :: zs => f.combine pre.length z (f.read pre) :: f.scan (pre ++ [z]) zs

/-- The layer at every position of `zs`. -/
def apply (f : Layer Z R) (zs : List Z) : List Z := f.scan [] zs

theorem length_scan (f : Layer Z R) (pre zs : List Z) : (f.scan pre zs).length = zs.length := by
  induction zs generalizing pre with
  | nil => rfl
  | cons z zs ih => simp [scan, ih]

theorem length_apply (f : Layer Z R) (zs : List Z) : (f.apply zs).length = zs.length :=
  f.length_scan [] zs

theorem scan_append_singleton (f : Layer Z R) (pre zs : List Z) (z : Z) :
    f.scan pre (zs ++ [z]) =
      f.scan pre zs ++ [f.combine (pre ++ zs).length z (f.read (pre ++ zs))] := by
  induction zs generalizing pre with
  | nil => simp [scan]
  | cons y zs ih => simp [scan, ih]

/-- **A layer is causal**: appending a position leaves the states before it
unchanged, and the new one comes from its state and the summary of them. -/
theorem apply_append_singleton (f : Layer Z R) (zs : List Z) (z : Z) :
    f.apply (zs ++ [z]) = f.apply zs ++ [f.combine zs.length z (f.read zs)] := by
  simpa [apply] using f.scan_append_singleton [] zs z

end Layer

/-- The layers `fs` in turn. -/
def run : List (Layer Z R) → List Z → List Z
  | [], zs => zs
  | f :: fs, zs => run fs (f.apply zs)

/-- The summaries that the layers `fs` read from the states `zs`, one per layer. -/
def summaries : List (Layer Z R) → List Z → List R
  | [], _ => []
  | f :: fs, zs => f.read zs :: summaries fs (f.apply zs)

/-- The state that the layers `fs` give a position of index `n` and state `z`
from the summaries `rs` they read before it. -/
def finish : List (Layer Z R) → ℕ → List R → Z → Z
  | [], _, _, z => z
  | _ :: _, _, [], z => z
  | f :: fs, n, r :: rs, z => finish fs n rs (f.combine n z r)

theorem length_summaries (fs : List (Layer Z R)) (zs : List Z) :
    (summaries fs zs).length = fs.length := by
  induction fs generalizing zs with
  | nil => rfl
  | cons f fs ih => simp [summaries, ih]

/-- **The layers are causal**: an appended position gets its state from its own
state and the summaries of the states before it. -/
theorem run_append_singleton (fs : List (Layer Z R)) (zs : List Z) (z : Z) :
    run fs (zs ++ [z]) = run fs zs ++ [finish fs zs.length (summaries fs zs) z] := by
  induction fs generalizing zs z with
  | nil => rfl
  | cons f fs ih => simp only [run, summaries, finish, f.apply_append_singleton, ih, f.length_apply]

/-- A causal model: each token embedded with its position, layers that read the
past through summaries in `R`, and an output read off each position's state. -/
structure Model (X Z R Y : Type*) where
  /-- The state of the token `x` at the position `i`, before the first layer. -/
  embed : ℕ → X → Z
  /-- The layers, applied in turn. -/
  layers : List (Layer Z R)
  /-- The output read off a state. -/
  out : Z → Y

namespace Model

/-- The outputs of `T` at the positions of `xs`. -/
def outputs (T : Model X Z R Y) (xs : List X) : List Y :=
  (run T.layers (xs.mapIdx T.embed)).map T.out

/-- The summaries that the layers of `T` read from the context `xs`. -/
def summary (T : Model X Z R Y) (xs : List X) : List R :=
  summaries T.layers (xs.mapIdx T.embed)

/-- The output of `T` at the token `x` of index `n`, from the summaries `rs`. -/
def reply (T : Model X Z R Y) (n : ℕ) (rs : List R) (x : X) : Y :=
  T.out (finish T.layers n rs (T.embed n x))

theorem length_summary (T : Model X Z R Y) (xs : List X) :
    (T.summary xs).length = T.layers.length :=
  length_summaries _ _

/-- **The output at an appended token depends only on it, its index and the
summaries of the context.** -/
theorem outputs_append_singleton (T : Model X Z R Y) (xs : List X) (x : X) :
    T.outputs (xs ++ [x]) = T.outputs xs ++ [T.reply xs.length (T.summary xs) x] := by
  simp [outputs, summary, reply, List.mapIdx_append, run_append_singleton]

theorem getLast?_outputs_append_singleton (T : Model X Z R Y) (xs : List X) (x : X) :
    (T.outputs (xs ++ [x])).getLast? = some (T.reply xs.length (T.summary xs) x) := by
  rw [outputs_append_singleton, List.getLast?_concat]

variable {K V : Type*} [Fintype K] [Fintype V] [Fintype R]

/-- **A model of the class recalls a dictionary only through its summaries**:
if, after each context `ctx D` of `n` tokens, the token `qry q` gets the output
`D q`, then `|V|^|K| ≤ |R|^L`. -/
theorem card_pow_le (T : Model X Z R V) (ctx : (K → V) → List X) (qry : K → X) {n : ℕ}
    (hlen : ∀ D, (ctx D).length = n)
    (h : ∀ D q, (T.outputs (ctx D ++ [qry q])).getLast? = some (D q)) :
    Fintype.card V ^ Fintype.card K ≤ Fintype.card R ^ T.layers.length := by
  have := Transformer.CausalRecall.card_pow_le_card_of_answers
    (M := List.Vector R T.layers.length)
    (fun D => ⟨T.summary (ctx D), T.length_summary _⟩)
    (fun m q => T.reply n m.1 (qry q)) fun D q => by
      have := h D q
      rwa [getLast?_outputs_append_singleton, hlen, Option.some_inj] at this
  simpa using this

/-- **A model of the class recalls few dictionaries in part**: for weights
`y ≤ x`, the dictionaries `D` for which `a` keys `q` get the output `D q` after
`ctx D` number at most `|R|^L (x + (|V| - 1) y)^|K| / (x^a y^(|K| - a))`. -/
theorem card_answered_mul_le [DecidableEq K] [DecidableEq V] (T : Model X Z R V)
    (ctx : (K → V) → List X) (qry : K → X) {n : ℕ} (hlen : ∀ D, (ctx D).length = n)
    {x y : ℕ} (hyx : y ≤ x) (a : ℕ) :
    #{D | a ≤ #{q | (T.outputs (ctx D ++ [qry q])).getLast? = some (D q)}} *
        (x ^ a * y ^ (Fintype.card K - a)) ≤
      Fintype.card R ^ T.layers.length * (x + (Fintype.card V - 1) * y) ^ Fintype.card K := by
  have := Transformer.CausalRecall.card_answered_mul_le
    (M := List.Vector R T.layers.length)
    (fun D => ⟨T.summary (ctx D), T.length_summary _⟩)
    (fun m q => T.reply n m.1 (qry q)) hyx a
  simpa only [card_vector, getLast?_outputs_append_singleton, hlen, Option.some_inj] using this

end Model

end Transformer.CausalRecall
